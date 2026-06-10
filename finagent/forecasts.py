"""Forecast persistence — every event-probability forecast, kept forever.

The rows are the future calibration corpus: each forecast starts
status='open' and is later resolved to outcome 1.0 / 0.0, at which
point Brier scores and calibration curves become computable. The
resolution tooling comes later; the schema is deliberately ready now
because calibration data only accumulates in calendar time — every
forecast NOT persisted is a data point lost.

Same SQLite DB + lazy-schema pattern as finagent/credits.py.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id                  TEXT PRIMARY KEY,
    question            TEXT NOT NULL,
    reframed_question   TEXT NOT NULL DEFAULT '',
    probability         REAL NOT NULL,
    p_low               REAL,
    p_high              REAL,
    n_ensemble          INTEGER NOT NULL DEFAULT 1,
    low_agreement       INTEGER NOT NULL DEFAULT 0,
    rationale           TEXT NOT NULL DEFAULT '',
    key_drivers_json    TEXT NOT NULL DEFAULT '[]',
    what_would_change   TEXT NOT NULL DEFAULT '',
    resolution_criteria TEXT NOT NULL DEFAULT '',
    horizon             TEXT NOT NULL DEFAULT '',
    base_rate           REAL,
    evidence_json       TEXT NOT NULL DEFAULT '[]',
    summary             TEXT NOT NULL DEFAULT '',
    owner               TEXT,
    status              TEXT NOT NULL DEFAULT 'open',   -- open | resolved | voided
    outcome             REAL,                            -- 1.0 / 0.0 once resolved
    resolved_at         REAL,
    created_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecasts_created ON forecasts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_owner   ON forecasts(owner);
CREATE INDEX IF NOT EXISTS idx_forecasts_status  ON forecasts(status);
"""


def _db_path() -> str:
    env = os.environ.get("FINAGENT_EXPERIMENT_DB", "").strip()
    if env:
        return env
    from finagent.experiments import _DEFAULT_PATH

    return str(_DEFAULT_PATH)


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_forecast(result: dict, *, owner: Optional[str] = None) -> str:
    """Persist a run_forecast() result dict. Returns the forecast id."""
    fid = uuid.uuid4().hex[:16]
    with _conn() as conn:
        conn.execute(
            "INSERT INTO forecasts (id, question, reframed_question, probability, "
            "p_low, p_high, n_ensemble, low_agreement, rationale, key_drivers_json, "
            "what_would_change, resolution_criteria, horizon, base_rate, "
            "evidence_json, summary, owner, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
            (
                fid,
                result.get("question", ""),
                result.get("reframed_question", ""),
                float(result.get("probability", 0.5)),
                result.get("p_low"),
                result.get("p_high"),
                int(result.get("n_ensemble", 1)),
                1 if result.get("low_agreement") else 0,
                result.get("rationale", ""),
                json.dumps(result.get("key_drivers", []), default=str),
                result.get("what_would_change_mind", ""),
                result.get("resolution_criteria", ""),
                result.get("horizon", ""),
                result.get("base_rate"),
                json.dumps(result.get("evidence", []), default=str),
                result.get("summary", ""),
                owner,
                time.time(),
            ),
        )
    return fid


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for src, dst in (("key_drivers_json", "key_drivers"), ("evidence_json", "evidence")):
        try:
            d[dst] = json.loads(d.pop(src) or "[]")
        except Exception:
            d[dst] = []
    d["low_agreement"] = bool(d.get("low_agreement"))
    return d


def get_forecast(forecast_id: str) -> Optional[dict[str, Any]]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM forecasts WHERE id = ?", (forecast_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_forecasts(
    *,
    owner: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM forecasts"
    args: list[Any] = []
    if owner:
        sql += " WHERE owner = ?"
        args.append(owner)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args.extend([max(1, min(int(limit), 200)), max(0, min(int(offset), 2000))])
    with _conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def resolve_forecast(forecast_id: str, outcome: float) -> bool:
    """Mark a forecast resolved with outcome 1.0 (happened) / 0.0 (didn't).
    The calibration harness consumes these. Returns False if not found
    or already resolved."""
    if outcome not in (0.0, 1.0):
        raise ValueError("outcome must be 0.0 or 1.0")
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE forecasts SET status='resolved', outcome=?, resolved_at=? "
            "WHERE id = ? AND status = 'open'",
            (float(outcome), time.time(), forecast_id),
        )
        return cur.rowcount > 0
