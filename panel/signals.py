"""Signal export — write the time-series + register it in the dashboard.

A "signal" here is a pandas Series (or DataFrame with a single
non-index column) of values produced by a research notebook that the
user wants to see on the dashboard. Examples:

  * SPY 252-day rolling Sharpe ratio (one float per trading day)
  * Pairs-trade z-score on SPY/TLT (one float per minute)
  * Bull/bear debate verdict score per ticker per day

The export does three things atomically:

  1. Write ``outputs/signals/<name>/series.parquet``
  2. Write ``outputs/signals/<name>/manifest.json``
  3. Upsert a row in ``signals`` + insert a row in ``signal_versions``
     in ``outputs/experiments.db``

The DB write is best-effort — if the SQLite file is missing or locked
the disk artefacts still land, the manifest still validates, and the
signal can be re-registered later by re-running the export. We log a
warning rather than crashing the notebook because killing a 20-minute
training run because the dashboard DB is locked is the wrong default.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import _store


logger = logging.getLogger(__name__)


def _series_path(name: str) -> Path:
    return _store.signals_dir() / name / "series.parquet"


def _manifest_path(name: str) -> Path:
    return _store.signals_dir() / name / "manifest.json"


# ── Public API ─────────────────────────────────────────────────────────


def export_signal(
    name: str,
    series,
    *,
    metadata: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist a pandas Series as a registered signal.

    Parameters
    ----------
    name
        Lowercase kebab-case slug. Same naming rules as ``save_model``.
        If a signal with this name already exists, this counts as a new
        *version* (a new ``signal_versions`` row) and the on-disk
        artefacts are overwritten with the latest series.
    series
        ``pandas.Series`` indexed by date/datetime, OR a
        ``pandas.DataFrame`` with exactly one non-index column. Index
        must be sorted ascending or convertible to a sorted DatetimeIndex.
    metadata
        Free-form dict declaring what the signal *means*. Recommended
        keys:

        * ``frequency``: 'daily' | 'minute' | 'tick' | 'event'
        * ``universe``: list[str] of tickers / instruments
        * ``description``: one-line plain-English summary
        * ``interpretation``: how to read the value ('higher = bullish',
          'long when > 0.5', 'z-score: enter at ±2', ...)
        * ``model_name``: name of the saved ``panel.save_model`` blob
          this signal is derived from (optional but recommended for
          reproducibility)
        * ``recipe_fingerprint``: SHA8 from the recipe that produced it
        * ``template``: template name (e.g. 'regime_modeling')
        * ``project``: project the signal belongs to

    run_id
        Override the auto-detected ``FINAGENT_RUN_ID`` env var. Useful
        when re-exporting a historical series outside the workflow.

    Returns
    -------
    Path to the manifest file. The dashboard discovers signals by
    walking ``outputs/signals/*/manifest.json`` if the DB is unavailable.
    """
    # Lazy-import pandas so the SDK loads in environments without it
    # (the data-mcp container doesn't ship pandas, for example).
    import pandas as pd  # noqa: WPS433 — local import is intentional

    name = _store.validate_name(name)

    # Normalise to a Series.
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(
                f"export_signal expects a Series or single-column DataFrame, "
                f"got DataFrame with shape {series.shape}"
            )
        series = series.iloc[:, 0]
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"export_signal expects pandas.Series, got {type(series).__name__}"
        )

    if series.empty:
        raise ValueError(f"export_signal: cannot export empty series for {name!r}")

    # Index must be a DatetimeIndex (or convertible).
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series = series.copy()
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise ValueError(
                f"export_signal: index must be DatetimeIndex (or convertible), "
                f"failed: {e}"
            ) from e

    # Sort + drop NaN values to give a clean canonical artefact.
    series = series.sort_index()
    n_total = len(series)
    series_clean = series.dropna()

    # Write parquet.
    series_path = _series_path(name)
    series_path.parent.mkdir(parents=True, exist_ok=True)
    df = series_clean.to_frame(name="value")
    df.index.name = "ts"
    df.to_parquet(series_path)

    last_value = float(series_clean.iloc[-1]) if len(series_clean) else None
    last_value_at = (
        series_clean.index[-1].isoformat() if len(series_clean) else None
    )

    # Write manifest.
    manifest = {
        "name": name,
        "kind": "signal",
        "series_path": str(series_path),
        "n_observations": int(len(series_clean)),
        "n_total_points": int(n_total),  # before dropna
        "first_value_at": (
            series_clean.index[0].isoformat() if len(series_clean) else None
        ),
        "last_value_at": last_value_at,
        "last_value": last_value,
        "metadata": dict(metadata or {}),
        "run_id": run_id or _store.current_run_id(),
    }
    manifest_path = _manifest_path(name)
    _store.write_manifest(manifest_path, manifest)
    logger.info(
        "panel.export_signal: wrote %s (%d obs, last=%s @ %s)",
        series_path, len(series_clean), last_value, last_value_at,
    )

    # Best-effort DB upsert — if it fails, the disk artefacts are still
    # canonical and a future re-export will reconcile.
    try:
        register_signal(
            name=name,
            run_id=run_id or _store.current_run_id(),
            n_observations=int(len(series_clean)),
            last_value=last_value,
            last_value_at=last_value_at,
            metadata=manifest["metadata"],
        )
    except Exception:
        logger.exception(
            "panel.export_signal: DB registration failed for %r; "
            "disk artefacts written, dashboard may need manual re-scan",
            name,
        )

    return manifest_path


def load_signal(name: str):
    """Return the signal as a ``pandas.Series`` indexed by datetime."""
    import pandas as pd

    name = _store.validate_name(name)
    path = _series_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no signal exported for {name!r} at {path}")
    df = pd.read_parquet(path)
    return df["value"]


def signal_manifest(name: str) -> dict[str, Any]:
    name = _store.validate_name(name)
    return _store.read_manifest(_manifest_path(name))


def list_signals() -> list[dict[str, Any]]:
    """Walk the signals/ tree. Used as a fallback when the DB is
    unavailable; the dashboard should normally hit the ``signals`` table
    via app.py's REST endpoint."""
    base = _store.signals_dir()
    if not base.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        mf = child / "manifest.json"
        if not mf.exists():
            continue
        try:
            out.append(_store.read_manifest(mf))
        except Exception:
            logger.exception("panel.list_signals: skipped malformed manifest %s", mf)
    return out


# ── DB registration ────────────────────────────────────────────────────


def _experiments_db_path() -> Path:
    """Locate the experiments SQLite DB. Same default as
    ``finagent.experiments._DEFAULT_PATH`` but we don't import that
    module here to keep the panel SDK independent of the agent stack
    (the SDK runs inside notebook kernels that may not have the agent
    deps installed).

    Env var precedence:
      1. ``FINAGENT_EXPERIMENT_DB`` — the canonical name used by
         ``finagent.experiments``.
      2. ``FINAGENT_DB``           — short alias accepted for
         backward compatibility with earlier SDK versions.
    """
    import os
    db_env = os.environ.get("FINAGENT_EXPERIMENT_DB") or os.environ.get("FINAGENT_DB")
    return Path(db_env or str(_store.outputs_dir() / "experiments.db"))


def register_signal(
    *,
    name: str,
    run_id: str | None,
    n_observations: int,
    last_value: float | None,
    last_value_at: str | None,
    metadata: Mapping[str, Any],
) -> str:
    """Upsert into ``signals`` + insert into ``signal_versions``.

    Returns the signal's primary-key id. Idempotent on ``name`` — second
    call with the same ``name`` updates the existing row's
    ``updated_at`` / ``last_value`` / ``last_value_at`` /
    ``n_observations`` and inserts a new ``signal_versions`` row.

    The schema is created on demand (``CREATE TABLE IF NOT EXISTS``) so
    the SDK works against a freshly-created DB without a prior call into
    ``finagent.experiments.ExperimentStore``.
    """
    import json

    db_path = _experiments_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SIGNALS_SCHEMA)
        now = time.time()
        meta = dict(metadata or {})
        # Pull common fields out of metadata into top-level columns so
        # the dashboard can filter on them without parsing JSON.
        frequency = str(meta.get("frequency") or "unknown")
        universe = json.dumps(meta.get("universe") or [])
        description = (meta.get("description") or "")[:500]
        project = meta.get("project")
        template = meta.get("template")
        recipe_fp = meta.get("recipe_fingerprint")

        existing = conn.execute(
            "SELECT id FROM signals WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            sig_id = existing[0]
            conn.execute(
                """
                UPDATE signals
                SET updated_at = ?,
                    n_observations = ?,
                    last_value = ?,
                    last_value_at = ?,
                    frequency = ?,
                    universe_json = ?,
                    description = ?,
                    project = COALESCE(?, project),
                    template = COALESCE(?, template),
                    recipe_fingerprint = COALESCE(?, recipe_fingerprint),
                    run_id = COALESCE(?, run_id),
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    now, n_observations, last_value, last_value_at,
                    frequency, universe, description,
                    project, template, recipe_fp, run_id,
                    json.dumps(meta, default=str),
                    sig_id,
                ),
            )
        else:
            sig_id = uuid.uuid4().hex[:16]
            conn.execute(
                """
                INSERT INTO signals (
                    id, name, project, run_id, recipe_fingerprint, template,
                    frequency, universe_json, description,
                    created_at, updated_at,
                    last_value, last_value_at, n_observations,
                    status, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    sig_id, name, project, run_id, recipe_fp, template,
                    frequency, universe, description,
                    now, now,
                    last_value, last_value_at, n_observations,
                    json.dumps(meta, default=str),
                ),
            )

        # Always log a new version row, whether it's the first export or
        # the 50th. Lets the UI show "last 10 retrains" without parsing
        # the parquet history.
        conn.execute(
            """
            INSERT INTO signal_versions (
                signal_id, run_id, inserted_at, n_observations,
                last_value, last_value_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig_id, run_id, now, n_observations,
                last_value, last_value_at,
                None,  # caller can extend this later via notes if desired
            ),
        )
        conn.commit()
        return sig_id
    finally:
        conn.close()


# ── Schema (mirrored from finagent/experiments.py for SDK independence) ──

_SIGNALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    project             TEXT,
    run_id              TEXT,
    recipe_fingerprint  TEXT,
    template            TEXT,
    frequency           TEXT NOT NULL DEFAULT 'unknown',
    universe_json       TEXT NOT NULL DEFAULT '[]',
    description         TEXT NOT NULL DEFAULT '',
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    last_value          REAL,
    last_value_at       TEXT,
    n_observations      INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'active',
    metadata_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_signals_project   ON signals(project);
CREATE INDEX IF NOT EXISTS idx_signals_template  ON signals(template);
CREATE INDEX IF NOT EXISTS idx_signals_updated   ON signals(updated_at DESC);

CREATE TABLE IF NOT EXISTS signal_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       TEXT NOT NULL,
    run_id          TEXT,
    inserted_at     REAL NOT NULL,
    n_observations  INTEGER,
    last_value      REAL,
    last_value_at   TEXT,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_signal_versions_signal ON signal_versions(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_versions_inserted ON signal_versions(inserted_at DESC);
"""
