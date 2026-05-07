"""SQLite-backed experiment store.

Single trader / small-team scope: SQLite with a JSON column per run is
plenty. The schema is narrow on purpose so swapping for MLflow later is
~50 LOC: just point ExperimentStore.get/list/insert at the MLflow client.

A "run" pins (recipe, notebook, status, metrics, error) and lives forever
unless explicitly deleted. Projects are derived — there is no separate
projects table; a project is just the set of runs that share a name.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .recipes import plausibility


_DEFAULT_PATH = Path(os.environ.get(
    "FINAGENT_EXPERIMENT_DB",
    str(Path(__file__).resolve().parents[1] / "outputs" / "experiments.db"),
))


@dataclass
class Run:
    id: str
    project: str
    name: str
    template: Optional[str]
    recipe_yaml: str
    recipe_hash: str
    status: str               # queued | running | completed | failed
    started_at: float         # unix
    finished_at: Optional[float]
    notebook_path: Optional[str]
    metrics_json: str         # serialized dict[str, float]
    error: Optional[str]
    search_id: Optional[str] = None
    search_iteration: Optional[int] = None
    bias_audit_json: Optional[str] = None
    hypothesis_verdict_json: Optional[str] = None
    fold_metrics_json: Optional[str] = None
    regime_metrics_json: Optional[str] = None
    tags_json: Optional[str] = None

    def metrics(self) -> dict[str, float | None]:
        # Scrub non-finite floats (NaN / ±Infinity) on the way out. Starlette's
        # JSONResponse serializes with allow_nan=False (strict JSON spec), so a
        # NaN in the response payload returns 500 from the project-runs endpoint
        # — the symptom we hit when strategy_metrics emits NaN for degenerate
        # folds (zero-std book, empty intersection, etc.). None renders as "—"
        # on the project page; NaN crashes the whole list endpoint.
        try:
            raw = json.loads(self.metrics_json) if self.metrics_json else {}
        except Exception:
            return {}
        return {k: _finite_or_none(v) for k, v in raw.items()}

    def tags(self) -> list[str]:
        """Decode the user-applied tag list, or [].

        Recognised pre-baked tags carry a colour convention on the
        frontend (candidate=blue, production=emerald, rejected=rose);
        anything else renders neutral. Free-form text is allowed.
        """
        if not self.tags_json:
            return []
        try:
            raw = json.loads(self.tags_json)
            return [str(t) for t in raw] if isinstance(raw, list) else []
        except Exception:
            return []

    def regime_metrics(self) -> list[dict[str, Any]]:
        """Decode the per-regime metric breakdown for unsupervised runs, or [].

        Each entry: {regime, n_obs, pct_of_oos, sharpe, sortino,
        annual_return, max_drawdown, hit_rate}. Only populated for
        target.kind == 'unsupervised_regime' runs.
        """
        if not self.regime_metrics_json:
            return []
        try:
            raw = json.loads(self.regime_metrics_json)
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def fold_metrics(self) -> list[dict[str, Any]]:
        """Decode the per-walk-forward-fold metric list, or [].

        Each entry is a dict like
            {"fold": 0, "start": "2018-01-01", "end": "2018-04-01",
             "n_obs": 60, "sharpe": 0.41, "sortino": 0.58, ...}
        Used by the C3 walk-forward stability dashboard. Empty for runs
        that predate C3 instrumentation or had no folds.
        """
        if not self.fold_metrics_json:
            return []
        try:
            raw = json.loads(self.fold_metrics_json)
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def hypothesis_verdict(self) -> Optional[dict[str, Any]]:
        """Decode the pre-registered-hypothesis verdict, or None.

        Verdict shape (when present):
            {
              "verdict": "PASS" | "FAIL" | "CANCEL",
              "summary": "<one-line>",
              "checks": [
                {"criterion": {metric, op, value}, "actual": <float|None>,
                 "passed": <bool>, "kind": "success"|"cancel"},
                ...
              ]
            }
        """
        if not self.hypothesis_verdict_json:
            return None
        try:
            return json.loads(self.hypothesis_verdict_json)
        except Exception:
            return None

    def bias_audit(self) -> Optional[dict[str, Any]]:
        """Decode the stored audit verdict, or None if not yet audited.

        The DB column is NULL until the post-run audit producer writes a
        verdict (see ``finagent.agents.bias_auditor`` and the hook in
        ``recipe_workflow``). The API surfaces ``bias_audit: null`` so
        frontends can render a "PENDING" badge without an extra round-trip.
        """
        if not self.bias_audit_json:
            return None
        try:
            return json.loads(self.bias_audit_json)
        except Exception:
            return None

    def as_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        metrics = self.metrics()
        d["metrics"] = metrics
        d["metrics_flags"] = plausibility.flag(metrics, self._bands())
        d["bias_audit"] = self.bias_audit()
        d["hypothesis_verdict"] = self.hypothesis_verdict()
        d["fold_metrics"] = self.fold_metrics()
        d["regime_metrics"] = self.regime_metrics()
        d["tags"] = self.tags()
        d.pop("metrics_json", None)
        d.pop("bias_audit_json", None)
        d.pop("hypothesis_verdict_json", None)
        d.pop("fold_metrics_json", None)
        d.pop("regime_metrics_json", None)
        d.pop("tags_json", None)
        return d

    def _bands(self) -> dict[str, tuple[float, float]]:
        """Resolve plausibility bands for this run's template.

        Templates may declare ``METADATA["plausibility"]`` to override the
        default bands (e.g. a crypto template might widen the Sharpe band
        because vol is higher and the daily-equity heuristics don't apply).
        Falls back to ``plausibility.DEFAULT_BANDS`` when the template is
        unknown or doesn't declare its own. Cached per template name to
        avoid re-importing on every public-dict serialization.
        """
        return _resolve_bands(self.template)


@dataclass
class Debate:
    """Bull/bear/moderator debate row.

    Stored on the ``debates`` table; one row per debate, immutable
    once status flips to completed/failed. The transcript is the full
    list of {speaker, phase, text, ts} entries; verdict is the
    moderator's DebateVerdict dict.
    """

    id: str
    ticker: str
    asset_class: str
    rounds: int
    status: str
    started_at: float
    finished_at: Optional[float]
    transcript_json: str
    verdict_json: Optional[str]
    error: Optional[str]
    source: str = "user"  # 'user' (manual) or 'scheduled' (cron)

    def transcript(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self.transcript_json) if self.transcript_json else []
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def verdict(self) -> Optional[dict[str, Any]]:
        if not self.verdict_json:
            return None
        try:
            return json.loads(self.verdict_json)
        except Exception:
            return None

    def as_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["transcript"] = self.transcript()
        d["verdict"] = self.verdict()
        d.pop("transcript_json", None)
        d.pop("verdict_json", None)
        return d


@dataclass
class Search:
    id: str
    project: str
    name: str
    submission_json: str          # full SearchSubmission as JSON
    policy: str                   # "random" | "grid"
    objective_json: str           # serialized Objective
    status: str                   # queued|running|completed|stopped|failed
    started_at: float
    finished_at: Optional[float]
    iterations: int
    best_run_id: Optional[str]
    best_metric: Optional[float]
    error: Optional[str]

    def submission(self) -> dict[str, Any]:
        try:
            return json.loads(self.submission_json) if self.submission_json else {}
        except Exception:
            return {}

    def objective(self) -> dict[str, Any]:
        try:
            return json.loads(self.objective_json) if self.objective_json else {}
        except Exception:
            return {}

    def as_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["submission"] = self.submission()
        d["objective"] = self.objective()
        d.pop("submission_json", None)
        d.pop("objective_json", None)
        return d


# Base schema — runs table is the original shape; new columns are added by
# the _migrate() pass below so existing DBs upgrade in place.
_SCHEMA_BASE = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    project       TEXT NOT NULL,
    name          TEXT NOT NULL,
    template      TEXT,
    recipe_yaml   TEXT NOT NULL,
    recipe_hash   TEXT NOT NULL,
    status        TEXT NOT NULL,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    notebook_path TEXT,
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    error         TEXT,
    bias_audit_json TEXT,
    hypothesis_verdict_json TEXT,
    fold_metrics_json TEXT,
    regime_metrics_json TEXT,
    tags_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);

CREATE TABLE IF NOT EXISTS searches (
    id              TEXT PRIMARY KEY,
    project         TEXT NOT NULL,
    name            TEXT NOT NULL,
    submission_json TEXT NOT NULL,
    policy          TEXT NOT NULL,
    objective_json  TEXT NOT NULL,
    status          TEXT NOT NULL,        -- queued|running|completed|stopped|failed
    started_at      REAL NOT NULL,
    finished_at     REAL,
    iterations      INTEGER NOT NULL DEFAULT 0,
    best_run_id     TEXT,
    best_metric     REAL,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_searches_project ON searches(project);
CREATE INDEX IF NOT EXISTS idx_searches_started ON searches(started_at DESC);

-- Cost ledger — one row per LLM call. Sum these for per-day / per-user /
-- per-purpose burn-rate. Per-call resolution is overkill for monthly
-- bookkeeping but useful when investigating a spike ("this run alone
-- chewed $4 in audit calls because it had 80 cells").
CREATE TABLE IF NOT EXISTS cost_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    user            TEXT,
    run_id          TEXT,
    purpose         TEXT NOT NULL,   -- 'bias_audit' / 'chat' / ...
    provider        TEXT NOT NULL,   -- 'openai' / 'anthropic'
    model           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_cost_ts      ON cost_events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_cost_run     ON cost_events(run_id);
CREATE INDEX IF NOT EXISTS idx_cost_purpose ON cost_events(purpose);

-- Debate runs — bull/bear/moderator multi-agent verdicts on a single
-- ticker. Transcript and verdict stored as JSON columns (one debate per
-- row, immutable once finished).
CREATE TABLE IF NOT EXISTS debates (
    id              TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    asset_class     TEXT NOT NULL,
    rounds          INTEGER NOT NULL DEFAULT 2,
    status          TEXT NOT NULL,           -- queued|running|completed|failed
    started_at      REAL NOT NULL,
    finished_at     REAL,
    transcript_json TEXT NOT NULL DEFAULT '[]',
    verdict_json    TEXT,
    error           TEXT,
    source          TEXT NOT NULL DEFAULT 'user'  -- 'user' | 'scheduled'
);
CREATE INDEX IF NOT EXISTS idx_debates_started ON debates(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_debates_ticker  ON debates(ticker);
"""

# Indexes that reference columns added by _migrate(). Created after the
# migration so they don't fail on legacy DBs.
#
# idx_debates_source MUST live here, not in _SCHEMA_BASE: legacy DBs that
# pre-date the daily-Nifty scheduler have a `debates` table without the
# `source` column. The `CREATE TABLE IF NOT EXISTS` at the top of
# _SCHEMA_BASE is a no-op on those DBs, so the column never gets added by
# that path; only `_migrate()` adds it via ALTER TABLE. Creating the
# index here, after _migrate has run, guarantees the column exists.
_SCHEMA_POST_MIGRATE = """
CREATE INDEX IF NOT EXISTS idx_runs_search    ON runs(search_id);
CREATE INDEX IF NOT EXISTS idx_debates_source ON debates(source);
"""


class ExperimentStore:
    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA_BASE)
            self._migrate(conn)
            conn.executescript(_SCHEMA_POST_MIGRATE)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent column additions for DBs that pre-date the searches feature."""
        existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        if "search_id" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN search_id TEXT")
        if "search_iteration" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN search_iteration INTEGER")
        if "bias_audit_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN bias_audit_json TEXT")
        if "hypothesis_verdict_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN hypothesis_verdict_json TEXT")
        if "fold_metrics_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN fold_metrics_json TEXT")
        if "regime_metrics_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN regime_metrics_json TEXT")
        if "tags_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN tags_json TEXT")

        # ── debates table column migrations ─────────────────────────
        try:
            existing_dbates = {r[1] for r in conn.execute("PRAGMA table_info(debates)")}
        except Exception:
            existing_dbates = set()
        if existing_dbates and "source" not in existing_dbates:
            conn.execute("ALTER TABLE debates ADD COLUMN source TEXT NOT NULL DEFAULT 'user'")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── writes ──────────────────────────────────────────────────────────

    def create_run(
        self,
        *,
        project: str,
        name: str,
        template: Optional[str],
        recipe_yaml: str,
        recipe_hash: str,
        search_id: Optional[str] = None,
        search_iteration: Optional[int] = None,
    ) -> Run:
        run_id = uuid.uuid4().hex[:16]
        now = time.time()
        run = Run(
            id=run_id,
            project=project,
            name=name,
            template=template,
            recipe_yaml=recipe_yaml,
            recipe_hash=recipe_hash,
            status="queued",
            started_at=now,
            finished_at=None,
            notebook_path=None,
            metrics_json="{}",
            error=None,
            search_id=search_id,
            search_iteration=search_iteration,
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, project, name, template, recipe_yaml, "
                "recipe_hash, status, started_at, metrics_json, search_id, search_iteration) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)",
                (run_id, project, name, template, recipe_yaml,
                 recipe_hash, "queued", now, search_id, search_iteration),
            )
        return run

    def update_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        notebook_path: Optional[str] = None,
        metrics: Optional[dict[str, float]] = None,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        sets: list[str] = []
        args: list[Any] = []
        if status is not None:
            sets.append("status = ?")
            args.append(status)
        if notebook_path is not None:
            sets.append("notebook_path = ?")
            args.append(notebook_path)
        if metrics is not None:
            # Scrub NaN/Inf before storing — keeps the DB JSON-spec-clean and
            # prevents a future schema migration from re-encountering this.
            clean = {k: _finite_or_none(v) for k, v in metrics.items()}
            sets.append("metrics_json = ?")
            args.append(json.dumps(clean))
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        if finished:
            sets.append("finished_at = ?")
            args.append(time.time())
        if not sets:
            return
        args.append(run_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE runs SET {', '.join(sets)} WHERE id = ?",
                args,
            )

    def delete_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def update_run_bias_audit(self, run_id: str, audit_json: str) -> None:
        """Persist the LLM-judge audit verdict (a JSON-serialised dict).

        Separate from ``update_run`` because the audit writes lag behind the
        run's terminal status — they happen on a background task — and we
        don't want to accidentally re-open a finished run's status field.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET bias_audit_json = ? WHERE id = ?",
                (audit_json, run_id),
            )

    def update_run_hypothesis_verdict(self, run_id: str, verdict_json: str) -> None:
        """Persist the pre-registered hypothesis verdict (PASS / FAIL / CANCEL)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET hypothesis_verdict_json = ? WHERE id = ?",
                (verdict_json, run_id),
            )

    def update_run_fold_metrics(self, run_id: str, fold_metrics_json: str) -> None:
        """Persist the per-walk-forward-fold metric list."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET fold_metrics_json = ? WHERE id = ?",
                (fold_metrics_json, run_id),
            )

    def update_run_regime_metrics(self, run_id: str, regime_metrics_json: str) -> None:
        """Persist the per-regime metric breakdown (unsupervised runs)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET regime_metrics_json = ? WHERE id = ?",
                (regime_metrics_json, run_id),
            )

    # ── cost ledger ──────────────────────────────────────────────────

    def record_cost_event(
        self,
        *,
        purpose: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        run_id: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        """Append a single cost-event row. One per LLM call."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cost_events (ts, user, run_id, purpose, provider, "
                "model, prompt_tokens, completion_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(), user, run_id, purpose, provider, model,
                    int(prompt_tokens), int(completion_tokens), float(cost_usd),
                ),
            )

    def cost_summary(self, days: int = 30) -> dict[str, Any]:
        """Aggregate the cost ledger for the last ``days`` days.

        Returns:
          {
            "window_days": days,
            "total_usd": <float>,
            "total_calls": <int>,
            "by_day":     [{date, calls, usd}, ...],
            "by_purpose": [{purpose, calls, usd}, ...],
            "by_user":    [{user, calls, usd}, ...],
            "by_model":   [{model, calls, usd}, ...],
            "top_runs":   [{run_id, calls, usd}, ...]   -- top 20
          }
        """
        cutoff = time.time() - days * 86400.0
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(SUM(cost_usd), 0) AS u "
                "FROM cost_events WHERE ts >= ?",
                (cutoff,),
            ).fetchone()
            by_day = conn.execute(
                "SELECT date(ts, 'unixepoch') AS d, COUNT(*) AS c, "
                "COALESCE(SUM(cost_usd), 0) AS u "
                "FROM cost_events WHERE ts >= ? GROUP BY d ORDER BY d",
                (cutoff,),
            ).fetchall()
            by_purpose = conn.execute(
                "SELECT purpose, COUNT(*) AS c, COALESCE(SUM(cost_usd), 0) AS u "
                "FROM cost_events WHERE ts >= ? GROUP BY purpose ORDER BY u DESC",
                (cutoff,),
            ).fetchall()
            by_user = conn.execute(
                "SELECT COALESCE(user, '') AS user, COUNT(*) AS c, "
                "COALESCE(SUM(cost_usd), 0) AS u "
                "FROM cost_events WHERE ts >= ? GROUP BY user ORDER BY u DESC",
                (cutoff,),
            ).fetchall()
            by_model = conn.execute(
                "SELECT model, COUNT(*) AS c, COALESCE(SUM(cost_usd), 0) AS u "
                "FROM cost_events WHERE ts >= ? GROUP BY model ORDER BY u DESC",
                (cutoff,),
            ).fetchall()
            top_runs = conn.execute(
                "SELECT COALESCE(run_id, '') AS run_id, COUNT(*) AS c, "
                "COALESCE(SUM(cost_usd), 0) AS u "
                "FROM cost_events WHERE ts >= ? AND run_id IS NOT NULL "
                "GROUP BY run_id ORDER BY u DESC LIMIT 20",
                (cutoff,),
            ).fetchall()

        def _rows(rows, key_col):
            return [{key_col: r[key_col], "calls": r["c"], "usd": r["u"]} for r in rows]

        return {
            "window_days": days,
            "total_usd": float(total["u"]),
            "total_calls": int(total["c"]),
            "by_day": [{"date": r["d"], "calls": r["c"], "usd": r["u"]} for r in by_day],
            "by_purpose": _rows(by_purpose, "purpose"),
            "by_user":    _rows(by_user, "user"),
            "by_model":   _rows(by_model, "model"),
            "top_runs":   _rows(top_runs, "run_id"),
        }

    # ── debates ──────────────────────────────────────────────────────

    def create_debate(
        self,
        *,
        ticker: str,
        asset_class: str,
        rounds: int,
        source: str = "user",
    ) -> Debate:
        debate_id = uuid.uuid4().hex[:16]
        now = time.time()
        d = Debate(
            id=debate_id,
            ticker=ticker,
            asset_class=asset_class,
            rounds=rounds,
            status="queued",
            started_at=now,
            finished_at=None,
            transcript_json="[]",
            verdict_json=None,
            error=None,
            source=source,
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO debates (id, ticker, asset_class, rounds, status, "
                "started_at, transcript_json, source) "
                "VALUES (?, ?, ?, ?, ?, ?, '[]', ?)",
                (debate_id, ticker, asset_class, rounds, "queued", now, source),
            )
        return d

    def update_debate(
        self,
        debate_id: str,
        *,
        status: Optional[str] = None,
        transcript: Optional[list[dict]] = None,
        verdict: Optional[dict] = None,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        sets: list[str] = []
        args: list[Any] = []
        if status is not None:
            sets.append("status = ?"); args.append(status)
        if transcript is not None:
            sets.append("transcript_json = ?"); args.append(json.dumps(transcript, default=str))
        if verdict is not None:
            sets.append("verdict_json = ?"); args.append(json.dumps(verdict, default=str))
        if error is not None:
            sets.append("error = ?"); args.append(error)
        if finished:
            sets.append("finished_at = ?"); args.append(time.time())
        if not sets:
            return
        args.append(debate_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE debates SET {', '.join(sets)} WHERE id = ?", args)

    def get_debate(self, debate_id: str) -> Optional[Debate]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM debates WHERE id = ?", (debate_id,)).fetchone()
        return _row_to_debate(row) if row else None

    def list_debates(self, ticker: Optional[str] = None, limit: int = 100) -> list[Debate]:
        sql = "SELECT * FROM debates"
        args: list[Any] = []
        if ticker:
            sql += " WHERE ticker = ?"; args.append(ticker)
        sql += " ORDER BY started_at DESC LIMIT ?"; args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_row_to_debate(r) for r in rows]

    def delete_debate(self, debate_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM debates WHERE id = ?", (debate_id,))

    # ── tags (existing) ──────────────────────────────────────────────

    def update_run_tags(self, run_id: str, tags_json: str) -> None:
        """Persist user-applied tags (workflow signal).

        Recognised pre-baked tags: 'candidate', 'production', 'rejected'
        — they get colour conventions on the frontend. Anything else
        is free-form text. Storage is just a JSON list; deduplication
        + normalisation happens at write time in the API handler.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET tags_json = ? WHERE id = ?",
                (tags_json, run_id),
            )

    # ── reads ───────────────────────────────────────────────────────────

    def get(self, run_id: str) -> Optional[Run]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self, project: Optional[str] = None,
                  limit: int = 200) -> list[Run]:
        sql = "SELECT * FROM runs"
        args: list[Any] = []
        if project:
            sql += " WHERE project = ?"
            args.append(project)
        sql += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_row_to_run(r) for r in rows]

    # ── searches ────────────────────────────────────────────────────────

    def create_search(
        self,
        *,
        project: str,
        name: str,
        submission_json: str,
        policy: str,
        objective_json: str,
    ) -> Search:
        sid = uuid.uuid4().hex[:16]
        now = time.time()
        search = Search(
            id=sid,
            project=project,
            name=name,
            submission_json=submission_json,
            policy=policy,
            objective_json=objective_json,
            status="queued",
            started_at=now,
            finished_at=None,
            iterations=0,
            best_run_id=None,
            best_metric=None,
            error=None,
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO searches (id, project, name, submission_json, policy, "
                "objective_json, status, started_at, iterations) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (sid, project, name, submission_json, policy, objective_json,
                 "queued", now),
            )
        return search

    def update_search(
        self,
        search_id: str,
        *,
        status: Optional[str] = None,
        iterations: Optional[int] = None,
        best_run_id: Optional[str] = None,
        best_metric: Optional[float] = None,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        sets: list[str] = []
        args: list[Any] = []
        if status is not None:
            sets.append("status = ?")
            args.append(status)
        if iterations is not None:
            sets.append("iterations = ?")
            args.append(iterations)
        if best_run_id is not None:
            sets.append("best_run_id = ?")
            args.append(best_run_id)
        if best_metric is not None:
            sets.append("best_metric = ?")
            args.append(best_metric)
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        if finished:
            sets.append("finished_at = ?")
            args.append(time.time())
        if not sets:
            return
        args.append(search_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE searches SET {', '.join(sets)} WHERE id = ?",
                args,
            )

    def get_search(self, search_id: str) -> Optional[Search]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM searches WHERE id = ?", (search_id,)
            ).fetchone()
        return _row_to_search(row) if row else None

    def list_searches(self, project: Optional[str] = None,
                      limit: int = 100) -> list[Search]:
        sql = "SELECT * FROM searches"
        args: list[Any] = []
        if project:
            sql += " WHERE project = ?"
            args.append(project)
        sql += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_row_to_search(r) for r in rows]

    def runs_in_search(self, search_id: str) -> list[Run]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE search_id = ? ORDER BY search_iteration ASC",
                (search_id,),
            ).fetchall()
        return [_row_to_run(r) for r in rows]

    def list_projects(self) -> list[dict[str, Any]]:
        sql = (
            "SELECT project, COUNT(*) as run_count, MAX(started_at) as last_run "
            "FROM runs GROUP BY project ORDER BY last_run DESC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            {"project": r["project"], "run_count": r["run_count"],
             "last_run": r["last_run"]}
            for r in rows
        ]


# Cache: template name → bands dict. Built lazily on first lookup. The
# template registry import is deferred to here because templates import
# from .recipes.types and we want to keep the experiments module
# import-cheap for tooling that just wants the Run dataclass.
_BANDS_CACHE: dict[Optional[str], dict[str, tuple[float, float]]] = {}


def _resolve_bands(template_name: Optional[str]) -> dict[str, tuple[float, float]]:
    if template_name in _BANDS_CACHE:
        return _BANDS_CACHE[template_name]
    bands = plausibility.DEFAULT_BANDS
    if template_name:
        try:
            from .recipes.templates import REGISTRY  # local import: avoid cycle at module load
            module = REGISTRY.get(template_name)
            metadata = getattr(module, "METADATA", None) if module else None
            declared = metadata.get("plausibility") if isinstance(metadata, dict) else None
            if isinstance(declared, dict):
                bands = declared
        except Exception:
            logging.getLogger(__name__).exception(
                "failed to resolve plausibility bands for template %s", template_name,
            )
    _BANDS_CACHE[template_name] = bands
    return bands


def _finite_or_none(v: Any) -> Any:
    """Return v unchanged unless it's a non-finite float, in which case None.

    Centralises the NaN/Inf scrub so both reads (Run.metrics) and writes
    (update_run) emit JSON-spec-compliant payloads. Starlette's JSONResponse
    rejects NaN/Infinity (allow_nan=False); leaving them in causes 500s on
    /api/projects/{name}/runs once any run records a degenerate metric.
    """
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _row_to_run(row: sqlite3.Row) -> Run:
    keys = row.keys()
    return Run(
        id=row["id"],
        project=row["project"],
        name=row["name"],
        template=row["template"],
        recipe_yaml=row["recipe_yaml"],
        recipe_hash=row["recipe_hash"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        notebook_path=row["notebook_path"],
        metrics_json=row["metrics_json"] or "{}",
        error=row["error"],
        search_id=(row["search_id"] if "search_id" in keys else None),
        search_iteration=(row["search_iteration"] if "search_iteration" in keys else None),
        bias_audit_json=(row["bias_audit_json"] if "bias_audit_json" in keys else None),
        hypothesis_verdict_json=(row["hypothesis_verdict_json"] if "hypothesis_verdict_json" in keys else None),
        fold_metrics_json=(row["fold_metrics_json"] if "fold_metrics_json" in keys else None),
        regime_metrics_json=(row["regime_metrics_json"] if "regime_metrics_json" in keys else None),
        tags_json=(row["tags_json"] if "tags_json" in keys else None),
    )


def _row_to_debate(row: sqlite3.Row) -> Debate:
    keys = row.keys()
    return Debate(
        id=row["id"],
        ticker=row["ticker"],
        asset_class=row["asset_class"],
        rounds=int(row["rounds"]) if row["rounds"] is not None else 2,
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        transcript_json=row["transcript_json"] or "[]",
        verdict_json=row["verdict_json"],
        error=row["error"],
        source=(row["source"] if "source" in keys else "user") or "user",
    )


def _row_to_search(row: sqlite3.Row) -> Search:
    return Search(
        id=row["id"],
        project=row["project"],
        name=row["name"],
        submission_json=row["submission_json"],
        policy=row["policy"],
        objective_json=row["objective_json"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        iterations=row["iterations"],
        best_run_id=row["best_run_id"],
        best_metric=row["best_metric"],
        error=row["error"],
    )


# Process-local default store handle.
_store: Optional[ExperimentStore] = None


def get_store() -> ExperimentStore:
    global _store
    if _store is None:
        _store = ExperimentStore()
    return _store
