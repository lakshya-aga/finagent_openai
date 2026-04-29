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
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


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

    def metrics(self) -> dict[str, float]:
        try:
            return json.loads(self.metrics_json) if self.metrics_json else {}
        except Exception:
            return {}

    def as_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics()
        d.pop("metrics_json", None)
        return d


_SCHEMA = """
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
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""


class ExperimentStore:
    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

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
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, project, name, template, recipe_yaml, "
                "recipe_hash, status, started_at, metrics_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')",
                (run_id, project, name, template, recipe_yaml,
                 recipe_hash, "queued", now),
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
            sets.append("metrics_json = ?")
            args.append(json.dumps(metrics))
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


def _row_to_run(row: sqlite3.Row) -> Run:
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
    )


# Process-local default store handle.
_store: Optional[ExperimentStore] = None


def get_store() -> ExperimentStore:
    global _store
    if _store is None:
        _store = ExperimentStore()
    return _store
