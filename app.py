from dotenv import load_dotenv
load_dotenv()

# Configure logging at INFO so logger.info() calls reach docker logs.
# Python's root logger defaults to WARNING — without this, the
# scheduler-startup banner, chart_smoke results, and the panel's
# tool-loop diagnostics all silently disappear and `docker logs
# synapse-finagent-1 | grep …` returns empty. Done at import time
# so it covers every module imported below.
import logging as _logging_setup
_logging_setup.basicConfig(
    level=_logging_setup.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,   # override uvicorn's logger config which kicks in later
)

# Phoenix tracing — auto-instruments openai / httpx / langchain calls
# when PHOENIX_COLLECTOR_ENDPOINT is set in the env. No-op locally.
# Must run before any LLM client is constructed so the instrumentation
# patches the SDK before first use.
from finagent.tracing import init_tracing
init_tracing()

# Daily Nifty 50 debate scheduler. APScheduler in-process; cron at
# 02:00 UTC (07:30 IST). No-op if apscheduler isn't installed locally.
from finagent.scheduler import start_scheduler, stop_scheduler

import asyncio
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from agent_workflow import run_workflow, WorkflowInput

# ── Rate Limiting ────────────────────────────────────────────────────────
# Simple in-memory sliding-window rate limiter. Tracks timestamps per key
# (session_id or IP). Configurable via RATE_LIMIT_RPM (default 10).
# Only applied to expensive endpoints: /api/chat, /api/debates, /api/recipes.

import time as _time
from collections import defaultdict as _defaultdict

_RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "10"))
_RATE_WINDOW = 60  # seconds
_rate_buckets: dict[str, list[float]] = _defaultdict(list)


def _check_rate_limit(key: str) -> bool:
    """Return True if the request should be allowed, False if rate-limited."""
    if _RATE_LIMIT_RPM <= 0:
        return True  # disabled
    now = _time.monotonic()
    bucket = _rate_buckets[key]
    # Prune old entries
    cutoff = now - _RATE_WINDOW
    _rate_buckets[key] = bucket = [t for t in bucket if t > cutoff]
    if len(bucket) >= _RATE_LIMIT_RPM:
        return False
    bucket.append(now)
    return True


_RATE_LIMITED_PATHS = {"/api/chat", "/api/debates", "/api/recipes"}

app = FastAPI()


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        if request.method in ("POST",) and request.url.path in _RATE_LIMITED_PATHS:
            # Key by session_id from JSON body if available, else by client IP
            key = request.client.host if request.client else "unknown"
            if not _check_rate_limit(key):
                return StarletteResponse(
                    content="Rate limit exceeded. Try again shortly.",
                    status_code=429,
                    headers={"Retry-After": str(_RATE_WINDOW)},
                )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


@app.on_event("startup")
async def _startup():
    # Scheduler must start AFTER the FastAPI event loop is running so
    # AsyncIOScheduler hooks the same loop the cron job needs to use.
    start_scheduler()
    # Clean up stale sessions from previous runs
    try:
        deleted = _cleanup_old_sessions()
        if deleted:
            logging.info("session cleanup: removed %d stale sessions", deleted)
    except Exception:
        logging.exception("session cleanup failed (non-fatal)")
    # Chart smoke-test: every container restart fires plot_ohlc_chart
    # for one US + one Indian ticker. Surfaces tz/dtype/upstream
    # regressions immediately rather than waiting for a debate to hit
    # them. Fire-and-forget so a slow upstream doesn't block startup.
    asyncio.create_task(_chart_smoke_test())


async def _chart_smoke_test() -> None:
    """Run the ohlc_chart diagnostic and log results.

    Mirrors `findata-diagnostics --only ohlc_chart` but fires inside the
    finagent process, where the chart actually runs at debate time.
    Catching regressions here instead of in data-mcp's container is
    important because finagent's pip install of `git+...data-mcp.git`
    can serve a stale layer if the deploy didn't use --no-cache.
    """
    # Tiny grace period — let FastAPI finish bringing the rest of the
    # app online before we hit yfinance. No correctness reason; just
    # keeps the startup log readable.
    await asyncio.sleep(2.0)
    try:
        from findata.ohlc_chart import plot_ohlc_chart
    except ImportError as e:
        logging.warning("chart_smoke: findata.ohlc_chart unavailable (%s)", e)
        return

    for ticker in ("AAPL", "RELIANCE.NS"):
        try:
            out = await asyncio.to_thread(
                plot_ohlc_chart, ticker,
                lookback_days=60, with_sr=False, with_indicators=False,
            )
        except Exception as e:
            logging.error(
                "chart_smoke: %s raised %s: %s — investigate immediately "
                "(this means plot_ohlc_chart's own try/except didn't catch a "
                "code path it should have)",
                ticker, type(e).__name__, e,
            )
            continue
        status = out.get("chart_status", "?")
        if status == "ok":
            logging.info("chart_smoke: %s OK", ticker)
        elif status == "no_data":
            logging.warning(
                "chart_smoke: %s no_data — likely yfinance rate-limit, "
                "not a code bug", ticker,
            )
        else:
            logging.error(
                "chart_smoke: %s status=%s summary=%r — REGRESSION, "
                "investigate before charts ship to users",
                ticker, status, (out.get("summary") or "")[:160],
            )


@app.on_event("shutdown")
async def _shutdown():
    stop_scheduler()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── SQLite-backed session store ──────────────────────────────────────────
# Replaces the old in-memory `_sessions: dict` so sessions survive
# process restarts. DB lives next to the outputs directory.

import sqlite3

_SESSION_DB_PATH = Path(os.environ.get("SESSION_DB_PATH", "outputs/sessions.db"))
_SESSION_TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "7"))


def _init_session_db() -> None:
    _SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_SESSION_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                notebook_path TEXT,
                history TEXT DEFAULT '[]',
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.commit()


def _get_session(sid: str) -> dict:
    with sqlite3.connect(str(_SESSION_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT notebook_path, history FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
    if row:
        return {"notebook_path": row[0], "history": json.loads(row[1] or "[]")}
    return _create_session(sid)


def _create_session(sid: str) -> dict:
    with sqlite3.connect(str(_SESSION_DB_PATH)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (sid,)
        )
        conn.commit()
    return {"notebook_path": None, "history": []}


def _update_session(sid: str, notebook_path: str | None, history: list) -> None:
    with sqlite3.connect(str(_SESSION_DB_PATH)) as conn:
        conn.execute(
            "UPDATE sessions SET notebook_path = ?, history = ?, updated_at = strftime('%s', 'now') WHERE session_id = ?",
            (notebook_path, json.dumps(history), sid),
        )
        conn.commit()


def _delete_session(sid: str) -> None:
    with sqlite3.connect(str(_SESSION_DB_PATH)) as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        conn.commit()


def _cleanup_old_sessions() -> int:
    """Delete sessions older than SESSION_TTL_DAYS. Returns count deleted."""
    cutoff = _time.time() - (_SESSION_TTL_DAYS * 86400)
    with sqlite3.connect(str(_SESSION_DB_PATH)) as conn:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
        )
        conn.commit()
        return cursor.rowcount


_init_session_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/session")
async def new_session():
    sid = str(uuid.uuid4())
    _create_session(sid)
    return {"session_id": sid}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id
    session = _get_session(sid)

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def _run():
        try:
            result = await run_workflow(
                WorkflowInput(input_as_text=req.message),
                existing_notebook_path=session["notebook_path"],
                prior_history=session["history"],
                progress_cb=progress_queue.put,
            )
            # Update session state
            notebook_path = result.get("notebook_path") or session["notebook_path"]
            history = list(session["history"])
            history.append(
                {"role": "user", "content": [{"type": "input_text", "text": req.message}]}
            )
            history.append(
                {"role": "assistant", "content": [{"type": "output_text", "text": result.get("output_text", "")}]}
            )
            _update_session(sid, notebook_path, history)
            await progress_queue.put({
                "type": "done",
                "mode": result.get("mode", "new"),
                "notebook_path": notebook_path,
                "summary": result.get("output_text", ""),  # used by question mode
            })
        except Exception as e:
            logging.exception("/api/chat run_workflow failed")
            await progress_queue.put(_client_error_payload(e))

    asyncio.create_task(_run())

    async def _stream():
        while True:
            try:
                update = await asyncio.wait_for(progress_queue.get(), timeout=600)
                yield f"data: {json.dumps(update)}\n\n"
                if update["type"] in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timed out'})}\n\n"
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/notebook/{session_id}")
async def download_notebook(session_id: str):
    session = _get_session(session_id)
    if not session["notebook_path"]:
        return {"error": "No notebook for this session"}
    path = Path(session["notebook_path"])
    if not path.exists():
        return {"error": "Notebook file not found"}
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    _delete_session(session_id)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Notebook browsing endpoints. The Synapse web app proxies these (auth-gated)
# so users can list and preview every notebook the agent has produced. The
# outputs directory is the docker volume mount point — `/app/outputs` in the
# container — so listings survive container restarts.
# ─────────────────────────────────────────────────────────────────────────────


_OUTPUTS_DIR = (Path(__file__).parent / "outputs").resolve()


# Cache parsed notebook metadata so repeated /api/notebooks calls don't re-read
# every .ipynb. Key: (filename, mtime). Value: parsed summary dict. When the
# file's mtime changes the old key won't match and we re-parse — no disk cache,
# plain in-memory dict.
_notebook_summary_cache: dict[tuple[str, float], dict] = {}


def _extract_notebook_summary(path: Path) -> dict:
    """Parse a notebook and pull out recipe identity + headline metrics.

    Returns a dict with the spec-defined shape (recipe_name, project, template,
    fingerprint, run_id, headline_metrics). On parse failure or missing
    metadata, fields default to None and headline_metrics is an empty dict —
    never raises, just logs a warning.

    Uses ``json.load`` (not ``nbformat.read``) since we only need the top-level
    metadata dict and don't want to pay validation cost for every cell.
    """
    blank = {
        "recipe_name": None,
        "project": None,
        "template": None,
        "fingerprint": None,
        "run_id": None,
        "headline_metrics": {},
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, ValueError) as e:
        logging.warning("notebook %s: could not parse JSON (%s)", path.name, e)
        return blank

    nb_meta = nb.get("metadata") if isinstance(nb, dict) else None
    if not isinstance(nb_meta, dict):
        return blank

    run_summary = nb_meta.get("finagent_run_summary")
    if not isinstance(run_summary, dict):
        run_summary = {}

    raw_metrics = run_summary.get("headline_metrics")
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    # NaN passes isinstance(_, float) and so would land in the response
    # payload — Starlette's JSONResponse uses allow_nan=False (strict JSON
    # spec) and 502s the whole notebooks endpoint when it encounters one.
    def _finite_or_none(v: object) -> float | int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)) and not (
            isinstance(v, float) and not math.isfinite(v)
        ):
            return v
        return None

    headline_metrics = {
        "sharpe": _finite_or_none(raw_metrics.get("sharpe")),
        "annual_return": _finite_or_none(raw_metrics.get("annual_return")),
        "max_drawdown": _finite_or_none(raw_metrics.get("max_drawdown")),
    }

    def _str_or_none(v: object) -> str | None:
        return v if isinstance(v, str) else None

    return {
        "recipe_name": _str_or_none(nb_meta.get("recipe_name")),
        "project": _str_or_none(run_summary.get("project")),
        "template": _str_or_none(run_summary.get("template")),
        "fingerprint": _str_or_none(nb_meta.get("recipe_fingerprint")),
        "run_id": _str_or_none(run_summary.get("run_id")),
        "headline_metrics": headline_metrics,
    }


def _notebook_summary_cached(path: Path, mtime: float) -> dict:
    """Memoized wrapper around ``_extract_notebook_summary`` keyed by (name, mtime)."""
    key = (path.name, mtime)
    cached = _notebook_summary_cache.get(key)
    if cached is not None:
        return cached
    summary = _extract_notebook_summary(path)
    _notebook_summary_cache[key] = summary
    return summary


def _safe_notebook_path(name: str) -> Path:
    """Resolve `name` inside the outputs directory or raise 404/403.

    Strips any path separators, resolves, then asserts the result still lives
    under the outputs root — defense against `../etc/passwd`-style traversal.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid notebook name")
    if not name.endswith(".ipynb"):
        raise HTTPException(status_code=400, detail="not a notebook")
    candidate = (_OUTPUTS_DIR / name).resolve()
    try:
        candidate.relative_to(_OUTPUTS_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="path escapes outputs")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="notebook not found")
    return candidate


@app.get("/api/notebooks")
async def list_notebooks():
    """Return every .ipynb in the outputs dir, newest first.

    Each entry now carries the recipe identity (name/project/template/
    fingerprint/run_id) and headline metrics (sharpe, annual_return) parsed
    out of the notebook's metadata + FINAGENT_RUN_SUMMARY marker — so the UI
    can render a useful list once a project has more than a handful of runs.
    Parsing is cached per (path, mtime); a stale entry is dropped when the
    file's mtime moves.
    """
    from finagent.experiments import get_store

    if not _OUTPUTS_DIR.exists():
        return {"notebooks": []}

    # Single read of the runs table — much cheaper than per-notebook lookups.
    # Index by run_id AND by recipe_hash (fingerprint) since notebook metadata
    # uses the latter as a fallback when no explicit run_id is stamped.
    runs_by_key: dict[str, dict] = {}
    try:
        for r in get_store().list_runs(limit=2000):
            d = r.as_public_dict()
            runs_by_key[r.id] = d
            if r.recipe_hash:
                runs_by_key.setdefault(r.recipe_hash, d)
    except Exception:
        logging.exception("could not load runs index for notebooks list")

    items = []
    seen_keys: set[tuple[str, float]] = set()
    for p in _OUTPUTS_DIR.glob("*.ipynb"):
        try:
            st = p.stat()
        except OSError:
            continue
        summary = _notebook_summary_cached(p, st.st_mtime)
        seen_keys.add((p.name, st.st_mtime))
        entry: dict = {
            "name": p.name,
            "size": st.st_size,
            "mtime": st.st_mtime,
            **summary,
        }
        # Join with the runs table to surface audit verdict + run status.
        # Frontend uses these to render the Quarantined / Flagged tabs and
        # to decide whether a notebook is eligible for comparison/export.
        run = None
        rid = entry.get("run_id")
        if rid and rid in runs_by_key:
            run = runs_by_key[rid]
        elif entry.get("fingerprint") and entry["fingerprint"] in runs_by_key:
            run = runs_by_key[entry["fingerprint"]]
        if run is not None:
            audit = run.get("bias_audit") or {}
            entry["bias_audit_verdict"] = audit.get("verdict") if audit else None
            entry["run_status"] = run.get("status")
            # Surface flag count too — Quarantined-style filtering can also
            # key on plausibility flags, not just audit verdict.
            entry["flag_count"] = len(run.get("metrics_flags") or {})
        items.append(entry)
    # Drop cache entries for notebooks that no longer exist on disk (or whose
    # mtime moved) so the cache tracks the directory rather than growing
    # forever.
    for stale in _notebook_summary_cache.keys() - seen_keys:
        _notebook_summary_cache.pop(stale, None)
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"notebooks": items}


@app.get("/api/notebooks/{name}")
async def get_notebook(name: str):
    """Return the raw .ipynb JSON so the UI can render it cell-by-cell."""
    path = _safe_notebook_path(name)
    return FileResponse(
        path,
        media_type="application/json",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/notebooks/{name}/download")
async def download_notebook_by_name(name: str):
    """Same file but with Content-Disposition forcing a download."""
    path = _safe_notebook_path(name)
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/x-ipynb+json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Recipes / Projects / Runs.
#
# A Recipe is a YAML config that pins data, target, features, model, and
# evaluation. Submitting one creates a Run record (SQLite-backed) and queues
# compilation + execution. The workflow harvests metrics from the run-summary
# marker the templated notebook prints, and stashes them on the Run so the
# Project page can render a sortable table.
# ─────────────────────────────────────────────────────────────────────────────


class _RecipeSubmission(BaseModel):
    yaml: str


@app.post("/api/recipes")
async def submit_recipe(req: _RecipeSubmission):
    """Compile and execute a recipe, return its run id + final status."""
    from finagent.recipe_workflow import run_recipe

    # Run on a worker thread — kernel boot + cell execution is sync and slow.
    result = await asyncio.to_thread(run_recipe, recipe_yaml=req.yaml)
    return result


@app.get("/api/projects")
async def list_projects():
    from finagent.experiments import get_store

    return {"projects": get_store().list_projects()}


@app.get("/api/projects/{name}/runs")
async def list_project_runs(name: str):
    from finagent.experiments import get_store

    runs = [r.as_public_dict() for r in get_store().list_runs(project=name)]
    return {"project": name, "runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    from finagent.experiments import get_store

    run = get_store().get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run.as_public_dict()


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    from finagent.experiments import get_store

    get_store().delete_run(run_id)
    return {"ok": True}


class _TagsBody(BaseModel):
    tags: List[str]


@app.patch("/api/runs/{run_id}/tags")
async def patch_run_tags(run_id: str, body: _TagsBody):
    """Replace the run's tag list. Trims, lowercases, dedupes, and caps
    at 12 tags / 32 chars each — a workflow signal, not a search index.
    """
    from finagent.experiments import get_store

    store = get_store()
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in body.tags[:24]:  # input cap before dedupe
        t = (raw or "").strip().lower()[:32]
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
        if len(cleaned) >= 12:
            break
    store.update_run_tags(run_id, json.dumps(cleaned))
    return {"ok": True, "tags": cleaned}


class _DebateSubmission(BaseModel):
    ticker: str
    asset_class: str = "us_equity"
    rounds: int = 2


# ── Signals (Phase 4 dashboard) ────────────────────────────────────────
#
# Notebooks publish signals via ``panel.export_signal(...)`` (writes
# parquet + manifest + DB row). The endpoints below are read-side
# helpers for the dashboard — list signals, fetch one signal's series
# for plotting, mark active/paused/archived. The "Add to dashboard"
# button is just status='paused' → status='active' once the user has
# eyeballed the signal.

@app.get("/api/signals")
async def list_signals_route(
    project: Optional[str] = None,
    status: Optional[str] = None,
):
    """List all registered signals, optionally filtered by project /
    status. Sort: most recently updated first."""
    from finagent.experiments import list_signals_db

    signals = [s.as_public_dict() for s in list_signals_db(project=project, status=status)]
    return {"signals": signals}


@app.get("/api/signals/{name}")
async def get_signal_route(name: str, versions: int = 0):
    """Fetch one signal's manifest + (optionally) its retraining
    history. ``versions=N`` returns the most recent N retrain events."""
    from finagent.experiments import get_signal_db, list_signal_versions_db

    sig = get_signal_db(name)
    if not sig:
        raise HTTPException(status_code=404, detail="signal not found")
    out = sig.as_public_dict()
    if versions > 0:
        out["versions"] = list_signal_versions_db(sig.id, limit=int(versions))
    return out


@app.get("/api/signals/{name}/series")
async def get_signal_series_route(name: str, tail: int = 0):
    """Return the signal's parquet contents as JSON-serializable rows.
    ``tail=N`` returns only the last N points; default returns all.

    The dashboard chart renderer hits this endpoint on every page-load,
    so callers should pass ``tail`` (~252 for one year of daily) when
    they don't need the full history.
    """
    try:
        import pandas as pd  # noqa: WPS433
        from panel import load_signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"panel unavailable: {e}")
    try:
        series = load_signal(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="signal series not found on disk")
    if tail > 0:
        series = series.tail(int(tail))
    points = [
        {"ts": ts.isoformat(), "value": float(v) if pd.notna(v) else None}
        for ts, v in series.items()
    ]
    return {"name": name, "n_points": len(points), "points": points}


class _SignalStatusBody(BaseModel):
    status: str  # 'active' | 'paused' | 'archived'


@app.patch("/api/signals/{name}/status")
async def patch_signal_status_route(name: str, body: _SignalStatusBody):
    """Update a signal's status. Used by the 'Add to dashboard'
    button (paused → active), 'Pause', and 'Archive' actions."""
    from finagent.experiments import update_signal_status_db

    try:
        ok = update_signal_status_db(name, body.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="signal not found")
    return {"ok": True, "name": name, "status": body.status}


@app.post("/api/debates")
async def start_debate(req: _DebateSubmission):
    """Kick off a bull/bear/moderator debate and stream events as SSE.

    Each line on the stream is a JSON event:
      {"type":"phase","phase":"bull_round_1","speaker":"bull_analyst",...}
      {"type":"message","speaker":"bull_analyst","phase":"bull_round_1","text":"..."}
      {"type":"verdict","data":{...DebateVerdict...}}
      {"type":"error","text":"..."}
      {"type":"done","debate_id":"..."}

    The debate is also persisted on the debates table so the frontend
    can re-render it after the stream closes.
    """
    from finagent.debate import run_debate
    from finagent.experiments import get_store

    store = get_store()
    rounds = max(1, min(4, int(req.rounds)))
    debate = store.create_debate(
        ticker=req.ticker.strip().upper()[:32],
        asset_class=(req.asset_class or "us_equity").strip().lower()[:32],
        rounds=rounds,
    )

    queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()

    async def _emit(evt: dict) -> None:
        await queue.put(evt)

    transcript_accum: list[dict] = []
    verdict_accum: dict | None = None

    async def _runner():
        nonlocal verdict_accum
        try:
            store.update_debate(debate.id, status="running")
            result = await run_debate(
                ticker=debate.ticker,
                asset_class=debate.asset_class,
                rounds=debate.rounds,
                emit=_emit_and_capture,
                debate_id=debate.id,
            )
            verdict_accum = result.get("verdict")
            store.update_debate(
                debate.id,
                status="completed",
                transcript=transcript_accum,
                verdict=verdict_accum,
                finished=True,
            )
        except Exception as exc:
            logging.exception("debate %s crashed", debate.id)
            store.update_debate(
                debate.id,
                status="failed",
                transcript=transcript_accum,
                error=str(exc),
                finished=True,
            )
        finally:
            await queue.put(None)  # sentinel: stream done

    async def _emit_and_capture(evt: dict) -> None:
        # Capture for persistence AND forward to the client stream.
        if evt.get("type") == "message":
            transcript_accum.append({
                "speaker": evt.get("speaker"),
                "phase": evt.get("phase"),
                "text": evt.get("text"),
                "ts": evt.get("ts"),
            })
        elif evt.get("type") == "verdict":
            # Persist incrementally so a stream reconnect can hit /get_debate
            # and see the verdict already saved.
            v = evt.get("data")
            if isinstance(v, dict):
                store.update_debate(debate.id, verdict=v, transcript=transcript_accum)
        await _emit(evt)

    asyncio.create_task(_runner())

    async def _stream():
        # Header event so the client immediately knows the debate id.
        yield f"data: {json.dumps({'type': 'started', 'debate_id': debate.id, 'ticker': debate.ticker, 'asset_class': debate.asset_class, 'rounds': debate.rounds})}\n\n"
        while True:
            evt = await queue.get()
            if evt is None:
                yield f"data: {json.dumps({'type': 'done', 'debate_id': debate.id})}\n\n"
                return
            try:
                yield f"data: {json.dumps(evt, default=str)}\n\n"
            except Exception:
                # Belt-and-braces: never let a bad event break the stream.
                continue

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/debates")
async def list_debates(ticker: Optional[str] = None, limit: int = 50):
    from finagent.experiments import get_store
    debates = get_store().list_debates(ticker=ticker, limit=max(1, min(200, int(limit))))
    return {"debates": [d.as_public_dict() for d in debates]}


@app.get("/api/debates/calendar")
async def debates_calendar(
    month: Optional[str] = None,  # "YYYY-MM"; default current month
    source: str = "scheduled",
):
    """Return debates grouped by date for a month-grid calendar view.

    The Nifty 50 daily-cron page consumes this. Filters to scheduled-source
    debates by default; pass source='all' or source='user' to widen.
    """
    from datetime import datetime, timezone, timedelta
    from finagent.experiments import get_store

    if month:
        try:
            year, mon = month.split("-")
            year, mon = int(year), int(mon)
        except Exception:
            raise HTTPException(status_code=400, detail="month must be YYYY-MM")
    else:
        now = datetime.now(timezone.utc)
        year, mon = now.year, now.month

    # Window covers the calendar's visible weeks (Sunday-anchored grid),
    # so a month view always has 5-6 rows pre-filled. Pad ±7 days.
    first = datetime(year, mon, 1, tzinfo=timezone.utc)
    last = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if mon == 12 else datetime(year, mon + 1, 1, tzinfo=timezone.utc)
    start_ts = (first - timedelta(days=7)).timestamp()
    end_ts = (last + timedelta(days=7)).timestamp()

    store = get_store()
    debates = store.list_debates(limit=1000)
    out_by_day: dict[str, list[dict]] = {}
    for d in debates:
        if not (start_ts <= d.started_at <= end_ts):
            continue
        if source != "all" and d.source != source:
            continue
        date_str = datetime.fromtimestamp(d.started_at, tz=timezone.utc).strftime("%Y-%m-%d")
        verdict = d.verdict() or {}
        out_by_day.setdefault(date_str, []).append({
            "id": d.id,
            "ticker": d.ticker,
            "asset_class": d.asset_class,
            "status": d.status,
            "started_at": d.started_at,
            "verdict_action": verdict.get("action"),
            "verdict_target": verdict.get("target_price"),
            "verdict_stoploss": verdict.get("stoploss"),
            "verdict_horizon": verdict.get("time_horizon"),
            "verdict_confidence": verdict.get("confidence"),
        })
    # Sort each day's bucket by started_at ascending so the row reads in
    # execution order.
    for k in out_by_day:
        out_by_day[k].sort(key=lambda r: r["started_at"])

    return {
        "month": f"{year:04d}-{mon:02d}",
        "source": source,
        "days": out_by_day,
    }


@app.post("/api/debates/scheduler/run-now")
async def trigger_scheduler_now(n: int = 5, rounds: int = 2):
    """Manually fire the daily Nifty cron (admin / smoke-test).

    Runs in the background — returns immediately with the queued
    ticker list so the caller can check progress on /api/debates.
    """
    from finagent.scheduler import select_least_recent, run_daily_nifty_debates

    selected = select_least_recent(max(1, min(50, int(n))))
    asyncio.create_task(run_daily_nifty_debates(n=len(selected), rounds=rounds))
    return {"queued": selected, "rounds": rounds}


@app.get("/api/debates/{debate_id}/performance")
async def debate_performance(debate_id: str):
    """Compute the actual return on a debate's underlying ticker since
    the debate landed, for the calendar's verdict-vs-reality column.

    Pulls historical OHLC for the window ``[debate_started, today]``,
    computes:
      * entry_price — close on the debate's first available trading day
      * latest_price — most recent close
      * return_pct — (latest - entry) / entry
      * direction_correct — True if the verdict pointed the right way
        (buy + positive return, sell + negative return); None when
        verdict is 'avoid' or 'unknown'
      * target_progress_pct — (current - entry) / (target - entry)
        for buy verdicts; clipped to [-2, 2]; None otherwise

    Falls back to a structured 'no_data' shape if the ticker can't be
    fetched (delisted, tz-mismatch, illiquid).
    """
    from datetime import datetime, timezone, timedelta
    from finagent.experiments import get_store

    d = get_store().get_debate(debate_id)
    if not d:
        raise HTTPException(status_code=404, detail="debate not found")

    verdict = d.verdict() or {}
    started = datetime.fromtimestamp(d.started_at, tz=timezone.utc)
    end = datetime.now(timezone.utc) + timedelta(days=1)

    out: dict = {
        "debate_id": debate_id,
        "ticker": d.ticker,
        "asset_class": d.asset_class,
        "started_at": d.started_at,
        "verdict_action": verdict.get("action"),
        "verdict_target": verdict.get("target_price"),
        "verdict_stoploss": verdict.get("stoploss"),
        "entry_price": None,
        "entry_date": None,
        "latest_price": None,
        "latest_date": None,
        "return_pct": None,
        "direction_correct": None,
        "target_progress_pct": None,
        "status": "no_data",
    }

    try:
        from findata.equity_prices import get_equity_prices
        df = await asyncio.to_thread(
            get_equity_prices,
            tickers=[d.ticker],
            start_date=(started - timedelta(days=2)).strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            fields=["Close"],
        )
    except Exception as exc:
        out["status"] = f"fetch_failed: {type(exc).__name__}"
        return out

    if df is None or df.empty:
        return out

    # Normalise to a flat Close series.
    import pandas as pd
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close = df["Close"][d.ticker].dropna()
        except Exception:
            close = pd.Series(dtype=float)
    else:
        close = df.get("Close", pd.Series(dtype=float)).dropna()

    if close.empty:
        return out

    # Entry: first close at or after started_at; latest: last close.
    started_naive = pd.Timestamp(started)
    if close.index.tz is not None and started_naive.tz is None:
        started_naive = started_naive.tz_localize("UTC")
    elif close.index.tz is None and started_naive.tz is not None:
        started_naive = started_naive.tz_localize(None)
    on_or_after = close[close.index >= started_naive]
    entry_series = on_or_after.iloc[:1] if not on_or_after.empty else close.iloc[:1]
    entry_price = float(entry_series.iloc[0])
    entry_date = entry_series.index[0].strftime("%Y-%m-%d")
    latest_price = float(close.iloc[-1])
    latest_date = close.index[-1].strftime("%Y-%m-%d")

    return_pct = (latest_price / entry_price) - 1.0 if entry_price else None

    direction_correct = None
    action = (verdict.get("action") or "").lower()
    if return_pct is not None and action in ("buy", "sell"):
        direction_correct = (
            (action == "buy" and return_pct > 0) or
            (action == "sell" and return_pct < 0)
        )

    target_progress_pct = None
    target = verdict.get("target_price")
    if (
        action == "buy"
        and isinstance(target, (int, float))
        and entry_price is not None
        and target != entry_price
    ):
        progress = (latest_price - entry_price) / (target - entry_price)
        target_progress_pct = max(-2.0, min(2.0, progress))

    out.update({
        "entry_price": entry_price,
        "entry_date": entry_date,
        "latest_price": latest_price,
        "latest_date": latest_date,
        "return_pct": return_pct,
        "direction_correct": direction_correct,
        "target_progress_pct": target_progress_pct,
        "status": "ok",
    })
    return out


@app.get("/api/debates/{debate_id}")
async def get_debate(debate_id: str):
    from finagent.experiments import get_store
    d = get_store().get_debate(debate_id)
    if not d:
        raise HTTPException(status_code=404, detail="debate not found")
    return d.as_public_dict()


@app.delete("/api/debates/{debate_id}")
async def delete_debate(debate_id: str):
    from finagent.experiments import get_store
    get_store().delete_debate(debate_id)
    return {"ok": True}


@app.get("/api/runs/{run_id}/tearsheet")
async def run_tearsheet(run_id: str):
    """Self-contained HTML tearsheet for a single run.

    Browser 'Save as PDF' produces a static copy suitable for the Friday
    investor memo. Server-side PDF generation deferred (weasyprint /
    headless-chromium add deployment complexity for a feature that
    browser print already covers).
    """
    from fastapi.responses import HTMLResponse
    from finagent.experiments import get_store
    from finagent.tearsheet import render_tearsheet

    run = get_store().get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    nb_path = Path(run.notebook_path) if run.notebook_path else None
    body = render_tearsheet(run.as_public_dict(), nb_path)
    return HTMLResponse(content=body)


# ─────────────────────────────────────────────────────────────────────────────
# Admin metrics dashboard.
#
# Tier-1 quality signals computed from the experiment store. Auth-gated at
# the synapse proxy layer (only the configured admin user reaches here).
# We deliberately don't enforce auth in finagent itself — keeps the backend
# composable; synapse decides who sees the dashboard.
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/admin/metrics")
async def admin_metrics(days: int = 7, keys: Optional[str] = None):
    """Compute Tier-1 metrics over a rolling window.

    Query params:
      days  — window length in days (default 7)
      keys  — comma-separated metric keys to include (default: all)
    """
    from finagent.metrics import compute_metrics

    if days <= 0 or days > 90:
        raise HTTPException(status_code=400, detail="days must be 1..90")
    selected = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
    return await asyncio.to_thread(compute_metrics, days=days, keys=selected)


@app.get("/admin/llm-config")
async def admin_llm_config():
    """Resolved (provider, model) per role — read by /api/admin/diagnostics
    on synapse so admins can see what model each agent is actually using.

    Surfaced post-L1 (model-swap dispatcher). Override per role via env:
      <ROLE>_PROVIDER=openai  <ROLE>_MODEL=gpt-5
      OPENAI_MODEL=gpt-4o     # global default applied when role-specific is unset
    """
    from finagent.llm import list_roles
    return {role: {"provider": p, "model": m} for role, (p, m) in list_roles().items()}


@app.get("/api/admin/costs")
async def admin_costs(days: int = 30):
    """Cost ledger summary — total $ spend + breakdowns by day, purpose,
    user, model, and top runs. Auth-gated at the synapse proxy layer.
    """
    from finagent.experiments import get_store

    days = max(1, min(365, int(days)))
    return await asyncio.to_thread(get_store().cost_summary, days)


@app.get("/api/admin/metrics/keys")
async def admin_metric_keys():
    """Return the registry of available metric keys for the toggle UI."""
    from finagent.metrics import list_metric_keys

    return {"keys": list_metric_keys()}


# ─────────────────────────────────────────────────────────────────────────────
# Template authoring (AI-drafted templates).
#
# A user describes a strategy in free text → POST /api/templates/draft runs
# the template-author agent, validates the contract via static AST + import
# check, writes the source to finagent/recipes/templates/_drafts/<slug>.py.
# Accepting a draft moves it out of _drafts/ so the registry's auto-
# discovery picks it up on next process start.
# Runtime verification (running compile() against a smoke recipe) is
# explicitly out of scope today.
# ─────────────────────────────────────────────────────────────────────────────


class _TemplateDraftRequest(BaseModel):
    description: str


@app.post("/api/templates/draft")
async def draft_template_endpoint(req: _TemplateDraftRequest):
    from finagent.templates_authoring import draft_template

    return await draft_template(req.description)


@app.get("/api/templates/drafts")
async def list_template_drafts():
    from finagent.templates_authoring import list_drafts

    return {"drafts": list_drafts()}


@app.post("/api/templates/drafts/{slug}/accept")
async def accept_template_draft(slug: str):
    from finagent.templates_authoring import accept_draft

    result = accept_draft(slug)
    if result.get("status") != "ok":
        raise HTTPException(status_code=400, detail=result.get("errors") or result)
    return result


@app.post("/api/templates/drafts/{slug}/reject")
async def reject_template_draft(slug: str):
    from finagent.templates_authoring import reject_draft

    return reject_draft(slug)


@app.get("/api/recipes/templates")
async def list_recipe_templates():
    """Return every registered template with its rich metadata.

    Each template module declares a METADATA dict (archetype, tagline,
    presets, etc.). The Recipe Builder uses this to render the gallery
    cards and pre-load preset YAML when the user picks a starting point.
    """
    from finagent.recipes.templates import REGISTRY

    out = []
    for name, module in REGISTRY.items():
        meta = getattr(module, "METADATA", None)
        if meta is None:
            out.append({"name": name})
        else:
            out.append({"name": name, **meta})
    return {"templates": out}


# ─────────────────────────────────────────────────────────────────────────────
# Searches.
#
# A Search is a budgeted sweep over recipe parameters driven by a hypothesis
# policy (random / grid for now; LLM-creative comes later). Every iteration
# is a real recipe submission backed by the existing run pipeline, so the
# resulting notebooks and lineage graphs work the same way.
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/api/searches")
async def submit_search(payload: dict):
    """Submit a search; returns the search id + final summary once done.

    Body shape mirrors finagent.searches.types.SearchSubmission.
    Synchronous from the caller's perspective — runs in a worker thread
    so the asyncio loop stays free for streaming chat etc.
    """
    from finagent.searches import SearchSubmission, execute_search

    try:
        sub = SearchSubmission.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid search: {exc}")

    return await asyncio.to_thread(execute_search, sub)


@app.get("/api/searches/{search_id}")
async def get_search(search_id: str):
    from finagent.experiments import get_store

    store = get_store()
    search = store.get_search(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="search not found")
    runs = [r.as_public_dict() for r in store.runs_in_search(search_id)]
    return {**search.as_public_dict(), "runs": runs}


@app.get("/api/projects/{name}/searches")
async def list_project_searches(name: str):
    from finagent.experiments import get_store

    searches = [s.as_public_dict() for s in get_store().list_searches(project=name)]
    return {"project": name, "searches": searches}


@app.post("/api/notebooks/{name}/run")
async def run_notebook_all_cells(name: str):
    """Re-execute every code cell in a notebook and persist outputs back.

    Unlike the validator agent's `validate_run` (which stops at the first
    error so it can fix it), this endpoint keeps going past errors so users
    get as much output as possible from the "Run all" button.
    """
    from finagent.functions.notebook_tools import run_all_cells_to_disk

    path = _safe_notebook_path(name)
    # Run in a worker thread — kernel boot + execution is sync and can take
    # tens of seconds; don't block the asyncio loop.
    result = await asyncio.to_thread(run_all_cells_to_disk, str(path), 180)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error") or "run failed")
    return result


@app.get("/api/notebooks/{name}/lineage")
async def get_notebook_lineage(name: str, method: str = "ast", refresh: bool = False):
    """Return a data-lineage graph for a notebook.

    method=ast      → static analysis (fast, always available)
    method=runtime  → runtime-traced lineage (requires the notebook to
                      execute cleanly in a fresh subprocess; slower)

    By default we read the cached graph from `nb.metadata['finagent_lineage'][method]`
    if it was computed during the workflow. Pass `refresh=true` to recompute.
    """
    import nbformat as _nbformat
    from finagent.lineage import extract_lineage_ast, extract_lineage_runtime

    if method not in ("ast", "runtime"):
        raise HTTPException(status_code=400, detail="method must be 'ast' or 'runtime'")

    path = _safe_notebook_path(name)
    if not refresh:
        try:
            with open(path, "r", encoding="utf-8") as f:
                nb = _nbformat.read(f, as_version=4)
            cached = (nb.metadata.get("finagent_lineage") or {}).get(method)
            if cached:
                return cached
        except Exception:
            logging.exception("could not read cached lineage for %s", name)

    extractor = extract_lineage_ast if method == "ast" else extract_lineage_runtime
    try:
        return extractor(str(path))
    except Exception as exc:
        logging.exception("lineage extraction failed for %s method=%s", name, method)
        raise HTTPException(status_code=500, detail=f"lineage failed: {exc}")


_VECTOR_STORE_ID = os.environ.get(
    "OPENAI_VECTOR_STORE_ID", "vs_69a81b0197a481919e14c2d66197af7d"
)
_UPLOAD_MAX_BYTES = 25 * 1024 * 1024  # 25 MB

@app.post("/uploads/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=415, detail="only PDF files are accepted")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"PDF too large (max {_UPLOAD_MAX_BYTES // (1024*1024)} MB)")

    filename = file.filename or "upload.pdf"
    # PDF upload uses the OpenAI Files + Vector Stores APIs specifically;
    # they don't have a clean LangChain / Anthropic equivalent, so this
    # one stays OpenAI-pinned even after the L3 migration. Routing
    # through the dispatcher anyway for telemetry consistency.
    from finagent.llm import get_llm_client
    client = get_llm_client("default")

    try:
        uploaded = await client.files.create(
            file=(filename, data, "application/pdf"),
            purpose="assistants",
        )
    except Exception as e:
        logging.exception("OpenAI Files.create failed for %s", filename)
        raise HTTPException(status_code=502, detail=f"files.create failed: {e}")

    try:
        vsf = await client.vector_stores.files.create(
            vector_store_id=_VECTOR_STORE_ID,
            file_id=uploaded.id,
        )
    except Exception as e:
        logging.exception("OpenAI vector_stores.files.create failed for %s", filename)
        raise HTTPException(status_code=502, detail=f"vector_stores.files.create failed: {e}")

    return {
        "filename": filename,
        "file_id": uploaded.id,
        "vector_store_id": _VECTOR_STORE_ID,
        "vector_store_file_id": vsf.id,
        "status": getattr(vsf, "status", "queued"),
        "bytes": len(data),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synapse web-compat endpoint.
#
# The Synapse Next.js app POSTs to /chat with {messages, user} and reads the
# body as plain text chunks (see FINAGENT_API.md in the synapse repo). This
# endpoint translates that contract onto run_workflow(): sessions are keyed by
# user.id so multi-turn notebook edits persist across browser tabs.
# ─────────────────────────────────────────────────────────────────────────────


class _WebMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class _WebUser(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None


class _WebChatRequest(BaseModel):
    messages: List[_WebMessage]
    user: _WebUser


def _to_internal_history(messages: List[_WebMessage]) -> list:
    """Convert web message list → agent SDK conversation history items."""
    history = []
    for m in messages:
        content_type = "output_text" if m.role == "assistant" else "input_text"
        history.append({
            "role": m.role,
            "content": [{"type": content_type, "text": m.content}],
        })
    return history


def _hint_for(e: Exception) -> Optional[str]:
    """Map an exception to a one-sentence remediation hint for the end user.

    Pattern-matches on the exception class and message. Returns None when no
    actionable hint applies — callers should treat that as "no hint" and not
    surface a placeholder. Class names are matched by string so we don't need
    to import optional SDKs (openai, anthropic) just to do isinstance checks.
    """
    cls = e.__class__.__name__
    msg = str(e).lower()

    if cls in ("ImportError", "ModuleNotFoundError"):
        return (
            "A required module is missing from the pinned environment. "
            "Try a recipe instead, or contact ops."
        )
    if cls == "TimeoutError" or isinstance(e, asyncio.TimeoutError):
        return (
            "The agent took too long. Try a recipe build (faster, "
            "deterministic) instead of free-form chat."
        )
    if cls in ("KeyError", "AttributeError"):
        return (
            "The agent referenced something that doesn't exist. This is "
            "usually a prompt-vs-tool mismatch — please screenshot the "
            "request and report it."
        )
    if cls == "RuntimeError" and ("rate limit" in msg or "quota" in msg):
        return "LLM provider rate-limited the request. Wait 30s and retry."
    return None


def _client_error_payload(e: Exception) -> dict:
    """Build the SSE/stream error event surfaced to the browser.

    Contract (consumed by the Synapse frontend in U7):
        {
            "type": "error",
            "error_class": <exception class name>,
            "message": <first line of str(e), truncated to 200 chars>,
            "hint": <remediation string or None>,
        }

    Tracebacks are deliberately *not* included — they are a security boundary.
    Full tracebacks are still emitted to server logs via logging.exception.
    """
    raw = str(e)
    message = raw.splitlines()[0][:200] if raw else "(no message)"
    return {
        "type": "error",
        "error_class": e.__class__.__name__,
        "message": message,
        "hint": _hint_for(e),
    }


@app.post("/chat")
async def chat_web(req: _WebChatRequest):
    # Last user message = new input. Everything before it = prior history.
    last_user_idx: Optional[int] = None
    for i in range(len(req.messages) - 1, -1, -1):
        if req.messages[i].role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        raise HTTPException(status_code=400, detail="request must contain at least one user message")

    last_user = req.messages[last_user_idx].content
    prior_history = _to_internal_history(req.messages[:last_user_idx])

    uid = req.user.id
    session = _sessions.setdefault(uid, {"notebook_path": None, "history": []})

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def _run():
        try:
            result = await run_workflow(
                WorkflowInput(input_as_text=last_user),
                existing_notebook_path=session["notebook_path"],
                # Web is authoritative for chat history; use what the client sent.
                prior_history=prior_history,
                progress_cb=progress_queue.put,
            )
            session["notebook_path"] = result.get("notebook_path") or session["notebook_path"]
            await progress_queue.put({
                "type": "done",
                "text": result.get("output_text", ""),
                "mode": result.get("mode", "new"),
                "notebook_path": session["notebook_path"],
            })
        except Exception as e:
            logging.exception("/chat run_workflow failed")
            await progress_queue.put(_client_error_payload(e))

    asyncio.create_task(_run())

    async def _stream():
        while True:
            try:
                update = await asyncio.wait_for(progress_queue.get(), timeout=600)
            except asyncio.TimeoutError:
                yield "\n\n**Error:** request timed out after 10 minutes.\n"
                break

            kind = update.get("type")
            if kind == "status":
                yield f"_{update.get('message', '')}_\n\n"
            elif kind == "trace":
                md = update.get("markdown", "")
                if md:
                    yield md + "\n\n"
            elif kind == "event":
                # Structured live events (phase/tool_call/notebook_outline/...).
                # Encoded as an HTML comment so ReactMarkdown drops it from
                # rendered output but the synapse client can sniff it from the
                # raw stream and surface it in the StepProgress / Outline UI.
                payload = update.get("data", {})
                try:
                    encoded = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    continue
                yield f"<!--FINAGENT_EVT {encoded}-->\n"
            elif kind == "done":
                text = update.get("text", "") or ""
                if text:
                    yield text
                nb_path = update.get("notebook_path")
                if nb_path:
                    yield f"\n\n---\n*Notebook updated: `{Path(nb_path).name}`*\n"
                break
            elif kind == "error":
                # Structured error contract (consumed by Synapse frontend in U7):
                # {type, error_class, message, hint}. Emit as a FINAGENT_EVT
                # comment so the client can sniff the raw stream, and also
                # render a plaintext fallback for older clients.
                try:
                    encoded = json.dumps(update, ensure_ascii=False)
                    yield f"<!--FINAGENT_EVT {encoded}-->\n"
                except Exception:
                    pass
                msg = update.get("message", "unknown error")
                hint = update.get("hint")
                err_cls = update.get("error_class")
                prefix = f"**Error ({err_cls}):**" if err_cls else "**Error:**"
                yield f"\n\n{prefix} {msg}\n"
                if hint:
                    yield f"\n_Hint: {hint}_\n"
                break
            else:
                # Unknown event type — surface a compact debug line rather than drop it.
                yield f"\n\n_{json.dumps(update)}_\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
