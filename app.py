from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from agent_workflow import run_workflow, WorkflowInput

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# session_id -> {"notebook_path": str | None, "history": list}
_sessions: dict = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/session")
async def new_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = {"notebook_path": None, "history": []}
    return {"session_id": sid}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id
    if sid not in _sessions:
        _sessions[sid] = {"notebook_path": None, "history": []}
    session = _sessions[sid]

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
            session["notebook_path"] = result.get("notebook_path") or session["notebook_path"]
            session["history"].append(
                {"role": "user", "content": [{"type": "input_text", "text": req.message}]}
            )
            session["history"].append(
                {"role": "assistant", "content": [{"type": "output_text", "text": result.get("output_text", "")}]}
            )
            await progress_queue.put({
                "type": "done",
                "mode": result.get("mode", "new"),
                "notebook_path": session["notebook_path"],
                "summary": result.get("output_text", ""),  # used by question mode
            })
        except Exception as e:
            await progress_queue.put({"type": "error", "message": str(e)})

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
    session = _sessions.get(session_id)
    if not session or not session["notebook_path"]:
        return {"error": "No notebook for this session"}
    path = Path(session["notebook_path"])
    if not path.exists():
        return {"error": "Notebook file not found"}
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Notebook browsing endpoints. The Synapse web app proxies these (auth-gated)
# so users can list and preview every notebook the agent has produced. The
# outputs directory is the docker volume mount point — `/app/outputs` in the
# container — so listings survive container restarts.
# ─────────────────────────────────────────────────────────────────────────────


_OUTPUTS_DIR = (Path(__file__).parent / "outputs").resolve()


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
    """Return every .ipynb in the outputs dir, newest first."""
    if not _OUTPUTS_DIR.exists():
        return {"notebooks": []}
    items = []
    for p in _OUTPUTS_DIR.glob("*.ipynb"):
        try:
            st = p.stat()
        except OSError:
            continue
        items.append({
            "name": p.name,
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
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
    client = AsyncOpenAI()

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
        except Exception:
            logging.exception("/chat run_workflow failed")
            await progress_queue.put({
                "type": "error",
                "message": "The research agent hit an unexpected error. Check server logs.",
            })

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
                yield f"\n\n**Error:** {update.get('message', 'unknown error')}\n"
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
