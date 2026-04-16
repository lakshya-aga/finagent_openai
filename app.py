from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
