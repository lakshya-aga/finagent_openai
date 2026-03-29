from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
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
                "summary": result.get("output_text", ""),
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
