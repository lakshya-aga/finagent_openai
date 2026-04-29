"""Streaming hooks: emit live, structured events for every agent tool call.

The original workflow only published trace markdown *after* each phase
completed, which meant the user stared at "Building notebook…" for minutes
with no incremental signal. ``StreamingHooks`` taps into the Agents SDK
``RunHooks`` lifecycle so the frontend gets a fresh event the moment a tool
fires, plus phase boundaries.

Event shape (delivered to ``progress_cb`` as ``{"type": "event", "data": ...}``)::

    {"type": "phase",          "phase": "plan|build|validate|...", "state": "start|end"}
    {"type": "tool_call",      "phase": "...", "agent": "...", "tool": "add_cell",
                                "state": "start|end", "snippet": "...", "args": "..."}
    {"type": "agent_lifecycle","phase": "...", "agent": "...", "state": "start|end"}
    {"type": "notebook_outline","path": "...", "cells": [{"idx", "type", "node_id", "rationale"}]}

Snippets and arg blobs are truncated to 200 chars each so the wire stays
lightweight even when a tool returns a large payload.
"""

from __future__ import annotations

import json  # noqa: F401  used by on_llm_end's args serialiser
import logging
from typing import Any, Awaitable, Callable, Optional

from agents import RunHooks


_TRUNC = 200


def _truncate(s: str, n: int = _TRUNC) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


class StreamingHooks(RunHooks):
    """RunHooks subclass that pushes structured events into a progress queue."""

    def __init__(
        self,
        progress_cb: Optional[Callable[[dict], Awaitable[None]]],
        phase: str,
    ) -> None:
        self._cb = progress_cb
        self._phase = phase
        # Cache the most-recent argument blob per tool-call so on_tool_end can
        # quote the args (the SDK only hands them to us during on_llm_end as
        # part of the response items, so we sniff from there too).
        self._last_args: dict[str, str] = {}

    async def _emit(self, payload: dict) -> None:
        if not self._cb:
            return
        try:
            await self._cb({"type": "event", "data": payload})
        except Exception:
            logging.exception("StreamingHooks emit failed")

    # ── lifecycle ────────────────────────────────────────────────────────
    async def on_agent_start(self, context, agent) -> None:
        await self._emit({
            "type": "agent_lifecycle",
            "phase": self._phase,
            "agent": getattr(agent, "name", "agent"),
            "state": "start",
        })

    async def on_agent_end(self, context, agent, output) -> None:
        await self._emit({
            "type": "agent_lifecycle",
            "phase": self._phase,
            "agent": getattr(agent, "name", "agent"),
            "state": "end",
        })

    # ── tool calls ───────────────────────────────────────────────────────
    async def on_tool_start(self, context, agent, tool) -> None:
        name = getattr(tool, "name", tool.__class__.__name__)
        await self._emit({
            "type": "tool_call",
            "phase": self._phase,
            "agent": getattr(agent, "name", "agent"),
            "tool": name,
            "state": "start",
        })

    async def on_tool_end(self, context, agent, tool, result) -> None:
        name = getattr(tool, "name", tool.__class__.__name__)
        snippet = _truncate(str(result) if result is not None else "")
        await self._emit({
            "type": "tool_call",
            "phase": self._phase,
            "agent": getattr(agent, "name", "agent"),
            "tool": name,
            "state": "end",
            "snippet": snippet,
        })

    # ── llm boundaries ───────────────────────────────────────────────────
    async def on_llm_end(self, context, agent, response) -> None:
        # Walk the LLM response and surface (a) reasoning summaries and
        # (b) the *arguments* of every function call the model just decided
        # on. The SDK emits `on_tool_start` shortly afterwards but doesn't
        # pass us the args, so this is the only hook with access to them.
        # Both event types are best-effort — the SDK doesn't promise a
        # stable item shape across versions.
        agent_name = getattr(agent, "name", "agent")
        try:
            for item in getattr(response, "output", []) or []:
                kind = getattr(item, "type", "")
                if kind == "reasoning":
                    summary = getattr(item, "summary", None) or []
                    bits = []
                    for s in summary:
                        text = getattr(s, "text", None)
                        if text:
                            bits.append(text)
                    if bits:
                        await self._emit({
                            "type": "reasoning",
                            "phase": self._phase,
                            "agent": agent_name,
                            "text": _truncate(" ".join(bits), 400),
                        })
                elif kind in ("function_call", "tool_call"):
                    name = getattr(item, "name", "") or "tool"
                    args = getattr(item, "arguments", "") or ""
                    call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
                    if isinstance(args, (dict, list)):
                        try:
                            args = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            args = str(args)
                    await self._emit({
                        "type": "tool_call_planned",
                        "phase": self._phase,
                        "agent": agent_name,
                        "tool": name,
                        "args": _truncate(str(args), 400),
                        "call_id": call_id,
                    })
        except Exception:
            pass


async def emit_phase(progress_cb, phase: str, state: str) -> None:
    """Convenience: emit a phase boundary event."""
    if not progress_cb:
        return
    await progress_cb({"type": "event", "data": {
        "type": "phase",
        "phase": phase,
        "state": state,
    }})


def build_notebook_outline(path: str) -> dict:
    """Read a saved .ipynb and produce a lightweight outline event payload.

    Pulls finagent provenance metadata (node_id, rationale) where present so
    the frontend can render "this cell came from DAG node n3_signal because X".
    """
    import nbformat

    cells_out = []
    try:
        nb = nbformat.read(open(path), as_version=4)
    except Exception as e:
        return {"type": "notebook_outline", "path": str(path), "error": str(e), "cells": []}

    for i, cell in enumerate(nb.cells):
        md = cell.metadata.get("finagent") if hasattr(cell, "metadata") else None
        title = ""
        source = cell.source or ""
        if cell.cell_type == "markdown":
            for line in source.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
            if not title:
                title = _truncate(source, 80)
        else:
            first = source.splitlines()[0] if source else ""
            title = _truncate(first, 80)

        cells_out.append({
            "idx": i,
            "type": cell.cell_type,
            "title": title,
            "node_id": (md or {}).get("node_id", ""),
            "rationale": (md or {}).get("rationale", ""),
        })

    return {"type": "notebook_outline", "path": str(path), "cells": cells_out}
