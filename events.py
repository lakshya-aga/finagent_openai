"""Typed agent events for streaming progress to UI clients.

The agent_workflow exposes its progress as a sequence of small dict events
(documented in FINAGENT_API.md). Each event has a ``type`` and event-specific
fields. The /chat endpoint serialises every event as one NDJSON line.

Event reference (kept in sync with synapse `agent-timeline.tsx`):

    {"type": "phase",     "name": "plan|build|validate|edit|answer"}
    {"type": "plan",      "nodes": [{"id","tool","description","depends_on"}]}
    {"type": "action",    "tool": str, "summary": str, "args": dict|None}
    {"type": "reasoning", "text": str}
    {"type": "fix",       "cell_index": int|None, "error_type": str|None,
                          "reasoning": str, "action": str, "result": str}
    {"type": "notebook",  "path": str, "num_cells": int}
    {"type": "answer",    "text": str}
    {"type": "error",     "message": str}
    {"type": "done",      "mode": str, "notebook_path": str|None}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# Deterministic tool→verb map. Keeps the timeline readable even when the
# model picks an obscure tool name. Values are present-tense phrases (the
# UI prepends a status icon, so "Adding cell" reads naturally).
ACTION_NAMES: Dict[str, str] = {
    # notebook plumbing
    "create_notebook": "Creating notebook",
    "add_cell": "Adding cell",
    "replace_cell": "Rewriting cell",
    "insert_cell": "Inserting cell",
    "delete_cell": "Deleting cell",
    "read_notebook": "Reading current notebook",
    "run_cell": "Running cell",
    "validate_run": "Running notebook end-to-end",
    "find_regex_in_notebook_code": "Searching notebook",
    "install_packages": "Installing packages",
    # fruit_thrower MCP
    "search_code": "Searching internal library",
    "get_unit_source": "Loading library source",
    "list_modules": "Listing internal modules",
    "get_module_summary": "Reading module summary",
    "index_repository": "Indexing repository",
    "get_index_stats": "Checking index stats",
    "generate_function": "Generating helper function",
    # data_mcp
    "search_tools": "Searching data tools",
    "get_tool_doc": "Reading tool docs",
    "list_all_tools": "Listing data tools",
    "request_data_source": "Requesting data source",
    # built-in tools
    "file_search": "Searching uploaded files",
}


def _coerce_args(raw: Any) -> Dict[str, Any]:
    """Tool-call arguments come as JSON strings, dicts, or pydantic objects."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    if hasattr(raw, "model_dump"):
        try:
            dumped = raw.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    return {}


def summarize_tool_call(name: str, args: Any) -> str:
    """One-line human description of a tool call.

    Prefers a hand-tuned verb from ACTION_NAMES, then enriches with the most
    informative argument (cell index, package list, query string, etc.). Falls
    back to the raw tool name so unknown tools still surface meaningfully.
    """
    verb = ACTION_NAMES.get(name, name.replace("_", " "))
    a = _coerce_args(args)

    if name in {"replace_cell", "insert_cell", "delete_cell", "run_cell"} and "cell_index" in a:
        return f"{verb} {a['cell_index']}"
    if name == "add_cell" and "cell_type" in a:
        return f"Adding {a['cell_type']} cell"
    if name == "install_packages" and isinstance(a.get("packages"), list):
        pkgs = ", ".join(str(p) for p in a["packages"][:6]) or "(none)"
        return f"Installing packages: {pkgs}"
    if name == "find_regex_in_notebook_code" and a.get("regex_pattern"):
        return f"Searching notebook for `{a['regex_pattern']}`"
    if name in {"search_code", "search_tools"} and a.get("query"):
        return f"{verb} for `{a['query']}`"
    if name == "get_tool_doc" and a.get("tool_name"):
        return f"Reading docs for `{a['tool_name']}`"
    if name == "get_unit_source" and a.get("name"):
        return f"Loading source for `{a['name']}`"
    if name == "request_data_source" and a.get("name"):
        return f"Requesting data source: `{a['name']}`"

    return verb


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_code_fences(raw: str) -> str:
    """Remove ```json ... ``` fences if the model wrapped its JSON output."""
    return _FENCE_RE.sub("", raw).strip()


def parse_plan(raw: str) -> List[Dict[str, Any]]:
    """Parse a planner JSON output into a list of DAG nodes.

    Returns ``[]`` if the output is not a JSON array of objects — the UI is
    happy to render an empty plan section, so silent failure is fine here.
    """
    if not raw:
        return []
    try:
        data = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and "nodes" in data and isinstance(data["nodes"], list):
        data = data["nodes"]
    if not isinstance(data, list):
        return []
    nodes = []
    for n in data:
        if not isinstance(n, dict):
            continue
        nodes.append({
            "id": n.get("id"),
            "tool": n.get("tool"),
            "description": n.get("description") or n.get("desc") or "",
            "depends_on": n.get("depends_on") or n.get("deps") or [],
        })
    return nodes


def parse_fixes(raw: str) -> List[Dict[str, Any]]:
    """Parse a validator JSON output into a list of fix events.

    Drops the trailing ``{"step": "FINAL", ...}`` sentinel — the workflow
    surfaces the final state via the ``answer`` event instead.
    """
    if not raw:
        return []
    try:
        data = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    fixes = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step", "")
        if isinstance(step, str) and step.upper() == "FINAL":
            continue
        fixes.append({
            "cell_index": entry.get("cell_index"),
            "error_type": entry.get("error_type"),
            "reasoning": entry.get("reasoning", ""),
            "action": entry.get("action", ""),
            "result": entry.get("result", ""),
        })
    return fixes


def extract_events(result: Any) -> List[Dict[str, Any]]:
    """Translate a Runner.run result's ``new_items`` into typed events.

    Tool calls become ``action`` events. Reasoning items become ``reasoning``
    events (trimmed). Tool outputs are not surfaced — the UI shows what the
    agent *did*, not the raw JSON it got back.
    """
    # Imported lazily so the module can be imported without the agents SDK
    # at unit-test time.
    from agents.items import ToolCallItem, ReasoningItem  # type: ignore

    events: List[Dict[str, Any]] = []
    for item in getattr(result, "new_items", []) or []:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = (
                getattr(raw, "name", None)
                or (raw.get("name") if isinstance(raw, dict) else None)
                or "tool"
            )
            args = (
                getattr(raw, "arguments", None)
                or (raw.get("arguments") if isinstance(raw, dict) else None)
            )
            events.append({
                "type": "action",
                "tool": name,
                "summary": summarize_tool_call(name, args),
            })
        elif isinstance(item, ReasoningItem):
            raw = item.raw_item
            parts: List[str] = []
            summary = getattr(raw, "summary", None)
            if summary:
                for s in summary:
                    text = getattr(s, "text", None)
                    if text:
                        parts.append(text)
            if parts:
                text = " ".join(parts).strip()
                if len(text) > 400:
                    text = text[:400].rsplit(" ", 1)[0] + "…"
                events.append({"type": "reasoning", "text": text})
    return events


def notebook_event(path: Optional[str], num_cells: Optional[int]) -> Optional[Dict[str, Any]]:
    """Build a ``notebook`` event, or ``None`` if there's nothing to report."""
    if not path:
        return None
    return {
        "type": "notebook",
        "path": str(path),
        "num_cells": int(num_cells) if num_cells is not None else 0,
    }
