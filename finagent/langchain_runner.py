"""Provider-neutral LangChain tool loop used by the LangGraph workflow."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .llm import _content_to_text, make_chat_for_role

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)


def _history_item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _history_to_messages(prior_history: Iterable[Any] | None) -> list[Any]:
    """Convert web/Agents-style history dictionaries into LangChain messages."""
    messages: list[Any] = []
    for item in prior_history or []:
        role = item.get("role") if isinstance(item, dict) else None
        text = _history_item_text(item).strip()
        if not text:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=text))
        elif role == "system":
            messages.append(SystemMessage(content=text))
        else:
            messages.append(HumanMessage(content=text))
    return messages


async def run_tool_loop(
    *,
    role: str,
    system: str,
    user: str,
    tools: Iterable[BaseTool],
    max_turns: int,
    progress_cb=None,
    phase: str = "",
    prior_history: Iterable[Any] | None = None,
) -> str:
    """Run a LangChain chat model with bound tools until it returns text.

    This intentionally mirrors the minimal semantics FinAgent needs from the
    old Agents SDK: model message, zero or more tool calls, tool outputs, repeat.
    The model itself is supplied by ``finagent.llm`` and can be OpenAI,
    Anthropic, Gemini, Ollama, etc.
    """
    tool_list = list(tools)
    by_name = {tool.name: tool for tool in tool_list}
    llm = make_chat_for_role(role).bind_tools(tool_list)
    messages = [
        SystemMessage(content=system),
        *_history_to_messages(prior_history),
        HumanMessage(content=user),
    ]
    last_text = ""

    for turn in range(max_turns):
        ai_msg = await llm.ainvoke(messages)
        messages.append(ai_msg)
        text = _content_to_text(getattr(ai_msg, "content", ""))
        if text.strip():
            last_text = text.strip()

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            return last_text

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tool_call_id = call.get("id")
            tool = by_name.get(name)
            if tool is None:
                output = f"Unknown tool {name!r}. Available tools: {sorted(by_name)}"
            else:
                if progress_cb:
                    await progress_cb(
                        {
                            "type": "event",
                            "data": {
                                "type": "tool_call",
                                "phase": phase,
                                "tool": name,
                                "turn": turn + 1,
                            },
                        }
                    )
                try:
                    output = await tool.ainvoke(args)
                except Exception as exc:
                    logger.exception("LangChain tool %s failed", name)
                    output = {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            messages.append(
                ToolMessage(
                    content=_stringify(output),
                    tool_call_id=tool_call_id or f"{name}_{turn}",
                    name=name,
                )
            )

    logger.warning("LangChain tool loop reached max_turns=%s phase=%s", max_turns, phase)
    if progress_cb:
        await progress_cb(
            {
                "type": "event",
                "data": {
                    "type": "turn_cap_exceeded",
                    "phase": phase,
                    "max_turns": max_turns,
                    "message": (
                        f"{phase or 'tool loop'} reached the {max_turns}-turn cap "
                        "before producing a final response."
                    ),
                },
            }
        )
    if last_text:
        return last_text
    return (
        f"Stopped after {max_turns} turns before the model produced a final "
        "answer. Check notebook/tool events for the last completed action."
    )
