"""Build a compact markdown summary of tool calls + reasoning from a RunResult."""

from __future__ import annotations

from agents.items import ReasoningItem, ToolCallItem, ToolCallOutputItem


def extract_trace_markdown(result) -> str:
    lines = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = (
                getattr(raw, "name", None)
                or (raw.get("name") if isinstance(raw, dict) else None)
                or "tool"
            )
            args = (
                getattr(raw, "arguments", None)
                or (raw.get("arguments", "") if isinstance(raw, dict) else "")
            )
            if isinstance(args, str) and len(args) > 120:
                args = args[:120] + "…"
            lines.append(f"> **{name}** `{args}`")
        elif isinstance(item, ToolCallOutputItem):
            raw = item.raw_item
            output = ""
            if isinstance(raw, dict):
                output = str(raw.get("output", ""))
            elif hasattr(raw, "output"):
                output = str(raw.output)
            if len(output) > 200:
                output = output[:200] + "…"
            if lines:
                lines[-1] += f" → {output}"
            else:
                lines.append(f"> → {output}")
        elif isinstance(item, ReasoningItem):
            raw = item.raw_item
            summaries = []
            if hasattr(raw, "summary") and raw.summary:
                for s in raw.summary:
                    if hasattr(s, "text") and s.text:
                        summaries.append(s.text)
            if summaries:
                text = " ".join(summaries)
                if len(text) > 300:
                    text = text[:300] + "…"
                lines.append(f"> 💭 _{text}_")
    return "\n".join(lines) if lines else ""
