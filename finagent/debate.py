"""Debate orchestrator — bull vs bear with moderator verdict.

Public entrypoint:
    run_debate(ticker, asset_class, *, rounds, emit) -> {transcript, verdict}

Streams events through the optional ``emit`` callable so the frontend
can render the debate live (turn-by-turn). Persists the final transcript
+ verdict on the ``debates`` table for the project-history view.

Event shape (each call to ``emit``):
    {
        "type": "phase" | "message" | "verdict" | "error",
        "phase": "bull_round_N" | "bear_round_N" | "verdict" | ...,
        "speaker": "bull_analyst" | "bear_analyst" | "moderator" | None,
        "text": "...",          # speaker's full message (final, not delta)
        "data": {...},          # only on type="verdict": parsed DebateVerdict
        "ts": <unix>
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from agents import Agent, Runner, RunConfig
from agents.mcp import MCPServerManager

from .agents.debate.agents import (
    DebateVerdict,
    bear_agent,
    bull_agent,
    moderator_agent,
)
from .mcp_connections import make_data_mcp


EmitFn = Callable[[dict], Awaitable[None]]


async def _noop_emit(_: dict) -> None:
    return None


# Per-tool output cap. We persist the full output so the UI can render
# real article titles + URLs + values; charts (~50KB base64 per call)
# are the only outputs that warrant truncation here. The truncated path
# strips the base64 so we keep title + summary but drop the bytes.
_OUTPUT_CAP_DEFAULT = 64_000   # ~64 KB per tool output, plenty for news/fundamentals
_OUTPUT_CAP_CHART = 1_000      # chart base64 stripped — bytes already shown inline


def _trim_chart_output(raw: str) -> str:
    """plot_ohlc_chart returns ~50KB base64 inside its JSON; the chart
    is already rendered inline in the message text via markdown_image,
    so for evidence persistence we strip image_base64 to keep payloads
    small. Keep title / summary / chart_status / params for context."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj.pop("image_base64", None)
            # Truncate the markdown_image field too — it embeds the same
            # base64. The agent's own message text contains the full one.
            mi = obj.get("markdown_image") or ""
            if len(mi) > 200:
                obj["markdown_image"] = mi[:160] + "…(truncated)"
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    return raw[:_OUTPUT_CAP_CHART]


def _extract_tool_evidence(
    result: Any, *, phase: str, speaker: str
) -> list[dict]:
    """Walk Runner.run() result.new_items and pair tool-call → tool-output.

    Each pair becomes one evidence record with the tool name, arguments,
    and the FULL output (subject to per-tool size caps). The UI groups
    by speaker so the user sees exactly which evidence each side cited.

    Returns a list of:
        {phase, speaker, tool, args (str), output (str), call_id, ts}
    """
    pending: dict[str, dict] = {}
    out: list[dict] = []
    now = time.time()

    for item in getattr(result, "new_items", []) or []:
        cls_name = type(item).__name__
        ri = getattr(item, "raw_item", None)
        if ri is None:
            continue
        ri_type = getattr(ri, "type", "") or ""

        # Tool call (the model decided to invoke a function tool).
        if cls_name == "ToolCallItem" or ri_type in ("function_call", "tool_call"):
            call_id = (
                getattr(ri, "call_id", None) or getattr(ri, "id", None) or ""
            )
            name = getattr(ri, "name", "") or "tool"
            args = getattr(ri, "arguments", "") or ""
            if isinstance(args, (dict, list)):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = str(args)
            pending[call_id] = {
                "phase": phase,
                "speaker": speaker,
                "tool": str(name),
                "args": str(args)[:4_000],   # arg blobs are small; 4K cap is generous
                "call_id": str(call_id),
                "ts": now,
            }

        # Tool output (the function returned).
        elif cls_name == "ToolCallOutputItem" or ri_type == "function_call_output":
            call_id = getattr(ri, "call_id", None) or ""
            base = pending.pop(
                call_id,
                {
                    "phase": phase, "speaker": speaker,
                    "tool": "unknown", "args": "",
                    "call_id": str(call_id), "ts": now,
                },
            )
            raw_output = str(getattr(ri, "output", "") or "")
            tool_name = base.get("tool", "")
            if tool_name == "plot_ohlc_chart":
                base["output"] = _trim_chart_output(raw_output)
            else:
                base["output"] = raw_output[:_OUTPUT_CAP_DEFAULT]
            out.append(base)

    # Surface any orphan calls (no matched output — rare, but possible
    # if the worker timed out mid-call). Keep them so the UI can show
    # "this tool was attempted" rather than silently swallow.
    for v in pending.values():
        v["output"] = ""
        v.setdefault("orphan", True)
        out.append(v)

    return out


async def run_debate(
    ticker: str,
    asset_class: str = "us_equity",
    *,
    rounds: int = 2,
    emit: Optional[EmitFn] = None,
    debate_id: Optional[str] = None,
    source: str = "user",
) -> dict[str, Any]:
    """Run a bull/bear/moderator debate.

    Args:
        ticker: e.g. "AAPL", "BTC-USD", "RELIANCE.NS".
        asset_class: free-form hint passed to the agents
            ("us_equity" / "crypto" / "indian_equity"). Used in the
            opening prompt so agents pick the right data fetchers.
        rounds: How many bull→bear exchange rounds before the moderator
            verdicts. Default 2 (bull opens, bear rebuts, bull counters,
            bear closes).
        emit: async callback for streaming events. Pass None for
            non-streaming use (tests, scheduled jobs).
        debate_id: pre-generated id to attach to events. If None, the
            orchestrator generates one — and we persist the row inline
            so the calendar / detail pages can pick it up after the
            stream ends.
        source: 'user' for manual submissions, 'scheduled' for the
            cron-driven Nifty 50 sweep. Persisted on the debate row;
            the calendar page filters on this.

    Returns:
        {debate_id, ticker, asset_class, rounds, transcript: [...],
         verdict: dict, started_at, finished_at}
    """
    emit = emit or _noop_emit
    # If the caller didn't pre-create a debate row (typical of the
    # scheduled / non-streaming path), create one now so the row exists
    # in the store from the moment the work starts.
    own_row = debate_id is None
    if own_row:
        from .experiments import get_store
        _store = get_store()
        _row = _store.create_debate(
            ticker=ticker,
            asset_class=asset_class,
            rounds=rounds,
            source=source,
        )
        debate_id = _row.id
        try:
            _store.update_debate(debate_id, status="running")
        except Exception:
            logging.exception("run_debate: could not flip status to running")
    started_at = time.time()

    # Stamp the agents with a fresh "now" — anchors any date-relative
    # reasoning ("the last earnings call was 3 weeks ago"). UTC keeps the
    # comparison portable across the deploy region and the user's tz.
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat(timespec="seconds")
    today_str = now_dt.strftime("%Y-%m-%d (%A)")

    transcript: list[dict] = []
    # Evidence trail: every tool call from every agent run, grouped by
    # speaker. Surfaced both via streaming (for live UIs) and persisted
    # on the debate row so the detail page can show "what the bull
    # actually read before claiming a +12% target".
    evidence: list[dict] = []

    async def _emit_evidence_batch(records: list[dict]) -> None:
        """Stream each tool call as its own event so live UIs can append
        them under the right speaker without waiting for the turn to end."""
        for rec in records:
            await emit({
                "type": "tool_call",
                **rec,
            })

    async def _emit_phase(phase: str, speaker: Optional[str] = None) -> None:
        await emit({
            "type": "phase",
            "phase": phase,
            "speaker": speaker,
            "ts": time.time(),
        })

    async def _emit_message(speaker: str, phase: str, text: str) -> None:
        msg = {
            "type": "message",
            "phase": phase,
            "speaker": speaker,
            "text": text,
            "ts": time.time(),
        }
        transcript.append(msg)
        await emit(msg)

    # Opening user message — what the debate is about.
    opening = (
        f"Asset under debate: {ticker} ({asset_class}).\n"
        f"As of {today_str}, build your case. Use tools to ground your claims."
    )

    # Bull and bear share the data-mcp connection so they can both
    # discover findata fetchers. Each agent gets its own MCPServerManager
    # because Agents SDK currently expects one server-set per Runner.run.
    # (We could share with care — but the SDK's lifecycle on streamable
    # HTTP is flaky enough that one-per-call is the safer default.)

    last_bull_text: Optional[str] = None
    last_bear_text: Optional[str] = None

    try:
        for round_i in range(1, rounds + 1):
            # ─── Bull turn ─────────────────────────────────────────
            phase = f"bull_round_{round_i}"
            await _emit_phase(phase, "bull_analyst")
            bull_input = opening if round_i == 1 else (
                f"{opening}\n\n"
                f"Round {round_i}. Your previous opening was:\n\n{last_bull_text}\n\n"
                f"The bear's reply was:\n\n{last_bear_text}\n\n"
                f"Now respond — engage with the bear's specific points, sharpen "
                f"your thesis, refine the levels."
            )
            async with MCPServerManager([make_data_mcp()]) as mgr:
                _bull = bull_agent(now_iso, today_str).clone(
                    mcp_servers=mgr.active_servers,
                )
                bull_result = await Runner.run(
                    _bull,
                    input=bull_input,
                    run_config=RunConfig(),
                    max_turns=20,
                )
            last_bull_text = bull_result.final_output_as(str)
            bull_ev = _extract_tool_evidence(
                bull_result, phase=phase, speaker="bull_analyst",
            )
            evidence.extend(bull_ev)
            await _emit_evidence_batch(bull_ev)
            await _emit_message("bull_analyst", phase, last_bull_text)

            # ─── Bear turn ─────────────────────────────────────────
            phase = f"bear_round_{round_i}"
            await _emit_phase(phase, "bear_analyst")
            bear_input = (
                f"{opening}\n\n"
                f"The bull's argument so far:\n\n{last_bull_text}\n\n"
                f"Now respond — engage with the bull's specific points, raise "
                f"the strongest counter-evidence, propose your own levels."
            )
            async with MCPServerManager([make_data_mcp()]) as mgr:
                _bear = bear_agent(now_iso, today_str).clone(
                    mcp_servers=mgr.active_servers,
                )
                bear_result = await Runner.run(
                    _bear,
                    input=bear_input,
                    run_config=RunConfig(),
                    max_turns=20,
                )
            last_bear_text = bear_result.final_output_as(str)
            bear_ev = _extract_tool_evidence(
                bear_result, phase=phase, speaker="bear_analyst",
            )
            evidence.extend(bear_ev)
            await _emit_evidence_batch(bear_ev)
            await _emit_message("bear_analyst", phase, last_bear_text)

        # ─── Moderator verdict ──────────────────────────────────────
        await _emit_phase("verdict", "moderator")
        # Reconstruct the full transcript as plain text for the moderator.
        debate_text = "\n\n".join(
            f"## {m['speaker']} ({m['phase']})\n{m['text']}" for m in transcript
        )
        mod_input = (
            f"Asset: {ticker} ({asset_class}). Today: {today_str}.\n\n"
            f"Below is the full bull/bear debate transcript. Read it carefully, "
            f"then issue your structured verdict.\n\n"
            f"{debate_text}"
        )
        _mod = moderator_agent(now_iso, today_str)
        mod_result = await Runner.run(
            _mod, input=mod_input, run_config=RunConfig(), max_turns=4,
        )
        verdict_obj: DebateVerdict = mod_result.final_output_as(DebateVerdict)
        verdict_dict = verdict_obj.model_dump()
        # Moderator typically has no tools (just synthesises), but capture
        # any anyway — keeps the contract consistent for future variants.
        mod_ev = _extract_tool_evidence(
            mod_result, phase="verdict", speaker="moderator",
        )
        if mod_ev:
            evidence.extend(mod_ev)
            await _emit_evidence_batch(mod_ev)
        await emit({
            "type": "verdict",
            "phase": "verdict",
            "speaker": "moderator",
            "data": verdict_dict,
            "ts": time.time(),
        })
        # Persist completion when we own the row (no streaming caller is
        # already doing so).
        if own_row:
            try:
                from .experiments import get_store
                get_store().update_debate(
                    debate_id,
                    status="completed",
                    transcript=transcript,
                    verdict=verdict_dict,
                    evidence=evidence,
                    finished=True,
                )
            except Exception:
                logging.exception("run_debate: failed to persist completion")

    except Exception as exc:
        logging.exception("debate %s failed (ticker=%s)", debate_id, ticker)
        await emit({
            "type": "error",
            "phase": "error",
            "text": f"{type(exc).__name__}: {exc}",
            "ts": time.time(),
        })
        if own_row:
            try:
                from .experiments import get_store
                get_store().update_debate(
                    debate_id,
                    status="failed",
                    transcript=transcript,
                    evidence=evidence,
                    error=str(exc),
                    finished=True,
                )
            except Exception:
                logging.exception("run_debate: failed to persist failure")
        raise

    finished_at = time.time()
    return {
        "debate_id": debate_id,
        "ticker": ticker,
        "asset_class": asset_class,
        "rounds": rounds,
        "transcript": transcript,
        "evidence": evidence,
        "verdict": verdict_dict,
        "started_at": started_at,
        "finished_at": finished_at,
    }
