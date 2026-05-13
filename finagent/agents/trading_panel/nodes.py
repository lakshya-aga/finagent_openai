"""LangGraph nodes — one async function per agent role.

Each node:
  1. Reads what it needs from PanelState
  2. Builds a prompt + (for analysts) binds tools to a ChatModel
  3. Runs a tool-call loop (analysts) or single-shot generation (managers)
  4. Writes its outputs to a fixed set of state keys

The keys are intentionally non-overlapping (no two nodes write to the
same key) so LangGraph's default "last-writer-wins" reducer is safe.

Streaming is wired through an opt-in ``emit`` callable on the state
(set by runner.py before .ainvoke). Each node calls emit() at phase
boundaries + after each tool call, mirroring the SSE event shape the
existing finagent.debate flow produces.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .llm_factory import make_chat_for_role
from .prompts import (
    BEAR_RESEARCHER_PROMPT,
    BULL_RESEARCHER_PROMPT,
    FUNDAMENTALS_ANALYST_PROMPT,
    MACRO_ANALYST_PROMPT,
    MARKET_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    RISK_DEBATOR_PROMPT,
    TRADER_PROMPT,
)
from .schemas import PortfolioDecision, ResearchPlan, TraderProposal
from .state import InvestDebateState, PanelState
from .tools import FUNDAMENTALS_TOOLS, MACRO_TOOLS, MARKET_TOOLS, NEWS_TOOLS


logger = logging.getLogger(__name__)


# Module-level emit hook. Set by runner.py before invoking the graph;
# nodes call _emit(...) and the runner forwards through the SSE channel.
# Falling back to a no-op keeps unit-testing trivial.
_emit_fn = None


def set_emit(fn) -> None:
    """Install the streaming callback. Pass None to disable."""
    global _emit_fn
    _emit_fn = fn


async def _emit(event: dict[str, Any]) -> None:
    if _emit_fn is None:
        return
    try:
        await _emit_fn(event)
    except Exception:
        logger.exception("trading_panel: emit failed (non-fatal)")


# ── Tool-call evidence capture ──────────────────────────────────────


def _shrink_for_llm_context(tool_name: str, output_str: str) -> str:
    """Strip heavyweight payloads (chart base64) before the tool output
    reaches the LLM's context window.

    The chart tool's full output is ~64KB — fine to capture once for
    the evidence trail, but catastrophic when it ends up embedded in
    every analyst report → every researcher / manager prompt. After
    one panel run with bull / bear / research_mgr / trader / risk /
    PM stages each carrying the chart through, the cumulative context
    exceeded gpt-5-mini's 272K-token limit and the request was
    rejected.

    The wrapped tool still returns its full output to the evidence
    layer (where the EvidenceList renders the actual chart for the
    user). For the LLM message context, we replace the heavy fields
    with terse placeholders so the model knows the chart was rendered
    + what its summary was, but doesn't carry the bytes around.
    """
    if tool_name != "plot_ohlc_chart" or not output_str:
        return output_str
    try:
        obj = json.loads(output_str)
    except Exception:
        return output_str
    if not isinstance(obj, dict):
        return output_str

    # Replace heavy fields with a status note. summary + chart_status
    # + ticker stay so the agent can talk about the chart in prose.
    md = obj.get("markdown_image", "") or ""
    if md and md.startswith("![") and "data:image" in md:
        obj["markdown_image"] = (
            "[chart rendered — full image is in the evidence panel; do NOT "
            "paste a markdown image link, the frontend handles inline display]"
        )
    obj["image_base64"] = "[stripped from LLM context to save tokens]"
    return json.dumps(obj, default=str)


def _summarise_tool_outcome(tool_name: str, output_str: str) -> str:
    """One-line summary of a tool's outcome for the user-visible heartbeat.

    Best-effort: parses output as JSON and picks fields that matter for
    each known tool family. Falls back to a generic 'ok' / 'error' read.
    Kept short (≤80 chars) so the streaming card stays compact.
    """
    if not output_str:
        return "ok"
    try:
        obj = json.loads(output_str)
    except Exception:
        # Non-JSON output (rare) — return a clipped raw preview.
        snippet = output_str.replace("\n", " ")[:60]
        return snippet or "ok"

    if not isinstance(obj, dict):
        return f"{type(obj).__name__} returned"

    # Error envelope (every wrapper produces this on failure).
    if "error" in obj and obj.get("error"):
        return f"error: {str(obj['error'])[:60]}"

    # Per-tool natural summaries.
    if "chart_status" in obj:
        return f"chart {obj['chart_status']}"
    if "articles" in obj:
        return f"{len(obj.get('articles') or [])} article(s)"
    if "company" in obj and "sector" in obj:  # GDELT shape
        c = len(obj.get("company") or [])
        s = len(obj.get("sector") or [])
        return f"GDELT: {c} company + {s} sector article(s)"
    if "best_order" in obj:
        sig = obj.get("signal", "?")
        ret = obj.get("forecast_return_pct")
        bits = [f"ARIMA, {sig}"]
        if isinstance(ret, (int, float)):
            bits.append(f"{ret:+.2f}%")
        return " · ".join(bits)
    if "indicators" in obj:
        return f"{len(obj.get('indicators') or {})} macro indicators"
    if "today" in obj and "one_year_ago" in obj:
        return f"yield curve: {obj.get('summary', 'ok')[:60]}"
    if "patterns" in obj:
        n = len(obj.get("patterns") or [])
        return f"{n} candlestick pattern(s)"
    if "levels" in obj:
        n = len(obj.get("levels") or [])
        return f"{n} S/R level(s)"
    if "hurst_exponent" in obj or "hurst" in obj:
        h = obj.get("hurst_exponent") or obj.get("hurst")
        regime = obj.get("regime", "")
        return f"Hurst {h:.2f}, {regime}" if isinstance(h, (int, float)) else (regime or "ok")
    if "rsi_14" in obj or "sma_50" in obj:
        rsi = obj.get("rsi_14") or obj.get("rsi")
        return f"RSI {rsi:.0f}, indicators ok" if isinstance(rsi, (int, float)) else "indicators ok"
    if "n_observations" in obj:
        return f"{obj.get('n_observations')} obs"

    # Generic dict — first 3 keys, value-clipped.
    keys = list(obj.keys())[:3]
    return ", ".join(f"{k}={str(obj[k])[:18]}" for k in keys) or "ok"


async def _invoke_structured_with_retry(
    *,
    llm,
    schema,
    system_prompt: str,
    user_prompt: str,
    speaker: str,
    max_retries: int = 2,
):
    """Invoke ``llm.with_structured_output(schema)`` with retry on
    validation failure.

    Smaller local models (Qwen 7b, Phi, etc.) emit malformed JSON often
    enough that one-shot structured output fails ~30-40% of the time.
    Retrying with the validation error fed back as feedback usually
    fixes it on the second attempt; total attempts capped at
    ``max_retries + 1`` to avoid runaway loops.

    On final failure, raises the last exception so the caller (and
    ultimately run_panel) can decide how to surface the error.
    """
    structured = llm.with_structured_output(schema)
    last_error: Optional[Exception] = None

    base_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries + 1):
        try:
            return await structured.ainvoke(base_messages)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "trading_panel: structured output failed for %s (attempt %d/%d): %s",
                speaker, attempt + 1, max_retries + 1, exc,
            )
            if attempt >= max_retries:
                break

            await _emit({
                "type": "structured_retry",
                "speaker": speaker,
                "attempt": attempt + 1,
                "error": str(exc)[:200],
            })

            # Re-prompt with the actual error so the model can correct.
            error_summary = str(exc)[:600]
            base_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
                HumanMessage(content=(
                    f"Your previous attempt failed schema validation:\n\n"
                    f"{error_summary}\n\n"
                    f"Re-emit the response. Match the schema EXACTLY: every "
                    f"required field present, enum values lowercase, numbers "
                    f"as numbers (not strings), no commentary outside the JSON."
                )),
            ]

    raise RuntimeError(
        f"structured output failed for {speaker} after {max_retries + 1} attempts; "
        f"last error: {last_error}"
    ) from last_error


async def _run_tool_loop(
    *,
    llm,
    tools: list,
    system_prompt: str,
    user_prompt: str,
    speaker: str,
    phase: str,
    panel_state: PanelState,
    max_iterations: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """Drive an analyst tool-call loop until the LLM produces a final
    text-only reply, or we hit max_iterations.

    Returns (final_text, evidence_records). Each evidence record is the
    same shape finagent.debate uses, so the existing UI's EvidenceList
    component renders them without changes.
    """
    bound = llm.bind_tools(tools)
    tool_index = {t.name: t for t in tools}

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    evidence: list[dict[str, Any]] = []

    # Per-phase tool-call cache. The model often re-calls the same tool
    # with identical args (4× chart calls observed in the AXISBANK
    # panel). Each redundant call: re-runs the tool (yfinance hit),
    # re-bloats the LLM context with the same output, and blows past
    # gpt-5-mini's 272K context cap. Cache by (tool, args-json) — if
    # we see a repeat, return the prior output INSTEAD of re-running.
    call_cache: dict[tuple[str, str], str] = {}
    # Hard cap on calls per tool name regardless of args. Belt-and-
    # braces in case the model nudges arg values to dodge the cache
    # (different lookback_days etc.).
    per_tool_limit = 1
    per_tool_count: dict[str, int] = {}

    # Log the initial prompt sizes — the user can grep
    # `docker logs synapse-finagent-1 | grep "panel/tool_loop"` to see
    # exactly what each analyst started with.
    logger.info(
        "panel/tool_loop START speaker=%s phase=%s sys_prompt=%d chars user_prompt=%d chars tools=%d (%s)",
        speaker, phase,
        len(system_prompt or ""), len(user_prompt or ""),
        len(tools), [t.name for t in tools],
    )

    for iteration in range(max_iterations):
        # Log the message-context size going INTO each LLM call.
        # If this grows unboundedly, that's the cause of slow inference.
        ctx_chars = sum(
            len(str(getattr(m, "content", ""))) for m in messages
        )
        logger.info(
            "panel/tool_loop INFER speaker=%s iter=%d/%d msgs=%d ctx_chars=%d",
            speaker, iteration + 1, max_iterations, len(messages), ctx_chars,
        )

        infer_start = time.time()
        ai_msg = await bound.ainvoke(messages)
        infer_dt = time.time() - infer_start

        n_tool_calls = len(getattr(ai_msg, "tool_calls", None) or [])
        ai_text_len = len(str(getattr(ai_msg, "content", "") or ""))
        logger.info(
            "panel/tool_loop AI_RESP speaker=%s iter=%d took=%.1fs tool_calls=%d text_len=%d",
            speaker, iteration + 1, infer_dt, n_tool_calls, ai_text_len,
        )

        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # Final answer — done.
            logger.info(
                "panel/tool_loop DONE speaker=%s phase=%s iters=%d total_evidence=%d final_text_len=%d",
                speaker, phase, iteration + 1,
                len(evidence), ai_text_len,
            )
            return (ai_msg.content or "", evidence)

        # Execute every tool call the LLM requested. Order doesn't matter
        # within a turn since each tool is read-only and side-effect-free.
        for call in tool_calls:
            tool_name = call.get("name") or call.get("tool")
            tool_args = call.get("args") or {}
            call_id = call.get("id") or str(uuid.uuid4())[:12]

            logger.info(
                "panel/tool_loop CALL speaker=%s tool=%s args=%s",
                speaker, tool_name,
                json.dumps(tool_args, default=str)[:300],
            )

            tool_fn = tool_index.get(tool_name)

            # ── Deduplication / call-count guards ─────────────────
            # Refuse repeated calls (same tool + same args) and any call
            # past the per-tool hard cap. Returns a synthesized error
            # envelope so the model sees "you already called this" and
            # writes the report instead of looping.
            args_key = json.dumps(tool_args, sort_keys=True, default=str)
            cache_key = (tool_name or "", args_key)
            n_prior = per_tool_count.get(tool_name or "", 0)

            if cache_key in call_cache:
                output = call_cache[cache_key]
                logger.warning(
                    "panel/tool_loop CACHE_HIT speaker=%s tool=%s args=%s — returning cached result",
                    speaker, tool_name, args_key[:100],
                )
            elif n_prior >= per_tool_limit:
                output = json.dumps({
                    "error": (
                        f"You have already called {tool_name} {n_prior} time(s) "
                        f"this turn. Each tool may be called at most "
                        f"{per_tool_limit} time(s) per analyst phase. Please "
                        f"refer to the prior result and write your final report."
                    ),
                    "tool": tool_name,
                })
                logger.warning(
                    "panel/tool_loop CALL_LIMIT speaker=%s tool=%s n_prior=%d — refusing",
                    speaker, tool_name, n_prior,
                )
            elif tool_fn is None:
                output = json.dumps({"error": f"unknown tool: {tool_name}"})
            else:
                # 60s per-tool hard timeout. yfinance has no built-in
                # timeout and can hang indefinitely on a network blip
                # or rate-limit; without this cap a single bad tool
                # call freezes the whole panel. 60s is generous for
                # any of our findata calls (typical: 1-5s).
                try:
                    async def _invoke():
                        try:
                            return await tool_fn.ainvoke(tool_args)
                        except NotImplementedError:
                            return await asyncio.to_thread(tool_fn.invoke, tool_args)
                    output = await asyncio.wait_for(_invoke(), timeout=60.0)
                    # Record successful invocation for dedup + cap.
                    per_tool_count[tool_name or ""] = n_prior + 1
                except asyncio.TimeoutError:
                    logger.warning(
                        "trading_panel: tool %s hung past 60s, killing call",
                        tool_name,
                    )
                    output = json.dumps({
                        "error": "tool call timed out after 60s (likely yfinance rate-limit or hung network)",
                        "tool": tool_name,
                    })
                except Exception as exc:
                    logger.exception(
                        "trading_panel: tool %s raised", tool_name,
                    )
                    output = json.dumps({
                        "error": f"{type(exc).__name__}: {exc}",
                        "tool": tool_name,
                    })

            output_str = output if isinstance(output, str) else json.dumps(output, default=str)

            # Shrink the LLM-visible payload (strip chart base64). The
            # full output is still captured in the evidence trail
            # below, so the user sees the actual chart in the UI.
            llm_visible = _shrink_for_llm_context(tool_name or "", output_str)

            # Populate the dedup cache with the SHRUNK output so a
            # repeated call returns the placeholder, not the 64KB blob.
            call_cache[cache_key] = llm_visible

            logger.info(
                "panel/tool_loop RESP speaker=%s tool=%s output_len=%d (shrunk=%d) preview=%r",
                speaker, tool_name, len(output_str), len(llm_visible),
                llm_visible[:200].replace("\n", " "),
            )

            messages.append(ToolMessage(content=llm_visible, tool_call_id=call_id))

            ev = {
                "phase": phase,
                "speaker": speaker,
                "tool": tool_name or "unknown",
                "args": json.dumps(tool_args, default=str)[:4_000],
                "output": output_str[:64_000],
                "call_id": call_id,
                "ts": time.time(),
            }
            evidence.append(ev)
            await _emit({"type": "tool_call", **ev})

            # ALSO emit a brief, user-visible status message so the UI
            # (which only renders type='message' / 'phase' / 'verdict' /
            # 'error') sees activity during long tool loops. Without
            # this, the analyst's 3-minute gathering phase looks blank
            # in the browser and users assume the panel is hung.
            outcome = _summarise_tool_outcome(tool_name or "unknown", output_str)
            await _emit({
                "type": "message",
                "phase": phase,
                "speaker": speaker,
                "text": f"`→ {tool_name}` — {outcome}",
                "ts": time.time(),
                "interim": True,
                # Tag the message with the tool + ticker so the frontend can
                # offer an "expand" chevron that fetches the OHLC chart for
                # that asset on demand. The ticker is the panel's subject
                # ticker (state["ticker"]); tool name lets the UI decide
                # whether to expose the expand affordance (chart-relevant
                # tools only) without re-parsing the text.
                "data": {
                    "tool": tool_name,
                    "ticker": panel_state.get("ticker"),
                    "asset_class": panel_state.get("asset_class"),
                },
            })

    # Hit iteration cap — return whatever the last AI message had.
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    text = (last_ai.content if last_ai else "") or ""
    logger.warning(
        "trading_panel: tool loop hit max_iterations=%d for %s",
        max_iterations, speaker,
    )
    return text, evidence


# ── Stage 1: analysts ───────────────────────────────────────────────


async def market_analyst_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "market_analyst", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "market_analyst",
        "speaker": "market_analyst",
        "text": "*Gathering technical signals — chart, indicators, S/R, regime, patterns.*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_analyst")
    sys_prompt = MARKET_ANALYST_PROMPT.format(
        ticker=state["ticker"], today_iso=state["today_iso"],
    )
    user_prompt = (
        f"Asset: {state['ticker']} ({state['asset_class']}). "
        f"Run your tool kit and produce the market analyst report."
    )
    text, evidence = await _run_tool_loop(
        llm=llm, tools=MARKET_TOOLS,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        speaker="market_analyst", phase="market_analyst",
        panel_state=state,
    )

    # Inject the chart from the evidence trail back into the analyst's
    # report. The LLM only saw a stripped placeholder for the chart
    # (to avoid blowing the context cap), so the model could not paste
    # the markdown_image itself. Pull the FULL chart out of the
    # evidence record and prepend it so the user sees the chart with
    # SMA + RSI overlays + S/R lines inline above the analyst's prose.
    chart_md = _extract_chart_from_evidence(evidence)
    if chart_md:
        text = chart_md + "\n\n" + text

    await _emit({"type": "message", "phase": "market_analyst",
                 "speaker": "market_analyst", "text": text})
    await _emit({"type": "phase", "phase": "market_analyst", "state": "end"})
    return {"market_report": text, "evidence": state.get("evidence", []) + evidence}


def _extract_chart_from_evidence(evidence: list[dict[str, Any]]) -> str:
    """Pull the first successful plot_ohlc_chart's markdown_image string
    out of the evidence list. Returns empty string if no chart succeeded.

    The evidence record has the FULL tool output (pre-shrink), so the
    base64-embedded image is intact here — only the LLM-facing
    ToolMessage was stripped down to placeholder."""
    for ev in evidence:
        if ev.get("tool") != "plot_ohlc_chart":
            continue
        raw = ev.get("output") or ""
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        md = obj.get("markdown_image") or ""
        # Only inject if it's a real chart, not the italic fallback.
        if md.startswith("![") and "data:image" in md:
            return md
    return ""


async def news_analyst_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "news_analyst", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "news_analyst",
        "speaker": "news_analyst",
        "text": "*Pulling recent headlines — yfinance + GDELT (with tone scoring).*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_analyst")
    sys_prompt = NEWS_ANALYST_PROMPT.format(
        ticker=state["ticker"], today_iso=state["today_iso"],
    )
    user_prompt = (
        f"Asset: {state['ticker']} ({state['asset_class']}). "
        f"Pick a sensible company name + sector keyword for the GDELT "
        f"queries (e.g. for RELIANCE.NS: 'Reliance Industries' / "
        f"'Indian conglomerates'), run the tools, write the news report."
    )
    text, evidence = await _run_tool_loop(
        llm=llm, tools=NEWS_TOOLS,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        speaker="news_analyst", phase="news_analyst",
        panel_state=state,
    )
    await _emit({"type": "message", "phase": "news_analyst",
                 "speaker": "news_analyst", "text": text})
    await _emit({"type": "phase", "phase": "news_analyst", "state": "end"})
    return {"news_report": text, "evidence": state.get("evidence", []) + evidence}


async def fundamentals_analyst_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "fundamentals_analyst", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "fundamentals_analyst",
        "speaker": "fundamentals_analyst",
        "text": "*Pulling 30 fundamentals fields, analyst consensus, earnings calendar, returns stats.*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_analyst")
    sys_prompt = FUNDAMENTALS_ANALYST_PROMPT.format(
        ticker=state["ticker"], today_iso=state["today_iso"],
    )
    user_prompt = (
        f"Asset: {state['ticker']} ({state['asset_class']}). "
        f"Run the fundamentals tool kit and write the report."
    )
    text, evidence = await _run_tool_loop(
        llm=llm, tools=FUNDAMENTALS_TOOLS,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        speaker="fundamentals_analyst", phase="fundamentals_analyst",
        panel_state=state,
    )
    await _emit({"type": "message", "phase": "fundamentals_analyst",
                 "speaker": "fundamentals_analyst", "text": text})
    await _emit({"type": "phase", "phase": "fundamentals_analyst", "state": "end"})
    return {"fundamentals_report": text, "evidence": state.get("evidence", []) + evidence}


async def macro_analyst_node(state: PanelState) -> dict[str, Any]:
    """Macro analyst — runs AFTER the fundamentals analyst so its prompt
    can reference the company's debt + margin profile when reasoning
    about rate sensitivity and inflation pass-through."""
    await _emit({"type": "phase", "phase": "macro_analyst", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "macro_analyst",
        "speaker": "macro_analyst",
        "text": "*Reading the macro tape — rates, inflation, credit spreads, FX, commodities, yield curve.*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_analyst")
    # Hint country from asset_class so the macro tool can do the right
    # thing for .NS (still pulls US-side macro + USD/INR + a flag note).
    country_hint = "IN" if state.get("asset_class", "").lower().startswith("indian") else "US"
    sys_prompt = MACRO_ANALYST_PROMPT.format(
        ticker=state["ticker"], today_iso=state["today_iso"],
    )
    # Pass the fundamentals report through as part of the user prompt so
    # the macro analyst can connect macro to the company's specific
    # balance sheet / margin profile rather than writing generic essays.
    fundamentals_excerpt = state.get("fundamentals_report") or "(not available)"
    user_prompt = (
        f"Asset: {state['ticker']} ({state['asset_class']}). "
        f"Country hint for fetch_macro_snapshot: '{country_hint}'.\n\n"
        f"For your reasoning about debt-financing cost + margin pressure, "
        f"the fundamentals analyst's report is:\n\n"
        f"---\n{fundamentals_excerpt}\n---\n\n"
        f"Run the macro tool kit and write the report — connect each "
        f"macro print to the specific company in the fundamentals excerpt."
    )
    text, evidence = await _run_tool_loop(
        llm=llm, tools=MACRO_TOOLS,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        speaker="macro_analyst", phase="macro_analyst",
        panel_state=state,
    )
    await _emit({"type": "message", "phase": "macro_analyst",
                 "speaker": "macro_analyst", "text": text})
    await _emit({"type": "phase", "phase": "macro_analyst", "state": "end"})
    return {"macro_report": text, "evidence": state.get("evidence", []) + evidence}


# ── Stage 2: bull / bear ────────────────────────────────────────────


def _format_analyst_reports(state: PanelState) -> str:
    parts = []
    for label, key in (
        ("Market analyst", "market_report"),
        ("News analyst", "news_report"),
        ("Fundamentals analyst", "fundamentals_report"),
        ("Macro analyst", "macro_report"),
    ):
        body = state.get(key) or ""
        if body:
            parts.append(f"### {label}\n\n{body}")
    return "\n\n---\n\n".join(parts) if parts else "(no analyst reports yet)"


def _format_prior_debate(deb: InvestDebateState) -> str:
    if not deb or deb.get("count", 0) == 0:
        return ""
    bits = []
    if deb.get("bull_history"):
        bits.append(f"PRIOR BULL ARGUMENTS:\n{deb['bull_history']}")
    if deb.get("bear_history"):
        bits.append(f"PRIOR BEAR ARGUMENTS:\n{deb['bear_history']}")
    return "\n\n".join(bits)


async def bull_researcher_node(state: PanelState) -> dict[str, Any]:
    deb = state.get("investment_debate") or InvestDebateState(
        bull_history="", bear_history="", current_response="",
        last_speaker="", count=0,
    )
    round_n = (deb.get("count", 0) // 2) + 1
    phase = f"bull_round_{round_n}"
    await _emit({"type": "phase", "phase": phase, "state": "start",
                 "speaker": "bull_researcher"})

    llm = make_chat_for_role("panel_researcher")
    sys_prompt = BULL_RESEARCHER_PROMPT.format(
        ticker=state["ticker"],
        analyst_reports=_format_analyst_reports(state),
        prior_debate=_format_prior_debate(deb),
    )
    msg = await llm.ainvoke([SystemMessage(content=sys_prompt),
                              HumanMessage(content="Write your bull turn now.")])
    text = msg.content or ""
    new_deb = InvestDebateState(
        bull_history=(deb.get("bull_history", "") + "\n\n" + text).strip(),
        bear_history=deb.get("bear_history", ""),
        current_response=text,
        last_speaker="bull",
        count=deb.get("count", 0) + 1,
    )
    await _emit({"type": "message", "phase": phase,
                 "speaker": "bull_researcher", "text": text})
    await _emit({"type": "phase", "phase": phase, "state": "end"})
    return {"investment_debate": new_deb}


async def bear_researcher_node(state: PanelState) -> dict[str, Any]:
    deb = state.get("investment_debate") or InvestDebateState(
        bull_history="", bear_history="", current_response="",
        last_speaker="", count=0,
    )
    round_n = (deb.get("count", 0) // 2) + 1
    phase = f"bear_round_{round_n}"
    await _emit({"type": "phase", "phase": phase, "state": "start",
                 "speaker": "bear_researcher"})

    llm = make_chat_for_role("panel_researcher")
    sys_prompt = BEAR_RESEARCHER_PROMPT.format(
        ticker=state["ticker"],
        analyst_reports=_format_analyst_reports(state),
        prior_debate=_format_prior_debate(deb),
    )
    msg = await llm.ainvoke([SystemMessage(content=sys_prompt),
                              HumanMessage(content="Write your bear turn now.")])
    text = msg.content or ""
    new_deb = InvestDebateState(
        bull_history=deb.get("bull_history", ""),
        bear_history=(deb.get("bear_history", "") + "\n\n" + text).strip(),
        current_response=text,
        last_speaker="bear",
        count=deb.get("count", 0) + 1,
    )
    await _emit({"type": "message", "phase": phase,
                 "speaker": "bear_researcher", "text": text})
    await _emit({"type": "phase", "phase": phase, "state": "end"})
    return {"investment_debate": new_deb}


# ── Stage 3: research manager (structured output) ───────────────────


def _format_debate_transcript(deb: Optional[InvestDebateState]) -> str:
    if not deb:
        return "(no debate)"
    bits = []
    if deb.get("bull_history"):
        bits.append(f"### Bull\n{deb['bull_history']}")
    if deb.get("bear_history"):
        bits.append(f"### Bear\n{deb['bear_history']}")
    return "\n\n".join(bits)


async def research_manager_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "research_manager", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "research_manager",
        "speaker": "research_manager",
        "text": "*Synthesising the bull/bear debate into a recommendation…*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_research_manager")
    sys_prompt = RESEARCH_MANAGER_PROMPT.format(
        analyst_reports=_format_analyst_reports(state),
        debate_transcript=_format_debate_transcript(state.get("investment_debate")),
    )
    plan: ResearchPlan = await _invoke_structured_with_retry(
        llm=llm, schema=ResearchPlan,
        system_prompt=sys_prompt,
        user_prompt="Emit the ResearchPlan now.",
        speaker="research_manager",
    )
    plan_dict = plan.model_dump()
    await _emit({"type": "research_plan", "phase": "research_manager",
                 "speaker": "research_manager", "data": plan_dict})
    await _emit({"type": "phase", "phase": "research_manager", "state": "end"})
    return {"research_plan": plan_dict}


# ── Stage 4: trader (structured output) ─────────────────────────────


async def trader_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "trader", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "trader",
        "speaker": "trader",
        "text": "*Translating the research plan into entry / target / stop levels…*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_trader")
    sys_prompt = TRADER_PROMPT.format(
        research_plan=json.dumps(state.get("research_plan") or {}, indent=2),
        analyst_reports=_format_analyst_reports(state),
    )
    proposal: TraderProposal = await _invoke_structured_with_retry(
        llm=llm, schema=TraderProposal,
        system_prompt=sys_prompt,
        user_prompt="Emit the TraderProposal now.",
        speaker="trader",
    )
    prop_dict = proposal.model_dump()
    await _emit({"type": "trader_proposal", "phase": "trader",
                 "speaker": "trader", "data": prop_dict})
    await _emit({"type": "phase", "phase": "trader", "state": "end"})
    return {"trader_proposal": prop_dict}


# ── Stage 5: risk debator (single LLM call, three perspectives) ─────


async def risk_debator_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "risk_review", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "risk_review",
        "speaker": "risk_debator",
        "text": "*Stress-testing the proposal — aggressive / conservative / neutral lenses…*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_risk")
    sys_prompt = RISK_DEBATOR_PROMPT.format(
        trader_proposal=json.dumps(state.get("trader_proposal") or {}, indent=2),
        analyst_reports=_format_analyst_reports(state),
    )
    msg = await llm.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content="Run the three-lens risk review now."),
    ])
    review = msg.content or ""
    await _emit({"type": "message", "phase": "risk_review",
                 "speaker": "risk_debator", "text": review})
    await _emit({"type": "phase", "phase": "risk_review", "state": "end"})
    return {"risk_review": review}


# ── Stage 6: portfolio manager (structured output) ──────────────────


async def portfolio_manager_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "portfolio_manager", "state": "start"})
    await _emit({
        "type": "message",
        "phase": "portfolio_manager",
        "speaker": "portfolio_manager",
        "text": "*Issuing the final verdict — rating, target, stop, time horizon, key risks…*",
        "ts": time.time(),
        "interim": True,
    })
    llm = make_chat_for_role("panel_pm")
    sys_prompt = PORTFOLIO_MANAGER_PROMPT.format(
        analyst_reports=_format_analyst_reports(state),
        research_plan=json.dumps(state.get("research_plan") or {}, indent=2),
        trader_proposal=json.dumps(state.get("trader_proposal") or {}, indent=2),
        risk_review=state.get("risk_review") or "",
    )
    decision: PortfolioDecision = await _invoke_structured_with_retry(
        llm=llm, schema=PortfolioDecision,
        system_prompt=sys_prompt,
        user_prompt="Emit the PortfolioDecision now.",
        speaker="portfolio_manager",
    )
    dec_dict = decision.model_dump()
    await _emit({"type": "portfolio_decision", "phase": "portfolio_manager",
                 "speaker": "portfolio_manager", "data": dec_dict})
    await _emit({"type": "phase", "phase": "portfolio_manager", "state": "end"})
    return {"portfolio_decision": dec_dict}
