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
    MARKET_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    RISK_DEBATOR_PROMPT,
    TRADER_PROMPT,
)
from .schemas import PortfolioDecision, ResearchPlan, TraderProposal
from .state import InvestDebateState, PanelState
from .tools import FUNDAMENTALS_TOOLS, MARKET_TOOLS, NEWS_TOOLS


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


async def _run_tool_loop(
    *,
    llm,
    tools: list,
    system_prompt: str,
    user_prompt: str,
    speaker: str,
    phase: str,
    panel_state: PanelState,
    max_iterations: int = 10,
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

    for iteration in range(max_iterations):
        ai_msg = await bound.ainvoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # Final answer — done.
            return (ai_msg.content or "", evidence)

        # Execute every tool call the LLM requested. Order doesn't matter
        # within a turn since each tool is read-only and side-effect-free.
        for call in tool_calls:
            tool_name = call.get("name") or call.get("tool")
            tool_args = call.get("args") or {}
            call_id = call.get("id") or str(uuid.uuid4())[:12]

            tool_fn = tool_index.get(tool_name)
            if tool_fn is None:
                output = json.dumps({"error": f"unknown tool: {tool_name}"})
            else:
                try:
                    output = await tool_fn.ainvoke(tool_args)
                except NotImplementedError:
                    # Tool isn't async-native — call sync.
                    output = tool_fn.invoke(tool_args)
                except Exception as exc:
                    logger.exception(
                        "trading_panel: tool %s raised", tool_name,
                    )
                    output = json.dumps({
                        "error": f"{type(exc).__name__}: {exc}",
                        "tool": tool_name,
                    })

            output_str = output if isinstance(output, str) else json.dumps(output, default=str)

            messages.append(ToolMessage(content=output_str, tool_call_id=call_id))

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
    await _emit({"type": "message", "phase": "market_analyst",
                 "speaker": "market_analyst", "text": text})
    await _emit({"type": "phase", "phase": "market_analyst", "state": "end"})
    return {"market_report": text, "evidence": state.get("evidence", []) + evidence}


async def news_analyst_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "news_analyst", "state": "start"})
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


# ── Stage 2: bull / bear ────────────────────────────────────────────


def _format_analyst_reports(state: PanelState) -> str:
    parts = []
    for label, key in (
        ("Market analyst", "market_report"),
        ("News analyst", "news_report"),
        ("Fundamentals analyst", "fundamentals_report"),
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
    llm = make_chat_for_role("panel_research_manager")
    structured = llm.with_structured_output(ResearchPlan)
    sys_prompt = RESEARCH_MANAGER_PROMPT.format(
        analyst_reports=_format_analyst_reports(state),
        debate_transcript=_format_debate_transcript(state.get("investment_debate")),
    )
    plan: ResearchPlan = await structured.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content="Emit the ResearchPlan now."),
    ])
    plan_dict = plan.model_dump()
    await _emit({"type": "research_plan", "phase": "research_manager",
                 "speaker": "research_manager", "data": plan_dict})
    await _emit({"type": "phase", "phase": "research_manager", "state": "end"})
    return {"research_plan": plan_dict}


# ── Stage 4: trader (structured output) ─────────────────────────────


async def trader_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "trader", "state": "start"})
    llm = make_chat_for_role("panel_trader")
    structured = llm.with_structured_output(TraderProposal)
    sys_prompt = TRADER_PROMPT.format(
        research_plan=json.dumps(state.get("research_plan") or {}, indent=2),
        analyst_reports=_format_analyst_reports(state),
    )
    proposal: TraderProposal = await structured.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content="Emit the TraderProposal now."),
    ])
    prop_dict = proposal.model_dump()
    await _emit({"type": "trader_proposal", "phase": "trader",
                 "speaker": "trader", "data": prop_dict})
    await _emit({"type": "phase", "phase": "trader", "state": "end"})
    return {"trader_proposal": prop_dict}


# ── Stage 5: risk debator (single LLM call, three perspectives) ─────


async def risk_debator_node(state: PanelState) -> dict[str, Any]:
    await _emit({"type": "phase", "phase": "risk_review", "state": "start"})
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
    llm = make_chat_for_role("panel_pm")
    structured = llm.with_structured_output(PortfolioDecision)
    sys_prompt = PORTFOLIO_MANAGER_PROMPT.format(
        analyst_reports=_format_analyst_reports(state),
        research_plan=json.dumps(state.get("research_plan") or {}, indent=2),
        trader_proposal=json.dumps(state.get("trader_proposal") or {}, indent=2),
        risk_review=state.get("risk_review") or "",
    )
    decision: PortfolioDecision = await structured.ainvoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content="Emit the PortfolioDecision now."),
    ])
    dec_dict = decision.model_dump()
    await _emit({"type": "portfolio_decision", "phase": "portfolio_manager",
                 "speaker": "portfolio_manager", "data": dec_dict})
    await _emit({"type": "phase", "phase": "portfolio_manager", "state": "end"})
    return {"portfolio_decision": dec_dict}
