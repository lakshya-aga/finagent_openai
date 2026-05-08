"""LangGraph state for the trading panel.

A flat TypedDict that every node reads + writes by key. Sub-states
for the bull/bear debate and (later) any multi-round risk debate are
nested dicts so we can keep round counters + per-speaker history
separate from the main message log.

We deliberately don't use ``MessagesState`` from langgraph.graph
because the panel doesn't follow a single conversation thread — each
analyst has its own tool-loop, and the researchers see structured
analyst-reports rather than raw chat history.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class InvestDebateState(TypedDict, total=False):
    """Bull/bear back-and-forth state."""
    bull_history: str          # full bullish argument so far
    bear_history: str          # full bearish argument so far
    current_response: str      # most recent response
    last_speaker: str          # "bull" | "bear"
    count: int                 # number of turns taken (2 = one full round)


class PanelState(TypedDict, total=False):
    """The full panel's blackboard.

    Every node reads what it needs from prior keys and writes its
    output into the keys it owns. LangGraph's reducer is the default
    "last-writer-wins" — each node writes a fresh copy of its own
    keys, which is fine since we don't have parallel writers to the
    same key.
    """
    # ── Inputs (set once at the start) ──
    ticker: str
    asset_class: str           # "us_equity" | "indian_equity" | "crypto" | ...
    today_iso: str             # "2026-05-08" — anchors date-relative reasoning
    panel_id: str              # uuid for tracing + persistence
    rounds: int                # debate rounds (default 2)

    # ── Stage 1: analyst reports (markdown strings) ──
    market_report: Optional[str]
    news_report: Optional[str]
    fundamentals_report: Optional[str]
    macro_report: Optional[str]

    # ── Stage 1b: tool-call evidence captured per analyst ──
    # Same shape as the existing finagent.debate evidence trail so the
    # frontend EvidenceList can render it without changes.
    evidence: list[dict[str, Any]]

    # ── Stage 2: bull/bear debate ──
    investment_debate: InvestDebateState

    # ── Stage 3: research manager output ──
    research_plan: Optional[dict[str, Any]]   # ResearchPlan.model_dump()

    # ── Stage 4: trader output ──
    trader_proposal: Optional[dict[str, Any]] # TraderProposal.model_dump()

    # ── Stage 5: risk debator output (single LLM call, 3 perspectives) ──
    risk_review: Optional[str]                # markdown summary

    # ── Stage 6: final portfolio decision ──
    portfolio_decision: Optional[dict[str, Any]]   # PortfolioDecision.model_dump()


def initial_state(
    *,
    ticker: str,
    asset_class: str,
    today_iso: str,
    panel_id: str,
    rounds: int = 2,
) -> PanelState:
    """Construct a fresh state with sensible defaults for every key.

    Pre-filling every key (even with None / empty) avoids LangGraph's
    'KeyError on first read' footgun in conditional edges.
    """
    return PanelState(
        ticker=ticker,
        asset_class=asset_class,
        today_iso=today_iso,
        panel_id=panel_id,
        rounds=rounds,
        market_report=None,
        news_report=None,
        fundamentals_report=None,
        macro_report=None,
        evidence=[],
        investment_debate=InvestDebateState(
            bull_history="",
            bear_history="",
            current_response="",
            last_speaker="",
            count=0,
        ),
        research_plan=None,
        trader_proposal=None,
        risk_review=None,
        portfolio_decision=None,
    )
