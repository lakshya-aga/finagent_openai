"""LangGraph wiring for the trading panel.

  START
    ↓
  market_analyst        ← (one tool-loop per analyst)
    ↓
  news_analyst
    ↓
  fundamentals_analyst
    ↓
  bull_researcher  ⇆  bear_researcher       ← N rounds of back-and-forth
    ↓
  research_manager  → ResearchPlan
    ↓
  trader            → TraderProposal
    ↓
  risk_debator      → markdown risk review
    ↓
  portfolio_manager → PortfolioDecision
    ↓
  END

Analysts run sequentially (not concurrent) so the streaming UI surfaces
one full report before starting the next — better UX than 3 partial
reports interleaving. If wall-clock matters more than UX later, swap
to a parallel-fan-out pattern with a state reducer.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    bear_researcher_node,
    bull_researcher_node,
    fundamentals_analyst_node,
    macro_analyst_node,
    market_analyst_node,
    news_analyst_node,
    portfolio_manager_node,
    research_manager_node,
    risk_debator_node,
    trader_node,
)
from .state import PanelState


def _should_continue_debate(state: PanelState) -> str:
    """After each researcher turn, alternate or terminate.

    Each round = bull + bear (count += 2). When count >= 2 * rounds, the
    debate is done and we move on to the research manager.
    """
    deb = state.get("investment_debate") or {}
    count = deb.get("count", 0)
    rounds = state.get("rounds", 2)
    if count >= 2 * rounds:
        return "research_manager"
    last = deb.get("last_speaker", "")
    if last == "bull":
        return "bear_researcher"
    return "bull_researcher"


def build_panel_graph() -> "StateGraph":
    """Assemble + compile the LangGraph workflow."""
    g = StateGraph(PanelState)

    # Nodes
    g.add_node("market_analyst", market_analyst_node)
    g.add_node("news_analyst", news_analyst_node)
    g.add_node("fundamentals_analyst", fundamentals_analyst_node)
    g.add_node("macro_analyst", macro_analyst_node)
    g.add_node("bull_researcher", bull_researcher_node)
    g.add_node("bear_researcher", bear_researcher_node)
    g.add_node("research_manager", research_manager_node)
    g.add_node("trader", trader_node)
    g.add_node("risk_debator", risk_debator_node)
    g.add_node("portfolio_manager", portfolio_manager_node)

    # Edges. Macro lands AFTER fundamentals so its prompt can read the
    # company's debt + margin profile while reasoning about rate
    # sensitivity. Researchers see all four analyst reports before
    # arguing.
    g.add_edge(START, "market_analyst")
    g.add_edge("market_analyst", "news_analyst")
    g.add_edge("news_analyst", "fundamentals_analyst")
    g.add_edge("fundamentals_analyst", "macro_analyst")
    g.add_edge("macro_analyst", "bull_researcher")

    # Bull/bear back-and-forth, controlled by _should_continue_debate.
    g.add_conditional_edges(
        "bull_researcher",
        _should_continue_debate,
        {
            "bear_researcher": "bear_researcher",
            "research_manager": "research_manager",
        },
    )
    g.add_conditional_edges(
        "bear_researcher",
        _should_continue_debate,
        {
            "bull_researcher": "bull_researcher",
            "research_manager": "research_manager",
        },
    )

    g.add_edge("research_manager", "trader")
    g.add_edge("trader", "risk_debator")
    g.add_edge("risk_debator", "portfolio_manager")
    g.add_edge("portfolio_manager", END)

    return g.compile()
