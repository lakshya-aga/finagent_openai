"""trading_panel — TradingAgents-style multi-stage debate.

A 4-stage hierarchy that replaces the linear bull→bear→moderator
flow with explicit specialist analysts, a researcher debate, a
trader, and a risk-debate-then-portfolio-manager pipeline. Modelled
after https://github.com/TauricResearch/TradingAgents but built on
LangChain + LangGraph (rather than tying us to a single SDK) so any
chat model — OpenAI, Anthropic, Gemini, Ollama (Qwen / Llama 3 /
DeepSeek), Azure — can fill any role.

Stages
------
  1. Analysts          (Market, News, Fundamentals)        — gather evidence
  2. Researcher debate (Bull ⇆ Bear, N rounds)             — argue
  3. Research Manager  → ResearchPlan                      — decide direction
  4. Trader            → TraderProposal                    — translate to trade
  5. Risk debator      (aggressive + neutral + conservative one-shot)
  6. Portfolio Manager → PortfolioDecision                 — final rating

Sits alongside the existing ``finagent.debate`` module — same
``debates`` table, same SSE streamer shape — until the panel proves
out and we can retire the simpler debate.
"""

from .runner import run_panel
from .schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)

__all__ = [
    "run_panel",
    "PortfolioDecision",
    "PortfolioRating",
    "ResearchPlan",
    "TraderAction",
    "TraderProposal",
]
