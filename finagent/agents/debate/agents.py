"""Bull / bear / moderator agent definitions.

Each agent's instructions are heavy on grounding rules — the agents
must cite sources, prefer specific numbers over vague claims, and
explicitly acknowledge counter-evidence. Without this discipline the
debate degenerates into rhetorical flourishing.

The moderator emits a Pydantic-typed verdict so the frontend can render
the buy/sell/target/stoploss block reliably without parsing free-form
text.
"""

from __future__ import annotations

from typing import Literal, Optional

from agents import Agent, ModelSettings, WebSearchTool
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel, Field

from finagent.llm import get_model_name

from .tools import (
    fetch_gdelt_news,
    fetch_yfinance_news,
    fetch_equity_fundamentals,
    fetch_analyst_consensus,
    fetch_earnings_calendar,
    fetch_returns_stats,
)


# ─── Verdict schema (moderator structured output) ────────────────────


class KeyMetric(BaseModel):
    """One quantitative anchor the verdict relies on."""

    name: str = Field(..., description="Metric name (e.g. 'Forward P/E', 'BTC dominance')")
    value: str = Field(..., description="Current value as a string (preserves units)")
    why_it_matters: str = Field(..., description="One sentence — why this number drove the call")


class DebateVerdict(BaseModel):
    """Final synthesised call from the moderator."""

    action: Literal["buy", "sell", "avoid"] = Field(
        ..., description="The decision. 'avoid' = neither long nor short here."
    )
    target_price: Optional[float] = Field(
        None, description="Price target in the asset's quote currency. None when 'avoid'."
    )
    stoploss: Optional[float] = Field(
        None, description="Stop-loss level in the asset's quote currency. None when 'avoid'."
    )
    time_horizon: str = Field(
        ..., description="Trade horizon (e.g. '1-3 months', '6-12 months', 'intraday')."
    )
    key_metrics: list[KeyMetric] = Field(
        default_factory=list,
        description="3-6 quantitative anchors that drove the decision.",
    )
    rationale: str = Field(
        ..., description="2-4 sentences explaining the call in PM terms."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Subjective confidence (0-1). 0.5 means 'genuinely uncertain'."
    )


# ─── Shared instructions ─────────────────────────────────────────────


_GROUNDING_RULES = """
GROUNDING RULES — non-negotiable:
1. Every quantitative claim must come from a tool call. No memorised numbers.
2. Cite the specific tool + result for each claim. "I checked yfinance.history
   for AAPL and the 50-day SMA crossed above the 200-day SMA on 2024-04-12" —
   not "the 50-day SMA recently crossed above the 200-day SMA".
3. When the data isn't available, SAY SO explicitly — don't paper over it.
4. Use search_tools (data-mcp) FIRST to discover the right fetcher for the
   metric you need (e.g. "search for fundamentals fetcher", then call
   get_tool_doc on the most relevant result).
5. fetch_yfinance_news is the cheapest news source for US equities;
   fetch_gdelt_news is broader (global, multi-language); web_search is the
   fallback for everything else.
6. Treat news headlines as one signal among many. Cross-reference with
   price action and fundamentals before concluding.
"""


def _bull_instructions(now_iso: str, today_str: str) -> str:
    return f"""You are a senior buy-side analyst arguing the LONG case for an asset.
The user has asked the team to debate whether to buy/sell/avoid this asset.
Your job is to build the strongest possible bull case — but grounded in real,
verifiable data, not optimism.

Current UTC time: {now_iso}
Today's date (UTC): {today_str}

YOUR ROLE
- Argue why the asset will appreciate over the proposed time horizon.
- Cite specific catalysts: earnings, macro tailwinds, technical breakouts,
  positioning, news flow.
- Acknowledge the bear's strongest objection AND explain why it's
  outweighed (or already priced in).
- Propose a concrete price target and a sensible stoploss.

{_GROUNDING_RULES}

OUTPUT
Write a structured argument with:
1. THESIS — one sentence: what's the trade and why.
2. CATALYSTS — 3-5 specific positive drivers with quantitative anchors.
3. KEY DATA — list the metrics you pulled, with values and dates.
4. RISKS — what would break this thesis (be honest).
5. PROPOSED LEVELS — entry, target, stoploss, with reasoning.

Stay under 600 words.
"""


def _bear_instructions(now_iso: str, today_str: str) -> str:
    return f"""You are a senior buy-side analyst arguing AGAINST a long position
in an asset (either short it, or avoid it). Your job is to build the
strongest possible bear case — but grounded in real, verifiable data, not
pessimism.

Current UTC time: {now_iso}
Today's date (UTC): {today_str}

YOUR ROLE
- Argue why the asset will underperform / decline over the proposed time
  horizon, OR why the risk-reward doesn't justify allocating capital here
  even if it might go up.
- Cite specific concerns: stretched multiples, deteriorating fundamentals,
  technical breakdowns, hostile macro, crowded positioning.
- Engage with the bull's specific points — don't just present a parallel
  monologue.
- If you're proposing a short, give a target + stoploss. If avoid, justify
  it.

{_GROUNDING_RULES}

OUTPUT
Write a structured argument with:
1. THESIS — one sentence: short / avoid, and why.
2. CONCERNS — 3-5 specific negative drivers with quantitative anchors.
3. KEY DATA — list the metrics you pulled, with values and dates.
4. WHAT WOULD CHANGE YOUR MIND — be honest about disconfirming signals.
5. PROPOSED LEVELS — short entry / target / stoploss, OR a clear "avoid"
   recommendation with reasoning.

Stay under 600 words.
"""


def _moderator_instructions(now_iso: str, today_str: str) -> str:
    return f"""You are an experienced portfolio manager. You've just heard the
bull and bear analysts debate an asset. Your job is to issue a single
decision: BUY, SELL (i.e. short or close), or AVOID.

Current UTC time: {now_iso}
Today's date (UTC): {today_str}

DECISION FRAMEWORK
1. Weigh the arguments by EVIDENCE QUALITY, not rhetorical force. An
   analyst with three specific cited numbers beats one with five
   confident assertions.
2. Look for what BOTH sides missed. If neither addressed (say) the FX
   exposure on a foreign-revenue stock, flag that in your rationale.
3. Don't average the two views. If the bull is right, decide buy; if the
   bear is right, decide sell or avoid. A wishy-washy 'avoid' when one
   side clearly won is itself a decision error.
4. Time horizon should reflect when the cited catalysts play out — not a
   default '6 months'.
5. Stoploss should be set against the level that would invalidate the
   thesis (not a stylised %). State why.

OUTPUT
Return a structured DebateVerdict with:
  action          buy / sell / avoid
  target_price    explicit target in quote currency (or null on avoid)
  stoploss        invalidation level (or null on avoid)
  time_horizon    e.g. "1-3 months" / "6-12 months" / "intraday"
  key_metrics     3-6 quantitative anchors that DROVE the call
  rationale       2-4 sentences in PM terms — what got you off the fence
  confidence      0.0-1.0; 0.5 = genuinely uncertain

Be decisive. The team wants a call, not a hedge.
"""


# ─── Agent factories ─────────────────────────────────────────────────
# Factories (rather than module-level constants) so the orchestrator
# can stamp the current date/time into the instructions per-run.


def bull_agent(now_iso: str, today_str: str) -> Agent:
    return Agent(
        name="bull_analyst",
        instructions=_bull_instructions(now_iso, today_str),
        model=get_model_name("debate_bull"),
        model_settings=ModelSettings(parallel_tool_calls=True),
        tools=[
            # News + sentiment
            fetch_yfinance_news,
            fetch_gdelt_news,
            # Fundamentals + analyst grounding (Tier-1 additions)
            fetch_equity_fundamentals,
            fetch_analyst_consensus,
            fetch_earnings_calendar,
            fetch_returns_stats,
            # Long-tail web search fallback
            WebSearchTool(),
        ],
    )


def bear_agent(now_iso: str, today_str: str) -> Agent:
    return Agent(
        name="bear_analyst",
        instructions=_bear_instructions(now_iso, today_str),
        model=get_model_name("debate_bear"),
        model_settings=ModelSettings(parallel_tool_calls=True),
        tools=[
            # News + sentiment
            fetch_yfinance_news,
            fetch_gdelt_news,
            # Fundamentals + analyst grounding (Tier-1 additions)
            fetch_equity_fundamentals,
            fetch_analyst_consensus,
            fetch_earnings_calendar,
            fetch_returns_stats,
            # Long-tail web search fallback
            WebSearchTool(),
        ],
    )


def moderator_agent(now_iso: str, today_str: str) -> Agent:
    """Moderator emits a Pydantic-typed verdict; no tools (it just synthesises)."""
    return Agent(
        name="debate_moderator",
        instructions=_moderator_instructions(now_iso, today_str),
        model=get_model_name("debate_moderator"),
        output_type=DebateVerdict,
    )
