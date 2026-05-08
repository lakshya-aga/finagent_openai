"""Pydantic schemas for the trading panel's three structured outputs.

Mirrors TradingAgents' three-tier rating model so we stay compatible
with their evaluation methodology, but with the field descriptions
adapted to our context (.NS support, our metric vocabulary, our chart
fallback semantics).

Each manager emits one of these. The detail page renders all three
side-by-side, and the existing ``DebateVerdict`` is computed by
projection at the end (so the calendar / performance / verdict-card
UIs keep working unchanged).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Shared rating types ─────────────────────────────────────────────


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager.

    BUY / OVERWEIGHT — directionally long, sizing differs.
    HOLD             — genuinely balanced; no edge.
    UNDERWEIGHT/SELL — directionally short.
    """
    BUY = "buy"
    OVERWEIGHT = "overweight"
    HOLD = "hold"
    UNDERWEIGHT = "underweight"
    SELL = "sell"


class TraderAction(str, Enum):
    """3-tier transaction direction the Trader emits.

    The PM later refines BUY into BUY vs OVERWEIGHT (sizing) — the
    Trader only commits to the directional bet itself.
    """
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"


# ── Stage 3 output: ResearchPlan ────────────────────────────────────


class ResearchPlan(BaseModel):
    """The Research Manager's synthesis of the bull/bear debate.

    Hand-off to the Trader: pins the directional view, captures which
    side carried the argument, and translates that into concrete
    instructions the Trader can act on.
    """
    model_config = ConfigDict(use_enum_values=True)

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Reserve HOLD for genuinely "
            "balanced evidence; otherwise commit to the side with the "
            "stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "2-4 sentences. Summarise the strongest bull and bear points, "
            "then state which arguments led to the recommendation."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete next-step instructions for the Trader. Position "
            "sizing guidance consistent with the rating (BUY > OVERWEIGHT "
            "> HOLD > UNDERWEIGHT > SELL in conviction)."
        ),
    )


# ── Stage 4 output: TraderProposal ──────────────────────────────────


class TraderProposal(BaseModel):
    """The Trader's translation of the ResearchPlan into a trade.

    Includes concrete entry / stoploss / sizing if the action is not
    HOLD. The Risk Debator and Portfolio Manager downstream may revise
    this, but the Trader's job is to commit to a specific shape.
    """
    model_config = ConfigDict(use_enum_values=True)

    action: TraderAction
    reasoning: str = Field(
        description=(
            "2-4 sentences anchored in specific evidence from the analyst "
            "reports + the ResearchPlan. Cite numbers, not vibes."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Entry price target in the asset's quote currency. None for HOLD.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stoploss level in the asset's quote currency. None for HOLD.",
    )
    target_price: Optional[float] = Field(
        default=None,
        description="Profit target. None for HOLD.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Sizing guidance, e.g. '5% of portfolio' or '2% NAV at risk'.",
    )


# ── Stage 6 output: PortfolioDecision ───────────────────────────────


class PortfolioDecision(BaseModel):
    """The Portfolio Manager's final, post-risk-review verdict.

    This is the artifact the user reads. It supersedes the Trader's
    proposal — if the risk debate flagged something material, the PM
    will revise rating / sizing / target / horizon here.
    """
    model_config = ConfigDict(use_enum_values=True)

    rating: PortfolioRating
    executive_summary: str = Field(
        description=(
            "2-4 sentences covering entry strategy, sizing, key risk "
            "levels, and time horizon."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the "
            "analysts + debate + risk review. If the risk debate flagged "
            "a concern, address it here."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Target price in the asset's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stoploss level in the asset's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Recommended holding period, e.g. '3-6 months'.",
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description=(
            "3-5 bullet-style strings — the specific things that would "
            "invalidate this thesis. Surface them so the user can monitor."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Subjective confidence 0..1. 0.5 = genuinely uncertain. "
            "Use the analyst-debate-risk consensus to calibrate."
        ),
    )
