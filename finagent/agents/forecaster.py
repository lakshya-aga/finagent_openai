"""Event-probability forecaster — the Torchcast-style surface.

Pipeline (cheap by construction):

  1. RESEARCH (one Agents-SDK agent, gpt-5-mini, Exa search tools,
     max ~4 searches): classify the question as forecastable,
     restate it as a resolvable claim with a horizon, gather an
     evidence digest with sources, and estimate a base rate.

  2. ENSEMBLE (three parallel NO-TOOL forecast passes on the same
     digest): each pass independently produces a probability +
     rationale. Independent passes over shared evidence is the
     classic cheap calibration trick — the median washes out
     single-pass anchoring, and the spread is an honest
     disagreement signal.

  3. AGGREGATE: median probability, [min, max] spread, drivers from
     the median pass. Spread > 0.25 flags low agreement.

Cost ≈ one research pass (~30-60k tokens mini) + three nano passes
(~5k each) + 3-4 Exa searches — comparable to a panel run, well
under a cent of Exa.

Forecasts persist with status='open' so a future calibration harness
can score Brier once outcomes resolve — the schema is the point;
resolution tooling comes later.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Schemas ────────────────────────────────────────────────────────


class EvidenceItem(BaseModel):
    fact: str = Field(description="One decision-relevant fact, stated plainly.")
    source_url: str = Field(default="", description="URL the fact came from.")
    date: str = Field(default="", description="Publication date if known (YYYY-MM-DD).")


class ResearchDigest(BaseModel):
    """Stage-1 output: is this forecastable, and what does the web say."""

    forecastable: bool = Field(
        description=(
            "True only if the question is about an uncertain FUTURE event "
            "that will resolve observably true/false. 'What is the capital "
            "of France' or 'should I buy NVDA' → False."
        ),
    )
    reframed_question: str = Field(
        description=(
            "The question restated as a precise, resolvable claim with an "
            "explicit horizon, e.g. 'The RBI announces a repo-rate cut at "
            "or before its October 2026 policy meeting.' Empty if not "
            "forecastable."
        ),
    )
    resolution_criteria: str = Field(
        default="",
        description="How an observer would decide true/false at the horizon.",
    )
    horizon: str = Field(
        default="", description="Resolution date/event, e.g. '2026-10-01'."
    )
    base_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Outside-view prior: how often does this KIND of event happen "
            "in comparable situations? Anchor from history, not vibes."
        ),
    )
    base_rate_reasoning: str = Field(
        default="", description="One sentence on where the base rate comes from."
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="5-10 decision-relevant facts from the searches, with sources.",
    )
    summary: str = Field(
        default="", description="<=600 chars: the evidence picture in plain English."
    )


class ForecastPass(BaseModel):
    """One independent ensemble member's read of the digest."""

    probability: float = Field(
        ge=0.01,
        le=0.99,
        description=(
            "P(claim resolves TRUE). Anchor on the base rate, then adjust "
            "for the evidence. Never 0 or 1 — reserve <0.05 and >0.95 for "
            "near-certainties with overwhelming evidence."
        ),
    )
    rationale: str = Field(description="<=400 chars: base rate → adjustments → number.")
    key_drivers: list[str] = Field(
        default_factory=list, description="3-5 factors that most move this probability."
    )
    what_would_change_mind: str = Field(
        default="", description="The single observation that would most shift this."
    )


# ── Stage 1: research agent (Exa tools) ────────────────────────────


_RESEARCH_INSTRUCTIONS = """You are a forecasting RESEARCHER. You receive one question
about a possible future event.

STEP 1 — Decide if it is FORECASTABLE: an uncertain FUTURE event that will
resolve observably true or false by some horizon. Questions of fact, opinion,
or advice are NOT forecastable — set forecastable=false and stop (no searches).

STEP 2 — If forecastable, restate it as a precise resolvable claim with an
explicit horizon and resolution criteria.

STEP 3 — Search the web with `exa_search`. Run 2-4 DIFFERENT searches:
  * base rates / historical frequency of this kind of event
  * the latest developments (use recent_only=true)
  * expert or market commentary on this specific event
Do NOT run more than 4 searches. Extract decision-relevant FACTS with their
source URLs and dates — not opinions about what will happen.

STEP 4 — Estimate the OUTSIDE-VIEW base rate from comparable historical
situations, citing your reasoning in one sentence.

Emit ONLY the structured ResearchDigest."""


def _make_exa_tool():
    """Build the exa_search function_tool lazily so this module imports
    cleanly when the Agents SDK or EXA_API_KEY are absent."""
    from agents import function_tool

    @function_tool
    def exa_search(query: str, recent_only: bool = False) -> dict:
        """Semantic web search. Returns {results: [{title, url,
        published_date, snippet}]}. Set recent_only=true to restrict to
        the last 30 days (use for 'latest developments' queries)."""
        from finagent import exa

        try:
            return {"results": exa.search(query, num_results=5, recent_only=recent_only)}
        except Exception as e:  # surface as data, never crash the loop
            return {"error": f"{type(e).__name__}: {e}", "results": []}

    return exa_search


async def _research(question: str) -> ResearchDigest:
    from agents import Agent, ModelSettings, Runner

    from finagent.llm import get_model_name

    agent = Agent(
        name="ForecastResearcher",
        instructions=_RESEARCH_INSTRUCTIONS,
        model=get_model_name("forecaster"),
        tools=[_make_exa_tool()],
        model_settings=ModelSettings(),
        output_type=ResearchDigest,
    )
    result = await Runner.run(
        agent,
        input=f"Question: {question}",
        # 4 searches + structured answer fits comfortably; the cap stops
        # a pathological search loop from burning tokens.
        max_turns=10,
    )
    return result.final_output_as(ResearchDigest)


# ── Stage 2: ensemble forecast passes (no tools) ───────────────────


_FORECAST_INSTRUCTIONS = """You are a calibrated FORECASTER. You receive a resolvable
claim, a horizon, an outside-view base rate, and an evidence digest.

Produce P(claim resolves TRUE):
  1. START from the base rate (outside view).
  2. ADJUST for each piece of evidence — small adjustments for weak evidence,
     large only for strong, recent, directly-relevant facts.
  3. CHECK against overconfidence: if the evidence is thin or conflicting,
     stay near the base rate. Never output 0 or 1.

Emit ONLY the structured ForecastPass."""


async def _forecast_pass(digest: ResearchDigest) -> Optional[ForecastPass]:
    from agents import Agent, ModelSettings, Runner

    from finagent.llm import get_model_name

    evidence_lines = "\n".join(
        f"- {e.fact} ({e.date or 'n.d.'}; {e.source_url or 'no source'})"
        for e in digest.evidence
    )
    prompt = (
        f"Claim: {digest.reframed_question}\n"
        f"Horizon: {digest.horizon}\n"
        f"Resolution criteria: {digest.resolution_criteria}\n"
        f"Outside-view base rate: {digest.base_rate:.2f} "
        f"({digest.base_rate_reasoning})\n\n"
        f"Evidence:\n{evidence_lines}\n\n"
        f"Summary: {digest.summary}\n\n"
        f"Produce the ForecastPass."
    )
    try:
        agent = Agent(
            name="ForecastPass",
            instructions=_FORECAST_INSTRUCTIONS,
            model=get_model_name("forecaster_pass"),
            model_settings=ModelSettings(),
            output_type=ForecastPass,
        )
        result = await Runner.run(agent, input=prompt, max_turns=1)
        return result.final_output_as(ForecastPass)
    except Exception:
        logger.exception("forecast pass failed")
        return None


# ── Public entrypoint ──────────────────────────────────────────────


async def run_forecast(question: str, *, n_ensemble: int = 3) -> dict:
    """Full pipeline. Returns a dict ready for persistence/SSE:

    {forecastable, question, reframed_question, probability, p_low,
     p_high, low_agreement, rationale, key_drivers,
     what_would_change_mind, resolution_criteria, horizon, base_rate,
     evidence: [...], summary, n_ensemble}

    Raises only on infrastructure failure (research stage crash); a
    non-forecastable question returns {forecastable: False, reason}.
    """
    digest = await _research(question)
    if not digest.forecastable:
        return {
            "forecastable": False,
            "question": question,
            "reason": (
                "Not a forecastable event — ask about an uncertain future "
                "event that will resolve observably true or false "
                "(e.g. 'Will the RBI cut rates before October?')."
            ),
        }

    passes = [
        p
        for p in await asyncio.gather(
            *(_forecast_pass(digest) for _ in range(max(1, n_ensemble)))
        )
        if p is not None
    ]
    if not passes:
        raise RuntimeError("all ensemble forecast passes failed")

    probs = sorted(p.probability for p in passes)
    median_p = statistics.median(probs)
    # The pass whose probability sits closest to the median narrates the
    # aggregate — its reasoning best represents the ensemble's centre.
    rep = min(passes, key=lambda p: abs(p.probability - median_p))
    spread = probs[-1] - probs[0]

    return {
        "forecastable": True,
        "question": question,
        "reframed_question": digest.reframed_question,
        "probability": round(median_p, 3),
        "p_low": round(probs[0], 3),
        "p_high": round(probs[-1], 3),
        "low_agreement": spread > 0.25,
        "rationale": rep.rationale,
        "key_drivers": rep.key_drivers,
        "what_would_change_mind": rep.what_would_change_mind,
        "resolution_criteria": digest.resolution_criteria,
        "horizon": digest.horizon,
        "base_rate": digest.base_rate,
        "evidence": [e.model_dump() for e in digest.evidence],
        "summary": digest.summary,
        "n_ensemble": len(passes),
    }


def aggregate_probabilities(probs: list[float]) -> dict:
    """Pure aggregation helper (unit-tested without LLM calls)."""
    if not probs:
        raise ValueError("no probabilities to aggregate")
    s = sorted(probs)
    return {
        "probability": statistics.median(s),
        "p_low": s[0],
        "p_high": s[-1],
        "low_agreement": (s[-1] - s[0]) > 0.25,
    }
