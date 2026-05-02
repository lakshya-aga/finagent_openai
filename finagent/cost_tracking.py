"""LLM cost tracking — pricing table + record helper.

UAT §5 #12: 'Cost / usage observability. Show me $ spent per run, broken
down by Anthropic/OpenAI/data calls, with a per-user budget. I cannot
deploy this firmwide without a meter.'

V1 ships:
  * Hardcoded pricing for the models we actually call (gpt-4o-mini for
    bias audit, gpt-4o + gpt-5 for orchestration when they're used).
    Models we don't recognise return cost = $0; the call is still
    recorded so we can spot the gap.
  * record_cost_event() — convenience wrapper that fishes
    response.usage out of the OpenAI SDK shape, computes USD, and
    inserts into the cost_events ledger.
  * Best-effort: every exception is caught so a misbehaving cost
    tracker never breaks the actual LLM call.

Phase 3:
  * Live pricing pulled from the provider invoice API (so we don't need
    to track price changes by hand).
  * Anthropic SDK support (their usage shape is similar but not
    identical).
  * Per-workspace budgets + soft limits.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


# Per-1M-token prices in USD. Update when the provider changes their
# rate card. Source: openai.com/api/pricing (snapshotted 2026-04).
_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    # model_name : (input, output)
    "gpt-4o-mini":    (0.150, 0.600),
    "gpt-4o":         (2.500, 10.000),
    "gpt-4":          (30.000, 60.000),
    "gpt-3.5-turbo":  (0.500, 1.500),
    # gpt-5 is in beta as of this snapshot; pricing is best-known estimate
    "gpt-5":          (5.000, 20.000),
    "gpt-5-mini":     (0.250, 1.000),
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Compute $ cost from a (model, prompt_tokens, completion_tokens) tuple.

    Returns 0 for unrecognised models — better to record the call with
    cost=0 than drop it entirely.
    """
    rates = _PRICING_USD_PER_1M.get(model)
    if rates is None:
        # Try a prefix match — handles dated suffixes like 'gpt-4o-mini-2024-07-18'.
        for known, r in _PRICING_USD_PER_1M.items():
            if model.startswith(known):
                rates = r
                break
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (
        (prompt_tokens / 1_000_000.0) * in_rate
        + (completion_tokens / 1_000_000.0) * out_rate
    )


def record_cost_event(
    *,
    response: Any,
    purpose: str,
    model: str,
    provider: str = "openai",
    run_id: Optional[str] = None,
    user: Optional[str] = None,
) -> None:
    """Persist a cost row from an OpenAI / Anthropic SDK response.

    Best-effort — every exception is caught and logged so the recording
    layer can never break the actual LLM call. ``response`` should be
    the SDK response object (has ``.usage.prompt_tokens`` /
    ``.usage.completion_tokens`` for OpenAI, or the equivalent dict).

    Caller passes the model name explicitly because the SDK responses
    sometimes omit it or report a normalised version (gpt-4o-mini-2024-
    07-18 vs gpt-4o-mini).
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        prompt_tokens = 0
        completion_tokens = 0
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or
                                (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0))
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or
                                    (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0))
        cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)

        # Defer the DB import: experiments.py imports finagent.recipes,
        # which can be heavy at module-load time. The lazy import keeps
        # the cost_tracking module cheap to import from agents that may
        # not always end up recording.
        from .experiments import get_store
        get_store().record_cost_event(
            purpose=purpose,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            run_id=run_id,
            user=user,
        )
    except Exception:
        # Never propagate. A misbehaving cost layer must not be allowed
        # to fail the underlying agent run.
        logging.exception("record_cost_event failed (purpose=%s model=%s)", purpose, model)
