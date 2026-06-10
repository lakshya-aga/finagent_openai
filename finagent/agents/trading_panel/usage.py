"""Token-usage instrumentation for the trading panel.

A process-global, thread-safe counter fed by a LangChain callback that
``llm_factory._make_openai`` attaches to every ChatOpenAI it builds.
Counts input / cached-input / output tokens per model name.

Why a global counter instead of per-run attribution: panel runs execute
concurrently (the daily cron runs 10+ at once), and LangChain callbacks
have no ambient run-id. Daily totals are what the cost question needs —
``snapshot()`` before/after a batch gives an exact delta for that batch,
which run_daily_all_50 logs and records into the cost ledger.

Cached-input tokens are tracked separately because OpenAI bills them at
~10% of the input rate — the cache-hit ratio is the main signal for
whether prompt-prefix structure is working.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# model_name -> {"input": int, "cached_input": int, "output": int, "calls": int}
_totals: dict[str, dict[str, int]] = {}


def _bump(model: str, input_t: int, cached_t: int, output_t: int) -> None:
    with _lock:
        bucket = _totals.setdefault(
            model, {"input": 0, "cached_input": 0, "output": 0, "calls": 0}
        )
        bucket["input"] += input_t
        bucket["cached_input"] += cached_t
        bucket["output"] += output_t
        bucket["calls"] += 1


def snapshot() -> dict[str, dict[str, int]]:
    """Deep copy of the per-model totals. Take one before and one after
    a batch; ``diff(before, after)`` is the batch's exact usage."""
    with _lock:
        return {m: dict(v) for m, v in _totals.items()}


def diff(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for model, vals in after.items():
        prev = before.get(model, {})
        delta = {k: vals.get(k, 0) - prev.get(k, 0) for k in vals}
        if any(delta.values()):
            out[model] = delta
    return out


class TokenUsageHandler(BaseCallbackHandler):
    """Reads token usage off every LLM response and feeds the counter.

    Handles both the modern ``usage_metadata`` shape (AIMessage) and the
    legacy ``llm_output["token_usage"]`` shape, because langchain-openai
    has moved between them across minor versions. Always best-effort:
    instrumentation must never break an actual panel call.
    """

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        try:
            for gens in getattr(response, "generations", []) or []:
                for gen in gens:
                    msg = getattr(gen, "message", None)
                    if msg is None:
                        continue
                    usage = getattr(msg, "usage_metadata", None) or {}
                    if not usage:
                        continue
                    input_t = int(usage.get("input_tokens", 0) or 0)
                    output_t = int(usage.get("output_tokens", 0) or 0)
                    details = usage.get("input_token_details", {}) or {}
                    cached_t = int(details.get("cache_read", 0) or 0)
                    model = (
                        getattr(msg, "response_metadata", {}) or {}
                    ).get("model_name") or "unknown"
                    _bump(model, input_t, cached_t, output_t)
                    return  # one usage record per response is enough

            # Legacy fallback shape.
            llm_output = getattr(response, "llm_output", None) or {}
            tu = llm_output.get("token_usage") or {}
            if tu:
                _bump(
                    llm_output.get("model_name", "unknown"),
                    int(tu.get("prompt_tokens", 0) or 0),
                    0,
                    int(tu.get("completion_tokens", 0) or 0),
                )
        except Exception:
            logger.debug("usage handler failed (non-fatal)", exc_info=True)


# Singleton — one handler shared by every ChatOpenAI the factory builds.
usage_handler = TokenUsageHandler()
