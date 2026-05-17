"""Prediction sources.

v1 sources:
  - manual entry (POST /api/paper-trading/predictions)
  - seed_from_debates(date) — pull verdicts from the existing daily
    Nifty 50 debate scheduler and convert {buy → +1, sell → -1,
    avoid → 0}.

Later v2 sources (not yet implemented):
  - daily stock_analyst agent per ticker (cheap LLM, ~$0.05/day)
  - portfolio_manager agent that overlays its own filter on top
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from . import store, universe

logger = logging.getLogger(__name__)


# Map a debate verdict ``action`` string to a paper-trading direction.
# Anything outside these three falls through to 0 — better than
# guessing wrong on edge cases ("hold", "underweight", etc.).
_ACTION_TO_DIRECTION: dict[str, int] = {
    "buy":   +1,
    "long":  +1,
    "sell":  -1,
    "short": -1,
    "avoid":  0,
    "hold":   0,
}


def record_prediction(
    date: str,
    ticker: str,
    direction: int,
    *,
    confidence: float | None = None,
    reasoning: str | None = None,
    target_price: float | None = None,
    stop_loss_price: float | None = None,
    time_horizon: str | None = None,
    max_hold_days: int | None = None,
    source: str = "manual",
) -> int:
    """Public wrapper around store.upsert_prediction with universe check."""
    if ticker not in universe.NIFTY50_TICKERS:
        raise ValueError(
            f"{ticker!r} is not in NIFTY50_TICKERS — paper-trading book "
            "only covers Nifty 50 in v1"
        )
    return store.upsert_prediction(
        date=date, ticker=ticker, direction=direction,
        confidence=confidence, reasoning=reasoning,
        target_price=target_price, stop_loss_price=stop_loss_price,
        time_horizon=time_horizon, max_hold_days=max_hold_days,
        source=source,
    )


def commit_predictions(date: str, directions: dict[str, int], *, source: str = "manual") -> dict:
    """Batch insert. Used by the portfolio-agent's commit tool +
    the admin POST endpoint.

    Returns ``{accepted: int, rejected: [(ticker, reason), ...]}``.
    """
    accepted = 0
    rejected: list[tuple[str, str]] = []
    for ticker, direction in directions.items():
        try:
            record_prediction(date, ticker, int(direction), source=source)
            accepted += 1
        except Exception as e:
            rejected.append((ticker, str(e)))
    return {"accepted": accepted, "rejected": rejected}


# ── Seed from existing daily debates ───────────────────────────────


def seed_from_debates(
    date: str,
    *,
    tickers: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> dict:
    """Convert today's existing debate verdicts into predictions.

    Walks every Debate row where ``finished_at`` falls on the given
    UTC date AND ticker is in NIFTY50_TICKERS (or the optional filter
    list). The most-recent verdict per ticker wins on ties.

    Setting ``overwrite=False`` (default) skips tickers that already
    have a prediction for ``date`` from any source — lets a manual
    override survive a re-seed.
    """
    from finagent.experiments import get_store
    debate_store = get_store()

    target_tickers = set(tickers or universe.NIFTY50_TICKERS)
    existing = {p["ticker"] for p in store.list_predictions(date=date)} if not overwrite else set()

    # Pull a generous slab — we don't have a per-date debate query, so
    # we scan the last N and filter.
    debates = debate_store.list_debates(limit=500)
    chosen_by_ticker: dict[str, dict] = {}
    for d in debates:
        if d.ticker not in target_tickers:
            continue
        # Match by finished date (UTC). Skip in-flight + failed.
        if not d.finished_at:
            continue
        from datetime import datetime, timezone
        finished_dt = datetime.fromtimestamp(d.finished_at, tz=timezone.utc).date().isoformat()
        if finished_dt != date:
            continue
        verdict = d.verdict() or {}
        if not verdict.get("action"):
            continue
        # Most-recent-wins on ties.
        if d.ticker not in chosen_by_ticker or d.finished_at > chosen_by_ticker[d.ticker]["finished_at"]:
            chosen_by_ticker[d.ticker] = {"verdict": verdict, "finished_at": d.finished_at, "debate_id": d.id}

    accepted = 0
    skipped_existing = 0
    skipped_no_mapping = 0
    for ticker, info in chosen_by_ticker.items():
        if ticker in existing:
            skipped_existing += 1
            continue
        verdict = info["verdict"]
        action = (verdict.get("action") or "").lower().strip()
        direction = _ACTION_TO_DIRECTION.get(action)
        if direction is None:
            skipped_no_mapping += 1
            logger.warning("paper_trading.seed: unknown action %r for %s — skipping",
                           verdict.get("action"), ticker)
            continue
        record_prediction(
            date=date, ticker=ticker, direction=direction,
            confidence=verdict.get("confidence"),
            reasoning=verdict.get("rationale") or verdict.get("rationale_short"),
            target_price=verdict.get("target_price"),
            stop_loss_price=verdict.get("stoploss"),
            source=f"debate:{info['debate_id']}",
        )
        accepted += 1

    logger.info(
        "paper_trading.seed_from_debates(%s): accepted=%d skipped_existing=%d "
        "skipped_no_mapping=%d coverage=%d/%d tickers",
        date, accepted, skipped_existing, skipped_no_mapping,
        accepted + skipped_existing, len(target_tickers),
    )
    return {
        "date": date,
        "accepted": accepted,
        "skipped_existing": skipped_existing,
        "skipped_no_mapping": skipped_no_mapping,
        "coverage": accepted + skipped_existing,
        "universe_size": len(target_tickers),
    }
