"""Stock analyst — per-ticker daily buy/sell/target/stop_loss/horizon.

Cheap, fast, single-LLM-call agent (~$0.001 on gpt-5-mini). Designed
to run for ALL 50 Nifty 50 tickers every morning before market open,
not the 5/day rotation the heavier multi-agent debate does.

Output is a Pydantic ``StockRecommendation`` that lands directly in
the paper-trading ``predictions`` table via
``predictions.record_prediction(..., source='agent:stock_analyst')``.

Recommendation shape mirrors a triple-barrier method:
  - upper barrier   : target_price (take-profit)
  - lower barrier   : stop_loss_price (stop-out)
  - vertical barrier: max_hold_days (time exit)

The agent is intentionally LIGHTWEIGHT — it receives precomputed
context (price history, recent news count, fundamentals summary) in
its prompt rather than reading via tools. One LLM call per ticker;
50 parallel batches finish in <2 minutes wall-clock.

For the deeper multi-agent debate (used by the /app/debate UI),
see finagent/agents/trading_panel — that's a different surface
that runs on demand for individual analysis, not a daily book.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ── Output schema ──────────────────────────────────────────────────


class StockRecommendation(BaseModel):
    """One daily recommendation for one ticker. Direction-encoded so
    paper_trading.predictions.record_prediction can ingest directly.
    """

    action: str = Field(
        description="One of: 'buy' (go long) | 'sell' (go short) | 'avoid' (stay neutral). Be conservative — when unsure, choose 'avoid'.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0..1.0 — your subjective certainty. 0.5 = coinflip; only emit 'buy' or 'sell' with confidence >= 0.55.",
    )
    target_price: Optional[float] = Field(
        default=None,
        description="Take-profit price in the asset's quote currency (INR for .NS tickers). Required when action != 'avoid'; null when 'avoid'.",
    )
    stop_loss_price: Optional[float] = Field(
        default=None,
        description="Stop-loss price in the asset's quote currency. Required when action != 'avoid'; null when 'avoid'.",
    )
    max_hold_days: Optional[int] = Field(
        default=None,
        description="Vertical barrier — close the position after this many TRADING days even if neither target nor stop is hit. Typical: 3-15. Null when 'avoid'.",
    )
    reasoning: str = Field(
        description="<=200 chars. One-sentence thesis in plain English. Cite the dominant signal (e.g. 'RSI oversold + earnings beat', 'sector rotation out of IT').",
    )


# ── Prompts ────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are STOCK_ANALYST for an Indian-equity intraday/swing book.

You receive precomputed context for ONE ticker and emit a single
recommendation as a structured StockRecommendation object.

Decision framework (triple-barrier-ish):
  - Direction (buy / sell / avoid): pick "buy" or "sell" only when
    your conviction is >= 0.55. Otherwise "avoid".
  - Target price (upper barrier): expected profit-take level. For
    "buy", target > current_price; for "sell", target < current_price.
    Set to a price the asset can plausibly reach within max_hold_days
    given its recent volatility (use ATR / recent range as a guide).
  - Stop loss (lower barrier): symmetric risk cap. Typical risk/reward
    1:1.5 to 1:3. For "buy", stop < current_price; for "sell", stop >
    current_price.
  - Max hold days (vertical barrier): 3-15 trading days. Use shorter
    horizons (3-5d) for momentum / breakout calls; longer (10-15d)
    for mean-reversion / fundamentals plays.

Be conservative: "avoid" is the safe default when signals conflict
or are weak. A neutral call costs nothing; a wrong direction call
loses real money.

Emit ONLY the StructuredRecommendation — no commentary."""


# ── Runner ─────────────────────────────────────────────────────────


async def analyse_ticker(
    ticker: str,
    *,
    current_price: float,
    context: str,
    model: Optional[str] = None,
) -> StockRecommendation:
    """Run the analyst for one ticker.

    ``context`` is a free-form text blob the caller assembles
    (recent OHLC summary, news headlines, fundamentals snapshot).
    We don't shape it from inside — the caller knows what's worth
    surfacing per ticker. Keeping the context-build outside this
    function means we can ship the agent without locking in any
    particular news/fundamentals source.

    Never raises: on any failure (SDK unavailable, LLM error, schema
    drift) returns ``StockRecommendation(action='avoid', confidence=0,
    reasoning='analyst unavailable: <err>')`` so the caller can fold
    it into the daily batch without try/except plumbing.
    """
    try:
        from agents import Agent, Runner, ModelSettings
        from finagent.llm import get_model_name
    except ImportError as e:
        logger.warning("stock_analyst: agents SDK unavailable (%s)", e)
        return StockRecommendation(
            action="avoid", confidence=0.0, reasoning=f"analyst unavailable: SDK missing ({e})",
        )

    model_name = model or _resolve_model()

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Current price: ₹{current_price:.2f}\n"
        f"\nContext:\n{context}\n"
        f"\nProduce the StockRecommendation."
    )

    try:
        agent = Agent(
            name="StockAnalyst",
            instructions=_SYSTEM_PROMPT,
            model=model_name,
            model_settings=ModelSettings(),
            output_type=StockRecommendation,
        )
        result = await Runner.run(agent, input=user_prompt, max_turns=1)
        rec = result.final_output_as(StockRecommendation)
    except Exception as exc:
        logger.warning("stock_analyst: LLM call failed for %s (%s)", ticker, exc)
        return StockRecommendation(
            action="avoid", confidence=0.0,
            reasoning=f"analyst unavailable: {type(exc).__name__}",
        )

    # Defence-in-depth: normalise the action verb in case the model
    # emits a near-miss ("hold", "long", etc.).
    rec = _normalise_recommendation(rec)
    return rec


def _normalise_recommendation(rec: StockRecommendation) -> StockRecommendation:
    """Coerce off-grid action verbs + clear price fields on 'avoid'."""
    action = (rec.action or "").strip().lower()
    if action in ("long", "buy"):
        action = "buy"
    elif action in ("short", "sell"):
        action = "sell"
    else:
        action = "avoid"

    # On 'avoid' the price fields are meaningless — drop them so
    # downstream consumers don't accidentally treat 'avoid' as having
    # an actionable SL/TP.
    if action == "avoid":
        return rec.model_copy(update={
            "action": action,
            "target_price": None,
            "stop_loss_price": None,
            "max_hold_days": None,
        })
    return rec.model_copy(update={"action": action})


def _resolve_model() -> str:
    try:
        from finagent.llm import get_model_name
        return get_model_name("stock_analyst")
    except Exception:
        try:
            from finagent.llm import get_model_name
            return get_model_name("intent_classifier")
        except Exception:
            return "gpt-4o-mini"


# ── Direction mapping for paper_trading.predictions ────────────────

_ACTION_TO_DIRECTION = {"buy": +1, "sell": -1, "avoid": 0}


def recommendation_to_prediction_kwargs(
    rec: StockRecommendation, *, ticker: str, date: str, source: str,
) -> dict:
    """Translate a StockRecommendation into the kwargs the paper_trading
    ``record_prediction`` API accepts. Centralises the mapping so the
    daily-cron callsite stays a single line."""
    direction = _ACTION_TO_DIRECTION.get(rec.action, 0)
    return dict(
        date=date, ticker=ticker, direction=direction,
        confidence=rec.confidence, reasoning=rec.reasoning,
        target_price=rec.target_price,
        stop_loss_price=rec.stop_loss_price,
        max_hold_days=rec.max_hold_days,
        source=source,
    )


# ── Daily batch runner ─────────────────────────────────────────────


async def run_daily_all_50(
    date: str,
    *,
    context_for: Optional["callable"] = None,
    concurrency: int = 5,
) -> dict:
    """Run the analyst for every Nifty 50 ticker on ``date``, with
    bounded parallelism. Writes results directly into the predictions
    table via ``record_prediction(source='agent:stock_analyst')``.

    ``context_for(ticker)`` is a caller-supplied coroutine that returns
    the (current_price, context_str) tuple for that ticker. Defaults to
    the lightweight built-in helper below that pulls 5d OHLC + recent
    news count from findata.

    Returns ``{date, n_analysed, n_buy, n_sell, n_avoid, n_failed}``.
    Sub-task failures are captured into the per-ticker recommendation's
    ``reasoning`` field with status='avoid' — the batch never raises.
    """
    from finagent.paper_trading import universe, predictions as ptp

    context_for = context_for or _default_context_for
    tickers = universe.NIFTY50_TICKERS

    sem = asyncio.Semaphore(concurrency)
    results: dict[str, StockRecommendation] = {}

    async def _one(ticker: str) -> None:
        async with sem:
            try:
                price, ctx = await context_for(ticker)
            except Exception as e:
                results[ticker] = StockRecommendation(
                    action="avoid", confidence=0.0,
                    reasoning=f"context fetch failed: {type(e).__name__}",
                )
                return
            results[ticker] = await analyse_ticker(
                ticker, current_price=price, context=ctx,
            )

    await asyncio.gather(*(_one(t) for t in tickers))

    # Persist
    n_buy = n_sell = n_avoid = n_failed = 0
    for ticker, rec in results.items():
        if "unavailable" in (rec.reasoning or "") or "failed" in (rec.reasoning or ""):
            n_failed += 1
        if rec.action == "buy":   n_buy += 1
        elif rec.action == "sell": n_sell += 1
        else:                      n_avoid += 1
        kwargs = recommendation_to_prediction_kwargs(
            rec, ticker=ticker, date=date, source="agent:stock_analyst",
        )
        try:
            ptp.record_prediction(**kwargs)
        except Exception:
            logger.exception("stock_analyst: failed to persist %s", ticker)

    logger.info(
        "stock_analyst: daily run for %s — buy=%d sell=%d avoid=%d failed=%d",
        date, n_buy, n_sell, n_avoid, n_failed,
    )
    return {
        "date": date, "n_analysed": len(results),
        "n_buy": n_buy, "n_sell": n_sell, "n_avoid": n_avoid,
        "n_failed": n_failed,
    }


async def _default_context_for(ticker: str) -> tuple[float, str]:
    """Lightweight context builder: pull 5d OHLC summary + recent
    news-count from findata. Designed to be FAST (<1s per ticker)
    so the daily batch stays under 2 minutes.

    Returns (current_price, context_str).
    """
    def _gather() -> tuple[float, str]:
        from findata.equity_prices import get_equity_prices
        import pandas as pd

        # Last 30 days for momentum + range context.
        try:
            df = get_equity_prices(
                tickers=[ticker],
                start_date=(pd.Timestamp.utcnow() - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            )
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return 0.0, "(no recent price data available)"

        # Flatten potential MultiIndex into a single-ticker frame.
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"][ticker] if ticker in df["Close"].columns else df["Close"].iloc[:, 0]
            high  = df["High"][ticker]  if ticker in df["High"].columns  else df["High"].iloc[:, 0]
            low   = df["Low"][ticker]   if ticker in df["Low"].columns   else df["Low"].iloc[:, 0]
        else:
            close = df["Close"]; high = df["High"]; low = df["Low"]

        current_price = float(close.iloc[-1])
        ret_5d   = (close.iloc[-1] / close.iloc[-5] - 1.0) * 100.0 if len(close) >= 5 else 0.0
        ret_20d  = (close.iloc[-1] / close.iloc[-20] - 1.0) * 100.0 if len(close) >= 20 else 0.0
        atr_pct  = float(((high - low) / close).tail(14).mean()) * 100.0  # 14-day ATR%
        range_lo = float(low.tail(20).min())
        range_hi = float(high.tail(20).max())

        ctx = (
            f"Price: ₹{current_price:.2f}\n"
            f"5-day return: {ret_5d:+.2f}%\n"
            f"20-day return: {ret_20d:+.2f}%\n"
            f"14-day ATR: {atr_pct:.2f}% of price\n"
            f"20-day range: ₹{range_lo:.2f} – ₹{range_hi:.2f}\n"
            f"(news + fundamentals not surfaced in v1 — keeping the per-ticker call\n"
            f"cheap; richer context is the trading-panel debate for /app/debate)"
        )
        return current_price, ctx

    return await asyncio.to_thread(_gather)
