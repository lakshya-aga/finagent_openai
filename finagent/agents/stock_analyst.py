"""Stock analyst — per-ticker daily buy/sell/target/stop_loss/horizon.

TWO modes, picked by the ``STOCK_ANALYST_MODE`` env var:

  * ``panel`` (default) — runs the full Tauric / TradingAgents-style
    multi-agent panel from ``finagent.agents.trading_panel`` per
    ticker. Three analyst nodes (market / news / fundamentals) →
    bull/bear research debate → research_manager → trader →
    risk_debator → portfolio_manager synthesis. ~10 LLM calls per
    ticker (~30-90s wall-clock with panel-internal tool use). The
    panel's ``PortfolioDecision`` is converted into a
    ``StockRecommendation`` and written to the predictions table.

  * ``single`` (legacy) — one LLM call per ticker with a precomputed
    text-context blob (returns / ATR / range). Cheap (~$0.001 on
    gpt-4o-mini, ~1s wall-clock) but information-thin → tended to
    return ``avoid`` on every ticker because nothing in the context
    forced a directional commitment. Kept as the rescue fallback when
    the panel fails for any specific ticker.

Output (both modes) is a Pydantic ``StockRecommendation`` that lands
in the paper-trading ``predictions`` table via
``predictions.record_prediction(..., source='agent:stock_analyst')``.

Recommendation shape mirrors a triple-barrier method:
  - upper barrier   : target_price (take-profit)
  - lower barrier   : stop_loss_price (stop-out)
  - vertical barrier: max_hold_days (time exit)

For the on-demand /app/debate UI, ``trading_panel.run_panel`` is
called directly (with persist=True so it shows up in the debates
table). The stock_analyst path passes persist=False — these daily
50-ticker runs are ephemeral; only the resulting Recommendation
matters and that lands in the predictions table.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
        ge=0.0,
        le=1.0,
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
    """Run the analyst for one ticker — dispatches to the panel by
    default, falls back to the single-call path on any failure.

    Mode selection (``STOCK_ANALYST_MODE`` env var):
      * unset / ``"panel"`` — multi-agent Tauric panel (default)
      * ``"single"``         — legacy single-LLM-call path

    ``current_price`` + ``context`` only feed the single-call path;
    the panel fetches its own context via tool calls (yfinance,
    news, fundamentals) inside its analyst nodes.

    Never raises: on any failure returns
    ``StockRecommendation(action='avoid', confidence=0,
    reasoning='analyst unavailable: <err>')``.
    """
    mode = os.environ.get("STOCK_ANALYST_MODE", "panel").strip().lower()

    if mode == "panel":
        rec = await _analyse_via_panel(ticker)
        # Panel returned a usable recommendation OR an explicit avoid
        # with reasoning. Either way it's authoritative; don't fall
        # back unless the panel itself threw.
        if rec is not None:
            return rec
        # Panel raised — log and fall through to the single-call rescue.
        logger.warning(
            "stock_analyst: panel returned None for %s, falling back to single-call",
            ticker,
        )

    return await _analyse_via_single_call(
        ticker,
        current_price=current_price,
        context=context,
        model=model,
    )


async def _analyse_via_panel(ticker: str) -> Optional[StockRecommendation]:
    """Run the full trading panel for one ticker and project the
    PortfolioDecision into a StockRecommendation.

    Returns ``None`` on hard failure (panel raised); the caller then
    falls back to the cheap single-call path. Returns a
    ``StockRecommendation(action='avoid', ...)`` on soft failures
    (panel ran but produced no decision) so the row still persists
    with a reason rather than silently dropping.
    """
    try:
        from .trading_panel import run_panel
    except ImportError as e:
        logger.warning("stock_analyst: trading_panel unavailable (%s)", e)
        return None

    rounds = _panel_rounds()
    try:
        result = await run_panel(
            ticker=ticker,
            asset_class="indian_equity",
            rounds=rounds,
            emit=None,  # no SSE — server-side only
            persist=False,  # daily 50-ticker runs are ephemeral;
            # only the prediction row matters
            source="agent:stock_analyst",
        )
    except Exception as e:
        logger.warning("stock_analyst: panel raised for %s (%s)", ticker, e)
        return None

    verdict = result.get("verdict") or {}
    if not verdict:
        return StockRecommendation(
            action="avoid",
            confidence=0.0,
            reasoning="panel produced no verdict",
        )

    rec = StockRecommendation(
        action=verdict.get("action", "avoid"),
        confidence=float(verdict.get("confidence") or 0.0),
        target_price=verdict.get("target_price"),
        stop_loss_price=verdict.get("stoploss"),
        max_hold_days=_horizon_to_days(verdict.get("time_horizon")),
        reasoning=(verdict.get("rationale") or "")[:200]
        or "panel verdict (no rationale)",
    )
    return _normalise_recommendation(rec)


async def _analyse_via_single_call(
    ticker: str,
    *,
    current_price: float,
    context: str,
    model: Optional[str] = None,
) -> StockRecommendation:
    """Legacy single-LLM-call analyst. Cheap, information-thin —
    kept as the rescue path when the panel fails for a specific ticker."""
    try:
        from agents import Agent, ModelSettings, Runner
    except ImportError as e:
        logger.warning("stock_analyst: agents SDK unavailable (%s)", e)
        return StockRecommendation(
            action="avoid",
            confidence=0.0,
            reasoning=f"analyst unavailable: SDK missing ({e})",
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
            action="avoid",
            confidence=0.0,
            reasoning=f"analyst unavailable: {type(exc).__name__}",
        )

    return _normalise_recommendation(rec)


# ── Horizon translation ──────────────────────────────────────────────


# Map the panel's free-form time_horizon strings into trading-day
# integers for the triple-barrier max_hold_days field. Falls through
# to 10 days (a sensible swing-trade default) on unknown values
# rather than dropping to None — None would make the position
# carry indefinitely, which is worse than picking a wrong horizon.
_HORIZON_TO_DAYS: dict[str, int] = {
    "intraday": 1,
    "1d": 1,
    "short": 3,
    "short_term": 3,
    "short-term": 3,
    "medium": 10,
    "medium_term": 10,
    "medium-term": 10,
    "swing": 10,
    "long": 30,
    "long_term": 30,
    "long-term": 30,
    "unspecified": 10,
    "unknown": 10,
}


def _horizon_to_days(horizon: Optional[str]) -> int:
    """Translate the panel's ``time_horizon`` field to ``max_hold_days``."""
    if not horizon:
        return 10
    return _HORIZON_TO_DAYS.get(str(horizon).strip().lower(), 10)


def _panel_rounds() -> int:
    """Number of bull→bear debate rounds inside the panel.

    Default 1 (one round = two researcher turns) keeps the per-ticker
    LLM count at ~9 nodes and the wall-clock at ~30-60s. Bump to 2
    via ``STOCK_ANALYST_PANEL_ROUNDS=2`` for deeper debate at roughly
    double the cost."""
    try:
        return max(1, int(os.environ.get("STOCK_ANALYST_PANEL_ROUNDS", "1")))
    except (TypeError, ValueError):
        return 1


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
        return rec.model_copy(
            update={
                "action": action,
                "target_price": None,
                "stop_loss_price": None,
                "max_hold_days": None,
            }
        )
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
    rec: StockRecommendation,
    *,
    ticker: str,
    date: str,
    source: str,
) -> dict:
    """Translate a StockRecommendation into the kwargs the paper_trading
    ``record_prediction`` API accepts. Centralises the mapping so the
    daily-cron callsite stays a single line."""
    direction = _ACTION_TO_DIRECTION.get(rec.action, 0)
    return dict(
        date=date,
        ticker=ticker,
        direction=direction,
        confidence=rec.confidence,
        reasoning=rec.reasoning,
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
    concurrency: Optional[int] = None,
) -> dict:
    """Run the analyst for every Nifty 50 ticker on ``date``. Writes
    results directly into the predictions table via
    ``record_prediction(source='agent:stock_analyst')``.

    ``concurrency`` default depends on ``STOCK_ANALYST_MODE``:

      * panel mode (default): **10** simultaneous panels. Each panel
        fires ~10 LLM calls (3 analysts + bull/bear debate +
        research_manager + trader + risk_debator + portfolio_manager),
        so 10 panels = ~100 in-flight LLM calls — comfortably under
        OpenAI Tier-2 rate limits (5000 RPM, 450k TPM) and finishing
        in ~5-7 minutes wall-clock for all 50 tickers.

      * single mode: **50** simultaneous calls — one per ticker, one
        LLM call each, so the slowest round-trip gates total wall-clock.
        Bump back down only if a free-tier key is throttling.

      * ``STOCK_ANALYST_CONCURRENCY`` env var overrides both defaults.

    ``context_for(ticker)`` is only consumed by the single-mode
    fallback (the panel fetches its own context via tool calls inside
    its analyst nodes). Defaults to the lightweight built-in helper
    that pulls 5d OHLC + ATR from findata.

    Returns ``{date, n_analysed, n_buy, n_sell, n_avoid, n_failed, ...}``.
    Sub-task failures are captured into the per-ticker recommendation's
    ``reasoning`` field with status='avoid' — the batch never raises.
    """
    if concurrency is None:
        env_override = os.environ.get("STOCK_ANALYST_CONCURRENCY", "").strip()
        if env_override:
            try:
                concurrency = max(1, int(env_override))
            except ValueError:
                concurrency = None
        if concurrency is None:
            mode = os.environ.get("STOCK_ANALYST_MODE", "panel").strip().lower()
            concurrency = 10 if mode == "panel" else 50
    from finagent.paper_trading import predictions as ptp
    from finagent.paper_trading import universe

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
                    action="avoid",
                    confidence=0.0,
                    reasoning=f"context fetch failed: {type(e).__name__}",
                )
                return
            results[ticker] = await analyse_ticker(
                ticker,
                current_price=price,
                context=ctx,
            )

    await asyncio.gather(*(_one(t) for t in tickers))

    # Persist + collect per-ticker outcomes so the admin response can
    # surface exactly what happened without forcing the operator to
    # ssh in and tail logs.
    n_buy = n_sell = n_avoid = n_failed = n_persisted = 0
    persist_errors: list[dict] = []
    per_ticker: list[dict] = []
    for ticker, rec in results.items():
        is_failed = "unavailable" in (rec.reasoning or "") or "failed" in (
            rec.reasoning or ""
        )
        if is_failed:
            n_failed += 1
        if rec.action == "buy":
            n_buy += 1
        elif rec.action == "sell":
            n_sell += 1
        else:
            n_avoid += 1
        kwargs = recommendation_to_prediction_kwargs(
            rec,
            ticker=ticker,
            date=date,
            source="agent:stock_analyst",
        )
        try:
            ptp.record_prediction(**kwargs)
            n_persisted += 1
        except Exception as e:
            logger.exception("stock_analyst: failed to persist %s", ticker)
            persist_errors.append(
                {"ticker": ticker, "error": f"{type(e).__name__}: {e}"[:200]}
            )
        per_ticker.append(
            {
                "ticker": ticker,
                "action": rec.action,
                "confidence": rec.confidence,
                "target_price": rec.target_price,
                "stop_loss_price": rec.stop_loss_price,
                "reasoning": (rec.reasoning or "")[:200],
            }
        )

    logger.info(
        "stock_analyst: daily run for %s — buy=%d sell=%d avoid=%d failed=%d persisted=%d",
        date,
        n_buy,
        n_sell,
        n_avoid,
        n_failed,
        n_persisted,
    )
    return {
        "date": date,
        "n_analysed": len(results),
        "n_persisted": n_persisted,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_avoid": n_avoid,
        "n_failed": n_failed,
        # Truncated to top-5 + bottom-5 by reasoning length so the
        # admin endpoint response stays readable.
        "sample_per_ticker": (per_ticker[:10] if len(per_ticker) > 10 else per_ticker),
        "persist_errors": persist_errors[:10],
    }


async def _default_context_for(ticker: str) -> tuple[float, str]:
    """Lightweight context builder: pull 5d OHLC summary + recent
    news-count from findata. Designed to be FAST (<1s per ticker)
    so the daily batch stays under 2 minutes.

    Returns (current_price, context_str).
    """

    def _gather() -> tuple[float, str]:
        import pandas as pd
        from findata.equity_prices import get_equity_prices

        # Last 30 days for momentum + range context.
        try:
            df = get_equity_prices(
                tickers=[ticker],
                start_date=(pd.Timestamp.utcnow() - pd.Timedelta(days=45)).strftime(
                    "%Y-%m-%d"
                ),
                end_date=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            )
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return 0.0, "(no recent price data available)"

        # Flatten potential MultiIndex into a single-ticker frame.
        if isinstance(df.columns, pd.MultiIndex):
            close = (
                df["Close"][ticker]
                if ticker in df["Close"].columns
                else df["Close"].iloc[:, 0]
            )
            high = (
                df["High"][ticker]
                if ticker in df["High"].columns
                else df["High"].iloc[:, 0]
            )
            low = (
                df["Low"][ticker]
                if ticker in df["Low"].columns
                else df["Low"].iloc[:, 0]
            )
        else:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]

        current_price = float(close.iloc[-1])
        ret_5d = (
            (close.iloc[-1] / close.iloc[-5] - 1.0) * 100.0 if len(close) >= 5 else 0.0
        )
        ret_20d = (
            (close.iloc[-1] / close.iloc[-20] - 1.0) * 100.0
            if len(close) >= 20
            else 0.0
        )
        atr_pct = float(((high - low) / close).tail(14).mean()) * 100.0  # 14-day ATR%
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
