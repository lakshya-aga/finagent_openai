"""End-of-day close routine + weighting math.

The two strategies share a single ``run_eod_close`` skeleton; only
the weight-computation step differs.

  equal_weight: w_i = direction_i / Σ |direction_j|
  market_cap:   w_i = direction_i × mcap_i / Σ mcap_j   (signed)

Both produce a {ticker → signed weight} dict where Σ |w_i| ≤ 1. The
strategy keeps gross exposure ≤ 100% always — leverage isn't
modeled in v1.

EOD pipeline per (date, strategy):

  1. read today's predictions (direction per ticker)
  2. compute target_weights via strategy formula
  3. fetch yesterday's open trades + portfolio snapshot
  4. diff:
       open new trades for fresh nonzero directions
       close trades whose direction flipped or went to 0
       ₹20 per opened-or-closed trade
  5. fetch today's close prices for every ticker holding a position
  6. mark-to-market each open trade → unrealized PnL
  7. daily PnL = Σ unrealized PnL changes since yesterday + realized
       PnL of trades closed today − transaction costs
  8. equity_today = equity_yesterday + daily_pnl
  9. write portfolio_snapshots + position_snapshots
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date as date_cls
from typing import Optional

from . import STARTING_CAPITAL, STRATEGIES, TRANSACTION_COST, store, universe

logger = logging.getLogger(__name__)


# ── Weight computation ─────────────────────────────────────────────


def compute_equal_weights(directions: dict[str, int]) -> dict[str, float]:
    """w_i = direction_i / Σ |direction_j|.

    With 30 longs and 20 shorts (all magnitude 1), each long gets
    +1/50 and each short -1/50 — gross exposure 100%, net +20%.
    """
    nonzero = {t: int(d) for t, d in directions.items() if d != 0}
    denom = sum(abs(d) for d in nonzero.values())
    if denom == 0:
        return {}
    return {t: d / denom for t, d in nonzero.items()}


def compute_market_cap_weights(
    directions: dict[str, int],
    market_caps: dict[str, float],
) -> dict[str, float]:
    """w_i = direction_i × mcap_i / Σ mcap_j (over nonzero directions).

    Tickers with missing mcap silently drop — the rest renormalise
    so Σ |w_i| ≤ 1. Logs a warning per drop so the operator can see
    coverage issues in the daily log.
    """
    nonzero = {t: int(d) for t, d in directions.items() if d != 0}
    contributing: list[tuple[str, int, float]] = []
    for t, d in nonzero.items():
        mc = market_caps.get(t)
        if mc is None or mc <= 0:
            logger.warning(
                "paper_trading: market-cap missing for %s — dropping from "
                "market_cap-weighted portfolio for this day",
                t,
            )
            continue
        contributing.append((t, d, mc))
    denom = sum(mc for _, _, mc in contributing)
    if denom == 0:
        return {}
    return {t: d * mc / denom for t, d, mc in contributing}


# ── Close-price fetch ──────────────────────────────────────────────


async def fetch_close_prices(tickers: list[str]) -> dict[str, float]:
    """Single batched yfinance call for the last close of every ticker.

    Returns ``{ticker: close_price}``. Missing tickers are omitted —
    the caller falls back to the previous snapshot's price for that
    ticker (i.e. zero return for the day).
    """
    if not tickers:
        return {}

    def _fetch() -> dict[str, float]:
        import yfinance as yf

        # period='5d' so we get a row even on Mondays / post-holiday.
        df = yf.download(
            tickers=tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        out: dict[str, float] = {}
        if df is None or df.empty:
            return out
        # yfinance returns a MultiIndex when len(tickers) > 1, flat
        # columns when len(tickers) == 1.
        if len(tickers) == 1:
            t = tickers[0]
            try:
                px = float(df["Close"].dropna().iloc[-1])
                out[t] = px
            except Exception:
                pass
            return out
        for t in tickers:
            try:
                px = float(df[t]["Close"].dropna().iloc[-1])
                out[t] = px
            except Exception:
                continue
        return out

    return await asyncio.to_thread(_fetch)


# ── EOD close ──────────────────────────────────────────────────────


@dataclass
class EodReport:
    date: str
    strategy: str
    equity_value: float
    daily_pnl: float
    transaction_costs: float
    trades_opened: int
    trades_closed: int
    n_open_positions: int


async def run_eod_close(
    date: str,
    strategy: str,
    *,
    close_prices: Optional[dict[str, float]] = None,
) -> EodReport:
    """End-of-day close for one (date, strategy). Idempotent — re-running
    overwrites the snapshot but doesn't re-charge transaction costs
    twice (the trades table records each open-or-close exactly once
    via diff against open_trades).

    Pass ``close_prices`` to skip the yfinance fetch (used by tests +
    bulk-replay). Otherwise we fetch lazily.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    # 1. read today's predictions
    preds = store.list_predictions(date=date)
    directions = {p["ticker"]: int(p["direction"]) for p in preds}

    # 2. compute target weights
    if strategy == "equal_weight":
        target = compute_equal_weights(directions)
    else:
        mcaps_table = store.get_market_caps()
        mcaps = {
            t: v["market_cap"] for t, v in mcaps_table.items() if v.get("market_cap")
        }
        if not mcaps:
            # First-run: refresh in-band so we don't return an empty
            # MCW portfolio on day one.
            mcaps = await universe.refresh_market_caps()
        target = compute_market_cap_weights(directions, mcaps)

    # 3. yesterday's open trades
    open_trades = store.list_open_trades(strategy)
    prev_snap = store.previous_snapshot(strategy, date)
    prev_equity = prev_snap["equity_value"] if prev_snap else STARTING_CAPITAL

    # 4. diff
    by_ticker = {t["ticker"]: t for t in open_trades}
    target_tickers = set(target.keys())
    open_tickers = set(by_ticker.keys())

    to_open = target_tickers - open_tickers
    to_keep = target_tickers & open_tickers
    to_close = open_tickers - target_tickers
    # Also close any open trade whose direction flipped sign.
    for t in list(to_keep):
        if (target[t] > 0) != (by_ticker[t]["direction"] > 0):
            to_close.add(t)
            to_open.add(t)
            to_keep.discard(t)

    # 5. close prices for everything we'll touch
    tickers_needing_px = list(to_open | to_keep | to_close)
    if close_prices is None:
        close_prices = await fetch_close_prices(tickers_needing_px)

    # 6. accounting
    transaction_costs = 0.0
    realized_pnl = 0.0
    trades_opened = 0
    trades_closed = 0

    for ticker in to_close:
        trade = by_ticker[ticker]
        px = close_prices.get(ticker, trade["open_price"])
        # PnL on notional from the ORIGINAL entry sizing. For paper
        # accounting we keep this simple: pnl = direction × (close - open) ×
        # equity_at_entry × open_weight ÷ open_price. Since open_weight
        # is already a fraction-of-equity, this is just:
        notional_at_entry = abs(trade["open_weight"]) * prev_equity
        pct_move = (px - trade["open_price"]) / trade["open_price"]
        pnl = trade["direction"] * pct_move * notional_at_entry
        realized_pnl += pnl
        transaction_costs += TRANSACTION_COST
        trades_closed += 1
        reason = "neutral" if ticker not in target else "direction_change"
        store.close_trade(
            trade["id"],
            closed_at=date,
            close_price=px,
            realized_pnl=pnl,
            close_reason=reason,
        )

    for ticker in to_open:
        if ticker not in close_prices:
            logger.warning(
                "paper_trading: no close price for %s — skipping open", ticker
            )
            continue
        direction = 1 if target[ticker] > 0 else -1
        store.open_trade(
            strategy=strategy,
            ticker=ticker,
            direction=direction,
            opened_at=date,
            open_price=close_prices[ticker],
            open_weight=target[ticker],
            transaction_cost=TRANSACTION_COST,
        )
        transaction_costs += TRANSACTION_COST
        trades_opened += 1

    # 7. MTM every position that's still open (= to_keep + freshly opened to_open).
    # Re-read open_trades so we capture the rows we just inserted.
    current_open = store.list_open_trades(strategy)

    unrealized_pnl_total = 0.0
    n_long = sum(1 for t in current_open if t["direction"] > 0)
    n_short = sum(1 for t in current_open if t["direction"] < 0)
    n_neutral = max(0, len(universe.NIFTY50_TICKERS) - n_long - n_short)

    # Clear stale position snapshots for this (date, strategy) before re-writing.
    store.clear_position_snapshots(date, strategy)

    gross_notional = 0.0
    net_notional = 0.0
    for trade in current_open:
        ticker = trade["ticker"]
        px = close_prices.get(ticker, trade["open_price"])
        notional_at_entry = abs(trade["open_weight"]) * prev_equity
        pct_move = (px - trade["open_price"]) / trade["open_price"]
        unrealized = trade["direction"] * pct_move * notional_at_entry
        unrealized_pnl_total += unrealized
        gross_notional += notional_at_entry
        net_notional += trade["direction"] * notional_at_entry
        days_held = _days_between(trade["opened_at"], date)
        store.upsert_position_snapshot(
            {
                "date": date,
                "strategy": strategy,
                "ticker": ticker,
                "direction": trade["direction"],
                "weight": trade["open_weight"],
                "notional": notional_at_entry,
                "entry_date": trade["opened_at"],
                "entry_price": trade["open_price"],
                "current_price": px,
                "days_held": days_held,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": pct_move * trade["direction"],
            }
        )

    # 8. daily PnL = realized (from closes) + Δunrealized − transaction costs.
    # Δunrealized over the day: today's unrealized − yesterday's unrealized
    # for positions that were open yesterday. For positions opened today
    # the contribution is 0 (open_price = today's close_price → no move yet).
    # For positions closed today the realized term already captures it.
    yesterday_unrealized_total = (
        sum(
            p["unrealized_pnl"]
            for p in store.list_positions(strategy, prev_snap["date"])
            if prev_snap
        )
        if prev_snap
        else 0.0
    )

    daily_pnl = (
        realized_pnl
        + (unrealized_pnl_total - yesterday_unrealized_total)
        - transaction_costs
    )
    equity_today = prev_equity + daily_pnl
    daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0.0

    # 9. write portfolio snapshot
    store.upsert_portfolio_snapshot(
        {
            "date": date,
            "strategy": strategy,
            "equity_value": equity_today,
            "cash": equity_today,  # see __init__ note: cash folded in
            "gross_exposure": gross_notional / equity_today if equity_today else 0.0,
            "net_exposure": net_notional / equity_today if equity_today else 0.0,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return,
            "transaction_costs": transaction_costs,
            "n_long": n_long,
            "n_short": n_short,
            "n_neutral": n_neutral,
        }
    )

    logger.info(
        "paper_trading: eod-close %s %s — equity=₹%.2f daily_pnl=₹%+.2f "
        "(%+.2f%%) opened=%d closed=%d open=%d costs=₹%.2f",
        date,
        strategy,
        equity_today,
        daily_pnl,
        daily_return * 100,
        trades_opened,
        trades_closed,
        len(current_open),
        transaction_costs,
    )

    return EodReport(
        date=date,
        strategy=strategy,
        equity_value=equity_today,
        daily_pnl=daily_pnl,
        transaction_costs=transaction_costs,
        trades_opened=trades_opened,
        trades_closed=trades_closed,
        n_open_positions=len(current_open),
    )


async def run_eod_close_all(date: str) -> list[EodReport]:
    """Convenience: run EOD close for every strategy. Used by the cron
    entrypoint + the admin "run now" endpoint."""
    return [await run_eod_close(date, s) for s in STRATEGIES]


# ── small helpers ──────────────────────────────────────────────────


def _days_between(start_iso: str, end_iso: str) -> int:
    a = date_cls.fromisoformat(start_iso)
    b = date_cls.fromisoformat(end_iso)
    return max(0, (b - a).days)
