"""Intraday execution — open at market open, monitor SL/TP triggers,
finalize at market close.

Replaces the v1 "open everything at close, MTM at close" EOD batch
with a more realistic three-phase day:

    09:20 IST   capture_open_prices(today)
                 ↳ for each prediction (direction != 0) with no
                   open trade yet for this strategy, open the trade
                   at today's *opening* print. Snapshots target +
                   stop_loss onto the trade row so a later prediction
                   edit doesn't move the trigger.

    every 15m   monitor_triggers()
                 ↳ for each currently-open trade with target or
                   stop_loss set, fetch the latest LTP. If price
                   crossed the trigger, close the trade at the
                   trigger price (or LTP if a more favourable fill).

    15:35 IST   finalize_eod(today)
                 ↳ close any trades whose direction flipped or went
                   to zero in *today's* predictions (carry-forward
                   from yesterday's book). MTM remaining open trades
                   with today's close LTP. Write portfolio_snapshots
                   + position_snapshots — same shape v1 produced, so
                   the dashboard reads unchanged.

Idempotent at every phase:
  - capture_open_prices skips tickers that already have an open trade
  - monitor_triggers only acts on trades it can close (no-op when no
    triggers hit)
  - finalize_eod uses UNIQUE(date, strategy) on the snapshot tables
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import STARTING_CAPITAL, TRANSACTION_COST, STRATEGIES
from . import store, universe
from .engine import (
    compute_equal_weights, compute_market_cap_weights, _days_between,
)
from .quotes import get_quote_source

logger = logging.getLogger(__name__)


# ── Phase 1: capture opening prices ──────────────────────────────


@dataclass
class OpenCaptureReport:
    date: str
    strategy: str
    trades_opened: int
    skipped_already_open: int
    skipped_no_price: int
    transaction_costs: float


async def capture_open_prices(
    date: str,
    strategy: str,
    *,
    open_prices: Optional[dict[str, float]] = None,
) -> OpenCaptureReport:
    """Open fresh positions at the day's opening print for every
    prediction with direction != 0 that doesn't already have an open
    trade for this strategy.

    Pass ``open_prices`` to skip the quote-source fetch (tests +
    backfills). Otherwise we fetch lazily.

    The transaction cost is logged into the trade row but NOT yet
    debited from equity — that happens in ``finalize_eod`` when the
    daily snapshot rolls up costs. Keeps equity accounting in one
    place even though trades open intraday.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    preds = store.list_predictions(date=date)
    directions = {p["ticker"]: int(p["direction"]) for p in preds}
    # Snapshot the prediction's target + stoploss onto each trade we open.
    pred_meta = {p["ticker"]: p for p in preds}

    if strategy == "equal_weight":
        target = compute_equal_weights(directions)
    else:
        mcaps_table = store.get_market_caps()
        mcaps = {t: v["market_cap"] for t, v in mcaps_table.items() if v.get("market_cap")}
        if not mcaps:
            mcaps = await universe.refresh_market_caps()
        target = compute_market_cap_weights(directions, mcaps)

    open_trades = store.list_open_trades(strategy)
    already_open = {t["ticker"] for t in open_trades}
    fresh = [t for t in target if t not in already_open]

    if open_prices is None and fresh:
        open_prices = {}
        # One call per ticker; could batch by extending quote_source if
        # the cost matters. yfinance one-by-one for 50 tickers = ~10s.
        src = get_quote_source()
        # We use parallel coroutines so the whole batch finishes in
        # ~one round-trip's time rather than serially.
        results = await asyncio.gather(
            *(src.get_day_open(t, date) for t in fresh),
            return_exceptions=True,
        )
        for t, r in zip(fresh, results):
            if isinstance(r, Exception):
                logger.warning("paper_trading.intraday: open-price fetch failed for %s: %s", t, r)
                continue
            if r is not None:
                open_prices[t] = r

    opened = 0
    no_price = 0
    txn = 0.0
    for ticker in fresh:
        if not open_prices or ticker not in open_prices:
            no_price += 1
            continue
        direction = 1 if target[ticker] > 0 else -1
        meta = pred_meta.get(ticker, {})
        store.open_trade(
            strategy=strategy, ticker=ticker, direction=direction,
            opened_at=date, open_price=open_prices[ticker],
            open_weight=target[ticker],
            transaction_cost=TRANSACTION_COST,
            target_price=meta.get("target_price"),
            stop_loss_price=meta.get("stop_loss_price"),
            opened_via="market_open",
        )
        opened += 1
        txn += TRANSACTION_COST

    report = OpenCaptureReport(
        date=date, strategy=strategy,
        trades_opened=opened,
        skipped_already_open=len(already_open & set(target.keys())),
        skipped_no_price=no_price,
        transaction_costs=txn,
    )
    logger.info(
        "paper_trading.intraday: capture_open_prices(%s, %s) — opened=%d "
        "skipped_already_open=%d skipped_no_price=%d cost=₹%.2f",
        date, strategy, opened, report.skipped_already_open, no_price, txn,
    )
    return report


# ── Phase 2: monitor triggers ────────────────────────────────────


@dataclass
class TriggerReport:
    date: str
    strategy: str
    trades_checked: int
    triggered_target: int
    triggered_stop_loss: int


async def monitor_triggers(
    date: str,
    strategy: str,
    *,
    ltps: Optional[dict[str, float]] = None,
) -> TriggerReport:
    """Scan every open trade for SL/TP hits. Close the trade at the
    trigger price (NOT the actual LTP) on a hit — paper-realistic for
    a market order assumption with no slippage modelling.

    Trigger logic (for a LONG, direction=+1):
        price >= target_price  → close at target_price (TP)
        price <= stop_loss_price → close at stop_loss_price (SL)
    For a SHORT (direction=-1):
        price <= target_price  → close at target_price (TP)
        price >= stop_loss_price → close at stop_loss_price (SL)

    If both trigger in the same window (unlikely intraday, but
    possible on a gap-open), the STOP_LOSS wins — risk control
    takes precedence over profit-taking.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    open_trades = [
        t for t in store.list_open_trades(strategy)
        if (t.get("target_price") is not None) or (t.get("stop_loss_price") is not None)
    ]
    if not open_trades:
        return TriggerReport(date=date, strategy=strategy, trades_checked=0,
                             triggered_target=0, triggered_stop_loss=0)

    tickers = [t["ticker"] for t in open_trades]
    if ltps is None:
        ltps = await get_quote_source().get_ltps(tickers)

    triggered_tp = 0
    triggered_sl = 0
    now_ts = time.time()

    for trade in open_trades:
        px = ltps.get(trade["ticker"])
        if px is None:
            continue
        direction = int(trade["direction"])
        tp = trade.get("target_price")
        sl = trade.get("stop_loss_price")

        # SL takes precedence on a tie — risk over reward.
        sl_hit = False
        tp_hit = False
        if direction > 0:
            if sl is not None and px <= sl:
                sl_hit = True
            elif tp is not None and px >= tp:
                tp_hit = True
        else:  # direction < 0
            if sl is not None and px >= sl:
                sl_hit = True
            elif tp is not None and px <= tp:
                tp_hit = True

        if not (sl_hit or tp_hit):
            continue

        trigger_price = sl if sl_hit else tp
        close_reason = "stop_loss" if sl_hit else "target"
        # Realised PnL on this trade's notional sizing. The notional
        # is computed from yesterday's equity × |open_weight| — same
        # convention as engine.run_eod_close.
        prev_snap = store.previous_snapshot(strategy, date)
        prev_equity = prev_snap["equity_value"] if prev_snap else STARTING_CAPITAL
        notional = abs(trade["open_weight"]) * prev_equity
        pct_move = (trigger_price - trade["open_price"]) / trade["open_price"]
        pnl = direction * pct_move * notional

        store.close_trade(
            trade["id"], closed_at=date,
            close_price=trigger_price, realized_pnl=pnl,
            close_reason=close_reason, closed_ts=now_ts,
        )
        if sl_hit:
            triggered_sl += 1
        else:
            triggered_tp += 1
        logger.info(
            "paper_trading.intraday: %s triggered %s on %s at ₹%.2f "
            "(open ₹%.2f, %s%.2f%%) pnl=₹%+.2f",
            strategy, close_reason, trade["ticker"], trigger_price,
            trade["open_price"], "+" if pct_move >= 0 else "",
            pct_move * 100, pnl,
        )

    return TriggerReport(
        date=date, strategy=strategy,
        trades_checked=len(open_trades),
        triggered_target=triggered_tp,
        triggered_stop_loss=triggered_sl,
    )


# ── Phase 3: finalize EOD ────────────────────────────────────────


@dataclass
class FinalizeReport:
    date: str
    strategy: str
    equity_value: float
    daily_pnl: float
    transaction_costs: float
    closed_for_direction_change: int
    closed_for_market_close: int
    n_open_positions: int


async def finalize_eod(
    date: str,
    strategy: str,
    *,
    close_prices: Optional[dict[str, float]] = None,
) -> FinalizeReport:
    """End-of-day reconciliation:
      - close any trade whose ticker dropped out of today's nonzero
        directions (close_reason='direction_change')
      - close any trade whose direction flipped sign vs today's
        prediction (also direction_change)
      - mark every remaining open trade to market at today's close
      - write portfolio_snapshots + position_snapshots

    Idempotent on UNIQUE(date, strategy). Re-running re-MTMs but
    doesn't re-charge transaction costs because the close-side
    trades have already been written by the previous run.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    preds = store.list_predictions(date=date)
    directions = {p["ticker"]: int(p["direction"]) for p in preds}
    if strategy == "equal_weight":
        target = compute_equal_weights(directions)
    else:
        mcaps_table = store.get_market_caps()
        mcaps = {t: v["market_cap"] for t, v in mcaps_table.items() if v.get("market_cap")}
        if not mcaps:
            mcaps = await universe.refresh_market_caps()
        target = compute_market_cap_weights(directions, mcaps)

    open_trades = store.list_open_trades(strategy)
    open_by_ticker = {t["ticker"]: t for t in open_trades}

    # Trades to close = ticker dropped from today's target, OR direction
    # flipped sign. (Already-triggered SL/TP closes will not appear in
    # open_trades — they're closed_at != NULL.)
    to_close: dict[str, str] = {}  # ticker → reason
    for ticker, trade in open_by_ticker.items():
        if ticker not in target:
            to_close[ticker] = "direction_change"
        elif (target[ticker] > 0) != (trade["direction"] > 0):
            to_close[ticker] = "direction_change"

    # Close prices needed for: trades-to-close + remaining open MTM.
    tickers_needing_px = list(set(to_close.keys()) | set(open_by_ticker.keys()))
    if close_prices is None and tickers_needing_px:
        close_prices = await get_quote_source().get_ltps(tickers_needing_px)

    prev_snap = store.previous_snapshot(strategy, date)
    prev_equity = prev_snap["equity_value"] if prev_snap else STARTING_CAPITAL

    realized_pnl = 0.0
    transaction_costs = 0.0
    closed_dir = 0
    for ticker, reason in to_close.items():
        trade = open_by_ticker[ticker]
        px = (close_prices or {}).get(ticker, trade["open_price"])
        notional = abs(trade["open_weight"]) * prev_equity
        pct_move = (px - trade["open_price"]) / trade["open_price"]
        pnl = trade["direction"] * pct_move * notional
        realized_pnl += pnl
        transaction_costs += TRANSACTION_COST
        closed_dir += 1
        store.close_trade(
            trade["id"], closed_at=date, close_price=px,
            realized_pnl=pnl, close_reason=reason,
        )

    # Refresh open trades after the close pass.
    current_open = store.list_open_trades(strategy)

    # MTM each remaining trade with today's close.
    unrealized_total = 0.0
    gross_notional = 0.0
    net_notional = 0.0
    store.clear_position_snapshots(date, strategy)
    for trade in current_open:
        ticker = trade["ticker"]
        px = (close_prices or {}).get(ticker, trade["open_price"])
        notional = abs(trade["open_weight"]) * prev_equity
        pct_move = (px - trade["open_price"]) / trade["open_price"]
        unrealized = trade["direction"] * pct_move * notional
        unrealized_total += unrealized
        gross_notional += notional
        net_notional += trade["direction"] * notional
        store.upsert_position_snapshot({
            "date": date, "strategy": strategy, "ticker": ticker,
            "direction": trade["direction"], "weight": trade["open_weight"],
            "notional": notional,
            "entry_date": trade["opened_at"], "entry_price": trade["open_price"],
            "current_price": px,
            "days_held": _days_between(trade["opened_at"], date),
            "unrealized_pnl": unrealized,
            "unrealized_pnl_pct": pct_move * trade["direction"],
        })

    # Pick up any transaction costs from trades OPENED today (market-open
    # phase recorded them on the trade row; we charge equity here so the
    # snapshot reflects them).
    todays_opens = [
        t for t in store.list_trades(strategy, limit=200, only_closed=False)
        if t.get("opened_at") == date and t.get("opened_via") == "market_open"
    ]
    transaction_costs += TRANSACTION_COST * len(todays_opens)

    # Pick up triggered closes from today (SL/TP fires that happened
    # during monitor_triggers and so don't appear in to_close).
    triggered_today = [
        t for t in store.list_trades(strategy, limit=500, only_closed=True)
        if t.get("closed_at") == date and t.get("close_reason") in ("stop_loss", "target")
    ]
    closed_triggered = len(triggered_today)
    triggered_pnl = sum((t.get("realized_pnl") or 0.0) for t in triggered_today)
    triggered_costs = TRANSACTION_COST * closed_triggered
    realized_pnl += triggered_pnl
    transaction_costs += triggered_costs

    yesterday_unrealized = sum(
        p["unrealized_pnl"]
        for p in store.list_positions(strategy, prev_snap["date"])
    ) if prev_snap else 0.0

    daily_pnl = realized_pnl + (unrealized_total - yesterday_unrealized) - transaction_costs
    equity_today = prev_equity + daily_pnl
    daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0.0

    n_long = sum(1 for t in current_open if t["direction"] > 0)
    n_short = sum(1 for t in current_open if t["direction"] < 0)
    n_neutral = max(0, len(universe.NIFTY50_TICKERS) - n_long - n_short)

    store.upsert_portfolio_snapshot({
        "date": date, "strategy": strategy,
        "equity_value": equity_today, "cash": equity_today,
        "gross_exposure": gross_notional / equity_today if equity_today else 0.0,
        "net_exposure":   net_notional   / equity_today if equity_today else 0.0,
        "daily_pnl": daily_pnl, "daily_return_pct": daily_return,
        "transaction_costs": transaction_costs,
        "n_long": n_long, "n_short": n_short, "n_neutral": n_neutral,
    })

    logger.info(
        "paper_trading.intraday: finalize_eod(%s, %s) — equity=₹%.2f "
        "daily_pnl=₹%+.2f closed_dir=%d closed_triggered=%d open=%d costs=₹%.2f",
        date, strategy, equity_today, daily_pnl, closed_dir, closed_triggered,
        len(current_open), transaction_costs,
    )
    return FinalizeReport(
        date=date, strategy=strategy, equity_value=equity_today,
        daily_pnl=daily_pnl, transaction_costs=transaction_costs,
        closed_for_direction_change=closed_dir,
        closed_for_market_close=closed_triggered,
        n_open_positions=len(current_open),
    )


# ── Convenience: all-strategy wrappers ──────────────────────────


async def capture_open_prices_all(date: str) -> list[OpenCaptureReport]:
    return [await capture_open_prices(date, s) for s in STRATEGIES]


async def monitor_triggers_all(date: str) -> list[TriggerReport]:
    return [await monitor_triggers(date, s) for s in STRATEGIES]


async def finalize_eod_all(date: str) -> list[FinalizeReport]:
    return [await finalize_eod(date, s) for s in STRATEGIES]
