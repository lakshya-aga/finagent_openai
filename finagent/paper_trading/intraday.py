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
from typing import Optional

from . import STARTING_CAPITAL, STRATEGIES, TRANSACTION_COST, store, universe
from .engine import (
    _days_between,
    compute_equal_weights,
    compute_market_cap_weights,
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
        mcaps = {
            t: v["market_cap"] for t, v in mcaps_table.items() if v.get("market_cap")
        }
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
                logger.warning(
                    "paper_trading.intraday: open-price fetch failed for %s: %s", t, r
                )
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
            strategy=strategy,
            ticker=ticker,
            direction=direction,
            opened_at=date,
            open_price=open_prices[ticker],
            open_weight=target[ticker],
            transaction_cost=TRANSACTION_COST,
            target_price=meta.get("target_price"),
            stop_loss_price=meta.get("stop_loss_price"),
            max_hold_days=meta.get("max_hold_days"),
            opened_via="market_open",
        )
        opened += 1
        txn += TRANSACTION_COST

    report = OpenCaptureReport(
        date=date,
        strategy=strategy,
        trades_opened=opened,
        skipped_already_open=len(already_open & set(target.keys())),
        skipped_no_price=no_price,
        transaction_costs=txn,
    )
    logger.info(
        "paper_trading.intraday: capture_open_prices(%s, %s) — opened=%d "
        "skipped_already_open=%d skipped_no_price=%d cost=₹%.2f",
        date,
        strategy,
        opened,
        report.skipped_already_open,
        no_price,
        txn,
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
        t
        for t in store.list_open_trades(strategy)
        if (t.get("target_price") is not None) or (t.get("stop_loss_price") is not None)
    ]
    if not open_trades:
        return TriggerReport(
            date=date,
            strategy=strategy,
            trades_checked=0,
            triggered_target=0,
            triggered_stop_loss=0,
        )

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
            trade["id"],
            closed_at=date,
            close_price=trigger_price,
            realized_pnl=pnl,
            close_reason=close_reason,
            closed_ts=now_ts,
        )
        if sl_hit:
            triggered_sl += 1
        else:
            triggered_tp += 1
        logger.info(
            "paper_trading.intraday: %s triggered %s on %s at ₹%.2f "
            "(open ₹%.2f, %s%.2f%%) pnl=₹%+.2f",
            strategy,
            close_reason,
            trade["ticker"],
            trigger_price,
            trade["open_price"],
            "+" if pct_move >= 0 else "",
            pct_move * 100,
            pnl,
        )

    return TriggerReport(
        date=date,
        strategy=strategy,
        trades_checked=len(open_trades),
        triggered_target=triggered_tp,
        triggered_stop_loss=triggered_sl,
    )


# ── EOD intraday replay (catches triggers between 15-min snapshots) ─


@dataclass
class ReplayReport:
    date: str
    strategy: str
    trades_scanned: int
    triggered_target: int
    triggered_stop_loss: int


async def replay_intraday_triggers(
    date: str,
    strategy: str,
    *,
    bars: Optional[dict] = None,
) -> ReplayReport:
    """Walk today's 1-minute OHLC bars and close any open trade whose
    target or stop_loss was breached during the day — even if the
    15-min ``monitor_triggers`` cron missed the touch.

    Runs at EOD just before ``finalize_eod`` so the daily snapshot
    reflects every trigger that ACTUALLY happened intraday, not
    just the ones the periodic monitor sampled.

    Trigger price = the SL/TP level itself (paper-realistic market
    order with no slippage). closed_ts = the timestamp of the 1m
    bar that breached, so the trade-history UI shows the real
    intra-day fill clock.

    Trade ordering note: ``monitor_triggers`` may have already
    closed some of today's trades. ``list_open_trades`` only
    returns currently-open ones, so we don't double-close.

    ``bars`` lets tests inject pre-built OHLC frames without
    touching yfinance. Shape: ``{ticker: DataFrame with
    'High'/'Low' columns indexed by tz-naive Timestamp}``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    open_trades = [
        t
        for t in store.list_open_trades(strategy)
        if t.get("target_price") is not None or t.get("stop_loss_price") is not None
    ]
    if not open_trades:
        return ReplayReport(
            date=date,
            strategy=strategy,
            trades_scanned=0,
            triggered_target=0,
            triggered_stop_loss=0,
        )

    tickers = sorted({t["ticker"] for t in open_trades})
    if bars is None:
        bars = await _fetch_intraday_1m(tickers, date)

    triggered_tp = 0
    triggered_sl = 0
    prev_snap = store.previous_snapshot(strategy, date)
    prev_equity = prev_snap["equity_value"] if prev_snap else STARTING_CAPITAL

    for trade in open_trades:
        df = bars.get(trade["ticker"])
        if df is None or len(df) == 0:
            continue
        # Trades opened today have entry timestamp == today's first
        # bar; we only walk bars AT OR AFTER the open price's bar.
        # For simplicity (and to align with capture_open_prices using
        # the day's first OHLC open), walk every bar in the day.
        first_hit = _scan_bars_for_trigger(
            df,
            direction=int(trade["direction"]),
            target=trade.get("target_price"),
            stop_loss=trade.get("stop_loss_price"),
        )
        if first_hit is None:
            continue
        trigger_price, close_reason, hit_ts = first_hit

        notional = abs(trade["open_weight"]) * prev_equity
        pct_move = (trigger_price - trade["open_price"]) / trade["open_price"]
        pnl = int(trade["direction"]) * pct_move * notional

        store.close_trade(
            trade["id"],
            closed_at=date,
            close_price=trigger_price,
            realized_pnl=pnl,
            close_reason=close_reason,
            closed_ts=hit_ts.timestamp() if hasattr(hit_ts, "timestamp") else None,
        )
        if close_reason == "target":
            triggered_tp += 1
        else:
            triggered_sl += 1
        logger.info(
            "paper_trading.intraday.replay: %s %s %s breached at ₹%.2f "
            "(ts=%s, open ₹%.2f) pnl=₹%+.2f",
            strategy,
            close_reason,
            trade["ticker"],
            trigger_price,
            hit_ts,
            trade["open_price"],
            pnl,
        )

    return ReplayReport(
        date=date,
        strategy=strategy,
        trades_scanned=len(open_trades),
        triggered_target=triggered_tp,
        triggered_stop_loss=triggered_sl,
    )


def _scan_bars_for_trigger(
    df, *, direction: int, target: Optional[float], stop_loss: Optional[float]
):
    """Walk a 1m OHLC frame forward; return (trigger_price, reason, ts)
    of the FIRST bar whose high/low breached target or stop_loss.

    Stop-loss wins on a same-bar tie (risk-first). For LONG positions
    we check (high >= target) and (low <= stop). For SHORTs, mirrored.
    """
    for ts, row in df.iterrows():
        try:
            hi = float(row.get("High"))
            lo = float(row.get("Low"))
        except (TypeError, ValueError):
            continue
        if direction > 0:
            sl_hit = stop_loss is not None and lo <= stop_loss
            tp_hit = target is not None and hi >= target
            if sl_hit:
                return stop_loss, "stop_loss", ts
            if tp_hit:
                return target, "target", ts
        else:
            sl_hit = stop_loss is not None and hi >= stop_loss
            tp_hit = target is not None and lo <= target
            if sl_hit:
                return stop_loss, "stop_loss", ts
            if tp_hit:
                return target, "target", ts
    return None


async def _fetch_intraday_1m(tickers: list[str], date: str) -> dict:
    """One yfinance batch call → ``{ticker: DataFrame of 1m bars for
    the trading day}``. yfinance supports up to 7 days of 1m history,
    so this works for today's EOD replay (today + previous 6 days).
    """
    if not tickers:
        return {}

    def _fetch() -> dict:
        import pandas as pd
        import yfinance as yf

        out: dict = {}
        try:
            df = yf.download(
                tickers=tickers,
                period="2d",
                interval="1m",
                progress=False,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
            if df is None or df.empty:
                return out
        except Exception as e:
            logger.warning(
                "paper_trading.intraday.replay: yfinance 1m fetch failed (%s)", e
            )
            return out

        # Trim each ticker's frame to bars on the target date (UTC) —
        # otherwise an extra session would double-count triggers.
        target_date = pd.Timestamp(date).date()

        if len(tickers) == 1:
            sub = df
            sub = (
                sub[sub.index.normalize().date == target_date]
                if hasattr(sub.index, "normalize")
                else sub
            )
            if not sub.empty:
                out[tickers[0]] = sub
            return out

        for t in tickers:
            try:
                sub = df[t].dropna(how="all")
                # MultiIndex columns flatten; normalise the index date.
                idx_dates = pd.Series(sub.index).dt.tz_localize(None).dt.date
                sub = sub[idx_dates.values == target_date]
                if not sub.empty:
                    out[t] = sub
            except Exception:
                continue
        return out

    return await asyncio.to_thread(_fetch)


async def replay_intraday_triggers_all(date: str) -> list[ReplayReport]:
    """Convenience: run replay for every strategy. Used by the EOD
    cron — called BEFORE finalize_eod_all so closures land in the
    same daily snapshot."""
    return [await replay_intraday_triggers(date, s) for s in STRATEGIES]


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
        mcaps = {
            t: v["market_cap"] for t, v in mcaps_table.items() if v.get("market_cap")
        }
        if not mcaps:
            mcaps = await universe.refresh_market_caps()
        target = compute_market_cap_weights(directions, mcaps)

    open_trades = store.list_open_trades(strategy)
    open_by_ticker = {t["ticker"]: t for t in open_trades}

    # Trades to close. Three reasons may apply (priority in this order):
    #   time_horizon      — third barrier: days_held > max_hold_days
    #   direction_change  — ticker dropped from today's target, OR
    #                       direction flipped sign vs today's prediction
    # Already-triggered SL/TP closes won't appear in open_trades since
    # they have closed_at != NULL after monitor_triggers or replay.
    to_close: dict[str, str] = {}
    for ticker, trade in open_by_ticker.items():
        max_hold = trade.get("max_hold_days")
        days_held = _days_between(trade["opened_at"], date)
        if max_hold is not None and days_held >= int(max_hold):
            to_close[ticker] = "time_horizon"
            continue
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
            trade["id"],
            closed_at=date,
            close_price=px,
            realized_pnl=pnl,
            close_reason=reason,
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
        store.upsert_position_snapshot(
            {
                "date": date,
                "strategy": strategy,
                "ticker": ticker,
                "direction": trade["direction"],
                "weight": trade["open_weight"],
                "notional": notional,
                "entry_date": trade["opened_at"],
                "entry_price": trade["open_price"],
                "current_price": px,
                "days_held": _days_between(trade["opened_at"], date),
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": pct_move * trade["direction"],
            }
        )

    # Pick up any transaction costs from trades OPENED today (market-open
    # phase recorded them on the trade row; we charge equity here so the
    # snapshot reflects them).
    todays_opens = [
        t
        for t in store.list_trades(strategy, limit=200, only_closed=False)
        if t.get("opened_at") == date and t.get("opened_via") == "market_open"
    ]
    transaction_costs += TRANSACTION_COST * len(todays_opens)

    # Pick up triggered closes from today (SL/TP fires that happened
    # during monitor_triggers and so don't appear in to_close).
    triggered_today = [
        t
        for t in store.list_trades(strategy, limit=500, only_closed=True)
        if t.get("closed_at") == date
        and t.get("close_reason") in ("stop_loss", "target")
    ]
    closed_triggered = len(triggered_today)
    triggered_pnl = sum((t.get("realized_pnl") or 0.0) for t in triggered_today)
    triggered_costs = TRANSACTION_COST * closed_triggered
    realized_pnl += triggered_pnl
    transaction_costs += triggered_costs

    yesterday_unrealized = (
        sum(
            p["unrealized_pnl"]
            for p in store.list_positions(strategy, prev_snap["date"])
        )
        if prev_snap
        else 0.0
    )

    daily_pnl = (
        realized_pnl + (unrealized_total - yesterday_unrealized) - transaction_costs
    )
    equity_today = prev_equity + daily_pnl
    daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0.0

    n_long = sum(1 for t in current_open if t["direction"] > 0)
    n_short = sum(1 for t in current_open if t["direction"] < 0)
    n_neutral = max(0, len(universe.NIFTY50_TICKERS) - n_long - n_short)

    store.upsert_portfolio_snapshot(
        {
            "date": date,
            "strategy": strategy,
            "equity_value": equity_today,
            "cash": equity_today,
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
        "paper_trading.intraday: finalize_eod(%s, %s) — equity=₹%.2f "
        "daily_pnl=₹%+.2f closed_dir=%d closed_triggered=%d open=%d costs=₹%.2f",
        date,
        strategy,
        equity_today,
        daily_pnl,
        closed_dir,
        closed_triggered,
        len(current_open),
        transaction_costs,
    )
    return FinalizeReport(
        date=date,
        strategy=strategy,
        equity_value=equity_today,
        daily_pnl=daily_pnl,
        transaction_costs=transaction_costs,
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


# ── Rebalance-at-close (the new single-shot daily cycle) ─────────


@dataclass
class RebalanceReport:
    date: str
    strategy: str
    equity_value: float
    daily_pnl: float
    transaction_costs: float
    triggered_target: int  # from intraday replay
    triggered_stop_loss: int  # from intraday replay
    closed_for_direction_change: int
    opened_at_close: int
    n_open_positions: int


async def rebalance_at_close(
    date: str,
    strategy: str,
    *,
    close_prices: Optional[dict[str, float]] = None,
    intraday_bars: Optional[dict] = None,
    force: bool = False,
) -> RebalanceReport:
    """Single daily cycle: replay → close direction-changes → open new
    at today's close → MTM survivors → snapshot.

    Replaces the start-of-day-open / EOD-close split. Positions opened
    by this routine are held at today's close price and carry over to
    the next trading day until a future rebalance closes them.

    Idempotent on UNIQUE(date, strategy). Safe to re-run; replay won't
    double-close (open_trades excludes already-closed rows) and the
    snapshot upsert overwrites the previous row.

    Trading-day gate: skips ``date`` cleanly if NSE was closed (weekend
    or holiday) so callers (admin endpoints, manual replays) can't
    accidentally book "direction flip" rows at stale Friday prices.
    Pass ``force=True`` to bypass the gate — only useful for tests with
    mocked close_prices on a synthetic date.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    if not force:
        from .calendar import is_nse_trading_day, next_nse_trading_day

        if not is_nse_trading_day(date):
            logger.info(
                "paper_trading.rebalance: SKIP %s %s (not an NSE trading day, next: %s)",
                date,
                strategy,
                next_nse_trading_day(date).isoformat(),
            )
            # Surface as an empty report (zero touches, zero PnL) rather
            # than raising so the cron handler doesn't have to special-case.
            prev_snap = store.previous_snapshot(strategy, date)
            prev_equity = prev_snap["equity_value"] if prev_snap else STARTING_CAPITAL
            return RebalanceReport(
                date=date,
                strategy=strategy,
                equity_value=prev_equity,
                daily_pnl=0.0,
                transaction_costs=0.0,
                triggered_target=0,
                triggered_stop_loss=0,
                closed_for_direction_change=0,
                opened_at_close=0,
                n_open_positions=len(store.list_open_trades(strategy)),
            )

    # ── 1. Replay intraday 1m bars to catch SL/TP fires ──────────
    replay_report = await replay_intraday_triggers(
        date,
        strategy,
        bars=intraday_bars,
    )

    # ── 2. Compute today's target weights ────────────────────────
    preds = store.list_predictions(date=date)
    directions = {p["ticker"]: int(p["direction"]) for p in preds}
    pred_meta = {p["ticker"]: p for p in preds}

    if strategy == "equal_weight":
        target = compute_equal_weights(directions)
    else:
        mcaps_table = store.get_market_caps()
        mcaps = {
            t: v["market_cap"] for t, v in mcaps_table.items() if v.get("market_cap")
        }
        if not mcaps:
            mcaps = await universe.refresh_market_caps()
        target = compute_market_cap_weights(directions, mcaps)

    # ── 3. Diff against currently-open trades ────────────────────
    open_trades = store.list_open_trades(strategy)
    open_by_ticker = {t["ticker"]: t for t in open_trades}

    to_close: dict[str, str] = {}
    for ticker, trade in open_by_ticker.items():
        # Time barrier expires first — close on age, even if direction unchanged
        max_hold = trade.get("max_hold_days")
        days_held = _days_between(trade["opened_at"], date)
        if max_hold is not None and days_held >= int(max_hold):
            to_close[ticker] = "time_horizon"
            continue
        if ticker not in target:
            to_close[ticker] = "direction_change"
        elif (target[ticker] > 0) != (trade["direction"] > 0):
            to_close[ticker] = "direction_change"

    # Open everything that should be on the book today and isn't
    # already held in the right direction. Tickers being closed for
    # direction_change / time_horizon are eligible to be re-opened
    # in today's direction at today's close — without this the short
    # we just closed would never get flipped long.
    held_and_keeping = set(open_by_ticker.keys()) - set(to_close.keys())
    to_open = {t for t in target.keys() if t not in held_and_keeping}

    # ── 4. Fetch close prices for everything we touch ────────────
    needed = list(set(to_close.keys()) | to_open | set(open_by_ticker.keys()))
    if close_prices is None and needed:
        close_prices = await get_quote_source().get_ltps(needed)
    close_prices = close_prices or {}

    prev_snap = store.previous_snapshot(strategy, date)
    prev_equity = prev_snap["equity_value"] if prev_snap else STARTING_CAPITAL

    # ── 5. Close direction-change / time-horizon trades ──────────
    realized_pnl = 0.0
    transaction_costs = 0.0
    closed_dir = 0
    for ticker, reason in to_close.items():
        trade = open_by_ticker[ticker]
        px = close_prices.get(ticker, trade["open_price"])
        notional = abs(trade["open_weight"]) * prev_equity
        pct_move = (px - trade["open_price"]) / trade["open_price"]
        pnl = trade["direction"] * pct_move * notional
        realized_pnl += pnl
        transaction_costs += TRANSACTION_COST
        closed_dir += 1
        store.close_trade(
            trade["id"],
            closed_at=date,
            close_price=px,
            realized_pnl=pnl,
            close_reason=reason,
        )

    # ── 6. Open new trades at today's CLOSE (the key behaviour shift)
    opened_at_close = 0
    for ticker in to_open:
        if ticker not in close_prices:
            logger.warning(
                "paper_trading.rebalance: no close price for %s — skipping open", ticker
            )
            continue
        direction = 1 if target[ticker] > 0 else -1
        meta = pred_meta.get(ticker, {})
        store.open_trade(
            strategy=strategy,
            ticker=ticker,
            direction=direction,
            opened_at=date,
            open_price=close_prices[ticker],
            open_weight=target[ticker],
            transaction_cost=TRANSACTION_COST,
            target_price=meta.get("target_price"),
            stop_loss_price=meta.get("stop_loss_price"),
            max_hold_days=meta.get("max_hold_days"),
            opened_via="rebalance_close",
        )
        opened_at_close += 1
        transaction_costs += TRANSACTION_COST

    # Pick up triggered closes (from the replay step above) — their
    # realized PnL needs to roll into today's daily_pnl + their txn
    # cost into today's costs.
    triggered_today = [
        t
        for t in store.list_trades(strategy, limit=500, only_closed=True)
        if t.get("closed_at") == date
        and t.get("close_reason") in ("stop_loss", "target")
    ]
    realized_pnl += sum((t.get("realized_pnl") or 0.0) for t in triggered_today)
    transaction_costs += TRANSACTION_COST * len(triggered_today)

    # ── 7. MTM survivors + write snapshots ───────────────────────
    current_open = store.list_open_trades(strategy)
    unrealized_total = 0.0
    gross_notional = 0.0
    net_notional = 0.0
    store.clear_position_snapshots(date, strategy)
    for trade in current_open:
        ticker = trade["ticker"]
        px = close_prices.get(ticker, trade["open_price"])
        notional = abs(trade["open_weight"]) * prev_equity
        pct_move = (px - trade["open_price"]) / trade["open_price"]
        unrealized = trade["direction"] * pct_move * notional
        unrealized_total += unrealized
        gross_notional += notional
        net_notional += trade["direction"] * notional
        store.upsert_position_snapshot(
            {
                "date": date,
                "strategy": strategy,
                "ticker": ticker,
                "direction": trade["direction"],
                "weight": trade["open_weight"],
                "notional": notional,
                "entry_date": trade["opened_at"],
                "entry_price": trade["open_price"],
                "current_price": px,
                "days_held": _days_between(trade["opened_at"], date),
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": pct_move * trade["direction"],
            }
        )

    yesterday_unrealized = (
        sum(
            p["unrealized_pnl"]
            for p in store.list_positions(strategy, prev_snap["date"])
        )
        if prev_snap
        else 0.0
    )

    daily_pnl = (
        realized_pnl + (unrealized_total - yesterday_unrealized) - transaction_costs
    )
    equity_today = prev_equity + daily_pnl
    daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0.0

    n_long = sum(1 for t in current_open if t["direction"] > 0)
    n_short = sum(1 for t in current_open if t["direction"] < 0)
    n_neutral = max(0, len(universe.NIFTY50_TICKERS) - n_long - n_short)

    store.upsert_portfolio_snapshot(
        {
            "date": date,
            "strategy": strategy,
            "equity_value": equity_today,
            "cash": equity_today,
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
        "paper_trading.rebalance(%s, %s) — equity=₹%.2f daily_pnl=₹%+.2f "
        "opened=%d closed_dir=%d triggered=(tp:%d sl:%d) open=%d costs=₹%.2f",
        date,
        strategy,
        equity_today,
        daily_pnl,
        opened_at_close,
        closed_dir,
        replay_report.triggered_target,
        replay_report.triggered_stop_loss,
        len(current_open),
        transaction_costs,
    )

    return RebalanceReport(
        date=date,
        strategy=strategy,
        equity_value=equity_today,
        daily_pnl=daily_pnl,
        transaction_costs=transaction_costs,
        triggered_target=replay_report.triggered_target,
        triggered_stop_loss=replay_report.triggered_stop_loss,
        closed_for_direction_change=closed_dir,
        opened_at_close=opened_at_close,
        n_open_positions=len(current_open),
    )


async def rebalance_at_close_all(date: str) -> list[RebalanceReport]:
    """Run the daily rebalance for both strategies. Used by the
    cron + the admin POST /api/paper-trading/rebalance endpoint."""
    return [await rebalance_at_close(date, s) for s in STRATEGIES]
