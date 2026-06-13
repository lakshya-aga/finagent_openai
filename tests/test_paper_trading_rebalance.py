"""Paper-trading rebalance + positions-panel regression tests.

Pure-SQLite tier — no LLM, no network, no kernel. Each test points the
paper-trading store at a tmp DB by monkeypatching ``store._db_path`` so
the real schema + SQL run without touching ``outputs/``.

Covers two bugs behind "the portfolio didn't update on the new debates":

  1. ``store.list_positions`` defaulted to ``MAX(position_snapshots.date)``.
     On a day the rebalance opens zero positions it writes a portfolio
     snapshot but no position-snapshot rows, so the positions panel fell
     back to an older day and showed STALE positions that disagreed with
     the (correct, flat) header. It now anchors to the latest *portfolio*
     snapshot date.

  2. A rebalance that wanted to open positions but got no usable close
     price for any of them silently skipped every open and left the book
     flat — indistinguishable on the dashboard from a genuine all-'avoid'
     day. The report now carries ``opens_requested`` /
     ``opens_skipped_no_price`` / ``quote_source_ok`` so the cron can
     flag a quote-source outage loudly.
"""

from __future__ import annotations

import asyncio

import pytest

from finagent.paper_trading import store


@pytest.fixture()
def pt_db(tmp_path, monkeypatch):
    db = tmp_path / "paper-trading-test.db"
    monkeypatch.setattr(store, "_db_path", lambda: str(db))
    # Force schema (re)creation against this tmp DB rather than
    # short-circuiting on a previous test's process-wide flag.
    monkeypatch.setattr(store, "_SCHEMA_CREATED", False)
    return db


def _portfolio_snap(date: str, strategy: str, **over):
    base = dict(
        date=date,
        strategy=strategy,
        equity_value=100_000.0,
        cash=100_000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl=0.0,
        daily_return_pct=0.0,
        transaction_costs=0.0,
        n_long=0,
        n_short=0,
        n_neutral=50,
    )
    base.update(over)
    return base


def _position_snap(date: str, strategy: str, ticker: str, **over):
    base = dict(
        date=date,
        strategy=strategy,
        ticker=ticker,
        direction=-1,
        weight=-0.5,
        notional=50_000.0,
        entry_date=date,
        entry_price=1_000.0,
        current_price=1_000.0,
        days_held=0,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
    )
    base.update(over)
    return base


def test_list_positions_anchors_to_portfolio_snapshot_date(pt_db):
    """A flat day (portfolio snapshot written, zero position rows) returns
    [] — not stale positions carried over from the last day that held any."""
    strat = "equal_weight"

    # Day 1 — a real short is held; both snapshot tables get a row.
    store.upsert_portfolio_snapshot(
        _portfolio_snap("2026-06-11", strat, gross_exposure=1.0, n_short=2, n_neutral=48)
    )
    store.upsert_position_snapshot(_position_snap("2026-06-11", strat, "HDFCBANK.NS"))

    # Day 2 — rebalance opened nothing: portfolio snapshot exists but
    # there are NO position_snapshots rows for 2026-06-12.
    store.upsert_portfolio_snapshot(_portfolio_snap("2026-06-12", strat))

    # Default (date=None) anchors to the latest PORTFOLIO snapshot
    # (2026-06-12) → zero positions, consistent with the flat header —
    # rather than falling back to 2026-06-11's stale row.
    assert store.list_positions(strat) == []

    # Explicit older date still reads that day's held positions.
    held = store.list_positions(strat, "2026-06-11")
    assert [p["ticker"] for p in held] == ["HDFCBANK.NS"]


def test_rebalance_flags_quote_source_outage(pt_db):
    """When the quote source returns no close prices, every open is
    skipped and the report marks ``quote_source_ok == False`` so the
    cron can distinguish an outage from a genuine all-'avoid' day."""
    from finagent.paper_trading import intraday, predictions

    date = "2026-06-12"
    # Two actionable predictions for an empty book.
    predictions.record_prediction(
        date=date, ticker="HDFCBANK.NS", direction=-1, source="manual"
    )
    predictions.record_prediction(
        date=date, ticker="ICICIBANK.NS", direction=-1, source="manual"
    )

    report = asyncio.run(
        intraday.rebalance_at_close(
            date,
            "equal_weight",
            close_prices={},  # simulate a total quote-source failure
            intraday_bars={},  # no intraday replay fetch
            force=True,  # bypass the NSE trading-calendar gate
        )
    )

    assert report.opens_requested == 2
    assert report.opened_at_close == 0
    assert report.opens_skipped_no_price == 2
    assert report.quote_source_ok is False
    assert report.n_open_positions == 0


def test_rebalance_healthy_when_prices_available(pt_db):
    """Sanity inverse: with close prices present the book actually opens
    and ``quote_source_ok`` stays True."""
    from finagent.paper_trading import intraday, predictions

    date = "2026-06-12"
    predictions.record_prediction(
        date=date, ticker="HDFCBANK.NS", direction=-1, source="manual"
    )
    predictions.record_prediction(
        date=date, ticker="ICICIBANK.NS", direction=1, source="manual"
    )

    report = asyncio.run(
        intraday.rebalance_at_close(
            date,
            "equal_weight",
            close_prices={"HDFCBANK.NS": 1500.0, "ICICIBANK.NS": 1100.0},
            intraday_bars={},
            force=True,
        )
    )

    assert report.opens_requested == 2
    assert report.opened_at_close == 2
    assert report.opens_skipped_no_price == 0
    assert report.quote_source_ok is True
    assert report.n_open_positions == 2
