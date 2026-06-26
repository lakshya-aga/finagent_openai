"""EOD yfinance fallback for paper-trading close prices.

When the primary quote source (Groww) can't price a ticker — e.g. the
market-data REST API returns 403 "Access forbidden" — the daily
close-rebalance should fill the gap from yfinance's daily (EOD) close
rather than silently skipping the position (which booked opened=0 and
looked like "no signal"). Gated by PAPER_TRADING_EOD_FALLBACK; the
fallback only fills tickers the primary couldn't price.
"""

from __future__ import annotations

import asyncio

import pytest

from finagent.paper_trading.quotes import fetch_close_prices


class _FakePrimary:
    """Stands in for GrowwSource. Returns whatever prices it was given;
    an empty dict simulates the 403 'Access forbidden' outage."""

    name = "groww"

    def __init__(self, prices):
        self._prices = prices

    async def get_ltps(self, tickers):
        return dict(self._prices)


class _FakeFallback:
    """Stands in for the yfinance EOD source; records what it was asked
    so we can assert it's only consulted for the primary's misses."""

    name = "yfinance"

    def __init__(self, prices):
        self._prices = prices
        self.asked_for = None

    async def get_eod_closes(self, tickers):
        self.asked_for = list(tickers)
        return {t: self._prices[t] for t in tickers if t in self._prices}


def test_eod_fallback_fills_when_primary_returns_nothing(monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_EOD_FALLBACK", "1")
    primary = _FakePrimary({})  # Groww 403 → empty
    fallback = _FakeFallback({"SBIN.NS": 1035.1, "RELIANCE.NS": 1309.5})

    prices = asyncio.run(
        fetch_close_prices(
            ["SBIN.NS", "RELIANCE.NS"], primary=primary, fallback=fallback
        )
    )

    assert prices == {"SBIN.NS": 1035.1, "RELIANCE.NS": 1309.5}
    assert fallback.asked_for == ["SBIN.NS", "RELIANCE.NS"]


def test_eod_fallback_only_fills_primary_misses(monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_EOD_FALLBACK", "1")
    primary = _FakePrimary({"SBIN.NS": 1000.0})  # Groww priced one
    fallback = _FakeFallback({"RELIANCE.NS": 1309.5})

    prices = asyncio.run(
        fetch_close_prices(
            ["SBIN.NS", "RELIANCE.NS"], primary=primary, fallback=fallback
        )
    )

    # primary value preserved; fallback consulted only for the miss
    assert prices == {"SBIN.NS": 1000.0, "RELIANCE.NS": 1309.5}
    assert fallback.asked_for == ["RELIANCE.NS"]


def test_eod_fallback_disabled_by_env(monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_EOD_FALLBACK", "0")
    primary = _FakePrimary({})
    fallback = _FakeFallback({"SBIN.NS": 1035.1})

    prices = asyncio.run(
        fetch_close_prices(["SBIN.NS"], primary=primary, fallback=fallback)
    )

    assert prices == {}
    assert fallback.asked_for is None  # fallback never consulted when disabled


@pytest.mark.needs_yfinance
def test_get_eod_closes_real_yfinance_daily():
    """Integration: the real yfinance daily endpoint prices NSE tickers
    from CI/docker (skipped on the host that lacks yfinance). This is
    the endpoint the EOD fallback depends on — the intraday 1m endpoint
    is the one Yahoo blocks from cloud IPs, not this one."""
    from finagent.paper_trading.quotes import YFinanceSource

    out = asyncio.run(
        YFinanceSource().get_eod_closes(["RELIANCE.NS", "SBIN.NS"])
    )
    if not out:
        pytest.skip(
            "yfinance returned no data (rate-limited / transient) — daily "
            "endpoint verified working live on the VM"
        )
    assert set(out) <= {"RELIANCE.NS", "SBIN.NS"}
    assert all(isinstance(v, float) and v > 0 for v in out.values())
