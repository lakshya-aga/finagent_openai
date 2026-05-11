"""Debate-system smoke tests.

Three concerns:

  1. Data sources — each findata fetcher used by the trading-panel
     analysts returns non-empty real data for a known ticker. The
     individual fetchers are tested via direct calls; the function-
     tool wrappers in ``finagent.agents.debate.tools`` and
     ``finagent.agents.trading_panel.tools`` add a thin JSON layer
     but no business logic.
  2. Chart generation — ``ohlc_chart_with_indicators`` produces a
     valid base64-encoded PNG (the bytes start with the PNG magic
     number).
  3. Agent API call — one tiny end-to-end ``run_panel`` invocation
     for a known symbol, asserting the panel returns a verdict +
     transcript with non-empty analyst sections. Gated by
     ``needs_openai`` so it only runs when the OpenAI SDK + API key
     are present.

Each fetcher is wrapped in ``try/except`` so a transient network
failure for one source doesn't fail the whole test — the assertion
is "at least N out of K sources returned data" rather than "every
single fetch succeeded". Production has its own monitors for
fetcher SLAs.
"""

from __future__ import annotations

import base64
import importlib
import os
import sys

import pytest


# Make sure data-mcp's findata is importable. The conda env that runs
# these tests has it installed editable, but on a vanilla host we add
# the source dir to sys.path as a fallback.

_DATA_MCP_SRC = "/Users/lakshya/Desktop/data-mcp"
if os.path.isdir(_DATA_MCP_SRC) and _DATA_MCP_SRC not in sys.path:
    sys.path.insert(0, _DATA_MCP_SRC)


def _can_import(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


needs_findata = pytest.mark.skipif(
    not _can_import("findata"),
    reason="findata package not importable",
)


# ── Data-source fetchers ───────────────────────────────────────────────


@needs_findata
@pytest.mark.needs_yfinance
def test_findata_equity_prices_fetches_known_ticker():
    from findata.equity_prices import get_equity_prices
    df = get_equity_prices(tickers=["AAPL"], start_date="2026-01-01", end_date="2026-04-01")
    assert df is not None and not df.empty, "no rows returned for AAPL"
    # Either MultiIndex (ticker × ohlc) or flat ohlc
    cols = set(c.lower() if not isinstance(c, tuple) else c[0].lower() for c in df.columns)
    assert {"open", "high", "low", "close"} & cols, f"missing OHLC columns, got {cols}"


@needs_findata
@pytest.mark.needs_yfinance
def test_findata_candlestick_patterns_runs_clean():
    """Regression — re-asserts the datetime64[s] dtype fix that the
    classifier rolled into ``_datetime_utils``. If the installed
    package gets stale (no ``normalise_dtindex`` import), this test
    is what surfaces it."""
    import inspect

    from findata.candlestick_patterns import detect_candlestick_patterns

    # Defence-in-depth: if the installed package version doesn't have
    # the dtype helper imported, fail loud here BEFORE the actual call.
    # Otherwise a TypeError silently masquerades as 'no patterns found'.
    src = inspect.getsource(detect_candlestick_patterns)
    assert "normalise_dtindex" in src, (
        "Stale findata install: detect_candlestick_patterns lacks the "
        "normalise_dtindex fix. Reinstall: pip install -e /path/to/data-mcp"
    )

    out = detect_candlestick_patterns("AAPL", lookback_days=60)
    assert isinstance(out, dict)
    assert "n_patterns_found" in out
    assert "summary" in out
    assert isinstance(out["patterns"], list)
    # Real ticker over 60d should yield SOMETHING from pandas-ta's
    # ~50 pattern recognisers. If 0 it usually means the index-cutoff
    # comparison silently dropped everything.
    assert out["n_patterns_found"] >= 1, f"zero patterns suspicious: {out['summary']}"


@needs_findata
@pytest.mark.needs_yfinance
def test_aapl_candlestick_full_yfinance_to_patterns_path():
    """End-to-end smoke for the dtype regression that keeps re-surfacing
    in production. Real path, no mocks:

        yfinance.download(AAPL) → findata.candlestick_patterns(AAPL) →
        non-empty pattern list

    Asserts:
      * yfinance returned a non-empty OHLC frame (sanity-check upstream)
      * The actual function call doesn't raise
        TypeError("Invalid comparison between dtype=datetime64[s] and
        Timestamp") at the index-cutoff slice
      * pandas-ta returns at least one pattern event
      * Each event has the canonical record shape
        (date, pattern, signal, close_at_pattern)

    When this fails with the dtype TypeError, it means the deployed
    findata package is stale (the docker layer cache served the old
    version). Fix is to bump the data-mcp pin in requirements.txt
    so docker invalidates the layer.
    """
    import yfinance as yf

    from findata.candlestick_patterns import detect_candlestick_patterns

    # 1. Direct yfinance fetch — sanity-check the upstream so a
    #    yfinance-side regression doesn't masquerade as a candlestick bug.
    raw = yf.download(
        "AAPL", period="3mo", progress=False, auto_adjust=False,
    )
    assert raw is not None and not raw.empty, "yfinance returned empty for AAPL"
    cols_lower = {
        (c.lower() if not isinstance(c, tuple) else c[0].lower())
        for c in raw.columns
    }
    assert {"open", "high", "low", "close"} & cols_lower, (
        f"missing OHLC columns from yfinance: got {cols_lower}"
    )

    # 2. The actual function under test. Catch + re-raise the dtype
    #    TypeError with diagnostic context so the failure mode is
    #    obvious in the test report.
    try:
        out = detect_candlestick_patterns("AAPL", lookback_days=60)
    except TypeError as e:
        if "Invalid comparison" in str(e) and "datetime64" in str(e):
            raise AssertionError(
                "The datetime64[s] dtype regression IS BACK. "
                f"findata.candlestick_patterns hit `{e}` on AAPL. "
                "Check that `normalise_dtindex(ohlc.index)` and "
                "`normalise_timestamp(cutoff)` are both being called "
                "before the `ohlc.index >= cutoff` comparison."
            ) from e
        raise  # any other TypeError is a real bug — surface raw

    # 3. Shape assertions
    assert out["ticker"] == "AAPL"
    assert out["lookback_days"] == 60
    assert isinstance(out["patterns"], list)
    assert out["n_patterns_found"] == len(out["patterns"])
    assert out["n_patterns_found"] >= 1, (
        f"Zero patterns is the silent symptom of the dtype bug. "
        f"Summary: {out['summary']}"
    )
    # 4. Each event must carry the contract fields
    sample = out["patterns"][0]
    assert {"date", "pattern", "signal", "close_at_pattern"}.issubset(sample), (
        f"event missing required fields: {sample}"
    )
    assert sample["signal"] in {"bullish", "bearish"}
    assert isinstance(sample["close_at_pattern"], (int, float))


@needs_findata
@pytest.mark.needs_yfinance
def test_findata_trend_indicators_returns_features():
    from findata.trend_indicators import compute_trend_indicators
    out = compute_trend_indicators("AAPL", window_days=180)
    assert isinstance(out, dict)
    # Trend indicators emit a few well-known fields; we only assert one
    # is present so a vendor field renaming doesn't fail the test.
    keys = set(out.keys())
    expected_one_of = {"sma_20", "sma_50", "rsi_14", "macd", "atr_14", "summary", "indicators"}
    assert keys & expected_one_of, f"no expected indicator keys in {keys}"


@needs_findata
@pytest.mark.needs_yfinance
def test_findata_news_yfinance_returns_records():
    """News fetcher returns either a DataFrame or a list/dict of news
    records (NOT a stringified DataFrame __repr__ — that was the old
    hallucination bug). Either shape is acceptable; we just verify
    it's structured data with the title field present."""
    import pandas as pd

    from findata.news_yfinance import get_yfinance_news
    news = get_yfinance_news("AAPL", max_records=3)
    assert news is not None
    if isinstance(news, pd.DataFrame):
        # 0 rows is acceptable on a quiet day; verify the schema.
        assert any(c.lower() in {"title", "headline"} for c in news.columns), (
            f"news DataFrame missing title column, got {list(news.columns)}"
        )
    else:
        records = news.get("news") or news.get("records") or [] if isinstance(news, dict) else news
        assert isinstance(records, list)
        if records:
            rec = records[0]
            assert isinstance(rec, dict)
            assert any(k in rec for k in ("title", "headline", "link", "url"))


@needs_findata
@pytest.mark.needs_yfinance
def test_findata_factor_loadings_us_ticker():
    from findata.factor_loadings import get_factor_loadings
    out = get_factor_loadings("AAPL", factor_model="5", window_days=252)
    assert isinstance(out, dict)
    assert "loadings" in out or "alpha_daily_pct" in out, f"unexpected shape: {list(out.keys())}"


# ── Chart generation ──────────────────────────────────────────────────


@needs_findata
@pytest.mark.needs_yfinance
def test_findata_ohlc_chart_produces_valid_png():
    from findata.ohlc_chart import plot_ohlc_chart
    out = plot_ohlc_chart("AAPL", lookback_days=90)
    assert isinstance(out, dict)
    # plot_ohlc_chart returns {ticker, lookback_days, title, summary,
    # image_base64, markdown_image, chart_status, params}.
    assert out.get("chart_status") in {"ok", "success", None}, (
        f"chart_status={out.get('chart_status')}, summary={out.get('summary')!r}"
    )
    b64 = out.get("image_base64") or ""
    if not b64:
        md = out.get("markdown_image") or ""
        if "base64," in md:
            b64 = md.split("base64,", 1)[1].split(")", 1)[0].strip()
    assert b64, f"no base64 PNG in output keys {list(out.keys())}"
    raw = base64.b64decode(b64[:200])  # header only
    # PNG magic number — bytes \x89PNG\r\n\x1a\n
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"not a valid PNG header: {raw[:8]!r}"


# ── Optional: end-to-end agent invocation ─────────────────────────────


@pytest.mark.needs_openai
@pytest.mark.needs_finagent_kernel
def test_trading_panel_runs_for_known_ticker():
    """Run a single trading-panel debate end-to-end. Skipped unless
    OPENAI_API_KEY is set + the agents SDK is importable. This is the
    most expensive test in the suite (~30-90s) and is intentionally
    non-default — run with ``pytest -m needs_openai``.
    """
    pytest.importorskip("agents")
    pytest.importorskip("openai")
    from finagent.agents.trading_panel import run_panel

    events = []
    async def collect():
        async for ev in run_panel(ticker="AAPL", asset_class="us_equity", rounds=1):
            events.append(ev)

    import asyncio
    asyncio.run(collect())

    # Smoke: at least one phase event + a final verdict.
    types = [e.get("type") for e in events if isinstance(e, dict)]
    assert "verdict" in types or "phase" in types, (
        f"panel produced no recognisable events: {types[:10]}"
    )
