"""LangChain @tool wrappers around findata.

Same underlying functions as ``finagent/agents/debate/tools.py``, but
exposed via langchain_core.tools.tool so LangGraph nodes can bind
them to a ChatModel via ``llm.bind_tools([...])``.

We deliberately don't share code with debate/tools.py — that file
uses Agents-SDK ``@function_tool``, which produces a different
metadata shape. Forking the wrappers is cheaper than building an
adapter that pretends both decorators are the same thing.

Per-role tool subsets (assigned in nodes.analysts):

  Market analyst       → indicators / S/R / patterns / regime / chart / arima
  News analyst         → yfinance_news / gdelt_news / web_search (optional)
  Fundamentals analyst → fundamentals / analyst_consensus / earnings / returns_stats
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool


logger = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────


def _safe_call(fn_name: str, fn, **kwargs) -> str:
    """Invoke a findata function and return JSON. On any exception,
    return a JSON error envelope so the agent can decide whether to
    retry / change params / give up — never raises out to the graph."""
    try:
        result = fn(**kwargs)
    except Exception as exc:
        logger.exception("trading_panel tool %s failed", fn_name)
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "tool": fn_name,
        })
    return json.dumps(result, default=str)


# ── News / sentiment ────────────────────────────────────────────────


@tool
def fetch_yfinance_news(ticker: str, max_records: int = 15) -> str:
    """Fetch recent yfinance news headlines for a ticker.

    Returns JSON: {ticker, articles: [{title, publisher, link, ts, ...}]}.
    Best for US-listed equities; sparse coverage for .NS tickers.
    """
    from findata.news_yfinance import get_yfinance_news
    return _safe_call("fetch_yfinance_news", get_yfinance_news,
                      ticker=ticker, max_records=max_records)


@tool
def fetch_gdelt_news(
    company_query: str,
    sector_query: str = "",
    days: int = 7,
    max_records: int = 25,
) -> str:
    """Fetch GDELT news with tone scores for a company AND its sector.

    Two parallel GDELT queries:
      - ``company_query`` (required, e.g. "Apple Inc iPhone Services" or
        "Reliance Industries petrochemicals")
      - ``sector_query`` (optional, e.g. "consumer electronics" or
        "Indian conglomerates")

    Returns JSON with both result lists, each article carrying
    {title, source, url, ts, tone}. Tone ∈ [-10, 10]; positive = upbeat.
    Use the tone column to weight headlines (don't just count them).

    Note: NO ``ticker`` parameter — GDELT is keyword-based, not symbol-
    based. Pass a natural-language company name as company_query.
    """
    from findata.news_gdelt import get_gdelt_news
    sector = sector_query if sector_query else None
    return _safe_call("fetch_gdelt_news", get_gdelt_news,
                      company_query=company_query,
                      sector_query=sector,
                      days=days,
                      max_records=max_records)


# ── Fundamentals ────────────────────────────────────────────────────


@tool
def fetch_equity_fundamentals(ticker: str) -> str:
    """30 fundamentals fields via yfinance: name / sector / industry,
    market_cap, P/E (trailing + forward), PEG, P/B, P/S, EV/EBITDA,
    EV/revenue, profit_margin, operating_margin, gross_margin, ROE, ROA,
    revenue_growth, earnings_growth, total_cash, total_debt,
    free_cash_flow, operating_cash, dividend_yield, payout_ratio, beta,
    current_price, 52w high/low. JSON."""
    from findata.fundamentals import get_equity_fundamentals
    # findata.get_equity_fundamentals takes a LIST of tickers and returns
    # a DataFrame; one row per ticker. Wrap the singular @tool param into
    # a list and convert the row-frame to a dict for the agent.
    return _safe_call(
        "fetch_equity_fundamentals",
        lambda **kw: get_equity_fundamentals(tickers=[kw["ticker"]]).to_dict(orient="index"),
        ticker=ticker,
    )


@tool
def fetch_analyst_consensus(ticker: str) -> str:
    """Wall-Street consensus: target_mean/high/low/median, current_price,
    implied upside %, recommendation key (strong_buy/buy/hold/.../sell),
    recommendation mean (1=strong_buy, 5=sell), # analysts. JSON."""
    from findata.analyst_consensus import get_analyst_consensus
    return _safe_call(
        "fetch_analyst_consensus",
        lambda **kw: get_analyst_consensus(tickers=[kw["ticker"]]).to_dict(orient="index"),
        ticker=ticker,
    )


@tool
def fetch_earnings_calendar(ticker: str) -> str:
    """Past + upcoming earnings rows. Each row: date, EPS estimate,
    EPS actual, surprise %, is_past flag. JSON."""
    from findata.earnings_calendar import get_earnings_calendar
    return _safe_call(
        "fetch_earnings_calendar",
        lambda **kw: get_earnings_calendar(ticker=kw["ticker"]).reset_index().to_dict(orient="records"),
        ticker=ticker,
    )


@tool
def fetch_returns_stats(ticker: str, window_days: int = 252) -> str:
    """Annualised vol / Sharpe / max DD / beta vs SPY / alpha over a
    rolling window. window_days default 252 (~1y trading days). JSON.
    Note: parameter is ``window_days`` (matches findata signature)."""
    from findata.returns_stats import compute_returns_stats
    return _safe_call(
        "fetch_returns_stats",
        lambda **kw: compute_returns_stats(
            ticker=kw["ticker"], window_days=kw["window_days"],
        ).to_dict(),
        ticker=ticker, window_days=window_days,
    )


# ── Technical analysis ──────────────────────────────────────────────


@tool
def compute_trend_indicators(ticker: str, lookback_days: int = 252) -> str:
    """SMA(50/200) + EMA(20/50) + RSI(14) + MACD + ADX + Bollinger snapshot.
    Includes semantic flags (golden_cross, oversold, etc.). JSON."""
    from findata.trend_indicators import compute_trend_indicators as _ti
    return _safe_call("compute_trend_indicators", _ti,
                      ticker=ticker, lookback_days=lookback_days)


@tool
def compute_support_resistance(
    ticker: str,
    lookback_days: int = 252,
    n_levels: int = 5,
) -> str:
    """Algorithmic S/R levels with touch counts + nearest support / resistance
    for the current price. Uses scipy.signal.find_peaks + KMeans. JSON."""
    from findata.support_resistance import compute_support_resistance as _sr
    return _safe_call("compute_support_resistance", _sr,
                      ticker=ticker, lookback_days=lookback_days,
                      n_levels=n_levels)


@tool
def detect_candlestick_patterns(ticker: str, lookback_days: int = 60) -> str:
    """pandas-ta cdl_pattern("all"): all ~60 patterns with bullish/bearish
    flags + dates. JSON."""
    from findata.candlestick_patterns import detect_candlestick_patterns as _detect
    return _safe_call("detect_candlestick_patterns", _detect,
                      ticker=ticker, lookback_days=lookback_days)


@tool
def compute_trend_regime(ticker: str, window_days: int = 252) -> str:
    """Hurst exponent (R/S) + linear-regression drift over the window.
    Classifies regime as trending / mean-reverting / random. JSON."""
    from findata.trend_regime import compute_trend_regime as _tr
    return _safe_call("compute_trend_regime", _tr,
                      ticker=ticker, window_days=window_days)


@tool
def arima_forecast(
    ticker: str,
    lookback_days: int = 365,
    forecast_days: int = 20,
) -> str:
    """SARIMA grid-search → forward forecast with 95% CI + bullish/bearish/
    neutral signal. Returns JSON {best_order, signal, forecast_return_pct,
    summary, ...}. Use as a quantitative anchor in your KEY DATA section."""
    from findata.arima_forecast import fit_arima_forecast as _fit
    return _safe_call("arima_forecast", _fit,
                      ticker=ticker, lookback_days=lookback_days,
                      forecast_days=forecast_days)


@tool
def fetch_macro_snapshot(country: str = "US") -> str:
    """Curated macro snapshot — interest rates, inflation, growth, credit
    spreads, FX, commodities — plus 30/90/365-day changes for each.

    One call, ~25 indicators grouped by theme. Use this to ground claims
    about debt-financing cost (rates), demand (consumer sentiment / PMI),
    sector rotation (curve shape), risk appetite (credit spreads + VIX).

    For .NS / Indian tickers pass country='IN' — the snapshot still
    returns the US-side macro plus USD/INR + dollar index, with a note
    flagging which Indian-domestic series aren't on FRED.

    Returns JSON with keys: indicators (label → value/unit/changes),
    groups (theme → [labels]), summary (one-line regime read), as_of.
    """
    from findata.macro_indicators import get_macro_snapshot
    return _safe_call("fetch_macro_snapshot", get_macro_snapshot, country=country)


@tool
def fetch_yield_curve() -> str:
    """US Treasury yield curve snapshot — every tenor (3M to 30Y), plus the
    same curve a year ago for shape comparison. Use this to read whether
    the curve is inverted / flat / steepening, which drives sector rotation
    (banks ⇄ rates, REITs ⇄ long end, growth ⇄ short end).

    Returns JSON: {today: [{tenor, yield}], one_year_ago: [...], summary}.
    """
    from findata.macro_indicators import get_yield_curve
    return _safe_call("fetch_yield_curve", get_yield_curve)


@tool
def plot_ohlc_chart(
    ticker: str,
    lookback_days: int = 252,
    with_sr: bool = True,
    with_indicators: bool = True,
) -> str:
    """Render an OHLC candlestick chart with optional S/R + 50/200 SMA + RSI.

    Returns JSON with a ``markdown_image`` string ready to paste verbatim
    into the analyst's report — when the panel renders, the chart shows
    inline. Falls back to italic ``*Chart unavailable: ...*`` if yfinance
    is rate-limited or the renderer fails; paste the fallback as-is, do
    NOT improvise an apology. Call once per analyst turn — it's the
    primary visual the user sees.
    """
    from findata.ohlc_chart import plot_ohlc_chart as _plot
    return _safe_call("plot_ohlc_chart", _plot,
                      ticker=ticker, lookback_days=lookback_days,
                      with_sr=with_sr, with_indicators=with_indicators)


# ── Tool kits per analyst role ──────────────────────────────────────


# Imported by analyst nodes via ``from .tools import MARKET_TOOLS`` etc.
# Keeping them as module-level constants makes LangChain's bind_tools
# call read like "give the market analyst its specific kit".


MARKET_TOOLS = [
    compute_trend_indicators,
    compute_support_resistance,
    detect_candlestick_patterns,
    compute_trend_regime,
    arima_forecast,
    plot_ohlc_chart,            # our visual layer — TradingAgents doesn't have this
]

NEWS_TOOLS = [
    fetch_yfinance_news,
    fetch_gdelt_news,
]

FUNDAMENTALS_TOOLS = [
    fetch_equity_fundamentals,
    fetch_analyst_consensus,
    fetch_earnings_calendar,
    fetch_returns_stats,
]

MACRO_TOOLS = [
    fetch_macro_snapshot,
    fetch_yield_curve,
]


__all__ = [
    "fetch_yfinance_news", "fetch_gdelt_news",
    "fetch_equity_fundamentals", "fetch_analyst_consensus",
    "fetch_earnings_calendar", "fetch_returns_stats",
    "compute_trend_indicators", "compute_support_resistance",
    "detect_candlestick_patterns", "compute_trend_regime",
    "arima_forecast", "plot_ohlc_chart",
    "fetch_macro_snapshot", "fetch_yield_curve",
    "MARKET_TOOLS", "NEWS_TOOLS", "FUNDAMENTALS_TOOLS", "MACRO_TOOLS",
]
