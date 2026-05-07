"""Executable function tools for debate agents.

These wrap data-mcp's findata modules with the OpenAI Agents SDK
``@function_tool`` decorator so the bull / bear analysts can invoke
them directly, without going through the data-mcp discovery layer
(``search_tools`` → ``get_tool_doc`` → write code → execute). The
debate agents are chat-style; they need executable tools with
returnable JSON, not just code-writing aids.

Each wrapper:
  * imports the findata function lazily so the module stays cheap
    to import even when only some agents need it
  * accepts plain primitive args (the SDK serialises Pydantic models
    fine but plain types are simpler for the model to reason about)
  * returns JSON strings (not DataFrames) — the model needs text it
    can quote in its argument

Web search is the OpenAI hosted ``WebSearchTool`` (built-in to the
SDK). It's added directly in agents.py, not here.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from agents import function_tool


# ─── News ────────────────────────────────────────────────────────────


@function_tool
async def fetch_yfinance_news(ticker: str, max_records: int = 15) -> str:
    """Fetch recent news headlines for a US-listed ticker via yfinance.

    Best for company-specific news on US equities. For non-US equities,
    crypto, or industry/sector context, use fetch_gdelt_news instead.

    Args:
        ticker: Yahoo Finance symbol, e.g. AAPL, MSFT.
        max_records: 1-30, default 15.

    Returns:
        JSON of {ticker, articles: [{published, title, publisher, link, summary}]}.
    """
    try:
        from findata.news_yfinance import get_yfinance_news
        df = get_yfinance_news(ticker, max_records=max_records)
        if df.empty:
            return json.dumps({"ticker": ticker, "articles": []})
        # Convert DataFrame → list of dicts with ISO timestamps for the LLM.
        df = df.reset_index()
        df["published"] = df["published"].astype(str)
        return json.dumps(
            {"ticker": ticker, "articles": df.to_dict(orient="records")},
            default=str,
        )
    except Exception as exc:
        logging.exception("fetch_yfinance_news failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker, "articles": []})


@function_tool
async def fetch_gdelt_news(
    company_query: str,
    sector_query: Optional[str] = None,
    days: int = 7,
    max_records: int = 25,
) -> str:
    """Fetch recent news from GDELT for a company AND (optionally) its sector.

    Each article carries GDELT's average tone score (-100..+100), plus
    URL, source country, domain, language. GDELT is free, no API key,
    multi-language coverage. Best for broad trawls of public sentiment;
    pair with fetch_yfinance_news for narrower US-equity coverage.

    Args:
        company_query: Required. Company-specific GDELT query
            (e.g. "NVIDIA AI chip data center").
        sector_query: Optional broader industry/sector query
            (e.g. "semiconductor manufacturing").
        days: Lookback window 1-30, default 7.
        max_records: Per-bucket cap 1-75, default 25 (so 50 max total
            when sector_query is set).

    Returns:
        JSON of {company: {query, articles}, sector: {query, articles}}.
        Each article has: seendate, title, url, domain, sourcecountry,
        language, tone, query_kind.
    """
    try:
        from findata.news_gdelt import get_gdelt_news
        result = get_gdelt_news(
            company_query=company_query,
            sector_query=sector_query,
            days=days,
            max_records=max_records,
        )
        out = {}
        for kind, df in result.items():
            df2 = df.reset_index()
            if "seendate" in df2.columns:
                df2["seendate"] = df2["seendate"].astype(str)
            out[kind] = {
                "query": str(df2["query"].iloc[0]) if not df2.empty and "query" in df2 else "",
                "articles": df2.to_dict(orient="records"),
            }
        return json.dumps(out, default=str)
    except Exception as exc:
        logging.exception("fetch_gdelt_news failed company_query=%r", company_query)
        return json.dumps({"error": str(exc), "company": {"articles": []}, "sector": {"articles": []}})


# ─── Fundamentals / consensus / events / risk stats ──────────────────


@function_tool
async def fetch_equity_fundamentals(tickers: list[str]) -> str:
    """Fetch a fundamentals snapshot for one or more equity tickers.

    Returns valuation multiples (P/E, P/B, EV/EBITDA), profitability
    (ROE, margins), growth (revenue, earnings YoY), balance sheet (cash,
    debt, FCF), dividend yield, beta, and 52-week range. One row per
    ticker.

    Args:
        tickers: Yahoo Finance ticker symbols, e.g. ["AAPL", "MSFT"].

    Returns:
        JSON of {tickers: {ticker: {field: value, ...}}}.
    """
    try:
        from findata.fundamentals import get_equity_fundamentals
        df = get_equity_fundamentals(tickers)
        if df.empty:
            return json.dumps({"tickers": {}})
        return json.dumps({"tickers": df.to_dict(orient="index")}, default=str)
    except Exception as exc:
        logging.exception("fetch_equity_fundamentals failed tickers=%r", tickers)
        return json.dumps({"error": str(exc), "tickers": {}})


@function_tool
async def fetch_analyst_consensus(tickers: list[str]) -> str:
    """Fetch the current Wall-Street consensus on one or more equity tickers.

    Returns mean/high/low target prices, current price, implied upside,
    recommendation key (strong_buy / buy / hold / underperform / sell),
    recommendation mean (1=strong buy, 5=sell), number of analysts.

    Use this to anchor your target_price reasoning — without it, target
    numbers tend to look suspiciously round.

    Args:
        tickers: Yahoo Finance ticker symbols.

    Returns:
        JSON of {tickers: {ticker: {target_mean, upside_pct,
        recommendation_key, num_analysts, ...}}}.
    """
    try:
        from findata.analyst_consensus import get_analyst_consensus
        df = get_analyst_consensus(tickers)
        if df.empty:
            return json.dumps({"tickers": {}})
        return json.dumps({"tickers": df.to_dict(orient="index")}, default=str)
    except Exception as exc:
        logging.exception("fetch_analyst_consensus failed tickers=%r", tickers)
        return json.dumps({"error": str(exc), "tickers": {}})


@function_tool
async def fetch_earnings_calendar(
    ticker: str,
    days_back: int = 365,
    days_forward: int = 90,
) -> str:
    """Fetch past + upcoming earnings rows for a single ticker.

    Each row is one earnings event with EPS estimate vs. actual (when
    reported) and surprise %. Use for time-horizon reasoning — does
    the thesis depend on the next print, or play out on a longer arc?

    Args:
        ticker: Yahoo Finance ticker symbol.
        days_back: How far back, 1-1825. Default 365.
        days_forward: How far forward, 0-365. Default 90.

    Returns:
        JSON of {ticker, events: [{date, eps_estimate, eps_actual,
        surprise_pct, is_past}, ...]}.
    """
    try:
        from findata.earnings_calendar import get_earnings_calendar
        df = get_earnings_calendar(ticker, days_back=days_back, days_forward=days_forward)
        if df.empty:
            return json.dumps({"ticker": ticker, "events": []})
        df = df.reset_index()
        df["date"] = df["date"].astype(str)
        return json.dumps({"ticker": ticker, "events": df.to_dict(orient="records")}, default=str)
    except Exception as exc:
        logging.exception("fetch_earnings_calendar failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker, "events": []})


@function_tool
async def fetch_returns_stats(
    ticker: str,
    window_days: int = 252,
    benchmark: Optional[str] = "SPY",
    risk_free_rate: float = 0.0,
) -> str:
    """Annualised return / vol / Sharpe / max drawdown / beta over a window.

    Pulls fresh price history and computes the standard risk-stat pack.
    Use this whenever you need to ground a "this stock is x% volatile"
    or "this beta makes it a tactical hedge" claim.

    Args:
        ticker: Yahoo Finance ticker symbol.
        window_days: Lookback in days, 20-2520. Default 252 (~1y).
        benchmark: Beta benchmark ticker. Default 'SPY'. None to skip beta.
        risk_free_rate: Annual rate for Sharpe. 0.0 default; pass ~0.045
            for current T-bill yield.

    Returns:
        JSON with annual_return, annual_vol, sharpe, max_drawdown, beta,
        alpha_annual, corr_to_benchmark, n_obs.
    """
    try:
        from findata.returns_stats import compute_returns_stats
        s = compute_returns_stats(
            ticker, window_days=window_days,
            benchmark=benchmark, risk_free_rate=risk_free_rate,
        )
        return json.dumps(s.to_dict(), default=str)
    except Exception as exc:
        logging.exception("fetch_returns_stats failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker})


# ─── Technical analysis ──────────────────────────────────────────────


@function_tool
async def detect_candlestick_patterns(ticker: str, lookback_days: int = 90) -> str:
    """Detect candlestick patterns (hammer, engulfing, doji, morning star, …)
    on a ticker's recent OHLC history.

    Returns the non-zero pattern events ordered most-recent first, plus
    a one-line summary the analyst can quote verbatim.

    Use this for technical-pattern claims like 'AAPL printed a bullish
    hammer at $186 support on 2026-04-22.'

    Args:
        ticker: Yahoo Finance ticker symbol.
        lookback_days: Window in days, 20-1825. Default 90.

    Returns:
        JSON of {ticker, lookback_days, n_patterns_found, patterns:
        [{date, pattern, signal, close_at_pattern}], summary}.
    """
    try:
        from findata.candlestick_patterns import detect_candlestick_patterns as _detect
        return json.dumps(_detect(ticker, lookback_days=lookback_days), default=str)
    except Exception as exc:
        logging.exception("detect_candlestick_patterns failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker, "patterns": []})


@function_tool
async def compute_support_resistance(
    ticker: str,
    lookback_days: int = 252,
    n_levels: int = 5,
    tolerance_pct: float = 0.5,
) -> str:
    """Detect candidate support / resistance levels on a ticker.

    Algorithmic detection — peaks/troughs (scipy.signal.find_peaks)
    clustered into n_levels via KMeans, ranked by touch count.

    Use this to ground claims like 'support at $185 with 4 touches,
    last bounce 2026-04-19'.

    Args:
        ticker: Yahoo Finance ticker symbol.
        lookback_days: Window in days, 30-1825. Default 252.
        n_levels: Number of S/R candidates, 2-12. Default 5.
        tolerance_pct: Touch tolerance band in %, default 0.5.

    Returns:
        JSON of {ticker, current_price, levels: [{price, type, touches,
        last_touch, strength}], nearest_support, nearest_resistance,
        summary}.
    """
    try:
        from findata.support_resistance import compute_support_resistance as _sr
        return json.dumps(
            _sr(ticker, lookback_days=lookback_days,
                n_levels=n_levels, tolerance_pct=tolerance_pct),
            default=str,
        )
    except Exception as exc:
        logging.exception("compute_support_resistance failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker, "levels": []})


@function_tool
async def compute_trend_indicators(ticker: str, window_days: int = 252) -> str:
    """Snapshot of trend / momentum / volatility indicators on a ticker.

    Computes SMA 20/50/200, EMA 12/26, RSI(14), MACD, ADX, Bollinger
    Bands. Returns latest values plus pre-computed semantic flags
    (above_50d, golden_cross_recent, rsi_state, macd_bullish_cross_5d,
    adx_state, bb_state) so claims like 'RSI 72 overbought, ADX 28
    confirms strong trend, golden cross 12d ago' are one tool call.

    Args:
        ticker: Yahoo Finance ticker symbol.
        window_days: Window in days, 220-2520. Default 252.

    Returns:
        JSON of {ticker, current_price, trend, momentum, volatility,
        trend_strength, summary}.
    """
    try:
        from findata.trend_indicators import compute_trend_indicators as _ti
        return json.dumps(_ti(ticker, window_days=window_days), default=str)
    except Exception as exc:
        logging.exception("compute_trend_indicators failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker})


@function_tool
async def compute_trend_regime(ticker: str, window_days: int = 252) -> str:
    """Classify a ticker as trending / random-walk / mean-reverting via
    Hurst exponent + linear-regression drift on log prices.

    Hurst < 0.45 → mean-reverting; ≈ 0.5 → random walk; > 0.55 →
    trending. The R² of the linear fit says how cleanly the trend
    line fits.

    Use this before claiming 'this is genuinely trending' vs 'this is
    just noisy and the recent move is regression to mean'.

    Args:
        ticker: Yahoo Finance ticker symbol.
        window_days: Window in days, 60-2520. Default 252.

    Returns:
        JSON of {ticker, hurst_exponent, regime, linear_slope_annualized,
        slope_r2, n_obs, summary}.
    """
    try:
        from findata.trend_regime import compute_trend_regime as _tr
        return json.dumps(_tr(ticker, window_days=window_days), default=str)
    except Exception as exc:
        logging.exception("compute_trend_regime failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker})


@function_tool
async def arima_forecast(
    ticker: str,
    lookback_days: int = 365,
    forecast_days: int = 20,
) -> str:
    """Fit a SARIMA model and forecast future prices as a quantitative trend signal.

    Internally runs a small grid-search over (p,d,q) ∈ {0,1,2}×{0,1}×{0,1,2}
    and three seasonal options (none, weekly AR/MA, weekly seasonal-MA),
    picks the model with the lowest AIC, and emits a forecast for the
    next ``forecast_days`` trading days with 95% confidence intervals.

    Use this in addition to (not instead of) compute_trend_indicators
    and plot_ohlc_chart. ARIMA gives you a *number* with a CI ("ARIMA
    forecasts +2.3% over 20d, 95% CI [-1.1%, +5.7%]"); the chart and
    indicators give you the visual / level reads. Cite the ARIMA result
    as one quantitative anchor in your KEY DATA section.

    Args:
        ticker: Yahoo-format ticker (e.g. "TATATECH.NS", "AAPL").
        lookback_days: History window for the fit, 90-1825. Default 365.
        forecast_days: Trading days to project forward, 1-60. Default 20.

    Returns:
        JSON of {ticker, status, best_order, best_seasonal_order, aic, bic,
        forecast (list of {date, mean, lower_95, upper_95}), forecast_return_pct,
        forecast_return_lower_pct, forecast_return_upper_pct, signal, summary}.
        ``signal`` is "bullish" (CI lower bound > 0), "bearish" (CI upper < 0),
        or "neutral" (CI straddles zero — the most common case).
    """
    try:
        from findata.arima_forecast import fit_arima_forecast as _fit
        return json.dumps(
            _fit(ticker, lookback_days=lookback_days, forecast_days=forecast_days),
            default=str,
        )
    except Exception as exc:
        logging.exception("arima_forecast failed ticker=%r", ticker)
        return json.dumps({
            "ticker": ticker,
            "status": "wrapper_error",
            "error": f"{type(exc).__name__}: {exc}",
            "signal": "neutral",
            "summary": f"ARIMA unavailable for {ticker}: {exc}",
        })


@function_tool
async def plot_ohlc_chart(
    ticker: str,
    lookback_days: int = 252,
    with_sr: bool = True,
    with_indicators: bool = True,
) -> str:
    """Render an OHLC candlestick chart with optional S/R + indicator overlays.

    Returns a paste-ready markdown image snippet that, when included
    inline in your response text, renders the chart in the user-facing
    debate transcript. Standard pattern:

        ![AAPL 252d](data:image/png;base64,...)

    The chart includes:
      * candlesticks for the lookback window
      * 50-day + 200-day SMA overlay (when with_indicators)
      * RSI(14) subplot below (when with_indicators)
      * up to 5 algorithmic support/resistance horizontal lines
        (when with_sr) — green = support, red = resistance

    Use this once per analyst turn to anchor the technical narrative
    visually. Don't call repeatedly with the same args — generates the
    same chart.

    Args:
        ticker: Yahoo Finance ticker symbol.
        lookback_days: Window in days, 30-1825. Default 252.
        with_sr: Overlay support/resistance lines. Default True.
        with_indicators: Overlay 50/200 SMA + RSI subplot. Default True.

    Returns:
        JSON of {ticker, title, summary, markdown_image, image_base64,
        params}. Embed `markdown_image` directly in your response text
        for inline rendering.
    """
    try:
        from findata.ohlc_chart import plot_ohlc_chart as _plot
        result = _plot(
            ticker,
            lookback_days=lookback_days,
            with_sr=with_sr,
            with_indicators=with_indicators,
        )
        # We send the full base64 down to the agent so it can paste the
        # markdown_image into its response. The base64 is large (~50KB)
        # but the model gets only the markdown_image string in its
        # output — the image bytes flow through to the user-facing
        # transcript via the rendered markdown.
        return json.dumps(result, default=str)
    except Exception as exc:
        logging.exception("plot_ohlc_chart failed ticker=%r", ticker)
        msg = f"{type(exc).__name__}: {exc}"
        # Synthesise a paste-ready italic fallback so the agent can still
        # follow the "paste markdown_image verbatim" instruction without
        # improvising an apologetic generic-error sentence.
        return json.dumps({
            "ticker": ticker,
            "error": msg,
            "image_base64": "",
            "markdown_image": f"*Chart unavailable for {ticker}: {msg}*",
            "chart_status": "wrapper_error",
        })


# Public surface — keep this list explicit so it's obvious which tools
# debate agents pick up when they import from this module.
__all__ = [
    # News / sentiment
    "fetch_yfinance_news",
    "fetch_gdelt_news",
    # Fundamentals + analyst grounding
    "fetch_equity_fundamentals",
    "fetch_analyst_consensus",
    "fetch_earnings_calendar",
    "fetch_returns_stats",
    # Technical analysis (NEW)
    "detect_candlestick_patterns",
    "compute_support_resistance",
    "compute_trend_indicators",
    "arima_forecast",
    "compute_trend_regime",
    "plot_ohlc_chart",
]
