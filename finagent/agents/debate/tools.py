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


# Public surface — keep this list explicit so it's obvious which tools
# debate agents pick up when they import from this module.
__all__ = [
    "fetch_yfinance_news",
    "fetch_gdelt_news",
    "fetch_equity_fundamentals",
    "fetch_analyst_consensus",
    "fetch_earnings_calendar",
    "fetch_returns_stats",
]
