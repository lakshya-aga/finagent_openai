"""Function tools for debate agents.

Two tools that aren't covered by the existing data-mcp discovery layer:
  * fetch_gdelt_news  — recent GDELT articles by ticker / company name
  * fetch_yfinance_news — Yahoo Finance news headlines for a US ticker

WebSearchTool (the OpenAI hosted tool) covers the long tail of "what's
the latest" questions, but these two are predictably structured and
cheap, so we want the agents to default to them before paying for a
broader web search.

All tools accept the agent's current notion of "now" via context — the
orchestrator stamps the run start time into each agent's instructions
so dates in queries are always grounded.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx
from agents import function_tool


_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@function_tool
async def fetch_gdelt_news(
    query: str,
    days: int = 7,
    max_records: int = 25,
) -> str:
    """Fetch recent news articles matching a query from GDELT.

    GDELT is a global news monitor with broad coverage and no API key
    required. Returns a JSON list of articles with title, url, source,
    publication date, sentiment tone, and a one-line snippet.

    Use this when you want a wide trawl of the public-news view on a
    company or theme — works well for both bull and bear research
    because the result mixes sentiment.

    Args:
        query: GDELT query string (e.g. company name, ticker, theme).
        days: Lookback window in days. Capped at 30; default 7.
        max_records: Max articles to return. Default 25, max 75.

    Returns:
        JSON string of {articles: [...], total: int, query: str}.
    """
    days = max(1, min(30, int(days)))
    max_records = max(1, min(75, int(max_records)))
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "JSON",
        "timespan": f"{days}d",
        "maxrecords": str(max_records),
        "sort": "DateDesc",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(_GDELT_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logging.exception("fetch_gdelt_news failed query=%r", query)
        return json.dumps({"error": str(exc), "query": query, "articles": []})

    articles = []
    for a in (data.get("articles") or [])[:max_records]:
        articles.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "source": a.get("sourcecountry") or a.get("domain"),
            "domain": a.get("domain"),
            "published": a.get("seendate"),
            "tone": a.get("tone"),
            "language": a.get("language"),
        })
    return json.dumps({"query": query, "total": len(articles), "articles": articles}, default=str)


@function_tool
async def fetch_yfinance_news(ticker: str, max_records: int = 15) -> str:
    """Fetch recent news headlines for a US-listed ticker via yfinance.

    Returns a JSON list of {title, publisher, link, published, summary}.
    Use for company-specific news where Yahoo Finance has decent
    coverage (US equities). For crypto / non-US equities, use
    fetch_gdelt_news or web_search instead.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. AAPL, MSFT).
        max_records: Max articles to return. Default 15, max 30.

    Returns:
        JSON string of {ticker, articles: [...]}.
    """
    max_records = max(1, min(30, int(max_records)))
    try:
        # yfinance is sync; run on a worker thread.
        def _fetch():
            import yfinance as yf
            t = yf.Ticker(ticker)
            try:
                items = t.news or []
            except Exception:
                items = []
            return items
        items = await asyncio.to_thread(_fetch)
    except Exception as exc:
        logging.exception("fetch_yfinance_news failed ticker=%r", ticker)
        return json.dumps({"error": str(exc), "ticker": ticker, "articles": []})

    out = []
    for it in items[:max_records]:
        # yfinance shape varies by version; defensively pull each field.
        out.append({
            "title": it.get("title") or it.get("content", {}).get("title"),
            "publisher": it.get("publisher") or it.get("content", {}).get("provider", {}).get("displayName"),
            "link": it.get("link") or it.get("content", {}).get("canonicalUrl", {}).get("url"),
            "published": it.get("providerPublishTime") or it.get("content", {}).get("pubDate"),
            "summary": (it.get("content") or {}).get("summary") or it.get("summary"),
        })
    return json.dumps({"ticker": ticker, "articles": out}, default=str)
