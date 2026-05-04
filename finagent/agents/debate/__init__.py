"""Debate agents — bull / bear / moderator.

Multi-agent setup that takes a ticker, gathers context (prices, news,
fundamentals), and runs a structured debate between two analysts. A
moderator synthesises the transcript into a buy/sell/avoid verdict
with target, stoploss, time horizon, and key metrics.

Reuses the existing OpenAI Agents SDK + data-mcp pattern from the chat
workflow. Adds a small set of news / web-search function tools so the
analysts can ground their arguments in live information.
"""

from .agents import bull_agent, bear_agent, moderator_agent
from .tools import fetch_gdelt_news, fetch_yfinance_news

__all__ = [
    "bull_agent",
    "bear_agent",
    "moderator_agent",
    "fetch_gdelt_news",
    "fetch_yfinance_news",
]
