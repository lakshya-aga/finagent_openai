"""Adapters for provider-hosted tools used by legacy agent paths."""

from __future__ import annotations

import logging
from typing import Any

from finagent.llm import get_role

logger = logging.getLogger(__name__)


def hosted_web_search_tools(role: str) -> list[Any]:
    """Return hosted web-search tools if the role's provider supports them.

    OpenAI Agents SDK exposes ``WebSearchTool`` as a provider-hosted tool.
    Anthropic/Gemini/Ollama do not expose that exact object through this SDK,
    so the neutral behavior is to return no hosted search tool and let the
    workflow rely on explicit data/news tools instead.
    """
    cfg = get_role(role)
    if not cfg.capabilities.hosted_web_search:
        logger.debug(
            "hosted web search unavailable for role=%s provider=%s",
            role,
            cfg.provider,
        )
        return []
    from agents import WebSearchTool

    return [WebSearchTool()]
