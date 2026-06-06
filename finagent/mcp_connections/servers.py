"""MCP endpoints + factories.

Two MCP servers feed the agents:
  - fruit_thrower (HTTP streamable): code-RAG over the internal fin-kit library.
  - data_mcp (SSE): documentation index for data-fetch wrappers.

Both URLs are env-overridable so the same code runs on a laptop and inside
docker-compose. Each agent that needs MCP access takes a *fresh* server
instance — sharing one across agents creates connection-pool collisions, so
the public API here is the `make_*` factories. The module-level `mcp_*`
placeholders exist only to satisfy `Agent(...)` at definition time; at
runtime the workflow swaps them out via `agent.clone(mcp_servers=...)`.
"""

from __future__ import annotations

import os
from agents import function_tool
from agents.mcp import (
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
)

FRUIT_THROWER_URL = os.environ.get("FRUIT_THROWER_URL", "http://localhost:8090/mcp/")
DATA_MCP_URL = os.environ.get("DATA_MCP_URL", "http://localhost:8000/sse")

FRUIT_THROWER_TOOLS = [
    "search_code",
    "get_unit_source",
    "list_modules",
    "get_module_summary",
    "index_repository",
    "get_index_stats",
    "generate_function",
]
DATA_MCP_TOOLS = [
    "search_tools",
    "get_tool_doc",
    "list_all_tools",
    "request_data_source",
]


def make_fruit_thrower() -> MCPServerStreamableHttp:
    return MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(url=FRUIT_THROWER_URL),
        name="fruit_thrower",
        tool_filter={"allowed_tool_names": FRUIT_THROWER_TOOLS},
        require_approval="never",
    )


def make_data_mcp() -> MCPServerSse:
    return MCPServerSse(
        params=MCPServerSseParams(url=DATA_MCP_URL),
        name="data_mcp",
        tool_filter={"allowed_tool_names": DATA_MCP_TOOLS},
        require_approval="never",
    )


def file_search_tools() -> list:
    """Return hosted file-search tools for legacy Agents SDK agents.

    Non-OpenAI knowledge backends can return an empty list, which keeps agent
    construction clean while making the missing hosted capability explicit.
    """
    from finagent.retrieval import hosted_file_search_tools

    return hosted_file_search_tools()


@function_tool
def unavailable_file_search(query: str = ""):
    """Explain that hosted file search is unavailable for this knowledge backend."""
    backend = os.environ.get("KNOWLEDGE_STORE_BACKEND", "openai")
    return {
        "success": False,
        "error": (
            "Hosted file search is unavailable for "
            f"KNOWLEDGE_STORE_BACKEND={backend!r}. Use "
            "KNOWLEDGE_STORE_BACKEND=openai with OPENAI_VECTOR_STORE_ID for "
            "OpenAI hosted file search, or migrate this call site to a "
            "provider-neutral retrieval backend."
        ),
        "query": query,
    }


def make_file_search():
    """Backward-compatible single-tool helper.

    Prefer ``file_search_tools()`` in new code because non-hosted knowledge
    stores may expose zero hosted tools. Legacy callers still receive a valid
    tool object so they do not accidentally pass ``None`` into an Agents SDK
    tool list.
    """
    tools = file_search_tools()
    return tools[0] if tools else unavailable_file_search


file_search = make_file_search()
