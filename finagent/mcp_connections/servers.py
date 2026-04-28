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

from agents import FileSearchTool
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
DATA_MCP_TOOLS = ["search_tools", "get_tool_doc", "list_all_tools", "request_data_source"]


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


# Default vector store for the agent's curated PDFs / research notes.
_VECTOR_STORE_ID = os.environ.get(
    "OPENAI_VECTOR_STORE_ID", "vs_69a81b0197a481919e14c2d66197af7d"
)


def make_file_search() -> FileSearchTool:
    return FileSearchTool(vector_store_ids=[_VECTOR_STORE_ID])


# Single shared FileSearchTool instance — it has no per-call state and is safe
# to share across agents.
file_search = make_file_search()
