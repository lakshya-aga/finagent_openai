"""MCP server smoke tests.

Two MCP servers are wired up by ``finagent.mcp_connections.servers``:

  * ``fruit_thrower`` (HTTP streamable, code-RAG over fin-kit)
  * ``data_mcp``     (SSE, doc index for findata wrappers)

These tests don't run the full agent stack — they just open the MCP
connection, list available tools, and call one no-side-effect tool to
prove the round-trip works. Skipped via ``needs_mcp`` when the
servers aren't reachable on the configured URLs.
"""

from __future__ import annotations

import asyncio
import os

import pytest


@pytest.mark.needs_mcp
def test_data_mcp_lists_tools_and_runs_search():
    """Connect, list tools, and run the ``search_tools`` query.
    ``search_tools`` is read-only; safe to call in CI."""
    pytest.importorskip("agents")
    from finagent.mcp_connections.servers import make_data_mcp

    async def run():
        server = make_data_mcp()
        await server.connect()
        try:
            tools = await server.list_tools()
            assert tools, "data_mcp returned zero tools"
            tool_names = {t.name for t in tools}
            # Sanity — at least one of the expected tools is exposed.
            expected = {"search_tools", "get_tool_doc", "list_all_tools"}
            assert tool_names & expected, (
                f"data_mcp missing expected tools, got {sorted(tool_names)}"
            )
            # Run a real query.
            if "search_tools" in tool_names:
                result = await server.call_tool(
                    "search_tools", {"query": "candlestick"}
                )
                assert result is not None
        finally:
            await server.cleanup()

    asyncio.run(run())


@pytest.mark.needs_mcp
def test_fruit_thrower_lists_tools_and_runs_search():
    """Connect, list tools, run ``search_code``. Read-only."""
    pytest.importorskip("agents")
    from finagent.mcp_connections.servers import make_fruit_thrower

    async def run():
        server = make_fruit_thrower()
        await server.connect()
        try:
            tools = await server.list_tools()
            assert tools, "fruit_thrower returned zero tools"
            names = {t.name for t in tools}
            expected = {"search_code", "list_modules", "get_module_summary"}
            assert names & expected, (
                f"fruit_thrower missing expected tools, got {sorted(names)}"
            )
            if "search_code" in names:
                result = await server.call_tool("search_code", {"query": "sharpe"})
                assert result is not None
        finally:
            await server.cleanup()

    asyncio.run(run())


def test_mcp_factory_imports_clean():
    """Even when the MCP servers aren't reachable, the factory module
    must import without raising — the agents' module-level Agent(...)
    construction depends on the placeholder server instances. This
    is the test that catches a bad refactor of servers.py before any
    agent ever starts."""
    pytest.importorskip("agents")
    from finagent.mcp_connections import servers
    assert hasattr(servers, "make_data_mcp")
    assert hasattr(servers, "make_fruit_thrower")
    assert hasattr(servers, "DATA_MCP_TOOLS")
    assert hasattr(servers, "FRUIT_THROWER_TOOLS")
    assert "search_tools" in servers.DATA_MCP_TOOLS
    assert "search_code" in servers.FRUIT_THROWER_TOOLS
