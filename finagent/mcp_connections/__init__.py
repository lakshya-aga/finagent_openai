import logging

from .servers import (
    FRUIT_THROWER_URL,
    DATA_MCP_URL,
    FRUIT_THROWER_TOOLS,
    DATA_MCP_TOOLS,
    make_fruit_thrower,
    make_data_mcp,
    make_file_search,
    file_search,
)

_logger = logging.getLogger(__name__)


# ── Knowledge-MCP stubs ────────────────────────────────────────────────
#
# `finagent.workflow` imports `optional_knowledge_mcp` and references
# `make_knowledge_mcp(...)` at several call sites. The knowledge-MCP
# server has not been implemented yet, so these stubs return None.
# The `mcp_servers()` helper below filters out None values so
# MCPServerManager never receives a None entry.
#
# Replace these stubs with a real factory when the knowledge-MCP
# server lands.

def make_knowledge_mcp():
    """Stub: no-op until the real knowledge-MCP server lands."""
    _logger.debug("knowledge-MCP not configured — stub returning None")
    return None


def optional_knowledge_mcp():
    """Stub: returns None until the real knowledge-MCP server lands."""
    return None


def mcp_servers():
    """Build the standard MCP server list, filtering out None entries.

    Use this instead of manually assembling
    ``[make_fruit_thrower(), make_data_mcp(), make_knowledge_mcp()]``
    so that unavailable optional servers (like knowledge-MCP) are
    silently skipped rather than passed as None to MCPServerManager.
    """
    servers = [
        make_fruit_thrower(),
        make_data_mcp(),
        make_knowledge_mcp(),
    ]
    return [s for s in servers if s is not None]
