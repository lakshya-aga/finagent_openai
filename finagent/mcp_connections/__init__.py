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


# ── Knowledge-MCP stubs ────────────────────────────────────────────────
#
# `finagent.workflow` (commit c3b17a0, "notebook naming") imports
# `optional_knowledge_mcp` and references `make_knowledge_mcp(...)` at
# several call sites. Neither symbol was ever defined — the
# notebook-naming commit landed half a feature. Without these stubs
# `import finagent.workflow` raises ImportError at module load and
# the test suite + the orchestration agent both fail to start.
#
# These stubs unblock the import. `make_knowledge_mcp()` returns None
# so the call sites that do `MCPServerManager([..., make_knowledge_mcp()])`
# pass a None into the manager — most agent SDK versions tolerate it,
# but a follow-up should filter Nones explicitly. Replace with the
# real factory when the knowledge-MCP server lands.

def make_knowledge_mcp():
    """Stub: no-op until the real knowledge-MCP server lands."""
    return None


def optional_knowledge_mcp():
    """Stub: returns None until the real knowledge-MCP server lands."""
    return None
