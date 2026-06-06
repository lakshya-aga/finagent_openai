"""LangChain-native tools for FinAgent workflow execution.

The legacy notebook agents used OpenAI Agents SDK ``FunctionTool`` objects.
This module exposes the same operational surface as ordinary LangChain tools
so the LangGraph workflow can run with OpenAI, Anthropic, Gemini, Ollama, or
any other provider supported by the shared model registry.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from .functions.cell_tools import (
    add_cell_impl,
    create_notebook_impl,
    delete_cell_impl,
    insert_cell_impl,
    replace_cell_impl,
)
from .functions.notebook_tools import (
    find_regex_in_notebook_code_impl,
    read_notebook_impl,
    validate_run_impl,
)
from .functions.packages import install_packages_impl
from .mcp_connections import make_data_mcp, make_fruit_thrower

logger = logging.getLogger(__name__)


def _json_result(value: Any) -> str:
    """Serialize tool output compactly for model-readable ToolMessages."""
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)


async def _to_thread(func, *args, **kwargs) -> str:
    return _json_result(await asyncio.to_thread(func, *args, **kwargs))


async def _create_notebook_tool() -> str:
    return await _to_thread(create_notebook_impl)


async def _add_cell_tool(
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
) -> str:
    return await _to_thread(add_cell_impl, cell_type, content, dag_node_id, rationale)


async def _replace_cell_tool(
    cell_index: int,
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
) -> str:
    return await _to_thread(
        replace_cell_impl,
        cell_index,
        cell_type,
        content,
        dag_node_id,
        rationale,
    )


async def _insert_cell_tool(
    cell_index: int,
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
) -> str:
    return await _to_thread(
        insert_cell_impl,
        cell_index,
        cell_type,
        content,
        dag_node_id,
        rationale,
    )


async def _delete_cell_tool(cell_index: int) -> str:
    return await _to_thread(delete_cell_impl, cell_index)


async def _read_notebook_tool() -> str:
    return await _to_thread(read_notebook_impl)


async def _find_regex_tool(regex_pattern: str, case_sensitive: bool = False) -> str:
    return await _to_thread(
        find_regex_in_notebook_code_impl,
        regex_pattern,
        case_sensitive,
    )


async def _validate_run_tool(
    max_cells: int = 9999,
    timeout: int = 120,
    prelude: str = "",
) -> str:
    return await _to_thread(validate_run_impl, max_cells, timeout, prelude)


async def _install_packages_tool(packages: list[str]) -> str:
    return await _to_thread(install_packages_impl, packages)


def notebook_build_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            coroutine=_create_notebook_tool,
            name="create_notebook",
            description=(
                "Create an empty notebook at the next available output path. "
                "Call this before adding cells for a new notebook."
            ),
        ),
        StructuredTool.from_function(
            coroutine=_add_cell_tool,
            name="add_cell",
            description=(
                "Append a markdown or code cell to the current notebook. "
                "For code cells, pass dag_node_id and rationale for lineage."
            ),
        ),
    ]


def notebook_edit_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            coroutine=_read_notebook_tool,
            name="read_notebook",
            description="Read all cells, outputs, and FinAgent provenance metadata.",
        ),
        StructuredTool.from_function(
            coroutine=_replace_cell_tool,
            name="replace_cell",
            description="Replace an existing notebook cell while preserving provenance.",
        ),
        StructuredTool.from_function(
            coroutine=_insert_cell_tool,
            name="insert_cell",
            description="Insert a markdown or code cell at a specific index.",
        ),
        StructuredTool.from_function(
            coroutine=_delete_cell_tool,
            name="delete_cell",
            description="Delete a notebook cell at a specific index.",
        ),
        StructuredTool.from_function(
            coroutine=_add_cell_tool,
            name="add_cell",
            description="Append a markdown or code cell to the notebook.",
        ),
    ]


def notebook_validation_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            coroutine=_read_notebook_tool,
            name="read_notebook",
            description="Read the full current notebook before validating or repairing.",
        ),
        StructuredTool.from_function(
            coroutine=_validate_run_tool,
            name="validate_run",
            description=(
                "Run notebook code cells in a persistent kernel, write outputs, "
                "and return the first error if any."
            ),
        ),
        StructuredTool.from_function(
            coroutine=_install_packages_tool,
            name="install_packages",
            description=(
                "Install Python packages into the current environment when "
                "validation proves a non-protected package is missing."
            ),
        ),
        StructuredTool.from_function(
            coroutine=_replace_cell_tool,
            name="replace_cell",
            description="Replace a broken notebook cell with a corrected version.",
        ),
        StructuredTool.from_function(
            coroutine=_find_regex_tool,
            name="find_regex_in_notebook_code",
            description="Search code cells with a regex and return matching snippets.",
        ),
    ]


def _json_schema_type_to_py(schema: dict[str, Any]) -> Any:
    typ = schema.get("type")
    if typ == "integer":
        return int
    if typ == "number":
        return float
    if typ == "boolean":
        return bool
    if typ == "array":
        return list[Any]
    if typ == "object":
        return dict[str, Any]
    return str


def _args_model(tool_name: str, schema: dict[str, Any]):
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        py_type = _json_schema_type_to_py(prop)
        default: Any = ...
        if name not in required:
            default = prop.get("default", None)
        description = prop.get("description")
        fields[name] = (py_type, Field(default, description=description))
    return create_model(f"{tool_name}_args", **fields)


class LangChainMCPToolContext:
    """Connect MCP servers and expose their tools as LangChain StructuredTools."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._servers = []
        self.tools: list[StructuredTool] = []

    async def __aenter__(self) -> "LangChainMCPToolContext":
        try:
            for server in (make_fruit_thrower(), make_data_mcp()):
                await server.connect()
                self._servers.append(server)
                self._stack.push_async_callback(server.cleanup)
                for mcp_tool in await server.list_tools():
                    self.tools.append(self._make_tool(server, mcp_tool))
        except Exception:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._stack.aclose()

    def _make_tool(self, server, mcp_tool) -> StructuredTool:
        tool_name = mcp_tool.name
        args_schema = _args_model(tool_name, mcp_tool.inputSchema or {})

        async def _call(**kwargs) -> str:
            logger.info("LANGCHAIN MCP TOOL CALL: %s args=%s", tool_name, kwargs)
            result = await server.call_tool(tool_name, kwargs or None)
            return _json_result(result)

        return StructuredTool.from_function(
            coroutine=_call,
            name=tool_name,
            description=mcp_tool.description or f"Call MCP tool {tool_name}.",
            args_schema=args_schema,
        )
