"""Backwards-compatible shim.

The agent workflow has been split into ``finagent/`` (functions, agents,
mcp_connections, workflow). This module re-exports the public surface
(``run_workflow`` and ``WorkflowInput``) so existing callers like ``app.py``
continue to work without changes.
"""

from finagent.workflow import WorkflowInput, run_workflow

__all__ = ["WorkflowInput", "run_workflow"]
