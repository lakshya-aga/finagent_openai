"""finagent package: agent workflow split into functions, agents, mcp_connections."""

from .workflow import WorkflowInput, run_workflow

__all__ = ["run_workflow", "WorkflowInput"]
