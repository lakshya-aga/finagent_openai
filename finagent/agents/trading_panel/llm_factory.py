"""Trading-panel compatibility wrappers around the shared LLM registry."""

from __future__ import annotations

from finagent.llm import (
    list_role_configs,
    make_chat,
    make_chat_for_role,
    parse_model_spec,
    role_spec,
)


def parse_spec(spec: str) -> tuple[str, str]:
    return parse_model_spec(spec)


def list_role_specs() -> dict[str, str]:
    """Return the active role -> provider:model mapping for panel roles."""
    return {
        role: cfg["spec"]
        for role, cfg in list_role_configs().items()
        if role.startswith("panel_")
    }


__all__ = [
    "list_role_specs",
    "make_chat",
    "make_chat_for_role",
    "parse_spec",
    "role_spec",
]
