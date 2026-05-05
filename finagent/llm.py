"""Model-swap dispatcher.

Every LLM call in the codebase routes through this module. The goal is
two-fold:

1. **Per-role model control** — each agent has a logical role
   (``planner``, ``orchestrator``, ``bias_auditor``, ``debate``, …).
   This module returns the right model name + provider for that role,
   read from environment overrides with sensible defaults.

2. **Future provider swap** — today the only supported provider is
   ``openai`` because the OpenAI Agents SDK requires it. After the
   LangGraph migration (Phase L3+) the same role keys can resolve to
   Anthropic / Google / local Ollama clients without touching the
   agent code.

Configuration:
  Set ``<ROLE>_PROVIDER`` and ``<ROLE>_MODEL`` to override per role.
  Examples:
    PLANNER_MODEL=gpt-5
    BIAS_AUDITOR_MODEL=gpt-4o-mini
    DEBATE_PROVIDER=anthropic         # post-L3 only
    DEBATE_MODEL=claude-3-5-sonnet     # post-L3 only

  Falls back to ``OPENAI_MODEL`` (a global default) and finally to the
  hardcoded default in ``_DEFAULTS`` below.
"""

from __future__ import annotations

import os
from typing import Any


# ─── Role registry ──────────────────────────────────────────────────
# (provider, model). Add a new agent role by appending here.
# Provider is informational today (only "openai" is wired); becomes
# load-bearing in L3 once we switch agent runtimes.

_DEFAULTS: dict[str, tuple[str, str]] = {
    # Chat workflow
    "intent_classifier": ("openai", "gpt-4o-mini"),
    "chat_planner":      ("openai", "gpt-5"),
    "chat_orchestrator": ("openai", "gpt-5"),
    "chat_validator":    ("openai", "gpt-5"),
    "chat_question":     ("openai", "gpt-4o"),
    "chat_edit_planner":      ("openai", "gpt-5"),
    "chat_edit_orchestrator": ("openai", "gpt-5"),
    # Recipe / template authoring
    "template_author":   ("openai", "gpt-5"),
    # Audit + verdict layers
    "bias_auditor":      ("openai", "gpt-4o-mini"),
    # Debate package
    "debate_bull":       ("openai", "gpt-4o"),
    "debate_bear":       ("openai", "gpt-4o"),
    "debate_moderator":  ("openai", "gpt-4o"),
    # Generic fallback for un-keyed callers
    "default":           ("openai", "gpt-4o"),
}


def get_role_config(role: str) -> tuple[str, str]:
    """Resolve (provider, model) for a role from env then defaults.

    Lookup order:
      1. ``<ROLE>_PROVIDER`` / ``<ROLE>_MODEL`` (per-role override)
      2. ``OPENAI_MODEL`` (global default — back-compat with existing
         deploys that pre-date this dispatcher)
      3. ``_DEFAULTS[role]``
      4. ``_DEFAULTS["default"]``
    """
    role_key = role.upper().replace("-", "_")
    default_provider, default_model = _DEFAULTS.get(role, _DEFAULTS["default"])
    provider = os.getenv(f"{role_key}_PROVIDER", default_provider)
    model = (
        os.getenv(f"{role_key}_MODEL")
        or os.getenv("OPENAI_MODEL")
        or default_model
    )
    return provider, model


def get_model_name(role: str) -> str:
    """Convenience wrapper — just the model string for a role."""
    return get_role_config(role)[1]


def get_llm_client(role: str) -> Any:
    """Return an async client for direct, single-call LLM use.

    For agents that are still on the OpenAI Agents SDK (everything
    except the post-L3 migrated set), call this when you need the raw
    chat-completions client — e.g. the intent classifier, the bias
    auditor, the SSE chat helper.

    Today: returns ``openai.AsyncOpenAI``. Adding Anthropic / Gemini
    / local-OpenAI-compatible just means another branch here.
    """
    provider, _ = get_role_config(role)
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI()
    if provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "ANTHROPIC_PROVIDER requires `pip install anthropic`"
            ) from exc
        return AsyncAnthropic()
    raise ValueError(
        f"Unsupported provider {provider!r} for role {role!r}. "
        f"Set <ROLE>_PROVIDER to one of: openai, anthropic."
    )


def list_roles() -> dict[str, tuple[str, str]]:
    """Snapshot of every role's resolved (provider, model). Useful for
    /admin/diagnostics and similar surfaces that want to render the
    current config to a UI.
    """
    return {role: get_role_config(role) for role in _DEFAULTS if role != "default"}
