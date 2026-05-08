"""Provider-agnostic LangChain ChatModel factory.

The whole point of LangChain in this module is **swappable models**.
A role spec like ``"openai:gpt-4o"`` or ``"ollama:qwen2.5:14b"``
returns a ready-to-use BaseChatModel that the rest of the panel can
bind tools to and prompt structured-output schemas against.

Why this lives here and not in finagent.llm:
  - finagent.llm wraps OpenAI's raw async client (OpenAI Agents SDK
    convention). The trading panel uses LangChain's BaseChatModel
    abstraction. Different layer — separate factory.
  - This factory is opt-in (only the panel imports it). The existing
    finagent.llm dispatch keeps powering everything else.
  - Lets the panel run on free local models (Qwen / Llama 3 / DeepSeek
    via Ollama) without forcing the rest of the stack onto LangChain.

Supported providers (each gated on its package being importable so
unused ones don't add deploy weight):

  openai:<model>      — gpt-4o, gpt-4o-mini, gpt-5, ...
  anthropic:<model>   — claude-sonnet-4-5, claude-haiku-4-5, ...
  google:<model>      — gemini-2.0-flash, gemini-2.5-pro, ...
  ollama:<model>      — qwen2.5:14b, llama3.1:8b, deepseek-r1:32b, ...

Default role-mapping mirrors finagent.llm's pattern: each role gets
a default that can be overridden via env (PANEL_<ROLE>_MODEL).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ── Role defaults ───────────────────────────────────────────────────


# Defaults: **gpt-5-mini for every role**. Single-model panel keeps
# cost + observability simple, and gpt-5-mini's structured-output
# adherence is good enough that we don't need the gpt-4o uplift on
# the deep-thinking roles.
#
# Approximate cost per panel run:
#   3 analysts × ~6 tool calls each   ~$0.05
#   macro analyst × ~2 tool calls     ~$0.02
#   bull/bear × 1 round               ~$0.02
#   research_mgr + trader + risk + pm ~$0.08
#   ────────────────────────────────────────
#   ≈ $0.15-0.20 per ticker
#
# Daily Nifty 5-stock cron @ ~$0.75/day ≈ $25/month. Within reason.
#
# To bump deep-thinking roles to full gpt-5 (sharper structured output
# at ~10× the per-token cost) when output quality matters:
#   export PANEL_RESEARCH_MANAGER_MODEL=openai:gpt-5
#   export PANEL_PM_MODEL=openai:gpt-5
#
# To flip the panel onto self-hosted Qwen later:
#   export PANEL_DEFAULT_MODEL=ollama:qwen2.5:14b-instruct
# (and ensure an Ollama daemon is reachable at OLLAMA_BASE_URL).
#
# To swap to Anthropic:
#   export PANEL_DEFAULT_MODEL=anthropic:claude-haiku-4-5
_ROLE_DEFAULTS: dict[str, tuple[str, str]] = {
    "panel_analyst":          ("openai", "gpt-5-mini"),
    "panel_researcher":       ("openai", "gpt-5-mini"),
    "panel_research_manager": ("openai", "gpt-5-mini"),
    "panel_trader":           ("openai", "gpt-5-mini"),
    "panel_risk":             ("openai", "gpt-5-mini"),
    "panel_pm":               ("openai", "gpt-5-mini"),
}


def _env_override(role: str) -> Optional[str]:
    """Look up an env-var override for a role.

    Format: ``PANEL_ANALYST_MODEL=ollama:qwen2.5:14b``
    Returns the raw "provider:model" string if set, else None.
    """
    key = f"PANEL_{role.removeprefix('panel_').upper()}_MODEL"
    val = os.environ.get(key)
    if val:
        return val.strip()
    # Also accept a global override that swaps every role at once.
    return os.environ.get("PANEL_DEFAULT_MODEL")


def parse_spec(spec: str) -> tuple[str, str]:
    """Split a "provider:model" spec.

    Handles the awkward case where the model name itself contains a
    colon (Ollama tags: ``qwen2.5:14b``, ``llama3.1:70b-instruct``).
    The first colon is the provider boundary; everything after is the
    model name.
    """
    spec = (spec or "").strip()
    if ":" not in spec:
        raise ValueError(
            f"model spec {spec!r} must be 'provider:model' "
            f"(e.g. 'openai:gpt-4o' or 'ollama:qwen2.5:14b')"
        )
    provider, _, model = spec.partition(":")
    return provider.strip().lower(), model.strip()


def role_spec(role: str) -> str:
    """Resolve a role to its 'provider:model' spec.

    Order of precedence:
        env override (PANEL_<ROLE>_MODEL or PANEL_DEFAULT_MODEL)
      > _ROLE_DEFAULTS table
      > first entry in _ROLE_DEFAULTS as a last-resort fallback
    """
    override = _env_override(role)
    if override:
        return override
    if role in _ROLE_DEFAULTS:
        provider, model = _ROLE_DEFAULTS[role]
        return f"{provider}:{model}"
    # Unknown role — log + fall back to the analyst default.
    logger.warning("trading_panel.llm_factory: unknown role %r; using analyst default", role)
    provider, model = _ROLE_DEFAULTS["panel_analyst"]
    return f"{provider}:{model}"


# ── Provider-specific constructors ──────────────────────────────────


def _make_openai(model: str, **kw: Any):
    from langchain_openai import ChatOpenAI
    # ChatOpenAI auto-reads OPENAI_API_KEY from env. Pass temperature
    # via kw — we keep it low for managers (deterministic JSON shape)
    # and let the caller override for analysts (a touch of variety
    # gives more nuanced reports).
    return ChatOpenAI(model=model, **kw)


def _make_anthropic(model: str, **kw: Any):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, **kw)


def _make_google(model: str, **kw: Any):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, **kw)


def _make_ollama(model: str, **kw: Any):
    """Local-first: connect to a running Ollama daemon.

    Ollama supports tool-calling for Qwen 2.5+, Llama 3.1+, and
    DeepSeek R1+. Older models won't bind tools cleanly — the panel
    will log a warning if structured output fails and fall back to a
    text-parse path.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        # Fall back to the legacy community package if the dedicated
        # one isn't installed (common in older deploys).
        from langchain_community.chat_models import ChatOllama  # type: ignore
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, base_url=base_url, **kw)


_PROVIDER_FACTORIES = {
    "openai": _make_openai,
    "anthropic": _make_anthropic,
    "google": _make_google,
    "gemini": _make_google,                # alias
    "ollama": _make_ollama,
}


def make_chat(spec: str, **kw: Any):
    """Construct a BaseChatModel from a 'provider:model' spec.

    Extra ``kw`` are forwarded to the provider's constructor. Common
    knobs:
        temperature: float = 0.0..1.0   (default 0.3 in this module)
        timeout: int                    (seconds)
        max_tokens: int

    Raises:
        ImportError if the provider's langchain package isn't
        installed. Surface this early — silent fallback to a wrong
        model would be much worse than a clean import error.
    """
    provider, model = parse_spec(spec)
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise ValueError(
            f"unknown provider {provider!r}. Available: "
            f"{sorted(_PROVIDER_FACTORIES.keys())}"
        )

    # gpt-5 family quirks: temperature is locked to 1.0 server-side
    # (any other value silently destabilises the request — observed as
    # an indefinite hang in the panel before timeouts were added).
    # Skip the default-temperature setdefault for gpt-5*.
    is_gpt5 = provider == "openai" and model.lower().startswith("gpt-5")
    if not is_gpt5:
        # Default temperature: low — every panel role benefits from
        # consistency more than creativity. gpt-5 ignores this.
        kw.setdefault("temperature", 0.3)

    # Default per-request timeout: 120s. Without this, a hung request
    # blocks the panel forever — the python process appears %CPU=0,
    # STATE=sleeping, and the user thinks the panel crashed silently.
    # 120s is generous for a single chat completion; panel has ≥10
    # calls so cumulative budget stays sane.
    kw.setdefault("timeout", 120)
    # Cap retries low so a model-tier issue surfaces fast instead of
    # hiding behind exponential backoff.
    kw.setdefault("max_retries", 2)
    return factory(model, **kw)


def make_chat_for_role(role: str, **kw: Any):
    """Convenience: resolve role → spec → ChatModel in one call."""
    return make_chat(role_spec(role), **kw)


# ── Diagnostics ─────────────────────────────────────────────────────


def list_role_specs() -> dict[str, str]:
    """Return the currently-active role → spec mapping. Useful for the
    /admin/llm-config diagnostics page."""
    return {role: role_spec(role) for role in _ROLE_DEFAULTS}
