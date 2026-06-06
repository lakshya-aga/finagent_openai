"""Shared model registry and LangChain chat-model factory.

This is the single model-selection surface for FinAgent.

The registry intentionally distinguishes between:

* logical roles (``chat_planner``, ``bias_auditor``, ``panel_pm``),
* provider/model specs (``openai:gpt-5-mini``, ``anthropic:claude-...``),
* provider capabilities (tool calling, structured output, streaming), and
* legacy OpenAI-client access for code that is still on the OpenAI Agents SDK.

True model agnosticism does not mean pretending every model behaves the same.
It means call sites ask for the capability they need and receive a compatible
LangChain ``BaseChatModel`` without hard-coding a vendor SDK.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class ModelCapabilities:
    """Provider/model behavior the orchestration layer can route on."""

    tool_calling: bool = True
    structured_output: bool = True
    streaming: bool = True
    hosted_file_search: bool = False
    hosted_web_search: bool = False
    local_runtime: bool = False
    notes: str = ""


@dataclass(frozen=True)
class RoleConfig:
    """Resolved configuration for one logical LLM role."""

    role: str
    provider: str
    model: str
    capabilities: ModelCapabilities

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"


_DEFAULTS: dict[str, tuple[str, str]] = {
    # Chat workflow.
    "intent_classifier": ("openai", "gpt-5-mini"),
    "chat_planner": ("openai", "gpt-5"),
    "chat_orchestrator": ("openai", "gpt-5"),
    "chat_validator": ("openai", "gpt-5"),
    "chat_question": ("openai", "gpt-5-mini"),
    "chat_edit_planner": ("openai", "gpt-5"),
    "chat_edit_orchestrator": ("openai", "gpt-5"),
    # Recipe / template authoring.
    "template_author": ("openai", "gpt-5"),
    # Audit + verdict layers.
    "bias_auditor": ("openai", "gpt-5-mini"),
    # Paper-trading daily analyst rescue path.
    "stock_analyst": ("openai", "gpt-5-mini"),
    # Notebook-name suggester.
    "name_suggester": ("openai", "gpt-5-mini"),
    # Legacy debate roles.
    "debate_bull": ("openai", "gpt-5-mini"),
    "debate_bear": ("openai", "gpt-5-mini"),
    "debate_moderator": ("openai", "gpt-5-mini"),
    # LangGraph trading panel roles.
    "panel_analyst": ("openai", "gpt-5-mini"),
    "panel_researcher": ("openai", "gpt-5-mini"),
    "panel_research_manager": ("openai", "gpt-5-mini"),
    "panel_trader": ("openai", "gpt-5-mini"),
    "panel_risk": ("openai", "gpt-5-mini"),
    "panel_pm": ("openai", "gpt-5-mini"),
    # Generic fallback.
    "default": ("openai", "gpt-5-mini"),
}


_PROVIDER_CAPABILITIES: dict[str, ModelCapabilities] = {
    "openai": ModelCapabilities(
        hosted_file_search=True,
        hosted_web_search=True,
        notes="OpenAI chat models plus optional hosted tools.",
    ),
    "anthropic": ModelCapabilities(
        hosted_file_search=False,
        hosted_web_search=False,
        notes="Claude through LangChain. Hosted OpenAI tools are unavailable.",
    ),
    "google": ModelCapabilities(
        hosted_file_search=False,
        hosted_web_search=False,
        notes="Gemini through LangChain. Install langchain-google-genai.",
    ),
    "gemini": ModelCapabilities(
        hosted_file_search=False,
        hosted_web_search=False,
        notes="Alias for google.",
    ),
    "ollama": ModelCapabilities(
        tool_calling=True,
        structured_output=False,
        streaming=True,
        local_runtime=True,
        notes=(
            "Local Ollama runtime. Structured output depends on the model; "
            "call sites should tolerate parser fallback."
        ),
    ),
}


def parse_model_spec(spec: str, *, default_provider: str = "openai") -> tuple[str, str]:
    """Parse ``provider:model`` while preserving model names containing colons.

    Plain model names are accepted for backwards compatibility and resolve to
    ``default_provider``. This keeps existing env vars such as
    ``OPENAI_MODEL=gpt-5-mini`` working.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("model spec cannot be empty")
    if ":" not in spec:
        return default_provider.strip().lower(), spec
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(
            f"model spec {spec!r} must be 'provider:model' or a plain model name"
        )
    return provider, model


def _role_env_key(role: str) -> str:
    return role.upper().replace("-", "_")


def _env_model_override(role: str, default_provider: str) -> tuple[str, str] | None:
    role_key = _role_env_key(role)
    provider = os.getenv(f"{role_key}_PROVIDER")
    model = os.getenv(f"{role_key}_MODEL")
    if model:
        return parse_model_spec(model, default_provider=provider or default_provider)
    if provider:
        _, default_model = _DEFAULTS.get(role, _DEFAULTS["default"])
        return provider.strip().lower(), default_model

    # Trading-panel compatibility: historically these used PANEL_* env names.
    if role.startswith("panel_"):
        suffix = role.removeprefix("panel_").upper()
        panel_model = os.getenv(f"PANEL_{suffix}_MODEL")
        if panel_model:
            return parse_model_spec(panel_model, default_provider=default_provider)
        panel_default = os.getenv("PANEL_DEFAULT_MODEL")
        if panel_default:
            return parse_model_spec(panel_default, default_provider=default_provider)

    global_spec = os.getenv("FINAGENT_DEFAULT_MODEL")
    if global_spec:
        return parse_model_spec(global_spec, default_provider=default_provider)

    # Backwards compatibility with old OpenAI-only deployments.
    openai_model = os.getenv("OPENAI_MODEL")
    if openai_model:
        return "openai", openai_model

    return None


def get_role(role: str) -> RoleConfig:
    """Resolve a role to provider/model/capability metadata."""
    default_provider, default_model = _DEFAULTS.get(role, _DEFAULTS["default"])
    override = _env_model_override(role, default_provider)
    if override is None:
        provider, model = default_provider, default_model
    else:
        provider, model = override
    provider = provider.strip().lower()
    caps = _PROVIDER_CAPABILITIES.get(provider)
    if caps is None:
        raise ValueError(
            f"Unsupported provider {provider!r} for role {role!r}. "
            f"Supported providers: {sorted(_PROVIDER_CAPABILITIES)}"
        )
    if provider == "gemini":
        provider = "google"
    return RoleConfig(role=role, provider=provider, model=model, capabilities=caps)


def get_role_config(role: str) -> tuple[str, str]:
    """Backwards-compatible tuple surface used by existing diagnostics."""
    cfg = get_role(role)
    return cfg.provider, cfg.model


def get_model_name(role: str) -> str:
    """Backwards-compatible model-name helper.

    Legacy OpenAI Agents SDK call sites can only consume the model string. New
    provider-neutral code should prefer ``role_spec`` or ``make_chat_for_role``.
    """
    return get_role(role).model


def role_spec(role: str) -> str:
    return get_role(role).spec


def list_roles() -> dict[str, tuple[str, str]]:
    return {role: get_role_config(role) for role in _DEFAULTS if role != "default"}


def list_role_configs() -> dict[str, dict[str, Any]]:
    """Diagnostics-friendly snapshot including provider capabilities."""
    out: dict[str, dict[str, Any]] = {}
    for role in _DEFAULTS:
        if role == "default":
            continue
        cfg = get_role(role)
        out[role] = {
            "provider": cfg.provider,
            "model": cfg.model,
            "spec": cfg.spec,
            "capabilities": cfg.capabilities.__dict__,
        }
    return out


def _make_openai(model: str, **kw: Any):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, **kw)


def _make_anthropic(model: str, **kw: Any):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, **kw)


def _make_google(model: str, **kw: Any):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, **kw)


def _make_ollama(model: str, **kw: Any):
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, base_url=base_url, **kw)


_PROVIDER_FACTORIES = {
    "openai": _make_openai,
    "anthropic": _make_anthropic,
    "google": _make_google,
    "gemini": _make_google,
    "ollama": _make_ollama,
}


def make_chat(spec: str, **kw: Any):
    """Construct a LangChain chat model from a provider/model spec."""
    provider, model = parse_model_spec(spec)
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise ValueError(
            f"unknown provider {provider!r}. Available: {sorted(_PROVIDER_FACTORIES)}"
        )

    is_gpt5 = provider == "openai" and model.lower().startswith("gpt-5")
    if not is_gpt5:
        kw.setdefault("temperature", 0.2)
    kw.setdefault("timeout", 120)
    kw.setdefault("max_retries", 2)
    return factory(model, **kw)


def make_chat_for_role(role: str, **kw: Any):
    return make_chat(role_spec(role), **kw)


async def ainvoke_text(
    role: str,
    *,
    system: str,
    user: str,
    **model_kwargs: Any,
) -> str:
    """Provider-neutral one-shot text call."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = make_chat_for_role(role, **model_kwargs)
    result = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(result, "content", result)
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content or "")


async def ainvoke_structured(
    role: str,
    schema: type[SchemaT],
    *,
    system: str,
    user: str,
    **model_kwargs: Any,
) -> SchemaT:
    """Provider-neutral one-shot structured-output call."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = make_chat_for_role(role, **model_kwargs)
    structured = llm.with_structured_output(schema)
    result = await structured.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        return schema.model_validate(result)
    return schema.model_validate_json(str(result))


def get_llm_client(role: str) -> Any:
    """Compatibility adapter for legacy OpenAI-client call sites.

    New code should not use this. It intentionally refuses non-OpenAI roles so
    provider lock-in is visible instead of silently misrouting to OpenAI.
    """
    cfg = get_role(role)
    if cfg.provider != "openai":
        raise RuntimeError(
            f"Role {role!r} resolves to {cfg.spec!r}, but this legacy path "
            "requires an OpenAI-compatible raw client. Migrate the call site "
            "to make_chat_for_role()/ainvoke_* or set the role provider to openai."
        )
    from openai import AsyncOpenAI

    return AsyncOpenAI()
