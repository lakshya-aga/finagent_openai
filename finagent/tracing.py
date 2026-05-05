"""Phoenix / OpenTelemetry tracing initialisation.

Self-hosted alternative to LangSmith. Phoenix is a Docker container
shipped alongside finagent (see synapse/docker-compose.yml). When the
``PHOENIX_COLLECTOR_ENDPOINT`` env var is set, this module's
``init_tracing()`` registers an OTLP exporter and turns on
auto-instrumentation for the OpenAI client (covers everything that
flows through finagent.llm.get_llm_client) and httpx (covers the
GDELT fetcher + any direct HTTP calls).

The OpenAI Agents SDK's tracer is NOT replaced — it still emits its
own traces to the OpenAI dashboard. After the L3 LangGraph migration
that side becomes irrelevant; for now the two trace surfaces coexist.

Auto-instrumentation is best-effort: missing optional packages are
logged and skipped so a misconfigured deploy never crashes finagent
boot. Re-running ``init_tracing()`` is safe (it short-circuits after
the first successful registration).
"""

from __future__ import annotations

import logging
import os

_REGISTERED = False


def init_tracing() -> bool:
    """Initialise Phoenix tracing. Returns True when active.

    No-ops cleanly when:
      * ``PHOENIX_COLLECTOR_ENDPOINT`` is unset (typical for local dev)
      * ``arize-phoenix-otel`` isn't installed
      * The Phoenix container is unreachable (logged, not raised)
    """
    global _REGISTERED
    if _REGISTERED:
        return True

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        logging.info("phoenix tracing: PHOENIX_COLLECTOR_ENDPOINT unset; skipping")
        return False

    project = os.environ.get("PHOENIX_PROJECT_NAME", "synapse")

    try:
        from phoenix.otel import register
    except ImportError:
        logging.warning(
            "phoenix tracing: arize-phoenix-otel not installed; "
            "spans will not be exported. Install: pip install arize-phoenix-otel"
        )
        return False

    try:
        # register() returns a tracer_provider and globally activates it.
        # auto_instrument=True hooks every supported library on the path
        # (openai, httpx, requests, langchain, llama-index, …) so we get
        # spans for tool calls + LLM calls without per-callsite wrapping.
        register(
            project_name=project,
            endpoint=endpoint,
            auto_instrument=True,
        )
    except Exception:
        logging.exception("phoenix tracing: register() failed; spans will not be exported")
        return False

    logging.info("phoenix tracing active: project=%s endpoint=%s", project, endpoint)
    _REGISTERED = True
    return True
