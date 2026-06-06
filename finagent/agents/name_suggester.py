"""Name-suggester agent.

A tiny one-shot LLM call that turns a user's research request into a
short kebab-case base name for the notebook file. Runs early in the
new-notebook workflow, BEFORE planning, so:

  - The notebook gets a meaningful filename right at creation time
    instead of an opaque ``notebook_5.ipynb``.
  - The frontend can echo the chosen name as the build progresses
    (`📓 cross-sector-momentum-v1__20260510-143015.ipynb`).
  - Future signal-registration logic can use the slug as a stable
    handle for the parquet file + scheduled job.

No tools, no MCPs, no streaming — single ChatCompletion call, fast
(< 1 second) and cheap (~ 0.001 USD on gpt-5-mini). Output is a
strict Pydantic model so we don't have to parse free text.

The datetime suffix that guarantees uniqueness is appended by
``finagent.functions.notebook_io._path_for_named`` — this agent only
produces the base slug.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NotebookName(BaseModel):
    """Structured output for the name suggester."""

    name: str = Field(
        description=(
            "A short kebab-case base name (lowercase, hyphens only, no spaces, "
            "no special chars, no file extension). 3-6 words max, ~20-50 "
            "characters. Should describe the research idea concretely "
            "(e.g. 'cross-sector-momentum-v1', 'btc-vol-targeting-12mo'). "
            "Do not include dates, fingerprints, or counters — those are "
            "appended automatically by the file-path helper."
        ),
    )


_NAME_SUGGESTER_INSTRUCTIONS = """You are a NAME-SUGGESTER for research notebooks.

Read the user's research request and produce ONE short, kebab-case base
name for the notebook file. The name should:

  - Be 3-6 words, lowercase, hyphens between words only (no underscores,
    no spaces, no special characters, no file extension)
  - Describe the research IDEA concretely so an analyst scanning a list
    of notebook filenames can guess what's inside without opening it
  - Avoid generic words ("notebook", "research", "analysis", "v1") unless
    they meaningfully disambiguate (e.g. "v1" is fine if the user said
    "first version")
  - Avoid dates / timestamps — those are appended automatically
  - Avoid fingerprints, hashes, or sequence numbers

Examples:

  Request:  "Build me a momentum signal for the Nifty 50"
  Name:     nifty50-momentum-signal

  Request:  "I want to backtest a pairs trade on SPY/TLT with cointegration"
  Name:     spy-tlt-pairs-cointegration

  Request:  "Test if VIX > 30 is a good entry signal for a long-only S&P book"
  Name:     vix-30-spx-long-entry

  Request:  "Run a regime-switching analysis on Bitcoin returns"
  Name:     bitcoin-regime-switching

Emit ONLY the structured ``NotebookName`` — no commentary, no preamble.
"""


# kebab-case validator — we slugify defensively after the LLM responds in
# case the model emits underscores / capitals despite the prompt.
_KEBAB_CLEAN_RE = re.compile(r"[^a-z0-9-]+")


def _force_kebab(s: str) -> str:
    """Coerce arbitrary string → kebab-case slug (lowercase, hyphens only).

    Defence-in-depth: even a perfectly-prompted LLM occasionally emits
    snake_case or camelCase. We canonicalise so the filename helper
    sees what it expects.
    """
    s = (s or "").strip().lower()
    # Replace runs of underscores / spaces / dots with single hyphens.
    s = re.sub(r"[\s_\.]+", "-", s)
    # Drop anything that isn't lowercase alphanumeric or hyphen.
    s = _KEBAB_CLEAN_RE.sub("", s)
    # Collapse runs of hyphens.
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "notebook"


async def suggest_notebook_name(user_request: str) -> str:
    """Run the name suggester for a single user request and return the
    canonical kebab-case slug. Always returns a non-empty string —
    falls back to ``notebook`` on any LLM failure so the build proceeds
    even if the agent is unavailable."""
    try:
        from finagent.llm import ainvoke_structured

        out = await ainvoke_structured(
            "name_suggester",
            NotebookName,
            system=_NAME_SUGGESTER_INSTRUCTIONS,
            user=f"User request:\n\n{user_request}",
        )
    except Exception as exc:
        logger.warning("name_suggester: LLM call failed, using fallback (%s)", exc)
        return "notebook"

    slug = _force_kebab(out.name)
    if not slug:
        return "notebook"
    return slug

