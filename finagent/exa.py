"""Minimal Exa search client for the forecaster's evidence gathering.

Deliberately a raw httpx call rather than the exa-py SDK — one POST,
no new dependency in requirements.txt (httpx ships with the openai
package). Defensive parsing: Exa's response fields shift between
"auto"/"neural" modes, so every field access tolerates absence.

Env:
  EXA_API_KEY  — required; raises ExaNotConfigured with a clear
                 message when missing so the endpoint can surface
                 "operator needs to add the key" instead of a 500.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_EXA_SEARCH_URL = "https://api.exa.ai/search"

# Cap per-result text so 5 results stay ~3-4k tokens total in the
# research agent's context instead of five full articles.
_MAX_TEXT_CHARS = 1500


class ExaNotConfigured(RuntimeError):
    pass


def exa_available() -> bool:
    return bool(os.environ.get("EXA_API_KEY", "").strip())


def search(
    query: str,
    *,
    num_results: int = 5,
    recent_only: bool = False,
) -> list[dict[str, Any]]:
    """One Exa search → [{title, url, published_date, snippet}].

    ``recent_only`` constrains to the last 30 days — the forecaster
    uses it for "latest developments" queries while leaving base-rate
    queries unconstrained. Synchronous (callers run it in a thread via
    the agent tool layer); 20s timeout; never returns more than 8.
    """
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        raise ExaNotConfigured(
            "EXA_API_KEY is not set — add it to the deployment secrets "
            "to enable event forecasting."
        )

    import httpx

    body: dict[str, Any] = {
        "query": query,
        "type": "auto",
        "numResults": max(1, min(int(num_results), 8)),
        "contents": {"text": {"maxCharacters": _MAX_TEXT_CHARS}},
    }
    if recent_only:
        from datetime import datetime, timedelta, timezone

        body["startPublishedDate"] = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    resp = httpx.post(
        _EXA_SEARCH_URL,
        json=body,
        headers={"x-api-key": key, "content-type": "application/json"},
        timeout=20.0,
    )
    resp.raise_for_status()
    payload = resp.json()

    out: list[dict[str, Any]] = []
    for r in payload.get("results") or []:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "title": (r.get("title") or "").strip()[:300],
                "url": r.get("url") or "",
                "published_date": r.get("publishedDate") or "",
                "snippet": (r.get("text") or "").strip()[:_MAX_TEXT_CHARS],
            }
        )
    logger.info("exa.search(%r) → %d results", query[:80], len(out))
    return out
