"""LLM-judge bias audit producer.

A tiny one-shot LLM call that audits a completed run for the common
research-bias mistakes a senior quant would catch on a code-review:

  * lookahead bias (`.shift(-N)`, future-dated joins, normalising on
    full-period statistics)
  * train/test leakage (fitting scalers/PCA on combined train+test,
    row-shuffled CV on time-series)
  * in-sample evaluation (reporting Sharpe on the training slice)
  * sample-size adequacy (Sharpe on 6 months of daily data is noise)
  * metric pickiness / p-hacking (tuning a threshold then reporting the
    best, hidden ``for x in ...`` then ``max()`` patterns)
  * survivorship bias (universe is "currently listed" only)

Modeled on ``finagent/agents/name_suggester.py`` — the cleanest "tiny
one-shot LLM call with Pydantic structured output" pattern in the repo.
The agents SDK is lazy-imported so this module stays loadable even in
environments where the SDK isn't installed (CI, alt runtimes); on any
failure (missing API key, network error, malformed response) we return
``BiasAudit(verdict="PENDING", ...)`` so the run is never blocked and
the frontend can render a grey "pending" pill.

Wired post-completion by ``finagent.recipe_workflow._spawn_bias_audit``
— the audit runs as a background task so the user-visible ``completed``
status fires the moment the kernel finishes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ── Schema ─────────────────────────────────────────────────────────────


class BiasReason(BaseModel):
    """One finding from the audit checklist. Multiple per audit are fine."""

    check_name: str = Field(
        description=(
            "One of: lookahead, train_test_leakage, in_sample_eval, "
            "sample_size, metric_pickiness, survivorship, other"
        )
    )
    severity: str = Field(
        description="One of: info, warn, fail",
    )
    evidence: str = Field(
        description=(
            "<=200 chars; cite the specific cell/metric/code that triggered "
            "the flag (e.g. 'cell 7: df[\"y\"].shift(-1)', "
            "'metrics.sharpe=4.2 implausible', 'no train/test split visible')."
        ),
    )


class BiasAudit(BaseModel):
    """Structured verdict the UI can render directly."""

    verdict: str = Field(
        description="One of: PASS, FLAGGED, FAILED, PENDING",
    )
    reasons: list[BiasReason] = Field(
        default_factory=list,
        description="Empty for clean PASS",
    )
    summary: str = Field(
        description="<=300 chars; plain-English overall assessment",
    )


# ── Prompt ─────────────────────────────────────────────────────────────


_BIAS_AUDITOR_INSTRUCTIONS = """You are a senior quant reviewing a trading-research notebook for common biases.
Read the cells, recipe, and computed metrics. Score each of these checks (skip
any that don't apply):

1. Lookahead: does any code use `.shift(-N)`, future-dated joins, or compute
   features after the label window? Normalising by full-period statistics
   counts here too.
2. Train/test leakage: are train and test sets disjoint in time? Any `fit()`
   on combined train+test? Cross-validation that splits by row instead of by
   time?
3. In-sample evaluation: are reported metrics (Sharpe, accuracy, PnL) computed
   on the *same* data the model was fit on?
4. Sample size: are reported metrics from <100 trades, <2 years of daily data,
   or <30 events? Sharpe on 6 months of data is unreliable.
5. Metric pickiness / p-hacking: are multiple thresholds / parameters tested
   with only the best reported? Any `for threshold in ...` followed by
   selecting the max? Any sweep that reports just the winner?
6. Survivorship: does the universe contain only currently-listed tickers (no
   delistings)?

For each finding, emit ONE entry in `reasons`:
  - `check_name`: one of {lookahead, train_test_leakage, in_sample_eval,
    sample_size, metric_pickiness, survivorship, other}
  - `severity`:
      * `info`  — best-practice nudge, doesn't invalidate the result
      * `warn`  — materially weakens the result, human should review
      * `fail`  — invalidates the result entirely
  - `evidence`: <=200 chars; cite a specific cell index, metric name, or code
    snippet so the user can jump straight to it

Verdict rules:
  * PASS    — no `warn` or `fail` reasons (info-only is still PASS)
  * FLAGGED — at least one `warn`, no `fail`
  * FAILED  — at least one `fail`

`summary` is <=300 chars of plain-English overall assessment for the badge
tooltip. If the notebook is empty or unreadable, return FLAGGED with a single
`other` reason explaining why. Never emit PENDING — that's reserved for the
caller's error path.
"""


# ── Notebook digest ────────────────────────────────────────────────────


# Keep input under the model's effective context. 30K chars ≈ 7-8K tokens for
# code, well within gpt-4o-mini's 128K window after the system prompt + the
# recipe + metrics serialisation.
_MAX_PAYLOAD_CHARS = 30_000


def _digest_notebook(notebook_json: dict) -> list[dict[str, str]]:
    """Strip a notebook down to ``[{cell_type, source}, ...]``.

    We deliberately drop ``outputs`` entirely — base64 PNG charts and large
    dataframe HTML reps would blow the context budget for zero added signal
    on the bias question, which is mostly about reading the code. Markdown
    cells are kept (they often narrate the methodology and let the model
    catch "we report the best fold" style claims).
    """
    if not isinstance(notebook_json, dict):
        return []
    cells = notebook_json.get("cells")
    if not isinstance(cells, list):
        return []
    digest: list[dict[str, str]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in ("code", "markdown"):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str):
            source = str(source)
        digest.append({"cell_type": cell_type, "source": source})
    return digest


def _build_user_payload(
    notebook_json: dict,
    recipe: dict | None,
    metrics: dict | None,
    template_name: str | None,
) -> str:
    """Pack the audit input into a single user message, capped at 30K chars.

    Order matters: recipe + metrics + template come first because they're
    the cheapest signals (always small, always relevant). Notebook cells
    come last so truncation, when it bites, drops the lowest-signal tail
    of the notebook rather than the recipe header.
    """
    import json as _json

    try:
        recipe_str = _json.dumps(recipe or {}, default=str, indent=2)
    except Exception:
        recipe_str = "{}"
    try:
        metrics_str = _json.dumps(metrics or {}, default=str, indent=2)
    except Exception:
        metrics_str = "{}"
    template_str = template_name or "(none)"

    header = (
        f"TEMPLATE: {template_str}\n\n"
        f"RECIPE:\n{recipe_str}\n\n"
        f"METRICS:\n{metrics_str}\n\n"
        f"NOTEBOOK CELLS (cell_type + source only — outputs stripped):\n"
    )

    cells = _digest_notebook(notebook_json)
    cell_chunks: list[str] = []
    used = len(header)
    truncated_at: Optional[int] = None
    for idx, cell in enumerate(cells):
        chunk = f"# --- Cell {idx} ({cell['cell_type']}) ---\n{cell['source']}\n"
        if used + len(chunk) > _MAX_PAYLOAD_CHARS:
            truncated_at = idx
            break
        cell_chunks.append(chunk)
        used += len(chunk)

    body = "\n".join(cell_chunks) if cell_chunks else "(no cells)"
    if truncated_at is not None:
        body += (
            f"\n\n# ... {len(cells) - truncated_at} more cells truncated to "
            f"fit context ..."
        )

    return header + body


# ── Public API ─────────────────────────────────────────────────────────


async def audit_run(
    *,
    notebook_json: dict,
    recipe: dict | None,
    metrics: dict | None,
    template_name: str | None,
) -> BiasAudit:
    """Audit a completed run and return a structured ``BiasAudit``.

    Always returns a ``BiasAudit`` — on any error (missing API key, missing
    SDK, network timeout, malformed response) returns
    ``BiasAudit(verdict="PENDING", reasons=[], summary="auditor unavailable: <reason>")``
    so the run completion path is never blocked.
    """
    # Lazy imports — keep this module loadable in environments where the
    # Agents SDK isn't installed (tests, alternative runtimes, ops scripts
    # that just want to deserialize a stored audit).
    try:
        from agents import Agent, Runner, ModelSettings
    except ImportError as exc:
        logger.warning("bias_auditor: agents SDK unavailable (%s)", exc)
        return BiasAudit(
            verdict="PENDING",
            reasons=[],
            summary=f"auditor unavailable: agents SDK not installed ({exc})",
        )

    try:
        from finagent.llm import get_model_name
        model_name = get_model_name("bias_auditor")
    except Exception as exc:
        logger.warning("bias_auditor: model resolution failed, falling back (%s)", exc)
        model_name = "gpt-4o-mini"

    try:
        user_payload = _build_user_payload(
            notebook_json or {}, recipe, metrics, template_name,
        )
    except Exception as exc:
        logger.exception("bias_auditor: payload build failed")
        return BiasAudit(
            verdict="PENDING",
            reasons=[],
            summary=f"auditor unavailable: payload build failed ({exc})",
        )

    try:
        agent = Agent(
            name="BiasAuditor",
            instructions=_BIAS_AUDITOR_INSTRUCTIONS,
            model=model_name,
            model_settings=ModelSettings(),
            output_type=BiasAudit,
        )
        result = await Runner.run(
            agent,
            input=user_payload,
            max_turns=1,
        )
        verdict = result.final_output_as(BiasAudit)
    except Exception as exc:
        logger.warning("bias_auditor: LLM call failed (%s)", exc)
        return BiasAudit(
            verdict="PENDING",
            reasons=[],
            summary=f"auditor unavailable: {exc}",
        )

    # Normalise the verdict in case the model emits a near-miss like "Pass".
    # Defence-in-depth: even with structured output, a stale model can drift
    # off-grid — recompute deterministically from the severities so the badge
    # never goes blank.
    v = (verdict.verdict or "").strip().upper()
    if v not in {"PASS", "FLAGGED", "FAILED"}:
        sevs = {(r.severity or "").strip().lower() for r in verdict.reasons}
        if "fail" in sevs:
            v = "FAILED"
        elif "warn" in sevs:
            v = "FLAGGED"
        else:
            v = "PASS"
    verdict.verdict = v
    return verdict
