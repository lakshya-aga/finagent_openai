"""LLM-judge bias audit producer.

Every completed run is post-processed by this module: an OpenAI structured-
output call inspects the executed notebook + recipe + harvested metrics and
emits a verdict (PASS / FLAGGED / FAILED) with a per-check reason list. The
audit runs *after* the run is reported as ``completed`` to the user — see
``finagent/recipe_workflow.py`` — so the user-visible critical path is not
slowed by the extra round-trip.

Cost profile: one ``gpt-4o-mini`` call per completed run, capped at 2000
output tokens. The prompt is ~600 tokens of system instructions plus a
truncated notebook digest (code cells + first 20 lines of stdout per output)
so we stay well inside the model's context window even on long notebooks.

The audit looks at *intent*, not raw output dumps — that's why we strip
images / large dataframes from outputs before sending. Catching bias
patterns is mostly about reading the code; the metrics dict is included
to flag implausible Sharpes and tiny OOS samples.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


# ── Schema ─────────────────────────────────────────────────────────────


class AuditReason(BaseModel):
    check_name: str = Field(
        description=(
            "Stable identifier for the check, e.g. 'lookahead_bias', "
            "'train_test_leakage', 'in_sample_evaluation', 'sample_size', "
            "'implausible_metrics', 'survivorship_bias', 'selection_bias'."
        )
    )
    severity: Literal["info", "warning", "critical"]
    evidence: str = Field(
        description=(
            "1-3 sentence explanation citing specific cells, variable names, "
            "or metric values that motivated this finding."
        )
    )


class BiasAuditVerdict(BaseModel):
    verdict: Literal["PASS", "FLAGGED", "FAILED"]
    summary: str = Field(description="One paragraph plain-English overall assessment.")
    reasons: list[AuditReason] = Field(default_factory=list)


# JSON schema fed to OpenAI structured-output. Keep in sync with the
# Pydantic models above; we hand-roll it because OpenAI's `response_format`
# requires `additionalProperties: false` everywhere and an explicit
# `required` list for every object — both stricter than Pydantic's default
# `model_json_schema()` output.
_VERDICT_JSON_SCHEMA = {
    "name": "BiasAuditVerdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "summary", "reasons"],
        "properties": {
            "verdict": {"type": "string", "enum": ["PASS", "FLAGGED", "FAILED"]},
            "summary": {"type": "string"},
            "reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["check_name", "severity", "evidence"],
                    "properties": {
                        "check_name": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "critical"],
                        },
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ── Prompt ─────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a senior quantitative researcher auditing a backtest \
notebook for methodological flaws. You are skeptical by default — your job is \
to catch the kinds of mistakes that make a strategy look great in-sample and \
fall apart in production.

Run through this checklist on every notebook:

1. Look-ahead bias: features computed using values from time t+k where k>=0 \
   (e.g. `.shift(-1)`, using `df['close']` as both feature and target without \
   a lag, normalising by full-period statistics).
2. Train/test leakage: target derived from the same window used for feature \
   fitting; scalers / PCA / feature selectors fit on the full dataset before \
   the split.
3. In-sample evaluation: Sharpe / accuracy / pnl computed on the training \
   slice rather than held-out data.
4. Sample size adequacy: fewer than ~200 out-of-sample observations is \
   suspect for any claim about a Sharpe.
5. Implausible metrics: Sharpe > 3, annual_return > 100%, max_drawdown < 5% \
   — flag for human review even if you cannot prove the cause.
6. Survivorship bias: universe is "current S&P 500 members", "currently \
   listed", or similar — historical results are inflated by dropped names.
7. Selection bias in metric reporting: only the best fold / parameter / \
   asset is reported; metrics differ wildly across runs but only one is shown.

For each finding emit ONE entry in `reasons` with:
- `check_name`: one of the seven check ids above (snake_case)
- `severity`: `critical` if it invalidates the result, `warning` if it \
  materially weakens it, `info` for best-practice notes
- `evidence`: cite the specific cell, variable, or metric value

Verdict rules:
- PASS: no warnings or criticals
- FLAGGED: at least one warning, no criticals
- FAILED: at least one critical

If the notebook is empty or unreadable, return FLAGGED with a single \
`audit_error` reason. Be concise — the summary is one paragraph, evidence \
is 1-3 sentences."""


# ── Notebook digest ────────────────────────────────────────────────────


_MAX_OUTPUT_LINES = 20
_MAX_CELL_SOURCE_CHARS = 4000


def _digest_notebook(notebook_json: dict) -> str:
    """Compress a notebook JSON down to code + truncated stdout per cell.

    We keep code cells verbatim (capped per cell) and append at most
    `_MAX_OUTPUT_LINES` lines of stream output. Markdown cells are skipped
    — the audit is about code, not narration. Images / dataframe HTML are
    dropped entirely.
    """
    cells = notebook_json.get("cells", []) if isinstance(notebook_json, dict) else []
    chunks: list[str] = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if len(source) > _MAX_CELL_SOURCE_CHARS:
            source = source[:_MAX_CELL_SOURCE_CHARS] + "\n# ...truncated..."
        out_lines: list[str] = []
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") != "stream":
                continue
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            for line in text.splitlines():
                out_lines.append(line)
                if len(out_lines) >= _MAX_OUTPUT_LINES:
                    break
            if len(out_lines) >= _MAX_OUTPUT_LINES:
                break
        chunk = f"# --- Cell {idx} ---\n{source}"
        if out_lines:
            chunk += "\n# stdout:\n" + "\n".join(f"# {l}" for l in out_lines)
        chunks.append(chunk)
    return "\n\n".join(chunks)


# ── LLM call ───────────────────────────────────────────────────────────


_AUDIT_MODEL = "gpt-4o-mini"


async def _call_llm(
    user_payload: str,
    run_id: Optional[str] = None,
) -> BiasAuditVerdict:
    """Hit gpt-4o-mini with structured-output and parse the result.

    Isolated as a thin function so tests can monkey-patch
    `_call_llm` directly and exercise `audit_run`'s error handling without
    needing an API key.
    """
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=_AUDIT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        response_format={"type": "json_schema", "json_schema": _VERDICT_JSON_SCHEMA},
        max_tokens=2000,
        temperature=0,
    )
    # Cost ledger — recorded best-effort; never fails the call.
    try:
        from finagent.cost_tracking import record_cost_event
        record_cost_event(
            response=resp,
            purpose="bias_audit",
            model=_AUDIT_MODEL,
            run_id=run_id,
        )
    except Exception:
        pass
    raw = resp.choices[0].message.content or "{}"
    return BiasAuditVerdict.model_validate_json(raw)


async def audit_run(
    notebook_json: dict,
    recipe_yaml: str,
    metrics: dict[str, Any],
    run_id: Optional[str] = None,
) -> BiasAuditVerdict:
    """Audit a completed run and return a structured verdict.

    Always returns a `BiasAuditVerdict` — on any error (missing API key,
    timeout, malformed JSON), returns FLAGGED with a single `audit_error`
    reason so callers don't have to reason about exceptions.
    """
    try:
        digest = _digest_notebook(notebook_json or {})
        if not digest:
            digest = "# (notebook is empty)"
        try:
            metrics_str = json.dumps(metrics or {}, default=str, indent=2)
        except Exception:
            metrics_str = "{}"
        user_payload = (
            "RECIPE YAML:\n"
            f"{recipe_yaml or '# (no recipe)'}\n\n"
            "METRICS:\n"
            f"{metrics_str}\n\n"
            "NOTEBOOK DIGEST (code cells + first lines of stdout):\n"
            f"{digest}"
        )
        return await _call_llm(user_payload, run_id=run_id)
    except Exception as exc:
        logging.exception("bias audit failed")
        return BiasAuditVerdict(
            verdict="FLAGGED",
            summary=(
                "Audit could not complete; treating run as flagged for human "
                "review until the auditor can be re-run."
            ),
            reasons=[
                AuditReason(
                    check_name="audit_error",
                    severity="warning",
                    evidence=str(exc),
                )
            ],
        )
