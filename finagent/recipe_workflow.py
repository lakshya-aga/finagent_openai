"""Run a Recipe end-to-end: compile, build notebook, execute, harvest metrics.

Flow:
  1. Compile recipe → list of CellSpec.
  2. Materialise a fresh notebook on disk with those cells (provenance
     metadata stamped from CellSpec.dag_node_id / rationale).
  3. Run all cells with the existing kernel runner (continues past errors).
  4. Tail the run's stdout for ``FINAGENT_RUN_SUMMARY {...}`` to extract
     metrics; persist them on the experiment Run.
  5. Stash both AST + runtime lineage on the notebook so the Project page
     and Graph view work the same as for AI-generated runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .experiments import get_store
from .functions.notebook_io import _OUTPUTS_DIR, _get_latest_path, _path_for_recipe
from .functions.notebook_tools import run_all_cells_to_disk
from .lineage import extract_lineage_ast
from .recipes.compiler import compile_recipe
from .recipes.types import Recipe, recipe_from_yaml


_SUMMARY_RE = re.compile(r"^FINAGENT_RUN_SUMMARY\s+(\{.*\})\s*$", re.MULTILINE)


def run_recipe(
    *,
    recipe_yaml: str,
    progress_cb=None,
    search_id: Optional[str] = None,
    search_iteration: Optional[int] = None,
) -> dict:
    """Compile + execute a recipe; return run metadata.

    When called from the search executor, ``search_id`` and
    ``search_iteration`` thread through to the run record so the project
    page can group + order runs by their parent search.
    """
    store = get_store()
    recipe = recipe_from_yaml(recipe_yaml)

    run = store.create_run(
        project=recipe.project,
        name=recipe.name,
        template=recipe.template,
        recipe_yaml=recipe_yaml,
        recipe_hash=recipe.fingerprint(),
        search_id=search_id,
        search_iteration=search_iteration,
    )
    logging.info("recipe run created id=%s project=%s name=%s",
                 run.id, recipe.project, recipe.name)

    try:
        cells = compile_recipe(recipe)
        if cells is None:
            raise ValueError(
                f"recipe has no template; cannot compile deterministically. "
                f"Set `template:` to one of: regime_modeling"
            )

        notebook_path = _materialise_notebook(recipe, cells)
        store.update_run(run.id, status="running",
                         notebook_path=str(notebook_path))

        result = run_all_cells_to_disk(str(notebook_path), timeout=180)

        # Harvest metrics from the run-summary marker if any cell emitted it.
        metrics = _extract_metrics_from_notebook(notebook_path)

        # Stash AST lineage immediately; runtime lineage is the same path the
        # Graph viewer will fetch on demand.
        try:
            lineage = extract_lineage_ast(str(notebook_path))
            _stash_lineage_on_notebook(notebook_path, "ast", lineage)
        except Exception:
            logging.exception("AST lineage failed for recipe %s", run.id)

        status = "completed" if not result.get("errors") else "failed"
        error = (
            "; ".join(
                f"cell {e['cell_index']}: {e.get('ename')}: {e.get('evalue')}"
                for e in result.get("errors", [])
            )
            if result.get("errors") else None
        )
        store.update_run(
            run.id,
            status=status,
            metrics=metrics,
            error=error,
            finished=True,
        )

        # Pre-registered hypothesis verdict — synchronous, deterministic,
        # cheap. Just compares the actual metrics against the success /
        # cancel criteria pinned in the recipe YAML. We do this inline
        # (not async) because the verdict is part of the run's terminal
        # state from the user's POV. Skipped on failed runs (no metrics
        # to evaluate against) and on recipes with no hypothesis block.
        if status == "completed" and recipe.hypothesis is not None:
            try:
                verdict = _evaluate_hypothesis(recipe.hypothesis, metrics)
                store.update_run_hypothesis_verdict(run.id, json.dumps(verdict))
            except Exception:
                logging.exception("hypothesis evaluation failed run_id=%s", run.id)

        # Fire-and-forget bias audit. Failed runs are skipped — there is no
        # methodology to evaluate when the kernel crashed before producing
        # results. We deliberately do NOT await this: the user-visible run
        # finishes the moment status flips to "completed"; the audit
        # populates `bias_audit_json` whenever it lands. The frontend
        # exposes that as `bias_audit: null` until the verdict arrives.
        if status == "completed":
            _spawn_bias_audit(run.id, notebook_path, recipe_yaml, metrics)

        return {
            "run_id": run.id,
            "status": status,
            "notebook_path": str(notebook_path),
            "metrics": metrics,
            "errors": result.get("errors", []),
        }
    except Exception as exc:
        logging.exception("recipe run failed id=%s", run.id)
        store.update_run(run.id, status="failed", error=str(exc), finished=True)
        return {"run_id": run.id, "status": "failed", "error": str(exc)}


# ── helpers ─────────────────────────────────────────────────────────────


# Libraries whose versions are stamped into every recipe notebook. These
# match the pinned scientific stack the orchestrator advertises to the
# planner (see finagent/agents/orchestration.py). A missing import means
# the package isn't installed in this environment — we surface that as
# ``null`` rather than crashing, so the metadata stamping never fails a
# run.
_PINNED_LIBS = (
    "pandas", "numpy", "scipy", "scikit-learn", "statsmodels",
    "hmmlearn", "xgboost", "matplotlib", "yfinance", "findata",
    "openai", "fastapi",
)


def _capture_library_versions() -> dict[str, str | None]:
    """Snapshot the exact versions of every pinned library at compile time."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        return {}
    out: dict[str, str | None] = {}
    for lib in _PINNED_LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = None
        except Exception:
            out[lib] = None
    return out


def _materialise_notebook(recipe: Recipe, cells) -> Path:
    """Build a fresh notebook on disk from a CellSpec list.

    File path is derived from the recipe identity (name + fingerprint +
    UTC minute timestamp) so the Notebooks list is human-searchable. The
    legacy `notebook_N.ipynb` naming is reserved for free-form chat-agent
    notebooks that don't carry a recipe identity.
    """
    try:
        fp = recipe.fingerprint()
    except Exception:
        fp = None
    path = _path_for_recipe(recipe.name, fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(cells=[])
    nb.metadata["kernelspec"] = {
        "display_name": "FinAgent Python",
        "language": "python",
        "name": "finagent-python",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    nb.metadata["finagent_recipe"] = {
        "name": recipe.name,
        "project": recipe.project,
        "template": recipe.template,
        "fingerprint": recipe.fingerprint(),
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        # Reproducibility fingerprint (UAT P2 #1): seed + pinned-library
        # versions + data vintage. Sufficient for a researcher to
        # reconstruct the exact environment that produced this notebook.
        "seed": recipe.seed,
        "library_versions": _capture_library_versions(),
        "data_vintage": {
            var: {
                "kind": ds.kind,
                "start": getattr(ds, "start", None),
                "end": getattr(ds, "end", None),
            }
            for var, ds in recipe.data.items()
        },
    }

    for spec in cells:
        if spec.cell_type == "markdown":
            cell = new_markdown_cell(spec.content)
        else:
            cell = new_code_cell(spec.content)
        if spec.dag_node_id or spec.rationale:
            md = cell.metadata.setdefault("finagent", {})
            if spec.dag_node_id:
                md["node_id"] = spec.dag_node_id
            if spec.rationale:
                md["rationale"] = spec.rationale
        nb.cells.append(cell)

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return path


def _extract_metrics_from_notebook(path: Path) -> dict[str, float]:
    """Tail the saved notebook for the FINAGENT_RUN_SUMMARY stream output."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception:
        return {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") != "stream":
                continue
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            m = _SUMMARY_RE.search(text)
            if m:
                try:
                    summary = json.loads(m.group(1))
                except Exception:
                    return {}
                return {k: float(v) for k, v in (summary.get("metrics") or {}).items()
                        if isinstance(v, (int, float))}
    return {}


def _stash_lineage_on_notebook(path: Path, method: str, lineage: dict) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bucket = nb.metadata.setdefault("finagent_lineage", {})
        bucket[method] = lineage
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except Exception:
        logging.exception("could not stash lineage")


# ── hypothesis verdict evaluator ────────────────────────────────────────────


def _check_one(actual: float | None, op: str, target: float) -> bool:
    """Apply a comparison op. None / NaN actuals always fail."""
    if actual is None:
        return False
    try:
        a = float(actual)
    except (TypeError, ValueError):
        return False
    if a != a:  # NaN
        return False
    if op == ">=": return a >= target
    if op == ">":  return a >  target
    if op == "<=": return a <= target
    if op == "<":  return a <  target
    if op == "==": return a == target
    if op == "!=": return a != target
    return False


def _evaluate_hypothesis(hypothesis, metrics: dict) -> dict:
    """Compare the run's actual metrics against the pre-registered
    success_criteria + cancel_criteria. Returns a structured verdict dict
    suitable for serialization onto the run row.

    Decision rule:
      * If ANY cancel_criterion holds → verdict = "CANCEL".
      * Else if ALL success_criteria hold → "PASS".
      * Else → "FAIL".

    Empty success_criteria + at-least-one cancel_criterion is supported
    (nothing-but-cancel-bands is a valid spec for "this idea has no
    explicit success threshold but I want to know when it's clearly
    broken"). The Pydantic validator already requires at least one
    criterion across both arrays.
    """
    checks: list[dict] = []
    cancel_hit = False
    for c in hypothesis.cancel_criteria:
        actual = metrics.get(c.metric)
        passed = _check_one(actual, c.op, c.value)
        if passed:
            cancel_hit = True
        checks.append({
            "criterion": c.model_dump(),
            "actual": actual,
            "passed": passed,
            "kind": "cancel",
        })

    success_all = bool(hypothesis.success_criteria)
    for c in hypothesis.success_criteria:
        actual = metrics.get(c.metric)
        passed = _check_one(actual, c.op, c.value)
        if not passed:
            success_all = False
        checks.append({
            "criterion": c.model_dump(),
            "actual": actual,
            "passed": passed,
            "kind": "success",
        })

    if cancel_hit:
        verdict = "CANCEL"
        summary = "Hypothesis cancelled — at least one cancel-criterion was hit."
    elif success_all:
        verdict = "PASS"
        summary = "Hypothesis confirmed — every pre-registered success criterion held."
    else:
        verdict = "FAIL"
        summary = "Hypothesis not confirmed — at least one success criterion did not hold."

    return {
        "verdict": verdict,
        "summary": summary,
        "thesis": hypothesis.thesis,
        "checks": checks,
    }


# ── bias audit hook ─────────────────────────────────────────────────────────


def _spawn_bias_audit(
    run_id: str,
    notebook_path: Path,
    recipe_yaml: str,
    metrics: dict,
) -> None:
    """Kick the bias audit off the run's critical path.

    Two execution contexts call ``run_recipe``:
      • the ASGI app worker, which already has a running event loop —
        ``asyncio.create_task`` is the right primitive there;
      • the synchronous search executor / CLI, which has no loop —
        we spin up a daemon thread that owns its own event loop so
        the audit can still complete without blocking the caller.

    Either way, every failure is swallowed and logged: a misbehaving
    auditor must never break a successful run.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop.create_task(_run_bias_audit(run_id, notebook_path, recipe_yaml, metrics))
        return

    def _thread_target() -> None:
        try:
            asyncio.run(
                _run_bias_audit(run_id, notebook_path, recipe_yaml, metrics)
            )
        except Exception:
            logging.exception("bias audit thread crashed run_id=%s", run_id)

    threading.Thread(
        target=_thread_target,
        name=f"bias-audit-{run_id}",
        daemon=True,
    ).start()


async def _run_bias_audit(
    run_id: str,
    notebook_path: Path,
    recipe_yaml: str,
    metrics: dict,
) -> None:
    """Read the executed notebook, call the auditor, persist the verdict."""
    try:
        from .agents.bias_auditor import audit_run  # local import: optional dep

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook_json = json.load(f)
        except Exception:
            logging.exception("could not read notebook for audit run_id=%s", run_id)
            notebook_json = {"cells": []}

        verdict = await audit_run(notebook_json, recipe_yaml, metrics)
        get_store().update_run_bias_audit(
            run_id, json.dumps(verdict.model_dump())
        )
        logging.info(
            "bias audit complete run_id=%s verdict=%s", run_id, verdict.verdict
        )
    except Exception:
        logging.exception("bias audit failed run_id=%s", run_id)
