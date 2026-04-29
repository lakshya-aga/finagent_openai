"""Search executor.

Drives one Search end-to-end:

  1. policy.propose(history) → mutations dict
  2. apply_mutations(base_recipe_yaml, mutations) → variant YAML
  3. dedup by recipe fingerprint against history; skip if duplicate
  4. recipe_workflow.run_recipe(variant_yaml, search_id, iter)
  5. read metric, update history, update best
  6. stop when budget exhausted, max_no_improvement triggered, or policy
     returns None (grid exhausted)

Designed to run synchronously inside a worker thread — the recipe
runner already does kernel boot + execution; we don't add any extra
asyncio. Background-task wrapping happens at the FastAPI layer.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import yaml

from ..experiments import ExperimentStore, Search, get_store
from ..recipe_workflow import run_recipe
from ..recipes.types import recipe_from_yaml
from .policy import make_policy
from .types import (
    SearchHistoryEntry,
    SearchSubmission,
)


logger = logging.getLogger(__name__)


def execute_search(
    submission: SearchSubmission,
    *,
    store: Optional[ExperimentStore] = None,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Run a search to completion. Returns the final search row + history."""
    store = store or get_store()
    base_recipe = recipe_from_yaml(submission.base_recipe_yaml)
    project = base_recipe.project
    search_name = name or f"{base_recipe.name}::{submission.policy}"

    search = store.create_search(
        project=project,
        name=search_name,
        submission_json=submission.model_dump_json(),
        policy=submission.policy,
        objective_json=submission.objective.model_dump_json(),
    )
    store.update_search(search.id, status="running")
    logger.info("search %s started: project=%s, policy=%s, budget=%s",
                search.id, project, submission.policy, submission.budget.max_runs)

    policy = make_policy(submission.policy, submission.space, submission.seed)
    history: list[SearchHistoryEntry] = []
    seen_fingerprints: set[str] = set()
    best_metric: Optional[float] = None
    best_run_id: Optional[str] = None
    no_improve_streak = 0
    started_at = time.time()
    completed_iters = 0
    error_message: Optional[str] = None

    try:
        while True:
            # ── Budget gates ──────────────────────────────────────────
            if completed_iters >= submission.budget.max_runs:
                break
            if (
                submission.budget.max_seconds is not None
                and time.time() - started_at > submission.budget.max_seconds
            ):
                logger.info("search %s hit max_seconds budget", search.id)
                break
            if (
                submission.budget.max_no_improvement is not None
                and no_improve_streak >= submission.budget.max_no_improvement
            ):
                logger.info("search %s early-stopped: %d no-improvement runs",
                            search.id, no_improve_streak)
                break

            # ── Propose, dedup, retry ─────────────────────────────────
            mutations = _propose_unique(
                policy, history, submission, base_recipe, seen_fingerprints,
                max_retries=20,
            )
            if mutations is None:
                # Policy exhausted (grid done, or random hit too many duplicates)
                logger.info("search %s policy exhausted at iter=%d",
                            search.id, completed_iters)
                break

            variant_yaml, variant_recipe = _apply_mutations(
                submission.base_recipe_yaml, mutations,
                naming_iter=completed_iters,
            )
            seen_fingerprints.add(variant_recipe.fingerprint())

            # ── Run the variant ──────────────────────────────────────
            run_result = run_recipe(
                recipe_yaml=variant_yaml,
                search_id=search.id,
                search_iteration=completed_iters,
            )
            run_id = run_result.get("run_id", "")
            run_status = run_result.get("status", "failed")
            metric_val = _read_metric(run_result, submission.objective.metric)

            entry = SearchHistoryEntry(
                iteration=completed_iters,
                run_id=run_id,
                recipe_fingerprint=variant_recipe.fingerprint(),
                mutations=mutations,
                metric=metric_val,
                status=run_status,
                error=run_result.get("error"),
            )
            history.append(entry)

            improved = submission.objective.is_better(metric_val, best_metric)
            if improved:
                best_metric = metric_val
                best_run_id = run_id
                no_improve_streak = 0
                logger.info("search %s iter %d → new best %s=%.6f",
                            search.id, completed_iters,
                            submission.objective.metric, metric_val)
            else:
                no_improve_streak += 1

            completed_iters += 1
            store.update_search(
                search.id,
                iterations=completed_iters,
                best_run_id=best_run_id,
                best_metric=best_metric,
            )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("search %s failed", search.id)

    final_status = "failed" if error_message else "completed"
    store.update_search(
        search.id,
        status=final_status,
        iterations=completed_iters,
        best_run_id=best_run_id,
        best_metric=best_metric,
        error=error_message,
        finished=True,
    )

    return {
        "search_id": search.id,
        "status": final_status,
        "iterations": completed_iters,
        "best_run_id": best_run_id,
        "best_metric": best_metric,
        "history": [h.model_dump() for h in history],
        "error": error_message,
    }


# ── helpers ─────────────────────────────────────────────────────────────


def _propose_unique(
    policy,
    history,
    submission,
    base_recipe,
    seen: set[str],
    *,
    max_retries: int,
):
    """Repeatedly ask the policy for a proposal until we get one whose
    rendered recipe fingerprint we haven't seen, or we exhaust retries.
    Stops gracefully when the policy returns None (grid done)."""
    for _ in range(max_retries + 1):
        mutations = policy.propose(history)
        if mutations is None:
            return None
        try:
            _, recipe_obj = _apply_mutations(
                submission.base_recipe_yaml, mutations, naming_iter=0,
            )
        except Exception as exc:
            logger.warning("mutation rejected: %s", exc)
            continue
        fp = recipe_obj.fingerprint()
        if fp not in seen:
            return mutations
    return None


def _apply_mutations(
    base_yaml: str,
    mutations: dict[str, Any],
    *,
    naming_iter: int,
):
    """Apply path-based mutations to the base recipe and re-validate.

    Field paths use dot-notation; list indices are not supported (a sweep
    over `features[2].params.window` would need a richer DSL). For 'features'
    as a whole we accept a list value; otherwise the path is dotted scalar.
    """
    data = yaml.safe_load(base_yaml) or {}
    for path, value in mutations.items():
        _set_path(data, path, value)
    # Tag the variant name so notebooks land with distinct files.
    data["name"] = f"{data.get('name', 'variant')}__iter{naming_iter:03d}"
    new_yaml = yaml.safe_dump(data, sort_keys=False)
    recipe = recipe_from_yaml(new_yaml)
    return new_yaml, recipe


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _read_metric(run_result: dict[str, Any], metric: str) -> Optional[float]:
    metrics = run_result.get("metrics") or {}
    val = metrics.get(metric)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
