"""Admin metrics — Tier-1 quality signals computed off the experiment store.

Most metrics are derivable straight from the existing `runs` and `searches`
tables; we don't need a separate counter table for this MVP. Each metric
function returns a single float (or None when there's no data) so the
admin dashboard can render a sparkline + headline number per metric.

The default rolling window is 7 days; callers can override. Adding a new
metric is a matter of writing one function and registering it in
``METRICS`` — the API handler takes care of dispatch and per-day series.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .experiments import ExperimentStore, get_store


@dataclass
class MetricSpec:
    key: str
    label: str
    description: str
    direction: str             # "up_is_good" | "down_is_good"
    fmt: str                   # "percent" | "ratio" | "seconds" | "count"
    fn: Callable[[ExperimentStore, float, float], Optional[float]]


def _runs_in_window(store: ExperimentStore, start_ts: float, end_ts: float):
    """Return only runs that *finished* within the window. Unfinished runs
    don't contribute to quality signals — they're either still running or
    failed during setup, both of which we'd rather wait or surface separately."""
    rows = []
    for r in store.list_runs(limit=2000):
        if r.finished_at is None:
            continue
        if r.finished_at < start_ts or r.finished_at > end_ts:
            continue
        rows.append(r)
    return rows


# ── individual metric functions ─────────────────────────────────────────


def _notebook_pass_rate(store, start, end) -> Optional[float]:
    rows = _runs_in_window(store, start, end)
    if not rows:
        return None
    passed = sum(1 for r in rows if r.status == "completed" and not (r.error or "").strip())
    return passed / len(rows)


def _recipe_success_rate(store, start, end) -> Optional[float]:
    rows = [r for r in _runs_in_window(store, start, end) if r.template]
    if not rows:
        return None
    return sum(1 for r in rows if r.status == "completed") / len(rows)


def _avg_time_to_metric(store, start, end) -> Optional[float]:
    rows = [
        r for r in _runs_in_window(store, start, end)
        if r.status == "completed" and r.metrics()
    ]
    if not rows:
        return None
    durations = [
        (r.finished_at - r.started_at)
        for r in rows
        if r.finished_at is not None and r.started_at is not None
    ]
    if not durations:
        return None
    return sum(durations) / len(durations)


def _failed_run_count(store, start, end) -> Optional[float]:
    rows = _runs_in_window(store, start, end)
    return float(sum(1 for r in rows if r.status == "failed"))


def _total_runs(store, start, end) -> Optional[float]:
    return float(len(_runs_in_window(store, start, end)))


def _provenance_coverage(store, start, end) -> Optional[float]:
    """% of code cells whose finagent.node_id metadata is set, scanned across
    notebooks produced by recipe runs in the window."""
    import nbformat
    from pathlib import Path

    rows = [
        r for r in _runs_in_window(store, start, end)
        if r.notebook_path and Path(r.notebook_path).exists()
    ]
    if not rows:
        return None
    total = 0
    stamped = 0
    for r in rows:
        try:
            with open(r.notebook_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except Exception:
            continue
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            total += 1
            md = cell.metadata.get("finagent") if hasattr(cell, "metadata") else None
            if md and md.get("node_id"):
                stamped += 1
    if total == 0:
        return None
    return stamped / total


def _output_completeness(store, start, end) -> Optional[float]:
    """% of code cells with at least one output (across runs in window)."""
    import nbformat
    from pathlib import Path

    rows = [
        r for r in _runs_in_window(store, start, end)
        if r.notebook_path and Path(r.notebook_path).exists()
    ]
    if not rows:
        return None
    total = 0
    with_outputs = 0
    for r in rows:
        try:
            with open(r.notebook_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except Exception:
            continue
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            total += 1
            if cell.get("outputs"):
                with_outputs += 1
    if total == 0:
        return None
    return with_outputs / total


def _reproducibility_drift(store, start, end) -> Optional[float]:
    """For recipe_hashes that ran 2+ times in the window, return the median
    relative spread of their headline metric. Lower is better."""
    rows = [
        r for r in _runs_in_window(store, start, end)
        if r.recipe_hash and r.metrics()
    ]
    if not rows:
        return None
    by_hash: dict[str, list[dict]] = {}
    for r in rows:
        by_hash.setdefault(r.recipe_hash, []).append(r.metrics())

    drifts: list[float] = []
    for metrics_list in by_hash.values():
        if len(metrics_list) < 2:
            continue
        # Pick the metric all runs share with non-zero values.
        common_keys = set.intersection(*(set(m.keys()) for m in metrics_list))
        for key in common_keys:
            values = [float(m[key]) for m in metrics_list if isinstance(m.get(key), (int, float))]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            if mean == 0:
                continue
            spread = (max(values) - min(values)) / abs(mean)
            drifts.append(spread)
            break  # one drift sample per recipe_hash
    if not drifts:
        return None
    drifts.sort()
    return drifts[len(drifts) // 2]


def _search_count(store, start, end) -> Optional[float]:
    """Count of searches that finished in the window."""
    rows = [
        s for s in store.list_searches(limit=500)
        if s.finished_at is not None and start <= s.finished_at <= end
    ]
    return float(len(rows))


# ── registry ────────────────────────────────────────────────────────────


METRICS: dict[str, MetricSpec] = {
    "notebook_pass_rate": MetricSpec(
        key="notebook_pass_rate",
        label="Notebook pass rate",
        description="% of finished runs whose notebook executed cleanly with no error cells.",
        direction="up_is_good",
        fmt="percent",
        fn=_notebook_pass_rate,
    ),
    "recipe_success_rate": MetricSpec(
        key="recipe_success_rate",
        label="Recipe success rate",
        description="% of templated-recipe runs that completed.",
        direction="up_is_good",
        fmt="percent",
        fn=_recipe_success_rate,
    ),
    "avg_time_to_metric": MetricSpec(
        key="avg_time_to_metric",
        label="Avg time to metric",
        description="Mean wall-clock seconds from run start to metrics emitted.",
        direction="down_is_good",
        fmt="seconds",
        fn=_avg_time_to_metric,
    ),
    "failed_run_count": MetricSpec(
        key="failed_run_count",
        label="Failed runs",
        description="Runs that finished with status=failed in the window.",
        direction="down_is_good",
        fmt="count",
        fn=_failed_run_count,
    ),
    "total_runs": MetricSpec(
        key="total_runs",
        label="Total runs",
        description="Every run that finished in the window (any status).",
        direction="up_is_good",
        fmt="count",
        fn=_total_runs,
    ),
    "provenance_coverage": MetricSpec(
        key="provenance_coverage",
        label="Provenance coverage",
        description="% of code cells in window's notebooks with finagent.node_id metadata.",
        direction="up_is_good",
        fmt="percent",
        fn=_provenance_coverage,
    ),
    "output_completeness": MetricSpec(
        key="output_completeness",
        label="Output completeness",
        description="% of code cells in window's notebooks with at least one output.",
        direction="up_is_good",
        fmt="percent",
        fn=_output_completeness,
    ),
    "reproducibility_drift": MetricSpec(
        key="reproducibility_drift",
        label="Reproducibility drift",
        description="Median relative metric spread across recipe re-runs.",
        direction="down_is_good",
        fmt="ratio",
        fn=_reproducibility_drift,
    ),
    "search_count": MetricSpec(
        key="search_count",
        label="Searches completed",
        description="Search artifacts finished in the window.",
        direction="up_is_good",
        fmt="count",
        fn=_search_count,
    ),
}


def compute_metrics(
    *,
    days: int = 7,
    keys: Optional[list[str]] = None,
    store: Optional[ExperimentStore] = None,
) -> dict:
    """Headline value per metric for a `days`-rolling window, plus a daily
    series so the dashboard can sparkline.

    Returns shape::

      {
        "window_days": 7,
        "as_of": <unix>,
        "metrics": {
          "<key>": {
            "label": ..., "description": ..., "direction": ..., "fmt": ...,
            "value": <float | None>,
            "series": [{"day": "YYYY-MM-DD", "value": <float | None>}, ...]
          }
        }
      }
    """
    store = store or get_store()
    end = time.time()
    start = end - days * 86400
    selected = list(keys) if keys else list(METRICS.keys())

    out_metrics: dict[str, dict] = {}
    for key in selected:
        spec = METRICS.get(key)
        if spec is None:
            continue
        try:
            value = spec.fn(store, start, end)
        except Exception:
            value = None
        series: list[dict] = []
        for d in range(days):
            day_start = end - (days - d) * 86400
            day_end = day_start + 86400
            try:
                v = spec.fn(store, day_start, day_end)
            except Exception:
                v = None
            day_iso = time.strftime("%Y-%m-%d", time.gmtime(day_start))
            series.append({"day": day_iso, "value": v})
        out_metrics[key] = {
            "label": spec.label,
            "description": spec.description,
            "direction": spec.direction,
            "fmt": spec.fmt,
            "value": value,
            "series": series,
        }

    return {
        "window_days": days,
        "as_of": end,
        "metrics": out_metrics,
    }


def list_metric_keys() -> list[dict]:
    """Light registry view for the toggle UI — no SQL queried."""
    return [
        {
            "key": s.key,
            "label": s.label,
            "description": s.description,
            "direction": s.direction,
            "fmt": s.fmt,
        }
        for s in METRICS.values()
    ]
