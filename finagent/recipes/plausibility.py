"""Plausibility flags for run metrics.

Hedge-fund UAT surfaced a credibility gap: the project page would happily
render a Sharpe of 4.6 or an annual return of 707x next to a green
"COMPLETED" pill, with no warning. A junior analyst could screenshot that
into a memo before noticing the unit error or look-ahead bug that produced
it. This module is the backend half of the fix — it inspects a metrics
dict and emits human-readable warnings for each value that falls outside a
plausible band. The frontend is then free to render an amber chip next to
the offending number.

Design notes:

  * Bands are *inclusive* on both endpoints. A Sharpe of exactly 3.0 does
    not flag; a Sharpe of 3.01 does.
  * We flag, we do **not** clamp. Clamping at the API boundary would hide
    bugs (look-ahead, leakage, unit mismatch) that a portfolio manager
    needs to see. The DB keeps the raw value; the public payload gets a
    sibling ``metrics_flags`` field.
  * Templates compile metrics with optional book prefixes
    (``model_<m>`` / ``value_<m>`` / ``momentum_<m>`` / ``buy_and_hold_<m>``)
    so the project page can sort each book independently. We strip those
    prefixes when looking up bands so a namespaced metric like
    ``value_sharpe`` still hits the ``sharpe`` band.
  * NaN / Infinity / None are skipped — those are already scrubbed to None
    by ``experiments._finite_or_none`` and "no value" is not implausible,
    just absent.
  * Pure module: only stdlib + typing imports. No agents/tools dependency,
    so it is safe to call from the experiment serializer hot path without
    pulling in the heavy SDK side of the codebase.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional


# Default plausibility bands for an equity strategy at daily frequency.
# Each entry is (low_inclusive, high_inclusive). Anything outside the band
# is suspicious enough that a human should look before trusting the run.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "sharpe": (-3.0, 3.0),
    "sortino": (-5.0, 5.0),
    "calmar": (-50.0, 50.0),
    "annual_return": (-1.0, 1.0),         # -100% to +100% per year
    "total_return": (-10.0, 50.0),        # -1000% to +5000% cumulative
    "max_drawdown": (-1.0, 0.0),          # always non-positive, > -100%
    "turnover": (0.0, 5.0),               # daily turnover; 5 = 500%/day
    "win_rate": (0.0, 1.0),
}


# Book prefixes a template may apply to a metric key. Stripped before band
# lookup so ``value_sharpe`` resolves to the ``sharpe`` band.
_BOOK_PREFIXES: tuple[str, ...] = (
    "model_",
    "value_",
    "momentum_",
    "buy_and_hold_",
)


def _strip_book_prefix(key: str) -> str:
    for prefix in _BOOK_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def flag(
    metrics: Mapping[str, float | int | None],
    bands: Optional[Mapping[str, tuple[float, float]]] = None,
) -> dict[str, str]:
    """Return ``{metric_key: warning_message}`` for each out-of-band value.

    Parameters
    ----------
    metrics
        Mapping of metric key to numeric value (or None / NaN, which are
        skipped — "no value" is absent, not implausible).
    bands
        Optional override of the per-metric band table. When None, falls
        back to ``DEFAULT_BANDS``. Metrics not in the band table are
        skipped silently, so adding new metric names upstream never
        causes spurious flags.

    Returns
    -------
    dict[str, str]
        Empty when every metric is in-band. Otherwise the warning string
        is specific enough to read in isolation, e.g.
        ``"sharpe 8.42 is outside plausibility band (-3.0, 3.0); "
        "likely lookahead or look-back-too-short"``.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    flags: dict[str, str] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool):
            # bools are an int subclass — exclude explicitly so a
            # stray True/False can't slip through the comparison.
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if not isinstance(value, (int, float)):
            continue
        band = bands.get(_strip_book_prefix(key))
        if band is None:
            continue
        lo, hi = band
        if value < lo or value > hi:
            flags[key] = (
                f"{key} {float(value):.2f} is outside plausibility band "
                f"({float(lo)}, {float(hi)}); "
                "likely lookahead or look-back-too-short"
            )
    return flags


def flags_for_template(
    template_name: Optional[str],
    metrics: Mapping[str, float | int | None],
    template_bands_override: Optional[Mapping[str, tuple[float, float]]] = None,
) -> dict[str, str]:
    """Resolve template-specific bands and return flagged metrics.

    Merges ``template_bands_override`` over ``DEFAULT_BANDS`` so a
    template can widen one metric's band (e.g. crypto template loosening
    the Sharpe band) without restating the rest. ``template_name`` is
    accepted for symmetry with the caller's API but doesn't drive
    lookup here — band resolution from the template registry happens
    in ``finagent.experiments._resolve_bands`` to keep this module
    free of agents/tools imports.
    """
    bands: dict[str, tuple[float, float]] = dict(DEFAULT_BANDS)
    if template_bands_override:
        for k, v in template_bands_override.items():
            bands[k] = (float(v[0]), float(v[1]))
    return flag(metrics, bands)


__all__ = ["DEFAULT_BANDS", "flag", "flags_for_template"]
