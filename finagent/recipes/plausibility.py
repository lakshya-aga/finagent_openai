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
"""

from __future__ import annotations

import math


# Default plausibility bands for an equity strategy at daily frequency.
# Each entry is (low_inclusive, high_inclusive). Anything outside the band
# is suspicious enough that a human should look before trusting the run.
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "sharpe": (-3, 3),
    "annual_return": (-1, 1),
    "total_return": (-10, 50),
    "calmar": (-50, 50),
    "max_drawdown": (-1, 0),
    "turnover": (0, 5),
    "sortino": (-5, 5),
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
    metrics: dict[str, float | None],
    bands: dict[str, tuple[float, float]] = DEFAULT_BANDS,
) -> dict[str, str]:
    """Return ``{metric_key: warning_message}`` for each out-of-band value.

    None and non-finite values (NaN, ±Infinity) are skipped — those mean
    "no value", which is not implausible. Keys without a band entry (after
    stripping book prefixes) are also skipped, so adding new metric names
    upstream never causes spurious flags.
    """
    flags: dict[str, str] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        band = bands.get(_strip_book_prefix(key))
        if band is None:
            continue
        lo, hi = band
        if value < lo or value > hi:
            flags[key] = (
                f"{value:.2f} outside expected range ({lo}, {hi}) "
                "— likely look-ahead, leakage, or unit mismatch"
            )
    return flags


__all__ = ["DEFAULT_BANDS", "flag"]
