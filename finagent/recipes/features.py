"""Feature builders referenced by recipe `features:` entries.

Each builder takes a generic ``**locals_kw`` of the host cell's namespace
plus its declared params. That lets a template say::

    _feat.build("returns", window=1, asset=asset_returns)

without a hard-coded contract on which DataFrame holds what.

New builders plug into ``_REGISTRY``. Names there are stable — recipes
reference them by string.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def build(name: str, **kwargs: Any) -> pd.DataFrame:
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"unknown feature builder: {name!r}. "
            f"Add it to finagent/recipes/features.py:_REGISTRY"
        )
    return fn(**kwargs)


def _first_numeric_frame(locals_kw: dict[str, Any]) -> pd.DataFrame:
    """Find the most likely 'prices' DataFrame in the local namespace."""
    for v in locals_kw.values():
        if isinstance(v, pd.DataFrame) and v.select_dtypes("number").shape[1] > 0:
            return v
    raise RuntimeError("no numeric DataFrame in scope; declare an explicit `asset=` arg")


# ── Builders ────────────────────────────────────────────────────────────


def _returns_lookback(window: int = 1, asset: pd.DataFrame | None = None,
                      **locals_kw: Any) -> pd.DataFrame:
    df = asset if asset is not None else _first_numeric_frame(locals_kw)
    out = df.select_dtypes("number").pct_change(window).rename(
        columns=lambda c: f"{c}_ret_{window}d"
    )
    return out


def _rolling_vol(window: int = 20, asset: pd.DataFrame | None = None,
                 **locals_kw: Any) -> pd.DataFrame:
    df = asset if asset is not None else _first_numeric_frame(locals_kw)
    rets = df.select_dtypes("number").pct_change()
    out = rets.rolling(window).std().rename(
        columns=lambda c: f"{c}_vol_{window}d"
    )
    return out


def _zscore(window: int = 60, asset: pd.DataFrame | None = None,
            **locals_kw: Any) -> pd.DataFrame:
    df = asset if asset is not None else _first_numeric_frame(locals_kw)
    nums = df.select_dtypes("number")
    rolling_mean = nums.rolling(window).mean()
    rolling_std = nums.rolling(window).std().replace(0, np.nan)
    out = ((nums - rolling_mean) / rolling_std).rename(
        columns=lambda c: f"{c}_z_{window}"
    )
    return out


def _macro_z_scores(window: int = 252, source: pd.DataFrame | None = None,
                    **locals_kw: Any) -> pd.DataFrame:
    """Z-score every column of the named macro frame."""
    df = source if source is not None else _first_numeric_frame(locals_kw)
    nums = df.select_dtypes("number")
    rolling_mean = nums.rolling(window).mean()
    rolling_std = nums.rolling(window).std().replace(0, np.nan)
    return ((nums - rolling_mean) / rolling_std).rename(
        columns=lambda c: f"macro_{c}_z_{window}"
    )


_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "returns_lookback": _returns_lookback,
    "rolling_vol": _rolling_vol,
    "zscore": _zscore,
    "macro_z_scores": _macro_z_scores,
}
