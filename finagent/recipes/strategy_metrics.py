"""Strategy-level financial metrics.

Used by template-generated notebooks to evaluate trading books in the same
language a portfolio manager uses — Sharpe, Sortino, max drawdown, turnover,
hit rate — instead of model-fit metrics like log-likelihood. Pure
pandas/numpy; no fitting, no I/O.

Conventions:

  * ``returns`` is a DataFrame indexed by date, one column per asset, values
    are simple per-period returns (decimal, e.g. 0.012 = +1.2%).
  * ``weights`` is a DataFrame with the same index/columns as ``returns``,
    values are portfolio weights (sum-to-1 typical, sum-to-0 for dollar-
    neutral). The book's per-period return is the row-wise dot product
    ``(weights.shift(1) * returns).sum(axis=1)`` — shifted by one because
    you only earn today's return on yesterday's position.
  * ``periods_per_year`` defaults to 252 (US/India equity trading days).
    Override for crypto (365), monthly (12), etc.

Every metric returns a single float (or NaN when the input is degenerate).
None of these helpers raise on empty input — they return NaN — so the
caller can lay out a metric table without try/except per cell.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def book_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Per-period book returns from a weights frame and an asset-returns frame.

    Shifts weights by one period — you earn today's return on yesterday's
    position. Aligns columns; missing columns contribute zero.
    """
    if weights is None or returns is None:
        return pd.Series(dtype=float)
    if weights.empty or returns.empty:
        return pd.Series(dtype=float, index=returns.index if returns is not None else None)
    common_cols = weights.columns.intersection(returns.columns)
    if len(common_cols) == 0:
        return pd.Series(0.0, index=returns.index)
    w = weights[common_cols].reindex(returns.index).fillna(0.0)
    r = returns[common_cols].fillna(0.0)
    return (w.shift(1) * r).sum(axis=1)


def total_return(book: pd.Series) -> float:
    """Cumulative compounded return over the whole window."""
    s = _clean(book)
    if s.empty:
        return float("nan")
    return float((1.0 + s).prod() - 1.0)


def annual_return(book: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualised return."""
    s = _clean(book)
    if s.empty or len(s) < 2:
        return float("nan")
    cum = (1.0 + s).prod()
    if cum <= 0:
        return float("nan")
    return float(cum ** (periods_per_year / len(s)) - 1.0)


def sharpe(
    book: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio. risk_free is a per-period excess (e.g. 0.0
    for a zero-rate assumption, 0.00015 for ~3.7% annual / 252)."""
    s = _clean(book)
    if s.empty or len(s) < 2:
        return float("nan")
    excess = s - risk_free
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(
    book: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino — Sharpe but penalising only the downside std."""
    s = _clean(book)
    if s.empty or len(s) < 2:
        return float("nan")
    excess = s - risk_free
    downside = excess[excess < 0]
    if downside.empty:
        return float("nan")
    dsd = downside.std(ddof=1)
    if not np.isfinite(dsd) or dsd == 0:
        return float("nan")
    return float(excess.mean() / dsd * np.sqrt(periods_per_year))


def max_drawdown(book: pd.Series) -> float:
    """Worst peak-to-trough drawdown of the equity curve, as a negative
    number (e.g. -0.18 = -18%)."""
    s = _clean(book)
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(book: pd.Series, periods_per_year: int = 252) -> float:
    """Annual return / |max drawdown|. NaN if no drawdown."""
    ar = annual_return(book, periods_per_year=periods_per_year)
    mdd = max_drawdown(book)
    if not np.isfinite(ar) or not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(ar / abs(mdd))


def apply_costs(
    book: pd.Series,
    weights: pd.DataFrame,
    *,
    bps_per_side: float = 0.0,
    borrow_bps: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Net out transaction costs from a book's gross daily returns.

    Two cost components, additive:

      ``bps_per_side`` · Per-trade execution cost in basis points
        (bid/ask half-spread + commission + slippage). Charged on
        per-day gross turnover (Σ |Δw|). A 100%-long book that swaps
        half its names pays ``0.5 × bps_per_side / 10000`` that day.
        The first row's "trade" (initial allocation) is NOT charged —
        we only charge for actual rebalances.

      ``borrow_bps`` · Annualised short-borrow rate in basis points.
        Charged per-day on gross short exposure (Σ |w_i| where w_i < 0)
        divided by ``periods_per_year``.

    Market impact is NOT modelled — see Costs docstring in types.py for
    the rationale. Phase-3 work.

    Returns a new Series; does not mutate ``book``.
    """
    if book is None or book.empty or weights is None or weights.empty:
        return book

    w = weights.fillna(0.0)

    # Transaction cost: |Δw| × bps_per_side / 10000.
    turnover_per_day = w.diff().abs().sum(axis=1).fillna(0.0)
    turnover_per_day.iloc[0] = 0.0  # don't charge for the initial allocation
    txn_cost = turnover_per_day * (bps_per_side / 10000.0)

    # Borrow cost: |shorts| × borrow_bps / 10000 / periods_per_year.
    short_exposure = w.clip(upper=0.0).abs().sum(axis=1)
    borrow_cost = short_exposure * (borrow_bps / 10000.0) / periods_per_year

    # Align indices in case the book slice and weights frame don't overlap
    # exactly — book.index wins.
    aligned_txn = txn_cost.reindex(book.index, fill_value=0.0)
    aligned_borrow = borrow_cost.reindex(book.index, fill_value=0.0)

    return book - aligned_txn - aligned_borrow


def turnover(weights: pd.DataFrame) -> float:
    """Mean per-period gross weight change. A 100%-long book that swaps
    half its names every day has turnover ≈ 0.5; a buy-and-hold book has
    turnover ≈ 0."""
    if weights is None or weights.empty:
        return float("nan")
    diffs = weights.fillna(0.0).diff().abs().sum(axis=1)
    diffs = diffs.iloc[1:]  # first row's diff is just the initial allocation
    if diffs.empty:
        return float("nan")
    return float(diffs.mean())


def hit_rate(book: pd.Series) -> float:
    """Fraction of periods with strictly positive return."""
    s = _clean(book)
    if s.empty:
        return float("nan")
    return float((s > 0).mean())


def exposure(weights: pd.DataFrame) -> float:
    """Average gross weight (|long| + |short|). 1.0 = always fully invested."""
    if weights is None or weights.empty:
        return float("nan")
    return float(weights.abs().sum(axis=1).mean())


def summary(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Compute the standard metric pack for one (weights, returns) pair."""
    book = book_returns(weights, returns)
    return {
        "total_return": total_return(book),
        "annual_return": annual_return(book, periods_per_year=periods_per_year),
        "sharpe": sharpe(book, risk_free=risk_free, periods_per_year=periods_per_year),
        "sortino": sortino(book, risk_free=risk_free, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(book),
        "calmar": calmar(book, periods_per_year=periods_per_year),
        "turnover": turnover(weights),
        "hit_rate": hit_rate(book),
        "exposure": exposure(weights),
    }


# ── strategy book builders (used by templates) ─────────────────────────


def value_book(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    long_only: bool = True,
    bottom_quantile: float = 0.34,
) -> pd.DataFrame:
    """Long the laggards. For multi-asset universes: cross-sectional,
    long the bottom quantile of trailing-cumulative-return (mean-reversion
    over the lookback). For single-asset: time-series rule, long when the
    cum return over the lookback is below zero (price below MA proxy).

    Returns a weights DataFrame with the same index/columns as ``prices``.
    """
    rets = prices.pct_change()
    cum = (1.0 + rets).rolling(lookback).apply(lambda r: r.prod() - 1.0, raw=True)
    if prices.shape[1] == 1:
        signal = (cum < 0).astype(float)
        return signal
    ranks = cum.rank(axis=1, pct=True)
    long_mask = (ranks <= bottom_quantile).astype(float)
    if long_only:
        weights = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        short_mask = (ranks >= (1.0 - bottom_quantile)).astype(float)
        weights = (long_mask - short_mask).div(
            (long_mask + short_mask).sum(axis=1).replace(0, np.nan), axis=0,
        ).fillna(0.0)
    return weights


def momentum_book(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    long_only: bool = True,
    top_quantile: float = 0.34,
) -> pd.DataFrame:
    """Long the leaders. Cross-sectional 12-1 momentum (ranks by cumulative
    return over ``lookback`` periods, excluding the most recent ``skip``).
    Single-asset case: time-series rule, long when 12-1 cum return > 0.
    """
    rets = prices.pct_change()
    cum = (1.0 + rets).rolling(lookback).apply(lambda r: r.prod() - 1.0, raw=True)
    cum_skip = cum.shift(skip)
    if prices.shape[1] == 1:
        signal = (cum_skip > 0).astype(float)
        return signal
    ranks = cum_skip.rank(axis=1, pct=True)
    long_mask = (ranks >= (1.0 - top_quantile)).astype(float)
    if long_only:
        weights = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        short_mask = (ranks <= top_quantile).astype(float)
        weights = (long_mask - short_mask).div(
            (long_mask + short_mask).sum(axis=1).replace(0, np.nan), axis=0,
        ).fillna(0.0)
    return weights


def buy_and_hold_book(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight buy-and-hold across all columns. Useful as the trivial
    baseline."""
    cols = prices.columns
    if len(cols) == 0:
        return pd.DataFrame(index=prices.index)
    w = pd.DataFrame(1.0 / len(cols), index=prices.index, columns=cols)
    return w


def regime_strategy_mapping(
    train_book_returns: dict[str, pd.Series],
    train_regimes: pd.Series,
    *,
    min_observations: int = 10,
) -> dict:
    """In a single training fold, decide which strategy in
    ``train_book_returns`` had the best Sharpe inside each regime label.

    Returns ``{regime_label: strategy_key}``. Falls back to the first
    strategy key when a regime has too few observations to estimate
    Sharpe reliably (caps the in-sample noise that would otherwise
    bias the OOS comparison).
    """
    fallback = next(iter(train_book_returns.keys()))
    out: dict = {}
    for state in pd.unique(train_regimes.dropna()):
        mask = (train_regimes == state) & train_regimes.notna()
        if mask.sum() < min_observations:
            out[state] = fallback
            continue
        scores: dict[str, float] = {}
        for key, ret in train_book_returns.items():
            r = ret.loc[mask].dropna()
            if r.empty or r.std(ddof=1) == 0 or not np.isfinite(r.std(ddof=1)):
                scores[key] = float("-inf")
            else:
                scores[key] = float(r.mean() / r.std(ddof=1))
        out[state] = max(scores, key=scores.get) if scores else fallback
    return out


# ── helpers ─────────────────────────────────────────────────────────────


def _clean(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()
