"""Grid-search the (θ, indicator, ε) tuple for each classifier and
report the held-out test-set Table-4 reproduction.

Mirrors the paper's ``CustomCrossValidation`` + Section 2.6 grid:

  theta ∈ {0.01, 0.015, 0.02}
  DC indicator ∈ {TMV, T, R}
  epsilon ∈ {0.6, 0.65, 0.7, 0.75, 0.8}
  classifier ∈ {naive_bayes, logistic_regression, svm}

45 combinations per classifier × 3 classifiers = 135 pipeline fits.
Each fit is ~1s on the S&P 500 split, so the whole run lands in
under three minutes on a laptop. We could trivially parallelise via
``concurrent.futures.ProcessPoolExecutor`` but the sequential version
is easier to follow and the wall-clock is already fine.

The grid-search picks the (θ, indicator, ε) that maximises PROFIT
on the validation set's regime-dependent strategy. The paper picks
profit; we honour that choice so the optimal parameters land in the
same cell of the grid for like-for-like comparison.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .directional_change import get_data
from .pipeline import Pipeline, Splits

logger = logging.getLogger(__name__)


# Paper's grid (Table 2).
DEFAULT_GRID: dict = {
    "theta": [0.01, 0.015, 0.02],
    "dc_indicator": ["TMV", "T", "R"],
    "epsilon": [0.6, 0.65, 0.7, 0.75, 0.8],
}


# ── grid search ────────────────────────────────────────────────────


@dataclass
class CustomCrossValidation:
    """Sweep a parameter grid for one classifier and report the
    profit-maximising tuple.

    Single-classifier per instance (mirrors the paper, which builds one
    ``CustomCrossValidation`` per classifier and compares the three
    optima side-by-side in Table 3). Use ``run_benchmark`` to drive
    all three classifiers end-to-end.

    Parameters
    ----------
    prices : pd.Series
        Full price series (semi-daily).
    classifier : str
        ``"naive_bayes"`` / ``"logistic_regression"`` / ``"svm"``.
    grid : dict
        Parameter grid; defaults to the paper's Table 2 spec.
    splits : Splits
        Date boundaries; defaults to the paper's splits.
    verbose : bool
        Print per-cell progress to stdout.
    """

    prices: pd.Series
    classifier: str = "naive_bayes"
    grid: dict = field(default_factory=lambda: dict(DEFAULT_GRID))
    splits: Splits = field(default_factory=Splits)
    verbose: bool = True

    # Filled by .fit()
    losses: list[dict] = field(init=False, default_factory=list)
    optimal_params: dict | None = field(init=False, default=None)
    optimal_loss: dict | None = field(init=False, default=None)

    def fit(self) -> "CustomCrossValidation":
        """Sweep the grid, score each cell on the validation set's
        regime-dependent profit, cache the optimal parameters."""
        self.losses = []
        best_profit = -np.inf
        grid_iter = list(ParameterGrid(self.grid))
        n_total = len(grid_iter)
        t0 = time.time()

        for idx, params in enumerate(grid_iter, start=1):
            cell_t0 = time.time()
            try:
                pipe = Pipeline(
                    prices=self.prices,
                    splits=self.splits,
                    classifier=self.classifier,
                    eval_on="valid",
                    **params,
                ).fit()
                metrics = pipe.trading_metrics
            except Exception as e:
                logger.warning(
                    "regime_dc grid: %s %s failed (%s)",
                    self.classifier,
                    params,
                    e,
                )
                metrics = {"profit": -np.inf, "sharpe": 0.0, "mdd": 0.0, "n_trades": 0}

            row = {**params, **metrics, "_cell_secs": round(time.time() - cell_t0, 2)}
            self.losses.append(row)

            if metrics["profit"] > best_profit:
                best_profit = metrics["profit"]
                self.optimal_params = dict(params)
                self.optimal_loss = dict(metrics)

            if self.verbose:
                print(
                    f"  [{idx:>3d}/{n_total}] {self.classifier:>20s}  "
                    f"θ={params['theta']:.3f}  ind={params['dc_indicator']:>3s}  "
                    f"ε={params['epsilon']:.2f}  →  "
                    f"profit={metrics['profit']:+.4f}  sharpe={metrics['sharpe']:+.2f}  "
                    f"mdd={metrics['mdd']:.4f}  ({row['_cell_secs']}s)"
                )

        if self.verbose:
            print(
                f"  {self.classifier} grid done in {time.time() - t0:.1f}s — "
                f"optimal {self.optimal_params} profit={best_profit:+.4f}"
            )
        return self

    # Convenience accessor.
    def results_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.losses)
            .sort_values("profit", ascending=False)
            .reset_index(drop=True)
        )


# ── end-to-end benchmark ───────────────────────────────────────────


def run_benchmark(
    *,
    prices: Optional[pd.Series] = None,
    splits: Optional[Splits] = None,
    classifiers: tuple[str, ...] = ("naive_bayes", "logistic_regression", "svm"),
    verbose: bool = True,
) -> dict:
    """Reproduce the paper end-to-end.

    Steps:
      1. Pull S&P 500 (^GSPC) semi-daily prices over the full span
         covered by ``splits`` (defaults to 2005-01-01 → 2022-12-31).
      2. For each classifier, sweep the full grid and pick the
         profit-optimal (θ, indicator, ε).
      3. Refit each classifier on its optimal cell and evaluate the
         REGIME-DEPENDENT + MEAN-REVERTING + MOMENTUM strategies on
         the held-out test split (2020 → 2022).
      4. Return a dict with the per-classifier grid-search results,
         optimal parameters, and Table-4 reproduction frame.

    Returns
    -------
    dict with keys::

        {
          "optimal":   {clf: {theta, dc_indicator, epsilon}, ...},
          "table4":    pd.DataFrame  # rows = classifier, cols = (regime+control) × (profit, sharpe, mdd)
          "grids":     {clf: pd.DataFrame, ...}
          "prices":    pd.Series  (re-usable; the caller can drop)
        }
    """
    splits = splits or Splits()
    if prices is None:
        if verbose:
            print(
                f"regime_dc.run_benchmark: pulling ^GSPC {splits.train_start} → {splits.test_end}"
            )
        prices = get_data(
            tickers=["^GSPC"],
            start_date=splits.train_start,
            delta_hours=6.5,
            end_date=splits.test_end,
        )
        if verbose:
            print(f"  pulled {len(prices)} semi-daily observations")

    optimal: dict[str, dict] = {}
    grids: dict[str, pd.DataFrame] = {}
    table4_rows: list[dict] = []

    for c in classifiers:
        if verbose:
            print(f"\nregime_dc.run_benchmark: grid sweep — {c}")
        cv = CustomCrossValidation(
            prices=prices,
            classifier=c,
            splits=splits,
            verbose=verbose,
        ).fit()
        optimal[c] = dict(cv.optimal_params or {})
        grids[c] = cv.results_df()

        # Refit on optimal + run on TEST split.
        if verbose:
            print(f"  refitting {c} on {optimal[c]} and evaluating on TEST split")
        pipe = Pipeline(
            prices=prices,
            splits=splits,
            classifier=c,
            eval_on="test",
            **optimal[c],
        ).fit()
        test_metrics = pipe.evaluate_test()
        table4_rows.append(
            {
                "classifier": c,
                **{f"opt_{k}": v for k, v in optimal[c].items()},
                "regime_profit": test_metrics["regime_dependent"]["profit"],
                "regime_sharpe": test_metrics["regime_dependent"]["sharpe"],
                "regime_mdd": test_metrics["regime_dependent"]["mdd"],
                "regime_trades": test_metrics["regime_dependent"]["n_trades"],
                "mean_reverting_profit": test_metrics["mean_reverting"]["profit"],
                "mean_reverting_sharpe": test_metrics["mean_reverting"]["sharpe"],
                "mean_reverting_mdd": test_metrics["mean_reverting"]["mdd"],
                "momentum_profit": test_metrics["momentum"]["profit"],
                "momentum_sharpe": test_metrics["momentum"]["sharpe"],
                "momentum_mdd": test_metrics["momentum"]["mdd"],
            }
        )

    table4 = pd.DataFrame(table4_rows)
    return {
        "optimal": optimal,
        "table4": table4,
        "grids": grids,
        "prices": prices,
    }
