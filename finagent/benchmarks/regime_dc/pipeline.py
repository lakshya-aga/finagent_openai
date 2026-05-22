"""End-to-end pipeline: data → DC indicators → HMM labels → classifier
→ predicted regimes → trading strategy.

Single-asset, single-classifier, single-(θ, ε, indicator) tuple.
``CustomCrossValidation`` (grid_search.py) sweeps the tuples.

Faithful port of the paper's ``cross_validation.Pipeline`` with the
deprecated chained-indexing pandas idioms removed.

Splits mirror the paper:

  train       2005-01-01 → 2017-12-31
  validation  2018-01-01 → 2019-12-31
  test        2020-01-01 → 2022-12-31
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import classifiers as clf
from . import directional_change as dc_mod
from . import hmm as hmm_mod
from . import strategy as strat_mod

logger = logging.getLogger(__name__)


# ── split definitions ──────────────────────────────────────────────


@dataclass
class Splits:
    """Date boundaries for the three-way split. Defaults match the
    paper (Section 1 + Section 2.6)."""
    train_start: str = "2005-01-01"
    train_end:   str = "2017-12-31"
    valid_start: str = "2018-01-01"
    valid_end:   str = "2019-12-31"
    test_start:  str = "2020-01-01"
    test_end:    str = "2022-12-31"


# ── pipeline ───────────────────────────────────────────────────────


@dataclass
class Pipeline:
    """One pipeline run for one (θ, indicator, ε, classifier) tuple.

    Parameters
    ----------
    prices : pd.Series
        Full price series (semi-daily, from ``directional_change.get_data``).
    splits : Splits
        Date boundaries for train/valid/test.
    theta : float
        DC threshold. Grid: {0.01, 0.015, 0.02}.
    dc_indicator : str
        Which DC indicator to feed the HMM/classifier. ``"R"`` / ``"T"`` / ``"TMV"``.
    epsilon : float
        Probability threshold for predicting the abnormal regime (label 1).
        Grid: {0.6, 0.65, 0.7, 0.75, 0.8}.
    classifier : str
        ``"naive_bayes"`` / ``"logistic_regression"`` / ``"svm"``.
    trade_threshold : float
        |TMV| trigger for entering a position. Paper: 0.5.
    n_regimes : int
        HMM components. Paper: 2.
    eval_on : str
        ``"valid"`` (default; used by grid search) or ``"test"``.
    """

    prices: pd.Series
    splits: Splits = field(default_factory=Splits)
    theta: float = 0.01
    dc_indicator: str = "R"
    epsilon: float = 0.5
    classifier: str = "naive_bayes"
    trade_threshold: float = strat_mod.TRADE_THRESHOLD
    n_regimes: int = 2
    eval_on: str = "valid"

    # Filled by .fit()
    indicator_train: pd.Series = field(init=False, default=None)
    indicator_valid: pd.Series = field(init=False, default=None)
    indicator_test:  pd.Series = field(init=False, default=None)
    regimes_train:   pd.Series = field(init=False, default=None)
    regimes_valid:   pd.Series = field(init=False, default=None)
    regimes_test:    pd.Series = field(init=False, default=None)
    hmm_model:       object    = field(init=False, default=None)
    clf_model:       object    = field(init=False, default=None)
    # Per-split DC events + indicator dict for downstream visualisation
    _dc:         dict = field(init=False, default_factory=dict)
    _indicators: dict = field(init=False, default_factory=dict)

    # ── helpers ────────────────────────────────────────────────────

    def _slice(self, split: str) -> pd.Series:
        s = self.splits
        if split == "train":
            return self.prices.loc[s.train_start:s.train_end]
        if split == "valid":
            return self.prices.loc[s.valid_start:s.valid_end]
        if split == "test":
            return self.prices.loc[s.test_start:s.test_end]
        raise ValueError(f"unknown split {split!r}")

    def _indicator_for(self, split: str) -> pd.Series:
        return self._indicators[self.dc_indicator][split]

    # ── main entry ─────────────────────────────────────────────────

    def fit(self, *, verbose: bool = False) -> "Pipeline":
        """Run the full pipeline on this (θ, indicator, ε, classifier)
        tuple. Idempotent — repeated calls overwrite the cached state.
        Returns ``self`` for chaining."""
        # 1. Compute DC events + the three indicators per split.
        self._dc = {}
        self._indicators = {"R": {}, "T": {}, "TMV": {}}
        for split in ("train", "valid", "test"):
            prices_split = self._slice(split)
            d, tmv, T, R = dc_mod.compute_indicators(prices_split, theta=self.theta)
            self._dc[split] = d
            self._indicators["R"][split]   = R
            self._indicators["T"][split]   = T
            self._indicators["TMV"][split] = tmv

        self.indicator_train = self._indicator_for("train")
        self.indicator_valid = self._indicator_for("valid")
        self.indicator_test  = self._indicator_for("test")

        # 2. Fit the HMM on the train indicator → labels.
        self.regimes_train, self.hmm_model = hmm_mod.fit_hmm(
            n_components=self.n_regimes,
            indicator=self.indicator_train,
            verbose=verbose,
        )

        # 3. Train the classifier on (train indicator, HMM labels).
        self.clf_model, valid_pred = clf.fit_predict(
            self.classifier,
            X_train=self.indicator_train.values,
            y_train=self.regimes_train.values.astype(int),
            X_test=self.indicator_valid.values,
            epsilon=self.epsilon,
        )
        self.regimes_valid = pd.Series(valid_pred, index=self.indicator_valid.index)

        # Also predict on test — cheap, lets the same fit serve both
        # cross-val and final eval.
        test_pred = clf.predict(self.clf_model, self.indicator_test.values, epsilon=self.epsilon)
        self.regimes_test = pd.Series(test_pred, index=self.indicator_test.index)

        return self

    # ── strategy eval ──────────────────────────────────────────────

    def _run_strategies(self, split: str) -> dict[str, dict]:
        """Run the three strategies (regime-dependent + 2 controls) on
        the chosen split and return their metrics dicts."""
        prices_split = self._slice(split)
        regimes_split = self.regimes_valid if split == "valid" else self.regimes_test
        dc_split = self._dc[split]

        ev = strat_mod.build_event_frame(
            prices=prices_split, dc=dc_split,
            regimes=regimes_split, theta=self.theta,
        )

        out_regime = strat_mod.regime_dependent(ev, threshold=self.trade_threshold)
        out_meanrev = strat_mod.mean_reverting_control(ev, threshold=self.trade_threshold)
        out_momentum = strat_mod.momentum_control(ev, threshold=self.trade_threshold)

        return {
            "regime_dependent":  strat_mod.metrics_summary(out_regime,   name="regime_dependent"),
            "mean_reverting":    strat_mod.metrics_summary(out_meanrev,  name="mean_reverting"),
            "momentum":          strat_mod.metrics_summary(out_momentum, name="momentum"),
        }

    @property
    def trading_metrics(self) -> dict:
        """Metrics on the cross-validation target split — defaults to
        validation. Used by ``CustomCrossValidation`` to score a grid
        cell. Returns the REGIME-DEPENDENT strategy's metrics (the
        paper grid-searches over profit on the regime-dependent book)."""
        return self._run_strategies(self.eval_on)["regime_dependent"]

    def evaluate_test(self) -> dict[str, dict]:
        """Run all three strategies on the held-out test split (Table 4
        reproduction). Should be called only after ``fit`` has been run
        with the optimal (θ, indicator, ε) from cross-validation."""
        return self._run_strategies("test")
