"""Hidden Markov Model fit + label standardization.

Faithful port of the paper's ``hidden_markov_model.py``. Two
behaviours worth flagging:

1. ``fit_hmm`` runs hmmlearn ten times with different
   ``random_state`` seeds and picks the run with the highest
   log-likelihood. EM is prone to local-minima on multimodal
   indicator histograms — this is the workaround the hmmlearn
   docs themselves recommend.

2. ``standardize_regime_labels`` flips the 0/1 labels so the
   NORMAL (low-volatility) regime is always 0. HMM has no
   notion of "which state is which" — the labels are just
   indices — so without this step the downstream classifier
   would be fitting on labels whose meaning rotates with the
   random seed.

   The flip rule is "if state 0 occupies < 50% of the timeline
   (or state 1 occupies > 50%), swap." This rests on the
   assumption that the long-run majority regime is the normal
   one — fragile for short samples but fine for the multi-year
   training set the paper uses.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)


def fit_hmm(
    n_components: int,
    indicator: pd.Series,
    *,
    n_restarts: int = 10,
    n_iter: int = 1000,
    covariance_type: str = "full",
    verbose: bool = False,
) -> tuple[pd.Series, "hmm.GaussianHMM"]:
    """Fit a GaussianHMM on ``indicator`` and return ``(regimes,
    best_model)``.

    The model is fit ``n_restarts`` times with different random seeds
    (mirroring the official hmmlearn tutorial — EM is prone to local
    minima). The run with the highest log-likelihood wins, then we
    standardise the labels via ``standardize_regime_labels`` so
    regime 0 is always the long-run majority (normal) state.
    """
    X = indicator.to_numpy().reshape(-1, 1)

    models = []
    scores = []
    for seed in range(n_restarts):
        m = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
        )
        try:
            m.fit(X)
            score = m.score(X)
        except Exception as e:
            if verbose:
                logger.warning("regime_dc.fit_hmm: seed=%d failed (%s)", seed, e)
            continue
        models.append(m)
        scores.append(score)

    if not models:
        raise RuntimeError("regime_dc.fit_hmm: every restart failed")

    best = models[int(np.argmax(scores))]

    regimes = pd.Series(best.predict(X), index=indicator.index)
    regimes = standardize_regime_labels(regimes, verbose=verbose)
    return regimes, best


def standardize_regime_labels(
    regimes: pd.Series, *, verbose: bool = False
) -> pd.Series:
    """Flip labels so regime 0 is always the long-run majority state.

    Mirrors the paper's ``standardize_regime_labels``. The flip is a
    plain ``1 - regimes`` when EITHER:

      * the initial regime is 0 and it occupies <= 50% of the
        timeline (i.e. the model labelled the minority state as 0); OR
      * the initial regime is 1 and it occupies >= 50% (i.e. the
        model labelled the majority state as 1).

    Time-weighting (rather than count-weighting) matters because the
    DC-event indices aren't equispaced — a long stretch in one regime
    with few DC events shouldn't be dwarfed by a short jittery stretch
    with many events.
    """
    if regimes.empty:
        return regimes

    initial_regime = regimes.iloc[0]

    # If only one unique label appears, the model decided everything
    # is one regime. Assume it's normal (label 0) — flip iff that
    # label is currently 1.
    if regimes.nunique() == 1:
        if initial_regime == 1:
            if verbose:
                logger.info("regime_dc: only one regime label, flipping 1→0")
            return 1 - regimes
        return regimes

    total_duration_seconds = (regimes.index[-1] - regimes.index[0]).total_seconds()
    duration_initial = 0.0
    prev_time = regimes.index[0]
    for time, regime in regimes.iloc[1:].items():
        if regime == initial_regime:
            duration_initial += (time - prev_time).total_seconds()
        prev_time = time

    frac_initial = (
        duration_initial / total_duration_seconds if total_duration_seconds else 0.0
    )
    if verbose:
        logger.info(
            "regime_dc.standardize: initial=%d frac=%.3f total_secs=%.0f",
            int(initial_regime),
            frac_initial,
            total_duration_seconds,
        )

    flip = (initial_regime == 0 and frac_initial <= 0.5) or (
        initial_regime == 1 and frac_initial >= 0.5
    )
    if flip:
        if verbose:
            logger.info("regime_dc.standardize: flipping labels")
        return 1 - regimes
    return regimes
