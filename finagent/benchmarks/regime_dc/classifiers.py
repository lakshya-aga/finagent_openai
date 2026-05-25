"""Binary regime classifiers — NBC, LR, SVM — with ε-threshold predict.

Faithful port of ``NaiveBayesClassifier.py`` / ``logistic_regression.py``
/ ``svm.py`` from the paper, collapsed into one module since the three
classes share the same train/predict shape. The only model-specific
detail is the constructor; everything else is identical.

Each classifier exposes:

  * ``fit(X_train, y_train)`` → trained sklearn estimator
  * ``predict(model, X, epsilon)`` → 0/1 labels, with regime 1
    (abnormal) emitted only when ``P[y=1|X] >= epsilon``. Defaults
    to the standard 0.5 threshold; the paper grid-searches over
    {0.6, 0.65, 0.7, 0.75, 0.8} so the abnormal regime is only
    flagged on high-confidence predictions.

The probability threshold ε is the third hyperparameter in the grid
(alongside θ and the DC indicator choice). Picking a high ε lets the
strategy stay in mean-reversion mode longer, which the paper found
to help — see Table 3.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class _BinaryClassifier(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> None: ...
    def predict(self, X: ArrayLike) -> np.ndarray: ...
    def predict_proba(self, X: ArrayLike) -> np.ndarray: ...


# ── factories ───────────────────────────────────────────────────────


def make_nbc() -> GaussianNB:
    """Naive Bayes with Gaussian kernels — the paper's headline classifier.
    Only one covariate (the chosen DC indicator) so the conditional-
    independence assumption is moot."""
    return GaussianNB()


def make_lr() -> LogisticRegression:
    """Unpenalised logistic regression. The paper uses
    ``penalty='none'`` in scikit-learn ≤1.1; that string was deprecated
    and removed — modern sklearn takes ``penalty=None``. Behaviour is
    identical: the L2 term is dropped from the objective."""
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)


def make_svm() -> "object":
    """SVC with RBF kernel + standard-scaled features, ``probability=True``
    so we can apply the ε threshold. Wrapped in a Pipeline so the scaler
    refits on each training fold."""
    return make_pipeline(StandardScaler(), SVC(probability=True))


_FACTORIES = {
    "naive_bayes": make_nbc,
    "logistic_regression": make_lr,
    "svm": make_svm,
}


def build_classifier(name: str) -> _BinaryClassifier:
    """Look up a classifier by name. Raises ``KeyError`` on unknown
    names so a typo in the grid surfaces loud rather than silently
    skipping."""
    try:
        return _FACTORIES[name]()
    except KeyError as e:
        raise KeyError(
            f"unknown classifier {name!r}; expected one of {list(_FACTORIES)}",
        ) from e


# ── train + predict ────────────────────────────────────────────────


def train(name: str, X: ArrayLike, y: ArrayLike) -> _BinaryClassifier:
    """Fit a fresh classifier of the given ``name`` on (X, y)."""
    model = build_classifier(name)
    model.fit(np.asarray(X).reshape(-1, 1), np.asarray(y))
    return model


def predict(
    model: _BinaryClassifier, X: ArrayLike, *, epsilon: float = 0.5
) -> np.ndarray:
    """Predict 0/1 regime labels with an ε threshold on P[y=1|X].

    ``epsilon=0.5`` reduces to the model's native ``predict`` (which
    is what the paper's wrapper does — there's no scenario in the
    grid where a non-default ε would round-trip to the default). Any
    other value triggers the probability-based decision rule.
    """
    X_arr = np.asarray(X).reshape(-1, 1)
    if epsilon == 0.5:
        return np.asarray(model.predict(X_arr)).astype(int)
    probs = model.predict_proba(X_arr)
    return np.where(probs[:, 1] >= float(epsilon), 1, 0)


def fit_predict(
    name: str,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    *,
    epsilon: float = 0.5,
) -> tuple[_BinaryClassifier, np.ndarray]:
    """Convenience: fit on (X_train, y_train) then predict on X_test.
    Returns ``(trained_model, predicted_labels)`` so callers can hold
    on to the model for inspection (decision boundary plots etc.)."""
    model = train(name, X_train, y_train)
    return model, predict(model, X_test, epsilon=epsilon)
