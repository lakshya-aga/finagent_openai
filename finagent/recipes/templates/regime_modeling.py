"""Regime modeling template.

Supports both unsupervised (HMM, KMeans, GMM) and supervised (XGBoost,
LightGBM, RandomForest, etc.) regime detection from the same recipe
shape. Branches on `recipe.target.kind`.

The template emits a list of `CellSpec` tuples — the compiler hands them
to the existing `add_cell` infrastructure, so generated notebooks get
the same syntax-highlighted, provenance-stamped treatment as AI-built
ones. No model is invented: the recipe pins the class, this module just
wires data → features → fit → eval → backtest in deterministic positions.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass

from ..types import Feature, Recipe


@dataclass
class CellSpec:
    cell_type: str                 # "code" or "markdown"
    content: str
    dag_node_id: str
    rationale: str = ""


TEMPLATE_NAME = "regime_modeling"


# Metadata surfaced to the frontend Recipe Builder. Each template declares
# its archetype (so the gallery can group cards), a short pitch, the field
# combinations it supports, and one or more named presets that act as
# starting points. Presets are full recipe YAML strings — when the user
# picks one, the editor pre-loads it.
METADATA = {
    "name": TEMPLATE_NAME,
    "title": "Market regime detection",
    "archetype": "regime",
    "tagline": "Classify the market into discrete regimes (risk-on / risk-off / volatile) and trade against the state.",
    "description": (
        "Build either an unsupervised regime model (HMM, GMM) or a supervised "
        "regime classifier (XGBoost, RandomForest) on the same feature pipeline. "
        "Walk-forward evaluation; emits standard asset_returns + asset_weights "
        "frames so the project page can compare runs side-by-side."
    ),
    "supports": {
        "targets": ["unsupervised_regime", "supervised_classification"],
        "models": [
            "hmmlearn.hmm.GaussianHMM",
            "sklearn.mixture.GaussianMixture",
            "xgboost.XGBClassifier",
            "sklearn.ensemble.RandomForestClassifier",
        ],
        "metrics": [
            "log_likelihood", "regime_persistence", "transition_entropy",
            "accuracy", "f1",
        ],
    },
    "presets": [
        {
            "key": "hmm_3state",
            "label": "HMM · 3-state (unsupervised)",
            "summary": "Gaussian HMM on SPY/TLT/GLD with macro features.",
            "yaml": (
                "name: regime_hmm_v1\n"
                "project: regime_research\n"
                "template: regime_modeling\n"
                "description: HMM regime model on SPY/TLT/GLD with macro features.\n\n"
                "data:\n"
                "  prices:\n"
                "    kind: yfinance\n"
                "    tickers: [SPY, TLT, GLD]\n"
                "    start: 2018-01-01\n"
                "  macro:\n"
                "    kind: fred\n"
                "    series_ids: [VIXCLS, T10Y2Y, DGS10]\n"
                "    start: 2018-01-01\n\n"
                "target:\n"
                "  kind: unsupervised_regime\n"
                "  method: hmm\n"
                "  n_states: 3\n\n"
                "features:\n"
                "  - name: returns_lookback\n"
                "    params: { window: 1 }\n"
                "  - name: rolling_vol\n"
                "    params: { window: 20 }\n\n"
                "model:\n"
                "  class_path: hmmlearn.hmm.GaussianHMM\n"
                "  params:\n"
                "    n_components: 3\n"
                "    covariance_type: full\n"
                "    n_iter: 100\n\n"
                "evaluation:\n"
                "  splits: walk_forward\n"
                "  train_window: 504\n"
                "  test_window: 21\n"
                "  metrics: [log_likelihood, regime_persistence, transition_entropy]\n\n"
                "seed: 42\n"
            ),
        },
        {
            "key": "xgb_classifier",
            "label": "XGBoost · forward-return-sign (supervised)",
            "summary": "XGBoost classifier predicting 5-day forward return sign on SPY.",
            "yaml": (
                "name: regime_xgb_v1\n"
                "project: regime_research\n"
                "template: regime_modeling\n"
                "description: XGBoost classifier predicting 5d forward return sign.\n\n"
                "data:\n"
                "  prices:\n"
                "    kind: yfinance\n"
                "    tickers: [SPY]\n"
                "    start: 2018-01-01\n\n"
                "target:\n"
                "  kind: supervised_classification\n"
                "  label_strategy: future_return_sign\n"
                "  horizon_days: 5\n\n"
                "features:\n"
                "  - name: returns_lookback\n"
                "    params: { window: 1 }\n"
                "  - name: rolling_vol\n"
                "    params: { window: 20 }\n"
                "  - name: zscore\n"
                "    params: { window: 60 }\n\n"
                "model:\n"
                "  class_path: xgboost.XGBClassifier\n"
                "  params:\n"
                "    n_estimators: 200\n"
                "    max_depth: 4\n"
                "    learning_rate: 0.05\n"
                "    use_label_encoder: false\n"
                "    eval_metric: logloss\n\n"
                "evaluation:\n"
                "  splits: walk_forward\n"
                "  train_window: 504\n"
                "  test_window: 21\n"
                "  metrics: [accuracy, f1]\n\n"
                "seed: 42\n"
            ),
        },
    ],
}


def supports(recipe: Recipe) -> bool:
    """Return True if this template can compile the recipe."""
    if recipe.template != TEMPLATE_NAME:
        return False
    if recipe.target.kind not in (
        "unsupervised_regime",
        "supervised_classification",
    ):
        return False
    return True


def compile(recipe: Recipe) -> list[CellSpec]:
    """Build the full notebook as a list of CellSpec items."""
    cells: list[CellSpec] = []

    # ── Step 1 — Header ────────────────────────────────────────────────
    cells.append(_md_header(recipe))

    # ── Step 2 — Imports ───────────────────────────────────────────────
    cells.append(_md_step(
        2, "Imports + recipe metadata",
        "n2_imports",
        "Pin the recipe identity and numerical seed up front so re-runs are deterministic.",
    ))
    cells.append(_code_imports(recipe))

    # ── Step 3 — Load data ─────────────────────────────────────────────
    cells.append(_md_step(
        3, "Load data",
        "n3_load",
        "Pull every datasource declared in the recipe.",
    ))
    cells.append(_code_load_data(recipe))

    # ── Step 4 — Build features ────────────────────────────────────────
    cells.append(_md_step(
        4, "Build features",
        "n4_features",
        "Derive the feature matrix from raw data. Each builder maps to a recipe entry.",
    ))
    cells.append(_code_features(recipe))

    # ── Step 5 — Build target (kind-specific) ─────────────────────────
    cells.append(_md_step(
        5, "Build target",
        "n5_target",
        f"target.kind={recipe.target.kind}; method={recipe.target.method or '-'}",
    ))
    cells.append(_code_target(recipe))

    # ── Step 6 — Walk-forward fit/predict loop ─────────────────────────
    cells.append(_md_step(
        6, "Walk-forward training and prediction",
        "n6_walk_forward",
        f"Splits={recipe.evaluation.splits}; train={recipe.evaluation.train_window}; "
        f"test={recipe.evaluation.test_window}.",
    ))
    cells.append(_code_walk_forward(recipe))

    # ── Step 7 — Metrics ───────────────────────────────────────────────
    cells.append(_md_step(
        7, "Compute metrics",
        "n7_metrics",
        f"Metrics: {', '.join(recipe.evaluation.metrics)}",
    ))
    cells.append(_code_metrics(recipe))

    # ── Step 8 — Backtest output (asset_returns / asset_weights) ──────
    cells.append(_md_step(
        8, "Build asset_returns and asset_weights",
        "n8_backtest",
        "Map regime predictions to portfolio weights and run the standard backtest hook.",
    ))
    cells.append(_code_backtest(recipe))

    # ── Step 9 — Persist run summary for the experiment store ─────────
    cells.append(_md_step(
        9, "Run summary",
        "n9_summary",
        "Print a JSON summary the experiment runner harvests for the Project page.",
    ))
    cells.append(_code_summary(recipe))

    return cells


# ── Cell builders ───────────────────────────────────────────────────────


def _md_header(recipe: Recipe) -> CellSpec:
    notes = (recipe.notes or "").strip()
    notes_block = f"\n\n{notes}" if notes else ""
    body = textwrap.dedent(f"""\
        # {recipe.name}

        **Project:** `{recipe.project}` · **Template:** `regime_modeling` · **Fingerprint:** `{recipe.fingerprint()}`

        {recipe.description.strip() if recipe.description else "Regime modeling research notebook compiled from a Recipe."}{notes_block}
        """)
    return CellSpec("markdown", body, "n1_header", "Recipe-rendered notebook header.")


def _md_step(n: int, title: str, node: str, rationale: str) -> CellSpec:
    body = textwrap.dedent(f"""\
        ### Step {n} — {title}

        **Node:** `{node}`

        **Why:** {rationale}
        """)
    return CellSpec("markdown", body, node, "")


def _code_imports(recipe: Recipe) -> CellSpec:
    """Imports + frozen RECIPE constant.

    Built as a plain ``"\\n".join(lines)`` instead of a ``textwrap.dedent``
    f-string. The previous implementation interpolated ``json.dumps(indent=2)``
    into a dedented f-string — but the multi-line JSON's continuation lines
    don't share the f-string's indentation, so ``dedent`` mis-computed the
    common-indent and the imports came out at column 8 (IndentationError on
    line 1 of the cell). Building line-by-line avoids that whole class of bug.
    """
    module, cls = recipe.model.class_path.rsplit(".", 1)
    # Compact JSON keeps RECIPE on a single line — easier on parsers, no
    # indentation interaction with the surrounding cell source.
    recipe_json = json.dumps(recipe.model_dump(mode="json"), default=str)
    lines = [
        "from __future__ import annotations",
        "import json",
        "import math",
        "from dataclasses import dataclass",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        f"from {module} import {cls}",
        "",
        "# Recipe metadata, pinned in a constant so the cell is reproducible.",
        f"RECIPE = {recipe_json}",
        f"SEED = {recipe.seed}",
        "np.random.seed(SEED)",
    ]
    return CellSpec("code", "\n".join(lines), "n2_imports", "Imports + frozen recipe blob.")


def _code_load_data(recipe: Recipe) -> CellSpec:
    """Generate one loader call per declared data source. Loader names are
    looked up in finagent.recipes.loaders so this stays declarative."""
    lines: list[str] = ["from finagent.recipes import loaders", ""]
    for var, ds in recipe.data.items():
        spec_blob = json.dumps(ds.model_dump(mode="json"), default=str)
        lines.append(f"{var} = loaders.load({spec_blob!r})")
    lines.append("")
    lines.append("# Sanity preview")
    if recipe.data:
        first = next(iter(recipe.data))
        lines.append(f"print({first}.shape if hasattr({first}, 'shape') else len({first}))")
    return CellSpec("code", "\n".join(lines), "n3_load", "Declarative datasource loaders.")


def _code_features(recipe: Recipe) -> CellSpec:
    """Each Feature gets a builder lookup. Unknown names raise — researcher
    sees the failing feature explicitly instead of a silent skip.
    """
    lines: list[str] = [
        "from finagent.recipes import features as _feat",
        "",
        "_feature_frames = []",
    ]
    for feat in recipe.features:
        params_blob = json.dumps(feat.params, default=str)
        lines.append(f"_feature_frames.append(_feat.build({feat.name!r}, **{params_blob}, **locals()))")
    lines.extend([
        "",
        "X = pd.concat(_feature_frames, axis=1).dropna(how='any')",
        "print(f'feature matrix: shape={X.shape}, columns={list(X.columns)[:8]}…')",
    ])
    return CellSpec("code", "\n".join(lines), "n4_features", "Feature matrix from declarative builders.")


def _code_target(recipe: Recipe) -> CellSpec:
    t = recipe.target
    if t.kind == "unsupervised_regime":
        # Unsupervised: the target IS what the model produces during fit.
        # We just align indices and let the walk-forward step fit + assign.
        body = textwrap.dedent("""\
            # Unsupervised regime modeling: the target is produced by the model
            # itself during fit. We don't pre-compute labels; instead we'll
            # `fit_predict` inside the walk-forward loop and treat the assigned
            # state as the regime label.
            y = pd.Series(index=X.index, dtype="float64", name="state")
            print(f'unsupervised target placeholder; will be filled per fold')
            """)
    elif t.kind == "supervised_classification":
        strategy = t.label_strategy or "future_return_sign"
        h = t.horizon_days or 5
        if strategy in ("future_return_sign", "next_return_sign"):
            body = textwrap.dedent(f"""\
                # Supervised classification: forward-return-sign over a {h}-day horizon.
                # Pick the first numeric column in X's source frame as the asset return.
                ASSET_RETURNS = next(
                    (df for df in [{', '.join(recipe.data.keys())}] if isinstance(df, pd.DataFrame) and df.select_dtypes('number').shape[1] > 0),
                    None,
                )
                if ASSET_RETURNS is None:
                    raise RuntimeError("no numeric DataFrame found in declared data sources for target labels")
                _ret = ASSET_RETURNS.select_dtypes('number').iloc[:, 0]
                _fwd = _ret.shift(-{h}).rolling({h}).sum()
                y = (_fwd > 0).astype(int).reindex(X.index).dropna()
                X = X.reindex(y.index)
                print(f'supervised target: positive={{(y==1).mean():.3f}}, negative={{(y==0).mean():.3f}}')
                """)
        elif strategy == "vol_quantile":
            thr = t.threshold if t.threshold is not None else 0.66
            body = textwrap.dedent(f"""\
                # Supervised classification: high-vol vs low-vol regime by rolling quantile.
                ASSET_RETURNS = next(
                    (df for df in [{', '.join(recipe.data.keys())}] if isinstance(df, pd.DataFrame) and df.select_dtypes('number').shape[1] > 0),
                    None,
                )
                if ASSET_RETURNS is None:
                    raise RuntimeError("no numeric DataFrame found for vol-quantile labels")
                _ret = ASSET_RETURNS.select_dtypes('number').iloc[:, 0]
                _vol = _ret.rolling({t.horizon_days or 21}).std()
                _q = _vol.expanding(252).quantile({thr})
                y = (_vol > _q).astype(int).reindex(X.index).dropna()
                X = X.reindex(y.index)
                print(f'vol-quantile target: high={{(y==1).mean():.3f}}, low={{(y==0).mean():.3f}}')
                """)
        else:
            body = f"raise NotImplementedError({strategy!r})"
    else:
        body = f"raise NotImplementedError(target_kind={t.kind!r})"
    return CellSpec("code", body, "n5_target", f"target={t.kind}/{t.method or '-'}.")


def _code_walk_forward(recipe: Recipe) -> CellSpec:
    """Walk-forward (or expanding-window) fit/predict loop.

    Built line-by-line with explicit indentation. The previous version mixed
    ``textwrap.dedent`` with ``textwrap.indent + .strip()`` interpolated mid-
    line via an f-string — which dropped the substituted block's continuation
    lines outside the for-loop body. Lines came out at column 0 instead of
    column 4, causing IndentationError + cascade of NameErrors.
    """
    e = recipe.evaluation
    train, test = e.train_window, e.test_window
    model_kwargs = json.dumps(recipe.model.params, default=str)
    cls_name = recipe.model.class_path.rsplit(".", 1)[1]
    is_unsupervised = recipe.target.kind == "unsupervised_regime"

    if e.splits == "walk_forward":
        loop_def = [
            "from typing import Iterator",
            "",
            f"def _walk_forward(n: int, train: int = {train}, test: int = {test}) -> Iterator[tuple[slice, slice]]:",
            "    start = 0",
            "    while start + train + test <= n:",
            "        yield slice(start, start + train), slice(start + train, start + train + test)",
            "        start += test  # rolling step",
        ]
    elif e.splits == "expanding_window":
        loop_def = [
            "from typing import Iterator",
            "",
            f"def _walk_forward(n: int, train: int = {train}, test: int = {test}) -> Iterator[tuple[slice, slice]]:",
            "    start = 0",
            "    tr_end = train",
            "    while tr_end + test <= n:",
            "        yield slice(start, tr_end), slice(tr_end, tr_end + test)",
            "        tr_end += test  # train window grows",
        ]
    else:
        loop_def = [
            "raise NotImplementedError('only walk_forward and expanding_window are templated today')",
        ]

    # Lines INSIDE the for-loop body all start at column 4. Use try/except to
    # tolerate models that don't accept random_state (instead of inspecting
    # __init__.__code__.co_varnames at runtime, which is fragile).
    if is_unsupervised:
        fit_lines = [
            "    try:",
            f"        model = {cls_name}(**{model_kwargs}, random_state=SEED)",
            "    except TypeError:",
            f"        model = {cls_name}(**{model_kwargs})",
            "    model.fit(X.iloc[tr])",
            "    preds = model.predict(X.iloc[te])",
        ]
    else:
        fit_lines = [
            f"    model = {cls_name}(**{model_kwargs})",
            "    model.fit(X.iloc[tr], y.iloc[tr])",
            "    preds = model.predict(X.iloc[te])",
        ]

    body_lines = [
        *loop_def,
        "",
        "oos_predictions = pd.Series(index=X.index, dtype='float64', name='pred')",
        "for tr, te in _walk_forward(len(X)):",
        *fit_lines,
        "    oos_predictions.iloc[te] = preds",
        "oos_predictions = oos_predictions.dropna()",
        "print(f'OOS predictions: {len(oos_predictions)} rows')",
    ]
    return CellSpec("code", "\n".join(body_lines), "n6_walk_forward", "Walk-forward fit/predict loop.")


_METRIC_CODE = {
    "accuracy": "metrics_out['accuracy'] = float((y.reindex(oos_predictions.index) == oos_predictions).mean())",
    "f1": (
        "from sklearn.metrics import f1_score\n"
        "metrics_out['f1'] = float(f1_score(y.reindex(oos_predictions.index), oos_predictions, average='macro'))"
    ),
    "log_likelihood": (
        "metrics_out['log_likelihood'] = float(getattr(model, 'score', lambda *_: float('nan'))(X.iloc[-1:].values))"
    ),
    "regime_persistence": (
        "metrics_out['regime_persistence'] = float((oos_predictions.shift(1) == oos_predictions).mean())"
    ),
    "transition_entropy": (
        "import numpy as _np\n"
        "_t = pd.crosstab(oos_predictions.shift(1), oos_predictions, normalize='index').fillna(0).values\n"
        "_with_log = _np.where(_t > 0, _t * _np.log(_t + 1e-12), 0.0)\n"
        "metrics_out['transition_entropy'] = float(-_with_log.sum())"
    ),
    "sharpe": (
        "_strat_returns = oos_predictions.diff().fillna(0).abs() * 0  # placeholder; real Sharpe lands in step 8\n"
        "metrics_out['sharpe'] = float('nan')"
    ),
}


def _code_metrics(recipe: Recipe) -> CellSpec:
    lines = ["metrics_out: dict[str, float] = {}"]
    for m in recipe.evaluation.metrics:
        snippet = _METRIC_CODE.get(m)
        if snippet is None:
            lines.append(f"# unknown metric {m!r}; skipped (add to _METRIC_CODE in template)")
            continue
        lines.append(snippet)
    lines.append("for k, v in metrics_out.items():")
    lines.append("    print(f'  {k}: {v:.4f}' if v == v else f'  {k}: nan')")
    return CellSpec("code", "\n".join(lines), "n7_metrics", "Compute requested metrics.")


def _code_backtest(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        # Translate regime predictions into a tradable weight series.
        # For unsupervised regimes: equal-weight long when in state 0, flat otherwise — a placeholder
        # the researcher should replace with a real allocation policy.
        ASSET_RETURNS_DF = next(
            (v for v in list(globals().values()) if isinstance(v, pd.DataFrame) and v.select_dtypes('number').shape[1] > 0),
            None,
        )
        if ASSET_RETURNS_DF is None:
            raise RuntimeError("no DataFrame available to derive asset_returns")
        _asset = ASSET_RETURNS_DF.select_dtypes('number').iloc[:, [0]]
        _asset.columns = [_asset.columns[0]]
        asset_returns = _asset.pct_change().fillna(0).reindex(oos_predictions.index)

        # Default policy: long-when-state-0; researcher overrides by editing this cell.
        _signal = (oos_predictions == 0).astype(int)
        asset_weights = pd.DataFrame(
            {asset_returns.columns[0]: _signal.reindex(asset_returns.index).fillna(0)}
        )
        print(f'asset_returns shape={asset_returns.shape}, asset_weights shape={asset_weights.shape}')
        """)
    return CellSpec("code", body, "n8_backtest", "Default policy maps state→weights; override here.")


def _code_summary(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        # Emit a JSON line the experiment runner picks up to populate the Project page.
        # The harness greps for the marker prefix below — keep it intact.
        import json as _json
        SUMMARY = {
            'recipe_name': RECIPE['name'],
            'project': RECIPE['project'],
            'fingerprint': RECIPE.get('fingerprint') if 'fingerprint' in RECIPE else None,
            'metrics': metrics_out,
            'rows_oos': int(len(oos_predictions)),
        }
        print('FINAGENT_RUN_SUMMARY ' + _json.dumps(SUMMARY, default=str))
        """)
    return CellSpec("code", body, "n9_summary", "Run-summary marker for harness ingestion.")
