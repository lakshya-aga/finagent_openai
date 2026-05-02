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
        # Financial metrics. Each is also emitted with a book-prefix
        # (model_<m> / value_<m> / momentum_<m> / buy_and_hold_<m>) so the
        # project page can sort by any one and compare strategies head-to-head.
        "metrics": [
            "sharpe", "sortino", "annual_return", "total_return",
            "max_drawdown", "calmar", "turnover", "hit_rate", "exposure",
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
                "  metrics: [sharpe, sortino, annual_return, max_drawdown, turnover]\n\n"
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
                "  metrics: [sharpe, sortino, annual_return, max_drawdown, turnover]\n\n"
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

    # ── Step 7 — Reference strategy books ─────────────────────────────
    cells.append(_md_step(
        7, "Build reference strategy books",
        "n7_reference_books",
        "Construct value (laggard), momentum (12-1), and buy-and-hold "
        "weight frames — the strategies a researcher would compare against.",
    ))
    cells.append(_code_reference_books(recipe))

    # ── Step 8 — Model-driven book (switching for unsup, signal for sup) ─
    cells.append(_md_step(
        8, "Build model-driven book",
        "n8_model_book",
        "Unsupervised: walk-forward learn regime→strategy mapping in-sample, "
        "switch out-of-sample. Supervised: use the prediction as a long/flat signal.",
    ))
    cells.append(_code_model_book(recipe))

    # ── Step 9 — Financial metrics for every book + canonical output ──
    cells.append(_md_step(
        9, "Financial metrics + asset_returns / asset_weights",
        "n9_metrics",
        "Sharpe / Sortino / drawdown / turnover for each strategy. "
        "asset_weights / asset_returns bound to the model book — the "
        "platform's canonical comparison surface.",
    ))
    cells.append(_code_financial_metrics(recipe))

    # ── Step 10 — Charts ──────────────────────────────────────────────
    cells.append(_md_step(
        10, "Charts",
        "n10_charts",
        "Equity curves for every book, drawdown for the model book, "
        "and (unsupervised only) a regime ribbon overlaying price.",
    ))
    cells.append(_code_charts(recipe))

    # ── Step 11 — Decomposition (asset contribution + per-year returns) ─
    cells.append(_md_step(
        11, "Decomposition",
        "n11_decomposition",
        "Where did the returns come from? Per-asset contribution bar chart "
        "across books, plus a calendar-year return table for like-for-like "
        "year comparison.",
    ))
    cells.append(_code_decomposition(recipe))

    # ── Step 12 — Persist run summary for the experiment store ────────
    cells.append(_md_step(
        12, "Run summary",
        "n12_summary",
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

    Two bugs to avoid here:

    1. textwrap.dedent + json.dumps(indent=2) interpolated into an f-string
       breaks indentation because the multi-line JSON's continuation lines
       have no f-string indent. Solved by building the cell line-by-line.

    2. json.dumps emits JSON literals (``null``, ``true``, ``false``) that
       are NameErrors when pasted as Python source. Solved by wrapping the
       JSON string in ``json.loads(...)`` at runtime so RECIPE is a real
       dict regardless of which keys carry None/True/False.
    """
    module, cls = recipe.model.class_path.rsplit(".", 1)
    # Compact JSON keeps RECIPE on a single line — and we wrap with
    # json.loads(<repr>) so the cell evaluates cleanly even when the recipe
    # contains None / True / False values.
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
        f"RECIPE = json.loads({recipe_json!r})",
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
        # repr() renders Python-literal syntax (None / True / False) so the
        # kwargs work as a Python expression. json.dumps would emit
        # null/true/false → NameError when the cell evaluates.
        params_blob = repr(feat.params)
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
    # repr() renders None/True/False instead of null/true/false. The
    # XGBoost preset has `use_label_encoder: false` which would NameError
    # if json-serialised into Python source.
    model_kwargs = repr(recipe.model.params)
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


def _code_reference_books(recipe: Recipe) -> CellSpec:
    """Builds the reference strategies (value, momentum, buy-and-hold) on
    the prices DataFrame from recipe.data. These are the books a researcher
    would compare *against* — without them you can't tell whether the
    regime model adds value or just makes a different mistake.

    Two bugs we hit and now defend against here:

    1. Picking the wrong frame. The previous version iterated globals() and
       took the first DataFrame — fragile, because notebook scope can have
       feature matrices, macro frames, IPython repl variables in
       indeterminate order. Now we hardcode the prices variable name from
       the recipe at compile time.

    2. Volume contamination. yfinance multi-ticker downloads return a
       MultiIndex column frame ``(field, ticker)`` with all of OHLCV +
       dividends. ``select_dtypes('number')`` keeps Volume, whose daily
       pct_change is routinely ±10×. Compounded across 2000 days that
       hits 10^45 territory — exactly what the user reported.
    """
    # Bake the prices variable in at compile time. Convention across all
    # presets: the FIRST key in recipe.data is the prices frame. Fall back
    # to the first source whose kind is a price source if convention is
    # broken.
    keys = list(recipe.data.keys())
    if not keys:
        raise ValueError("recipe.data is empty — cannot build reference books")
    price_kinds = {"yfinance", "binance", "coingecko", "csv", "fin_kit"}
    prices_var = next(
        (k for k in keys if recipe.data[k].kind in price_kinds),
        keys[0],
    )

    body = textwrap.dedent(f"""\
        from finagent.recipes import strategy_metrics as sm

        # Use the recipe-declared prices source ({prices_var!r}) directly
        # instead of guessing from globals(). The previous heuristic
        # occasionally picked up the FRED macro frame or the feature
        # matrix, producing nonsense compound returns.
        ASSET_PRICES = {prices_var}
        if not isinstance(ASSET_PRICES, pd.DataFrame) or ASSET_PRICES.empty:
            raise RuntimeError(f"prices source {prices_var!r} is not a usable DataFrame")

        # yfinance with multiple tickers returns columns as a MultiIndex
        # (field, ticker). Flatten to a single price-per-ticker frame:
        # prefer 'Adj Close', fall back to 'Close', else first level.
        if isinstance(ASSET_PRICES.columns, pd.MultiIndex):
            _levels = list(ASSET_PRICES.columns.get_level_values(0).unique())
            _field = next(
                (f for f in ['Adj Close', 'Close', 'close', 'adj_close'] if f in _levels),
                _levels[0],
            )
            ASSET_PRICES = ASSET_PRICES[_field].copy()

        # Drop OHLCV / corporate-action columns by name — these survive a
        # single-ticker yfinance download and contaminate pct_change().
        # Whitelist by elimination: keep columns whose name doesn't match
        # any known non-price field.
        _drop_lower = {{'open', 'high', 'low', 'volume', 'dividends',
                       'stock splits', 'capital gains', 'adj volume'}}
        ASSET_PRICES = ASSET_PRICES.loc[
            :, [c for c in ASSET_PRICES.columns
                if str(c).strip().lower() not in _drop_lower]
        ]
        ASSET_PRICES = ASSET_PRICES.select_dtypes('number')

        if ASSET_PRICES.shape[1] == 0:
            raise RuntimeError(
                f"no price columns survived filtering on {{prices_var!r}}; "
                f"original columns were {{list({prices_var}.columns)[:8]}}"
            )

        # Sanity check daily returns. ETF / index returns above ±50% in a
        # single day indicate bad data (split not adjusted, stale tick,
        # macro series with sign flip in pct_change). Drop offending rows
        # so they don't poison compounded metrics.
        asset_returns_full = ASSET_PRICES.pct_change()
        _bad = asset_returns_full.abs() > 0.5
        if _bad.any().any():
            _n_bad = int(_bad.sum().sum())
            print(f'WARNING: {{_n_bad}} return cells > 50% daily move — zeroing.')
            asset_returns_full = asset_returns_full.where(~_bad, 0.0)

        # Reference books — same lookbacks across all regime studies so they
        # stay comparable across recipes. Tweak via the recipe later if you
        # want a sweep over windows.
        _value_w = sm.value_book(ASSET_PRICES, lookback=252)
        _momentum_w = sm.momentum_book(ASSET_PRICES, lookback=252, skip=21)
        _bh_w = sm.buy_and_hold_book(ASSET_PRICES)

        print(f'reference books built — assets: {{list(ASSET_PRICES.columns)}}')
        print(f'  shape: {{ASSET_PRICES.shape}}')
        print(f'  daily return percentiles: '
              f'1%={{float(asset_returns_full.stack().quantile(0.01)):.4f}}, '
              f'99%={{float(asset_returns_full.stack().quantile(0.99)):.4f}}')
        print(f'  value avg exposure   : {{sm.exposure(_value_w):.2f}}')
        print(f'  momentum avg exposure: {{sm.exposure(_momentum_w):.2f}}')
        """)
    return CellSpec("code", body, "n7_reference_books",
                    "Value + momentum + buy-and-hold reference weights.")


def _code_model_book(recipe: Recipe) -> CellSpec:
    """Build the model-driven book.

    Unsupervised path: walk-forward learn `regime → strategy` mapping from
    in-sample value/momentum returns within each fold's train window, then
    apply the mapping on the test window. Strategy choice is *learned*, not
    hardcoded — that's what makes the comparison meaningful.

    Supervised path: prediction is binary (forward-return-sign), so the
    book is "long when prediction=1, flat when prediction=0". No regime
    switching — the prediction itself is the signal.
    """
    is_unsupervised = recipe.target.kind == "unsupervised_regime"
    if is_unsupervised:
        body = textwrap.dedent("""\
            # ── Unsupervised path: regime-learned switching ────────────────
            # For each walk-forward fold:
            #   1. Inside the train window, compute value_book + momentum_book
            #      *book returns* aligned to the fitted regime labels.
            #   2. For each regime label, choose the strategy with the best
            #      in-sample Sharpe.
            #   3. Apply the mapping on the test window's regimes.
            from finagent.recipes import strategy_metrics as sm

            model_weights = pd.DataFrame(0.0, index=ASSET_PRICES.index, columns=ASSET_PRICES.columns)

            # We re-walk the same fold layout as Step 6. The regime labels
            # produced inside Step 6 (via fit_predict per fold) are reused via
            # `oos_predictions`; for the *training* fold we need fresh
            # in-sample regime labels — so we refit the model briefly on each
            # train slice purely to get train labels (cheap; same shape).
            _value_book_full = sm.book_returns(_value_w, asset_returns_full)
            _momentum_book_full = sm.book_returns(_momentum_w, asset_returns_full)

            for tr, te in _walk_forward(len(X)):
                # Fit a fresh model on this fold's train window to get train labels
                _m = type(model)(**RECIPE['model']['params'])
                if 'random_state' in type(_m).__init__.__code__.co_varnames:
                    _m = type(model)(**{**RECIPE['model']['params'], 'random_state': SEED})
                _m.fit(X.iloc[tr])
                _train_labels = pd.Series(_m.predict(X.iloc[tr]), index=X.index[tr])

                # In-sample regime-conditional strategy returns
                _train_idx = X.index[tr]
                _vb_train = _value_book_full.reindex(_train_idx).fillna(0.0)
                _mb_train = _momentum_book_full.reindex(_train_idx).fillna(0.0)
                _mapping = sm.regime_strategy_mapping(
                    {'value': _vb_train, 'momentum': _mb_train},
                    _train_labels,
                )

                # Apply mapping on the test window
                _test_idx = X.index[te]
                _test_labels = oos_predictions.reindex(_test_idx)
                for ts, label in _test_labels.items():
                    if pd.isna(label):
                        continue
                    pick = _mapping.get(label, 'value')
                    src = _value_w if pick == 'value' else _momentum_w
                    if ts in src.index:
                        model_weights.loc[ts] = src.loc[ts]
            print(f'model book built (regime-switched). nonzero weeks: {(model_weights.abs().sum(axis=1) > 0).sum()}')
        """)
    else:
        body = textwrap.dedent("""\
            # ── Supervised path: long-when-prediction=1, flat otherwise ────
            # Equal-weight long across all assets in scope when the model
            # signals up; flat (cash) otherwise. This is the simplest faithful
            # use of a binary classifier — fancier weight schemes (size by
            # predicted probability, etc.) are easy to swap in here.
            from finagent.recipes import strategy_metrics as sm

            _signal = oos_predictions.reindex(ASSET_PRICES.index).fillna(0).astype(float)
            _signal = _signal.clip(lower=0.0, upper=1.0)
            _eq_w = sm.buy_and_hold_book(ASSET_PRICES)  # row-normalised equal-weight
            model_weights = _eq_w.mul(_signal, axis=0).fillna(0.0)
            print(f'model book built (signal-driven). long weeks: {(_signal > 0).sum()}')
        """)
    return CellSpec("code", body, "n8_model_book",
                    "Model book: regime-switched (unsup) or signal-driven (sup).")


def _code_financial_metrics(recipe: Recipe) -> CellSpec:
    """Compute Sharpe / Sortino / drawdown / turnover / etc. for every book
    and emit them under namespaced keys so the project page can sort and
    compare. Binds asset_returns + asset_weights to the model book so the
    platform's downstream lineage / replay still works."""
    requested = list(recipe.evaluation.metrics)
    body = textwrap.dedent(f"""\
        from finagent.recipes import strategy_metrics as sm

        _books = {{
            'value':         _value_w,
            'momentum':      _momentum_w,
            'buy_and_hold':  _bh_w,
            'model':         model_weights,
        }}

        # Compute the standard pack for every book, then namespace each metric
        # by book so the project page columns are <book>_<metric>.
        metrics_out: dict[str, float] = {{}}
        for _book_name, _w in _books.items():
            _pack = sm.summary(_w, asset_returns_full)
            for _k, _v in _pack.items():
                metrics_out[f'{{_book_name}}_{{_k}}'] = _v

        # Headline aliases that Project-page filters use today (sharpe,
        # sortino, etc.) refer to the MODEL book — that's the strategy we'd
        # actually trade. Reference books surface as model_*-prefixed extras.
        for _alias in {requested!r}:
            _src = f'model_{{_alias}}'
            if _src in metrics_out:
                metrics_out[_alias] = metrics_out[_src]

        # Pretty-print so the notebook output is human-readable too.
        print()
        print('book          sharpe   sortino   ann_ret    max_dd   turnover')
        for _name in ['model', 'value', 'momentum', 'buy_and_hold']:
            print(
                f'  {{_name:13s}} '
                f'{{metrics_out[_name + "_sharpe"]:>6.2f}}  '
                f'{{metrics_out[_name + "_sortino"]:>7.2f}}  '
                f'{{metrics_out[_name + "_annual_return"] * 100:>+6.2f}}%  '
                f'{{metrics_out[_name + "_max_drawdown"] * 100:>+6.2f}}%  '
                f'{{metrics_out[_name + "_turnover"]:>6.3f}}'
            )

        # Bind the canonical platform contract: asset_returns + asset_weights
        # represent the model's actual book. The reference comparisons are
        # captured in metrics_out (and surfaced on the project page) but
        # don't pollute the lineage / kernel state for downstream cells.
        asset_returns = asset_returns_full.fillna(0.0)
        asset_weights = model_weights.fillna(0.0)
        print(f'\\nasset_returns shape={{asset_returns.shape}}, asset_weights shape={{asset_weights.shape}}')
        """)
    return CellSpec("code", body, "n9_metrics",
                    "Financial metrics for value / momentum / buy-and-hold / model + canonical asset_returns/weights.")


def _code_charts(recipe: Recipe) -> CellSpec:
    """Render the headline charts as inline PNGs.

    Three plots — equity curves (all books), drawdown of the model book, and
    (unsupervised only) a regime ribbon overlaying the first asset price. We
    explicitly call ``plt.show()`` so the kernel emits ``image/png`` outputs;
    without ``show()`` the figure object is the cell value and only ``text/
    plain`` (the repr) ends up in the notebook — which is how every prior run
    landed without renderable charts.
    """
    is_unsupervised = recipe.target.kind == "unsupervised_regime"
    body = textwrap.dedent("""\
        # NOTE: do NOT call matplotlib.use(...) here — the Jupyter kernel sets
        # the inline backend on startup, which routes figure outputs into the
        # cell as image/png. Forcing 'Agg' would suppress that and the notebook
        # would render with text/plain only (which is how earlier runs landed
        # without visible charts).
        import matplotlib.pyplot as plt
        from finagent.recipes import strategy_metrics as sm

        # ── Equity curves: growth of $1 for each book ───────────────────
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _palette = {
            'model':         '#2563eb',
            'value':         '#16a34a',
            'momentum':      '#f59e0b',
            'buy_and_hold':  '#94a3b8',
        }
        for _name, _w in _books.items():
            _bk = sm.book_returns(_w, asset_returns_full).fillna(0.0)
            if _bk.empty:
                continue
            _eq = (1.0 + _bk).cumprod()
            ax.plot(_eq.index, _eq.values, label=_name, linewidth=1.5,
                    color=_palette.get(_name))
        ax.set_title('Equity curves — model vs. reference books')
        ax.set_ylabel('Growth of $1 (compounded)')
        ax.legend(loc='best', frameon=False)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        # ── Drawdown of the model book ──────────────────────────────────
        _bk_m = sm.book_returns(model_weights, asset_returns_full).fillna(0.0)
        if not _bk_m.empty:
            _eq_m = (1.0 + _bk_m).cumprod()
            _peak = _eq_m.cummax()
            _dd = _eq_m / _peak - 1.0
            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.fill_between(_dd.index, _dd.values, 0.0,
                            color='#dc2626', alpha=0.35, linewidth=0)
            ax.plot(_dd.index, _dd.values, color='#991b1b', linewidth=1.0)
            ax.set_title('Model book drawdown')
            ax.set_ylabel('Drawdown')
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()
    """)
    if is_unsupervised:
        body += textwrap.dedent("""\

            # ── Regime ribbon overlay (unsupervised only) ───────────────
            # Shade the price track by the OOS regime label so a researcher
            # can visually check whether transitions line up with regime
            # changes. First numeric column of ASSET_PRICES is the proxy.
            _price = ASSET_PRICES.iloc[:, 0].dropna()
            _labels = oos_predictions.reindex(_price.index)
            if _labels.notna().any():
                fig, ax = plt.subplots(figsize=(10, 3.2))
                ax.plot(_price.index, _price.values, color='#0f172a', linewidth=1.0)
                _state_colors = ['#dbeafe', '#fee2e2', '#dcfce7', '#fef9c3', '#ede9fe']
                _states = sorted(_labels.dropna().unique())
                _ymin, _ymax = float(_price.min()), float(_price.max())
                for _i, _state in enumerate(_states):
                    _mask = (_labels == _state).reindex(_price.index, fill_value=False)
                    ax.fill_between(_price.index, _ymin, _ymax,
                                    where=_mask.values,
                                    color=_state_colors[_i % len(_state_colors)],
                                    alpha=0.6, linewidth=0,
                                    label=f'state {int(_state)}')
                ax.set_title(f'{ASSET_PRICES.columns[0]} with OOS regime ribbon')
                ax.set_ylabel(str(ASSET_PRICES.columns[0]))
                ax.legend(loc='best', frameon=False, ncol=min(4, len(_states)))
                ax.grid(alpha=0.25)
                plt.tight_layout()
                plt.show()
        """)
    return CellSpec("code", body, "n10_charts",
                    "Equity curves + drawdown + (unsup) regime ribbon as inline PNGs.")


def _code_decomposition(recipe: Recipe) -> CellSpec:
    """Per-asset contribution bar plot + per-year return table.

    Two questions every PM asks after the headline numbers:

      1. Which assets actually drove the returns? "The model book made
         12%/yr" hides the fact that 11% of it came from one ticker. The
         per-asset contribution (sum over time of weight × return) makes
         this visible. Stacked side-by-side per book so you can also see
         that, e.g., value loaded onto TLT and momentum onto SPY.

      2. How does the model compare year-by-year? A 4% Sharpe with one
         great year and three flat ones is a different beast from a 1.5
         Sharpe with consistent 8%/yr. The calendar-year table makes the
         year-by-year story explicit.

    Both rendered inline: bars as image/png, table as HTML via display().
    """
    body = textwrap.dedent("""\
        import matplotlib.pyplot as plt
        from finagent.recipes import strategy_metrics as sm

        # ── Per-asset contribution to total return for each book ────────
        # contribution_a = sum_t (w_{t-1, a} * r_{t, a})
        # i.e. the time-summed dollar return earned through asset `a`.
        # This is the additive decomposition: sum across assets equals
        # the total simple-return sum (NOT the compounded return — exact
        # decomposition of the compounded number requires log-returns,
        # but the simple-sum view is the one PMs actually read).
        _palette = {
            'model':         '#2563eb',
            'value':         '#16a34a',
            'momentum':      '#f59e0b',
            'buy_and_hold':  '#94a3b8',
        }
        _contrib_rows = {}
        for _name, _w in _books.items():
            _aligned_w = _w.reindex(asset_returns_full.index).fillna(0.0).shift(1)
            _per_asset = (_aligned_w * asset_returns_full.fillna(0.0)).sum(axis=0)
            _contrib_rows[_name] = _per_asset

        contribution_df = pd.DataFrame(_contrib_rows)
        # Order assets by absolute model contribution so the largest bars
        # sit on the left — easier to read on a wide screen.
        if 'model' in contribution_df.columns and contribution_df['model'].abs().sum() > 0:
            contribution_df = contribution_df.reindex(
                contribution_df['model'].abs().sort_values(ascending=False).index
            )

        # Grouped bar chart: one group per asset, one bar per book.
        _book_order = [b for b in ['model', 'value', 'momentum', 'buy_and_hold']
                       if b in contribution_df.columns]
        _n_books = len(_book_order)
        _assets = list(contribution_df.index)
        _n_assets = len(_assets)
        if _n_assets > 0 and _n_books > 0:
            import numpy as _np
            _bar_w = 0.8 / max(_n_books, 1)
            _x = _np.arange(_n_assets)
            fig, ax = plt.subplots(figsize=(max(8, _n_assets * 1.2), 4.0))
            for _i, _book in enumerate(_book_order):
                _vals = (contribution_df[_book].values * 100.0)  # to percent
                ax.bar(
                    _x + (_i - (_n_books - 1) / 2) * _bar_w,
                    _vals,
                    width=_bar_w,
                    label=_book,
                    color=_palette.get(_book),
                )
            ax.axhline(0, color='#475569', linewidth=0.5)
            ax.set_xticks(_x)
            ax.set_xticklabels(_assets, rotation=0)
            ax.set_ylabel('Total contribution (% of $1)')
            ax.set_title('Per-asset contribution to total return — by book')
            ax.legend(loc='best', frameon=False)
            ax.grid(axis='y', alpha=0.25)
            plt.tight_layout()
            plt.show()

            # Print the underlying table too — useful for screen-reader
            # audiences and for reading exact numbers off the bars.
            print('Per-asset contribution (% of $1):')
            print((contribution_df * 100.0).round(2).to_string())

        # ── Calendar-year returns by book ───────────────────────────────
        # Compound within each year: (1 + r).groupby(year).prod() - 1
        _yearly_rows = {}
        for _name, _w in _books.items():
            _bk = sm.book_returns(_w, asset_returns_full).fillna(0.0)
            if _bk.empty:
                continue
            _yearly_rows[_name] = ((1.0 + _bk).groupby(_bk.index.year).prod() - 1.0)
        yearly_df = pd.DataFrame(_yearly_rows)
        # Reorder columns to the canonical book order so headers don't
        # randomise based on dict insertion order across kernel restarts.
        _ordered_cols = [b for b in ['model', 'value', 'momentum', 'buy_and_hold']
                         if b in yearly_df.columns]
        yearly_df = yearly_df[_ordered_cols] if _ordered_cols else yearly_df

        # Add an "all" row at the bottom for the full-window compounded
        # return — same as model_total_return etc. but inline so the
        # reader doesn't have to scroll back to the metrics step.
        if not yearly_df.empty:
            _full = ((1.0 + yearly_df).prod() - 1.0)
            _full.name = 'all'
            yearly_df_with_full = pd.concat([yearly_df, _full.to_frame().T])

            # Render as HTML for a clean notebook table; format as percentages.
            try:
                from IPython.display import display
                _styled = (yearly_df_with_full * 100.0).round(2).style.format('{:+.2f}%')
                display(_styled)
            except Exception:
                # Fallback if IPython display isn't wired up — print plain
                print('Calendar-year returns (%):')
                print((yearly_df_with_full * 100.0).round(2).to_string())
        """)
    return CellSpec("code", body, "n11_decomposition",
                    "Per-asset contribution bars + per-year return table.")


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
            'rows_oos': int(len(oos_predictions)) if 'oos_predictions' in dir() else 0,
        }
        print('FINAGENT_RUN_SUMMARY ' + _json.dumps(SUMMARY, default=str))
        """)
    return CellSpec("code", body, "n12_summary", "Run-summary marker for harness ingestion.")
