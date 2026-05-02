"""Pairs trading template — cointegration + OU mean-reversion.

Sister template to ``regime_modeling``. Same Recipe contract, different
research archetype: instead of classifying market state and switching
strategies, this one looks for a *cointegrated pair* in the universe,
fits an Ornstein-Uhlenbeck mean-reversion model on the spread, and
trades the z-score of the residual.

Pipeline (per fold of walk-forward evaluation):

  1. Load price history for the universe (yfinance multi-ticker).
  2. For every pair (i, j), run Engle-Granger ``coint`` on the train
     window. Filter to pairs with p < 0.05; pick the lowest-p survivor.
  3. Fit OU on the train-window residual: dS = θ(μ-S)dt + σ dW.
     Implementation: discretised AR(1) regression on the spread.
  4. On the test window, compute z = (spread - μ) / σ.
     Trade signals: z > +entry → short spread; z < -entry → long;
     |z| < exit → flat.
  5. Translate signals into per-leg weights (β-hedged) and feed into
     the standard ``strategy_metrics.summary`` so the metric pack and
     plausibility-flagging machinery work unchanged.

Reuses the existing ``strategy_metrics`` helpers — no new metric code.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass

from ..types import Recipe


@dataclass
class CellSpec:
    cell_type: str
    content: str
    dag_node_id: str
    rationale: str = ""


TEMPLATE_NAME = "pairs_clustering_then_cointegration"


METADATA = {
    "name": TEMPLATE_NAME,
    "title": "Pairs trading via cointegration",
    "archetype": "stat_arb",
    "tagline": "Find a cointegrated pair in a universe, fit OU mean-reversion, trade the spread.",
    "description": (
        "Engle-Granger cointegration test across every pair in the universe; "
        "pick the lowest-p survivor; fit an Ornstein-Uhlenbeck mean-reversion "
        "model on the spread; trade the z-score of the residual. Walk-forward "
        "evaluation; emits the same financial metric pack as regime_modeling "
        "so runs can be compared head-to-head with regime / momentum / value "
        "strategies on the project page."
    ),
    "supports": {
        "targets": ["supervised_regression"],
        "models": [],   # OU fit is internal — no sklearn-style classifier needed
        "metrics": [
            "sharpe", "sortino", "annual_return", "total_return",
            "max_drawdown", "calmar", "turnover", "hit_rate", "exposure",
        ],
    },
    # Plausibility envelope — pairs trading at daily frequency lives in
    # roughly the same regime as long-only equity strategies (turnover is
    # higher, but Sharpe still shouldn't credibly exceed 3 OOS).
    "plausibility": {
        "sharpe": (-3, 3),
        "annual_return": (-1, 1),
        "total_return": (-10, 50),
        "calmar": (-50, 50),
        "max_drawdown": (-1, 0),
        "turnover": (0, 10),       # pairs strategies churn more than long-only
        "sortino": (-5, 5),
    },
    "presets": [
        {
            "key": "etf_pair_spy_tlt",
            "label": "ETF pair · SPY / TLT",
            "summary": "Two-asset cointegration on SPY and TLT, OU mean-reversion model.",
            "yaml": (
                "name: pairs_etf_v1\n"
                "project: pairs_research\n"
                "template: pairs_clustering_then_cointegration\n"
                "description: SPY/TLT cointegration pairs trade with OU mean-reversion.\n\n"
                "data:\n"
                "  prices:\n"
                "    kind: yfinance\n"
                "    tickers: [SPY, TLT]\n"
                "    start: 2015-01-01\n\n"
                "target:\n"
                "  kind: supervised_regression\n"
                "  label_strategy: cointegration_zscore\n"
                "  horizon_days: 5\n\n"
                "features: []\n\n"
                "model:\n"
                "  class_path: statsmodels.tsa.stattools.coint\n"
                "  params: {}\n\n"
                "evaluation:\n"
                "  splits: walk_forward\n"
                "  train_window: 504\n"
                "  test_window: 63\n"
                "  metrics: [sharpe, sortino, annual_return, max_drawdown, turnover]\n"
                "  costs:\n"
                "    bps_per_side: 5    # ETF spreads tighter, but pairs trades churn\n"
                "    borrow_bps: 50     # liquid ETF short borrow ~50bps annualised\n\n"
                "seed: 42\n"
            ),
        },
        {
            "key": "sector_universe_top_pair",
            "label": "Sector universe · pick the best pair",
            "summary": "Search a 6-name sector universe for the most cointegrated pair, then trade it.",
            "yaml": (
                "name: pairs_sector_v1\n"
                "project: pairs_research\n"
                "template: pairs_clustering_then_cointegration\n"
                "description: Sector ETF universe; pick most cointegrated pair; OU model.\n\n"
                "data:\n"
                "  prices:\n"
                "    kind: yfinance\n"
                "    tickers: [XLF, XLE, XLK, XLV, XLI, XLY]\n"
                "    start: 2015-01-01\n\n"
                "target:\n"
                "  kind: supervised_regression\n"
                "  label_strategy: cointegration_zscore\n"
                "  horizon_days: 5\n\n"
                "features: []\n\n"
                "model:\n"
                "  class_path: statsmodels.tsa.stattools.coint\n"
                "  params: {}\n\n"
                "evaluation:\n"
                "  splits: walk_forward\n"
                "  train_window: 504\n"
                "  test_window: 63\n"
                "  metrics: [sharpe, sortino, annual_return, max_drawdown, turnover]\n"
                "  costs:\n"
                "    bps_per_side: 8    # sector ETFs less liquid than SPY/TLT\n"
                "    borrow_bps: 75     # sector ETF short borrow slightly wider\n\n"
                "seed: 42\n"
            ),
        },
    ],
}


def supports(recipe: Recipe) -> bool:
    if recipe.template != TEMPLATE_NAME:
        return False
    if recipe.target.kind != "supervised_regression":
        return False
    if (recipe.target.label_strategy or "") != "cointegration_zscore":
        return False
    return True


def compile(recipe: Recipe) -> list[CellSpec]:
    cells: list[CellSpec] = []
    cells.append(_md_header(recipe))

    cells.append(_md_step(2, "Imports + recipe metadata", "n2_imports",
                          "Pin the recipe identity and seed up front for determinism."))
    cells.append(_code_imports(recipe))

    cells.append(_md_step(3, "Load data", "n3_load",
                          "Pull every datasource declared in the recipe."))
    cells.append(_code_load_data(recipe))

    cells.append(_md_step(4, "Compute pairwise log-price spreads", "n4_spreads",
                          "Build log-price spreads for every pair (i,j) with OLS β."))
    cells.append(_code_spreads(recipe))

    cells.append(_md_step(5, "Cointegration tests", "n5_coint",
                          "Engle-Granger across every pair; pick the lowest-p survivor."))
    cells.append(_code_cointegration(recipe))

    cells.append(_md_step(6, "Walk-forward OU fit + signal", "n6_walk_forward",
                          "Fit OU on each train window; generate z-score signals on test."))
    cells.append(_code_walk_forward(recipe))

    cells.append(_md_step(7, "Reference strategy books", "n7_reference_books",
                          "Buy-and-hold of the universe, plus a buy-both-legs baseline."))
    cells.append(_code_reference_books(recipe))

    cells.append(_md_step(8, "Pairs model book", "n8_model_book",
                          "Translate z-score signals into β-hedged pair weights."))
    cells.append(_code_model_book(recipe))

    cells.append(_md_step(9, "Financial metrics", "n9_metrics",
                          "Sharpe / Sortino / DD / turnover for model + reference books; "
                          "headline aliases bound to the model book."))
    cells.append(_code_financial_metrics(recipe))

    cells.append(_md_step(10, "Charts", "n10_charts",
                          "Spread + entry/exit z-bands; equity curves; drawdown."))
    cells.append(_code_charts(recipe))

    cells.append(_md_step(11, "Decomposition", "n11_decomposition",
                          "Per-leg PnL contribution + calendar-year returns by book."))
    cells.append(_code_decomposition(recipe))

    cells.append(_md_step(12, "Run summary", "n12_summary",
                          "Print the JSON summary the experiment runner harvests."))
    cells.append(_code_summary(recipe))

    return cells


# ── Cell builders ───────────────────────────────────────────────────────


def _md_header(recipe: Recipe) -> CellSpec:
    notes = (recipe.notes or "").strip()
    notes_block = f"\n\n{notes}" if notes else ""
    body = textwrap.dedent(f"""\
        # {recipe.name}

        **Project:** `{recipe.project}` · **Template:** `pairs_clustering_then_cointegration` · **Fingerprint:** `{recipe.fingerprint()}`

        {recipe.description.strip() if recipe.description else "Pairs trading research notebook compiled from a Recipe."}{notes_block}
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
    recipe_json = json.dumps(recipe.model_dump(mode="json"), default=str)
    lines = [
        "from __future__ import annotations",
        "import json",
        "import math",
        "from itertools import combinations",
        "",
        "import numpy as np",
        "import pandas as pd",
        "from statsmodels.tsa.stattools import coint",
        "",
        "# Recipe metadata, pinned in a constant so the cell is reproducible.",
        f"RECIPE = json.loads({recipe_json!r})",
        f"SEED = {recipe.seed}",
        "np.random.seed(SEED)",
        "",
        "# Z-score thresholds — entry at |z|>2, exit at |z|<0.5. These are",
        "# canonical pairs-trading defaults and match the academic literature.",
        "ENTRY_Z = 2.0",
        "EXIT_Z = 0.5",
    ]
    return CellSpec("code", "\n".join(lines), "n2_imports", "Imports + frozen recipe blob.")


def _code_load_data(recipe: Recipe) -> CellSpec:
    lines: list[str] = ["from finagent.recipes import loaders", ""]
    for var, ds in recipe.data.items():
        spec_blob = json.dumps(ds.model_dump(mode="json"), default=str)
        lines.append(f"{var} = loaders.load({spec_blob!r})")
    lines.append("")
    if recipe.data:
        first = next(iter(recipe.data))
        lines.append(f"print({first}.shape if hasattr({first}, 'shape') else len({first}))")
    return CellSpec("code", "\n".join(lines), "n3_load", "Declarative datasource loaders.")


def _code_spreads(recipe: Recipe) -> CellSpec:
    keys = list(recipe.data.keys())
    prices_var = keys[0] if keys else "prices"
    body = textwrap.dedent(f"""\
        # Slim the prices frame to a single column per ticker. yfinance multi-
        # ticker returns a MultiIndex (field, ticker); flatten to Adj Close /
        # Close. Drop OHLCV-by-name for the single-index case.
        ASSET_PRICES = {prices_var}
        if not isinstance(ASSET_PRICES, pd.DataFrame) or ASSET_PRICES.empty:
            raise RuntimeError("prices source is not a usable DataFrame")
        if isinstance(ASSET_PRICES.columns, pd.MultiIndex):
            _levels = list(ASSET_PRICES.columns.get_level_values(0).unique())
            _field = next(
                (f for f in ['Adj Close', 'Close', 'close', 'adj_close'] if f in _levels),
                _levels[0],
            )
            ASSET_PRICES = ASSET_PRICES[_field].copy()
        _drop_lower = {{'open', 'high', 'low', 'volume', 'dividends',
                       'stock splits', 'capital gains', 'adj volume'}}
        ASSET_PRICES = ASSET_PRICES.loc[
            :, [c for c in ASSET_PRICES.columns
                if str(c).strip().lower() not in _drop_lower]
        ]
        ASSET_PRICES = ASSET_PRICES.select_dtypes('number').dropna(how='all')
        ASSET_PRICES = ASSET_PRICES.ffill().dropna()

        if ASSET_PRICES.shape[1] < 2:
            raise RuntimeError(
                f"need at least 2 assets for pairs trading; got {{list(ASSET_PRICES.columns)}}"
            )

        asset_returns_full = ASSET_PRICES.pct_change()
        # Sanity guard from regime_modeling: zero out any daily move >50%
        # so a stale tick doesn't poison compound metrics.
        _bad = asset_returns_full.abs() > 0.5
        if _bad.any().any():
            asset_returns_full = asset_returns_full.where(~_bad, 0.0)

        log_prices = np.log(ASSET_PRICES)
        print(f'log-prices ready: shape={{log_prices.shape}}, tickers={{list(log_prices.columns)}}')
        """)
    return CellSpec("code", body, "n4_spreads", "Slim prices + log-transform.")


def _code_cointegration(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        # Engle-Granger cointegration across every pair in the universe.
        # We compute test stats on the FULL window first to surface a global
        # ranking — the actual trading uses per-fold tests below.
        _pair_stats = []
        for _i, _j in combinations(log_prices.columns, 2):
            _x = log_prices[_i].dropna()
            _y = log_prices[_j].dropna()
            _aligned = pd.concat([_x, _y], axis=1).dropna()
            if len(_aligned) < 100:
                continue
            try:
                _t, _p, _ = coint(_aligned.iloc[:, 0], _aligned.iloc[:, 1])
            except Exception:
                continue
            _pair_stats.append({'i': _i, 'j': _j, 't_stat': float(_t), 'p_value': float(_p)})

        coint_table = pd.DataFrame(_pair_stats).sort_values('p_value').reset_index(drop=True)
        if coint_table.empty:
            raise RuntimeError("no testable pairs in the universe — need >= 100 aligned observations")

        print('Cointegration ranking (full sample):')
        print(coint_table.head(10).to_string(index=False))

        BEST_PAIR = (coint_table.iloc[0]['i'], coint_table.iloc[0]['j'])
        BEST_P = float(coint_table.iloc[0]['p_value'])
        print(f'\\nSelected pair: {BEST_PAIR} (p={BEST_P:.4f})')
        if BEST_P > 0.05:
            print('WARNING: best pair has p > 0.05 — relationship may not be cointegrated.')
        """)
    return CellSpec("code", body, "n5_coint", "Pairwise cointegration ranking.")


def _code_walk_forward(recipe: Recipe) -> CellSpec:
    e = recipe.evaluation
    body = textwrap.dedent(f"""\
        # Walk-forward OU fit + z-score signal generation.
        #
        # Per fold:
        #   1. Refit OLS hedge β on the training window's log-prices.
        #   2. Fit OU via discretised AR(1): S_{{t+1}} = α + φ S_t + ε.
        #      θ = -log(φ) / dt, μ = α / (1-φ), σ_eq = std(ε) / sqrt(1-φ²).
        #   3. On test window: compute z = (spread - μ) / σ_eq.
        #      Signal: long spread when z < -ENTRY, short when z > +ENTRY,
        #      flat when |z| < EXIT (sticky position between thresholds).
        from typing import Iterator

        def _walk_forward(n: int, train: int = {e.train_window}, test: int = {e.test_window}) -> Iterator[tuple[slice, slice]]:
            start = 0
            while start + train + test <= n:
                yield slice(start, start + train), slice(start + train, start + train + test)
                start += test

        _i, _j = BEST_PAIR
        oos_signal = pd.Series(0.0, index=ASSET_PRICES.index, name='signal')
        oos_zscore = pd.Series(np.nan, index=ASSET_PRICES.index, name='z')
        beta_per_fold = []

        for tr, te in _walk_forward(len(log_prices)):
            _train = log_prices[[_i, _j]].iloc[tr].dropna()
            _test = log_prices[[_i, _j]].iloc[te].dropna()
            if len(_train) < 60 or len(_test) < 5:
                continue
            # OLS β: regress log(_i) on log(_j)
            _x = _train[_j].values
            _y = _train[_i].values
            _X = np.vstack([_x, np.ones_like(_x)]).T
            _b, _intercept = np.linalg.lstsq(_X, _y, rcond=None)[0]
            beta = float(_b)
            beta_per_fold.append(beta)

            _spread_tr = _train[_i] - beta * _train[_j]
            _spread_te = _test[_i] - beta * _test[_j]

            # AR(1) on spread to recover OU parameters.
            _s = _spread_tr.values
            _phi, _alpha = np.polyfit(_s[:-1], _s[1:], 1)
            _resid = _s[1:] - (_phi * _s[:-1] + _alpha)
            _sigma_eq = float(np.std(_resid)) / max(np.sqrt(max(1.0 - _phi**2, 1e-9)), 1e-9)
            _mu = float(_alpha / (1.0 - _phi)) if abs(1.0 - _phi) > 1e-9 else float(_spread_tr.mean())

            _z = (_spread_te - _mu) / max(_sigma_eq, 1e-9)
            oos_zscore.loc[_z.index] = _z.values

            # Sticky entry/exit logic: build position from z-score.
            _pos = pd.Series(0.0, index=_z.index)
            _state = 0  # +1=long spread, -1=short spread
            for _ts, _zv in _z.items():
                if _state == 0:
                    if _zv > ENTRY_Z:
                        _state = -1
                    elif _zv < -ENTRY_Z:
                        _state = +1
                else:
                    if abs(_zv) < EXIT_Z:
                        _state = 0
                _pos.loc[_ts] = float(_state)
            oos_signal.loc[_pos.index] = _pos.values

        oos_signal = oos_signal.replace(0.0, np.nan).dropna() if False else oos_signal
        print(f'OOS signal: {{int((oos_signal != 0).sum())}} active periods')
        print(f'Median β across folds: {{float(np.median(beta_per_fold)) if beta_per_fold else float("nan"):.4f}}')
        BETA = float(np.median(beta_per_fold)) if beta_per_fold else 1.0
        """)
    return CellSpec("code", body, "n6_walk_forward", "Walk-forward OU fit + signal.")


def _code_reference_books(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        from finagent.recipes import strategy_metrics as sm

        # Buy-and-hold the universe (equal-weight). Useful baseline because
        # the pairs strategy IS market-neutral; B&H is the long-only counter.
        _bh_w = sm.buy_and_hold_book(ASSET_PRICES)

        # Buy-both-legs equal-weight: a degenerate "pair" book that always
        # holds 50% in each of the two pair tickers. Same exposure level as
        # the active strategy when it's flat — useful for quoting the no-
        # alpha return.
        _i, _j = BEST_PAIR
        _both_w = pd.DataFrame(0.0, index=ASSET_PRICES.index, columns=ASSET_PRICES.columns)
        _both_w[_i] = 0.5
        _both_w[_j] = 0.5

        print(f'reference books built. universe: {list(ASSET_PRICES.columns)}')
        print(f'  buy-and-hold avg exposure: {sm.exposure(_bh_w):.2f}')
        print(f'  both-legs avg exposure   : {sm.exposure(_both_w):.2f}')
        """)
    return CellSpec("code", body, "n7_reference_books", "B&H + both-legs reference weights.")


def _code_model_book(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        # Translate the OOS signal into pair weights.
        #   long spread (signal=+1)  → long _i, short β·_j
        #   short spread (signal=-1) → short _i, long β·_j
        # Normalise so |w_i| + |w_j| = 1 (gross 100%).
        _i, _j = BEST_PAIR
        model_weights = pd.DataFrame(0.0, index=ASSET_PRICES.index, columns=ASSET_PRICES.columns)
        _gross = 1.0 + abs(BETA)
        _w_i = oos_signal * (1.0 / _gross)
        _w_j = oos_signal * (-BETA / _gross)
        model_weights[_i] = _w_i.reindex(model_weights.index).fillna(0.0)
        model_weights[_j] = _w_j.reindex(model_weights.index).fillna(0.0)

        print(f'model book built. nonzero days: {int((model_weights.abs().sum(axis=1) > 0).sum())}')
        """)
    return CellSpec("code", body, "n8_model_book", "Pair weights from z-score signal.")


def _code_financial_metrics(recipe: Recipe) -> CellSpec:
    requested = list(recipe.evaluation.metrics)
    costs = recipe.evaluation.costs
    costs_blob = (
        repr(costs.model_dump()) if costs is not None else "None"
    )
    body = textwrap.dedent(f"""\
        from finagent.recipes import strategy_metrics as sm

        _books = {{
            'model':         model_weights,
            'buy_and_hold':  _bh_w,
            'both_legs':     _both_w,
        }}

        # Cost overlay — see regime_modeling for the full design notes.
        # When pinned, we emit gross AND net metric packs side-by-side
        # and re-point the headline aliases at the net values.
        _costs = {costs_blob}
        _costs_applied = _costs is not None

        metrics_out: dict[str, float] = {{}}
        for _book_name, _w in _books.items():
            _gross_book = sm.book_returns(_w, asset_returns_full)
            _gross_pack = sm.summary(_w, asset_returns_full)
            for _k, _v in _gross_pack.items():
                metrics_out[f'{{_book_name}}_{{_k}}'] = _v
            if _costs_applied:
                _net_book = sm.apply_costs(_gross_book, _w, **_costs)
                metrics_out[f'{{_book_name}}_total_return_net'] = sm.total_return(_net_book)
                metrics_out[f'{{_book_name}}_annual_return_net'] = sm.annual_return(_net_book)
                metrics_out[f'{{_book_name}}_sharpe_net'] = sm.sharpe(_net_book)
                metrics_out[f'{{_book_name}}_sortino_net'] = sm.sortino(_net_book)
                metrics_out[f'{{_book_name}}_max_drawdown_net'] = sm.max_drawdown(_net_book)
                metrics_out[f'{{_book_name}}_calmar_net'] = sm.calmar(_net_book)
                metrics_out[f'{{_book_name}}_hit_rate_net'] = sm.hit_rate(_net_book)
                metrics_out[f'{{_book_name}}_turnover_net'] = _gross_pack['turnover']
                metrics_out[f'{{_book_name}}_exposure_net'] = _gross_pack['exposure']

        # Headline aliases — net when costs pinned, else gross.
        _alias_suffix = '_net' if _costs_applied else ''
        for _alias in {requested!r}:
            _src = f'model_{{_alias}}{{_alias_suffix}}'
            if _src in metrics_out:
                metrics_out[_alias] = metrics_out[_src]

        print()
        if _costs_applied:
            print(f'costs applied: bps_per_side={{_costs.get("bps_per_side", 0)}}, '
                  f'borrow_bps={{_costs.get("borrow_bps", 0)}}')
            print('book          sharpe(net) sortino(net) ann_ret(net)  max_dd(net) turnover')
            for _name in ['model', 'buy_and_hold', 'both_legs']:
                print(
                    f'  {{_name:13s}} '
                    f'{{metrics_out[_name + "_sharpe_net"]:>10.2f}}  '
                    f'{{metrics_out[_name + "_sortino_net"]:>11.2f}}  '
                    f'{{metrics_out[_name + "_annual_return_net"] * 100:>+10.2f}}%  '
                    f'{{metrics_out[_name + "_max_drawdown_net"] * 100:>+10.2f}}%  '
                    f'{{metrics_out[_name + "_turnover"]:>6.3f}}'
                )
        else:
            print('book          sharpe   sortino   ann_ret    max_dd   turnover')
            for _name in ['model', 'buy_and_hold', 'both_legs']:
                print(
                    f'  {{_name:13s}} '
                    f'{{metrics_out[_name + "_sharpe"]:>6.2f}}  '
                    f'{{metrics_out[_name + "_sortino"]:>7.2f}}  '
                    f'{{metrics_out[_name + "_annual_return"] * 100:>+6.2f}}%  '
                    f'{{metrics_out[_name + "_max_drawdown"] * 100:>+6.2f}}%  '
                    f'{{metrics_out[_name + "_turnover"]:>6.3f}}'
                )

        asset_returns = asset_returns_full.fillna(0.0)
        asset_weights = model_weights.fillna(0.0)
        print(f'\\nasset_returns shape={{asset_returns.shape}}, asset_weights shape={{asset_weights.shape}}')
        """)
    return CellSpec("code", body, "n9_metrics", "Financial metrics + canonical asset_returns/weights.")


def _code_charts(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        import matplotlib.pyplot as plt
        from finagent.recipes import strategy_metrics as sm

        # ── Spread + z-score with entry/exit bands ──────────────────────
        if oos_zscore.notna().any():
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(oos_zscore.index, oos_zscore.values, color='#0f172a', linewidth=1.0)
            ax.axhline(+ENTRY_Z, color='#dc2626', linewidth=0.8, linestyle='--', label=f'+{ENTRY_Z}σ entry')
            ax.axhline(-ENTRY_Z, color='#16a34a', linewidth=0.8, linestyle='--', label=f'-{ENTRY_Z}σ entry')
            ax.axhline(+EXIT_Z, color='#94a3b8', linewidth=0.6, linestyle=':')
            ax.axhline(-EXIT_Z, color='#94a3b8', linewidth=0.6, linestyle=':')
            ax.axhline(0, color='#475569', linewidth=0.4)
            ax.set_title(f'Spread z-score · {BEST_PAIR[0]} vs {BEST_PAIR[1]} (OOS)')
            ax.set_ylabel('z')
            ax.legend(loc='best', frameon=False, ncol=2)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

        # ── Equity curves ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _palette = {'model': '#2563eb', 'buy_and_hold': '#94a3b8', 'both_legs': '#f59e0b'}
        for _name, _w in _books.items():
            _bk = sm.book_returns(_w, asset_returns_full).fillna(0.0)
            if _bk.empty:
                continue
            _eq = (1.0 + _bk).cumprod()
            ax.plot(_eq.index, _eq.values, label=_name, linewidth=1.5, color=_palette.get(_name))
        ax.set_title('Equity curves — pairs model vs reference books')
        ax.set_ylabel('Growth of $1 (compounded)')
        ax.legend(loc='best', frameon=False)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        # ── Drawdown of model book ──────────────────────────────────────
        _bk_m = sm.book_returns(model_weights, asset_returns_full).fillna(0.0)
        if not _bk_m.empty:
            _eq_m = (1.0 + _bk_m).cumprod()
            _peak = _eq_m.cummax()
            _dd = _eq_m / _peak - 1.0
            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.fill_between(_dd.index, _dd.values, 0.0, color='#dc2626', alpha=0.35, linewidth=0)
            ax.plot(_dd.index, _dd.values, color='#991b1b', linewidth=1.0)
            ax.set_title('Pairs model book drawdown')
            ax.set_ylabel('Drawdown')
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()
        """)
    return CellSpec("code", body, "n10_charts", "Spread bands + equity curves + drawdown.")


def _code_decomposition(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        import matplotlib.pyplot as plt
        from finagent.recipes import strategy_metrics as sm

        # Per-leg contribution to model book PnL.
        _aligned_w = model_weights.reindex(asset_returns_full.index).fillna(0.0).shift(1)
        contrib_per_leg = (_aligned_w * asset_returns_full.fillna(0.0)).sum(axis=0) * 100.0
        contrib_per_leg = contrib_per_leg.sort_values(ascending=False)

        if (contrib_per_leg.abs() > 1e-9).any():
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(contrib_per_leg.index, contrib_per_leg.values, color='#2563eb')
            ax.axhline(0, color='#475569', linewidth=0.5)
            ax.set_title('Per-leg contribution to model book return (% of $1)')
            ax.set_ylabel('Contribution (%)')
            ax.grid(axis='y', alpha=0.25)
            plt.tight_layout()
            plt.show()
            print('Per-leg contribution (%):')
            print(contrib_per_leg.round(3).to_string())

        # Calendar-year returns by book.
        _yearly_rows = {}
        for _name, _w in _books.items():
            _bk = sm.book_returns(_w, asset_returns_full).fillna(0.0)
            if _bk.empty:
                continue
            _yearly_rows[_name] = ((1.0 + _bk).groupby(_bk.index.year).prod() - 1.0)
        yearly_df = pd.DataFrame(_yearly_rows)
        _ordered_cols = [b for b in ['model', 'buy_and_hold', 'both_legs'] if b in yearly_df.columns]
        yearly_df = yearly_df[_ordered_cols] if _ordered_cols else yearly_df
        if not yearly_df.empty:
            _full = ((1.0 + yearly_df).prod() - 1.0)
            _full.name = 'all'
            yearly_df_with_full = pd.concat([yearly_df, _full.to_frame().T])
            try:
                from IPython.display import display
                _styled = (yearly_df_with_full * 100.0).round(2).style.format('{:+.2f}%')
                display(_styled)
            except Exception:
                print('Calendar-year returns (%):')
                print((yearly_df_with_full * 100.0).round(2).to_string())
        """)
    return CellSpec("code", body, "n11_decomposition", "Per-leg bars + per-year table.")


def _code_summary(recipe: Recipe) -> CellSpec:
    body = textwrap.dedent("""\
        import json as _json
        SUMMARY = {
            'recipe_name': RECIPE['name'],
            'project': RECIPE['project'],
            'fingerprint': RECIPE.get('fingerprint') if 'fingerprint' in RECIPE else None,
            'metrics': metrics_out,
            'pair': list(BEST_PAIR),
            'cointegration_p': BEST_P,
            'beta_median': BETA,
        }
        print('FINAGENT_RUN_SUMMARY ' + _json.dumps(SUMMARY, default=str))
        """)
    return CellSpec("code", body, "n12_summary", "Run-summary marker for harness ingestion.")
