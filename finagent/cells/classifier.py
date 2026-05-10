"""Heuristic AST-based cell-role classifier.

Tags each notebook code cell with one or more roles drawn from a small
controlled vocabulary. The vocabulary is shaped by the train/infer
split (Phase 4) — every role either runs only at training time, only
at inference time, or both. The classifier doesn't know which side a
role belongs to; the splitter (``finagent/cells/splitter.py``) makes
that call from the role tags.

Heuristics, not magic
---------------------
We deliberately use AST + name matching, not an LLM. Reasons:

  * Determinism — the same cell text always classifies the same way,
    so the splitter is reproducible.
  * Speed — a notebook with 30 cells classifies in < 5 ms.
  * Cost — running an LLM on every cell of every notebook would dwarf
    the cost of the actual research call.

Where the heuristic is too coarse (e.g. a custom function named
``do_everything()`` that hides the real semantics), the cell ends up
tagged as ``other`` and the splitter punts on it. The orchestrator
prompt is updated to discourage this pattern in the first place; a
follow-up LLM auto-splitter can clean up the long tail.

Role vocabulary
---------------

  imports        - only Import / ImportFrom statements
  data_load      - calls to fetch_*, load_*, *.read_csv, yfinance.download,
                   findata.* fetchers, panel.load_signal, ...
  preprocess     - dataframe transforms (dropna, fillna, shift, rolling,
                   resample, merge, groupby, scale, standardise)
  train          - .fit(), OLS(), GBM/XGB/LGB constructors followed by .fit
  eval           - .predict(), .score(), summary tables, OOS metrics
  signal_export  - panel.export_signal(...) / panel.save_model(...)
  chart          - matplotlib / plt.* / mplfinance / fig.*
  summary        - the FINAGENT_RUN_SUMMARY print line
  other          - anything else (helper functions, glue code)

A cell can have multiple roles. ``needs_split`` returns True when the
roles span incompatible groups (e.g. data_load + train, or train + chart).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Iterable

import nbformat


logger = logging.getLogger(__name__)


ROLES = (
    "imports",
    "data_load",
    "preprocess",
    "train",
    "eval",
    "signal_export",
    "chart",
    "summary",
    "other",
)


# Roles that are SAFE to combine in a single cell. Any role pair NOT in
# this allowlist is flagged as needing a split.
_COMPATIBLE: dict[str, set[str]] = {
    "imports": {"imports"},
    "data_load": {"data_load", "preprocess"},        # fetch + immediate filter is fine
    "preprocess": {"preprocess", "data_load"},       # mirror above
    "train": {"train"},                              # train alone — never with chart/eval
    "eval": {"eval", "chart"},                       # eval that prints + plots is fine
    "chart": {"chart", "eval"},                      # chart that uses an eval result is fine
    "signal_export": {"signal_export"},              # always alone — single side-effect
    "summary": {"summary"},                          # always alone
    "other": {"other"},                              # punt
}


# Function-call substring matchers — chosen to be specific enough that
# false positives are rare. Order matters: more specific first.
_DATA_LOAD_FUNCS = (
    "panel.load_signal", "panel.load_model",
    "fetch_ohlcv", "fetch_news", "fetch_factor_loadings", "fetch_fundamentals",
    "fetch_world_themes", "fetch_sector_exposure",
    "yfinance.download", "yf.download",
    "pd.read_csv", "pd.read_parquet", "pd.read_excel", "read_sql",
    "findata.", "load_data", "load_dataset",
)

_PREPROCESS_FUNCS = (
    ".dropna", ".fillna", ".shift", ".rolling", ".resample",
    ".merge", ".groupby", ".pct_change", ".diff",
    "StandardScaler", "MinMaxScaler", "RobustScaler",
    ".pivot", ".melt", ".stack", ".unstack",
    "to_datetime",
)

_TRAIN_FUNCS = (
    ".fit(", ".fit_transform",
    "OLS(", "WLS(", "GLS(", "GLM(",
    "GradientBoosting", "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "RandomForest", "LogisticRegression", "LinearRegression",
    "KMeans", "GaussianMixture", "HMM",
    "curve_fit", "minimize",
)

_EVAL_FUNCS = (
    ".predict(", ".predict_proba", ".score(",
    ".summary(",
    "sharpe", "sortino", "max_drawdown", "calmar",
    "book_returns", "buy_and_hold_book", "summary_metrics",
    "mean_squared_error", "r2_score", "accuracy_score",
)

_SIGNAL_EXPORT_FUNCS = (
    "panel.export_signal", "panel.save_model",
    "export_signal(", "save_model(",
)

_CHART_FUNCS = (
    "plt.", "fig.", "ax.",
    "matplotlib.", "mplfinance.", "mpf.",
    ".plot(", ".hist(", ".scatter(", ".bar(", ".pie(",
    "savefig",
)


def _all_calls(tree: ast.AST) -> list[str]:
    """Best-effort string-render of every Call node's func attribute."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                out.append(ast.unparse(node.func))
            except Exception:
                pass
        elif isinstance(node, ast.Attribute):
            try:
                out.append(ast.unparse(node))
            except Exception:
                pass
    return out


def _has_substr(haystack: Iterable[str], needles: Iterable[str]) -> bool:
    return any(n in s for s in haystack for n in needles)


def classify_cell(source: str) -> set[str]:
    """Return the set of roles detected in ``source``.

    Empty set ⇒ blank cell. ``{"other"}`` ⇒ the heuristics couldn't
    place it (custom helper, glue code, comments-only). Multi-element
    sets are common and not always bad — see ``needs_split``.
    """
    source = (source or "").strip()
    if not source:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Cell has a syntax error — let the executor surface that;
        # don't pretend to classify it.
        logger.debug("classify_cell: syntax error, returning {'other'}")
        return {"other"}

    # Imports-only is the easiest signal.
    if all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body):
        return {"imports"}

    calls = _all_calls(tree)
    src = source  # raw source for substring scans the AST misses
    roles: set[str] = set()

    # FINAGENT_RUN_SUMMARY is a string literal — scan source, not AST.
    if "FINAGENT_RUN_SUMMARY" in src:
        roles.add("summary")
    if _has_substr(calls, _SIGNAL_EXPORT_FUNCS) or _has_substr([src], _SIGNAL_EXPORT_FUNCS):
        roles.add("signal_export")
    if _has_substr(calls, _DATA_LOAD_FUNCS) or _has_substr([src], _DATA_LOAD_FUNCS):
        roles.add("data_load")
    if _has_substr(calls, _PREPROCESS_FUNCS) or _has_substr([src], _PREPROCESS_FUNCS):
        roles.add("preprocess")
    if _has_substr(calls, _TRAIN_FUNCS) or _has_substr([src], _TRAIN_FUNCS):
        roles.add("train")
    if _has_substr(calls, _EVAL_FUNCS) or _has_substr([src], _EVAL_FUNCS):
        roles.add("eval")
    if _has_substr(calls, _CHART_FUNCS) or _has_substr([src], _CHART_FUNCS):
        roles.add("chart")

    # NOTE: we do NOT add "imports" when the cell is a mix of imports +
    # other code. The orchestrator is told to consolidate imports into
    # their own cell ("ONE ROLE PER CELL"), so a stray inline import
    # shouldn't promote a chart cell to "imports" and route it into
    # train.py. The pure-imports case is handled at the top of this
    # function with the `all(isinstance(...) for n in tree.body)` check.

    if not roles:
        roles.add("other")
    return roles


def needs_split(source: str) -> bool:
    """True when ``source`` mixes incompatible roles.

    ``{imports}`` alone, ``{data_load, preprocess}``, ``{eval, chart}``
    are all fine. ``{data_load, train}``, ``{train, chart}``,
    ``{train, signal_export}`` are not — those mix train-time and
    inference-time concerns or would force re-fetching at inference.
    """
    roles = classify_cell(source)
    if len(roles) <= 1:
        return False
    # All roles must be in EVERY OTHER role's compatibility set.
    for r in roles:
        compat = _COMPATIBLE.get(r, {r})
        if not (roles - {r}).issubset(compat):
            return True
    return False


def tag_notebook(path: Path) -> dict[str, list[str]]:
    """Walk a notebook, stamp each code cell's ``metadata.finagent.roles``
    with its detected roles, and return a ``{cell_index: [roles]}`` map.

    Idempotent: running twice produces the same metadata.
    """
    nb = nbformat.read(str(path), as_version=4)
    out: dict[str, list[str]] = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        roles = sorted(classify_cell(cell.source))
        meta = cell.metadata.setdefault("finagent", {})
        meta["roles"] = roles
        meta["needs_split"] = needs_split(cell.source)
        out[str(i)] = roles
    nbformat.write(nb, str(path))
    return out
