"""CLI entry point — reproduce Table 4 from the paper.

Usage::

    python -m finagent.benchmarks.regime_dc                  # full sweep
    python -m finagent.benchmarks.regime_dc --classifier nbc # NBC only
    python -m finagent.benchmarks.regime_dc --quick          # tiny grid

Reference numbers (paper Table 4, S&P 500 test set 2020 → 2022):

    classifier         regime_profit  sharpe  mdd      opt(θ,ε,ind)
    naive_bayes        1.0824         1.21    0.2506   (0.01, 0.80, R)
    logistic_regression 0.7528        0.75    0.2138   (0.02, 0.60, T)
    svm                 0.7528        0.75    0.2138   (0.02, 0.60, T)

Re-runs with newer yfinance / hmmlearn drift by ~5-10% on profit
(hmmlearn is non-deterministic across point releases; yfinance ticks
shift slightly under back-adjustments). The headline finding — NBC
beats LR and SVM, and all three beat the static controls — is robust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .grid_search import Splits, run_benchmark

# Short names accepted on the CLI for ergonomics.
_CLF_ALIASES = {
    "nbc": "naive_bayes",
    "naive_bayes": "naive_bayes",
    "lr": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "svm": "svm",
}


def _parse_classifier(name: str) -> str:
    key = name.strip().lower()
    if key not in _CLF_ALIASES:
        raise argparse.ArgumentTypeError(
            f"unknown classifier {name!r}; expected one of {sorted(set(_CLF_ALIASES))}",
        )
    return _CLF_ALIASES[key]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m finagent.benchmarks.regime_dc",
        description="Reproduce the Chen-Tsang / Baid-et-al directional-change "
        "regime detection benchmark on S&P 500.",
    )
    ap.add_argument(
        "--classifier",
        "-c",
        type=_parse_classifier,
        action="append",
        default=None,
        help="Run only this classifier (repeatable). Default: run all three.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Tiny grid (θ=0.01, ε=0.7, ind=R) for smoke-testing the "
        "pipeline — runs 1 cell per classifier instead of 45.",
    )
    ap.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="If given, dump the Table 4 reproduction to this path as JSON.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cell progress prints.",
    )
    args = ap.parse_args(argv)

    classifiers = (
        tuple(args.classifier)
        if args.classifier
        else ("naive_bayes", "logistic_regression", "svm")
    )
    splits = Splits()

    if args.quick:
        from . import grid_search as gs

        # Monkey-patch the default grid for this run — keeps the API surface
        # narrow (we don't expose grid as a CLI flag because the paper's
        # 45-cell grid is the canonical benchmark).
        gs.DEFAULT_GRID = {
            "theta": [0.01],
            "dc_indicator": ["R"],
            "epsilon": [0.7],
        }

    result = run_benchmark(
        splits=splits,
        classifiers=classifiers,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 80)
    print("TABLE 4 REPRODUCTION  (S&P 500, test set 2020-01-01 → 2022-12-31)")
    print("=" * 80)
    table4 = result["table4"]
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(table4.to_string(index=False))

    print(
        "\nOptimal hyperparameters per classifier (validation-set profit-maximising):"
    )
    for c, p in result["optimal"].items():
        print(
            f"  {c:>20s}  θ={p.get('theta')}  ε={p.get('epsilon')}  ind={p.get('dc_indicator')}"
        )

    print(
        "\nReference (paper Table 4 + Table 3 — NBC: θ=0.01, ε=0.80, ind=R):\n"
        "  naive_bayes          regime_profit=1.0824  sharpe=1.21  mdd=0.2506\n"
        "  logistic_regression  regime_profit=0.7528  sharpe=0.75  mdd=0.2138  (θ=0.02, ε=0.60, T)\n"
        "  svm                  regime_profit=0.7528  sharpe=0.75  mdd=0.2138  (θ=0.02, ε=0.60, T)\n"
        "  mean-rev control     regime_profit=0.3710  sharpe=0.48  mdd=0.1543  (NBC params)\n"
        "  momentum control     regime_profit=0.3541  sharpe=0.52  mdd=0.1546  (NBC params)"
    )

    if args.save_json:
        payload = {
            "optimal": result["optimal"],
            "table4": result["table4"].to_dict(orient="records"),
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nSaved JSON to {args.save_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
