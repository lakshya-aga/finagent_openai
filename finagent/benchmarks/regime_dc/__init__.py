"""Directional-Change + HMM + Classifier regime benchmark.

Faithful reproduction of:

  Baid, D., Prasad, R., Vishnoi, S., & Sharma, U. (2023).
  *Detecting Market Regime Changes for Trading.*
  CMU 46-927 Machine Learning II project.

itself a re-implementation of:

  Chen, J., & Tsang, E. P. K. (2021).
  *Detecting Regime Change in Computational Finance:
   Data Science, Machine Learning and Algorithmic Trading.* CRC Press.

Why we keep this around:

  Every new regime-detection template that ships in
  ``finagent.recipes.templates`` should be able to beat — or at
  least match — the headline numbers in the paper's Table 4 on
  the S&P 500 test set (2020-01-01 → 2022-12-31). If a new approach
  can't outperform a 2-state Gaussian HMM on directional-change
  indicators with a naive Bayes classifier, it isn't worth a place
  in the template gallery.

  This package therefore exists as a callable reference, NOT as a
  recipe template — recipes are LLM/UI-facing, this is the
  load-bearing baseline that lives outside the recipe system.

Reference numbers (paper Table 4, NBC optimal: θ=0.01, ε=0.8, R):

  Regime-based:        Profit 1.0824   Sharpe 1.21   MDD 0.2506
  Mean-rev control:    Profit 0.3710   Sharpe 0.48   MDD 0.1543
  Momentum control:    Profit 0.3541   Sharpe 0.52   MDD 0.1546

Usage:

    from finagent.benchmarks.regime_dc import run_benchmark
    table = run_benchmark()   # returns a DataFrame mirroring Table 4

CLI:

    python -m finagent.benchmarks.regime_dc
"""

from .pipeline import Pipeline
from .grid_search import CustomCrossValidation, run_benchmark
from .strategy import metrics_summary

__all__ = [
    "Pipeline",
    "CustomCrossValidation",
    "run_benchmark",
    "metrics_summary",
]
