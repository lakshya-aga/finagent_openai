"""Hypothesis policies. A policy turns history → next mutation dict.

Two policies are baseline-tier and ship together:

  random  — uniform sample per dimension. Good when the space has many
            dimensions, or you want a no-bias starting point. Even random
            search is the right answer if your fancy LLM policy can't beat
            it on this problem.
  grid    — Cartesian product. Exhaustive when the product fits the
            budget; pre-shuffled deterministically when it doesn't, so
            you still see broad coverage early.

Both are deterministic given the search seed — re-running a search with
the same seed reproduces the same iteration sequence. That's the trust
guarantee we need before we even contemplate adding LLM creativity.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from .types import (
    ChoiceDimension,
    Dimension,
    FloatDimension,
    IntDimension,
    SearchHistoryEntry,
    grid_iter,
)


class Policy(ABC):
    """Returns the next mutation dict, or None when exhausted."""

    @abstractmethod
    def propose(
        self, history: list[SearchHistoryEntry]
    ) -> Optional[dict[str, Any]]:
        ...


def _sample_one(rng: random.Random, dim: Dimension) -> Any:
    if isinstance(dim, IntDimension):
        lo, hi = dim.range
        if dim.step:
            options = dim.grid_values()
            return rng.choice(options)
        return rng.randint(lo, hi)
    if isinstance(dim, FloatDimension):
        lo, hi = dim.range
        if dim.step:
            return rng.choice(dim.grid_values())
        return rng.uniform(lo, hi)
    if isinstance(dim, ChoiceDimension):
        return rng.choice(dim.values)
    raise ValueError(f"unknown dimension type: {type(dim).__name__}")


class RandomPolicy(Policy):
    """Uniform per-dimension sample. Caller dedups by recipe fingerprint."""

    def __init__(self, space: list[Dimension], seed: int = 42) -> None:
        self.space = space
        self._rng = random.Random(seed)

    def propose(
        self, history: list[SearchHistoryEntry]
    ) -> Optional[dict[str, Any]]:
        return {dim.path: _sample_one(self._rng, dim) for dim in self.space}


class GridPolicy(Policy):
    """Cartesian product over declared values. Pre-shuffled with the seed
    so partial-budget runs still see a wide cross-section of cells.

    Continuous Float dimensions without a step are clamped to [lo, hi] —
    grid is meaningless without discretisation, but we don't reject the
    search outright; we just include the two endpoints.
    """

    def __init__(self, space: list[Dimension], seed: int = 42) -> None:
        self.space = space
        cells = list(grid_iter(space))
        random.Random(seed).shuffle(cells)
        self._cells = iter(cells)

    def propose(
        self, history: list[SearchHistoryEntry]
    ) -> Optional[dict[str, Any]]:
        return next(self._cells, None)


def make_policy(kind: str, space: list[Dimension], seed: int = 42) -> Policy:
    if kind == "random":
        return RandomPolicy(space, seed)
    if kind == "grid":
        return GridPolicy(space, seed)
    raise ValueError(f"unknown policy: {kind!r}")
