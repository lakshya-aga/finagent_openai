"""Pydantic shape for search artifacts.

A Search owns a base recipe, a search space, an objective, a budget, and
a hypothesis policy. The executor iterates: policy.propose(history) →
mutations → variant recipe → run_recipe() → metric → history. Stops when
the budget runs out or the policy returns nothing new.

The space DSL is intentionally boring: each dimension declares a recipe
field path, a type, and either a range (numeric) or a list of choices.
That's enough for sweeps over hyperparameters and structural toggles
without inventing a query language.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Iterator, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Search-space dimensions ─────────────────────────────────────────────


class IntDimension(BaseModel):
    type: Literal["int"] = "int"
    path: str
    range: tuple[int, int]
    step: Optional[int] = None  # grid step; defaults to 1

    @model_validator(mode="after")
    def _coherent(self) -> "IntDimension":
        lo, hi = self.range
        if hi < lo:
            raise ValueError(f"range[{self.path}]: max < min")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"step[{self.path}] must be positive")
        return self

    def grid_values(self) -> list[int]:
        lo, hi = self.range
        step = self.step or 1
        return list(range(lo, hi + 1, step))


class FloatDimension(BaseModel):
    type: Literal["float"] = "float"
    path: str
    range: tuple[float, float]
    step: Optional[float] = None  # grid step (numeric)

    @model_validator(mode="after")
    def _coherent(self) -> "FloatDimension":
        lo, hi = self.range
        if hi < lo:
            raise ValueError(f"range[{self.path}]: max < min")
        return self

    def grid_values(self) -> list[float]:
        lo, hi = self.range
        if self.step is None:
            # No step → can't grid; caller must use random sampling
            return [lo, hi]
        out: list[float] = []
        v = lo
        while v <= hi + 1e-12:
            out.append(round(v, 12))
            v += self.step
        return out


class ChoiceDimension(BaseModel):
    type: Literal["choice"] = "choice"
    path: str
    values: list[Any] = Field(..., min_length=1)

    def grid_values(self) -> list[Any]:
        return list(self.values)


Dimension = Union[IntDimension, FloatDimension, ChoiceDimension]


# ── Objective + Budget ──────────────────────────────────────────────────


class Objective(BaseModel):
    metric: str = Field(..., min_length=1)
    direction: Literal["max", "min"] = "max"

    def is_better(self, candidate: Optional[float], current_best: Optional[float]) -> bool:
        if candidate is None:
            return False
        if current_best is None:
            return True
        return (candidate > current_best) if self.direction == "max" else (candidate < current_best)


class Budget(BaseModel):
    max_runs: int = Field(10, ge=1, le=200)
    max_seconds: Optional[int] = Field(None, ge=10, description="wall clock cap")
    max_no_improvement: Optional[int] = Field(
        None,
        ge=1,
        description="early-stop after K consecutive runs without improvement",
    )


# ── Policy enum ─────────────────────────────────────────────────────────


PolicyKind = Literal["random", "grid"]


# ── Search submission + record ──────────────────────────────────────────


class SearchSubmission(BaseModel):
    """What the user POSTs to /api/searches."""

    base_recipe_yaml: str = Field(..., min_length=1)
    space: list[Dimension] = Field(..., min_length=1)
    objective: Objective
    budget: Budget
    policy: PolicyKind = "random"
    seed: int = 42
    notes: str = ""

    @field_validator("space")
    @classmethod
    def _unique_paths(cls, v: list[Dimension]) -> list[Dimension]:
        seen: set[str] = set()
        for dim in v:
            if dim.path in seen:
                raise ValueError(f"duplicate space path: {dim.path}")
            seen.add(dim.path)
        return v

    def grid_size(self) -> int:
        size = 1
        for dim in self.space:
            size *= max(1, len(dim.grid_values()))
            if size > 10_000:
                return size  # short-circuit; far past anything we'd run
        return size

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class SearchHistoryEntry(BaseModel):
    iteration: int
    run_id: str
    recipe_fingerprint: str
    mutations: dict[str, Any]
    metric: Optional[float] = None
    status: str
    error: Optional[str] = None


def grid_iter(space: list[Dimension]) -> Iterator[dict[str, Any]]:
    """Cartesian product over the declared space — yields one mutation dict per cell."""
    paths = [d.path for d in space]
    value_lists = [d.grid_values() for d in space]
    for combo in itertools.product(*value_lists):
        yield dict(zip(paths, combo))
