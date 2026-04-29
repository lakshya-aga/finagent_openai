"""Pydantic schema for research recipes.

A Recipe is the durable artifact — the notebook is one rendered view of it.
Two recipes that share a project name + template form a sweep; their
notebooks are comparable on the Project page.

Schema design notes:
  - Field-level validators reject impossible combinations early (e.g. an
    unsupervised target with a classification metric) so users see the
    error in the recipe editor, not 30s into a kernel run.
  - `template` selects a deterministic compiler path; if absent, the
    workflow falls back to the AI planner with the recipe rendered as
    a structured prompt.
  - Free-form `notes` field lets researchers narrate intent for future
    review.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Data ────────────────────────────────────────────────────────────────


class DataSource(BaseModel):
    """One input dataset. The `kind` discriminates the loader path."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["yfinance", "fred", "fin_kit", "csv", "fama_french", "cboe", "coingecko", "binance"]
    # All other fields are kind-specific; the loader knows what to look for.
    # Examples:
    #   yfinance:  tickers (list[str]), start (date), end (date), interval
    #   fred:      series_ids (list[str]), start, end
    #   csv:       path (str), date_column, ticker_column
    #   fin_kit:   function (str), kwargs (dict)


# ── Target ──────────────────────────────────────────────────────────────


TargetKind = Literal[
    "unsupervised_regime",
    "supervised_classification",
    "supervised_regression",
    "binary_event",
]


class Target(BaseModel):
    """What the model is trying to predict / discover."""

    kind: TargetKind
    # Common knobs across kinds — only some are meaningful per kind, the
    # validator below catches obvious mismatches.
    method: Optional[str] = None  # e.g. "hmm", "kmeans", "future_return_sign"
    n_states: Optional[int] = Field(None, ge=2, le=10)
    horizon_days: Optional[int] = Field(None, ge=1, le=63)
    label_strategy: Optional[Literal[
        "next_return_sign", "future_return_sign", "triple_barrier", "vol_quantile",
    ]] = None
    threshold: Optional[float] = None  # e.g. for triple_barrier or vol_quantile

    @model_validator(mode="after")
    def _coherent(self) -> "Target":
        if self.kind == "unsupervised_regime":
            if self.n_states is None:
                raise ValueError("unsupervised_regime requires n_states")
        elif self.kind == "supervised_classification":
            if self.label_strategy is None:
                raise ValueError("supervised_classification requires label_strategy")
            if self.horizon_days is None:
                raise ValueError("supervised_classification requires horizon_days")
        elif self.kind == "supervised_regression":
            if self.horizon_days is None:
                raise ValueError("supervised_regression requires horizon_days")
        return self


# ── Features ────────────────────────────────────────────────────────────


class Feature(BaseModel):
    """One feature engineering step. `name` references a known builder; the
    template / compiler maps name → function. Unknown names fall back to AI.
    """

    model_config = ConfigDict(extra="allow")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# ── Model ───────────────────────────────────────────────────────────────


class ModelSpec(BaseModel):
    """Fully-qualified class + kwargs."""

    class_path: str = Field(
        ..., description="Dotted path, e.g. 'hmmlearn.GaussianHMM'"
    )
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def _has_dot(cls, v: str) -> str:
        if "." not in v:
            raise ValueError("class_path must be dotted (module.ClassName)")
        return v


# ── Evaluation ──────────────────────────────────────────────────────────


SplitKind = Literal["walk_forward", "expanding_window", "kfold", "holdout"]


class Evaluation(BaseModel):
    splits: SplitKind
    train_window: Optional[int] = Field(None, ge=20, description="bars in each train fold")
    test_window: Optional[int] = Field(None, ge=1, description="bars in each test fold")
    n_folds: Optional[int] = Field(None, ge=2, le=20)
    metrics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> "Evaluation":
        needs_windows = self.splits in ("walk_forward", "expanding_window")
        if needs_windows and (self.train_window is None or self.test_window is None):
            raise ValueError(f"{self.splits} requires train_window and test_window")
        if self.splits == "kfold" and self.n_folds is None:
            raise ValueError("kfold requires n_folds")
        if not self.metrics:
            raise ValueError("at least one metric must be requested")
        return self


# ── Recipe ──────────────────────────────────────────────────────────────


class Recipe(BaseModel):
    """A reusable research run definition."""

    name: str = Field(..., min_length=1, max_length=80)
    project: str = Field(..., min_length=1, max_length=80)
    template: Optional[str] = Field(None, description="e.g. 'regime_modeling'")
    description: str = ""
    notes: str = ""
    seed: int = 42

    data: dict[str, DataSource]
    target: Target
    features: list[Feature]
    model: ModelSpec
    evaluation: Evaluation

    @field_validator("name", "project")
    @classmethod
    def _slug(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    def fingerprint(self) -> str:
        """Deterministic content hash — same recipe, same hash, same expected
        notebook. Used for caching + drift detection."""
        payload = self.model_dump(mode="json")
        # Sort keys so insertion order doesn't change the hash.
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def recipe_from_yaml(text: str) -> Recipe:
    """Parse a YAML string into a Recipe with full validation."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("recipe YAML must be a mapping at the top level")
    return Recipe.model_validate(data)
