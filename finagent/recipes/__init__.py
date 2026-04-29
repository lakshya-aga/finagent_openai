"""Research recipes: declarative configs that drive the agent pipeline.

A `Recipe` pins what data, target, features, model and evaluation a run
should use. The compiler turns it into a notebook deterministically when
a matching template exists; otherwise the existing AI planner takes over.
"""

from .types import (
    DataSource,
    Evaluation,
    Feature,
    ModelSpec,
    Recipe,
    Target,
    recipe_from_yaml,
)
from .compiler import compile_recipe, available_templates

__all__ = [
    "DataSource",
    "Evaluation",
    "Feature",
    "ModelSpec",
    "Recipe",
    "Target",
    "available_templates",
    "compile_recipe",
    "recipe_from_yaml",
]
