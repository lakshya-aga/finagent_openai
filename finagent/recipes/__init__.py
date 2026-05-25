"""Research recipes: declarative configs that drive the agent pipeline.

A `Recipe` pins what data, target, features, model and evaluation a run
should use. The compiler turns it into a notebook deterministically when
a matching template exists; otherwise the existing AI planner takes over.
"""

from .compiler import available_templates, compile_recipe
from .types import (
    DataSource,
    Evaluation,
    Feature,
    ModelSpec,
    Recipe,
    Target,
    recipe_from_yaml,
)

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
