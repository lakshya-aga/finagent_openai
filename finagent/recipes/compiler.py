"""Recipe → notebook compiler.

If a template matches, render the recipe deterministically (no LLM call).
Otherwise the caller falls back to the existing AI planner. The compiler
itself doesn't write to disk — it returns a list of CellSpec items the
workflow runner stitches into a notebook through the same `add_cell`
infrastructure that AI-generated runs use.
"""

from __future__ import annotations

from .templates import REGISTRY
from .types import Recipe


def available_templates() -> list[str]:
    return list(REGISTRY.keys())


def compile_recipe(recipe: Recipe):
    """Return a list of CellSpec, or None if no template matches."""
    if recipe.template is None:
        return None
    template = REGISTRY.get(recipe.template)
    if template is None:
        raise ValueError(
            f"unknown template: {recipe.template!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    if not template.supports(recipe):
        raise ValueError(
            f"template {recipe.template!r} does not support this recipe shape "
            f"(target.kind={recipe.target.kind}, model={recipe.model.class_path})"
        )
    return template.compile(recipe)
