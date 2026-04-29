"""Template registry. Each module exposes `TEMPLATE_NAME`, `supports(recipe)`,
and `compile(recipe) -> list[CellSpec]`."""

from . import regime_modeling

REGISTRY = {
    regime_modeling.TEMPLATE_NAME: regime_modeling,
}

__all__ = ["REGISTRY", "regime_modeling"]
