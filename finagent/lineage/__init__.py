"""Provenance / data-lineage extractors for generated notebooks.

Two methods, same output shape — wire either into the UI and let users compare.

  ast      — pure static analysis via stdlib `ast`. Fast, deterministic,
             no execution. Loses precision on chained calls and in-place
             mutation but covers ~90% of agent-generated notebooks.
  runtime  — runs the notebook in a fresh Python subprocess and diffs
             the global namespace per cell to attribute new/changed
             variables to that cell. Combined with AST-extracted call
             names, this gives runtime-confirmed lineage without
             needing a third-party tracer like lineapy.
"""

from .types import Lineage, LineageEdge, LineageNode
from .ast_extractor import extract_lineage_ast
from .runtime_extractor import extract_lineage_runtime

__all__ = [
    "Lineage",
    "LineageEdge",
    "LineageNode",
    "extract_lineage_ast",
    "extract_lineage_runtime",
]
