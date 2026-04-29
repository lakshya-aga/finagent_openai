"""Shared lineage shape so AST + runtime extractors emit comparable graphs."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


NodeKind = Literal["data", "call", "input", "output"]


class LineageNode(TypedDict, total=False):
    id: str
    label: str
    kind: NodeKind
    cell_idx: Optional[int]
    details: Optional[str]


class LineageEdge(TypedDict, total=False):
    id: str
    src: str
    dst: str
    label: Optional[str]


class Lineage(TypedDict, total=False):
    method: str          # "ast" | "runtime"
    generated_at: str
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    warnings: list[str]
    error: Optional[str]
    notebook_path: str


def empty_lineage(method: str, *, error: str = "") -> Lineage:
    from datetime import datetime, timezone

    return {
        "method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": [],
        "edges": [],
        "warnings": [],
        "error": error,
        "notebook_path": "",
    }


# Reasonable cap so even pathological notebooks don't return 50 MB of JSON.
MAX_NODES = 500
MAX_EDGES = 1500
