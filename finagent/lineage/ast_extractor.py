"""Static lineage extraction using stdlib `ast`.

Walks every code cell, parses it to an AST, and rewrites assignments of the
form ``targets = call(...)`` into a graph of:

    arg_var ──► call_node ──► target_var

Variables are tracked across cells via a single global symbol table because
Jupyter notebooks share state. Cells with syntax errors are skipped and
listed in ``warnings``.

Limitations (documented because users will hit them):
  - Chained method calls collapse to one call node per Assign — e.g.
    ``df.dropna().resample("D").mean()`` produces a single ``mean`` node
    with `df` as input. Useful enough; precise enough is the runtime
    extractor's job.
  - In-place mutation is invisible (``df.fillna(.., inplace=True)`` looks
    like a no-op).
  - Bare expressions, `for`/`if` bodies, function definitions are skipped.
"""

from __future__ import annotations

import ast
import builtins
from datetime import datetime, timezone
from typing import Iterable

import nbformat

from .types import (
    MAX_EDGES,
    MAX_NODES,
    Lineage,
    LineageEdge,
    LineageNode,
    empty_lineage,
)


_BUILTIN_NAMES: set[str] = set(dir(builtins)) | {
    "self", "cls", "True", "False", "None",
}


def extract_lineage_ast(notebook_path: str) -> Lineage:
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        out = empty_lineage("ast", error=f"could not open notebook: {e}")
        out["notebook_path"] = str(notebook_path)
        return out

    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []
    warnings: list[str] = []

    # var_name -> latest producing node id. One symbol table for the whole
    # notebook because Jupyter sells the same scope across cells.
    symbols: dict[str, str] = {}
    seq = 0

    def next_id(prefix: str) -> str:
        nonlocal seq
        seq += 1
        return f"{prefix}{seq}"

    def add_node(node: LineageNode) -> None:
        if len(nodes) < MAX_NODES:
            nodes.append(node)

    def add_edge(edge: LineageEdge) -> None:
        if len(edges) < MAX_EDGES:
            edges.append(edge)

    def get_or_create_input(var: str) -> str:
        """Return the producer node id for ``var`` — creating an input
        node if it has never been assigned in this notebook."""
        if var in symbols:
            return symbols[var]
        nid = next_id("in_")
        add_node({"id": nid, "label": var, "kind": "input"})
        symbols[var] = nid
        return nid

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source or ""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            warnings.append(f"cell {cell_idx}: syntax error ({exc.msg}); skipped")
            continue

        for stmt in tree.body:
            _handle_statement(
                stmt,
                cell_idx,
                symbols,
                add_node,
                add_edge,
                next_id,
                get_or_create_input,
            )

    # Mark terminal data nodes as outputs so the renderer can highlight
    # them — they are never read by anything downstream.
    src_set = {e.get("src") for e in edges}
    for n in nodes:
        if n.get("kind") == "data" and n.get("id") not in src_set:
            n["kind"] = "output"

    return {
        "method": "ast",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": edges,
        "warnings": warnings,
        "error": None,
        "notebook_path": str(notebook_path),
    }


# ─── per-statement handler ──────────────────────────────────────────────


def _handle_statement(
    stmt: ast.stmt,
    cell_idx: int,
    symbols: dict[str, str],
    add_node,
    add_edge,
    next_id,
    get_or_create_input,
) -> None:
    if isinstance(stmt, ast.Assign):
        # ``a, b = call(x)`` — extract the call once, link to all targets.
        if isinstance(stmt.value, ast.Call):
            _emit_call_assign(
                stmt.value,
                stmt.targets,
                cell_idx,
                symbols,
                add_node,
                add_edge,
                next_id,
                get_or_create_input,
            )
        else:
            # Plain ``a = b`` — alias, link b -> a so lineage carries through.
            inputs = sorted(_collect_loaded_names(stmt.value))
            targets = sorted(_extract_target_names(stmt.targets))
            for tgt in targets:
                tgt_id = next_id("var_")
                add_node({
                    "id": tgt_id,
                    "label": tgt,
                    "kind": "data",
                    "cell_idx": cell_idx,
                })
                for src in inputs:
                    if src in _BUILTIN_NAMES:
                        continue
                    add_edge({
                        "id": next_id("e_"),
                        "src": get_or_create_input(src),
                        "dst": tgt_id,
                        "label": "alias",
                    })
                symbols[tgt] = tgt_id
    elif isinstance(stmt, ast.AugAssign):
        # ``a += expr`` — same as Assign for our purposes.
        if isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            tgt_id = next_id("var_")
            add_node({
                "id": tgt_id,
                "label": target_name,
                "kind": "data",
                "cell_idx": cell_idx,
            })
            inputs = sorted(_collect_loaded_names(stmt.value)) + [target_name]
            for src in inputs:
                if src in _BUILTIN_NAMES:
                    continue
                add_edge({
                    "id": next_id("e_"),
                    "src": get_or_create_input(src),
                    "dst": tgt_id,
                    "label": type(stmt.op).__name__.lower(),
                })
            symbols[target_name] = tgt_id
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        # Bare call (e.g. plotting). Capture inputs but no produced var.
        call = stmt.value
        fn = _call_label(call)
        call_id = next_id("call_")
        add_node({
            "id": call_id,
            "label": fn,
            "kind": "call",
            "cell_idx": cell_idx,
        })
        for src in sorted(_collect_loaded_names(call)):
            if src in _BUILTIN_NAMES or src == fn.split(".")[0]:
                continue
            add_edge({
                "id": next_id("e_"),
                "src": get_or_create_input(src),
                "dst": call_id,
            })
    # else: import / def / class / control flow — ignored on purpose.


# ─── call assignment ────────────────────────────────────────────────────


def _emit_call_assign(
    call: ast.Call,
    targets: list[ast.expr],
    cell_idx: int,
    symbols: dict[str, str],
    add_node,
    add_edge,
    next_id,
    get_or_create_input,
) -> None:
    fn = _call_label(call)
    call_id = next_id("call_")
    add_node({
        "id": call_id,
        "label": fn,
        "kind": "call",
        "cell_idx": cell_idx,
    })

    # Inputs: every Name(Load) referenced in the call subtree, minus the
    # callable's own attribute-chain root (so ``df.method(x)`` doesn't
    # double-count `df` as both source and "the bound object").
    callable_root = fn.split(".")[0] if fn else ""
    arg_names = sorted(_collect_loaded_names(call))
    for src in arg_names:
        if src in _BUILTIN_NAMES:
            continue
        # Method-call: still record the bound object as a source — it's
        # the variable the call mutates / reads.
        add_edge({
            "id": next_id("e_"),
            "src": get_or_create_input(src),
            "dst": call_id,
        })
        if src == callable_root:
            # don't duplicate
            pass

    # Outputs: every name in the targets list. Tuple/List unpack expands.
    for target in targets:
        for tname in _extract_target_names_one(target):
            var_id = next_id("var_")
            add_node({
                "id": var_id,
                "label": tname,
                "kind": "data",
                "cell_idx": cell_idx,
            })
            add_edge({
                "id": next_id("e_"),
                "src": call_id,
                "dst": var_id,
                "label": tname,
            })
            symbols[tname] = var_id


# ─── helpers ────────────────────────────────────────────────────────────


def _call_label(call: ast.Call) -> str:
    """``foo``, ``df.method``, ``a.b.c.method`` — best-effort dotted name."""
    f = call.func
    parts: list[str] = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    elif isinstance(f, ast.Call):
        # ``foo()()`` — fall back to "<chained call>" rather than recurse.
        parts.append("<call>")
    if not parts:
        return "<call>"
    return ".".join(reversed(parts))


def _collect_loaded_names(node: ast.AST) -> set[str]:
    """All names loaded (read) inside a subtree."""
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            out.add(child.id)
    return out


def _extract_target_names(targets: Iterable[ast.expr]) -> set[str]:
    out: set[str] = set()
    for t in targets:
        out.update(_extract_target_names_one(t))
    return out


def _extract_target_names_one(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_extract_target_names_one(elt))
        return out
    if isinstance(target, ast.Starred):
        return _extract_target_names_one(target.value)
    if isinstance(target, ast.Subscript):
        # ``d[k] = v`` — attribute-write; skip rather than mis-name.
        return []
    if isinstance(target, ast.Attribute):
        return []
    return []
