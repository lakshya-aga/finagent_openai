"""Runtime-traced lineage extraction.

Spawns a fresh Python subprocess (so the host process's globals don't leak
in), executes each notebook code cell sequentially, and after each cell
records the namespace delta — which names appeared, which mutated, and
roughly how big each value is. That delta is then combined with the call
names AST extracted from the same cell to attribute "produced by call X
in cell N" rather than the AST-only "looks like an Assign in cell N".

Why this beats pure AST:
  - ``inplace=True`` mutations show up because the mutated object's hash
    changes between cells.
  - Variables introduced via ``exec()``, ``getattr``, or other dynamic
    constructs are visible because we just look at the globals dict.
  - Cells that fail at runtime stop the trace cleanly with the error
    attached to that cell.

Why we don't use lineapy (yet):
  - Adds a heavy dep + database, runtime is slower, current adoption is
    thin. Easy to swap in later by adding another extractor with the
    same Lineage shape.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any

import nbformat

from .ast_extractor import _call_label, _collect_loaded_names, _BUILTIN_NAMES
from .types import (
    MAX_EDGES,
    MAX_NODES,
    Lineage,
    LineageEdge,
    LineageNode,
    empty_lineage,
)


# Time cap so a runaway notebook doesn't pin the server. Configurable via
# env so users debugging long-running cells can extend it.
_TIMEOUT_SEC = int(os.environ.get("FINAGENT_LINEAGE_TIMEOUT", "180"))


# Code that runs in the spawned interpreter. Reads the notebook path from
# argv[1], executes each code cell in a shared globals dict, and emits
# per-cell namespace deltas as JSON to stdout.
_RUNNER = textwrap.dedent(
    """
    import json, sys, hashlib, traceback

    try:
        import nbformat
    except ImportError as e:
        print(json.dumps({"runner_error": f"nbformat missing: {e}"}))
        sys.exit(1)

    def _hash(v):
        try:
            return hashlib.sha1(repr(v).encode("utf-8", "replace")).hexdigest()[:12]
        except Exception:
            return ""

    def _summary(v):
        try:
            t = type(v).__name__
            shape = ""
            try:
                if hasattr(v, "shape"):
                    shape = f" shape={tuple(v.shape)}"
                elif hasattr(v, "__len__"):
                    shape = f" len={len(v)}"
            except Exception:
                pass
            return f"{t}{shape}"
        except Exception:
            return type(v).__name__

    nb = nbformat.read(open(sys.argv[1], encoding="utf-8"), as_version=4)
    g = {"__name__": "__main__"}
    timeline = []
    prev_hashes = {}

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        before = set(g.keys())
        try:
            exec(compile(cell.source, f"<cell {i}>", "exec"), g)
            err = None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        after = set(g.keys())
        new_names = sorted(n for n in (after - before) if not n.startswith("__"))

        # Detect mutated bindings (hash changed) for names that already
        # existed. Useful for catching `df.fillna(.., inplace=True)` etc.
        mutated = []
        for name in (after & before):
            if name.startswith("__"):
                continue
            try:
                h = _hash(g.get(name))
            except Exception:
                h = ""
            if name in prev_hashes and prev_hashes[name] != h:
                mutated.append(name)
            prev_hashes[name] = h
        for name in new_names:
            try:
                prev_hashes[name] = _hash(g.get(name))
            except Exception:
                prev_hashes[name] = ""

        details = {n: _summary(g.get(n)) for n in (new_names + mutated) if n in g}
        timeline.append({
            "cell": i,
            "new_vars": new_names,
            "mutated_vars": mutated,
            "details": details,
            "error": err,
        })
        if err:
            break

    print(json.dumps({"timeline": timeline}))
    """
)


def extract_lineage_runtime(notebook_path: str) -> Lineage:
    out = empty_lineage("runtime")
    out["notebook_path"] = str(notebook_path)

    if not os.path.exists(notebook_path):
        out["error"] = f"notebook not found: {notebook_path}"
        return out

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, notebook_path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        out["error"] = f"runtime trace timed out after {_TIMEOUT_SEC}s"
        return out
    except Exception as exc:
        out["error"] = f"could not spawn runtime tracer: {exc}"
        return out

    if proc.returncode != 0 and not proc.stdout:
        out["error"] = f"tracer exited {proc.returncode}: {proc.stderr.strip()[:400]}"
        return out

    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        payload = json.loads(last_line)
    except Exception as exc:
        out["error"] = f"could not parse tracer output: {exc}"
        return out

    if "runner_error" in payload:
        out["error"] = payload["runner_error"]
        return out

    timeline: list[dict[str, Any]] = payload.get("timeline", [])
    return _build_graph_from_timeline(notebook_path, timeline, out["warnings"])


# ─── graph builder ──────────────────────────────────────────────────────


def _build_graph_from_timeline(
    notebook_path: str,
    timeline: list[dict[str, Any]],
    warnings: list[str],
) -> Lineage:
    """Combine runtime namespace deltas with AST-extracted call names."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    cells_by_idx = {i: c for i, c in enumerate(nb.cells)}

    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []
    seq = 0
    symbols: dict[str, str] = {}

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
        if var in symbols:
            return symbols[var]
        nid = next_id("in_")
        add_node({"id": nid, "label": var, "kind": "input"})
        symbols[var] = nid
        return nid

    for entry in timeline:
        cell_idx = entry["cell"]
        new_vars: list[str] = entry.get("new_vars") or []
        mutated_vars: list[str] = entry.get("mutated_vars") or []
        details: dict[str, str] = entry.get("details") or {}
        err: str | None = entry.get("error")
        if err:
            warnings.append(f"cell {cell_idx} raised at runtime: {err}")

        cell = cells_by_idx.get(cell_idx)
        if cell is None:
            continue
        source = cell.source or ""

        # Best-effort: pick the first ``Assign(value=Call)`` per produced
        # var as the call we attribute it to. If the cell has multiple
        # calls, we link each new var to its corresponding Assign by
        # walking the AST in order.
        call_attribution = _attribute_calls_to_targets(source, new_vars + mutated_vars)

        for var in new_vars:
            attr = call_attribution.get(var)
            var_id = next_id("var_")
            add_node({
                "id": var_id,
                "label": var,
                "kind": "data",
                "cell_idx": cell_idx,
                "details": details.get(var, ""),
            })
            if attr is not None:
                fn_name, input_names = attr
                call_id = next_id("call_")
                add_node({
                    "id": call_id,
                    "label": fn_name,
                    "kind": "call",
                    "cell_idx": cell_idx,
                })
                for src in input_names:
                    if src in _BUILTIN_NAMES:
                        continue
                    add_edge({
                        "id": next_id("e_"),
                        "src": get_or_create_input(src),
                        "dst": call_id,
                    })
                add_edge({
                    "id": next_id("e_"),
                    "src": call_id,
                    "dst": var_id,
                    "label": var,
                })
            symbols[var] = var_id

        for var in mutated_vars:
            # Mutation: wrap into a call node labelled "<mutation>" with
            # the variable as input + output. Preserves continuity.
            attr = call_attribution.get(var)
            call_id = next_id("call_")
            add_node({
                "id": call_id,
                "label": (attr[0] if attr else "<mutation>"),
                "kind": "call",
                "cell_idx": cell_idx,
            })
            old_id = symbols.get(var)
            if old_id is None:
                old_id = get_or_create_input(var)
            add_edge({
                "id": next_id("e_"),
                "src": old_id,
                "dst": call_id,
            })
            new_id = next_id("var_")
            add_node({
                "id": new_id,
                "label": var,
                "kind": "data",
                "cell_idx": cell_idx,
                "details": details.get(var, ""),
            })
            add_edge({
                "id": next_id("e_"),
                "src": call_id,
                "dst": new_id,
                "label": "mutated",
            })
            symbols[var] = new_id

    src_set = {e.get("src") for e in edges}
    for n in nodes:
        if n.get("kind") == "data" and n.get("id") not in src_set:
            n["kind"] = "output"

    return {
        "method": "runtime",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": edges,
        "warnings": warnings,
        "error": None,
        "notebook_path": str(notebook_path),
    }


def _attribute_calls_to_targets(
    source: str, var_names: list[str]
) -> dict[str, tuple[str, list[str]]]:
    """For each interesting var, find the Assign whose target matches and
    return (call_name, input_names). Handles tuple/list unpack."""
    out: dict[str, tuple[str, list[str]]] = {}
    if not source.strip():
        return out
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out

    interest = set(var_names)

    def _names_in_target(target: ast.expr) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for elt in target.elts for n in _names_in_target(elt)]
        return []

    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        targets = []
        for t in stmt.targets:
            targets.extend(_names_in_target(t))
        hit = [t for t in targets if t in interest]
        if not hit:
            continue
        fn = _call_label(stmt.value)
        inputs = sorted(_collect_loaded_names(stmt.value))
        for t in hit:
            out[t] = (fn, inputs)
    return out
