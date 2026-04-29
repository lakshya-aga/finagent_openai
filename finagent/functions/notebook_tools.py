"""Read / search / validate tools used by the validator + edit agents."""

from __future__ import annotations

import logging
import queue
import re
import time
from pathlib import Path
from typing import Any, Dict

import nbformat
from jupyter_client import KernelManager

from agents import function_tool

from .kernel import _serialize_output
from .notebook_io import _get_current_path, _load_notebook, _save_notebook


@function_tool
def read_notebook() -> Dict[str, Any]:
    """Read the full current notebook: every cell with index, type, source, and outputs."""
    logging.info("TOOL CALL: read_notebook")
    nb = _load_notebook()
    cells = []
    for i, cell in enumerate(nb.cells):
        entry = {
            "cell_index": i,
            "cell_type": cell.cell_type,
            "source": cell.source,
        }
        if cell.cell_type == "code":
            entry["outputs"] = cell.get("outputs", [])
        # Surface provenance metadata so agents see which DAG node built each cell.
        finagent_md = cell.metadata.get("finagent")
        if finagent_md:
            entry["finagent"] = finagent_md
        cells.append(entry)
    return {"success": True, "num_cells": len(nb.cells), "cells": cells}


@function_tool
def find_regex_in_notebook_code(regex_pattern: str, case_sensitive: bool):
    """Search the current notebook for a regex; return matches with surrounding snippets."""
    logging.info(f"TOOL CALL: find_regex_in_notebook_code {regex_pattern!r}")
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        nb = _load_notebook()
    except Exception as e:
        raise ValueError(f"Could not parse notebook content: {e}")

    matches = []
    pattern = re.compile(regex_pattern, flags)
    for idx, cell in enumerate(nb.cells):
        source = cell.source or ""
        for match in pattern.finditer(source):
            start, end = match.span()
            snippet_start = max(0, start - 80)
            snippet_end = min(len(source), end + 80)
            matches.append({
                "cell_index": idx,
                "cell_type": cell.cell_type,
                "match_text": match.group(0),
                "span": [start, end],
                "snippet": source[snippet_start:snippet_end],
            })

    return {
        "success": True,
        "regex_pattern": regex_pattern,
        "case_sensitive": case_sensitive,
        "num_matches": len(matches),
        "matches": matches,
    }


def lint_notebook_imports(path: str) -> dict:
    """Static AST-walk over a notebook's code cells; flag every imported
    module that isn't (a) part of the stdlib or (b) importable in the
    current kernel.

    Why this exists: a Jupyter kernel boot for ``validate_run`` takes ~10s,
    and most module-not-found failures are decidable at parse time. The
    validator agent treats the result like a real cell error — same shape,
    cheaper to detect.

    Returns
    -------
    dict
        ``{"ok": bool, "missing": [{"module": str, "cell_index": int,
        "line": int}, ...]}``.
        ``ok=False`` means at least one import targets a non-importable
        module.
    """
    import ast
    import importlib.util
    import sys as _sys

    p = Path(path)
    if not p.exists():
        return {"ok": False, "missing": [], "error": f"notebook not found: {p}"}

    try:
        with open(p, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as exc:
        return {"ok": False, "missing": [], "error": f"could not read notebook: {exc}"}

    # Stdlib check works on Python 3.10+ via sys.stdlib_module_names; we run
    # on 3.11+, so this is reliable.
    stdlib = set(getattr(_sys, "stdlib_module_names", ()))

    missing: list[dict] = []
    seen: set[str] = set()  # dedup across cells

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source or ""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # let the kernel surface real syntax errors

        for node in ast.walk(tree):
            names: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append((alias.name, getattr(node, "lineno", 0)))
            elif isinstance(node, ast.ImportFrom):
                # `from x.y import z` → check root package x
                if node.level == 0 and node.module:
                    names.append((node.module, getattr(node, "lineno", 0)))
            for name, line in names:
                root = name.split(".")[0]
                if not root or root in stdlib or root in seen:
                    continue
                try:
                    spec = importlib.util.find_spec(root)
                except (ImportError, ValueError):
                    spec = None
                if spec is None:
                    seen.add(root)
                    missing.append({
                        "module": name,
                        "cell_index": cell_idx,
                        "line": line,
                    })

    return {"ok": not missing, "missing": missing}


def run_all_cells_to_disk(path: str, timeout: int = 120) -> dict:
    """Execute every code cell in a single persistent kernel, write outputs back.

    Unlike :func:`validate_run`, this does NOT stop at the first error — every
    cell gets a turn so the user-visible "Run all" button populates as much
    of the notebook as possible. Errors are still attached to the offending
    cells (same nbformat error output shape), so the viewer can highlight
    them, but downstream cells run too — which is correct for "what does my
    notebook look like, output-wise?" even when one cell crashes.

    Returns a summary dict with the per-cell execution status.
    """
    notebook_path = Path(path)
    if not notebook_path.exists():
        return {"success": False, "error": f"notebook not found: {notebook_path}"}

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    if not code_cells:
        return {"success": True, "executed_cells": 0, "errors": []}

    km = KernelManager(kernel_name="finagent-python")
    km.start_kernel()
    errors: list[dict] = []
    execution_count = 0

    try:
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=timeout)

        for cell_idx, cell in code_cells:
            if not cell.source.strip():
                cell.outputs = []
                continue

            execution_count += 1
            msg_id = kc.execute(cell.source)
            cell_outputs = []
            deadline = time.time() + timeout

            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                try:
                    msg = kc.get_iopub_msg(timeout=remaining)
                except queue.Empty:
                    break
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                if msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
                    break
                out = _serialize_output(msg)
                if out is not None:
                    cell_outputs.append(out)

            cell.outputs = []
            cell["execution_count"] = execution_count
            for out in cell_outputs:
                ot = out["output_type"]
                if ot == "stream":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="stream", name=out.get("name", "stdout"),
                        text=out.get("text", ""),
                    ))
                elif ot in {"display_data", "execute_result"}:
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type=ot, data=out.get("data", {}),
                        metadata=out.get("metadata", {}),
                        **({"execution_count": execution_count} if ot == "execute_result" else {}),
                    ))
                elif ot == "error":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="error",
                        ename=out.get("ename", ""),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", []),
                    ))

            err = next((o for o in cell_outputs if o["output_type"] == "error"), None)
            if err:
                errors.append({
                    "cell_index": cell_idx,
                    "ename": err.get("ename", ""),
                    "evalue": err.get("evalue", ""),
                })
                # KEY DIFFERENCE vs validate_run: do NOT break. Keep running so
                # later cells still get outputs even if one fails.

    finally:
        try:
            kc.stop_channels()
        except Exception:
            pass
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return {
        "success": True,
        "path": str(notebook_path),
        "executed_cells": execution_count,
        "error_count": len(errors),
        "errors": errors,
    }


@function_tool
def validate_run(max_cells: int, timeout: int, prelude: str):
    """Run the full notebook in one persistent kernel, write outputs back, save to disk."""
    path = _get_current_path()
    logging.info(f"TOOL CALL: validate_run path={path} max_cells={max_cells} timeout={timeout}")

    nb = _load_notebook()

    indexed_code_cells = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            indexed_code_cells.append((i, cell))
            if len(indexed_code_cells) >= max_cells:
                break

    if not indexed_code_cells:
        return {
            "success": True,
            "path": str(path),
            "message": "No code cells to execute",
            "executed_cells": 0,
        }

    # Pre-kernel import lint: a clean fail here saves ~10s of kernel boot
    # when the orchestrator has invented a module name. The validator agent
    # treats the returned shape like a real cell error.
    lint = lint_notebook_imports(str(path))
    if not lint.get("ok") and lint.get("missing"):
        first = lint["missing"][0]
        return {
            "success": False,
            "path": str(path),
            "status": "error",
            "executed_cells": 0,
            "first_error_cell_index": first.get("cell_index"),
            "outputs": [],
            "error": {
                "ename": "ModuleNotFoundError",
                "evalue": (
                    f"No module named {first['module']!r} (detected at parse "
                    f"time, before kernel boot). Other missing modules: "
                    f"{[m['module'] for m in lint['missing'][1:]] or 'none'}."
                ),
                "traceback": [],
                "missing_modules": lint["missing"],
                "from_lint": True,
            },
        }

    km = KernelManager(kernel_name="finagent-python")
    km.start_kernel()
    first_error_cell = None
    error_output = None
    execution_count = 0

    try:
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=timeout)

        if prelude:
            kc.execute(prelude)

        for cell_idx, cell in indexed_code_cells:
            if not cell.source.strip():
                cell.outputs = []
                continue

            execution_count += 1
            msg_id = kc.execute(cell.source)
            cell_outputs = []
            deadline = time.time() + timeout

            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                try:
                    msg = kc.get_iopub_msg(timeout=remaining)
                except queue.Empty:
                    break
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                if msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
                    break
                out = _serialize_output(msg)
                if out is not None:
                    cell_outputs.append(out)

            cell.outputs = []
            cell["execution_count"] = execution_count
            for out in cell_outputs:
                output_type = out["output_type"]
                if output_type == "stream":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="stream", name=out.get("name", "stdout"), text=out.get("text", "")
                    ))
                elif output_type in {"display_data", "execute_result"}:
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type=output_type,
                        data=out.get("data", {}),
                        metadata=out.get("metadata", {}),
                        **({"execution_count": execution_count} if output_type == "execute_result" else {}),
                    ))
                elif output_type == "error":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="error",
                        ename=out.get("ename", ""),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", []),
                    ))

            error_in_cell = next((o for o in cell_outputs if o["output_type"] == "error"), None)
            if error_in_cell:
                first_error_cell = cell_idx
                error_output = error_in_cell
                break

    finally:
        try:
            kc.stop_channels()
        except Exception:
            pass
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass

    _save_notebook(nb, path)

    success = error_output is None
    return {
        "success": success,
        "path": str(path),
        "status": "ok" if success else "error",
        "executed_cells": len(indexed_code_cells),
        "first_error_cell_index": first_error_cell,
        "outputs": [],
        "error": error_output,
    }
