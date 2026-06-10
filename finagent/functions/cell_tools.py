"""Notebook-cell @function_tool wrappers used by the orchestration agents.

Traceability hook: add_cell / replace_cell / insert_cell accept optional
`dag_node_id` and `rationale` strings. When provided, they are written to
`cell.metadata["finagent"]` so the frontend can render per-cell provenance
(which DAG node produced this cell, and *why*).
"""

from __future__ import annotations

import logging

import nbformat
from agents import function_tool

from .kernel import _run_code_in_kernel
from .notebook_io import (
    _ensure_parent_dir,
    _get_current_path,
    _get_latest_path,
    _load_notebook,
    _make_cell,
    _save_notebook,
    get_active_notebook_path,
    set_active_notebook_path,
)


def _apply_provenance(cell, dag_node_id: str, rationale: str) -> None:
    """Stamp finagent metadata onto a cell. Empty strings are ignored."""
    if not dag_node_id and not rationale:
        return
    md = cell.metadata.setdefault("finagent", {})
    if dag_node_id:
        md["node_id"] = dag_node_id
    if rationale:
        md["rationale"] = rationale


def create_notebook_impl():
    """Create an empty notebook for this build, idempotently."""
    import sys

    from nbformat.v4 import new_notebook

    # Idempotency guard: a build already has its notebook → return it.
    active = get_active_notebook_path()
    if active is not None and active.exists():
        logging.info(f"TOOL CALL: create_notebook (idempotent no-op) {active}")
        nb_existing = _load_notebook()
        return {
            "success": True,
            "path": str(active),
            "message": "Notebook already exists for this build",
            "num_cells": len(nb_existing.cells),
        }

    path = _get_latest_path()
    logging.info(f"TOOL CALL: create_notebook {path}")
    _ensure_parent_dir(path)

    nb = new_notebook(cells=[])
    nb.metadata["kernelspec"] = {
        "display_name": "FinAgent Python",
        "language": "python",
        "name": "finagent-python",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    _save_notebook(nb, path)
    # Pin this file as the active notebook so every subsequent add_cell /
    # validate_run resolves to it — even though its name won't match the
    # legacy notebook_N pattern. Without this, cell writes route to a stale
    # numbered notebook in outputs/.
    set_active_notebook_path(path)
    return {
        "success": True,
        "path": str(path),
        "message": "Notebook created",
        "num_cells": 0,
    }


@function_tool
def create_notebook():
    return create_notebook_impl()


def add_cell_impl(
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
):
    """Append a cell. Pass dag_node_id and a one-line rationale so the cell is traceable.

    For code cells, ALWAYS supply both fields. For markdown header cells that
    narrate the next code cell, the same dag_node_id is fine and rationale may be
    left empty.
    """
    path = _get_current_path()
    logging.info(f"TOOL CALL: add_cell type={cell_type} node={dag_node_id!r}")

    nb = _load_notebook()
    cell = _make_cell(cell_type, content)
    _apply_provenance(cell, dag_node_id, rationale)
    nb.cells.append(cell)
    _save_notebook(nb, path)

    return {
        "success": True,
        "path": str(path),
        "message": "Cell added",
        "cell_index": len(nb.cells) - 1,
        "cell_type": cell_type,
        "num_cells": len(nb.cells),
    }


@function_tool
def add_cell(
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
):
    return add_cell_impl(cell_type, content, dag_node_id, rationale)


def replace_cell_impl(
    cell_index: int,
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
):
    """Replace a cell in place. Re-state dag_node_id / rationale so provenance survives."""
    path = _get_current_path()
    logging.info(
        f"TOOL CALL: replace_cell idx={cell_index} type={cell_type} node={dag_node_id!r}"
    )

    nb = _load_notebook()
    if cell_index < 0 or cell_index >= len(nb.cells):
        raise IndexError(
            f"cell_index {cell_index} out of range for notebook with {len(nb.cells)} cells"
        )

    new = _make_cell(cell_type, content)
    # Carry forward any prior provenance, then overlay new fields if provided.
    prior = nb.cells[cell_index].metadata.get("finagent")
    if prior:
        new.metadata["finagent"] = dict(prior)
    _apply_provenance(new, dag_node_id, rationale)
    nb.cells[cell_index] = new
    _save_notebook(nb, path)

    return {
        "success": True,
        "path": str(path),
        "message": "Cell replaced",
        "cell_index": cell_index,
        "cell_type": cell_type,
        "num_cells": len(nb.cells),
    }


@function_tool
def replace_cell(
    cell_index: int,
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
):
    return replace_cell_impl(cell_index, cell_type, content, dag_node_id, rationale)


def insert_cell_impl(
    cell_index: int,
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
):
    """Insert a new cell at cell_index, shifting subsequent cells down."""
    path = _get_current_path()
    logging.info(
        f"TOOL CALL: insert_cell idx={cell_index} type={cell_type} node={dag_node_id!r}"
    )
    nb = _load_notebook()
    if cell_index < 0 or cell_index > len(nb.cells):
        return {
            "success": False,
            "error": f"cell_index {cell_index} out of range (0–{len(nb.cells)})",
        }
    cell = _make_cell(cell_type, content)
    _apply_provenance(cell, dag_node_id, rationale)
    nb.cells.insert(cell_index, cell)
    _save_notebook(nb, path)
    return {"success": True, "cell_index": cell_index, "num_cells": len(nb.cells)}


@function_tool
def insert_cell(
    cell_index: int,
    cell_type: str,
    content: str,
    dag_node_id: str = "",
    rationale: str = "",
):
    return insert_cell_impl(cell_index, cell_type, content, dag_node_id, rationale)


def delete_cell_impl(cell_index: int):
    """Delete the cell at cell_index, shifting subsequent cells up."""
    path = _get_current_path()
    logging.info(f"TOOL CALL: delete_cell idx={cell_index}")
    nb = _load_notebook()
    if cell_index < 0 or cell_index >= len(nb.cells):
        return {
            "success": False,
            "error": f"cell_index {cell_index} out of range (0–{len(nb.cells) - 1})",
        }
    del nb.cells[cell_index]
    _save_notebook(nb, path)
    return {"success": True, "deleted_index": cell_index, "num_cells": len(nb.cells)}


@function_tool
def delete_cell(cell_index: int):
    return delete_cell_impl(cell_index)


def run_cell_impl(path: str, cell_index: int, timeout: int):
    """Run a single cell after executing all preceding code cells as prelude."""
    path = _get_current_path()
    logging.info(f"TOOL CALL: run_cell idx={cell_index}")

    nb = _load_notebook()
    if cell_index < 0 or cell_index >= len(nb.cells):
        raise IndexError(
            f"cell_index {cell_index} out of range for notebook with {len(nb.cells)} cells"
        )

    target_cell = nb.cells[cell_index]
    if target_cell.cell_type != "code":
        return {
            "success": False,
            "path": str(path),
            "cell_index": cell_index,
            "message": "Target cell is not a code cell",
        }

    prelude = []
    for i in range(cell_index):
        cell = nb.cells[i]
        if cell.cell_type == "code":
            prelude.append(f"# --- cell {i} ---\n{cell.source}")
    prelude_code = "\n\n".join(prelude)
    final_code = (
        f"{prelude_code}\n\n# --- target cell {cell_index} ---\n{target_cell.source}"
        if prelude_code
        else target_cell.source
    )

    target_result = _run_code_in_kernel(final_code, timeout=timeout)

    target_cell.outputs = []
    for output in target_result["outputs"]:
        output_type = output["output_type"]
        if output_type == "stream":
            target_cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="stream",
                    name=output.get("name"),
                    text=output.get("text", ""),
                )
            )
        elif output_type in {"display_data", "execute_result"}:
            target_cell.outputs.append(
                nbformat.v4.new_output(
                    output_type=output_type,
                    data=output.get("data", {}),
                    metadata=output.get("metadata", {}),
                    execution_count=output.get("execution_count"),
                )
            )
        elif output_type == "error":
            target_cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="error",
                    ename=output.get("ename"),
                    evalue=output.get("evalue"),
                    traceback=output.get("traceback", []),
                )
            )

    _save_notebook(nb, path)

    return {
        "success": target_result["success"],
        "path": str(path),
        "cell_index": cell_index,
        "status": target_result["status"],
        "outputs": target_result["outputs"],
        "error": target_result["error"],
    }


@function_tool
def run_cell(path: str, cell_index: int, timeout: int):
    return run_cell_impl(path, cell_index, timeout)
