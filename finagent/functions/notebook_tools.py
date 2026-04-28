"""Read / search / validate tools used by the validator + edit agents."""

from __future__ import annotations

import logging
import queue
import re
import time
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
