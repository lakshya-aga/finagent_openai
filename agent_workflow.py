

from agents import function_tool, FileSearchTool, HostedMCPTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from agents.mcp import MCPServerSse, MCPServerSseParams, MCPServerManager, MCPServerStreamableHttp, MCPServerStreamableHttpParams
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import json
import time
import queue
import re
import subprocess

import sys

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from jupyter_client import KernelManager
import logging

logging.basicConfig(
    filename="finagent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def _get_latest_path():
    i = 1

    while True:
        path = Path(f"outputs/notebook_{i}.ipynb")
        if not path.exists():
            return path
        i += 1

def _get_current_path():
    files = os.listdir(Path("outputs"))
    files.sort()
    return Path.joinpath(Path("outputs"), files[-1])


def _ensure_parent_dir(path) -> None:
    logging.info(f"TOOL CALL: {locals()}")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_notebook():
    path = _get_current_path()
    logging.info(f"TOOL CALL: {locals()}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Notebook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _save_notebook(nb, path) -> None:
    logging.info(f"TOOL CALL: {locals()}")

    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def _make_cell(cell_type: str, content: str):
    logging.info(f"TOOL CALL: {locals()}")

    if cell_type == "code":
        return new_code_cell(content)
    if cell_type == "markdown":
        return new_markdown_cell(content)
    raise ValueError("cell_type must be 'code' or 'markdown'")


def _serialize_output(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    logging.info(f"TOOL CALL: {locals()}")

    msg_type = msg.get("msg_type")
    content = msg.get("content", {})

    if msg_type == "stream":
        return {
            "output_type": "stream",
            "name": content.get("name"),
            "text": content.get("text", ""),
        }

    if msg_type in {"display_data", "execute_result"}:
        return {
            "output_type": msg_type,
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
            "execution_count": content.get("execution_count"),
        }

    if msg_type == "error":
        return {
            "output_type": "error",
            "ename": content.get("ename"),
            "evalue": content.get("evalue"),
            "traceback": content.get("traceback", []),
        }

    return None


def _run_code_in_kernel(code: str, timeout: int = 120) -> Dict[str, Any]:

    """
    Execute code in a temporary Jupyter kernel and collect outputs.
    """
    logging.info(f"TOOL CALL: {locals()}")

    logging.info(f"_run_code_in_kernel using kernel_name=finagent-python")
    km = KernelManager(kernel_name="finagent-python")
    km.start_kernel()


    try:
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=timeout)

        msg_id = kc.execute(code)
        outputs = []
        execute_reply = None
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = kc.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                break

            parent = msg.get("parent_header", {})
            if parent.get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            if msg_type == "status" and msg.get("content", {}).get("execution_state") == "idle":
                break

            out = _serialize_output(msg)
            if out is not None:
                outputs.append(out)

        shell_deadline = time.time() + 5
        while time.time() < shell_deadline:
            try:
                reply = kc.get_shell_msg(timeout=0.5)
            except queue.Empty:
                continue
            if reply.get("parent_header", {}).get("msg_id") == msg_id:
                execute_reply = reply
                break

        status = None
        if execute_reply:
            status = execute_reply.get("content", {}).get("status")

        error = next((o for o in outputs if o["output_type"] == "error"), None)

        return {
            "success": error is None and status != "error",
            "status": status or ("error" if error else "ok"),
            "outputs": outputs,
            "error": error,
        }

    finally:
        try:
            kc.stop_channels()
        except Exception:
            pass
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@function_tool
def create_notebook():
    """
    Creates an empty notebook
    """
    # print("create Notebook called")
    path =  _get_latest_path()
    # print(path)
    logging.info(f"TOOL CALL: {locals()}")

    _ensure_parent_dir(path)
    # print("Created parent directory")


    nb = new_notebook(cells=[])
    # print("NB initialised")

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
    # print("Successfully Notebook saved")
    return {
        "success": True,
        "path": path,
        "message": "Notebook created",
        "num_cells": 0,
    }


@function_tool
def add_cell(cell_type: str, content: str):
    """
    Add a cell in the created notebook
    """
    path = _get_current_path()
    logging.info(f"TOOL CALL: {locals()}")

    nb = _load_notebook()
    cell = _make_cell(cell_type, content)
    nb.cells.append(cell)
    _save_notebook(nb, path)

    return {
        "success": True,
        "path": path,
        "message": "Cell added",
        "cell_index": len(nb.cells) - 1,
        "cell_type": cell_type,
        "num_cells": len(nb.cells),
    }


@function_tool
def replace_cell(cell_index: int, cell_type: str, content: str):
    """
    Replace a cell in the created notebook
    """
    path = _get_current_path()
    logging.info(f"TOOL CALL: {locals()}")

    nb = _load_notebook()

    if cell_index < 0 or cell_index >= len(nb.cells):
        raise IndexError(f"cell_index {cell_index} out of range for notebook with {len(nb.cells)} cells")

    nb.cells[cell_index] = _make_cell(cell_type, content)
    _save_notebook(nb, path)

    return {
        "success": True,
        "path": path,
        "message": "Cell replaced",
        "cell_index": cell_index,
        "cell_type": cell_type,
        "num_cells": len(nb.cells),
    }


@function_tool
def insert_cell(cell_index: int, cell_type: str, content: str):
    """
    Insert a new cell at the given index, shifting existing cells down.
    """
    path = _get_current_path()
    logging.info(f"TOOL CALL: {locals()}")
    nb = _load_notebook()
    if cell_index < 0 or cell_index > len(nb.cells):
        return {"success": False, "error": f"cell_index {cell_index} out of range (0–{len(nb.cells)})"}
    nb.cells.insert(cell_index, _make_cell(cell_type, content))
    _save_notebook(nb, path)
    return {"success": True, "cell_index": cell_index, "num_cells": len(nb.cells)}


@function_tool
def delete_cell(cell_index: int):
    """
    Delete the cell at the given index, shifting subsequent cells up.
    """
    path = _get_current_path()
    logging.info(f"TOOL CALL: {locals()}")
    nb = _load_notebook()
    if cell_index < 0 or cell_index >= len(nb.cells):
        return {"success": False, "error": f"cell_index {cell_index} out of range (0–{len(nb.cells)-1})"}
    del nb.cells[cell_index]
    _save_notebook(nb, path)
    return {"success": True, "deleted_index": cell_index, "num_cells": len(nb.cells)}


@function_tool
def run_cell(path: str, cell_index: int, timeout: int):
    """
    Run a cell to see the output
    """
    path = _get_current_path()
    logging.info(f"TOOL CALL: {locals()}")

    nb = _load_notebook()

    if cell_index < 0 or cell_index >= len(nb.cells):
        raise IndexError(f"cell_index {cell_index} out of range for notebook with {len(nb.cells)} cells")

    target_cell = nb.cells[cell_index]
    if target_cell.cell_type != "code":
        return {
            "success": False,
            "path": path,
            "cell_index": cell_index,
            "message": "Target cell is not a code cell",
        }

    # Execute all code cells up to and including cell_index so state is available.
    code_parts = []
    for i in range(cell_index + 1):
        cell = nb.cells[i]
        if cell.cell_type == "code":
            code_parts.append(f"# --- cell {i} ---\n{cell.source}")

    code_to_run = "\n\n".join(code_parts)
    result = _run_code_in_kernel(code_to_run, timeout=timeout, kernel_name="/Users/lakshya/miniconda3/envs/finagentv2/bin/python")

    # Attach outputs only to the target cell from the last execution chunk is hard to isolate
    # without executing cell-by-cell; for reliability, run target cell separately after prelude.
    prelude = []
    for i in range(cell_index):
        cell = nb.cells[i]
        if cell.cell_type == "code":
            prelude.append(f"# --- cell {i} ---\n{cell.source}")
    prelude_code = "\n\n".join(prelude)
    final_code = f"{prelude_code}\n\n# --- target cell {cell_index} ---\n{target_cell.source}" if prelude_code else target_cell.source

    target_result = _run_code_in_kernel(final_code, timeout=timeout, kernel_name="/Users/lakshya/miniconda3/envs/finagentv2/bin/python")

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
        "path": path,
        "cell_index": cell_index,
        "status": target_result["status"],
        "outputs": target_result["outputs"],
        "error": target_result["error"],
    }


@function_tool
def install_packages(packages: List[str]):
    """
    Install python packages in the current environment
    """
    logging.info(f"TOOL CALL: {locals()}")

    if not packages:
        return {
            "success": True,
            "message": "No packages requested",
            "installed": [],
        }

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "success": proc.returncode == 0,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "installed": packages if proc.returncode == 0 else [],
    }



@function_tool
def find_regex_in_notebook_code(
    regex_pattern: str,
    case_sensitive: bool,
):
    """
    Provide the regula expression to search the current notebook
    """
    logging.info(f"TOOL CALL: {locals()}")
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        nb = nbformat.reads(open(_get_latest_path()).read(), as_version=4)
    except Exception as e:
        raise ValueError(f"Could not parse notebook content: {e}")

    allowed = {"code", "markdown"}
    
    matches = []
    pattern = re.compile(regex_pattern, flags)

    for idx, cell in enumerate(nb.cells):

        source = cell.source or ""
        for match in pattern.finditer(source):
            start, end = match.span()
            snippet_start = max(0, start - 80)
            snippet_end = min(len(source), end + 80)
            snippet = source[snippet_start:snippet_end]

            matches.append({
                "cell_index": idx,
                "cell_type": cell.cell_type,
                "match_text": match.group(0),
                "span": [start, end],
                "snippet": snippet,
            })

    return {
        "success": True,
        "regex_pattern": regex_pattern,
        "case_sensitive": case_sensitive,
        "num_matches": len(matches),
        "matches": matches,
    }

@function_tool
def read_notebook() -> Dict[str, Any]:
    """
    Read the full current notebook, returning all cells with their index, type, source, and outputs.
    """
    logging.info(f"TOOL CALL: read_notebook")
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
        cells.append(entry)
    return {
        "success": True,
        "num_cells": len(nb.cells),
        "cells": cells,
    }

@function_tool
def validate_run(max_cells: int, timeout: int, prelude: str):
    """
    Run the full current notebook cell-by-cell in a single persistent kernel,
    write outputs back to each cell, and save the notebook to disk.
    """
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

        # Run prelude (if any) without attributing outputs to a cell
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

            # Write outputs back to the cell
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
                        **( {"execution_count": execution_count} if output_type == "execute_result" else {} )
                    ))
                elif output_type == "error":
                    cell.outputs.append(nbformat.v4.new_output(
                        output_type="error",
                        ename=out.get("ename", ""),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", []),
                    ))

            # Stop on first error
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

PROTECTED_PACKAGES = {"findata", "mlfinlab"}

@function_tool
def install_packages(packages: List[str]) -> Dict[str, Any]:
    """
    Install python packages into the current environment.
    Raises immediately if any package in PROTECTED_PACKAGES is requested,
    signalling the user must install those manually.
    """
    logging.info(f"TOOL CALL: install_packages {packages}")

    if not packages:
        return {"success": True, "message": "No packages requested", "installed": []}

    protected_requested = [p for p in packages if p.lower() in PROTECTED_PACKAGES]
    if protected_requested:
        return {
            "success": False,
            "fatal": True,
            "message": (
                f"Cannot auto-install protected package(s): {protected_requested}. "
                "Please install them manually in your environment "
                "(e.g. `pip install findata mlfinlab`) and re-run the workflow."
            ),
            "installed": [],
        }

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": proc.returncode == 0,
        "fatal": False,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "installed": packages if proc.returncode == 0 else [],
    }



file_search = FileSearchTool(
  vector_store_ids=[
    "vs_69a81b0197a481919e14c2d66197af7d"
  ]
)
mcp = MCPServerStreamableHttp(
  params=MCPServerStreamableHttpParams(url="http://localhost:8090/mcp/"),
  name="fruit_thrower",
  tool_filter={"allowed_tool_names": ["search_code", "get_unit_source", "list_modules", "get_module_summary", "index_repository", "get_index_stats", "generate_function"]},
  require_approval="never",
)
mcp1 = MCPServerSse(
  params=MCPServerSseParams(url="http://localhost:8000/sse"),
  name="data_mcp",
  tool_filter=["search_tools", "get_tool_doc", "list_all_tools", "request_data_source"],
  require_approval="never",
)
mcp2 = MCPServerStreamableHttp(
  params=MCPServerStreamableHttpParams(url="http://localhost:8090/mcp/"),
  name="fruit_thrower",
  tool_filter={"allowed_tool_names": ["search_code", "get_unit_source", "list_modules", "get_module_summary", "index_repository", "get_index_stats", "generate_function"]},
  require_approval="never",
)
mcp3 = MCPServerSse(
  params=MCPServerSseParams(url="http://localhost:8000/sse"),
  name="data_mcp",
  tool_filter=["search_tools", "get_tool_doc", "list_all_tools", "request_data_source"],
  require_approval="never",
)
planner = Agent(
  name="Planner",
  instructions="""You are a quant research workflow planner.
Your job is to convert research ideas into a COMPACT, EXECUTABLE DAG specification
using the available internal library for transformations and for fetching data. These can be queried using the MCPs.
PRIMARY GOAL
- Produce a SMALL DAG (typically 6-10 nodes; depending on the complexity of the task) that delivers the users request.
- If the request is a trading strategy, it should produce asset weight and returns in the end
CRITICAL CONSTRAINTS
- You DO NOT write code.
- You DO NOT assume file access.
- You MUST use existing tools. If a needed step has no tool, you must:
  (a) choose a simpler equivalent using existing tools, or
  (b) use popular libraries to implement it
- Prefer matrix-wide operations. Avoid per-asset loops.
DAG DESIGN RULES (IMPORTANT)
1) Use MACRO NODES, not micro steps.
   - Each node should represent a coherent stage (e.g., \"clean+align data\", \"compute features\", \"build signal\").
   - Do NOT split into tiny nodes like \"dropna\", \"shift\", \"astype\" unless essential.
2) Every node must map directly to ONE tool invocation.
   - If a stage needs multiple tool calls, split into at most 2 nodes for that stage.
3) Variables persist via node outputs only (worker memory resets).
   - Keep intermediate outputs minimal (3-6 total intermediates).
4) ALWAYS include lightweight diagnostics only if a tool supports it (debug flag).
   - Do NOT add separate \"diagnostics nodes\" unless explicitly requested.
REQUIRED WORKFLOW SHAPE
- Data Preparation (1-2 nodes)
- Signal Construction (1-2 nodes)
- Signal Normalization / Risk Controls (1-2 nodes)
- Portfolio Weights (1 node)
- Final asset_returns (1 node)
- Final asset_weights (1 node)
FINAL OUTPUT REQUIREMENT
- The last two nodes MUST output exactly:
  - asset_returns : pandas.DataFrame
  - asset_weights : pandas.DataFrame
OUTPUT FORMAT
Return a JSON list of nodes with:
- id, tool, description, depends_on, parameters, inputs, outputs
Keep descriptions short. Keep parameters explicit.
PLANNING HEURISTICS
- Choose the simplest valid path.
- Reuse existing tools aggressively.
- Avoid \"custom_code\" unless impossible.
- Prefer daily frequency unless specified otherwise.""",
  model="gpt-5",
  tools=[
    file_search,
  ],
  mcp_servers=[mcp, mcp1],
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="low"
    )
  )
)


orchestration_agent = Agent(
  name="Orchestration Agent",
  instructions="""You are a NOTEBOOK ASSEMBLY AGENT.
Your responsibility is to assemble and execute a Jupyter notebook
from a provided ordered task list.
You are NOT a researcher.
You are NOT allowed to invent logic.
You are NOT allowed to modify task implementations.
────────────────────────────────────────
CONTEXT
• A fully defined task list is provided.
• Each task can be completed using python code from the internal libraries
• Documentation of Internal libraries for manipulation of data and getting the data are available via MCPs
• You must connect tasks using notebook cells only.
────────────────────────────────────────
PRIMARY OBJECTIVE
Build a SINGLE executable notebook that:
1. Executes each task in order.
2. Passes outputs explicitly between tasks.
3. Produces two final DataFrames:
      - asset_weights
      - asset_returns
4. Calls the user-defined backtest function exactly once in the final cell.

When reasoning is required, keep it compact.
Do not describe tool actions in text.
Use tools to act.""",

  model="gpt-5",
  tools=[
    add_cell,
    create_notebook,
    # run_cell,
    file_search,
  ],
  mcp_servers=[mcp2, mcp3],
  model_settings=ModelSettings(
    parallel_tool_calls=True,
    store=True,
    reasoning=Reasoning(
      effort="low"
    )
  )
)



validatorandfixingagent = Agent(
    name="ValidatorAndFixingAgent",
    instructions="""You are a NOTEBOOK VALIDATION AND REPAIR AGENT.

Your job is to run the notebook, diagnose every error, and fix it — or escalate cleanly.

════════════════════════════════════════
STEP-BY-STEP LOOP  (repeat until notebook passes or you must stop)
════════════════════════════════════════

1. READ the full notebook with `read_notebook` so you know every cell index and source.
2. RUN the notebook with `validate_run` (use max_cells=9999, timeout=120, kernel_name="python3", prelude="").
3. If success=True → notebook is done. Report success.
4. If success=False → inspect `first_error_cell_index` and the `error` dict (ename, evalue, traceback).

DIAGNOSIS RULES (in priority order)
─────────────────────────────────────
A. ModuleNotFoundError / ImportError
   • Extract the missing module name from `error.evalue`.
   • If the module is "findata" or "mlfinlab":
       – STOP immediately.
       – Return a FATAL message: tell the user to install those packages
         manually (`pip install findata mlfinlab`) and re-run.
       – Do NOT attempt any further fixes.
   • Otherwise: call `install_packages([module_name])`.
       – If install_packages returns fatal=True, STOP and relay the message.
       – If install succeeds, go back to step 2.

B. AttributeError / NameError / TypeError / ValueError / KeyError / other logic errors
   • Use `find_regex_in_notebook_code` to locate the exact cell(s) containing
     the offending symbol or expression.
   • Consult the MCP documentation tools (mcp2 / mcp3) to look up the correct
     API — search for the class/function name, then `get_tool_doc` for details.
   • Use `find_regex_in_notebook_code` again to narrow down documentation output
     if it is large (search for the method or parameter name).
   • Apply the minimal correct fix with `replace_cell`.
   • Go back to step 2.

C. Repeated failure on same cell (same error twice in a row)
   • Try an alternative approach using documentation lookup.
   • If still failing after a second attempt, STOP and request human feedback
     with a clear description of the blocking issue.

GENERAL RULES
─────────────────────────────────────
- Never invent logic. Only fix what is provably wrong based on error messages
  and documentation.
- Never skip cells or comment them out to hide errors.
- Always re-run the full notebook after every fix to confirm no regressions.
- Keep fixes minimal — change only the broken line(s).
- Log every step internally before acting.

OUTPUT FORMAT
─────────────────────────────────────
When finished, return a JSON array — one object per fix attempt:

[
  {
    "step": "<short description of the error>",
    "cell_index": <int or null>,
    "error_type": "<ename>",
    "reasoning": "<why this error occurred and what the fix is>",
    "action": "<what tool was called and what change was made>",
    "result": "<outcome after re-run>"
  }
]

End with a final object:
  { "step": "FINAL", "result": "SUCCESS" | "FATAL: <reason>" | "HUMAN_NEEDED: <reason>" }
""",
    model="gpt-5",
    tools=[
        read_notebook,           # ← new: lets agent read all cells before acting
        validate_run,
        install_packages,        # ← updated: guards findata/mlfinlab
        replace_cell,
        find_regex_in_notebook_code,
    ],
    mcp_servers=[mcp2, mcp3],
    model_settings=ModelSettings(
        parallel_tool_calls=True,
        store=True,
        reasoning=Reasoning(effort="medium"),
    ),
)

edit_planner = Agent(
    name="EditPlanner",
    instructions="""You are a NOTEBOOK EDIT PLANNER.

You receive the content of an existing research notebook and a user request for changes.
Your job is to produce a MINIMAL diff spec — only touch cells that actually need to change.

OUTPUT FORMAT
─────────────────────────────────────
Return ONLY a JSON object (no markdown fences):

{
  "mode": "edit",
  "rationale": "<brief explanation of what needs to change and why>",
  "operations": [
    {"op": "replace",      "cell_index": <int>, "description": "<what this cell should do after the change>"},
    {"op": "insert_after", "cell_index": <int>, "description": "<what the new cell should do>"},
    {"op": "delete",       "cell_index": <int>, "reason":      "<why this cell is removed>"},
    {"op": "append",                            "description": "<what the new cell should do>"}
  ]
}

If the request requires a completely new notebook (fundamentally different topic/approach), return:
{"mode": "new", "rationale": "<why a fresh start is better>"}

RULES
─────────────────────────────────────
- Be minimal. A 2-cell change should have 2 operations, not 10.
- Reference cell indices exactly as shown in the notebook content.
- "replace"      — rewrite an existing cell in place.
- "insert_after" — insert a new cell immediately after cell_index.
- "delete"       — remove a cell entirely.
- "append"       — add a new cell at the very end.
- Preserve variable names and data-flow unless the user explicitly asks to rename things.
- Do NOT write any code yourself — descriptions only. The edit orchestration agent writes the code.
""",
    model="gpt-5",
    model_settings=ModelSettings(store=True, reasoning=Reasoning(effort="low")),
)


edit_orchestration_agent = Agent(
    name="EditOrchestrationAgent",
    instructions="""You are a NOTEBOOK EDIT AGENT.

You receive a diff spec (a JSON operations list) and must apply it to the current notebook.

════════════════════════════════════════
STEP-BY-STEP
════════════════════════════════════════
1. Call read_notebook to see current cell indices and content.
2. Sort and apply operations in this order to avoid index-shifting bugs:
   a. DELETE operations  — apply in DESCENDING cell_index order.
   b. REPLACE operations — apply in DESCENDING cell_index order.
   c. INSERT_AFTER ops   — apply in DESCENDING cell_index order (use insert_cell(cell_index+1, ...)).
   d. APPEND operations  — apply last, in listed order.
3. For every cell you write, use MCP tools to look up correct library APIs.
4. Ensure the notebook remains internally consistent after all operations
   (imports still present, variable names coherent, data flows intact).

TOOLS
─────────────────────────────────────
- read_notebook       — inspect current cells
- replace_cell        — rewrite a cell at an existing index
- insert_cell         — insert a new cell at a given index
- delete_cell         — delete a cell at a given index
- add_cell            — append a cell at the end
- MCP tools (fruit_thrower, data_mcp) — look up internal library APIs

OUTPUT FORMAT
─────────────────────────────────────
Return a JSON array of applied operations:
[
  {"op": "<op>", "cell_index": <int or null>, "action": "<what was done>", "result": "ok" | "<error>"}
]
End with: {"op": "FINAL", "result": "SUCCESS" | "FATAL: <reason>"}
""",
    model="gpt-5",
    tools=[read_notebook, replace_cell, insert_cell, delete_cell, add_cell],
    mcp_servers=[mcp2, mcp3],
    model_settings=ModelSettings(
        parallel_tool_calls=True,
        store=True,
        reasoning=Reasoning(effort="low"),
    ),
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(
    workflow_input: WorkflowInput,
    existing_notebook_path: Optional[str] = None,
    prior_history: Optional[list] = None,
    progress_cb=None,
):
  async def _emit(msg: str):
    if progress_cb:
      await progress_cb({"type": "status", "message": msg})

  logging.info(f"run_workflow mode={'edit' if existing_notebook_path else 'new'}")

  def _fresh_fruit_thrower():
    return MCPServerStreamableHttp(
      params=MCPServerStreamableHttpParams(url="http://localhost:8090/mcp/"),
      name="fruit_thrower",
      tool_filter={"allowed_tool_names": ["search_code", "get_unit_source", "list_modules", "get_module_summary", "index_repository", "get_index_stats", "generate_function"]},
      require_approval="never",
    )

  def _fresh_data_mcp():
    return MCPServerSse(
      params=MCPServerSseParams(url="http://localhost:8000/sse"),
      name="data_mcp",
      tool_filter={"allowed_tool_names": ["search_tools", "get_tool_doc", "list_all_tools", "request_data_source"]},
      require_approval="never",
    )

  with trace("Finagent"):
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = list(prior_history or [])

    # ── EDIT MODE ────────────────────────────────────────────────────────────
    if existing_notebook_path:
      # Read the existing notebook for planner context
      nb = nbformat.read(open(existing_notebook_path), as_version=4)
      nb_summary_lines = []
      for i, cell in enumerate(nb.cells):
        snippet = cell.source[:200].replace("\n", " ")
        nb_summary_lines.append(f"  [{i}] ({cell.cell_type}) {snippet}")
      nb_summary = "\n".join(nb_summary_lines)

      await _emit("Planning changes...")
      edit_planner_input = (
        f"EXISTING NOTEBOOK ({existing_notebook_path}):\n{nb_summary}\n\n"
        f"USER REQUEST: {workflow['input_as_text']}"
      )
      edit_planner_result_temp = await Runner.run(
        edit_planner,
        input=[{"role": "user", "content": [{"type": "input_text", "text": edit_planner_input}]}],
        run_config=RunConfig(),
      )
      diff_spec_raw = edit_planner_result_temp.final_output_as(str).strip()

      # Parse — if mode is "new", fall through to new-notebook flow below
      try:
        diff_spec = json.loads(diff_spec_raw)
      except Exception:
        diff_spec = {"mode": "new", "rationale": "Could not parse diff spec"}

      if diff_spec.get("mode") == "new":
        existing_notebook_path = None   # fall through to new-notebook flow
      else:
        await _emit("Applying edits...")
        async with MCPServerManager([_fresh_fruit_thrower(), _fresh_data_mcp()]) as _edit_mcp_mgr:
          _edit_orch = edit_orchestration_agent.clone(mcp_servers=_edit_mcp_mgr.active_servers)
          edit_orch_result_temp = await Runner.run(
            _edit_orch,
            input=[
              *conversation_history,
              {"role": "user", "content": [{"type": "input_text",
                "text": f"Apply this diff spec to the current notebook:\n{diff_spec_raw}"}]},
            ],
            run_config=RunConfig(),
            max_turns=40,
          )
        conversation_history.extend([item.to_input_item() for item in edit_orch_result_temp.new_items])

        await _emit("Validating notebook...")
        async with MCPServerManager([_fresh_fruit_thrower(), _fresh_data_mcp()]) as _val_mcp_mgr:
          _val = validatorandfixingagent.clone(mcp_servers=_val_mcp_mgr.active_servers)
          val_result_temp = await Runner.run(
            _val,
            input=conversation_history,
            run_config=RunConfig(),
            max_turns=20,
          )
        return {
          "output_text": val_result_temp.final_output_as(str),
          "notebook_path": existing_notebook_path,
          "mode": "edit",
        }

    # ── NEW NOTEBOOK MODE ────────────────────────────────────────────────────
    conversation_history.append({
      "role": "user",
      "content": [{"type": "input_text", "text": workflow["input_as_text"]}],
    })

    await _emit("Planning research...")
    async with MCPServerManager([_fresh_fruit_thrower(), _fresh_data_mcp()]) as _planner_mcp_mgr:
      _planner = planner.clone(mcp_servers=_planner_mcp_mgr.active_servers)
      planner_result_temp = await Runner.run(
        _planner,
        input=[
          *conversation_history,
          {"role": "user", "content": [{"type": "input_text",
            "text": f"Question: {workflow['input_as_text']}"}]},
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69a81a9aedf48190bc2aaab7923d4ae10e4febf1cb72186f"
        })
      )
    conversation_history.extend([item.to_input_item() for item in planner_result_temp.new_items])
    planner_result = {"output_text": planner_result_temp.final_output_as(str)}

    await _emit("Building notebook...")
    async with MCPServerManager([_fresh_fruit_thrower(), _fresh_data_mcp()]) as _orch_mcp_mgr:
      _orchestration_agent = orchestration_agent.clone(mcp_servers=_orch_mcp_mgr.active_servers)
      orchestration_agent_result_temp = await Runner.run(
        _orchestration_agent,
        input=[
          *conversation_history,
          {"role": "user", "content": [{"type": "input_text",
            "text": f"Question: {workflow['input_as_text'], planner_result}"}]},
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69a81a9aedf48190bc2aaab7923d4ae10e4febf1cb72186f",
        }),
        max_turns=40
      )
    conversation_history.extend([item.to_input_item() for item in orchestration_agent_result_temp.new_items])

    await _emit("Validating and fixing...")
    async with MCPServerManager([_fresh_fruit_thrower(), _fresh_data_mcp()]) as _val_mcp_mgr:
      _validatorandfixingagent = validatorandfixingagent.clone(mcp_servers=_val_mcp_mgr.active_servers)
      validatorandfixingagent_result_temp = await Runner.run(
        _validatorandfixingagent,
        input=conversation_history,
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69a81a9aedf48190bc2aaab7923d4ae10e4febf1cb72186f",
        }),
        max_turns=20
      )
    conversation_history.extend([item.to_input_item() for item in validatorandfixingagent_result_temp.new_items])

    return {
      "output_text": validatorandfixingagent_result_temp.final_output_as(str),
      "notebook_path": str(_get_current_path()),
      "mode": "new",
    }
