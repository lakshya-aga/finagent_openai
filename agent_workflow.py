

from agents import function_tool, FileSearchTool, HostedMCPTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
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

    PYTHON_PATH = os.environ.get("FINAGENT_PYTHON", "/Users/lakshya/miniconda3/envs/finagentv2/bin/python")   
    logging.info(f"_run_code_in_kernel using python_path={PYTHON_PATH}")
    km = KernelManager()
    km.kernel_cmd = [PYTHON_PATH, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
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
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
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
    Run the full current notebook in a single kernel execution.
    """
    PYTHON_PATH = os.environ.get("FINAGENT_PYTHON", "/Users/lakshya/miniconda3/envs/finagentv2/bin/python")
    path = _get_current_path()
    logging.info(f"TOOL CALL: validate_run path={path} max_cells={max_cells} timeout={timeout}")

    nb = _load_notebook()

    indexed_code_cells = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            indexed_code_cells.append((i, cell.source))
            if len(indexed_code_cells) >= max_cells:
                break

    if not indexed_code_cells:
        return {
            "success": True,
            "path": str(path),
            "message": "No code cells to execute",
            "executed_cells": 0,
        }

    chunks = []
    if prelude:
        chunks.append(f"# --- prelude ---\n{prelude}")
    for idx, source in indexed_code_cells:
        chunks.append(f"# --- cell {idx} ---\n{source}")

    result = _run_code_in_kernel(
        "\n\n".join(chunks),
        timeout=timeout,
    )

    # Identify the failing cell from traceback markers
    first_error_cell = None
    if not result["success"] and result.get("error"):
        traceback_text = "\n".join(result["error"].get("traceback", []))
        for idx, _ in reversed(indexed_code_cells):
            if f"cell {idx}" in traceback_text:
                first_error_cell = idx
                break

    return {
        "success": result["success"],
        "path": str(path),
        "status": result["status"],
        "executed_cells": len(indexed_code_cells),
        "first_error_cell_index": first_error_cell,
        "outputs": result["outputs"],
        "error": result["error"],
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
mcp = HostedMCPTool(tool_config={
  "type": "mcp",
  "server_label": "fruit_thrower",
  "allowed_tools": [
    "search_code",
    "get_unit_source",
    "list_modules",
    "get_module_summary",
    "index_repository",
    "get_index_stats"
  ],
  "headers": {
    "Authorization": "Bearer a30b5aa8f95c9fa548d3418051250e5e14e1122d45b0f801"
  },
  "require_approval": "never",
  "server_url": "https://antiques-viewpicture-cashiers-chargers.trycloudflare.com/mcp/"
})
mcp1 = HostedMCPTool(tool_config={
  "type": "mcp",
  "server_label": "data_mcp",
  "allowed_tools": [
    "search_tools",
    "get_tool_doc",
    "list_all_tools"
  ],
  "headers": {
    "Authorization": "Bearer a30b5aa8f95c9fa548d3418051250e5e14e1122d45b0f801"
  },
  "require_approval": "never",
  "server_url": "https://antiques-viewpicture-cashiers-chargers.trycloudflare.com/data-mcp/"
})
mcp2 = HostedMCPTool(tool_config={
  "type": "mcp",
  "server_label": "fruit_thrower",
  "allowed_tools": [
    "search_code",
    "get_unit_source",
    "list_modules",
    "get_module_summary",
    "index_repository",
    "get_index_stats"
  ],
  "headers": {
    "Authorization": "Bearer a30b5aa8f95c9fa548d3418051250e5e14e1122d45b0f801"
  },
  "require_approval": "never",
  "server_description": "code examples information",
  "server_url": "https://antiques-viewpicture-cashiers-chargers.trycloudflare.com/mcp/"
})
mcp3 = HostedMCPTool(tool_config={
  "type": "mcp",
  "server_label": "data_mcp",
  "allowed_tools": [
    "search_tools",
    "get_tool_doc",
    "list_all_tools"
  ],
  "headers": {
    "Authorization": "Bearer a30b5aa8f95c9fa548d3418051250e5e14e1122d45b0f801"
  },
  "require_approval": "never",
  "server_description": "mcp that provides code for fetching data - standardise input data format",
  "server_url": "https://antiques-viewpicture-cashiers-chargers.trycloudflare.com/data-mcp/"
})
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
    mcp,
    mcp1
  ],
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
    mcp2,
    mcp3
  ],
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
        mcp2,
        mcp3                   
    ],
    model_settings=ModelSettings(
        parallel_tool_calls=True,
        store=True,
        reasoning=Reasoning(effort="medium"),
    ),
)

class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):

  logging.info(f"TOOL CALL: {locals()}")
  with trace("Finagent"):
    state = {

    }
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    planner_result_temp = await Runner.run(
      planner,
      input=[
        *conversation_history,
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": f"Question: {workflow['input_as_text']}"
            }
          ]
        }
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69a81a9aedf48190bc2aaab7923d4ae10e4febf1cb72186f"
      })
    )

    conversation_history.extend([item.to_input_item() for item in planner_result_temp.new_items])

    planner_result = {
      "output_text": planner_result_temp.final_output_as(str)
    }
    orchestration_agent_result_temp = await Runner.run(
      orchestration_agent,
      input=[
        *conversation_history,
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": f"Question: {workflow['input_as_text'], planner_result}"
            }
          ]
        }
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69a81a9aedf48190bc2aaab7923d4ae10e4febf1cb72186f",
      }),
      max_turns=40
    )

    conversation_history.extend([item.to_input_item() for item in orchestration_agent_result_temp.new_items])

    orchestration_agent_result = {
      "output_text": orchestration_agent_result_temp.final_output_as(str)
    }
    validatorandfixingagent_result_temp = await Runner.run(
      validatorandfixingagent,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69a81a9aedf48190bc2aaab7923d4ae10e4febf1cb72186f",
      }),
      max_turns=20
    )

    conversation_history.extend([item.to_input_item() for item in validatorandfixingagent_result_temp.new_items])

    validatorandfixingagent_result = {
      "output_text": validatorandfixingagent_result_temp.final_output_as(str)
    }
    return validatorandfixingagent_result
