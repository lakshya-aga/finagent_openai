import logging
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import nbformat
from jupyter_client import KernelManager
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from pydantic import BaseModel

try:
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import create_react_agent

    LANGGRAPH_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - handled at runtime
    ChatOpenAI = None
    HumanMessage = None
    StateGraph = None
    START = None
    END = None
    LANGGRAPH_IMPORT_ERROR = exc

    def tool(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


logging.basicConfig(
    filename="finagent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

PROTECTED_PACKAGES = {"findata", "mlfinlab"}
DEFAULT_PYTHON_PATH = os.environ.get(
    "FINAGENT_PYTHON",
    "/Users/lakshya/miniconda3/envs/finagentv2/bin/python",
)
DEFAULT_MODEL = os.environ.get("FINAGENT_MODEL", "gpt-4.1")
MAX_FILE_PREVIEW_CHARS = 8000

PLANNER_PROMPT = """You are a quant research workflow planner.
Convert the user's request into a compact execution plan for a notebook-building agent.

Rules:
- Return a JSON list of macro steps.
- Keep the workflow small and executable.
- Prefer existing project code, common Python libraries, and clear dataflow.
- If this is a trading strategy, the final outputs must be `asset_returns` and `asset_weights`.
- Do not write full notebook code here.
- Use tools when you need local project context.

Each JSON object should contain:
- id
- description
- depends_on
- outputs
"""

ASSEMBLER_PROMPT = """You are a notebook assembly agent.
Build a single executable Jupyter notebook from the given request and plan.

Rules:
- Use tools to create and edit the notebook.
- Keep the notebook readable and modular.
- Prefer explicit intermediate variables over hidden state.
- Produce final DataFrames named `asset_weights` and `asset_returns`.
- If a backtest is requested, add exactly one final backtest cell.
- Do not narrate tool calls.
"""

VALIDATOR_PROMPT = """You are a notebook validation and repair agent.

Loop until the notebook validates or you hit a hard blocker:
1. Read the notebook.
2. Run validation.
3. If it fails, identify the first broken cell and apply the smallest justified fix.
4. Re-run validation after each fix.

Rules:
- Never remove functionality to hide an error.
- Stop immediately if a protected package such as `findata` or `mlfinlab` is missing.
- Keep fixes minimal and evidence-based.
- Return a compact JSON summary of attempts and a final status.
"""


def _outputs_dir() -> Path:
    path = Path("outputs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_latest_path() -> Path:
    i = 1
    while True:
        path = _outputs_dir() / f"notebook_{i}.ipynb"
        if not path.exists():
            return path
        i += 1


def _get_current_path() -> Path:
    files = sorted(_outputs_dir().glob("notebook_*.ipynb"))
    if not files:
        raise FileNotFoundError("No notebook found in outputs/. Create one first.")
    return files[-1]


def _ensure_parent_dir(path: str | Path) -> None:
    parent = Path(path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _load_notebook() -> nbformat.NotebookNode:
    path = _get_current_path()
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return nbformat.read(handle, as_version=4)


def _save_notebook(nb: nbformat.NotebookNode, path: str | Path) -> None:
    _ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        nbformat.write(nb, handle)


def _make_cell(cell_type: str, content: str) -> nbformat.NotebookNode:
    if cell_type == "code":
        return new_code_cell(content)
    if cell_type == "markdown":
        return new_markdown_cell(content)
    raise ValueError("cell_type must be 'code' or 'markdown'")


def _serialize_output(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    logging.info("Executing notebook code in kernel")

    km = KernelManager()
    km.kernel_cmd = [DEFAULT_PYTHON_PATH, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    km.start_kernel()

    try:
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=timeout)

        msg_id = kc.execute(code)
        outputs: List[Dict[str, Any]] = []
        execute_reply = None
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

        status = execute_reply.get("content", {}).get("status") if execute_reply else None
        error = next((output for output in outputs if output["output_type"] == "error"), None)

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


@tool
def create_notebook() -> Dict[str, Any]:
    """Create a new empty notebook under outputs/."""
    path = _get_latest_path()
    nb = new_notebook(cells=[])
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
    return {"success": True, "path": str(path), "num_cells": 0}


@tool
def add_cell(cell_type: str, content: str) -> Dict[str, Any]:
    """Append a notebook cell."""
    path = _get_current_path()
    nb = _load_notebook()
    nb.cells.append(_make_cell(cell_type, content))
    _save_notebook(nb, path)
    return {
        "success": True,
        "path": str(path),
        "cell_index": len(nb.cells) - 1,
        "cell_type": cell_type,
        "num_cells": len(nb.cells),
    }


@tool
def replace_cell(cell_index: int, cell_type: str, content: str) -> Dict[str, Any]:
    """Replace a notebook cell by index."""
    path = _get_current_path()
    nb = _load_notebook()

    if cell_index < 0 or cell_index >= len(nb.cells):
        raise IndexError(f"cell_index {cell_index} out of range for notebook with {len(nb.cells)} cells")

    nb.cells[cell_index] = _make_cell(cell_type, content)
    _save_notebook(nb, path)
    return {
        "success": True,
        "path": str(path),
        "cell_index": cell_index,
        "cell_type": cell_type,
        "num_cells": len(nb.cells),
    }


@tool
def run_cell(cell_index: int, timeout: int = 120) -> Dict[str, Any]:
    """Execute a single code cell, replaying prior code cells as prelude state."""
    path = _get_current_path()
    nb = _load_notebook()

    if cell_index < 0 or cell_index >= len(nb.cells):
        raise IndexError(f"cell_index {cell_index} out of range for notebook with {len(nb.cells)} cells")

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
    final_code = f"{prelude_code}\n\n# --- target cell {cell_index} ---\n{target_cell.source}" if prelude_code else target_cell.source

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


@tool
def install_packages(packages: List[str]) -> Dict[str, Any]:
    """Install Python packages unless they are protected project dependencies."""
    if not packages:
        return {"success": True, "installed": [], "message": "No packages requested"}

    protected_requested = [pkg for pkg in packages if pkg.lower() in PROTECTED_PACKAGES]
    if protected_requested:
        return {
            "success": False,
            "fatal": True,
            "installed": [],
            "message": (
                f"Cannot auto-install protected package(s): {protected_requested}. "
                "Install them manually and re-run the workflow."
            ),
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


@tool
def find_regex_in_notebook_code(regex_pattern: str, case_sensitive: bool = False) -> Dict[str, Any]:
    """Search code and markdown cells in the current notebook with a regex."""
    flags = 0 if case_sensitive else re.IGNORECASE
    nb = _load_notebook()
    matches = []
    pattern = re.compile(regex_pattern, flags)

    for idx, cell in enumerate(nb.cells):
        source = cell.source or ""
        for match in pattern.finditer(source):
            start, end = match.span()
            snippet = source[max(0, start - 80): min(len(source), end + 80)]
            matches.append(
                {
                    "cell_index": idx,
                    "cell_type": cell.cell_type,
                    "match_text": match.group(0),
                    "span": [start, end],
                    "snippet": snippet,
                }
            )

    return {
        "success": True,
        "regex_pattern": regex_pattern,
        "case_sensitive": case_sensitive,
        "num_matches": len(matches),
        "matches": matches,
    }


@tool
def read_notebook() -> Dict[str, Any]:
    """Read the current notebook including code cell outputs."""
    nb = _load_notebook()
    cells = []
    for i, cell in enumerate(nb.cells):
        entry = {"cell_index": i, "cell_type": cell.cell_type, "source": cell.source}
        if cell.cell_type == "code":
            entry["outputs"] = cell.get("outputs", [])
        cells.append(entry)
    return {"success": True, "num_cells": len(nb.cells), "cells": cells}


@tool
def validate_run(max_cells: int = 9999, timeout: int = 120, prelude: str = "") -> Dict[str, Any]:
    """Run the current notebook in one kernel and report the first failing code cell if any."""
    path = _get_current_path()
    nb = _load_notebook()

    indexed_code_cells = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            indexed_code_cells.append((i, cell.source))
            if len(indexed_code_cells) >= max_cells:
                break

    if not indexed_code_cells:
        return {"success": True, "path": str(path), "message": "No code cells to execute", "executed_cells": 0}

    chunks = []
    if prelude:
        chunks.append(f"# --- prelude ---\n{prelude}")
    for idx, source in indexed_code_cells:
        chunks.append(f"# --- cell {idx} ---\n{source}")

    result = _run_code_in_kernel("\n\n".join(chunks), timeout=timeout)

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


@tool
def search_workspace(query: str, glob: str = "*.py") -> Dict[str, Any]:
    """Search the workspace with ripgrep for project context."""
    cmd = ["rg", "-n", "--glob", glob, query, "."]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": proc.returncode in {0, 1},
        "query": query,
        "glob": glob,
        "stdout": proc.stdout[:MAX_FILE_PREVIEW_CHARS],
        "stderr": proc.stderr[:4000],
    }


@tool
def read_workspace_file(path: str, start_line: int = 1, end_line: int = 250) -> Dict[str, Any]:
    """Read a text file from the current workspace."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    file_path = file_path.resolve()

    workspace_root = Path.cwd().resolve()
    if workspace_root not in file_path.parents and file_path != workspace_root:
        raise ValueError(f"Path {file_path} is outside the workspace")

    lines = file_path.read_text(encoding="utf-8").splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    snippet = "\n".join(lines[start_idx:end_idx])
    return {
        "success": True,
        "path": str(file_path),
        "start_line": start_line,
        "end_line": end_idx,
        "content": snippet[:MAX_FILE_PREVIEW_CHARS],
    }


class WorkflowInput(BaseModel):
    input_as_text: str


class WorkflowState(TypedDict, total=False):
    user_request: str
    planner_output: str
    assembly_output: str
    validation_output: str
    final_output: Dict[str, Any]


_planner_agent = None
_assembler_agent = None
_validator_agent = None
_workflow_graph = None


def _require_langgraph() -> None:
    if LANGGRAPH_IMPORT_ERROR is not None:
        raise ImportError(
            "LangGraph dependencies are not installed. Install `langgraph`, "
            "`langchain-openai`, and `langchain-core` before running this workflow."
        ) from LANGGRAPH_IMPORT_ERROR


def _build_model() -> ChatOpenAI:
    _require_langgraph()
    return ChatOpenAI(model=DEFAULT_MODEL)


def _extract_last_message_text(result: Dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif hasattr(part, "text"):
                    parts.append(part.text)
            if parts:
                return "\n".join(parts)
    return ""


def _get_planner_agent():
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = create_react_agent(
            _build_model(),
            tools=[search_workspace, read_workspace_file],
            prompt=PLANNER_PROMPT,
        )
    return _planner_agent


def _get_assembler_agent():
    global _assembler_agent
    if _assembler_agent is None:
        _assembler_agent = create_react_agent(
            _build_model(),
            tools=[create_notebook, add_cell, replace_cell, run_cell, search_workspace, read_workspace_file],
            prompt=ASSEMBLER_PROMPT,
        )
    return _assembler_agent


def _get_validator_agent():
    global _validator_agent
    if _validator_agent is None:
        _validator_agent = create_react_agent(
            _build_model(),
            tools=[read_notebook, validate_run, replace_cell, install_packages, find_regex_in_notebook_code],
            prompt=VALIDATOR_PROMPT,
        )
    return _validator_agent


async def _planner_node(state: WorkflowState) -> WorkflowState:
    request = state["user_request"]
    result = await _get_planner_agent().ainvoke({"messages": [HumanMessage(content=request)]})
    return {"planner_output": _extract_last_message_text(result)}


async def _assembler_node(state: WorkflowState) -> WorkflowState:
    prompt = (
        f"User request:\n{state['user_request']}\n\n"
        f"Execution plan:\n{state['planner_output']}\n\n"
        "Create the notebook now."
    )
    result = await _get_assembler_agent().ainvoke({"messages": [HumanMessage(content=prompt)]})
    return {"assembly_output": _extract_last_message_text(result)}


async def _validator_node(state: WorkflowState) -> WorkflowState:
    prompt = (
        f"User request:\n{state['user_request']}\n\n"
        f"Planner output:\n{state['planner_output']}\n\n"
        f"Assembler output:\n{state['assembly_output']}\n\n"
        "Validate and fix the notebook."
    )
    result = await _get_validator_agent().ainvoke({"messages": [HumanMessage(content=prompt)]})
    validation_output = _extract_last_message_text(result)
    return {
        "validation_output": validation_output,
        "final_output": {"output_text": validation_output},
    }


def build_workflow_graph():
    global _workflow_graph
    if _workflow_graph is None:
        _require_langgraph()
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", _planner_node)
        graph.add_node("assemble", _assembler_node)
        graph.add_node("validate", _validator_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "assemble")
        graph.add_edge("assemble", "validate")
        graph.add_edge("validate", END)
        _workflow_graph = graph.compile()
    return _workflow_graph


async def run_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    logging.info("Running workflow for prompt: %s", workflow_input.input_as_text)
    app = build_workflow_graph()
    result = await app.ainvoke({"user_request": workflow_input.input_as_text})
    return result["final_output"]
