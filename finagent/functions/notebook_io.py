"""Pure-IO helpers for the active notebook on disk.

These functions never call agent tools — they are the building blocks the
@function_tool wrappers in cell_tools.py and notebook_tools.py call into.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def _get_latest_path() -> Path:
    """Return the next unused notebook_N.ipynb path in outputs/."""
    i = 1
    while True:
        path = _OUTPUTS_DIR / f"notebook_{i}.ipynb"
        if not path.exists():
            return path
        i += 1


def _notebook_index(name: str) -> int:
    """Extract the trailing N from `notebook_N.ipynb`. Returns -1 if not parseable."""
    if not name.endswith(".ipynb"):
        return -1
    stem = name[: -len(".ipynb")]
    if "_" not in stem:
        return -1
    tail = stem.rsplit("_", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return -1


def _get_current_path() -> Path:
    """Return the highest-numbered notebook_N.ipynb in outputs/.

    Lexicographic sorting is wrong here: ``notebook_10.ipynb`` < ``notebook_2.ipynb``
    on string comparison would route every cell-tool call to the older file
    once the workflow generates its 10th notebook. Sort by the integer stem
    instead so the "current" notebook tracks the freshest run.
    """
    candidates = [
        f for f in os.listdir(_OUTPUTS_DIR)
        if f.endswith(".ipynb") and _notebook_index(f) >= 0
    ]
    if not candidates:
        # Fall back to "any .ipynb" by mtime if nothing matches the
        # notebook_N convention — protects against custom-named uploads.
        any_ipynb = [f for f in os.listdir(_OUTPUTS_DIR) if f.endswith(".ipynb")]
        if not any_ipynb:
            raise FileNotFoundError(f"No notebooks found in {_OUTPUTS_DIR}")
        any_ipynb.sort(key=lambda f: (_OUTPUTS_DIR / f).stat().st_mtime)
        return _OUTPUTS_DIR / any_ipynb[-1]
    candidates.sort(key=_notebook_index)
    return _OUTPUTS_DIR / candidates[-1]


def _ensure_parent_dir(path) -> None:
    logging.info(f"TOOL CALL: _ensure_parent_dir {path}")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_notebook():
    path = _get_current_path()
    logging.info(f"TOOL CALL: _load_notebook {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Notebook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _save_notebook(nb, path) -> None:
    logging.info(f"TOOL CALL: _save_notebook {path}")
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def _make_cell(cell_type: str, content: str):
    if cell_type == "code":
        return new_code_cell(content)
    if cell_type == "markdown":
        return new_markdown_cell(content)
    raise ValueError("cell_type must be 'code' or 'markdown'")
