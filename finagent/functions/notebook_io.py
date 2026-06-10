"""Pure-IO helpers for the active notebook on disk.

These functions never call agent tools — they are the building blocks the
@function_tool wrappers in cell_tools.py and notebook_tools.py call into.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


# Per-process "next-notebook-name" hint set by the workflow before it
# invokes the orchestration agent. When the chat-intent classifier flags
# a request as a fresh notebook build, finagent.workflow runs the
# name_suggester agent → calls set_next_notebook_name(slug) → the next
# create_notebook() tool call picks up that slug + appends a datetime
# suffix. Cleared after one read so a subsequent run uses fresh state.
_NEXT_NOTEBOOK_BASE: str | None = None

# The notebook the agent is currently building/editing. Set by
# ``create_notebook`` (and ``set_active_notebook_path``) so that every
# subsequent cell-tool call (add_cell / replace_cell / validate_run / …)
# operates on the SAME file — even when its name doesn't match the legacy
# ``notebook_N`` pattern.
#
# Why this exists: ``_get_current_path`` used to return the highest
# ``notebook_N.ipynb`` by integer index. Once notebooks got human-readable
# names (``<slug>__<timestamp>.ipynb``), those names parse to index -1 and
# were IGNORED — so a fresh build that created ``my-strategy__….ipynb`` would
# have every add_cell silently write into a leftover ``notebook_6.ipynb``
# instead. This pointer pins the active file for the whole build.
_ACTIVE_NOTEBOOK_PATH: "Path | None" = None


def set_next_notebook_name(base_name: str | None) -> None:
    """Workflow-level hint for the next ``create_notebook`` call. Pass a
    kebab-case base (e.g. ``cross-sector-momentum-v1``); the path helper
    appends ``__YYYYMMDD-HHMMSS`` automatically. Pass ``None`` to clear.

    Also clears the active-notebook pointer: calling this signals that a
    fresh build is starting, so any active pointer left over from a prior
    build in the same process must not leak into this one (otherwise the
    idempotency guard in ``create_notebook`` would return the OLD notebook)."""
    global _NEXT_NOTEBOOK_BASE, _ACTIVE_NOTEBOOK_PATH
    _NEXT_NOTEBOOK_BASE = base_name
    _ACTIVE_NOTEBOOK_PATH = None


def get_active_notebook_path() -> "Path | None":
    """The notebook pinned for the current build, or None."""
    return _ACTIVE_NOTEBOOK_PATH


def set_active_notebook_path(path: "Path | str | None") -> None:
    """Pin the notebook every subsequent cell-tool call resolves to.

    ``create_notebook`` calls this with the file it just created so the
    orchestrator's ``add_cell`` calls land in the right place. Editing an
    existing notebook (edit flow) can also call this with the target path.
    Pass ``None`` to clear and fall back to mtime resolution."""
    global _ACTIVE_NOTEBOOK_PATH
    _ACTIVE_NOTEBOOK_PATH = Path(path) if path is not None else None


def _path_for_named(base_name: str, when: datetime | None = None) -> Path:
    """Build a notebook path from a user/LLM-suggested base name.

    Format: ``<slug>__<YYYYMMDD-HHMMSS>.ipynb``. The datetime suffix
    guarantees uniqueness without scanning the directory; collisions on
    the same second are disambiguated with a ``__2``, ``__3``... counter.
    """
    when = when or datetime.utcnow()
    slug = _slugify(base_name)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    base = f"{slug}__{stamp}"
    candidate = _OUTPUTS_DIR / f"{base}.ipynb"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = _OUTPUTS_DIR / f"{base}__{i}.ipynb"
        if not candidate.exists():
            return candidate
        i += 1


def _get_latest_path() -> Path:
    """Return the next path for a freshly-created notebook.

    Two paths:
      - If the workflow set a name hint via ``set_next_notebook_name``,
        consume it and produce ``<slug>__<datetime>.ipynb``. The hint is
        cleared after one read so subsequent builds don't reuse it.
      - Otherwise fall back to the legacy ``notebook_N.ipynb`` counter
        (manual runs that bypass the chat-intent classifier).
    """
    global _NEXT_NOTEBOOK_BASE
    if _NEXT_NOTEBOOK_BASE:
        base = _NEXT_NOTEBOOK_BASE
        _NEXT_NOTEBOOK_BASE = None  # consume; don't reuse
        return _path_for_named(base)

    i = 1
    while True:
        path = _OUTPUTS_DIR / f"notebook_{i}.ipynb"
        if not path.exists():
            return path
        i += 1


# Restrict file names to a conservative ASCII charset. Whitespace and
# anything that could break shell quoting / URL encoding gets folded to
# underscore. Two trailing underscores collapse to one.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(s: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", s.strip()).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "recipe"


def _path_for_recipe(
    recipe_name: str,
    fingerprint: str | None,
    when: datetime | None = None,
) -> Path:
    """Build a semantic notebook path for a recipe run.

    Format: ``<recipe>__<short-hash>__<YYYYMMDD-HHMM>.ipynb``. Reviewer
    feedback was that the previous ``notebook_N.ipynb`` scheme made the
    Notebooks list useless once a team has more than a handful of files
    (UAT P1 #3). The new pattern is searchable by recipe and sorted by
    timestamp without relying on metadata.

    Collisions are resolved by appending a counter (``__2``, ``__3``…).
    """
    when = when or datetime.utcnow()
    slug = _slugify(recipe_name)
    short = (fingerprint or "")[:8] or "nofp"
    stamp = when.strftime("%Y%m%d-%H%M")
    base = f"{slug}__{short}__{stamp}"
    candidate = _OUTPUTS_DIR / f"{base}.ipynb"
    if not candidate.exists():
        return candidate
    # Two runs of the same recipe finishing in the same minute — append a
    # disambiguating counter rather than overwriting.
    i = 2
    while True:
        candidate = _OUTPUTS_DIR / f"{base}__{i}.ipynb"
        if not candidate.exists():
            return candidate
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
    """Return the notebook the agent is currently working on.

    Resolution order:
      1. ``_ACTIVE_NOTEBOOK_PATH`` — the file ``create_notebook`` (or the
         edit flow) pinned for this build. This is authoritative and is the
         fix for the named-notebook routing bug: a build that created
         ``my-strategy__20260609-161526.ipynb`` must keep writing there, not
         silently fall through to a leftover ``notebook_6.ipynb``.
      2. Newest ``.ipynb`` by mtime — covers manual runs that bypass the
         chat flow (no active pointer set) and legacy ``notebook_N`` files.
         mtime is correct regardless of naming scheme, so it handles both
         ``notebook_N`` and ``<slug>__<timestamp>`` files uniformly.

    The previous implementation preferred the highest ``notebook_N`` integer,
    which IGNORED human-readable names (they parse to index -1) and routed
    cell writes into stale numbered notebooks.
    """
    global _ACTIVE_NOTEBOOK_PATH
    if _ACTIVE_NOTEBOOK_PATH is not None and _ACTIVE_NOTEBOOK_PATH.exists():
        return _ACTIVE_NOTEBOOK_PATH

    any_ipynb = [f for f in os.listdir(_OUTPUTS_DIR) if f.endswith(".ipynb")]
    if not any_ipynb:
        raise FileNotFoundError(f"No notebooks found in {_OUTPUTS_DIR}")
    any_ipynb.sort(key=lambda f: (_OUTPUTS_DIR / f).stat().st_mtime)
    return _OUTPUTS_DIR / any_ipynb[-1]


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
