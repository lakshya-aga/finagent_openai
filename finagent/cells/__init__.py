"""Cell-level utilities — role classifier, splitter, lint hooks.

The notebook author (whether human or LLM) often packs multiple jobs
into one cell — imports + data load, or train + chart, or several
preprocess steps + a model fit. That's fine for exploration but bad
for production: the train/infer split (Phase 4) needs to know which
cells *only* run at training time and which cells re-execute every
inference, and a cell that does both can't be cleanly partitioned.

This package provides:

  * :func:`classify_cell` — heuristic AST-based role detection
  * :func:`needs_split` — convenience guard that returns True when a
    cell mixes incompatible roles
  * :func:`tag_notebook` — walk a notebook on disk and stamp each
    cell's metadata with its detected roles (idempotent)

The auto-split agent (Phase 3 stretch goal) is not shipped here yet —
the orchestrator-prompt update + post-execution lint warnings are the
forcing function. If a cell consistently mixes roles we add an explicit
LLM splitter behind the scenes; for now we surface the warning to the
author.
"""

from __future__ import annotations

from .classifier import (
    ROLES,
    classify_cell,
    needs_split,
    tag_notebook,
)
from .splitter import split_notebook


__all__ = ["ROLES", "classify_cell", "needs_split", "tag_notebook", "split_notebook"]
