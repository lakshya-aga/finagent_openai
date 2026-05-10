"""Train / infer notebook → standalone Python script extractor.

Once a research notebook has been tagged by ``classifier.tag_notebook``,
the splitter partitions its cells into two scripts:

  * ``train.py``  — runs end-to-end on a fresh kernel and (re-)fits the
                    model. Includes: imports + data_load + preprocess +
                    train + signal_export + summary.
  * ``infer.py``  — runs end-to-end on a fresh kernel using the saved
                    model. Includes: imports + data_load (today's
                    window) + preprocess + load_model + score +
                    signal_export + summary.

The split is rule-based off the role tags. Cells tagged ``chart``,
``eval``, ``other`` are skipped (charts are an analyst affordance, not
production code). Cells tagged ``signal_export`` are emitted in *both*
scripts because the inference job also needs to publish results.

Limitations
-----------
This is a structural split, not a code refactor. If a cell does
``model = OLS(y, X).fit()`` and a later cell uses ``model.predict(...)``,
both end up in train.py — but infer.py needs to ``panel.load_model(...)``
to recreate ``model``. The convention we ship in the orchestrator
prompt is:

    # in the train cell:
    model = OLS(y, X).fit()
    panel.save_model("spy-momentum-252d", model, metadata={...})

    # in the eval cell (which becomes infer.py only):
    model = panel.load_model("spy-momentum-252d")  # or use the existing 'model'
    preds = model.predict(...)

The splitter trusts that convention. When it sees a ``train`` cell it
emits a comment in infer.py reminding the author to ``panel.load_model``.
A future LLM-assisted variant can do the rewrite automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import nbformat


logger = logging.getLogger(__name__)


# Cells with at least one of these roles go into train.py.
_TRAIN_ROLES = {"imports", "data_load", "preprocess", "train", "signal_export", "summary"}

# Cells with at least one of these roles go into infer.py.
_INFER_ROLES = {"imports", "data_load", "preprocess", "eval", "signal_export", "summary"}


def _cell_roles(cell) -> set[str]:
    return set((cell.metadata.get("finagent") or {}).get("roles") or [])


def _is_executable_code(cell) -> bool:
    if cell.cell_type != "code":
        return False
    if not (cell.source or "").strip():
        return False
    return True


def split_notebook(notebook_path: Path, *, output_dir: Path | None = None) -> dict[str, Path]:
    """Walk ``notebook_path`` and emit ``train.py`` + ``infer.py``.

    Parameters
    ----------
    notebook_path
        Path to the (tagged) ``.ipynb`` file. Must have been processed
        by ``classifier.tag_notebook`` first; cells without role tags
        are skipped with a warning.
    output_dir
        Where to write the scripts. Defaults to a sibling directory
        ``<notebook_dir>/scripts/<notebook_stem>/``.

    Returns
    -------
    ``{"train": Path, "infer": Path}`` pointing at the two scripts.
    """
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"notebook not found: {notebook_path}")
    nb = nbformat.read(str(notebook_path), as_version=4)

    if output_dir is None:
        output_dir = notebook_path.parent / "scripts" / notebook_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_lines: list[str] = []
    infer_lines: list[str] = []

    train_lines.append(_HEADER.format(kind="train", source=notebook_path.name))
    infer_lines.append(_HEADER.format(kind="infer", source=notebook_path.name))

    untagged = 0
    for i, cell in enumerate(nb.cells):
        if not _is_executable_code(cell):
            continue
        roles = _cell_roles(cell)
        if not roles:
            untagged += 1
            continue

        if roles & _TRAIN_ROLES:
            train_lines.append(f"\n# --- cell {i} :: roles={sorted(roles)} ---\n")
            train_lines.append(cell.source.rstrip() + "\n")

        # Infer.py logic. Two paths:
        #   1. Cell is a *pure* train cell (or train + imports) — emit
        #      a "load saved model" reminder comment and skip the body.
        #      The agent's discipline is to keep model_use cells separate
        #      from model_fit cells, so infer should already have a
        #      `panel.load_model(...)` cell elsewhere.
        #   2. Cell has at least one infer-side role — copy as-is.
        if "train" in roles and not (roles - {"train", "imports"}):
            infer_lines.append(
                f"\n# --- cell {i} :: TRAIN cell omitted from infer "
                "(load via panel.load_model) ---\n"
            )
        elif roles & _INFER_ROLES:
            infer_lines.append(f"\n# --- cell {i} :: roles={sorted(roles)} ---\n")
            infer_lines.append(cell.source.rstrip() + "\n")

    if untagged:
        logger.warning(
            "split_notebook: %d cells in %s had no role tags — re-run "
            "classifier.tag_notebook(...) first",
            untagged, notebook_path.name,
        )

    train_path = output_dir / "train.py"
    infer_path = output_dir / "infer.py"
    train_path.write_text("".join(train_lines))
    infer_path.write_text("".join(infer_lines))
    return {"train": train_path, "infer": infer_path}


_HEADER = '''"""Auto-generated {kind}.py from {source}.

Do NOT edit by hand. Regenerate via:

    from finagent.cells.splitter import split_notebook
    split_notebook("outputs/{source}")

This script is what the scheduler runs on a cron — keep it self-
contained, deterministic, and idempotent. If you need to change
training/inference behaviour, edit the source notebook and re-extract.
"""

from __future__ import annotations

'''
