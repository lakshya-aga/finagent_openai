"""Template draft authoring + acceptance lifecycle.

Glue between ``finagent.agents.template_author`` and the templates folder:

  - ``draft_template(description)`` runs the agent, validates the structural
    contract via static AST check + import probe, writes the source to
    ``finagent/recipes/templates/_drafts/<slug>.py``.
  - ``list_drafts()`` enumerates pending drafts on disk.
  - ``accept_draft(slug)`` moves the file out of ``_drafts/`` so the
    dynamic-discovery loader picks it up on next process start.
  - ``reject_draft(slug)`` deletes the file.

We deliberately do NOT execute ``compile()`` against a real recipe yet — the
user has explicitly deferred runtime verification. The static contract
check (TEMPLATE_NAME / METADATA / supports / compile present and signatures
look sensible) is the only gate today.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents import Runner

from .agents.template_author import template_author_agent


logger = logging.getLogger(__name__)


_TEMPLATES_DIR = Path(__file__).resolve().parent / "recipes" / "templates"
_DRAFTS_DIR = _TEMPLATES_DIR / "_drafts"
_DRAFTS_DIR.mkdir(exist_ok=True)


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{2,29}$")


def _slugify(s: str) -> str:
    """Coerce arbitrary text to our slug shape; falls back to a uuid suffix."""
    s = re.sub(r"[^a-z0-9_]+", "_", s.strip().lower()).strip("_")
    if not s:
        s = "template"
    if not s[0].isalpha():
        s = "t_" + s
    return s[:30]


# ── core API ────────────────────────────────────────────────────────────


async def draft_template(description: str) -> dict[str, Any]:
    """Run the author agent once; persist the draft if it passes the static check.

    Returns
    -------
    dict
        ``{"status": "ok"|"error", "slug": str, "path": str, "errors": [str]}``.
        On error the draft is NOT written.
    """
    if not isinstance(description, str) or len(description.strip()) < 20:
        return {
            "status": "error",
            "slug": "",
            "path": "",
            "errors": ["description must be at least 20 characters of intent"],
        }

    try:
        result = await Runner.run(
            template_author_agent,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": description.strip()}],
            }],
            max_turns=4,
        )
    except Exception as exc:
        logger.exception("template author agent failed")
        return {
            "status": "error",
            "slug": "",
            "path": "",
            "errors": [f"author agent failed: {type(exc).__name__}: {exc}"],
        }

    draft = result.final_output
    if draft is None:
        return {
            "status": "error",
            "slug": "",
            "path": "",
            "errors": ["author agent returned no structured output"],
        }

    slug = _slugify(getattr(draft, "slug", "") or "")
    if not _SLUG_RE.match(slug):
        slug = f"{slug or 'template'}_{uuid.uuid4().hex[:6]}"
    source = getattr(draft, "python_source", "") or ""

    errors = _validate_source(source, expected_template_name=draft.template_name or slug)
    metadata = {
        "slug": slug,
        "template_name": draft.template_name,
        "archetype": draft.archetype,
        "title": draft.title,
        "tagline": draft.tagline,
        "description": draft.description,
        "drafted_at": datetime.now(timezone.utc).isoformat(),
    }

    if errors:
        return {
            "status": "error",
            "slug": slug,
            "path": "",
            "errors": errors,
            "metadata": metadata,
            "source": source,
        }

    path = _write_draft(slug, source, metadata)
    return {
        "status": "ok",
        "slug": slug,
        "path": str(path),
        "errors": [],
        "metadata": metadata,
        "source": source,
    }


def list_drafts() -> list[dict[str, Any]]:
    """Enumerate pending drafts (slug, drafted_at, source preview)."""
    out: list[dict[str, Any]] = []
    for py_file in sorted(_DRAFTS_DIR.glob("*.py")):
        meta_file = py_file.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                meta = {}
        try:
            source = py_file.read_text()
        except OSError:
            source = ""
        out.append({
            "slug": py_file.stem,
            "path": str(py_file),
            "metadata": meta,
            "source_preview": source[:1200],
            "size": len(source),
            "mtime": py_file.stat().st_mtime,
        })
    return sorted(out, key=lambda d: d["mtime"], reverse=True)


def accept_draft(slug: str) -> dict[str, Any]:
    """Move ``_drafts/<slug>.py`` to ``templates/<slug>.py`` so the registry
    picks it up on next process start."""
    src_py = _DRAFTS_DIR / f"{slug}.py"
    src_json = _DRAFTS_DIR / f"{slug}.json"
    if not src_py.exists():
        return {"status": "error", "errors": [f"draft not found: {slug}"]}
    dst_py = _TEMPLATES_DIR / f"{slug}.py"
    if dst_py.exists():
        return {
            "status": "error",
            "errors": [f"a template named {slug!r} already exists in the registry"],
        }
    shutil.move(str(src_py), str(dst_py))
    if src_json.exists():
        src_json.unlink()
    return {"status": "ok", "slug": slug, "path": str(dst_py)}


def reject_draft(slug: str) -> dict[str, Any]:
    src_py = _DRAFTS_DIR / f"{slug}.py"
    src_json = _DRAFTS_DIR / f"{slug}.json"
    removed = False
    if src_py.exists():
        src_py.unlink()
        removed = True
    if src_json.exists():
        src_json.unlink()
    return {"status": "ok" if removed else "error", "slug": slug}


def count_drafts() -> int:
    return len(list(_DRAFTS_DIR.glob("*.py")))


# ── static contract check ───────────────────────────────────────────────


def _validate_source(source: str, expected_template_name: str) -> list[str]:
    """Static + dynamic checks. We run these *before* writing the draft to
    disk so a broken template never appears in _drafts/.

    Stages, in order — each stops the validator on first failure:

      1. ast.parse the module text.
      2. Module-level attrs present: TEMPLATE_NAME, METADATA, supports, compile.
      3. Import the module via importlib (exercises the relative
         ``from ..types import Recipe`` import).
      4. ``METADATA["presets"]`` is a non-empty list with a parseable YAML
         in each entry.
      5. Pick the first preset, build a Recipe from it, run ``compile()``.
      6. ast.parse every code cell in the returned list. This catches the
         JSON-vs-Python-literal class of bugs (null / true / false leaking
         into Python source) that already shipped on the hand-authored
         regime_modeling template — so AI-authored templates never ship
         them again.
    """
    from .recipes.types import recipe_from_yaml as _recipe_from_yaml

    errors: list[str] = []
    if not source.strip():
        return ["empty python_source"]

    # ── 1. parse the module text ───────────────────────────────────────
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"python_source has a syntax error: {exc.msg} (line {exc.lineno})"]

    # ── 2. module-level attrs ──────────────────────────────────────────
    has_template_name = False
    has_metadata = False
    has_supports = False
    has_compile = False

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "TEMPLATE_NAME":
                        has_template_name = True
                    elif target.id == "METADATA":
                        has_metadata = True
        elif isinstance(node, ast.FunctionDef):
            if node.name == "supports":
                has_supports = True
            elif node.name == "compile":
                has_compile = True

    if not has_template_name:
        errors.append("missing module-level TEMPLATE_NAME assignment")
    if not has_metadata:
        errors.append("missing module-level METADATA assignment")
    if not has_supports:
        errors.append("missing supports(recipe) function")
    if not has_compile:
        errors.append("missing compile(recipe) function")

    if errors:
        return errors

    # ── 3. import the module via importlib ─────────────────────────────
    # Spec name MUST sit two levels under finagent.recipes so that
    # `from ..types import Recipe` resolves to finagent.recipes.types.
    # We write the candidate as a sibling of the real templates package
    # so relative imports walk up cleanly.
    tmp_path = _TEMPLATES_DIR / f"_validate_candidate_{uuid.uuid4().hex[:8]}.py"
    module_name = f"finagent.recipes.templates.{tmp_path.stem}"
    module = None
    try:
        tmp_path.write_text(source)
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        if spec is None or spec.loader is None:
            errors.append("could not build import spec for the draft")
        else:
            module = importlib.util.module_from_spec(spec)
            # The candidate uses ``from ..types import Recipe``. Relative
            # imports require the module to be registered in sys.modules
            # *before* exec_module runs, otherwise the import machinery
            # can't resolve __package__.
            import sys as _sys
            _sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                errors.append(f"draft fails to import: {type(exc).__name__}: {exc}")
            else:
                for attr in ("TEMPLATE_NAME", "METADATA", "supports", "compile"):
                    if not hasattr(module, attr):
                        errors.append(f"imported draft is missing attr: {attr}")
                tn = getattr(module, "TEMPLATE_NAME", None)
                if tn and expected_template_name and tn != expected_template_name:
                    errors.append(
                        f"TEMPLATE_NAME ({tn!r}) doesn't match the slug "
                        f"({expected_template_name!r})"
                    )

        if errors or module is None:
            return errors

        # ── 4-6. compile() the first preset and ast-parse every cell ──
        meta = getattr(module, "METADATA", {}) or {}
        presets = meta.get("presets") or []
        if not isinstance(presets, list) or not presets:
            errors.append(
                "METADATA.presets is empty — every template must ship at "
                "least one working preset"
            )
            return errors

        first = presets[0]
        if not isinstance(first, dict) or "yaml" not in first:
            errors.append("METADATA.presets[0] missing 'yaml' key")
            return errors

        try:
            sample_recipe = _recipe_from_yaml(first["yaml"])
        except Exception as exc:
            errors.append(
                f"METADATA.presets[0].yaml fails Recipe validation: "
                f"{type(exc).__name__}: {exc}"
            )
            return errors

        try:
            cells = module.compile(sample_recipe)
        except Exception as exc:
            errors.append(
                f"compile() raised on the first preset: "
                f"{type(exc).__name__}: {exc}"
            )
            return errors

        if not isinstance(cells, list) or not cells:
            errors.append("compile() returned no cells (must return a non-empty list)")
            return errors

        for i, cell in enumerate(cells):
            ctype = getattr(cell, "cell_type", None)
            content = getattr(cell, "content", None)
            if ctype not in ("code", "markdown"):
                errors.append(f"cell {i}: cell_type must be 'code' or 'markdown', got {ctype!r}")
                continue
            if not isinstance(content, str):
                errors.append(f"cell {i}: content must be a string, got {type(content).__name__}")
                continue
            if ctype != "code":
                continue
            try:
                ast.parse(content)
            except SyntaxError as exc:
                errors.append(
                    f"cell {i} ({getattr(cell, 'dag_node_id', '?')}): "
                    f"generated Python source has a syntax error — "
                    f"{exc.msg} at line {exc.lineno}"
                )
                continue
            # Heuristic anti-pattern check for the JSON-leak family.
            # If the generated source contains a bare ``null``, ``true``,
            # or ``false`` token outside a string literal, it would
            # NameError on execution. Detect via tokenize so we don't
            # false-positive on string content.
            json_leak = _detect_json_literal_leak(content)
            if json_leak:
                errors.append(
                    f"cell {i} ({getattr(cell, 'dag_node_id', '?')}): "
                    f"generated source contains bare {json_leak!r} "
                    "outside a string literal — looks like json.dumps was "
                    "interpolated into Python source. Use repr() for "
                    "kwargs or wrap with json.loads() for a constant."
                )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            # Drop the stub module from sys.modules so the next validation
            # round doesn't pick up stale state if the auto-discovery loop
            # ever scans the templates directory mid-validation.
            sys = __import__("sys")
            sys.modules.pop(module_name, None)
        except OSError:
            pass
    return errors


def _detect_json_literal_leak(source: str) -> str | None:
    """Return the offending bare token (``null`` / ``true`` / ``false``)
    if the source has one outside a string literal, else None.

    Uses ``tokenize`` so quoted occurrences (legitimate string content)
    don't false-positive.
    """
    import io
    import token as _token
    import tokenize

    bad = {"null", "true", "false"}
    try:
        for tok in tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline):
            if tok.type == _token.NAME and tok.string in bad:
                return tok.string
    except tokenize.TokenizeError:
        # Source had a tokenize-level issue; ast.parse already caught
        # the structural problem so we just bail here.
        return None
    return None


def _write_draft(slug: str, source: str, metadata: dict[str, Any]) -> Path:
    py_path = _DRAFTS_DIR / f"{slug}.py"
    meta_path = _DRAFTS_DIR / f"{slug}.json"
    py_path.write_text(source)
    meta_path.write_text(json.dumps(metadata, indent=2))
    return py_path
