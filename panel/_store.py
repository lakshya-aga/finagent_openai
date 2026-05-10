"""Filesystem layout + manifest helpers shared by ``models`` and ``signals``.

Centralised so ``models.py`` and ``signals.py`` agree on directory
structure, manifest schema version, and atomic-write semantics. Keeping
this module dependency-free (stdlib only) means the panel SDK loads
even in barebones notebook environments.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# ── Paths ──────────────────────────────────────────────────────────────

_OUTPUTS_DIR = Path(
    os.environ.get(
        "FINAGENT_OUTPUTS_DIR",
        # Default: <repo_root>/outputs — same place notebooks + the
        # experiments DB live. Resolved relative to this file so the SDK
        # works whether it's imported from inside or outside the repo.
        str(Path(__file__).resolve().parents[1] / "outputs"),
    )
).resolve()

MODELS_DIR = _OUTPUTS_DIR / "models"
SIGNALS_DIR = _OUTPUTS_DIR / "signals"


def outputs_dir() -> Path:
    """Repo-root ``outputs/`` directory. Useful for tests."""
    return _OUTPUTS_DIR


def models_dir() -> Path:
    return MODELS_DIR


def signals_dir() -> Path:
    return SIGNALS_DIR


# ── Naming ─────────────────────────────────────────────────────────────

# Restrict signal/model names to a conservative slug — they end up as
# directory names + DB primary keys + URL path segments.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


def validate_name(name: str) -> str:
    """Reject names that aren't lowercase kebab-case slugs.

    The slug doubles as a directory name, a DB primary key, and a URL
    path segment, so we want a single conservative grammar. Empty string,
    underscores, spaces, dots, and uppercase are all rejected. Length
    bounds: 3-64 chars (first + middle{1,62} + last).
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid name {name!r}: must be lowercase kebab-case, "
            "3-64 chars, alphanumeric or hyphens, no leading/trailing hyphen"
        )
    return name


# ── Manifest read/write ────────────────────────────────────────────────

# Bumped whenever the manifest schema changes in a backward-incompatible
# way. Readers should tolerate older versions; writers always emit the
# current one.
MANIFEST_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic JSON write — temp file + rename so a crash mid-write
    doesn't leave a half-written manifest that subsequent reads choke on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    enriched.setdefault("written_at", utc_now_iso())
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:
        # If anything fails, scrub the temp so we don't leak it.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Run-id auto-detection ──────────────────────────────────────────────

def current_run_id() -> str | None:
    """The recipe_workflow exports ``FINAGENT_RUN_ID`` into the kernel
    env so notebooks can attribute their outputs back to the run that
    spawned them. Returns None when running outside a workflow (e.g. in
    a one-off interactive notebook)."""
    val = os.environ.get("FINAGENT_RUN_ID")
    return val if val else None
