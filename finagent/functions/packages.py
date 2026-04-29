"""install_packages tool. Refuses to auto-install protected packages."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, Dict, List

from agents import function_tool


PROTECTED_PACKAGES = {"findata", "mlfinlab"}


@function_tool
def install_packages(packages: List[str]) -> Dict[str, Any]:
    """Install python packages into the current environment.

    Refuses to auto-install anything in PROTECTED_PACKAGES — those must be set
    up by the operator (typically because they require manual configuration
    such as private indexes or hand-built wheels).
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

    # If pip itself reports the package doesn't exist on PyPI, mark fatal so
    # the validator stops retrying and can take the rewrite-or-escalate path.
    # We deliberately don't try to predict which names are "fake" — only act
    # on what pip actually told us.
    not_on_pypi = (
        proc.returncode != 0
        and (
            "No matching distribution found" in (proc.stderr or "")
            or "Could not find a version that satisfies the requirement"
            in (proc.stderr or "")
        )
    )

    out: Dict[str, Any] = {
        "success": proc.returncode == 0,
        "fatal": not_on_pypi,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "installed": packages if proc.returncode == 0 else [],
    }
    if not_on_pypi:
        out["reason"] = "module_not_on_pypi"
        out["message"] = (
            f"pip could not find {packages} on PyPI. The orchestrator may "
            "have referenced a non-existent module name. Locate the bad "
            "import via find_regex_in_notebook_code, then either inline "
            "the missing logic with replace_cell or escalate clearly to the "
            "user — do NOT keep retrying install_packages."
        )
    return out
