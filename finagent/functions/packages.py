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
    return {
        "success": proc.returncode == 0,
        "fatal": False,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "installed": packages if proc.returncode == 0 else [],
    }
