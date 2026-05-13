"""install_packages tool. Refuses to auto-install protected packages.

Runtime pip installs are gated by the ALLOW_RUNTIME_INSTALL env var
(default: 'false'). When disabled, install_packages returns an error
instead of mutating the environment. An allowlist mode is also available
via ALLOWED_RUNTIME_PACKAGES (comma-separated).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Dict, List

from agents import function_tool


PROTECTED_PACKAGES = {"findata", "mlfinlab"}

# Controls whether install_packages can actually run pip.
# Set to "true" to allow runtime installs (dev/sandbox mode).
# Default is "false" for production safety.
_ALLOW_RUNTIME_INSTALL = os.environ.get("ALLOW_RUNTIME_INSTALL", "false").lower() in (
    "true", "1", "yes",
)

# Optional allowlist: if set, only these packages can be installed at runtime.
# Comma-separated. Empty = no allowlist filtering (all non-protected allowed).
_ALLOWED_PACKAGES_RAW = os.environ.get("ALLOWED_RUNTIME_PACKAGES", "")
_ALLOWED_PACKAGES: set[str] | None = (
    {p.strip().lower() for p in _ALLOWED_PACKAGES_RAW.split(",") if p.strip()}
    if _ALLOWED_PACKAGES_RAW.strip()
    else None
)


@function_tool
def install_packages(packages: List[str]) -> Dict[str, Any]:
    """Install python packages into the current environment.

    Gated by ALLOW_RUNTIME_INSTALL env var (default: false).
    Refuses to auto-install anything in PROTECTED_PACKAGES — those must be set
    up by the operator (typically because they require manual configuration
    such as private indexes or hand-built wheels).
    """
    logging.info(f"TOOL CALL: install_packages {packages} (allowed={_ALLOW_RUNTIME_INSTALL})")

    if not packages:
        return {"success": True, "message": "No packages requested", "installed": []}

    # Gate: runtime installs disabled
    if not _ALLOW_RUNTIME_INSTALL:
        return {
            "success": False,
            "fatal": True,
            "reason": "runtime_install_disabled",
            "message": (
                "Runtime package installation is disabled (ALLOW_RUNTIME_INSTALL=false). "
                "The requested packages must be pre-installed in the environment. "
                "Add them to requirements.txt and rebuild the container, or set "
                "ALLOW_RUNTIME_INSTALL=true for dev/sandbox use."
            ),
            "installed": [],
        }

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

    # Allowlist check
    if _ALLOWED_PACKAGES is not None:
        blocked = [p for p in packages if p.lower() not in _ALLOWED_PACKAGES]
        if blocked:
            return {
                "success": False,
                "fatal": True,
                "reason": "not_in_allowlist",
                "message": (
                    f"Package(s) {blocked} are not in ALLOWED_RUNTIME_PACKAGES. "
                    "Add them to the allowlist or pre-install in requirements.txt."
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
