"""pytest fixtures + dep-availability markers shared by every test.

The host machine that runs these tests has pandas + numpy + sklearn +
nbformat + nbclient + matplotlib + statsmodels but NOT yfinance / openai
/ jupyter-kernel-spec for the "finagent-python" kernel. The docker
container has the full stack. Tests guard their imports so the host can
run the unit tier and CI / docker can run the integration tier.

Markers
-------

  needs_yfinance   - skipped when yfinance import fails
  needs_openai     - skipped when openai import fails (and OPENAI_API_KEY unset)
  needs_finagent_kernel - skipped when the 'finagent-python' Jupyter kernel
                          spec isn't installed (notebook-execution tests)
  needs_mcp        - skipped when the data-mcp / fruit-thrower URLs aren't
                     reachable (HEAD request timeout)
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def _kernel_installed(name: str) -> bool:
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        return name in KernelSpecManager().find_kernel_specs()
    except Exception:
        return False


def _url_reachable(url: str, timeout: float = 0.5) -> bool:
    try:
        u = urlparse(url)
        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_yfinance: requires yfinance importable")
    config.addinivalue_line("markers", "needs_openai: requires openai importable + OPENAI_API_KEY env var")
    config.addinivalue_line("markers", "needs_finagent_kernel: requires the 'finagent-python' kernelspec installed")
    config.addinivalue_line("markers", "needs_mcp: requires the data-mcp / fruit-thrower URLs reachable")


def pytest_collection_modifyitems(config, items):
    skip_if = {
        "needs_yfinance": (
            not _can_import("yfinance"),
            "yfinance not installed",
        ),
        "needs_openai": (
            not (_can_import("openai") and os.environ.get("OPENAI_API_KEY")),
            "openai or OPENAI_API_KEY missing",
        ),
        "needs_finagent_kernel": (
            not _kernel_installed("finagent-python"),
            "'finagent-python' Jupyter kernel not installed",
        ),
        "needs_mcp": (
            not all(_url_reachable(u) for u in (
                os.environ.get("FRUIT_THROWER_URL", "http://localhost:8090/mcp/"),
                os.environ.get("DATA_MCP_URL", "http://localhost:8000/sse"),
            )),
            "MCP servers not reachable",
        ),
    }
    for item in items:
        for marker_name, (should_skip, reason) in skip_if.items():
            if marker_name in item.keywords and should_skip:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    """Point the panel SDK + experiments DB at a temp directory so tests
    don't pollute the real outputs/. Yields the path."""
    db_path = str(tmp_path / "experiments.db")
    monkeypatch.setenv("FINAGENT_OUTPUTS_DIR", str(tmp_path))
    # Both env names — panel SDK prefers FINAGENT_EXPERIMENT_DB but
    # accepts FINAGENT_DB; finagent.experiments only reads the long name.
    monkeypatch.setenv("FINAGENT_EXPERIMENT_DB", db_path)
    monkeypatch.setenv("FINAGENT_DB", db_path)
    # Reset any cached panel._store paths so re-importing in a test sees
    # the new env vars.
    import importlib, sys
    for modname in list(sys.modules):
        if modname.startswith("panel"):
            importlib.reload(sys.modules[modname])
    return tmp_path
