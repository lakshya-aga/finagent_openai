"""Template registry.

Discovers every sibling ``*.py`` module in this package (other than
``__init__.py`` itself) and registers it by ``TEMPLATE_NAME``. The
``_drafts/`` subdirectory is intentionally NOT scanned — drafts only
become live after a human accepts them via ``accept_draft()`` (which
moves the file out of ``_drafts/`` and into this directory).

Each template module must expose the four-attribute contract:

    TEMPLATE_NAME : str
    METADATA      : dict
    supports(recipe) -> bool
    compile(recipe)  -> list[CellSpec]
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

from . import regime_modeling


logger = logging.getLogger(__name__)


def _build_registry() -> dict[str, Any]:
    """Auto-discover sibling template modules and build the name→module map.

    Hand-authored seed templates (``regime_modeling`` etc.) ship in this
    package directly. AI-authored templates land here after acceptance.
    The ``_drafts/`` subdirectory is excluded so unaccepted drafts can't
    affect production behavior.
    """
    registry: dict[str, Any] = {regime_modeling.TEMPLATE_NAME: regime_modeling}
    pkg_path = list(__path__)  # type: ignore[name-defined]
    for finder, name, ispkg in pkgutil.iter_modules(pkg_path):
        if ispkg or name.startswith("_") or name == "regime_modeling":
            continue
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except Exception:
            logger.exception("failed to import template module %s", name)
            continue
        template_name = getattr(module, "TEMPLATE_NAME", None)
        if not isinstance(template_name, str) or not template_name:
            logger.warning("template module %s is missing TEMPLATE_NAME; skipping", name)
            continue
        if template_name in registry:
            logger.warning(
                "template name collision: %s already registered; %s ignored",
                template_name, name,
            )
            continue
        for attr in ("METADATA", "supports", "compile"):
            if not hasattr(module, attr):
                logger.warning(
                    "template module %s missing attr %s; skipping", name, attr,
                )
                break
        else:
            registry[template_name] = module
    return registry


REGISTRY = _build_registry()

__all__ = ["REGISTRY", "regime_modeling"]
