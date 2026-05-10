"""Model persistence — pickled estimator + JSON manifest on disk.

The contract is intentionally narrow: any pickleable object can be
saved (sklearn pipelines, statsmodels Results wrappers, raw dicts, even
a tuple of ``(scaler, model)``). The SDK doesn't try to introspect the
estimator — that's the caller's job, declared via the ``metadata``
field. This keeps the SDK from coupling to any specific ML framework.

Layout:

    outputs/models/<name>/
        model.pkl
        manifest.json   # {framework, signature, features, training_window, ...}

Replacement semantics: ``save_model("foo", new_model)`` overwrites the
existing files for ``foo``. The DB layer (in ``signals.py``) keeps the
historical record via ``signal_versions`` rows; the on-disk model file
is always the latest. If you need point-in-time reproducibility, capture
the recipe fingerprint + git SHA in the manifest and re-run the
training notebook.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Mapping

from . import _store


logger = logging.getLogger(__name__)


def _model_path(name: str) -> Path:
    return _store.models_dir() / name / "model.pkl"


def _manifest_path(name: str) -> Path:
    return _store.models_dir() / name / "manifest.json"


def save_model(
    name: str,
    model: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Pickle ``model`` into ``outputs/models/<name>/model.pkl`` and
    write a manifest alongside it.

    Parameters
    ----------
    name
        Lowercase kebab-case slug. Doubles as the directory name and
        the foreign key linking models to signals downstream.
    model
        Any pickleable object. Sklearn pipelines, statsmodels Results,
        a tuple of ``(scaler, model)``, even a callable — anything that
        round-trips through ``pickle.dumps``.
    metadata
        Free-form dict declaring what this model is. Recommended keys:

        * ``framework``: 'sklearn' | 'statsmodels' | 'xgboost' | ...
        * ``signature``: human-readable like 'OLS(y ~ rolling_sharpe_252)'
        * ``features``: list[str] of input column names
        * ``target``: name of the target column
        * ``training_window``: [start_iso, end_iso]
        * ``hyperparameters``: dict of the actual fitted hyperparams
        * ``cv_strategy``: 'walk-forward' | 'k-fold' | ...

        Anything else passes through unchanged. The dict is JSON-
        serialised so values must be json-friendly (dates are
        stringified by ``write_manifest``).

    Returns
    -------
    Path to the manifest file (the canonical "this model exists"
    pointer).
    """
    name = _store.validate_name(name)
    path = _model_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "name": name,
        "kind": "model",
        "model_path": str(path),
        "size_bytes": path.stat().st_size,
        "metadata": dict(metadata or {}),
        "run_id": _store.current_run_id(),
    }
    manifest_path = _manifest_path(name)
    _store.write_manifest(manifest_path, manifest)
    logger.info("panel.save_model: wrote %s (%d bytes)", path, path.stat().st_size)
    return manifest_path


def load_model(name: str) -> Any:
    """Round-trip the model saved by ``save_model``.

    Raises ``FileNotFoundError`` if no model has been saved for ``name``.
    Caller is responsible for any framework-specific re-hydration steps
    beyond pickle (e.g. attaching a fresh device for torch tensors).
    """
    name = _store.validate_name(name)
    path = _model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no model saved for {name!r} at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def model_manifest(name: str) -> dict[str, Any]:
    """Return the manifest dict for a saved model. Useful for inference
    callers that need to know which features/columns the model expects
    without re-loading the pickle."""
    name = _store.validate_name(name)
    return _store.read_manifest(_manifest_path(name))


def list_models() -> list[dict[str, Any]]:
    """List every saved model's manifest. Used by the dashboard's
    'available models' picker. Skips directories without a manifest
    (incomplete writes, manual rm)."""
    base = _store.models_dir()
    if not base.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        mf = child / "manifest.json"
        if not mf.exists():
            continue
        try:
            out.append(_store.read_manifest(mf))
        except Exception:
            logger.exception("panel.list_models: skipped malformed manifest %s", mf)
    return out
