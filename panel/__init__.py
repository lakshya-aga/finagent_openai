"""``panel`` — research-notebook → production-signal SDK.

A small, opinionated facade that lets a notebook author do four things
without thinking about file layout, manifest schemas, or DB plumbing:

    import panel

    # 1. Persist a fitted model (sklearn / statsmodels / anything pickleable).
    panel.save_model("spy-momentum-252d", model, metadata={...})

    # 2. Reload it later (used by the inference job).
    model = panel.load_model("spy-momentum-252d")

    # 3. Materialise the latest features the model needs to score "today".
    df = panel.get_inference_inputs("spy-momentum-252d", fetch_fn=...)

    # 4. Publish the resulting time series as a registered signal.
    panel.export_signal("spy-momentum-252d", series, metadata={...})

The SDK lives at the *repo root* (not under ``finagent.``) so notebook
code that runs in a Jupyter kernel can ``import panel`` cleanly without
caring about the package layout of the agent that produced it. That's
the same reason ``findata`` lives at the root in the data-mcp repo.

Storage layout
--------------
Everything goes under ``outputs/`` next to the existing notebooks/ and
experiments.db:

    outputs/
        models/<signal_name>/
            model.pkl            # pickled estimator
            manifest.json        # framework, signature, feature list, vintage
        signals/<signal_name>/
            series.parquet       # the exported time-series
            manifest.json        # frequency, universe, description, ...

The signal registry (the ``signals`` + ``signal_versions`` tables in
``outputs/experiments.db``) is populated on every ``export_signal()``
call so the dashboard endpoint in ``app.py`` can list signals without
walking the disk.

Why a thin SDK and not a heavier framework
------------------------------------------
The notebook author is a quant first, an engineer second. They want to
write four lines and be done. The SDK swallows: path resolution,
manifest schema validation, parquet round-tripping, DB upsert, and the
``FINAGENT_RUN_ID`` env-var auto-detection that links a freshly-created
signal back to the run that produced it.

Public symbols are re-exported here so callers can ``from panel import
save_model, load_model, export_signal, get_inference_inputs`` without
caring about submodule layout.
"""

from __future__ import annotations

from .models import save_model, load_model, model_manifest, list_models
from .signals import (
    export_signal,
    load_signal,
    signal_manifest,
    list_signals,
    register_signal,  # exposed for tests; real callers use export_signal
)
from .inputs import get_inference_inputs


__all__ = [
    # Model lifecycle
    "save_model",
    "load_model",
    "model_manifest",
    "list_models",
    # Signal lifecycle
    "export_signal",
    "load_signal",
    "signal_manifest",
    "list_signals",
    "register_signal",
    # Inference inputs
    "get_inference_inputs",
]
