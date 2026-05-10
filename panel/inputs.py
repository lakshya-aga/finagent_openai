"""Inference input fetcher.

When the scheduler runs ``infer.py`` for a registered signal it needs
the *current-day* features that the model was trained on. The training
notebook had access to a fetch_fn (yfinance, internal tick API,
whatever); the inference job needs the same fetch_fn but operating on
"today" instead of the training window.

The SDK doesn't try to remember how a notebook fetched data — it just
provides a small wrapper that the caller hands a fetcher to and the
model manifest to, so the inference job has a clear "this is what I
need" surface.

Pattern:

    from panel import get_inference_inputs, load_model

    model = load_model("spy-momentum-252d")
    df = get_inference_inputs(
        signal_name="spy-momentum-252d",
        fetch_fn=lambda window: yfinance.download("SPY", start=window[0], end=window[1]),
    )
    score = model.predict(df[features])
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from . import models as _models


logger = logging.getLogger(__name__)


def get_inference_inputs(
    signal_name: str,
    *,
    fetch_fn: Callable[[tuple[str, str]], Any],
    lookback_days: int | None = None,
    end_date: datetime | None = None,
) -> Any:
    """Materialise the input window the inference job needs to score
    the model behind ``signal_name``.

    Parameters
    ----------
    signal_name
        Name passed to ``save_model`` / ``export_signal``.
    fetch_fn
        Caller-supplied fetcher. Receives a ``(start_iso, end_iso)``
        tuple and is expected to return whatever the training notebook
        used (typically a ``pandas.DataFrame``). The SDK doesn't impose
        a return type; whatever the fetcher returns is what the
        inference job sees.
    lookback_days
        Window length. Defaults to whatever the model manifest declared
        as ``lookback_days``, falling back to 365 if absent.
    end_date
        End of the window. Defaults to "now (UTC)". Tests / backfills
        can pass a fixed date.

    Returns
    -------
    Whatever ``fetch_fn`` returned. The SDK is deliberately untyped here
    — different signals need different shapes (dataframes, arrays,
    dicts of frames), and forcing a single return type would be
    presumptuous.
    """
    manifest = _models.model_manifest(signal_name)
    meta = manifest.get("metadata") or {}
    if lookback_days is None:
        lookback_days = int(meta.get("lookback_days") or 365)
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)
    window = (
        start_date.date().isoformat(),
        end_date.date().isoformat(),
    )
    logger.info(
        "panel.get_inference_inputs: signal=%s window=%s..%s",
        signal_name, window[0], window[1],
    )
    return fetch_fn(window)
