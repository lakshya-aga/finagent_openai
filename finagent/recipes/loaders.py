"""Recipe data-source loaders.

Templated notebook cells call ``loaders.load(spec)`` with a JSON-encoded
``DataSource`` dict. The dispatcher below maps ``kind`` → concrete fetch
function. New kinds plug in here without touching templates.

Returns a `pandas.DataFrame` indexed by date (where applicable).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def load(spec_json: str | dict[str, Any]) -> pd.DataFrame:
    spec = json.loads(spec_json) if isinstance(spec_json, str) else dict(spec_json)
    kind = spec.pop("kind")
    fn = _DISPATCH.get(kind)
    if fn is None:
        raise ValueError(f"unknown DataSource kind: {kind!r}")
    return fn(**spec)


def _yfinance(tickers: list[str], start: str, end: str | None = None,
              interval: str = "1d", **kwargs) -> pd.DataFrame:
    # Prefer findata wrapper when available (consistent column shape).
    try:
        from findata.equity_prices import get_equity_prices

        return get_equity_prices(tickers, start, end or "2099-12-31")
    except Exception:
        import yfinance as yf

        df = yf.download(tickers, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        return df


def _fred(series_ids: list[str], start: str | None = None,
          end: str | None = None, api_key: str | None = None,
          **kwargs) -> pd.DataFrame:
    from findata.fred import get_fred_series

    return get_fred_series(series_ids, start_date=start, end_date=end, api_key=api_key)


def _csv(path: str, **kwargs) -> pd.DataFrame:
    from findata.file_reader import get_file_data

    return get_file_data(path, **kwargs)


def _fama_french(factor_model: str = "3", start: str | None = None,
                 end: str | None = None, **kwargs) -> pd.DataFrame:
    from findata.fama_french import get_fama_french_factors

    return get_fama_french_factors(factor_model, start_date=start, end_date=end)


def _cboe(symbols: list[str] | None = None, start: str | None = None,
          end: str | None = None, **kwargs) -> pd.DataFrame:
    from findata.cboe_volatility import get_cboe_volatility_indices

    return get_cboe_volatility_indices(symbols, start_date=start, end_date=end)


def _coingecko(coin_id: str, vs_currency: str = "usd", days: int | str = 90,
               **kwargs) -> pd.DataFrame:
    from findata.coingecko import get_coingecko_ohlcv

    return get_coingecko_ohlcv(coin_id, vs_currency=vs_currency, days=days)


def _fin_kit(function: str, kwargs: dict[str, Any] | None = None,
             **_more) -> pd.DataFrame:
    """Generic adapter — looks up `function` as a dotted path under fin_kit."""
    import importlib

    module_path, fn_name = function.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    return fn(**(kwargs or {}))


_DISPATCH = {
    "yfinance": _yfinance,
    "fred": _fred,
    "csv": _csv,
    "fama_french": _fama_french,
    "cboe": _cboe,
    "coingecko": _coingecko,
    "fin_kit": _fin_kit,
}
