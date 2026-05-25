"""Directional-Change event detection + indicator computation.

Faithful port of the paper's ``directional_change.py``. The DC framework
samples the price series at extreme + DC-confirmation points (PEXT,
PDCC) rather than at fixed time intervals — the data dictates the
sampling. See Section 2.2 of the paper / Appendix A of Chen & Tsang.

Three indicators per DC trend:

  * TMV(n) = (PEXT(n) - PEXT(n-1)) / (PEXT(n-1) * θ)   total price move
  * T(n)   = tEXT(n) - tEXT(n-1)                       trend duration (days)
  * R(n)   = TMV(n) * θ / T(n)                         time-adjusted return

All three are inputs to the HMM that follows.

Compared to the original we keep ``get_DC_data_v2`` (the streaming
implementation) — the v1 implementation in the original repo had a
trend-flip edge case that v2 fixed.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# ── data loader ─────────────────────────────────────────────────────


def get_data(
    tickers: list[str],
    start_date: str,
    delta_hours: float = 6.5,
    end_date: str = "2022-12-31",
) -> pd.Series:
    """Pull daily OHLC from yfinance and interleave Open + Close with
    a ``delta_hours`` offset, producing a semi-daily price series.

    The paper picks a 12h delta on FX (24h day) but a half-trading-day
    (6.5h) delta on equities — the NYSE session is 09:30–16:00. The
    resulting series is one point at 00:00 (close) and one at 06:30
    (the following open) per calendar day, ~doubling the number of
    observations for the HMM.

    Returns a Series indexed by the offset timestamps, single column
    (the first ticker). Multi-ticker support is identical to the
    original — only the first ticker's column is returned (the paper
    only used ^GSPC).
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    # yf.download returns a MultiIndex (field, ticker) when given a
    # single ticker as a list, but a flat column index when given a
    # string. Normalise to flat single-series for both shapes.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0].dropna()
        open_ = data["Open"].iloc[:, 0].dropna()
    else:
        close = data["Close"].dropna()
        open_ = data["Open"].dropna()

    open_.index = open_.index + pd.Timedelta(hours=float(delta_hours))
    interleaved = pd.concat([close, open_]).sort_index()
    # name the series after the first ticker so downstream plot
    # legends look right.
    interleaved.name = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
    return interleaved


# ── DC event detection ──────────────────────────────────────────────


def _pct_change(start: float, end: float) -> float:
    return (end - start) / start


def get_DC_data(prices: pd.Series, theta: float) -> list[tuple]:
    """Walk the price series and emit (DCC_time, DCC_price, EXT_time,
    EXT_price) tuples one per directional-change confirmation.

    Streaming algorithm:
      * track the running high + low since the last trend reversal
      * when the price drops θ below the running high, we have
        confirmed a downtrend — emit (now, last_high) and reset
      * symmetrically for upturns

    Equivalent to the paper's ``get_DC_data_v2``. Linear in the
    length of the input.
    """
    if len(prices) < 2:
        return []

    last_high = last_low = prices.iloc[0]
    last_low_time = last_high_time = prices.index[0]
    is_downward_overshoot = is_upward_overshoot = False
    out: list[tuple] = []

    for timestamp, current_price in prices.iloc[1:].items():
        if _pct_change(last_high, current_price) <= -theta:
            if not is_downward_overshoot:
                # downtrend confirmed
                out.append((timestamp, current_price, last_high_time, last_high))
                is_downward_overshoot = True
                is_upward_overshoot = False
            last_high = current_price
            last_high_time = timestamp
        elif _pct_change(last_low, current_price) >= theta:
            if not is_upward_overshoot:
                # uptrend confirmed
                out.append((timestamp, current_price, last_low_time, last_low))
                is_upward_overshoot = True
                is_downward_overshoot = False
            last_low = current_price
            last_low_time = timestamp

        # always update the running extremes (used by the NEXT trend
        # in case the current overshoot keeps extending)
        if current_price <= last_low:
            last_low = current_price
            last_low_time = timestamp
        if current_price >= last_high:
            last_high = current_price
            last_high_time = timestamp

    return out


# ── indicator extraction ────────────────────────────────────────────


def split_dcc_ext(dc: list[tuple]) -> tuple[list, list, list, list]:
    """Unzip the DC tuple list into four parallel lists.
    Mirrors the paper's ``get_DCC_EXT``."""
    if not dc:
        return [], [], [], []
    dcc = [row[1] for row in dc]
    dcc_index = [row[0] for row in dc]
    ext = [row[3] for row in dc]
    ext_index = [row[2] for row in dc]
    return dcc, dcc_index, ext, ext_index


def get_TMV(dc: list[tuple], theta: float) -> pd.Series:
    """Total Price Movement series — one value per DC trend, indexed
    by the EXT timestamp ending that trend. First trend has no
    predecessor so it's dropped (mirrors the paper)."""
    _, _, ext, idx = split_dcc_ext(dc)
    ext_s = pd.Series(data=ext, index=idx)
    return ext_s.pct_change().dropna() / theta


def get_T(dc: list[tuple]) -> pd.Series:
    """Time-for-completion series. Hours are converted to fractional
    days (the unit Chen & Tsang use): ``days + hours/24``."""
    _, _, ext, idx = split_dcc_ext(dc)
    if len(idx) < 2:
        return pd.Series([], dtype=float)
    diffs = pd.Series(idx).diff().dropna()
    t = diffs.apply(lambda x: x.days + (x.seconds // 3600) / 24)
    t.index = idx[1:]
    return t


def get_R(tmv: pd.Series, T: pd.Series, theta: float) -> pd.Series:
    """Time-adjusted return: R = TMV * θ / T. Both inputs are aligned
    on the EXT-timestamp index so division is element-wise."""
    return tmv * theta / T


def compute_indicators(
    prices: pd.Series,
    theta: float,
) -> tuple[list[tuple], pd.Series, pd.Series, pd.Series]:
    """One-shot: DC events → (DC, TMV, T, R). Each indicator series
    is aligned on the SAME EXT-timestamp index, so a row in any of
    the three corresponds to the same trend in the others."""
    dc = get_DC_data(prices, theta)
    tmv = get_TMV(dc, theta)
    T = get_T(dc)
    R = get_R(tmv, T, theta)
    return dc, tmv, T, R
