"""Three trading strategies + metrics.

Faithful port of the paper's ``trading_strategy.py`` and
``generate_data.py``, rewritten in numpy to dodge the chained-indexing
warnings (and outright failures, on pandas ≥ 2.2) of the original.

Strategy state machine (identical for all three; only entry direction
and close-condition vary):

  state OPEN with position p ∈ {-1, +1}:
    * close at the next DCC or EXT_DCC event (and, for the regime-
      dependent strategy, on a regime flip).
    * MTM happens implicitly via the daily-return computation.

  state FLAT:
    * if |TMV| >= 0.5 (the trading threshold the paper calls
      ``threshold``, distinct from the DC threshold θ), open a new
      position in the appropriate direction (sign of TMV for momentum,
      -sign(TMV) for mean-reversion).

  TMV is tracked at EVERY price point as
      TMV(t) = (price(t) - last_EXT_price) / (last_EXT_price * θ)
  so the strategy can fire mid-trend, not only on DC events.

The three strategies (Table 1 in the paper):

  * CT1 / "mean_reverting": always mean-reverting, close on DCC
  * CT2 / "momentum"      : always momentum-following, close on DCC
  * JC1 / "regime_dependent": mean-reverting in regime 0 (low vol),
                              momentum in regime 1 (high vol),
                              close on DCC OR regime change

Three metrics (Appendix A):

  * profit: (1 + r).prod() - 1  — cumulative compound return
  * sharpe: r.mean() * 252 / (r.std() * sqrt(252))   (rf=0)
  * mdd   : Kadane min-subarray on simple returns
            (a proxy for max drawdown — see Eqs. 7-8 in the paper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .directional_change import split_dcc_ext

logger = logging.getLogger(__name__)


# Constants — match the paper.
TRADE_THRESHOLD = 0.5    # |TMV| trigger
INIT_CAPITAL    = 1.0    # arbitrary unit notional


# ── enriched event frame ───────────────────────────────────────────


def build_event_frame(
    prices: pd.Series,
    dc: list[tuple],
    regimes: pd.Series,
    theta: float,
) -> pd.DataFrame:
    """Augment the raw price series with the columns the strategy
    state machine needs:

      * ``price``  : the input prices
      * ``type``   : one of ``Other`` / ``DCC`` / ``EXT`` / ``EXT_DCC``
      * ``TMV``    : running TMV from the last EXT
      * ``regime`` : forward-filled regime label (-1 before the first
                     regime prediction is available, mirroring the paper)

    Returns the augmented frame, indexed positionally (0..N-1) — the
    strategy loop uses positional access. The original timestamps live
    in the ``time`` column.
    """
    df = pd.DataFrame({"price": prices.values}, index=np.arange(len(prices)))
    df["time"] = prices.index

    _, dcc_idx, _, ext_idx = split_dcc_ext(dc)
    dcc_set = set(dcc_idx)
    ext_set = set(ext_idx)

    types = np.full(len(df), "Other", dtype=object)
    for i, t in enumerate(prices.index):
        in_dcc = t in dcc_set
        in_ext = t in ext_set
        if in_dcc and in_ext:
            types[i] = "EXT_DCC"
        elif in_dcc:
            types[i] = "DCC"
        elif in_ext:
            types[i] = "EXT"
    df["type"] = types

    # Running TMV from the last EXT.
    tmv = np.zeros(len(df), dtype=float)
    trend = np.zeros(len(df), dtype=int)
    last_ext_price = np.nan
    for i in range(len(df)):
        if df["type"].iloc[i] in ("EXT", "EXT_DCC"):
            last_ext_price = df["price"].iloc[i]
        if np.isfinite(last_ext_price) and last_ext_price > 0:
            move = df["price"].iloc[i] - last_ext_price
            tmv[i] = move / (last_ext_price * theta)
            trend[i] = int(np.sign(move))
    df["TMV"]   = tmv
    df["trend"] = trend

    # Forward-fill regime; -1 before the first prediction is available.
    regime_col = np.full(len(df), -1, dtype=int)
    if not regimes.empty:
        reg_lookup = dict(zip(regimes.index, regimes.values.astype(int)))
        last_regime = -1
        first_regime_t = regimes.index[0]
        for i, t in enumerate(prices.index):
            if t in reg_lookup:
                last_regime = reg_lookup[t]
                regime_col[i] = last_regime
            elif t >= first_regime_t:
                regime_col[i] = last_regime
    df["regime"] = regime_col

    return df


# ── strategy state machines ────────────────────────────────────────


def _run_strategy(
    df: pd.DataFrame,
    *,
    mode: str,                       # "mean_reverting" | "momentum" | "regime_dependent"
    init_cap: float = INIT_CAPITAL,
    threshold: float = TRADE_THRESHOLD,
) -> pd.DataFrame:
    """Run the strategy state machine and return the augmented frame.

    Columns added: position, asset_cap, bank_cap, total_cap, daily_ret.
    Identical bookkeeping for all three modes; only the entry-direction
    rule and close-condition differ.

    Position sizing: ALL CAPITAL into the position at entry. This
    matches the paper — we're benchmarking strategy *shape*, not
    optimal sizing.
    """
    n = len(df)
    price  = df["price"].to_numpy(dtype=float)
    tmv    = df["TMV"].to_numpy(dtype=float)
    typ    = df["type"].to_numpy(dtype=object)
    regime = df["regime"].to_numpy(dtype=int)

    position  = np.zeros(n, dtype=float)
    asset_cap = np.zeros(n, dtype=float)
    bank_cap  = np.zeros(n, dtype=float)
    total_cap = np.zeros(n, dtype=float)
    daily_ret = np.zeros(n, dtype=float)
    bank_cap[0] = init_cap
    total_cap[0] = init_cap

    is_dcc_event = lambda t: t in ("DCC", "EXT_DCC")

    for i in range(1, n):
        if regime[i] == -1:
            # No regime info yet — sit on cash, mirroring the paper.
            position[i]  = position[i - 1]
            asset_cap[i] = asset_cap[i - 1]
            bank_cap[i]  = bank_cap[i - 1]
        else:
            if position[i - 1] == 0 and abs(tmv[i]) >= threshold:
                # Open a new position.
                if mode == "mean_reverting":
                    direction = -np.sign(tmv[i])
                elif mode == "momentum":
                    direction = +np.sign(tmv[i])
                elif mode == "regime_dependent":
                    # Normal regime (0) → mean-revert; abnormal (1) → momentum.
                    direction = -np.sign(tmv[i]) if regime[i] == 0 else +np.sign(tmv[i])
                else:
                    raise ValueError(f"unknown mode {mode!r}")
                position[i]  = direction * (total_cap[i - 1] / price[i])
                asset_cap[i] = position[i] * price[i]
                bank_cap[i]  = total_cap[i - 1] - asset_cap[i]
            elif position[i - 1] == 0:
                # No position, threshold not breached.
                position[i]  = position[i - 1]
                asset_cap[i] = position[i] * price[i]
                bank_cap[i]  = bank_cap[i - 1]
            else:
                # Already in a position — check close conditions.
                regime_flip = (mode == "regime_dependent"
                               and regime[i - 1] != regime[i])
                if not regime_flip and not is_dcc_event(typ[i]):
                    # Hold.
                    position[i]  = position[i - 1]
                    asset_cap[i] = position[i] * price[i]
                    bank_cap[i]  = bank_cap[i - 1]
                else:
                    # Close → all cash.
                    position[i]  = 0.0
                    asset_cap[i] = 0.0
                    bank_cap[i]  = abs(position[i - 1]) * price[i]

        total_cap[i] = bank_cap[i] + asset_cap[i]
        prev_total = total_cap[i - 1]
        daily_ret[i] = (total_cap[i] - prev_total) / prev_total if prev_total != 0 else 0.0

    out = df.copy()
    out["position"]  = position
    out["asset_cap"] = asset_cap
    out["bank_cap"]  = bank_cap
    out["total_cap"] = total_cap
    out["daily_ret"] = daily_ret
    return out


def regime_dependent(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Mean-revert during regime 0, momentum during regime 1.
    Close on DCC or regime change."""
    return _run_strategy(df, mode="regime_dependent", **kwargs)


def mean_reverting_control(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """CT1 — always mean-reverting. Close on DCC."""
    return _run_strategy(df, mode="mean_reverting", **kwargs)


def momentum_control(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """CT2 — always momentum-following. Close on DCC."""
    return _run_strategy(df, mode="momentum", **kwargs)


# ── metrics (Appendix A) ───────────────────────────────────────────


def profit(total_cap: pd.Series) -> float:
    """(end_capital - start_capital) / start_capital. The paper
    quotes this as the headline "profit" in Table 4."""
    start = float(total_cap.iloc[0])
    end   = float(total_cap.iloc[-1])
    return (end - start) / start if start != 0 else 0.0


def sharpe(daily_ret: pd.Series, *, regime_mask: pd.Series | None = None) -> float:
    """Annualised Sharpe with rf=0 on the strategy's per-observation
    returns.

    The paper's price series is semi-daily (open at +6.5h, close at
    +0h ⇒ ~504 obs/year), so the √-rule annualisation factor is
    ``√(2·252)``, NOT the textbook ``√252``. The paper's
    ``trading_strategy.get_sharpe`` is the one wired into its metrics
    pipeline and uses ``sqrt(2·252)``; reproducing that exactly is
    what brings us in line with Table 4. The earlier ``sharpe`` helper
    in the same file (with the plain 252 factor) is unused by the
    pipeline.

    We also exclude pre-regime warmup rows (``regime == -1``) when
    a mask is supplied — matching the paper's
    ``df = df[df['regime'] >= 0]`` filter. Without the filter the
    long stretch of zero returns before the first regime prediction
    deflates the mean and pulls Sharpe artificially low.
    """
    r = daily_ret.dropna()
    if regime_mask is not None:
        # Align by index, drop any row where regime is missing or -1.
        m = regime_mask.reindex(r.index).fillna(-1).astype(int) >= 0
        r = r[m.values]
    if len(r) == 0 or r.std(ddof=1) == 0:
        return 0.0
    # Use ddof=1 (sample std) — the paper passes ddof=1 explicitly.
    return float(np.sqrt(2.0 * 252.0) * r.mean() / r.std(ddof=1))


def max_drawdown(daily_ret: pd.Series) -> float:
    """Paper's proxy MDD = -min(cumsum(r) - cummax(cumsum(r))), i.e.
    the deepest dip of the simple-return cumulative below its running
    max. Returned as a positive number for table display.

    This isn't classical price-drawdown — it's a returns-based proxy
    the paper uses for cross-validation cheapness. We honour the
    convention so our numbers compare apples-to-apples."""
    r = daily_ret.dropna()
    if r.empty:
        return 0.0
    cum = r.cumsum()
    dd = (cum - cum.cummax()).min()
    return float(-dd)


def metrics_summary(out: pd.DataFrame, *, name: str = "strategy") -> dict:
    """One-shot metrics dict for a strategy output frame.

    Keys mirror Table 4: ``profit``, ``sharpe``, ``mdd``. The extra
    ``n_trades`` count is for instrumentation only — the paper doesn't
    report it but it's useful when debugging a strategy that "made
    nothing" (often: zero trades fired)."""
    daily = out["daily_ret"]
    total = out["total_cap"]
    # Mask off the pre-regime warmup (regime == -1) when computing
    # Sharpe so we match the paper's ``df[df['regime'] >= 0]`` filter.
    regime_mask = out["regime"] if "regime" in out.columns else None
    # Trade = transition into a nonzero position from zero.
    pos = out["position"].to_numpy()
    entries = int(((np.abs(pos) > 0) & (np.roll(np.abs(pos), 1) == 0)).sum())
    return {
        "name":    name,
        "profit":  round(profit(total), 4),
        "sharpe":  round(sharpe(daily, regime_mask=regime_mask), 2),
        "mdd":     round(max_drawdown(daily), 4),
        "n_trades": entries,
    }
