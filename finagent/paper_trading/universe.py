"""Nifty 50 universe + sector classification + market-cap refresh.

The ticker list lives here (it used to be in finagent.scheduler.NIFTY_50
when the now-removed daily-5-debate cron also consumed it; the
stock_analyst panel run is now the only daily writer over the
universe so the canonical location moved to paper_trading where it
gets used).

Sectors are hardcoded here because yfinance's sector field is
unreliable for Indian tickers (it sometimes returns "Industrials"
for a pure-bank like SBIN). Mapping is NSE's standard sector
classification as of 2026-05; refresh when the index reconstitutes.

Market caps are fetched lazily via yfinance (one request per
ticker) and cached in the `nifty50_universe` table by
``refresh_market_caps()``. The MCW strategy reads from the cached
value; if it's stale (>7 days), the next read triggers a refresh.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Optional


# ─── Nifty 50 universe ──────────────────────────────────────────────
# Source: NSE indices listing as of 2026-05. The Nifty 50 changes
# occasionally; refresh this list when the index reconstitutes.
# Tickers use the .NS suffix so yfinance routes them to NSE.
NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "BHARTIARTL.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",
    "KOTAKBANK.NS",
    "MARUTI.NS",
    "AXISBANK.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
    "TITAN.NS",
    "NTPC.NS",
    "BAJAJFINSV.NS",
    "ASIANPAINT.NS",
    "ONGC.NS",
    "ADANIENT.NS",
    "POWERGRID.NS",
    "WIPRO.NS",
    "JSWSTEEL.NS",
    "TATAMOTORS.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "BAJAJ-AUTO.NS",
    "NESTLEIND.NS",
    "BEL.NS",
    "TATASTEEL.NS",
    "GRASIM.NS",
    "HDFCLIFE.NS",
    "TRENT.NS",
    "SBILIFE.NS",
    "EICHERMOT.NS",
    "HINDALCO.NS",
    "TECHM.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "INDUSINDBK.NS",
    "APOLLOHOSP.NS",
    "BPCL.NS",
    "TATACONSUM.NS",
    "BRITANNIA.NS",
    "SHRIRAMFIN.NS",
    "HEROMOTOCO.NS",
]

logger = logging.getLogger(__name__)


# NSE standard sector for each Nifty 50 constituent.
# Kept independent of any vendor's classification — used for sector-
# exposure caps in the portfolio agent + the dashboard's sector chip.
SECTORS: dict[str, str] = {
    # Financials
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "INDUSINDBK.NS": "Banking",
    "BAJFINANCE.NS": "NBFC",
    "BAJAJFINSV.NS": "NBFC",
    "SHRIRAMFIN.NS": "NBFC",
    "HDFCLIFE.NS": "Insurance",
    "SBILIFE.NS": "Insurance",
    # IT
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT",
    "TECHM.NS": "IT",
    # Energy + Utilities
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "BPCL.NS": "Energy",
    "COALINDIA.NS": "Energy",
    "NTPC.NS": "Utilities",
    "POWERGRID.NS": "Utilities",
    # Consumer
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",
    "TATACONSUM.NS": "FMCG",
    "TITAN.NS": "Consumer Disc",
    "TRENT.NS": "Consumer Disc",
    "ASIANPAINT.NS": "Consumer Disc",
    # Auto
    "MARUTI.NS": "Auto",
    "M&M.NS": "Auto",
    "TATAMOTORS.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto",
    # Industrials + Materials
    "LT.NS": "Industrials",
    "BEL.NS": "Industrials",
    "ADANIPORTS.NS": "Industrials",
    "ULTRACEMCO.NS": "Materials",
    "GRASIM.NS": "Materials",
    "JSWSTEEL.NS": "Materials",
    "TATASTEEL.NS": "Materials",
    "HINDALCO.NS": "Materials",
    "ADANIENT.NS": "Conglomerate",
    # Healthcare
    "SUNPHARMA.NS": "Healthcare",
    "DRREDDY.NS": "Healthcare",
    "CIPLA.NS": "Healthcare",
    "APOLLOHOSP.NS": "Healthcare",
    # Telecom
    "BHARTIARTL.NS": "Telecom",
}

# Sanity check at import time — catches any silent drift between
# NIFTY50_TICKERS and SECTORS during ticker reconstitutions.
_missing = set(NIFTY50_TICKERS) - set(SECTORS.keys())
if _missing:
    logger.warning(
        "paper_trading.universe: %d tickers in NIFTY50_TICKERS have no "
        "SECTORS mapping — they will render as 'Unknown' on the dashboard: %s",
        len(_missing),
        sorted(_missing),
    )


def get_sector(ticker: str) -> str:
    """NSE sector for a Nifty 50 ticker; "Unknown" for misses."""
    return SECTORS.get(ticker, "Unknown")


# ── Market-cap refresh ──────────────────────────────────────────────

# Refresh stale rows (>7 days old) on read. Avoids forcing every
# dashboard load to wait on a yfinance batch.
_MARKET_CAP_TTL_SECS = 7 * 24 * 60 * 60


async def refresh_market_caps(
    tickers: Optional[Iterable[str]] = None,
    *,
    force: bool = False,
) -> dict[str, float]:
    """Refresh market caps for the given tickers (default: all of Nifty 50).

    Each ticker fires one yfinance request — slow (~0.3s per ticker
    for 50 = 15s). We run them in a thread pool so the event loop
    stays responsive. Caches into ``nifty50_universe`` so the next
    call within ``_MARKET_CAP_TTL_SECS`` (7 days) is a no-op unless
    ``force=True``.

    Returns a dict of ticker → market_cap (INR). Missing tickers are
    omitted; the caller decides how to handle them (e.g. fall back
    to equal-weight when MCW data is missing).
    """
    tickers = list(tickers or NIFTY50_TICKERS)
    out: dict[str, float] = {}

    from .store import get_market_caps, upsert_market_cap

    cached = get_market_caps(tickers)
    now = time.time()
    stale = [
        t
        for t in tickers
        if force
        or (now - (cached.get(t, {}).get("refreshed_at") or 0)) > _MARKET_CAP_TTL_SECS
    ]
    # Fresh rows: copy through.
    for t in tickers:
        if t not in stale and cached.get(t, {}).get("market_cap"):
            out[t] = cached[t]["market_cap"]

    if not stale:
        return out

    logger.info("paper_trading: refreshing %d market caps", len(stale))

    def _fetch_one(ticker: str) -> Optional[float]:
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info or {}
            mcap = info.get("marketCap")
            return float(mcap) if mcap is not None else None
        except Exception as e:
            logger.warning(
                "paper_trading: market-cap fetch failed for %s (%s)", ticker, e
            )
            return None

    # Sequential rather than parallel — yfinance is rate-limited and
    # 50 parallel requests reliably triggers 429s.
    for t in stale:
        mcap = await asyncio.to_thread(_fetch_one, t)
        if mcap is not None and mcap > 0:
            upsert_market_cap(t, mcap, get_sector(t))
            out[t] = mcap
    return out
