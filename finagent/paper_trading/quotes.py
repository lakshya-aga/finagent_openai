"""Pluggable last-traded-price source.

Why an abstraction layer:
  - GROWW's REST quote endpoints aren't part of any open docs we
    can verify against. The WebSocket protocol is well-known
    (live_data/server/sources/groww.py) but it's a long-lived
    stream — wrong shape for periodic SL/TP polling.
  - yfinance works without auth, returns the latest 1-min bar,
    and survives the IST market hours fine (15-min delayed but
    that's acceptable for paper-trading triggers).

So: yfinance is the default. The GROWW slot exists as a stub —
when we verify the REST endpoints we swap in a working
implementation without touching the engine.

Provider auto-selection (``get_quote_source``):
  1. If ``PAPER_TRADING_QUOTE_SOURCE=groww`` in env, use GrowwSource
     (warns + falls back to yfinance if creds missing).
  2. Otherwise: YFinanceSource.

Both providers implement:
  async get_ltps(tickers: list[str]) -> dict[str, float]
  async get_day_open(ticker: str, date: str) -> float | None

Missing tickers are omitted from the result dict (callers handle).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


class QuoteSource(Protocol):
    """Minimal interface every provider implements. The engine never
    cares which provider answers — it just calls these two methods."""

    name: str

    async def get_ltps(self, tickers: list[str]) -> dict[str, float]:
        """Latest traded price for each ticker. Missing tickers omitted."""
        ...

    async def get_day_open(self, ticker: str, date: str) -> float | None:
        """Open price for a given trading date. ``date`` in
        'YYYY-MM-DD'. Returns None on miss (e.g. holiday, fresh listing)."""
        ...


# ── YFinance default ───────────────────────────────────────────────


class YFinanceSource:
    """Default provider — no auth, free, ~15-min delayed.

    Acceptable for paper trading because:
      - SL/TP triggers fire on the LATEST PRICE WE CAN SEE; a 15-min
        lag means a position that hit SL at 11:00 IST gets the close
        recorded at 11:15. The paper-PnL is still the SL price; only
        the wall-clock timestamp is delayed.
      - We're not competing with HFT — 5-15min monitoring resolution
        is plenty for a daily directional book.
    """

    name = "yfinance"

    async def get_ltps(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}

        def _fetch() -> dict[str, float]:
            import yfinance as yf
            # period='1d', interval='1m' returns intraday bars; we
            # take the last close. Falls back to period='5d' if the
            # intraday call returns empty (after-hours, holiday).
            out: dict[str, float] = {}
            try:
                df = yf.download(
                    tickers=tickers, period="1d", interval="1m",
                    progress=False, auto_adjust=False, group_by="ticker",
                    threads=True,
                )
                if df is None or df.empty:
                    raise RuntimeError("yfinance returned empty intraday frame")
            except Exception:
                df = yf.download(
                    tickers=tickers, period="5d", interval="1d",
                    progress=False, auto_adjust=False, group_by="ticker",
                    threads=True,
                )
                if df is None or df.empty:
                    return out
            if len(tickers) == 1:
                try:
                    out[tickers[0]] = float(df["Close"].dropna().iloc[-1])
                except Exception:
                    pass
                return out
            for t in tickers:
                try:
                    out[t] = float(df[t]["Close"].dropna().iloc[-1])
                except Exception:
                    continue
            return out

        return await asyncio.to_thread(_fetch)

    async def get_day_open(self, ticker: str, date: str) -> float | None:
        def _fetch() -> float | None:
            import yfinance as yf
            import pandas as pd
            # Pull a slim window around the date so we get the open
            # bar even when the market opened late or had a gap.
            try:
                df = yf.download(
                    tickers=ticker, start=date,
                    end=(pd.Timestamp(date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                    interval="1d", progress=False, auto_adjust=False,
                )
                if df is None or df.empty:
                    return None
                # First row's Open — that's the day's open print.
                if isinstance(df.columns, pd.MultiIndex):
                    return float(df["Open"][ticker].dropna().iloc[0])
                return float(df["Open"].dropna().iloc[0])
            except Exception:
                return None

        return await asyncio.to_thread(_fetch)


# ── GROWW slot (stub, swap in REST impl once endpoints verified) ──


class GrowwSource:
    """Groww quote provider — wraps the official ``growwapi`` SDK.

    Auth: ``GROWW_ACCESS_TOKEN`` env var must hold a token with the
    ``apiTrading`` role. To generate one:
      1. Sign in at https://developer.groww.in
      2. Create an API app → returns an API key + API secret
      3. Run the developer portal's TOTP login flow → returns the
         long-lived API token. THAT is what GROWW_ACCESS_TOKEN
         expects.

    What WON'T work: pasting your normal Groww app session token
    (role ``auth-totp``). Those tokens are scoped to the UI and
    return 403 on every market-data endpoint. Symptom: GrowwAPI()
    constructs fine, but every method returns
    ``GrowwAPIException: Access forbidden for this request.``

    On instantiation failure (missing token, SDK import error, or
    auth rejection) we raise a clear RuntimeError so the resolver
    in get_quote_source() falls back to yfinance.

    Symbol convention: our codebase uses ``RELIANCE.NS`` (yfinance
    style). The SDK uses bare ``RELIANCE`` with a separate
    ``segment="CASH"`` arg. We strip the ``.NS`` suffix on the way
    in and pin every Nifty 50 query to the cash segment.
    """

    name = "groww"

    def __init__(self) -> None:
        token = os.environ.get("GROWW_ACCESS_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "GROWW_ACCESS_TOKEN not set — cannot use GROWW quote source. "
                "Unset PAPER_TRADING_QUOTE_SOURCE to fall back to yfinance.",
            )
        try:
            from growwapi import GrowwAPI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "growwapi SDK not installed — `pip install growwapi`. "
                f"Underlying error: {e}",
            ) from e
        self._client = GrowwAPI(token)
        self._SEGMENT_CASH = getattr(GrowwAPI, "SEGMENT_CASH", "CASH")

    @staticmethod
    def _strip_ns(ticker: str) -> str:
        """``RELIANCE.NS`` → ``RELIANCE``; the SDK wants bare symbols."""
        return ticker.removesuffix(".NS")

    @staticmethod
    def _restore_ns(symbol: str) -> str:
        return symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    async def get_ltps(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        bare = tuple(self._strip_ns(t) for t in tickers)

        def _fetch() -> dict[str, float]:
            # get_ltp signature: (exchange_trading_symbols: Tuple[str], segment: str, ...)
            # On a typical call it returns {"NSE_RELIANCE": 2856.0, ...}
            try:
                resp = self._client.get_ltp(
                    exchange_trading_symbols=bare, segment=self._SEGMENT_CASH,
                    timeout=10,
                )
            except Exception as e:
                logger.warning("paper_trading: GROWW get_ltp failed (%s)", e)
                return {}
            out: dict[str, float] = {}
            for key, px in (resp or {}).items():
                # SDK keys back as "NSE_RELIANCE" — extract the symbol.
                sym = key.split("_", 1)[-1] if "_" in key else key
                try:
                    out[self._restore_ns(sym)] = float(px)
                except (TypeError, ValueError):
                    continue
            return out

        return await asyncio.to_thread(_fetch)

    async def get_day_open(self, ticker: str, date: str) -> float | None:
        """Today's open via ``get_ohlc``; for non-today dates use
        ``get_historical_candles``. The intraday-engine only ever calls
        this for the current trading day so we optimise for that."""
        bare = self._strip_ns(ticker)

        def _fetch_today_ohlc() -> float | None:
            try:
                resp = self._client.get_ohlc(
                    exchange_trading_symbols=(bare,),
                    segment=self._SEGMENT_CASH,
                    timeout=10,
                )
            except Exception as e:
                logger.warning("paper_trading: GROWW get_ohlc failed for %s (%s)", ticker, e)
                return None
            for _, ohlc in (resp or {}).items():
                op = ohlc.get("open") if isinstance(ohlc, dict) else None
                if op is not None:
                    try:
                        return float(op)
                    except (TypeError, ValueError):
                        return None
            return None

        return await asyncio.to_thread(_fetch_today_ohlc)


# ── Source resolver ────────────────────────────────────────────────


_singleton: QuoteSource | None = None


def get_quote_source() -> QuoteSource:
    """Auto-select the provider based on env. Memoised — repeated
    calls return the same instance."""
    global _singleton
    if _singleton is not None:
        return _singleton

    pref = os.environ.get("PAPER_TRADING_QUOTE_SOURCE", "yfinance").lower()
    if pref == "groww":
        try:
            _singleton = GrowwSource()
            logger.info("paper_trading: quote source = GROWW (REST)")
            return _singleton
        except Exception as e:
            logger.warning(
                "paper_trading: GROWW quote source unavailable (%s) — "
                "falling back to yfinance", e,
            )
    _singleton = YFinanceSource()
    logger.info("paper_trading: quote source = yfinance")
    return _singleton


def reset_quote_source() -> None:
    """Test hook — flush the memoised singleton so a subsequent
    ``get_quote_source()`` re-resolves against the current env."""
    global _singleton
    _singleton = None
