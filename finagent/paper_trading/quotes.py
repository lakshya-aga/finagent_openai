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

    Auth (three input paths, picked by ``groww_auth.get_token``):
      1. ``GROWW_ACCESS_TOKEN`` env var → use as-is (manual override)
      2. ``GROWW_API_KEY`` + ``GROWW_TOTP_SECRET`` → mint via
         ``GrowwAPI.get_access_token(api_key, totp)``. Cached in-memory
         for 20h; auto-refreshes.
      3. None of the above → raise so the resolver in
         ``get_quote_source()`` falls back to yfinance.

    The minted token has the ``apiTrading`` role (vs the ``auth-totp``
    role a UI session token has). Only ``apiTrading`` tokens succeed
    against the market-data endpoints; pasting your normal Groww app
    session token returns 403 on every call.

    Symbol convention: our codebase uses ``RELIANCE.NS`` (yfinance
    style). The SDK uses bare ``RELIANCE`` with a separate
    ``segment="CASH"`` arg. We strip the ``.NS`` suffix on the way
    in and pin every Nifty 50 query to the cash segment.

    On a 401 mid-day (Groww revoked the token early) we invalidate
    the cache + re-mint on the next call.
    """

    name = "groww"

    def __init__(self) -> None:
        try:
            from growwapi import GrowwAPI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "growwapi SDK not installed — `pip install growwapi`. "
                f"Underlying error: {e}",
            ) from e
        # GrowwAPI is also needed as the static for token minting.
        self._GrowwAPI = GrowwAPI

        from . import groww_auth
        self._auth = groww_auth
        token = groww_auth.get_token()  # may raise — caller handles
        self._client = GrowwAPI(token)
        self._SEGMENT_CASH = getattr(GrowwAPI, "SEGMENT_CASH", "CASH")

    def _rebuild_after_refresh(self) -> None:
        """Re-mint the token (the cache decides whether actual mint
        runs vs returns the existing cached token) and rebuild the
        SDK client. Called when a 401 surfaces mid-day."""
        self._auth.invalidate()
        try:
            token = self._auth.get_token(force_refresh=True)
        except Exception as e:
            logger.warning("paper_trading: groww token refresh failed (%s)", e)
            return
        self._client = self._GrowwAPI(token)

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

        def _fetch_once() -> tuple[dict[str, float], bool]:
            """Returns (ltps, hit_auth_error). On auth_error caller
            re-mints the token and retries exactly once."""
            try:
                resp = self._client.get_ltp(
                    exchange_trading_symbols=bare, segment=self._SEGMENT_CASH,
                    timeout=10,
                )
            except Exception as e:
                msg = str(e).lower()
                auth_failed = (
                    "unauthorized" in msg or "401" in msg
                    or "invalid token" in msg or "forbidden" in msg
                    or "expired" in msg
                )
                if not auth_failed:
                    logger.warning("paper_trading: GROWW get_ltp failed (%s)", e)
                return {}, auth_failed
            out: dict[str, float] = {}
            for key, px in (resp or {}).items():
                sym = key.split("_", 1)[-1] if "_" in key else key
                try:
                    out[self._restore_ns(sym)] = float(px)
                except (TypeError, ValueError):
                    continue
            return out, False

        ltps, auth_failed = await asyncio.to_thread(_fetch_once)
        if auth_failed:
            logger.info("paper_trading: GROWW token rejected — re-minting and retrying once")
            self._rebuild_after_refresh()
            ltps, _ = await asyncio.to_thread(_fetch_once)
        return ltps

    async def get_day_open(self, ticker: str, date: str) -> float | None:
        """Today's open via ``get_ohlc``; for non-today dates use
        ``get_historical_candles``. The intraday-engine only ever calls
        this for the current trading day so we optimise for that."""
        bare = self._strip_ns(ticker)

        def _fetch_once() -> tuple[float | None, bool]:
            try:
                resp = self._client.get_ohlc(
                    exchange_trading_symbols=(bare,),
                    segment=self._SEGMENT_CASH,
                    timeout=10,
                )
            except Exception as e:
                msg = str(e).lower()
                auth_failed = (
                    "unauthorized" in msg or "401" in msg
                    or "invalid token" in msg or "forbidden" in msg
                    or "expired" in msg
                )
                if not auth_failed:
                    logger.warning("paper_trading: GROWW get_ohlc failed for %s (%s)", ticker, e)
                return None, auth_failed
            for _, ohlc in (resp or {}).items():
                op = ohlc.get("open") if isinstance(ohlc, dict) else None
                if op is not None:
                    try:
                        return float(op), False
                    except (TypeError, ValueError):
                        return None, False
            return None, False

        price, auth_failed = await asyncio.to_thread(_fetch_once)
        if auth_failed:
            self._rebuild_after_refresh()
            price, _ = await asyncio.to_thread(_fetch_once)
        return price


# ── Source resolver ────────────────────────────────────────────────


_singleton: QuoteSource | None = None


def get_quote_source() -> QuoteSource:
    """Auto-select the provider. Memoised — repeated calls return
    the same instance.

    Selection rules:
      * ``PAPER_TRADING_QUOTE_SOURCE=yfinance`` → force yfinance.
      * ``PAPER_TRADING_QUOTE_SOURCE=groww`` → force GROWW. Raises on
        construction failure (no fallback) so the operator sees the
        explicit choice failing.
      * unset OR ``=auto`` → use GROWW if any Groww credential set
        (``GROWW_ACCESS_TOKEN``, OR ``GROWW_API_KEY`` + ``GROWW_TOTP_SECRET``).
        Otherwise yfinance.

    This lets a deploy "just work" the moment the GROWW secrets land
    in the VM .env, without anyone remembering to flip a separate
    PAPER_TRADING_QUOTE_SOURCE flag.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    pref = os.environ.get("PAPER_TRADING_QUOTE_SOURCE", "auto").lower()

    if pref == "yfinance":
        _singleton = YFinanceSource()
        logger.info("paper_trading: quote source = yfinance (explicit)")
        return _singleton

    if pref == "groww":
        # Explicit — no fallback. Surface the failure loud.
        _singleton = GrowwSource()
        logger.info("paper_trading: quote source = GROWW (explicit)")
        return _singleton

    # auto: prefer GROWW when creds are present.
    has_token  = bool(os.environ.get("GROWW_ACCESS_TOKEN", "").strip())
    has_totp_pair = bool(
        os.environ.get("GROWW_API_KEY", "").strip()
        and os.environ.get("GROWW_TOTP_SECRET", "").strip()
    )
    if has_token or has_totp_pair:
        try:
            _singleton = GrowwSource()
            logger.info(
                "paper_trading: quote source = GROWW (auto; mint=%s)",
                "totp" if has_totp_pair and not has_token else "override-token",
            )
            return _singleton
        except Exception as e:
            logger.warning(
                "paper_trading: GROWW quote source unavailable (%s) — "
                "falling back to yfinance", e,
            )
    _singleton = YFinanceSource()
    logger.info("paper_trading: quote source = yfinance (auto; no GROWW creds)")
    return _singleton


def reset_quote_source() -> None:
    """Test hook — flush the memoised singleton so a subsequent
    ``get_quote_source()`` re-resolves against the current env."""
    global _singleton
    _singleton = None
