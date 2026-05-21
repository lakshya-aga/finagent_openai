"""Groww access-token minting + caching.

The official ``growwapi`` SDK takes a single pre-minted ``token`` arg
(a 24h JWT with the ``apiTrading`` role). It exposes a static
``GrowwAPI.get_access_token(api_key, totp)`` helper that does the
minting in one call — we use that exclusively. No HTTP guessing.

Three input paths, picked in this order:

  1. GROWW_ACCESS_TOKEN env var
     Manual override. If set, use it as-is. Useful for testing with a
     token minted via the developer-portal UI's "Generate Access Token"
     button, or for a deploy that doesn't want to ship its TOTP seed
     around.

  2. GROWW_API_KEY + GROWW_TOTP_SECRET env vars
     The recurring path. We use pyotp.TOTP(secret).now() to generate
     the current 6-digit code, then call
     GrowwAPI.get_access_token(api_key=..., totp=...) to mint a
     fresh JWT. The token is cached in this module's memory; we
     refresh it after 20h (JWTs are 24h, 4h safety margin).

  3. None of the above set
     We raise. The quote-source resolver catches this and falls
     back to yfinance, so the rest of the system keeps working.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


# Tokens are 24h-valid (per the iat/exp claims we've seen). Refresh
# at 20h so a stale token never blocks a market-hours call.
_TOKEN_TTL_SECS = 20 * 3600


@dataclass
class _CachedToken:
    token: str
    minted_at: float

    def is_fresh(self) -> bool:
        return (time.time() - self.minted_at) < _TOKEN_TTL_SECS


_cached: Optional[_CachedToken] = None
_lock = threading.Lock()


def get_token(force_refresh: bool = False) -> str:
    """Return a valid Groww access token. Mints + caches as needed.

    Thread-safe via a module-level lock so two concurrent
    ``GrowwSource`` constructions don't double-mint (rare but possible
    on app startup with overlapping schedulers).

    Raises ``RuntimeError`` if neither override token nor
    API_KEY+TOTP_SECRET pair is available.
    """
    global _cached

    # Fast path: in-memory token still fresh.
    if not force_refresh and _cached and _cached.is_fresh():
        return _cached.token

    with _lock:
        # Re-check under the lock (another thread may have just minted).
        if not force_refresh and _cached and _cached.is_fresh():
            return _cached.token

        # Path 1: explicit override token.
        override = os.environ.get("GROWW_ACCESS_TOKEN", "").strip()
        if override:
            _cached = _CachedToken(token=override, minted_at=time.time())
            logger.info("groww_auth: using GROWW_ACCESS_TOKEN override (no mint)")
            return override

        # Path 2: API_KEY + TOTP_SECRET → mint.
        api_key = os.environ.get("GROWW_API_KEY", "").strip()
        totp_secret = os.environ.get("GROWW_TOTP_SECRET", "").strip()
        if api_key and totp_secret:
            token = _mint_via_totp(api_key, totp_secret)
            if token:
                _cached = _CachedToken(token=token, minted_at=time.time())
                return token

        # Path 3: nothing available.
        raise RuntimeError(
            "No Groww credentials available — set either GROWW_ACCESS_TOKEN, "
            "or both GROWW_API_KEY + GROWW_TOTP_SECRET."
        )


def _mint_via_totp(api_key: str, totp_secret: str) -> Optional[str]:
    """Call GrowwAPI.get_access_token(api_key, totp). Returns the JWT
    on success, None on any failure. Failures get logged at WARNING
    so the operator can see them in container logs without the daily
    cron crashing."""
    try:
        from growwapi import GrowwAPI  # type: ignore[import-not-found]
        import pyotp
    except ImportError as e:
        logger.warning("groww_auth: SDK or pyotp not installed (%s)", e)
        return None

    try:
        totp_code = pyotp.TOTP(totp_secret).now()
    except Exception as e:
        # Bad TOTP seed (wrong base32, truncated, etc.) — pyotp raises here.
        logger.warning("groww_auth: TOTP code generation failed (%s)", e)
        return None

    try:
        token = GrowwAPI.get_access_token(api_key=api_key, totp=totp_code)
    except Exception as e:
        logger.warning("groww_auth: get_access_token failed (%s)", e)
        return None

    if not token or not isinstance(token, str):
        logger.warning("groww_auth: get_access_token returned %r", token)
        return None

    logger.info("groww_auth: minted fresh access token (%d chars)", len(token))
    return token


def invalidate() -> None:
    """Drop the cached token — next ``get_token()`` re-mints. Called
    by GrowwSource when a 401 indicates the token went stale early
    (Groww can revoke tokens before the 24h TTL on policy violations
    or admin action)."""
    global _cached
    with _lock:
        _cached = None
        logger.info("groww_auth: cached token invalidated")
