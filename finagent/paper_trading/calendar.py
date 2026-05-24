"""NSE trading-calendar helpers.

The paper-trading crons (``run_daily_stock_analyses`` at 09:30 UTC and
``run_paper_trading_rebalance`` at 10:00 UTC) fire every calendar day.
Without a market-holiday filter they happily run on Saturdays, Sundays,
Republic Day, etc. — producing "predictions" with no live market to
trade against, then closing positions at the previous trading day's
close price and booking artificial ``direction_flip`` rows with ₹0 PnL.

This module provides the gate. Each cron entry-point calls
``is_nse_trading_day(today)`` and no-ops cleanly when the answer is
False. ``/api/health`` also surfaces today's status so an operator can
quickly answer "did the cron skip today because it's a holiday?"
without ssh'ing the VM.

Preferred backend: ``pandas_market_calendars`` (NSE calendar built-in,
updated upstream when NSE publishes a new holiday list). When that
library isn't installed we fall back to a weekend-only check — which
is right for ~50 of the ~52 weeks/year and wrong only on the ~12 NSE
holidays per year. Installing the lib should be considered required
for production; the fallback is a soft-degrade so a dev environment
without the dep doesn't break.
"""

from __future__ import annotations

import logging
from datetime import date as _date_t
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Backend detection ───────────────────────────────────────────────


def _get_nse_calendar():
    """Lazy import + memoised handle on the NSE calendar.

    Returns the pandas_market_calendars NSE calendar object on success,
    or ``None`` if the library isn't installed (caller falls back to
    the weekend-only check).
    """
    try:
        import pandas_market_calendars as mcal  # type: ignore[import-not-found]
    except ImportError:
        return None
    return mcal.get_calendar("NSE")


# Cache the (year, month) → set-of-trading-day-ISO results so the
# Saturday cron pays the schedule() cost once per month, not once per
# call. lru_cache size 24 = two years of months.
@lru_cache(maxsize=24)
def _trading_days_for_month(year: int, month: int) -> Optional[frozenset[str]]:
    cal = _get_nse_calendar()
    if cal is None:
        return None
    # Slightly extend the window so a query on the 1st of the month
    # still sees the prior month's tail in case caller does next_open
    # math across the boundary.
    start = pd.Timestamp(year=year, month=month, day=1)
    # First day of the month AFTER `month` — pandas handles year roll.
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)
    try:
        sched = cal.schedule(start_date=start, end_date=end)
    except Exception as e:
        logger.warning("calendar: schedule() raised for %d-%02d (%s)", year, month, e)
        return None
    return frozenset(sched.index.strftime("%Y-%m-%d"))


# ── Public API ──────────────────────────────────────────────────────


def _coerce_date(when) -> _date_t:
    if isinstance(when, str):
        return pd.Timestamp(when).date()
    if isinstance(when, datetime):
        return when.date()
    if isinstance(when, _date_t):
        return when
    raise TypeError(f"calendar: unsupported date type {type(when).__name__}")


def is_nse_trading_day(when=None) -> bool:
    """Is ``when`` a date the NSE was/is open for trading?

    ``when`` accepts a ``str`` (``YYYY-MM-DD``), ``datetime``, or
    ``date``. Defaults to today UTC.

    Algorithm:
      1. If ``pandas_market_calendars`` is installed, consult the NSE
         calendar (handles weekends + every published holiday).
      2. Otherwise fall back to a weekend-only check. Correct on ~95%
         of days; only wrong on the ~12 NSE holidays per year.
    """
    d = _coerce_date(when) if when is not None else datetime.now(timezone.utc).date()
    days = _trading_days_for_month(d.year, d.month)
    if days is not None:
        return d.strftime("%Y-%m-%d") in days
    # Fallback — Monday=0 ... Sunday=6
    return d.weekday() < 5


def next_nse_trading_day(when=None) -> _date_t:
    """Strictly the NEXT trading day after ``when`` (today UTC by
    default). If ``when`` is a Friday the answer is the following
    Monday (or later if Monday is a holiday).
    """
    start = _coerce_date(when) if when is not None else datetime.now(timezone.utc).date()
    # Search up to 14 days forward — covers any conceivable holiday
    # cluster (e.g. Diwali + weekend + bank holiday).
    for delta in range(1, 15):
        candidate = start + pd.Timedelta(days=delta)
        d = candidate.date() if hasattr(candidate, "date") else candidate
        if is_nse_trading_day(d):
            return d
    raise RuntimeError(
        f"calendar: no NSE trading day found in 14 days after {start.isoformat()}",
    )


def calendar_backend() -> str:
    """Diagnostic helper for ``/api/health`` — which backend is in use?

    Returns ``"pandas_market_calendars"`` when the upstream lib is
    available (preferred), ``"weekend_only"`` when we're using the
    fallback (production should never be on this — install
    ``pandas_market_calendars``).
    """
    return "pandas_market_calendars" if _get_nse_calendar() is not None else "weekend_only"
