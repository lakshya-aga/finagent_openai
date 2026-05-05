"""Daily Nifty 50 debate scheduler.

Runs every day at 02:00 UTC (07:30 IST — 1h45m before the Indian market
opens at 09:15 IST). Picks the 5 Nifty 50 tickers least-recently
analysed and kicks a debate for each. With a 5-per-day cadence the
universe rotates every 10 days.

Single-instance APScheduler attached to the FastAPI event loop. We
specifically don't use a persistent jobstore — the cron is hardcoded
and registered on every startup, so a missed window (container restart
during the cron minute) is acceptable. The next day's run picks up
the slack via the round-robin selection rule.

Manual trigger: POST /api/debates/scheduler/run-now (admin-only, in
app.py) calls ``run_daily_nifty_debates`` directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional


# ─── Nifty 50 universe ──────────────────────────────────────────────
# Source: NSE indices listing as of 2026-05. The Nifty 50 changes
# occasionally; refresh this list when the index reconstitutes.
# Tickers use the .NS suffix so yfinance routes them to NSE.

NIFTY_50: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS",     "SBIN.NS", "LT.NS",       "HINDUNILVR.NS", "ITC.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "KOTAKBANK.NS", "MARUTI.NS", "AXISBANK.NS",
    "M&M.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TITAN.NS", "NTPC.NS",
    "BAJAJFINSV.NS", "ASIANPAINT.NS", "ONGC.NS", "ADANIENT.NS", "POWERGRID.NS",
    "WIPRO.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "BAJAJ-AUTO.NS", "NESTLEIND.NS", "BEL.NS", "TATASTEEL.NS", "GRASIM.NS",
    "HDFCLIFE.NS", "TRENT.NS", "SBILIFE.NS", "EICHERMOT.NS", "HINDALCO.NS",
    "TECHM.NS", "DRREDDY.NS", "CIPLA.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS",
    "BPCL.NS", "TATACONSUM.NS", "BRITANNIA.NS", "SHRIRAMFIN.NS", "HEROMOTOCO.NS",
]


def select_least_recent(n: int = 5) -> list[str]:
    """Return the N Nifty 50 tickers least-recently analysed by the scheduler.

    Looks at every existing 'scheduled'-source debate row in the store
    and keys by max(started_at) per ticker. Tickers with no prior
    scheduled debate float to the top (started_at defaulted to 0).
    Ties are broken by the alphabetical position in NIFTY_50 — keeps
    the rotation deterministic.
    """
    from .experiments import get_store

    store = get_store()
    last_seen: dict[str, float] = {t: 0.0 for t in NIFTY_50}
    try:
        for d in store.list_debates(limit=2000):
            if d.source != "scheduled":
                continue
            if d.ticker in last_seen and d.started_at > last_seen[d.ticker]:
                last_seen[d.ticker] = d.started_at
    except Exception:
        logging.exception("scheduler: failed to read prior debates")

    # Sort by (last_seen ascending, original index ascending) — tickers
    # never analysed (last_seen=0) come first, ordered by their listed
    # position so the very first run analyses the top-5 marquee names.
    ordered = sorted(
        enumerate(NIFTY_50),
        key=lambda pair: (last_seen[pair[1]], pair[0]),
    )
    return [t for _, t in ordered[: max(1, int(n))]]


async def run_daily_nifty_debates(n: int = 5, rounds: int = 2) -> dict:
    """Run scheduled debates on the N least-recently-analysed Nifty 50 tickers.

    Sequential rather than concurrent — runs in the background so we
    don't care about latency, and concurrency would multiply OpenAI
    rate-limit pressure during the cron window. Each debate persists
    incrementally; failures on one ticker don't stop the rest.

    Returns a summary {tickers: [...], succeeded: int, failed: int,
    debate_ids: [...]}.
    """
    from .debate import run_debate

    selected = select_least_recent(n)
    logging.info("scheduler: running daily debates on %s", selected)

    debate_ids: list[str] = []
    succeeded = 0
    failed = 0
    for ticker in selected:
        try:
            result = await run_debate(
                ticker=ticker,
                asset_class="indian_equity",
                rounds=rounds,
                emit=None,             # no SSE stream — purely server-side
                debate_id=None,
                source="scheduled",
            )
            debate_ids.append(result.get("debate_id", ""))
            succeeded += 1
        except Exception:
            logging.exception("scheduler: debate failed for %s", ticker)
            failed += 1

    summary = {
        "tickers": selected,
        "succeeded": succeeded,
        "failed": failed,
        "debate_ids": debate_ids,
    }
    logging.info("scheduler: completed daily debates: %s", summary)
    return summary


# ─── APScheduler integration ────────────────────────────────────────


_scheduler: Optional[object] = None  # AsyncIOScheduler, lazy import


def start_scheduler() -> None:
    """Start the daily Nifty 50 cron. No-op if APScheduler isn't installed
    (e.g. a slim local dev install) or if already started.

    The cron fires at 02:00 UTC (07:30 IST) — well before Indian market
    open at 09:15 IST so the verdicts are fresh when the user reads them.
    """
    global _scheduler
    if _scheduler is not None:
        return

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logging.warning(
            "scheduler: apscheduler not installed; daily Nifty debates DISABLED. "
            "Install: pip install apscheduler"
        )
        return

    sch = AsyncIOScheduler(timezone="UTC")
    sch.add_job(
        run_daily_nifty_debates,
        CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="nifty50_daily_debates",
        replace_existing=True,
        misfire_grace_time=3600,  # tolerate 1h late firing if the worker was down
    )
    sch.start()
    _scheduler = sch
    logging.info("scheduler: started; daily Nifty 50 debates at 02:00 UTC")


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)  # type: ignore[attr-defined]
        except Exception:
            logging.exception("scheduler: shutdown failed")
        _scheduler = None
