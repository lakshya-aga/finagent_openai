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
    _record_fire("nifty50_daily_debates", "ok", summary)
    return summary


# ─── APScheduler integration ────────────────────────────────────────


_scheduler: Optional[object] = None  # AsyncIOScheduler, lazy import


# Per-job execution telemetry — populated by the cron handler
# wrappers below and read by /api/health. Kept module-level (rather
# than persisted) on purpose: this is "did the cron fire since the
# process came up", not historical state. If the process restarts
# the dict resets, which is exactly what you want for diagnosing
# "did the new container actually run the cron yet".
_LAST_FIRE: dict[str, dict] = {}


def _record_fire(job_id: str, status: str, result: dict | None = None) -> None:
    """Stamp ``_LAST_FIRE[job_id]`` with the most recent invocation.

    ``status`` is one of ``ok`` / ``error`` / ``starting``.
    ``result`` is the (possibly truncated) return payload — handy
    for the health endpoint to surface n_buy / n_sell etc. without
    re-running the job.
    """
    import time as _time
    entry = {"status": status, "ts": _time.time()}
    if result is not None:
        # Trim noisy fields so the health response stays under a few KB.
        trimmed = {k: v for k, v in result.items() if k not in ("sample_per_ticker", "traceback")}
        if "reports" in trimmed and isinstance(trimmed["reports"], list):
            trimmed["reports"] = [
                {k: v for k, v in r.items() if k not in ("trades", "positions")}
                for r in trimmed["reports"]
            ]
        entry["summary"] = trimmed
    _LAST_FIRE[job_id] = entry


def get_last_fire_table() -> dict[str, dict]:
    """Read-only snapshot of ``_LAST_FIRE`` for the health endpoint."""
    return dict(_LAST_FIRE)


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

    # Paper-trading daily lifecycle. TWO cron jobs replace the v2
    # four-phase intraday pipeline. Positions now open AT TODAY'S
    # CLOSE and carry overnight; the daily rebalance is the single
    # write surface. UTC times below; IST is UTC+5:30.
    #
    #   09:30 UTC = 15:00 IST   Phase A: stock-analyst run for all 50
    #                            Nifty tickers. 30 min before market
    #                            close so the analyst sees a near-full
    #                            session of price action and the
    #                            recommendations feed straight into
    #                            the close-rebalance below. Runs FULLY
    #                            in parallel (concurrency=50) — ~$0.05
    #                            of LLM, finishes in 20–60s.
    #   10:00 UTC = 15:30 IST   Phase B: close-rebalance. Replays the
    #                            day's 1m tape to catch SL/TP fires,
    #                            closes any positions whose new
    #                            direction disagrees with today's
    #                            analyst output (or that hit the
    #                            max_hold_days time barrier), then
    #                            OPENS NEW positions AT TODAY'S CLOSE
    #                            and snapshots equity/exposure.
    #                            Positions carry overnight until the
    #                            next day's rebalance touches them.
    #
    # No more capture_opens / monitor / finalize / portfolio_agent —
    # the single rebalance routine subsumes all four. The intraday
    # replay still catches every SL/TP that fired during the day so
    # we lose no triple-barrier correctness.
    sch.add_job(
        run_daily_stock_analyses,
        CronTrigger(hour=9, minute=30, timezone="UTC"),
        id="paper_trading_daily_analyses",
        replace_existing=True, misfire_grace_time=1800,
    )
    sch.add_job(
        run_paper_trading_rebalance,
        CronTrigger(hour=10, minute=0, timezone="UTC"),
        id="paper_trading_rebalance",
        replace_existing=True, misfire_grace_time=3600,
    )

    sch.start()
    _scheduler = sch
    logging.info(
        "scheduler: started — Nifty debates 02:00 UTC + paper trading "
        "(analyst 09:30 UTC / 15:00 IST, close-rebalance 10:00 UTC / 15:30 IST)",
    )


async def run_daily_stock_analyses() -> dict:
    """Phase 0 cron target — fire the per-ticker stock_analyst for
    every Nifty 50 ticker, writing direction + target + stop_loss +
    max_hold_days into the predictions table. Runs ~$0.05 of LLM
    calls in ~90 seconds (5-way parallelism on gpt-4o-mini).

    On a top-level failure (import error, OpenAI auth, etc.) returns
    the exception text + a short traceback so the admin endpoint can
    show the operator exactly what went wrong without ssh'ing the VM.
    """
    import traceback
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    logging.info("scheduler: stock_analyst daily run for %s", today)
    _record_fire("paper_trading_daily_analyses", "starting")
    try:
        from .agents.stock_analyst import run_daily_all_50
    except Exception as e:
        logging.exception("scheduler: stock_analyst import failed")
        payload = {
            "date": today, "status": "import_failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
        }
        _record_fire("paper_trading_daily_analyses", "error", payload)
        return payload
    try:
        result = await run_daily_all_50(today)
        result.setdefault("status", "ok")
        _record_fire("paper_trading_daily_analyses", "ok", result)
        return result
    except Exception as e:
        logging.exception("scheduler: stock_analyst daily run failed")
        payload = {
            "date": today, "status": "run_failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
        }
        _record_fire("paper_trading_daily_analyses", "error", payload)
        return payload


async def run_paper_trading_rebalance() -> dict:
    """Single-shot daily lifecycle cron target.

    Runs at 10:00 UTC (15:30 IST = NSE close). Wraps the
    ``intraday.rebalance_at_close_all`` routine which:

      1. Replays today's 1m intraday tape to catch any SL/TP fires
         that happened during the session.
      2. Closes positions whose direction disagrees with today's
         stock_analyst output, or that hit the max_hold_days time
         barrier.
      3. OPENS NEW positions at today's close price (positions then
         carry overnight until the next rebalance touches them).
      4. MTMs survivors at today's close + writes
         portfolio_snapshots + position_snapshots.

    Returns a per-strategy report list. Idempotent on (date, strategy)
    so re-running the day overwrites the snapshot rather than
    duplicating; safe for the admin "run now" endpoint to call mid-day.

    On failure returns a structured error payload (rather than raising)
    so the admin endpoint can show the operator exactly what went wrong
    without ssh'ing the VM.
    """
    import traceback
    from datetime import datetime, timezone
    from .paper_trading import intraday

    today = datetime.now(timezone.utc).date().isoformat()
    logging.info("scheduler: paper-trading close-rebalance for %s", today)
    _record_fire("paper_trading_rebalance", "starting")
    try:
        reports = await intraday.rebalance_at_close_all(today)
        logging.info(
            "scheduler: rebalance %s — %s",
            today,
            ", ".join(
                f"{r.strategy} equity=₹{r.equity_value:.2f} pnl=₹{r.daily_pnl:+.2f} "
                f"open={r.n_open_positions} opened={r.opened_at_close} "
                f"closed_dir={r.closed_for_direction_change} "
                f"triggers=(tp:{r.triggered_target} sl:{r.triggered_stop_loss})"
                for r in reports
            ),
        )
        payload = {
            "date": today, "status": "ok",
            "reports": [r.__dict__ for r in reports],
        }
        _record_fire("paper_trading_rebalance", "ok", payload)
        return payload
    except Exception as e:
        logging.exception("scheduler: paper-trading rebalance failed for %s", today)
        payload = {
            "date": today, "status": "rebalance_failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
        }
        _record_fire("paper_trading_rebalance", "error", payload)
        return payload


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)  # type: ignore[attr-defined]
        except Exception:
            logging.exception("scheduler: shutdown failed")
        _scheduler = None


def get_registered_jobs() -> list[dict]:
    """Read-only snapshot of the registered APScheduler jobs for the
    health endpoint. Returns ``[]`` when the scheduler hasn't started
    (lets the caller distinguish "no scheduler" from "no jobs")."""
    if _scheduler is None:
        return []
    out: list[dict] = []
    try:
        for job in _scheduler.get_jobs():  # type: ignore[attr-defined]
            next_run = getattr(job, "next_run_time", None)
            out.append({
                "id":            getattr(job, "id", "?"),
                "next_run_time": next_run.isoformat() if next_run else None,
                "trigger":       str(getattr(job, "trigger", "?")),
            })
    except Exception as e:
        logging.warning("scheduler.get_registered_jobs failed: %s", e)
    return out
