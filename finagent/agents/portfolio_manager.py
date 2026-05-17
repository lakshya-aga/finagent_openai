"""Portfolio Manager agent — translates upstream debate verdicts into
the day's directional book for the paper-trading dashboard.

This agent is one layer ABOVE the per-ticker analyst. It does not
read price data or news — that work was done by the analyst whose
verdict it consumes. Its only job is portfolio construction:
filtering low-confidence calls, balancing long/short exposure,
maintaining continuity across days, and finally COMMITTING a
direction set via the single write tool.

Six tools, tight surface:

  Read:
    get_recommendations(date)
        → upstream analyst verdicts for `date`
    get_universe(date)
        → 50 Nifty tickers + sector + market cap
    get_recent_directions(ticker, lookback_days)
        → what this agent committed previously (continuity)
    get_portfolio_performance(strategy, lookback_days)
        → its own track record
    get_sector_for_ticker(ticker)
        → cheap NSE-classification lookup (no LLM)

  Write:
    commit_predictions(directions)
        → ONE batch call per day; atomic; validates ticker ∈ universe

Run via ``run_portfolio_manager(date)`` — wraps an Agents-SDK Agent
with the six function tools above. Idempotent: re-running for the
same date overwrites the directions via UNIQUE(date, ticker) on
the predictions table.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ── Tool surface ───────────────────────────────────────────────────


# Each tool is wrapped at runtime via @function_tool so the Agents
# SDK can introspect signatures. We define them here as plain
# functions so they remain unit-testable without the SDK.


def _impl_get_recommendations(date: str) -> str:
    """List the upstream debate verdicts that landed on ``date``.

    Returns a JSON string the agent reads via tool output.
    """
    from finagent.experiments import get_store

    store = get_store()
    # Pull recent debates and filter by finished-date.
    out = []
    for d in store.list_debates(limit=200):
        if not d.finished_at:
            continue
        finished_dt = datetime.fromtimestamp(d.finished_at, tz=timezone.utc).date().isoformat()
        if finished_dt != date:
            continue
        verdict = d.verdict() or {}
        out.append({
            "ticker": d.ticker,
            "action": verdict.get("action"),
            "confidence": verdict.get("confidence"),
            "target_price": verdict.get("target_price"),
            "stoploss": verdict.get("stoploss"),
            "rationale": (verdict.get("rationale") or "")[:300],
            "debate_id": d.id,
        })
    return json.dumps({"date": date, "n": len(out), "recommendations": out}, default=str)


def _impl_get_universe(date: str) -> str:
    """Return the 50 Nifty tickers with sector + last-known market cap.
    ``date`` is accepted for symmetry but the universe is static day-to-day
    (mcap refreshed via the 7-day TTL in paper_trading.universe)."""
    from finagent.paper_trading import universe, store

    mcaps = store.get_market_caps()
    out = []
    for ticker in universe.NIFTY50_TICKERS:
        info = mcaps.get(ticker, {})
        out.append({
            "ticker": ticker,
            "sector": universe.get_sector(ticker),
            "market_cap": info.get("market_cap"),
            "market_cap_refreshed_at": info.get("refreshed_at"),
        })
    return json.dumps({"date": date, "n": len(out), "universe": out}, default=str)


def _impl_get_recent_directions(ticker: Optional[str] = None, lookback_days: int = 30) -> str:
    """Past predictions committed by the portfolio manager.

    When ``ticker`` is set, returns that ticker's history; otherwise
    returns the last ``lookback_days`` of all directions across the
    universe (aggregated by date for compactness)."""
    from datetime import timedelta
    from finagent.paper_trading import store

    today = datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=int(lookback_days))).isoformat()
    if ticker:
        rows = [
            p for p in store.list_predictions(ticker=ticker)
            if p["date"] >= cutoff
        ]
        rows.sort(key=lambda r: r["date"], reverse=True)
        return json.dumps({
            "ticker": ticker,
            "lookback_days": lookback_days,
            "n": len(rows),
            "history": rows,
        }, default=str)

    # Aggregate by date — keeps the tool output compact for the LLM
    # (50 tickers × 30 days = 1500 rows would blow the context).
    by_date: dict[str, dict] = {}
    for p in store.list_predictions():
        if p["date"] < cutoff:
            continue
        d = by_date.setdefault(p["date"], {"date": p["date"], "n_long": 0, "n_short": 0, "n_flat": 0})
        if p["direction"] > 0: d["n_long"] += 1
        elif p["direction"] < 0: d["n_short"] += 1
        else: d["n_flat"] += 1
    days = sorted(by_date.values(), key=lambda r: r["date"], reverse=True)
    return json.dumps({"lookback_days": lookback_days, "by_date": days}, default=str)


def _impl_get_portfolio_performance(strategy: str = "equal_weight", lookback_days: int = 30) -> str:
    """Own track record — the agent reads this to see whether its
    recent calls have been paying off (or not)."""
    from datetime import timedelta
    from finagent.paper_trading import store

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=int(lookback_days))).isoformat()
    snaps = store.list_snapshots(strategy, start=start)
    if not snaps:
        return json.dumps({"strategy": strategy, "n": 0, "message": "no snapshots in window"})

    rets = [s["daily_return_pct"] for s in snaps if s["daily_return_pct"] is not None]
    cum_return = 1.0
    for r in rets:
        cum_return *= 1.0 + r
    import statistics
    mean = statistics.fmean(rets) if rets else 0.0
    std = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe_annualised = (mean / std) * (252 ** 0.5) if std > 0 else None
    return json.dumps({
        "strategy": strategy,
        "lookback_days": lookback_days,
        "n_observations": len(snaps),
        "cumulative_return_pct": cum_return - 1.0,
        "mean_daily_return_pct": mean,
        "stdev_daily_return_pct": std,
        "sharpe_annualised": sharpe_annualised,
        "first_snapshot_date": snaps[0]["date"],
        "last_snapshot_date": snaps[-1]["date"],
        "last_equity_value": snaps[-1]["equity_value"],
    }, default=str)


def _impl_get_sector_for_ticker(ticker: str) -> str:
    from finagent.paper_trading import universe
    return json.dumps({"ticker": ticker, "sector": universe.get_sector(ticker)})


# Write tool — returns its own structured payload for the agent's
# benefit (so it sees what landed vs what was rejected).
class CommitResult(BaseModel):
    accepted: int
    rejected: list[dict]
    n_long: int
    n_short: int
    n_neutral: int


def _impl_commit_predictions(date: str, directions_json: str) -> str:
    """Atomic write — every ticker in ``directions_json`` (a JSON map
    {ticker: direction}) is upserted into the predictions table for
    ``date`` with ``source='agent:portfolio_manager'``.

    Validates: directions ∈ {-1, 0, +1}; tickers ∈ NIFTY50_TICKERS.
    Returns counts + any rejected entries.
    """
    from finagent.paper_trading import predictions as ptp, universe

    try:
        directions = json.loads(directions_json)
    except Exception as e:
        return json.dumps({"error": f"directions_json is not valid JSON: {e}"})
    if not isinstance(directions, dict):
        return json.dumps({"error": "directions_json must decode to a JSON object {ticker: direction}"})

    accepted = 0
    rejected: list[dict] = []
    n_long = n_short = n_neutral = 0
    for ticker, raw_dir in directions.items():
        try:
            d = int(raw_dir)
        except (TypeError, ValueError):
            rejected.append({"ticker": ticker, "reason": f"direction not int-coercible: {raw_dir!r}"})
            continue
        if d not in (-1, 0, 1):
            rejected.append({"ticker": ticker, "reason": f"direction must be -1/0/+1, got {d}"})
            continue
        if ticker not in universe.NIFTY50_TICKERS:
            rejected.append({"ticker": ticker, "reason": "not in NIFTY50_TICKERS"})
            continue
        try:
            ptp.record_prediction(
                date=date, ticker=ticker, direction=d,
                source="agent:portfolio_manager",
            )
            accepted += 1
            if d > 0: n_long += 1
            elif d < 0: n_short += 1
            else: n_neutral += 1
        except Exception as e:
            rejected.append({"ticker": ticker, "reason": str(e)})

    logger.info(
        "portfolio_manager: commit_predictions(%s) — accepted=%d (L%d/S%d/N%d) rejected=%d",
        date, accepted, n_long, n_short, n_neutral, len(rejected),
    )
    return json.dumps({
        "accepted": accepted,
        "rejected": rejected,
        "n_long": n_long,
        "n_short": n_short,
        "n_neutral": n_neutral,
    })


# ── System prompt ──────────────────────────────────────────────────


_PORTFOLIO_MANAGER_INSTRUCTIONS = """You are PORTFOLIO_MANAGER for a Nifty 50 daily directional book.

Your goal: assign exactly one of {LONG (+1), SHORT (-1), NEUTRAL (0)}
to each of the 50 tickers, persisted via commit_predictions exactly
once per trading day.

A downstream pipeline turns your direction set into two portfolios
mathematically:

  - equal_weight:  w_i = direction_i / Σ |direction_j|
  - market_cap:    w_i = direction_i × mcap_i / Σ mcap_j  (over nonzero directions)

You do NOT pick prices, run charts, or read news. Trust the upstream
analyst's recommendation + confidence. Your value is portfolio
CONSTRUCTION:

  1. Filter weak calls. Default a ticker to 0 (NEUTRAL) when no
     upstream recommendation exists, OR when confidence < 0.55.
  2. Continuity. Use get_recent_directions to see what you set
     previously. If yesterday you went LONG and the analyst is
     still LONG today, keep LONG — don't flip-flop on noise.
  3. Balance. Avoid extreme net exposure: |n_long - n_short| ≤ 12
     out of 50 unless you have a strong macro thesis. Avoid
     sector concentration > 40% of nonzero positions in one sector
     (use get_sector_for_ticker to compute your draft set's
     sector spread before committing).
  4. Track record awareness. Use get_portfolio_performance to see
     how your equal_weight book has done recently. If you've been
     losing, lean toward smaller nonzero counts (more NEUTRALs)
     until conviction recovers.

PROCEDURE:

  1. Call get_recommendations(today) — see today's upstream verdicts.
  2. Call get_universe(today) — see the 50-ticker universe.
  3. Optionally call get_recent_directions() and get_portfolio_performance()
     for context.
  4. Build a dict mapping each of the 50 tickers to a direction
     in {-1, 0, +1}. Tickers without an upstream recommendation
     should default to 0.
  5. Call commit_predictions(today, directions_json=<json string>)
     EXACTLY ONCE with the full dict.
  6. After commit_predictions returns, output a short markdown
     summary of your decisions: net exposure, sector lean, any
     deliberate flips vs yesterday, anything you'd want a human
     to know.

When uncertain, output 0. Empty exposure is better than wrong
exposure that drags both books.
"""


# ── Runner ─────────────────────────────────────────────────────────


class PortfolioManagerReport(BaseModel):
    """What ``run_portfolio_manager`` returns to the caller."""
    date: str
    status: str = Field(description="ok | failed | unavailable")
    summary: str = Field(default="", description="agent's final markdown summary")
    accepted: int = 0
    rejected: int = 0
    n_long: int = 0
    n_short: int = 0
    n_neutral: int = 0
    error: Optional[str] = None


async def run_portfolio_manager(date: Optional[str] = None) -> PortfolioManagerReport:
    """Run the portfolio manager for ``date`` (default: today UTC).

    Idempotent — re-runs overwrite the day's predictions via
    UNIQUE(date, ticker) on the predictions table.

    Returns a structured report. Never raises — on any failure
    returns ``status='failed'`` with the error message so the
    caller (scheduler / admin endpoint) can surface it cleanly.
    """
    today = date or datetime.now(timezone.utc).date().isoformat()

    try:
        from agents import Agent, Runner, ModelSettings, function_tool
        from finagent.llm import get_model_name
    except ImportError as e:
        logger.warning("portfolio_manager: agents SDK unavailable (%s)", e)
        return PortfolioManagerReport(
            date=today, status="unavailable",
            error=f"agents SDK not importable: {e}",
        )

    # Wrap the impl functions as Agents-SDK tools. We do this inside
    # run_portfolio_manager so the module stays importable in
    # environments without the SDK (tests, CI).
    @function_tool
    async def get_recommendations(date: str) -> str:
        """List today's upstream debate verdicts. Returns JSON with
        {date, n, recommendations: [{ticker, action, confidence,
        target_price, stoploss, rationale, debate_id}]}."""
        return _impl_get_recommendations(date)

    @function_tool
    async def get_universe(date: str) -> str:
        """The 50 Nifty tickers with sector + market cap. JSON shape:
        {date, n, universe: [{ticker, sector, market_cap,
        market_cap_refreshed_at}]}."""
        return _impl_get_universe(date)

    @function_tool
    async def get_recent_directions(ticker: Optional[str] = None, lookback_days: int = 30) -> str:
        """Past directions you committed. When ticker is set, returns
        that ticker's row-by-row history; otherwise a per-day
        aggregate {n_long, n_short, n_flat} across the universe."""
        return _impl_get_recent_directions(ticker, lookback_days)

    @function_tool
    async def get_portfolio_performance(strategy: str = "equal_weight", lookback_days: int = 30) -> str:
        """Your own track record. JSON with cumulative return,
        Sharpe (annualised), mean/stdev of daily returns over the
        lookback window. strategy ∈ {equal_weight, market_cap}."""
        return _impl_get_portfolio_performance(strategy, lookback_days)

    @function_tool
    async def get_sector_for_ticker(ticker: str) -> str:
        """NSE sector for a Nifty 50 ticker. Cheap lookup, no LLM."""
        return _impl_get_sector_for_ticker(ticker)

    @function_tool
    async def commit_predictions(date: str, directions_json: str) -> str:
        """Atomic batch write. directions_json is a JSON string
        encoding a dict {ticker: -1|0|+1}. Validates and upserts;
        returns {accepted, rejected, n_long, n_short, n_neutral}.
        Call exactly ONCE per run."""
        return _impl_commit_predictions(date, directions_json)

    tools = [
        get_recommendations, get_universe, get_recent_directions,
        get_portfolio_performance, get_sector_for_ticker,
        commit_predictions,
    ]

    try:
        model = get_model_name("portfolio_manager")
    except Exception:
        try:
            model = get_model_name("intent_classifier")
        except Exception:
            model = "gpt-4o-mini"

    try:
        agent = Agent(
            name="PortfolioManager",
            instructions=_PORTFOLIO_MANAGER_INSTRUCTIONS,
            model=model,
            model_settings=ModelSettings(),
            tools=tools,
        )
        user_prompt = (
            f"Today is {today}. Build the daily directional book for "
            f"the Nifty 50 paper-trading portfolio. Follow the procedure "
            f"in your system prompt — read recommendations, build a 50-ticker "
            f"direction map, then commit_predictions once."
        )
        result = await Runner.run(
            agent, input=user_prompt,
            max_turns=15,   # 6 tools + the commit + summary leaves slack
        )
        summary_md = (result.final_output or "").strip() if hasattr(result, "final_output") else ""
    except Exception as exc:
        logger.exception("portfolio_manager: agent run failed for %s", today)
        return PortfolioManagerReport(
            date=today, status="failed", error=str(exc),
        )

    # Post-run: read the actual predictions that landed so the report
    # reflects ground truth (the agent's commit_predictions output is
    # already authoritative, but re-reading from the DB is more honest
    # — catches the case where the agent skipped the commit).
    from finagent.paper_trading import store
    preds = store.list_predictions(date=today)
    by_dir = {-1: 0, 0: 0, 1: 0}
    for p in preds:
        d = int(p["direction"])
        by_dir[d] = by_dir.get(d, 0) + 1

    return PortfolioManagerReport(
        date=today,
        status="ok" if preds else "failed",
        summary=summary_md,
        accepted=len(preds),
        rejected=0,
        n_long=by_dir.get(1, 0),
        n_short=by_dir.get(-1, 0),
        n_neutral=by_dir.get(0, 0),
        error=None if preds else "agent did not commit any predictions",
    )
