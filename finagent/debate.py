"""Debate orchestrator — public API contract preserved, internals delegated to the trading panel.

Public entrypoint:
    run_debate(ticker, asset_class, *, rounds, emit) -> {transcript, verdict}

This module USED to drive a linear bull→bear→moderator debate via the
OpenAI Agents SDK. As of 2026-05, it's a thin adapter that delegates
to ``finagent.agents.trading_panel.run_panel`` (TradingAgents-style
8-agent panel on LangGraph + LangChain) and reshapes the panel's
output into the existing ``debates``-table schema + SSE event shape.

Why a wrapper instead of a clean rename: every caller in the stack
(``app.py`` chat endpoint, ``finagent.scheduler``, the synapse UI's
SSE consumer + debate detail page + calendar + performance panel)
already speaks the legacy contract. Reshaping at this seam means
zero changes anywhere else; the panel's richer reasoning ships
under the existing UI without a frontend redesign.

Event mapping (panel → debate-shape):
  panel "phase" / "message" / "tool_call" / "verdict" / "error"
                            ↓ (mostly identity)
  debate "phase" / "message" / "tool_call" / "verdict" / "error"

Speaker mapping (panel → debate-shape):
  market_analyst / news_analyst / fundamentals_analyst / macro_analyst
  bull_researcher (was bull_analyst) / bear_researcher (was bear_analyst)
  research_manager / trader / risk_debator / portfolio_manager (was moderator)

Frontend: ``synapse/src/components/debate-views.tsx`` colour-codes
bull_analyst (emerald) and bear_analyst (rose); other speakers fall
through to a neutral grey panel — works fine for the new agents.

Kill-switch: set ``DEBATE_LEGACY=1`` to force the old SDK-based path
(only useful if the panel has a regression and you need to ship a
hotfix without rolling the panel back; the legacy code lives in
``finagent.debate_legacy``).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional


EmitFn = Callable[[dict], Awaitable[None]]


async def _noop_emit(_: dict) -> None:
    return None


def _legacy_mode() -> bool:
    """Allow flipping back to the OpenAI-Agents-SDK debate via env.

    DEBATE_LEGACY=1 only — every other value (including unset) routes
    through the panel.
    """
    return os.environ.get("DEBATE_LEGACY", "").strip() in {"1", "true", "yes"}


async def run_debate(
    ticker: str,
    asset_class: str = "us_equity",
    *,
    rounds: int = 2,
    emit: Optional[EmitFn] = None,
    debate_id: Optional[str] = None,
    source: str = "user",
) -> dict[str, Any]:
    """Run a multi-agent investment-thesis debate on a single ticker.

    Args:
        ticker: e.g. "AAPL", "BTC-USD", "RELIANCE.NS".
        asset_class: free-form hint passed to the agents
            ("us_equity" / "crypto" / "indian_equity").
        rounds: How many bull→bear rounds before the moderator verdicts.
        emit: async callback for streaming events.
        debate_id: pre-generated id; if None, generated.
        source: 'user' for manual submissions, 'scheduled' for the cron.

    Returns:
        Same shape as the legacy debate:
        {debate_id, ticker, asset_class, rounds, transcript, verdict,
         started_at, finished_at}.

    Implementation: forwards to trading_panel.run_panel, reshapes its
    multi-stage output into a flat transcript + projected verdict.
    """
    if _legacy_mode():
        # Hatch back to the old SDK-based debate. Kept only as a hot-
        # patch escape route; not the default path.
        from .debate_legacy import run_debate as _legacy
        return await _legacy(
            ticker, asset_class, rounds=rounds, emit=emit,
            debate_id=debate_id, source=source,
        )

    emit = emit or _noop_emit

    # Persist the row up front so the calendar / detail page see a
    # 'queued' entry the moment the request lands. Same lifecycle as
    # the legacy debate.
    own_row = debate_id is None
    if own_row:
        from .experiments import get_store
        try:
            _row = get_store().create_debate(
                ticker=ticker, asset_class=asset_class,
                rounds=rounds, source=source,
            )
            debate_id = _row.id
            get_store().update_debate(debate_id, status="running")
        except Exception:
            logging.exception("run_debate: pre-run persistence failed")
            debate_id = uuid.uuid4().hex[:16]

    started_at = time.time()
    transcript: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    final_verdict: Optional[dict[str, Any]] = None

    # ── Incremental persist helper ────────────────────────────────
    # Without this, the /app/debate/<id> detail page (which polls the
    # persisted row every 4s) shows an empty transcript for the entire
    # 5-10 minute panel run. Persisting after every message + every
    # tool_call is cheap (one SQLite UPDATE) and gives the user live
    # progress visibility on both the streaming hub AND the detail page.

    def _persist_progress() -> None:
        if not debate_id:
            return
        try:
            from .experiments import get_store
            get_store().update_debate(
                debate_id,
                transcript=transcript,
                evidence=evidence,
            )
        except Exception:
            logging.exception("run_debate: incremental persist failed (non-fatal)")

    # ── Adapter emit hook ─────────────────────────────────────────
    # Forward panel events to the caller's emit, also accumulate
    # transcript + evidence + persist incrementally.

    async def _adapter_emit(ev: dict) -> None:
        nonlocal final_verdict
        typ = ev.get("type")

        if typ == "message":
            # Panel emits one "message" per analyst report + each
            # bull/bear turn + the risk review. Same shape the legacy
            # debate produced; just pass through with debate_id stamp.
            msg = {
                "type": "message",
                "phase": ev.get("phase", ""),
                "speaker": ev.get("speaker", ""),
                "text": ev.get("text", ""),
                "ts": ev.get("ts", time.time()),
            }
            # Forward interim-flag through so the UI's compact-rendering
            # logic kicks in for status messages.
            if ev.get("interim"):
                msg["interim"] = True
            transcript.append(msg)
            await emit(msg)
            _persist_progress()
            return

        if typ == "tool_call":
            evidence.append({
                "phase": ev.get("phase", ""),
                "speaker": ev.get("speaker", ""),
                "tool": ev.get("tool", ""),
                "args": ev.get("args", ""),
                "output": ev.get("output", ""),
                "call_id": ev.get("call_id", ""),
                "ts": ev.get("ts", time.time()),
            })
            await emit(ev)
            # Persist evidence so the detail page's EvidenceList stays
            # current. Tool calls fire 4-8 times per analyst, ~30 times
            # total — still cheap as long as each is a single UPDATE.
            _persist_progress()
            return

        if typ == "verdict":
            final_verdict = ev.get("data", {})
            await emit(ev)
            # Persist verdict immediately so a page-reload during the
            # final stages still sees the moderator's call.
            if debate_id:
                try:
                    from .experiments import get_store
                    get_store().update_debate(
                        debate_id,
                        verdict=final_verdict,
                        transcript=transcript,
                        evidence=evidence,
                    )
                except Exception:
                    logging.exception("run_debate: verdict persist failed")
            return

        if typ in ("phase", "started", "done", "error",
                   "structured_retry", "research_plan",
                   "trader_proposal", "portfolio_decision"):
            # Pass through — UI either displays them or ignores.
            await emit(ev)
            return

        # Unknown event types — forward anyway so future panel events
        # don't get silently swallowed.
        await emit(ev)

    # ── Run the panel ─────────────────────────────────────────────

    try:
        from .agents.trading_panel import run_panel
        result = await run_panel(
            ticker=ticker,
            asset_class=asset_class,
            rounds=rounds,
            emit=_adapter_emit,
            panel_id=debate_id,
            persist=False,    # we own the persistence here, not the panel
            source=source,
        )
    except Exception as exc:
        logging.exception("run_debate: panel raised for %s", ticker)
        if own_row:
            try:
                from .experiments import get_store
                get_store().update_debate(
                    debate_id, status="failed",
                    transcript=transcript, evidence=evidence,
                    error=str(exc), finished=True,
                )
            except Exception:
                logging.exception("run_debate: failed to persist failure")
        await emit({"type": "error", "phase": "error",
                    "text": f"{type(exc).__name__}: {exc}",
                    "ts": time.time()})
        raise

    finished_at = time.time()
    verdict = final_verdict or result.get("verdict") or {}

    # ── Persist completion ────────────────────────────────────────
    if own_row:
        try:
            from .experiments import get_store
            get_store().update_debate(
                debate_id,
                status="completed",
                transcript=transcript,
                verdict=verdict,
                evidence=evidence,
                finished=True,
            )
        except Exception:
            logging.exception("run_debate: failed to persist completion")

    return {
        "debate_id": debate_id,
        "ticker": ticker,
        "asset_class": asset_class,
        "rounds": rounds,
        "transcript": transcript,
        "verdict": verdict,
        "started_at": started_at,
        "finished_at": finished_at,
        # Panel-specific extras for callers that want them. Existing
        # callers don't read these; new UI sections can.
        "panel": {
            "analyst_reports": result.get("analyst_reports"),
            "research_plan": result.get("research_plan"),
            "trader_proposal": result.get("trader_proposal"),
            "risk_review": result.get("risk_review"),
            "portfolio_decision": result.get("portfolio_decision"),
        },
    }
