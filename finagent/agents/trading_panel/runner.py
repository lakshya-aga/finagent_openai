"""Public entrypoint: ``run_panel(...)``.

Sets up the streaming hook, builds initial state, invokes the
LangGraph, and returns a dict shaped like ``finagent.debate.run_debate``
so the API layer + persistence + frontend stay backwards-compatible.

Persists a row in the existing ``debates`` table marked
``source="panel"`` (new value, alongside "user" / "scheduled") so the
calendar + detail pages can filter / colour / route them. Each stage's
structured output is preserved both in the row's ``evidence_json``
trail (raw tool calls) AND in the verdict_json (the final
PortfolioDecision projected to DebateVerdict shape for back-compat).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from . import nodes
from .graph import build_panel_graph
from .schemas import PortfolioDecision, PortfolioRating
from .state import initial_state


logger = logging.getLogger(__name__)


EmitFn = Callable[[dict], Awaitable[None]]


async def _noop(_: dict) -> None:
    return None


async def run_panel(
    ticker: str,
    asset_class: str = "us_equity",
    *,
    rounds: int = 2,
    emit: Optional[EmitFn] = None,
    panel_id: Optional[str] = None,
    persist: bool = True,
    source: str = "panel",
) -> dict[str, Any]:
    """Run the full trading panel.

    Args:
        ticker: e.g. "AAPL", "RELIANCE.NS", "BTC-USD".
        asset_class: hint for the analysts ("us_equity" / "indian_equity" /
            "crypto"). Surfaces in the system prompts.
        rounds: number of bull→bear→bull→bear rounds (default 2 = four
            researcher turns total).
        emit: async callback for streaming events. Same shape as
            finagent.debate. Pass None for non-streaming use (tests / cron).
        panel_id: pre-generated id. If None, generated here.
        persist: persist a row in the debates table (source='panel').
            Set False for dry-run unit tests.
        source: store-row source field ("panel" / "user" / "scheduled").

    Returns:
        {panel_id, ticker, asset_class, rounds, started_at, finished_at,
         analyst_reports, debate_transcript, research_plan, trader_proposal,
         risk_review, portfolio_decision, verdict, evidence}

        ``verdict`` is the PortfolioDecision projected onto the
        DebateVerdict shape so existing callers (verdict card, calendar
        pills, performance panel) keep working.
    """
    emit = emit or _noop
    panel_id = panel_id or uuid.uuid4().hex[:16]
    started_at = time.time()

    nodes.set_emit(emit)
    try:
        await emit({
            "type": "started",
            "panel_id": panel_id,
            "ticker": ticker,
            "asset_class": asset_class,
            "rounds": rounds,
            "ts": started_at,
        })

        # ── Optional persistence: create the row up front ──
        if persist:
            try:
                from finagent.experiments import get_store
                store = get_store()
                # Reuse the debates table — same UI / API path. The
                # 'panel' source value lets us filter later.
                store.create_debate(
                    ticker=ticker,
                    asset_class=asset_class,
                    rounds=rounds,
                    source=source,
                )
                # NOTE: we don't pass panel_id into create_debate (its
                # signature generates its own id). For the panel, we
                # treat the persistence as best-effort lineage rather
                # than primary state.
            except Exception:
                logger.exception("run_panel: pre-run persistence failed (continuing)")

        # ── Build initial state ──
        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = initial_state(
            ticker=ticker, asset_class=asset_class,
            today_iso=today_iso, panel_id=panel_id, rounds=rounds,
        )

        # ── Run the graph ──
        graph = build_panel_graph()
        # recursion_limit must comfortably exceed the worst-case node
        # count: 3 analysts + 2*rounds debate + 4 (mgr/trader/risk/pm).
        # For rounds=2 that's 11 nodes; cap at 50 for headroom.
        final_state = await graph.ainvoke(
            state,
            config={"recursion_limit": 50},
        )

        # ── Project PortfolioDecision → DebateVerdict for back-compat ──
        decision_dict = final_state.get("portfolio_decision") or {}
        verdict = _project_to_verdict(decision_dict)

        finished_at = time.time()

        result: dict[str, Any] = {
            "panel_id": panel_id,
            "ticker": ticker,
            "asset_class": asset_class,
            "rounds": rounds,
            "started_at": started_at,
            "finished_at": finished_at,
            "analyst_reports": {
                "market": final_state.get("market_report"),
                "news": final_state.get("news_report"),
                "fundamentals": final_state.get("fundamentals_report"),
            },
            "debate_transcript": final_state.get("investment_debate") or {},
            "research_plan": final_state.get("research_plan"),
            "trader_proposal": final_state.get("trader_proposal"),
            "risk_review": final_state.get("risk_review"),
            "portfolio_decision": decision_dict,
            "verdict": verdict,
            "evidence": final_state.get("evidence", []),
        }

        await emit({
            "type": "verdict",
            "phase": "verdict",
            "speaker": "portfolio_manager",
            "data": verdict,
            "ts": finished_at,
        })
        await emit({"type": "done", "ts": finished_at})

        return result

    except Exception as exc:
        logger.exception("run_panel failed for %s", ticker)
        await emit({
            "type": "error",
            "phase": "error",
            "text": f"{type(exc).__name__}: {exc}",
            "ts": time.time(),
        })
        raise
    finally:
        # Always uninstall the emit hook so a future panel run with a
        # different streamer doesn't accidentally fire into a closed
        # client.
        nodes.set_emit(None)


# ── Back-compat verdict projection ──────────────────────────────────


_RATING_TO_ACTION = {
    PortfolioRating.BUY.value:         "buy",
    PortfolioRating.OVERWEIGHT.value:  "buy",
    PortfolioRating.HOLD.value:        "avoid",
    PortfolioRating.UNDERWEIGHT.value: "sell",
    PortfolioRating.SELL.value:        "sell",
}


def _project_to_verdict(decision: dict[str, Any]) -> dict[str, Any]:
    """Project the panel's PortfolioDecision into the legacy DebateVerdict
    shape so the existing verdict card / calendar / performance panel
    don't need to know about the new schema.

    Mapping:
        BUY/OVERWEIGHT → action='buy'
        HOLD           → action='avoid'
        UNDERWEIGHT/SELL → action='sell'
    """
    if not decision:
        return {
            "action": "avoid", "target_price": None, "stoploss": None,
            "time_horizon": "unknown", "key_metrics": [],
            "rationale": "Panel produced no decision.", "confidence": 0.0,
        }
    rating = decision.get("rating") or PortfolioRating.HOLD.value
    return {
        "action": _RATING_TO_ACTION.get(rating, "avoid"),
        "target_price": decision.get("price_target"),
        "stoploss": decision.get("stop_loss"),
        "time_horizon": decision.get("time_horizon") or "unspecified",
        "key_metrics": [
            {"name": "rating", "value": rating, "why_it_matters": "5-tier panel rating"},
            *[{"name": "risk", "value": r[:80], "why_it_matters": "Identified by panel"}
              for r in (decision.get("key_risks") or [])[:4]],
        ],
        "rationale": (decision.get("executive_summary") or "")[:600],
        "confidence": float(decision.get("confidence") or 0.0),
    }
