"""Cost-lever unit tests: pricing prefix-match, usage counter, tier knobs."""

from __future__ import annotations

from finagent.agents.trading_panel import usage
from finagent.agents.trading_panel.tools import _safe_call
from finagent.cost_tracking import estimate_cost_usd


def test_pricing_longest_prefix_wins():
    """Dated model names must price at their OWN tier, not the gpt-5
    base rate ('gpt-5' prefix shadowing 'gpt-5-mini' overpriced mini
    traffic ~35×, observed live)."""
    mini = estimate_cost_usd("gpt-5-mini-2025-08-07", 1_000_000, 0)
    nano = estimate_cost_usd("gpt-5-nano-2025-08-07", 1_000_000, 0)
    full = estimate_cost_usd("gpt-5-2025-08-07", 1_000_000, 0)
    assert mini == 0.25
    assert nano == 0.05
    assert full == 5.0
    assert estimate_cost_usd("totally-unknown-model", 1_000_000, 0) == 0.0


def test_usage_counter_snapshot_diff():
    before = usage.snapshot()
    usage._bump("gpt-5-mini", 1000, 200, 50)
    usage._bump("gpt-5-mini", 500, 0, 25)
    usage._bump("gpt-5-nano", 100, 0, 10)
    delta = usage.diff(before, usage.snapshot())
    assert delta["gpt-5-mini"] == {
        "input": 1500,
        "cached_input": 200,
        "output": 75,
        "calls": 2,
    }
    assert delta["gpt-5-nano"]["calls"] == 1


def test_tool_output_cap(monkeypatch):
    monkeypatch.setenv("PANEL_TOOL_OUTPUT_MAX_CHARS", "1000")
    big = _safe_call("t", lambda: {"data": "x" * 50_000})
    assert len(big) < 1100
    assert "TRUNCATED" in big
    small = _safe_call("t", lambda: {"data": "ok"})
    assert "TRUNCATED" not in small


def test_tier_knobs_env_parsing(monkeypatch):
    from finagent.agents.stock_analyst import _panel_max, _panel_min_conf

    assert _panel_min_conf() == 0.6  # default
    assert _panel_max() == 15  # default
    monkeypatch.setenv("STOCK_ANALYST_PANEL_MIN_CONF", "0.75")
    monkeypatch.setenv("STOCK_ANALYST_PANEL_MAX", "8")
    assert _panel_min_conf() == 0.75
    assert _panel_max() == 8
    monkeypatch.setenv("STOCK_ANALYST_PANEL_MAX", "garbage")
    assert _panel_max() == 15  # falls back, never raises


def test_open_position_tickers_safe_on_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAGENT_EXPERIMENT_DB", str(tmp_path / "x.db"))
    from finagent.agents.stock_analyst import _open_position_tickers

    assert _open_position_tickers() == set()
