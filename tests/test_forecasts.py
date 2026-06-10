"""Forecast feature tests — aggregation math, Exa client guards, store."""

from __future__ import annotations

import pytest

from finagent import exa
from finagent import forecasts as fstore
from finagent.agents.forecaster import aggregate_probabilities


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAGENT_EXPERIMENT_DB", str(tmp_path / "f.db"))


# ── aggregation ──────────────────────────────────────────────────────


def test_aggregate_median_and_spread():
    agg = aggregate_probabilities([0.6, 0.7, 0.65])
    assert agg["probability"] == 0.65
    assert agg["p_low"] == 0.6 and agg["p_high"] == 0.7
    assert agg["low_agreement"] is False

    # Wide disagreement flags low agreement.
    agg = aggregate_probabilities([0.2, 0.55, 0.8])
    assert agg["low_agreement"] is True

    with pytest.raises(ValueError):
        aggregate_probabilities([])


# ── exa client guards ────────────────────────────────────────────────


def test_exa_requires_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert exa.exa_available() is False
    with pytest.raises(exa.ExaNotConfigured):
        exa.search("anything")


def test_exa_parses_defensively(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "A", "url": "https://x.com/a", "text": "t" * 9999},
                    {"weird": "shape"},  # missing every expected field
                    "not-a-dict",
                ]
            }

    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: FakeResp())
    out = exa.search("test query", num_results=99)  # also tests the cap path
    assert len(out) == 2  # the non-dict entry dropped
    assert out[0]["url"] == "https://x.com/a"
    assert len(out[0]["snippet"]) <= 1500  # text capped
    assert out[1] == {"title": "", "url": "", "published_date": "", "snippet": ""}


# ── store roundtrip + resolution ─────────────────────────────────────


def test_forecast_store_roundtrip_and_resolution(tmp_db):
    fid = fstore.save_forecast(
        {
            "question": "Will it rain?",
            "reframed_question": "Measurable rain falls in SG before 2026-12-31.",
            "probability": 0.72,
            "p_low": 0.65,
            "p_high": 0.8,
            "n_ensemble": 3,
            "low_agreement": False,
            "rationale": "Base rate is high.",
            "key_drivers": ["monsoon season"],
            "evidence": [{"fact": "It is monsoon season", "source_url": "https://e.x"}],
            "resolution_criteria": "Official SG weather record.",
            "horizon": "2026-12-31",
            "base_rate": 0.7,
        },
        owner="alice@x.com",
    )

    f = fstore.get_forecast(fid)
    assert f is not None
    assert f["probability"] == 0.72
    assert f["key_drivers"] == ["monsoon season"]
    assert f["evidence"][0]["fact"] == "It is monsoon season"
    assert f["status"] == "open"
    assert f["owner"] == "alice@x.com"

    # Listing: owner filter works; unknown owner sees nothing.
    assert len(fstore.list_forecasts(owner="alice@x.com")) == 1
    assert fstore.list_forecasts(owner="bob@x.com") == []
    assert len(fstore.list_forecasts()) == 1

    # Resolution flips status exactly once.
    assert fstore.resolve_forecast(fid, 1.0) is True
    assert fstore.resolve_forecast(fid, 0.0) is False  # already resolved
    f = fstore.get_forecast(fid)
    assert f["status"] == "resolved" and f["outcome"] == 1.0

    with pytest.raises(ValueError):
        fstore.resolve_forecast(fid, 0.5)
