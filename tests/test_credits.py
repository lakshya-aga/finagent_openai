"""Credits ledger + debate ownership/cache tests.

Pure-SQLite tier — no LLM, no network, no kernel. The credits module
resolves its DB path from FINAGENT_EXPERIMENT_DB at call time, so each
test points it at a tmp file via monkeypatch.
"""

from __future__ import annotations

import threading
import time

import pytest

from finagent import credits
from finagent.experiments import ExperimentStore


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "credits-test.db"
    monkeypatch.setenv("FINAGENT_EXPERIMENT_DB", str(db))
    return db


# ── accounts + ledger ────────────────────────────────────────────────


def test_signup_bonus_granted_once(tmp_db, monkeypatch):
    monkeypatch.setenv("CREDITS_SIGNUP_BONUS", "3")
    first = credits.ensure_account("Alice@Example.com")
    assert first == {"user_id": "alice@example.com", "balance": 3, "created": True}

    again = credits.ensure_account("alice@example.com ")
    assert again["created"] is False
    assert again["balance"] == 3  # bonus did NOT stack

    ledger = credits.history("alice@example.com")
    assert len(ledger) == 1
    assert ledger[0]["reason"] == "signup_bonus"
    assert ledger[0]["delta"] == 3


def test_charge_happy_path_and_insufficient(tmp_db):
    credits.ensure_account("bob@x.com")  # bonus = 3 default
    ok, bal = credits.charge("bob@x.com", 1, "analysis:NVDA")
    assert ok and bal == 2

    ok, bal = credits.charge("bob@x.com", 5, "analysis:AAPL")
    assert not ok and bal == 2  # untouched on refusal

    # Unknown account never charges.
    ok, bal = credits.charge("ghost@x.com", 1, "analysis:TSLA")
    assert not ok and bal == 0


def test_charge_is_atomic_under_race(tmp_db, monkeypatch):
    monkeypatch.setenv("CREDITS_SIGNUP_BONUS", "1")
    credits.ensure_account("racer@x.com")

    results: list[bool] = []
    barrier = threading.Barrier(2)

    def _try():
        barrier.wait()
        ok, _ = credits.charge("racer@x.com", 1, "analysis:RACE")
        results.append(ok)

    threads = [threading.Thread(target=_try) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Balance 1, two concurrent charges: exactly one wins, never both.
    assert sorted(results) == [False, True]
    assert credits.get_balance("racer@x.com") == 0


def test_refund_restores_balance_and_journals(tmp_db):
    credits.ensure_account("carol@x.com")
    credits.charge("carol@x.com", 1, "analysis:MSFT", ref_id="d123")
    new_bal = credits.refund("carol@x.com", 1, ref_id="d123")
    assert new_bal == 3

    reasons = [e["reason"] for e in credits.history("carol@x.com")]
    assert "refund:failed_run" in reasons


def test_grant_creates_account_without_bonus(tmp_db):
    bal = credits.grant("purchaser@x.com", 50, "purchase:starter")
    assert bal == 50  # no signup bonus stacked on top
    assert credits.get_balance("purchaser@x.com") == 50

    with pytest.raises(ValueError):
        credits.grant("purchaser@x.com", 0, "bad")
    with pytest.raises(ValueError):
        credits.grant("", 5, "bad")


# ── debate ownership + same-day cache ────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    return ExperimentStore(path=tmp_path / "experiments-test.db")


def test_debate_owner_roundtrip_and_filtering(store):
    a = store.create_debate(
        ticker="NVDA", asset_class="us_equity", rounds=2, owner="alice@x.com"
    )
    store.create_debate(
        ticker="AAPL", asset_class="us_equity", rounds=2, owner="bob@x.com"
    )
    store.create_debate(ticker="MSFT", asset_class="us_equity", rounds=1)  # system

    assert store.get_debate(a.id).owner == "alice@x.com"
    assert "owner" in store.get_debate(a.id).as_public_dict()

    mine = store.list_debates(owner="alice@x.com")
    assert [d.ticker for d in mine] == ["NVDA"]
    assert store.count_debates(owner="alice@x.com") == 1
    # No owner filter keeps the legacy list-everything behaviour.
    assert store.count_debates() == 3


def test_cache_hit_requires_completed_same_day_enough_rounds(store):
    d = store.create_debate(
        ticker="NVDA", asset_class="us_equity", rounds=2, owner="alice@x.com"
    )

    # Queued → no cache hit.
    assert (
        store.find_cached_debate(ticker="NVDA", asset_class="us_equity", min_rounds=1)
        is None
    )

    store.update_debate(d.id, status="completed", verdict={"action": "buy"}, finished=True)

    # Completed today → hit, regardless of requester (owner ignored).
    hit = store.find_cached_debate(ticker="NVDA", asset_class="us_equity", min_rounds=1)
    assert hit is not None and hit.id == d.id

    # A 2-round cached run satisfies a 2-round request…
    assert (
        store.find_cached_debate(ticker="NVDA", asset_class="us_equity", min_rounds=2)
        is not None
    )
    # …but never a deeper one.
    assert (
        store.find_cached_debate(ticker="NVDA", asset_class="us_equity", min_rounds=3)
        is None
    )

    # Different ticker / asset class → miss.
    assert (
        store.find_cached_debate(ticker="AAPL", asset_class="us_equity", min_rounds=1)
        is None
    )
    assert (
        store.find_cached_debate(ticker="NVDA", asset_class="crypto", min_rounds=1)
        is None
    )


def test_cache_ignores_stale_runs(store):
    d = store.create_debate(ticker="TSLA", asset_class="us_equity", rounds=2)
    store.update_debate(d.id, status="completed", finished=True)
    # Backdate the run two days — yesterday's analysis must not serve today.
    with store._conn() as conn:
        conn.execute(
            "UPDATE debates SET started_at = ? WHERE id = ?",
            (time.time() - 2 * 86400, d.id),
        )
    assert (
        store.find_cached_debate(ticker="TSLA", asset_class="us_equity", min_rounds=1)
        is None
    )
