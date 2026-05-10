"""Unit tests for the ``panel`` SDK (Phase 2).

Round-trips:
  * save_model + load_model + manifest
  * export_signal (parquet + manifest + DB row + version row)
  * list_models / list_signals walk
  * get_inference_inputs window math + fetch_fn invocation
  * Naming validation rejects bad slugs

Uses the ``isolated_outputs`` fixture so each test gets a fresh
``outputs/`` + ``experiments.db`` and never touches the real ones.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Module-level helper: pickleable (local classes inside test bodies are not).
class _MultiplyModel:
    def __init__(self, mult: int):
        self.mult = mult

    def predict(self, x):
        return x * self.mult


# ── save_model + load_model ────────────────────────────────────────────


def test_save_load_model_round_trip(isolated_outputs):
    import panel

    saved = panel.save_model(
        "spy-momentum-252d",
        _MultiplyModel(3),
        metadata={"framework": "test", "features": ["rs252"]},
    )
    assert saved.exists()
    assert saved.name == "manifest.json"

    loaded = panel.load_model("spy-momentum-252d")
    assert loaded.predict(7) == 21

    mf = panel.model_manifest("spy-momentum-252d")
    assert mf["name"] == "spy-momentum-252d"
    assert mf["metadata"]["features"] == ["rs252"]
    assert mf["kind"] == "model"
    assert mf["size_bytes"] > 0


def test_save_model_replaces_in_place(isolated_outputs):
    import panel

    panel.save_model("foo", {"v": 1}, metadata={"version": 1})
    panel.save_model("foo", {"v": 2}, metadata={"version": 2})
    loaded = panel.load_model("foo")
    assert loaded == {"v": 2}
    assert panel.model_manifest("foo")["metadata"]["version"] == 2


def test_load_model_missing_raises(isolated_outputs):
    import panel

    with pytest.raises(FileNotFoundError):
        panel.load_model("does-not-exist")


def test_list_models_walks_dir(isolated_outputs):
    import panel

    panel.save_model("alpha", "model-a", metadata={"k": "a"})
    panel.save_model("beta", "model-b", metadata={"k": "b"})
    listed = panel.list_models()
    names = sorted(m["name"] for m in listed)
    assert names == ["alpha", "beta"]


# ── export_signal ──────────────────────────────────────────────────────


def _toy_series(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.standard_normal(n).cumsum() * 0.05, index=idx, name="x")


def test_export_signal_round_trip_parquet_manifest_and_db(isolated_outputs):
    import panel

    s = _toy_series(252)
    panel.export_signal(
        "spy-momentum-252d",
        s,
        metadata={
            "frequency": "daily",
            "universe": ["SPY"],
            "description": "test signal",
            "project": "p1",
            "template": "regime_modeling",
            "recipe_fingerprint": "abc123",
        },
        run_id="run_xyz",
    )

    # On disk
    series_path = isolated_outputs / "signals" / "spy-momentum-252d" / "series.parquet"
    manifest_path = isolated_outputs / "signals" / "spy-momentum-252d" / "manifest.json"
    assert series_path.exists()
    assert manifest_path.exists()
    rt = pd.read_parquet(series_path)
    assert len(rt) == 252
    assert "value" in rt.columns

    mf = panel.signal_manifest("spy-momentum-252d")
    assert mf["n_observations"] == 252
    assert mf["last_value"] is not None
    assert mf["metadata"]["universe"] == ["SPY"]
    assert mf["run_id"] == "run_xyz"

    # Round-tripped through load_signal
    loaded = panel.load_signal("spy-momentum-252d")
    assert len(loaded) == 252
    assert loaded.iloc[-1] == pytest.approx(s.iloc[-1])

    # In the DB
    conn = sqlite3.connect(str(isolated_outputs / "experiments.db"))
    row = conn.execute("SELECT * FROM signals WHERE name = ?", ("spy-momentum-252d",)).fetchone()
    assert row is not None
    # PRAGMA table_info returns (cid, name, type, notnull, dflt, pk) — name is index 1.
    cols = {c[1]: row[i] for i, c in enumerate(conn.execute("PRAGMA table_info(signals)").fetchall())}
    assert cols["name"] == "spy-momentum-252d"
    assert cols["frequency"] == "daily"
    assert json.loads(cols["universe_json"]) == ["SPY"]
    assert cols["run_id"] == "run_xyz"
    assert cols["recipe_fingerprint"] == "abc123"
    assert cols["template"] == "regime_modeling"
    assert cols["project"] == "p1"
    assert cols["n_observations"] == 252
    assert cols["status"] == "active"

    versions = conn.execute("SELECT count(*) FROM signal_versions WHERE signal_id = ?", (cols["id"],)).fetchone()[0]
    assert versions == 1


def test_export_signal_second_export_is_a_new_version(isolated_outputs):
    import panel

    panel.export_signal("foo", _toy_series(10), metadata={"frequency": "daily"})
    panel.export_signal("foo", _toy_series(20, seed=1), metadata={"frequency": "daily"})
    panel.export_signal("foo", _toy_series(30, seed=2), metadata={"frequency": "daily"})

    conn = sqlite3.connect(str(isolated_outputs / "experiments.db"))
    rows = conn.execute("SELECT count(*) FROM signal_versions").fetchone()[0]
    assert rows == 3
    rows = conn.execute("SELECT count(*) FROM signals").fetchone()[0]
    assert rows == 1
    n_obs = conn.execute("SELECT n_observations FROM signals WHERE name = 'foo'").fetchone()[0]
    assert n_obs == 30


def test_export_signal_drops_nans_and_sorts_index(isolated_outputs):
    import panel

    idx = pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01", "2024-04-01"])
    s = pd.Series([1.0, 2.0, float("nan"), 4.0], index=idx)
    panel.export_signal("nan-test", s, metadata={"frequency": "daily"})
    loaded = panel.load_signal("nan-test")
    assert len(loaded) == 3  # NaN dropped
    assert list(loaded.index) == sorted(loaded.index)


def test_export_signal_rejects_empty(isolated_outputs):
    import panel

    with pytest.raises(ValueError, match="empty"):
        panel.export_signal("empty-sig", pd.Series([], dtype=float), metadata={})


def test_export_signal_accepts_dataframe_with_one_column(isolated_outputs):
    import panel

    s = _toy_series(50)
    df = s.to_frame(name="rs")
    panel.export_signal("df-test", df, metadata={"frequency": "daily"})
    assert panel.signal_manifest("df-test")["n_observations"] == 50


def test_export_signal_rejects_multicolumn_dataframe(isolated_outputs):
    import panel

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="single-column"):
        panel.export_signal("multicol", df, metadata={})


# ── list_signals ───────────────────────────────────────────────────────


def test_list_signals_walks_disk(isolated_outputs):
    import panel

    panel.export_signal("alpha", _toy_series(10), metadata={"frequency": "daily"})
    panel.export_signal("bravo", _toy_series(20), metadata={"frequency": "weekly"})
    listed = panel.list_signals()
    names = sorted(s["name"] for s in listed)
    assert names == ["alpha", "bravo"]


# ── get_inference_inputs ───────────────────────────────────────────────


def test_get_inference_inputs_invokes_fetch_fn_with_window(isolated_outputs):
    import panel

    panel.save_model("spy", {"k": 1}, metadata={"lookback_days": 90})
    captured: dict = {}

    def fake_fetch(window):
        captured["window"] = window
        return pd.DataFrame({"x": [1.0]})

    df = panel.get_inference_inputs("spy", fetch_fn=fake_fetch)
    assert len(df) == 1
    assert "window" in captured
    start, end = captured["window"]
    # ~90 days apart
    delta = (pd.Timestamp(end) - pd.Timestamp(start)).days
    assert delta in (89, 90, 91)


def test_get_inference_inputs_lookback_override_wins(isolated_outputs):
    import panel

    panel.save_model("spy", {"k": 1}, metadata={"lookback_days": 90})
    captured = {}

    def fake_fetch(window):
        captured["window"] = window
        return None

    panel.get_inference_inputs("spy", fetch_fn=fake_fetch, lookback_days=10)
    start, end = captured["window"]
    assert (pd.Timestamp(end) - pd.Timestamp(start)).days in (9, 10, 11)


# ── name validation ────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "",
    "a",                    # too short (min 4 with leading + trailing alnum)
    "Foo",                  # uppercase
    "foo_bar",              # underscore
    "foo bar",              # space
    "foo.bar",              # dot
    "-foo",                 # leading hyphen
    "foo-",                 # trailing hyphen
    "x" * 100,              # too long
])
def test_name_validation_rejects_bad_slugs(isolated_outputs, bad):
    import panel

    with pytest.raises((ValueError, TypeError)):
        panel.save_model(bad, {})


@pytest.mark.parametrize("good", [
    "foo",
    "spy-momentum-252d",
    "btc-vol-targeting-12mo",
    "x-y-z",
    "abc",
])
def test_name_validation_accepts_kebab(isolated_outputs, good):
    import panel

    # Should not raise — also writes a model so we know the path round-trips.
    panel.save_model(good, {"k": 1}, metadata={})
    assert panel.model_manifest(good)["name"] == good
