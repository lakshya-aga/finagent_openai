"""End-to-end notebook → signal test for the SPY 252-day rolling Sharpe.

What this test exercises (the actual research-loop product surface):

  1. Build a notebook on disk that loads SPY OHLCV (offline fixture so
     the test doesn't depend on yfinance / network), computes the
     rolling 252-day Sharpe ratio, fits a tiny model, persists the
     model via `panel.save_model`, and exports the signal series via
     `panel.export_signal`.
  2. Execute the notebook in-process via nbclient + the host's stock
     `python3` kernel (no need for the `finagent-python` kernelspec to
     run the test — the SDK has no docker dependencies).
  3. Verify the produced artefacts:
       - `outputs/models/spy-momentum-252d/{model.pkl, manifest.json}`
       - `outputs/signals/spy-momentum-252d/{series.parquet, manifest.json}`
       - `signals` row + `signal_versions` row in experiments.db
  4. Tag the executed notebook with cell roles, then split into
     `train.py` + `infer.py`. Verify both scripts contain the right
     statements (data load + model fit ⊆ train, data load + load_model
     + export ⊆ infer).
  5. Smoke the inference path: import infer.py's logic directly,
     reload the model with `panel.load_model`, and predict on a fresh
     window — assert finite output.

This is a unit-tier test (no LLM, no network). The companion
integration test that drives the actual chat-orchestrator agent +
yfinance is gated by the `needs_openai` + `needs_yfinance` markers
and lives in `test_spy_chat_workflow.py` (not shipped here yet).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


# ── Synthetic SPY OHLCV fixture (offline) ─────────────────────────────


def _synthetic_spy(n_years: int = 5, seed: int = 42) -> pd.DataFrame:
    """Reproducible synthetic daily OHLCV indexed by trading days.

    Drift + lognormal volatility — close enough to a real series for the
    notebook's Sharpe computation to produce sensible numbers, with no
    network dependency.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * 252
    daily_returns = rng.normal(loc=0.0004, scale=0.012, size=n_days)
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    return pd.DataFrame({
        "Open":  close * (1 + rng.normal(0, 0.001, n_days)),
        "High":  close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
        "Low":   close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
        "Close": close,
        "Volume": rng.integers(50_000_000, 150_000_000, n_days),
    }, index=idx)


# ── Notebook-text builder (one job per cell — follows the orchestrator rule) ──


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_spy_momentum_notebook(spy_csv: Path) -> nbformat.NotebookNode:
    """Construct the notebook in memory. Each cell does ONE job per
    ``finagent.cells.classifier`` rules so the splitter can partition
    cleanly into train.py / infer.py."""
    nb = nbformat.v4.new_notebook()

    cells = []

    # Bootstrap cell — ensures ``panel`` is importable from the kernel
    # subprocess. Tagged as imports so the splitter still routes
    # subsequent cells correctly. In production the panel package is
    # installed into the kernel's site-packages so this isn't needed;
    # for the test we inject the repo root onto sys.path.
    cells.append(nbformat.v4.new_code_cell(
        "import sys\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
    ))

    # Cell 1 — imports (role: imports)
    cells.append(nbformat.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "from sklearn.linear_model import LinearRegression\n"
        "import panel\n"
    ))

    # Cell 1 — data_load (role: data_load)
    cells.append(nbformat.v4.new_code_cell(
        f"df = pd.read_csv({str(spy_csv)!r}, index_col=0, parse_dates=True)\n"
        "assert len(df) > 0, 'data_load failed'\n"
    ))

    # Cell 2 — preprocess: returns + rolling Sharpe (role: preprocess)
    cells.append(nbformat.v4.new_code_cell(
        "rets = df['Close'].pct_change()\n"
        "rolling_mean = rets.rolling(252).mean()\n"
        "rolling_std  = rets.rolling(252).std()\n"
        "rolling_sharpe_252 = (rolling_mean / rolling_std).dropna() * (252 ** 0.5)\n"
    ))

    # Cell — preprocess: align X / y windows for fitting.
    # (role: preprocess only — doing the next-step shift + dropna here
    # so the train cell is purely .fit() per the one-role-per-cell rule.)
    cells.append(nbformat.v4.new_code_cell(
        "rs = rolling_sharpe_252.copy()\n"
        "y  = rets.reindex(rs.index).shift(-1).dropna()\n"
        "X  = rs.reindex(y.index).to_frame(name='rs252')\n"
    ))

    # Cell — train: fit a LinearRegression. Pure-train cell so the
    # splitter omits it from infer.py and emits the load_model reminder.
    cells.append(nbformat.v4.new_code_cell(
        "model = LinearRegression()\n"
        "model.fit(X, y)\n"
    ))

    # Cell — signal_export: save_model (role: signal_export)
    cells.append(nbformat.v4.new_code_cell(
        "panel.save_model('spy-momentum-252d', model, metadata={\n"
        "    'framework':'sklearn',\n"
        "    'signature':'LinearRegression(y_next ~ rolling_sharpe_252)',\n"
        "    'features':['rs252'],\n"
        "    'lookback_days': 365,\n"
        "    'training_window':[str(rs.index[0]), str(rs.index[-1])],\n"
        "})\n"
    ))

    # Cell 5 — eval: predict + compute walk-forward Sharpe of the
    # predicted signal (role: eval — paired with print, not chart, to
    # keep the cell single-purpose)
    cells.append(nbformat.v4.new_code_cell(
        "preds = pd.Series(model.predict(X).flatten(), index=X.index, name='signal')\n"
        "signal_sharpe = (preds.mean() / preds.std()) * (252 ** 0.5) if preds.std() else 0.0\n"
        "print(f'in_sample_signal_sharpe={signal_sharpe:.3f}')\n"
    ))

    # Cell 6 — signal_export: export_signal (role: signal_export)
    cells.append(nbformat.v4.new_code_cell(
        "panel.export_signal(\n"
        "    'spy-momentum-252d',\n"
        "    rolling_sharpe_252,\n"
        "    metadata={\n"
        "        'frequency':'daily',\n"
        "        'universe':['SPY'],\n"
        "        'description':'252-day rolling Sharpe ratio on SPY daily returns',\n"
        "        'interpretation':'higher = stronger trailing risk-adjusted momentum',\n"
        "        'model_name':'spy-momentum-252d',\n"
        "    },\n"
        ")\n"
    ))

    # Cell 7 — summary
    cells.append(nbformat.v4.new_code_cell(
        "import json\n"
        "summary = {\n"
        "    'recipe_name':'spy-momentum-252d',\n"
        "    'metrics': {'rolling_sharpe_last': float(rolling_sharpe_252.iloc[-1]),\n"
        "                'n_observations': int(len(rolling_sharpe_252))}\n"
        "}\n"
        "print('FINAGENT_RUN_SUMMARY ' + json.dumps(summary))\n"
    ))

    nb.cells = cells
    return nb


# ── The actual test ───────────────────────────────────────────────────


def _execute_notebook_in_process(nb_path: Path) -> nbformat.NotebookNode:
    """Run the notebook with nbclient against the host's default
    Python kernel. Returns the executed notebook (with outputs)."""
    from nbclient import NotebookClient

    nb = nbformat.read(str(nb_path), as_version=4)
    # Prefer a kernel that matches the current Python so all our deps
    # (pandas / sklearn / panel) are visible.
    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=60,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, str(nb_path))
    return nb


def test_spy_momentum_notebook_to_signal_round_trip(isolated_outputs):
    """Build → execute → split → infer round-trip for the SPY 252-day
    rolling Sharpe momentum signal. The full happy path of the
    research-loop → dashboard pipeline (Phases 1-4)."""

    # 1. Synthetic SPY fixture on disk (notebook will read_csv it).
    spy_df = _synthetic_spy(n_years=5)
    spy_csv = isolated_outputs / "spy_synth.csv"
    spy_df.to_csv(spy_csv)

    # 2. Build + write the notebook.
    nb = _build_spy_momentum_notebook(spy_csv)
    nb_path = isolated_outputs / "spy_momentum_test.ipynb"
    nbformat.write(nb, str(nb_path))

    # 3. Execute in-process. The notebook calls into `panel`; the
    #    isolated_outputs fixture has already pointed FINAGENT_OUTPUTS_DIR
    #    + FINAGENT_DB at our temp dir, so all artefacts land there.
    _execute_notebook_in_process(nb_path)

    # 4. Verify on-disk artefacts (model + signal).
    model_pkl   = isolated_outputs / "models"  / "spy-momentum-252d" / "model.pkl"
    model_mf    = isolated_outputs / "models"  / "spy-momentum-252d" / "manifest.json"
    signal_pq   = isolated_outputs / "signals" / "spy-momentum-252d" / "series.parquet"
    signal_mf   = isolated_outputs / "signals" / "spy-momentum-252d" / "manifest.json"

    assert model_pkl.exists(),  f"model pkl missing: {model_pkl}"
    assert model_mf.exists(),   f"model manifest missing: {model_mf}"
    assert signal_pq.exists(),  f"signal parquet missing: {signal_pq}"
    assert signal_mf.exists(),  f"signal manifest missing: {signal_mf}"

    # 5. Verify model manifest contents.
    mf = json.loads(model_mf.read_text())
    assert mf["name"] == "spy-momentum-252d"
    assert mf["metadata"]["framework"] == "sklearn"
    assert mf["metadata"]["features"] == ["rs252"]
    assert mf["metadata"]["lookback_days"] == 365

    # 6. Verify signal manifest + parquet round-trip.
    smf = json.loads(signal_mf.read_text())
    assert smf["metadata"]["frequency"] == "daily"
    assert smf["metadata"]["universe"] == ["SPY"]
    # 5 yrs * 252 - 252 (rolling NaNs) ~ 1008
    assert smf["n_observations"] >= 750

    series_rt = pd.read_parquet(signal_pq)
    assert len(series_rt) == smf["n_observations"]
    assert "value" in series_rt.columns
    assert series_rt["value"].notna().all()

    # 7. Verify DB row in signals + signal_versions.
    db = isolated_outputs / "experiments.db"
    assert db.exists()
    conn = sqlite3.connect(str(db))
    sig = conn.execute(
        "SELECT name, frequency, n_observations, status, last_value FROM signals "
        "WHERE name = 'spy-momentum-252d'"
    ).fetchone()
    assert sig is not None, "signal row not in DB"
    assert sig[0] == "spy-momentum-252d"
    assert sig[1] == "daily"
    assert sig[2] >= 750
    assert sig[3] == "active"
    assert sig[4] is not None

    versions = conn.execute(
        "SELECT count(*) FROM signal_versions"
    ).fetchone()[0]
    assert versions == 1, f"expected 1 version row, got {versions}"

    # 8. Tag + split the executed notebook into train.py / infer.py.
    spec = importlib.util.spec_from_file_location("classifier", "finagent/cells/classifier.py")
    classifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(classifier)
    spec = importlib.util.spec_from_file_location("splitter", "finagent/cells/splitter.py")
    splitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(splitter)

    role_map = classifier.tag_notebook(nb_path)
    # Cell 0 = bootstrap (sys.path injection — classifies as 'other').
    # Cell 1 = imports.
    assert role_map["1"] == ["imports"]
    # Cell 5 (model.fit) must be tagged train (and ONLY train so the
    # splitter emits the load_model reminder in infer).
    assert role_map["5"] == ["train"]
    # Cell 6 (save_model) must be tagged signal_export.
    assert "signal_export" in role_map["6"]
    # Cell 8 (export_signal) must be tagged signal_export.
    assert "signal_export" in role_map["8"]

    out = splitter.split_notebook(nb_path, output_dir=isolated_outputs / "scripts")
    train_src = out["train"].read_text()
    infer_src = out["infer"].read_text()

    # train.py contains the model fit + both signal-export calls.
    assert "model.fit" in train_src
    assert "panel.save_model" in train_src
    assert "panel.export_signal" in train_src

    # infer.py omits the train cell (load via panel.load_model reminder)
    # but contains the data_load + preprocess + signal_export.
    assert "TRAIN cell omitted" in infer_src
    assert "rolling_sharpe_252" in infer_src
    assert "panel.export_signal" in infer_src

    # 9. Inference round-trip — load the saved model, run it on a fresh
    #    in-memory window, confirm finite output.
    import panel
    loaded_model = panel.load_model("spy-momentum-252d")
    fresh_X = pd.DataFrame({"rs252": [0.5, 1.2, -0.3]})
    preds = loaded_model.predict(fresh_X)
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds)), "loaded model produced non-finite predictions"


def test_spy_momentum_signal_listed_via_dashboard_helpers(isolated_outputs):
    """After the round-trip test runs, the signal must be listable
    through the same helpers ``app.py`` uses for the dashboard
    endpoints. We exercise the helpers directly (the FastAPI route is
    a one-line wrapper)."""
    # Run the round-trip first to populate the DB + disk.
    test_spy_momentum_notebook_to_signal_round_trip(isolated_outputs)

    # Bypass finagent.__init__ (it eagerly imports openai). Build a
    # minimal package shim that satisfies the relative imports inside
    # finagent.experiments, then load that module directly.
    import sys
    import types

    sys.modules.pop("finagent", None)
    pkg = types.ModuleType("finagent")
    pkg.__path__ = ["finagent"]
    sys.modules["finagent"] = pkg
    recipes_pkg = types.ModuleType("finagent.recipes")
    recipes_pkg.__path__ = ["finagent/recipes"]
    sys.modules["finagent.recipes"] = recipes_pkg

    plaus_spec = importlib.util.spec_from_file_location(
        "finagent.recipes.plausibility", "finagent/recipes/plausibility.py"
    )
    plaus_mod = importlib.util.module_from_spec(plaus_spec)
    # IMPORTANT: register in sys.modules BEFORE exec_module so the
    # @dataclass decorator inside finagent.experiments can resolve
    # cls.__module__ at class-creation time (Python 3.13 strictly
    # requires the module to be in sys.modules during dataclass init).
    sys.modules["finagent.recipes.plausibility"] = plaus_mod
    plaus_spec.loader.exec_module(plaus_mod)

    exp_spec = importlib.util.spec_from_file_location(
        "finagent.experiments", "finagent/experiments.py"
    )
    exp = importlib.util.module_from_spec(exp_spec)
    sys.modules["finagent.experiments"] = exp
    exp_spec.loader.exec_module(exp)

    sigs = exp.list_signals_db()
    names = [s.name for s in sigs]
    assert "spy-momentum-252d" in names

    one = exp.get_signal_db("spy-momentum-252d")
    assert one is not None
    assert one.frequency == "daily"
    assert one.universe == ["SPY"]
    assert one.n_observations >= 750
    assert one.status == "active"

    versions = exp.list_signal_versions_db(one.id)
    assert len(versions) >= 1

    # Status update round-trip.
    assert exp.update_signal_status_db("spy-momentum-252d", "paused") is True
    assert exp.get_signal_db("spy-momentum-252d").status == "paused"
    assert exp.update_signal_status_db("spy-momentum-252d", "active") is True
    assert exp.get_signal_db("spy-momentum-252d").status == "active"

    with pytest.raises(ValueError):
        exp.update_signal_status_db("spy-momentum-252d", "bogus")
