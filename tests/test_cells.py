"""Tests for the cell-role classifier and train/infer splitter.

The classifier is a pure-AST heuristic — no LLM, deterministic, fast.
The splitter walks a tagged notebook and emits ``train.py`` + ``infer.py``
honouring the role-compatibility rules in the orchestrator prompt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pytest


# Load classifier + splitter via importlib so we don't drag in the full
# finagent.__init__ chain (which eagerly imports the OpenAI SDK).

def _load_module(rel_path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, rel_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def classifier():
    return _load_module("finagent/cells/classifier.py", "classifier")


@pytest.fixture(scope="module")
def splitter():
    # splitter imports nothing from finagent except itself, so it's safe.
    return _load_module("finagent/cells/splitter.py", "splitter")


# ── classifier — single-role detection ─────────────────────────────────


@pytest.mark.parametrize("source,expected", [
    # imports
    ("import pandas as pd\nimport numpy as np", {"imports"}),
    ("from sklearn.linear_model import LinearRegression", {"imports"}),
    # data_load
    ("df = yf.download('SPY', start='2020-01-01')", {"data_load"}),
    ("df = pd.read_csv('foo.csv')", {"data_load"}),
    ("df = panel.load_signal('spy-momentum-252d')", {"data_load"}),
    # train
    ("from sklearn.linear_model import LinearRegression\nm = LinearRegression()\nm.fit(X, y)",
     {"train"}),  # imports inside, but mostly fit
    ("model = OLS(y, X).fit()", {"train"}),
    # signal_export
    ("panel.export_signal('foo', s, metadata={})", {"signal_export"}),
    ("panel.save_model('foo', m, metadata={})", {"signal_export"}),
    # chart
    ("import matplotlib.pyplot as plt\nplt.plot(s)", {"chart"}),
    # summary
    ("print('FINAGENT_RUN_SUMMARY ' + json.dumps(s))", {"summary"}),
])
def test_classifier_single_role(classifier, source, expected):
    assert expected.issubset(classifier.classify_cell(source))


def test_classifier_blank_source(classifier):
    assert classifier.classify_cell("") == set()
    assert classifier.classify_cell("   \n  \n") == set()


def test_classifier_syntax_error_returns_other(classifier):
    assert classifier.classify_cell("def broken(:") == {"other"}


def test_classifier_unknown_helper_is_other(classifier):
    # Custom helper, no recognisable calls → other
    src = "def my_helper(x):\n    return x + 1\n\nresult = my_helper(5)"
    roles = classifier.classify_cell(src)
    assert "other" in roles


# ── classifier — needs_split rules ─────────────────────────────────────


def test_needs_split_train_plus_chart_is_bad(classifier):
    src = "m.fit(X, y)\nimport matplotlib.pyplot as plt\nplt.plot(m.predict(X))"
    assert classifier.needs_split(src) is True


def test_needs_split_data_load_plus_train_is_bad(classifier):
    src = "df = yf.download('SPY')\nm.fit(df[['x']], df['y'])"
    assert classifier.needs_split(src) is True


def test_needs_split_eval_plus_chart_is_ok(classifier):
    src = "preds = m.predict(X_test)\nplt.plot(preds)"
    assert classifier.needs_split(src) is False


def test_needs_split_data_load_plus_preprocess_is_ok(classifier):
    src = "df = yf.download('SPY')\ndf = df.dropna()"
    assert classifier.needs_split(src) is False


def test_needs_split_single_role_is_ok(classifier):
    assert classifier.needs_split("import pandas as pd") is False
    assert classifier.needs_split("panel.export_signal('x', s, metadata={})") is False


# ── tag_notebook idempotency + metadata stamping ───────────────────────


def _build_notebook(cells, path: Path):
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(c) if isinstance(c, str) else c
        for c in cells
    ]
    nbformat.write(nb, str(path))
    return path


def test_tag_notebook_stamps_roles_in_metadata(classifier, tmp_path):
    nb_path = _build_notebook(
        [
            "import pandas as pd",
            "df = yf.download('SPY')",
            "m = OLS(df['y'], df[['x']]).fit()",
            "panel.export_signal('foo', s, metadata={})",
        ],
        tmp_path / "test.ipynb",
    )
    out = classifier.tag_notebook(nb_path)
    assert "0" in out and "imports" in out["0"]
    assert "1" in out and "data_load" in out["1"]
    assert "3" in out and "signal_export" in out["3"]

    # Re-read and confirm metadata persisted.
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        roles = (cell.metadata.get("finagent") or {}).get("roles") or []
        assert isinstance(roles, list)


def test_tag_notebook_is_idempotent(classifier, tmp_path):
    nb_path = _build_notebook(["import pandas as pd"], tmp_path / "idem.ipynb")
    a = classifier.tag_notebook(nb_path)
    b = classifier.tag_notebook(nb_path)
    assert a == b


# ── splitter — train.py + infer.py emission ────────────────────────────


def test_split_emits_two_scripts(classifier, splitter, tmp_path):
    cells = [
        "import pandas as pd\nimport panel",                            # imports
        "df = yf.download('SPY', start='2020-01-01')",                  # data_load
        "df = df.dropna()",                                             # preprocess
        "model = OLS(df['y'], df[['x']]).fit()",                        # train
        "panel.save_model('m', model, metadata={})",                    # signal_export
        "preds = model.predict(df[['x']])",                             # eval
        "import matplotlib.pyplot as plt\nplt.plot(preds)",             # chart
        "panel.export_signal('foo', preds, metadata={'frequency':'daily'})",  # signal_export
        "print('FINAGENT_RUN_SUMMARY ' + json.dumps({'sharpe': 1.0}))", # summary
    ]
    nb_path = _build_notebook(cells, tmp_path / "split.ipynb")
    classifier.tag_notebook(nb_path)
    out = splitter.split_notebook(nb_path, output_dir=tmp_path / "out")

    assert out["train"].exists()
    assert out["infer"].exists()
    train_src = out["train"].read_text()
    infer_src = out["infer"].read_text()

    # train.py contains the train cell + signal_export + summary
    assert "OLS(df['y']" in train_src
    assert "panel.export_signal" in train_src
    assert "FINAGENT_RUN_SUMMARY" in train_src

    # infer.py omits the train cell (it's marked for omission)
    # but does include data_load + preprocess + eval + signal_export + summary
    assert "yf.download" in infer_src
    assert "df.dropna" in infer_src
    assert "model.predict" in infer_src
    assert "panel.export_signal" in infer_src
    # Charts excluded entirely from infer (still in train? No — chart isn't a TRAIN_ROLE)
    assert "plt.plot" not in train_src
    assert "plt.plot" not in infer_src


def test_split_emits_load_model_reminder_for_train_only_cell(classifier, splitter, tmp_path):
    cells = [
        "import pandas as pd",
        "df = pd.DataFrame({'x':[1,2,3],'y':[4,5,6]})",
        # Pure train cell with no other roles — splitter should emit a reminder.
        "model = OLS(df['y'], df[['x']]).fit()",
    ]
    nb_path = _build_notebook(cells, tmp_path / "trainonly.ipynb")
    classifier.tag_notebook(nb_path)
    out = splitter.split_notebook(nb_path)
    infer_src = out["infer"].read_text()
    assert "TRAIN cell omitted" in infer_src
    assert "load via panel.load_model" in infer_src
