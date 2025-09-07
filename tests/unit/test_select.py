import math
from pathlib import Path
import json
import pytest

from src.model.select import pick_best, persist_best_model, write_metrics

def test_pick_best_prefers_auc_then_accuracy():
    runs = {
        "a": {"accuracy": 0.60, "roc_auc": 0.55, "n_train": 1, "n_test": 1},
        "b": {"accuracy": 0.62, "roc_auc": 0.54, "n_train": 1, "n_test": 1},
        "c": {"accuracy": 0.59, "roc_auc": float("nan"), "n_train": 1, "n_test": 1},
    }
    best, met = pick_best(runs)
    assert best == "a"
    assert met["roc_auc"] == 0.55

def test_pick_best_raises_on_empty():
    with pytest.raises(ValueError):
        pick_best({})

def test_persist_best_model(tmp_path: Path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "model-abc.joblib").write_bytes(b"dummy")
    dst = persist_best_model(art, "abc")
    assert dst.exists()
    assert dst.name == "model.joblib"

def test_write_metrics(tmp_path: Path):
    art = tmp_path / "artifacts"
    path = write_metrics(art, {"x": 1})
    assert path.exists()
    assert json.loads(path.read_text())["x"] == 1
