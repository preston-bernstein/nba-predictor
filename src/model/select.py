"""Model selection & persistence helpers."""


from __future__ import annotations
from pathlib import Path
import json
import shutil
from typing import Dict, Tuple

from .metrics import selection_key  # (roc_auc (NaN→-inf), accuracy)

def pick_best(runs: Dict[str, dict]) -> Tuple[str, dict]:
    """
    Given a mapping {model_name -> metrics}, return (best_name, best_metrics)
    using selection_key: ROC AUC first (higher is better, NaN treated as -inf),
    then accuracy.
    """
    if not runs:
        raise ValueError("runs is empty")
    best_name, best_metrics = max(runs.items(), key=lambda kv: selection_key(kv[1]))
    return best_name, best_metrics

def persist_best_model(art_dir: Path, best_name: str) -> Path:
    """
    Copy artifacts/model-{best_name}.joblib to artifacts/model.joblib and
    return the destination path.
    """
    src = art_dir / f"model-{best_name}.joblib"
    dst = art_dir / "model.joblib"
    if not src.exists():
        raise FileNotFoundError(f"missing trained model file: {src}")
    art_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst

def write_metrics(art_dir: Path, metrics: dict) -> Path:
    """
    Write metrics JSON to artifacts/metrics.json and return the path.
    """
    path = art_dir / "metrics.json"
    art_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
