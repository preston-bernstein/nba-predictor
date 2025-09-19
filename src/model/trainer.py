from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy.typing as npt
import pandas as pd

from src import config
from src.model import metrics as metrics_mod
from src.model.datasets import (
    baseline_stats,
    load_features,
    pick_features,
    time_split,
    to_xy,
)
from src.model.models import get_models
from src.model.select import persist_best_model, pick_best, write_metrics


class Trainer:
    """
    Orchestrates the ML training flow:
      - load features
      - choose usable feature columns
      - time-based split
      - train 1+ models
      - pick best (ROC-AUC, then accuracy)
      - persist stable model path + write metrics

    You can override feats_path/art_dir for tests or experiments;
    defaults come from src.config.
    """

    def __init__(
        self,
        feats_path: Path | None = None,
        art_dir: Path | None = None,
        pref_features: Iterable[str] = ("delta_off", "delta_def", "delta_rest", "delta_elo"),
        min_features: int = 2,
        test_frac: float = 0.25,
    ) -> None:
        self.feats_path = feats_path or config.FEATS
        self.art_dir = art_dir or config.ART_DIR
        self.pref_features = list(pref_features)
        self.min_features = int(min_features)
        self.test_frac = float(test_frac)

    def prepare_data(self) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
        df = load_features(self.feats_path)
        used = pick_features(df, self.pref_features, self.min_features)
        train_df, test_df = time_split(df, test_frac=self.test_frac)
        return df, used, train_df, test_df

    def train_models(
        self,
        model_names: Iterable[str],
        X_tr: npt.ArrayLike,
        y_tr: npt.ArrayLike,
        X_te: npt.ArrayLike,
        y_te: npt.ArrayLike,
    ) -> dict[str, dict[str, float]]:
        runs: dict[str, dict[str, float]] = {}
        for name, model in get_models(model_names):
            m = metrics_mod.fit_and_score(model, X_tr, y_tr, X_te, y_te)
            # ensure artifact dir exists before dumping
            self.art_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.art_dir / f"model-{name}.joblib")
            runs[name] = m
        return runs

    def run(self, model_names: Iterable[str] = ("logreg",)) -> dict[str, Any]:
        # 1) prep
        _, used_feats, train_df, test_df = self.prepare_data()
        X_tr, y_tr = to_xy(train_df, used_feats)
        X_te, y_te = to_xy(test_df, used_feats)

        # 2) train each requested model
        runs = self.train_models(model_names, X_tr, y_tr, X_te, y_te)

        # 3) choose best & persist stable path
        best_name, best_metrics = pick_best(runs)
        persist_best_model(self.art_dir, best_name)

        # 4) assemble metrics & write
        base = baseline_stats(test_df)
        combined = {
            # flat metrics for best model (keeps existing tests happy)
            "n_train": best_metrics["n_train"],
            "n_test": best_metrics["n_test"],
            "accuracy": best_metrics["accuracy"],
            "roc_auc": best_metrics["roc_auc"],
            # extras
            "features_used": used_feats,
            "best_model": best_name,
            "runs": runs,
            **base,
        }
        write_metrics(self.art_dir, combined)
        return combined
