"""Training metrics and scoring utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping  # UP035
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, roc_auc_score


class ProbClassifier(Protocol):
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Any: ...
    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]: ...


def fit_and_score(
    model: ProbClassifier,
    X_tr: npt.ArrayLike,
    y_tr: npt.ArrayLike,
    X_te: npt.ArrayLike,
    y_te: npt.ArrayLike,
) -> dict[str, float | int]:
    """Train a model and compute evaluation metrics on the test set."""
    # Train
    model.fit(X_tr, y_tr)

    # Predict probabilities
    proba: npt.NDArray[np.float64] = np.asarray(model.predict_proba(X_te), dtype=np.float64)
    y_prob: npt.NDArray[np.float64] = proba[:, 1]

    # Class labels from probability threshold
    y_hat: npt.NDArray[np.int_] = (y_prob >= 0.5).astype(np.int_)

    # Ensure Sized for len()
    y_tr_arr = np.asarray(y_tr)
    y_te_arr = np.asarray(y_te)

    # Metrics
    roc_auc = float(roc_auc_score(y_te_arr, y_prob)) if len(np.unique(y_te_arr)) == 2 else 0.5
    acc = float(accuracy_score(y_te_arr, y_hat))

    return {
        "n_train": int(len(y_tr_arr)),
        "n_test": int(len(y_te_arr)),
        "accuracy": acc,
        "roc_auc": roc_auc,
    }


def selection_key(metrics: Mapping[str, float | int]) -> tuple[float, float]:
    """Primary: ROC AUC (NaN → -inf). Secondary: accuracy."""
    roc = float(metrics.get("roc_auc", float("nan")))  # may be absent; coerce to float
    roc_for_sel = roc if not math.isnan(roc) else float("-inf")
    acc = float(metrics.get("accuracy", float("-inf")))
    return (roc_for_sel, acc)
