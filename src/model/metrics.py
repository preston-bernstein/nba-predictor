"""Training metrics and scoring utilities."""


from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def fit_and_score(model, X_tr, y_tr, X_te, y_te) -> dict:
    """Train a model and compute evaluation metrics on the test set."""
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)

    # metrics
    if len(np.unique(y_te)) == 2:
        roc_auc = float(roc_auc_score(y_te, y_prob))
    else:
        # fallback when only one class in test set
        roc_auc = 0.5

    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, y_hat)),
        "roc_auc": roc_auc,
    }


def selection_key(metrics: dict) -> tuple[float, float]:
    """Primary: ROC AUC (NaN → -inf). Secondary: accuracy."""
    roc = metrics.get("roc_auc", float("nan"))
    roc_for_sel = roc if not np.isnan(roc) else float("-inf")
    acc = metrics.get("accuracy", float("-inf"))
    return (roc_for_sel, acc)
