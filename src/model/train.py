# src/model/train.py
from __future__ import annotations
from pathlib import Path
import argparse, json, shutil
import numpy as np
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

FEATS = Path("data_cache/features.csv")
ART = Path("artifacts")

# preferred features in order (we'll use whatever exists)
PREF_FEATS = ["delta_off", "delta_def", "delta_rest", "delta_elo"]
MIN_FEATS = 2  # require at least N features to train

def _time_split(df: pd.DataFrame, test_frac: float = 0.25):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)
    n_test = max(1, int(test_frac * n))
    train_df, test_df = df.iloc[:-n_test], df.iloc[-n_test:]
    return train_df, test_df

def _pick_features(df: pd.DataFrame) -> list[str]:
    used = [c for c in PREF_FEATS if c in df.columns]
    if len(used) < MIN_FEATS:
        raise ValueError(f"Not enough features. Found {used}, need ≥{MIN_FEATS} among {PREF_FEATS}")
    return used

def _fit_and_score(model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)

    # metrics
    # Use 0.5 (random) when ROC AUC is undefined (only one class in y_te)
    if len(np.unique(y_te)) == 2:
        roc_auc = float(roc_auc_score(y_te, y_prob))
    else:
        roc_auc = 0.5

    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, y_hat)),
        "roc_auc": float(roc_auc),
    }
    return metrics

def _get_models(requested: list[str]):
    all_defs = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    # keep order from requested; validate names
    out = []
    for name in requested:
        if name not in all_defs:
            raise ValueError(f"unknown model '{name}'. available: {sorted(all_defs)}")
        out.append((name, all_defs[name]))
    return out

def _selection_key(metrics: dict) -> tuple[float, float]:
    """Primary: ROC AUC (NaN -> -inf). Secondary: accuracy."""
    roc = metrics.get("roc_auc", float("nan"))
    roc_for_sel = roc if not np.isnan(roc) else float("-inf")
    acc = metrics.get("accuracy", float("-inf"))
    return (roc_for_sel, acc)

def main(models: list[str] | None = None):
    # default: keep current behavior (logreg only) so existing tests stay green
    if not models:
        models = ["logreg"]

    df = pd.read_csv(FEATS, parse_dates=["GAME_DATE"])
    used_feats = _pick_features(df)
    train_df, test_df = _time_split(df, test_frac=0.25)

    X_tr = train_df[used_feats].values
    y_tr = train_df["home_win"].values
    X_te = test_df[used_feats].values
    y_te = test_df["home_win"].values

    ART.mkdir(parents=True, exist_ok=True)

    # 1) train & record each requested model
    runs: dict[str, dict] = {}
    for name, model in _get_models(models):
        metrics = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        joblib.dump(model, ART / f"model-{name}.joblib")
        runs[name] = metrics

    # 2) select best by (roc_auc first, then accuracy)
    best_name, _ = max(runs.items(), key=lambda kv: _selection_key(kv[1]))
    shutil.copyfile(ART / f"model-{best_name}.joblib", ART / "model.joblib")

    # 3) assemble and write metrics
    base_home_rate = float(test_df["home_win"].mean())

    # flatten best run metrics to top-level to satisfy existing tests
    best_metrics = runs[best_name]
    combined_metrics = {
        # flat metrics for the selected/best model (tests look for these)
        "n_train": best_metrics["n_train"],
        "n_test": best_metrics["n_test"],
        "accuracy": best_metrics["accuracy"],
        "roc_auc": best_metrics["roc_auc"],

        # extras
        "features_used": used_feats,
        "best_model": best_name,
        "runs": runs,
        "baseline_home_rate": base_home_rate,
        "baseline_home_acc": base_home_rate,
    }

    (ART / "metrics.json").write_text(json.dumps(combined_metrics, indent=2))


    print("Saved models in ->", ART)
    print("Metrics ->", combined_metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],  # default keeps current behavior
        help="One or more: logreg rf",
    )
    args = ap.parse_args()
    main(args.models)
