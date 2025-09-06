# src/model/train.py
from pathlib import Path
import json, numpy as np, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

FEATS = Path("data_cache/features.csv")
ART = Path("artifacts")

# declare preferred features in order
PREF_FEATS = ["delta_off", "delta_def", "delta_rest", "delta_elo"]
MIN_FEATS = 2  # require at least N features to train

def main():
    df = pd.read_csv(FEATS, parse_dates=["GAME_DATE"])
    # sort by time; use last 25% as test
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)
    n_test = max(1, int(0.25 * n))
    train_df, test_df = df.iloc[:-n_test], df.iloc[-n_test:]

    used_feats = [c for c in PREF_FEATS if c in df.columns]
    if len(used_feats) < MIN_FEATS:
        raise ValueError(f"Not enough features. Found {used_feats}, need ≥{MIN_FEATS} among {PREF_FEATS}")

    X_tr = train_df[used_feats].values
    y_tr = train_df["home_win"].values
    X_te = test_df[used_feats].values
    y_te = test_df["home_win"].values

    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_te)) == 2:
        roc_auc = float(roc_auc_score(y_te, y_prob))
    else:
        roc_auc = float(accuracy_score(y_te, y_hat))  # safe fallback in [0,1]

    # Baselines
    base_home_rate = float((test_df["home_win"].mean()))
    # dumb baseline: always pick home
    base_home_acc = float(base_home_rate)

    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "features_used": used_feats,
        "accuracy": float(accuracy_score(y_te, y_hat)),
        "roc_auc": roc_auc,
        "baseline_home_rate": base_home_rate,
        "baseline_home_acc": base_home_acc,
    }

    ART.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ART / "model.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model ->", ART / "model.joblib")
    print("Metrics ->", metrics)

if __name__ == "__main__":
    main()
