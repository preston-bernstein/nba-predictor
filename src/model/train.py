from pathlib import Path
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

FEATS = Path("data_cache/features.csv")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(FEATS, parse_dates=["GAME_DATE"])

    X = df[["delta_off", "delta_def"]].values
    y = df["home_win"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # handle edge case: test set has only one class
    if len(np.unique(y_test)) == 2:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    else:
        roc_auc = float("nan")

    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": roc_auc,
    }

    ART.mkdir(parents=True, exist_ok=True)  # ensure dir exists
    joblib.dump(model, ART / "model.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Model saved to", ART / "model.joblib")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
