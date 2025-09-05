from fastapi import APIRouter, HTTPException
import numpy as np
from .core import load_games, load_model, matchup_features

router = APIRouter()

@router.get("/health")
def health(): return {"ok": True}

@router.get("/teams")
def teams():
    df = load_games()
    return sorted(set(df["home_team"]).union(df["away_team"]))

@router.get("/predict")
def predict(home: str, away: str, date: str | None = None):
    try:
        d_off, d_def = matchup_features(home, away, date=date)  # strict
    except ValueError as e:
        # convert domain errors to 422 for client
        raise HTTPException(status_code=422, detail=str(e))
    
    model = load_model()
    X = np.array([[d_off, d_def]], dtype=float)
    p = float(model.predict_proba(X)[:, 1][0])
    return {"home_team": home, "away_team": away, "as_of": date,
            "features": {"delta_off": d_off, "delta_def": d_def},
            "prob_home_win": p}
