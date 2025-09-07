# src/service/routes.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from . import deps
from .schemas import (
    HealthResponse,
    TeamListResponse,
    PredictQuery,
    FeatureDeltas,
    PredictResponse,
    ErrorResponse,
)

router = APIRouter()

# --- re-exports for test monkeypatching compatibility ---
# tests patch routes.load_games / routes.matchup_features / routes.load_model
load_games = deps.load_games
matchup_features = deps.matchup_features
load_model = deps.load_model
# --------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

@router.get("/teams", response_model=TeamListResponse)
def teams() -> TeamListResponse:
    df = load_games()
    all_teams = sorted(set(df["home_team"]).union(df["away_team"]))
    return TeamListResponse(teams=all_teams)

@router.get(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}},
)
@router.get("/predict", response_model=PredictResponse, responses={422: {"model": ErrorResponse}})
def predict(q: PredictQuery = Depends()) -> PredictResponse:
    try:
        # ✅ Always request the full mapping, not a 2-tuple
        deltas = matchup_features(q.home, q.away, date=q.date, return_dict=True)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    model = load_model()

    canonical = ["delta_off", "delta_def", "delta_rest", "delta_elo"]
    available = [k for k in canonical if k in deltas]

    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is None:
        order = available
    else:
        order = canonical[: int(n_expected)]  # model was trained in this order
        missing = [k for k in order if k not in deltas]
        if missing:
            # Return a 422 with a clear message rather than a 500
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient feature history for this date: missing {missing}; "
                       f"model expects {order}."
            )

    X = np.array([[float(deltas[k]) for k in order]], dtype=float)
    prob = float(model.predict_proba(X)[:, 1][0])

    return PredictResponse(
        home_team=q.home,
        away_team=q.away,
        as_of=q.date,
        features={k: float(deltas[k]) for k in order},
        prob_home_win=prob,
    )