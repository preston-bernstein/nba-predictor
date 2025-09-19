"""API routers."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from . import deps
from .errors import unprocessable  # tiny helper -> HTTP 422
from .schemas import (
    ErrorResponse,
    HealthResponse,
    PredictQuery,
    PredictResponse,
    TeamListResponse,
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
def predict(q: PredictQuery = Depends()) -> PredictResponse:  # noqa: B008
    """
    Normalize/validate teams inside deps.matchup_features, compute deltas,
    enforce deterministic feature order, and surface domain/history issues as 422.
    """
    try:
        # Always request the full mapping, not a 2-tuple
        deltas = matchup_features(q.home, q.away, date=q.date, return_dict=True)
    except ValueError as e:
        # Domain/history/type errors surface as 422, not 500
        raise unprocessable(str(e)) from e

    model = load_model()

    # Preferred order if the model exposes it (recommended to persist at train time)
    feature_columns = getattr(model, "feature_columns_", None)

    if feature_columns:
        order = list(feature_columns)
        missing = [k for k in order if k not in deltas]
        if missing:
            raise unprocessable(
                f"Insufficient feature history for this date: missing {missing}; "
                f"model expects {order}."
            )
    else:
        # Fallback to canonical order by prefix; cap at model.n_features_in_ if available
        canonical = ["delta_off", "delta_def", "delta_rest", "delta_elo"]
        available = [k for k in canonical if k in deltas]
        n_expected = getattr(model, "n_features_in_", None)
        order = available if n_expected is None else available[: int(n_expected)]
        missing = [k for k in order if k not in deltas]
        if missing:
            raise unprocessable(
                f"Insufficient feature history for this date: missing {missing}; "
                f"model expects {order}."
            )

    X = np.array([[float(deltas[k]) for k in order]], dtype=float)
    prob = float(model.predict_proba(X)[:, 1][0])

    # q.home/q.away may be raw inputs; deps handled normalization for features.
    # For response, echo the request values as-is or switch to canonical codes if you prefer.
    return PredictResponse(
        home_team=q.home,
        away_team=q.away,
        as_of=q.date,
        features={k: float(deltas[k]) for k in order},
        prob_home_win=prob,
    )
