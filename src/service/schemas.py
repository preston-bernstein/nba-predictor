from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

__all__ = [
    "HealthResponse",
    "TeamListResponse",
    "PredictQuery",
    "FeatureDeltas",
    "PredictResponse",
    "ErrorResponse",
]

class HealthResponse(BaseModel):
    ok: bool = True

class TeamListResponse(BaseModel):
    teams: list[str]

class ErrorResponse(BaseModel):
    detail: str

class PredictQuery(BaseModel):
    home: str
    away: str
    date: Optional[str] = None  # ISO date string like "2024-11-01"

class FeatureDeltas(BaseModel):
    # make extras forbidden so we don’t silently accept wrong keys
    model_config = {"extra": "forbid"}

    delta_off: float
    delta_def: float
    # optional extras (present when available)
    delta_rest: Optional[float] = None
    delta_elo: Optional[float] = None

class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    as_of: Optional[str] = None
    features: FeatureDeltas
    prob_home_win: float = Field(ge=0.0, le=1.0)