from __future__ import annotations

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
    date: str | None = None  # ISO date string like "2024-11-01"


class FeatureDeltas(BaseModel):
    model_config = {"extra": "forbid"}
    delta_off: float
    delta_def: float
    delta_rest: float | None = None
    delta_elo: float | None = None


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    as_of: str | None = None
    features: FeatureDeltas
    prob_home_win: float = Field(ge=0.0, le=1.0)
