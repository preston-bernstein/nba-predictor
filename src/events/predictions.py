"""Prediction for a game event, using the same code as GET /v1/predict."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from fastapi import HTTPException

from src import config
from src.service import routes
from src.service.schemas import PredictQuery

from .handler import PoisonMessage, Prediction


@lru_cache(maxsize=1)
def model_version() -> str:
    """Model file name plus the first 12 hex chars of its SHA-256."""
    path = config.MODEL
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.name}@{digest}"


def predict_game(game: dict[str, Any]) -> Prediction:
    """Call the service's `predict` with the team names, then with the team ids.

    A 422 from the service means the teams (or their history) are not usable.
    That is a poison message. Anything else is left to the caller as transient.
    """
    attempts = [
        (game["homeTeam"]["name"], game["awayTeam"]["name"]),
        (game["homeTeam"]["id"], game["awayTeam"]["id"]),
    ]
    first_detail = ""
    for home, away in attempts:
        try:
            result = routes.predict(PredictQuery(home=home, away=away, date=game["startDate"]))
        except HTTPException as exc:
            if exc.status_code != 422:
                raise
            first_detail = first_detail or str(exc.detail)
            continue
        return Prediction(result.prob_home_win, model_version())
    raise PoisonMessage(f"cannot predict: {first_detail}")
