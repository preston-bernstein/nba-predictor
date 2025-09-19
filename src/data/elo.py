from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EloConfig:
    base: float = 1500.0  # initial rating for unseen teams
    k: float = 20.0  # update magnitude
    home_adv: float = 50.0  # rating points for home advantage (used in EXPECTATION only)


def _expect_home(r_home: float, r_away: float, hadv: float) -> float:
    """
    Expected score for HOME given ratings and home advantage.
    E_home = 1 / (1 + 10^((R_away - (R_home + HAdv))/400))
    """
    return 1.0 / (1.0 + math.pow(10.0, (r_away - (r_home + hadv)) / 400.0))


def add_elo(games: pd.DataFrame, cfg: EloConfig | None = None) -> pd.DataFrame:
    """
    Compute pre-game Elo ratings per matchup (no leakage).
    Requires columns: GAME_DATE, home_team, home_score, away_team, away_score
    Returns a copy sorted by GAME_DATE with: home_elo_pre, away_elo_pre
    """
    # Avoid B008: instantiate inside the function
    cfg = EloConfig() if cfg is None else cfg

    required = {"GAME_DATE", "home_team", "home_score", "away_team", "away_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"add_elo: missing columns: {sorted(missing)}")

    # sort and copy to avoid mutating caller data
    g = games.sort_values("GAME_DATE").reset_index(drop=True).copy()

    ratings: dict[str, float] = {}
    home_pre, away_pre = [], []

    for _, row in g.iterrows():
        h = str(row["home_team"])
        a = str(row["away_team"])

        # Coerce scores to float, handle missing/invalid gracefully
        try:
            hs = float(row["home_score"])
            as_ = float(row["away_score"])
        except Exception as e:
            logger.debug("Elo parse failed at %s: %r", row.get("GAME_DATE"), e)
            raise ValueError(f"Non-numeric score at {row.get('GAME_DATE')}: {e}") from e

        if np.isnan(hs) or np.isnan(as_):
            raise ValueError(f"Non-numeric score at {row.get('GAME_DATE')}: NaN")

        rh = ratings.get(h, cfg.base)
        ra = ratings.get(a, cfg.base)

        # Store PRE-game ratings for features
        home_pre.append(rh)
        away_pre.append(ra)

        # Expected scores (home advantage only affects expectation)
        e_home = _expect_home(rh, ra, cfg.home_adv)

        # Actual outcomes; treat ties as 0.5 (BR usually has no ties, but be safe)
        if hs > as_:
            s_home = 1.0
        elif hs < as_:
            s_home = 0.0
        else:
            s_home = 0.5

        # Update post-game ratings
        rh_new = rh + cfg.k * (s_home - e_home)
        ra_new = ra + cfg.k * ((1.0 - s_home) - (1.0 - e_home))

        ratings[h] = rh_new
        ratings[a] = ra_new

    g["home_elo_pre"] = home_pre
    g["away_elo_pre"] = away_pre
    return g
