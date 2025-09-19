from io import StringIO

import pandas as pd

REQUIRED_COLS = {"Date", "Visitor/Neutral", "Home/Neutral"}


def parse_games(html: str) -> pd.DataFrame:
    """Parse a Basketball-Reference season index page into a tidy games DataFrame."""
    tables = pd.read_html(StringIO(html), flavor="lxml")
    game_tables = [t for t in tables if REQUIRED_COLS.issubset(set(map(str, t.columns)))]
    if not game_tables:
        raise ValueError("No game tables found on page")

    df = pd.concat(game_tables, ignore_index=True)

    # Keep rows with an actual date and scores
    df = df[df["Date"].notna()].copy()
    df["GAME_DATE"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])

    df = df.rename(
        columns={
            "Visitor/Neutral": "away_team",
            "PTS": "away_score",
            "Home/Neutral": "home_team",
            "PTS.1": "home_score",
        }
    )

    # Filter out postponed/headers; enforce ints
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    out = df[["GAME_DATE", "home_team", "home_score", "away_team", "away_score"]].copy()
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    # Natural key -> call it game_id
    out["game_id"] = out.apply(
        lambda r: f"{r.GAME_DATE.date()}::{r.away_team}@{r.home_team}", axis=1
    )

    # De-dupe on game_id
    out = out.drop_duplicates(subset=["game_id"]).sort_values("GAME_DATE").reset_index(drop=True)

    return out
