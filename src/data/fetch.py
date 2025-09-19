from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

from src import config
from src.service.normalizer import TeamNormalizeError, normalize_team

from .br_client import fetch_season_html
from .br_parse import parse_games

OUT_DIR = config.DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# --- Local fallbacks to avoid ingest failures if normalizer isn’t picked up ---
# (Uppercased, single-spaced keys.)
_FALLBACK_BR_FULL = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BRK",
    "CHARLOTTE HORNETS": "CHO",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHO",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}

# If we see a full label that ends with one of these nicknames, map to code.
# (Longest first so “TRAIL BLAZERS” wins over “BLAZERS”.)
_SUFFIX_NICK_TO_CODE: list[tuple[str, str]] = [
    ("TRAIL BLAZERS", "POR"),
    ("TIMBERWOLVES", "MIN"),
    ("GRIZZLIES", "MEM"),
    ("CAVALIERS", "CLE"),
    ("MAVERICKS", "DAL"),
    ("NUGGETS", "DEN"),
    ("PISTONS", "DET"),
    ("WARRIORS", "GSW"),
    ("ROCKETS", "HOU"),
    ("PACERS", "IND"),
    ("CLIPPERS", "LAC"),
    ("LAKERS", "LAL"),
    ("BUCKS", "MIL"),
    ("PELICANS", "NOP"),
    ("KNICKS", "NYK"),
    ("THUNDER", "OKC"),
    ("MAGIC", "ORL"),
    ("76ERS", "PHI"),
    ("SUNS", "PHO"),
    ("KINGS", "SAC"),
    ("SPURS", "SAS"),
    ("RAPTORS", "TOR"),
    ("JAZZ", "UTA"),
    ("WIZARDS", "WAS"),
    ("HAWKS", "ATL"),
    ("CELTICS", "BOS"),
    ("NETS", "BRK"),
    ("HORNETS", "CHO"),
    ("BULLS", "CHI"),
    ("HEAT", "MIA"),
]

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_PUNC_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def _clean_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(" " if unicodedata.category(ch).startswith("Z") else ch for ch in s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _PUNC_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s.upper()


def _norm_team_label(raw: str) -> str:
    """
    Strongly-normalize a team label to a BR code.

    Strategy:
      1) exact BR full-name fallback
      2) main normalizer.normalize_team
      3) nickname suffix fallback ("Portland Trail Blazers", etc.)
    """
    s = _clean_key(str(raw))

    if s in _FALLBACK_BR_FULL:
        return _FALLBACK_BR_FULL[s]

    try:
        return normalize_team(s)
    except TeamNormalizeError as e:
        for nick, code in _SUFFIX_NICK_TO_CODE:
            if s.endswith(" " + nick) or s == nick:
                return code

        hexes = " ".join(f"{ord(ch):04X}" for ch in s)
        msg = f"Unknown team in scraped data: {raw!r} (codepoints: {hexes})"
        raise ValueError(msg) from e


def _normalize_teams_inplace(df: pd.DataFrame) -> None:
    df["home_team"] = df["home_team"].map(_norm_team_label)
    df["away_team"] = df["away_team"].map(_norm_team_label)


def _drop_dupe_games(df: pd.DataFrame) -> pd.DataFrame:
    subset: Iterable[str] | None
    if "game_id" in df.columns:
        subset = ["game_id"]
    elif "game_key" in df.columns:
        subset = ["game_key"]
    else:
        subset = None
    if subset:
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    return df


def _post_parse_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df = _drop_dupe_games(df)
    _normalize_teams_inplace(df)
    return df


def years_span(start: int, end: int) -> list[int]:
    if end < start:
        raise SystemExit("--to must be >= --from")
    return list(range(start, end + 1))


def fetch_seasons(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        logging.info("fetching season %d", yr)
        html = fetch_season_html(yr)
        frames.append(parse_games(html))
    games = pd.concat(frames, ignore_index=True)
    return _post_parse_cleanup(games)


def main(seasons: list[int]) -> None:
    games = fetch_seasons(seasons)
    out_csv = OUT_DIR / "games.csv"
    games.to_csv(out_csv, index=False)
    logging.info("saved %d games -> %s", len(games), out_csv)


def _cli() -> None: # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from",
        dest="from_year",
        type=int,
        default=2019,
        help="Start season end-year (e.g., 2019 means 2018–19).",
    )
    ap.add_argument(
        "--to",
        dest="to_year",
        type=int,
        default=2025,
        help="End season end-year (inclusive).",
    )
    ap.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Explicit list of season end-years (overrides --from/--to).",
    )
    args = ap.parse_args()

    # args.seasons is Optional[List[int]]
    seasons: list[int] = (
        list(args.seasons) if args.seasons else years_span(args.from_year, args.to_year)
    )
    main(seasons)


if __name__ == "__main__": # pragma: no cover
    _cli()
