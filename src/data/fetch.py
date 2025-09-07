from pathlib import Path
import argparse
import logging
import pandas as pd
from src import config

from .br_client import fetch_season_html
from .br_parse import parse_games

OUT_DIR = config.DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def years_span(start: int, end: int) -> list[int]:
    if end < start:
        raise SystemExit("--to must be >= --from")
    return list(range(start, end + 1))

def main(seasons: list[int]):
    frames = []
    for yr in seasons:
        logging.info("fetching season %d", yr)
        html = fetch_season_html(yr)
        frames.append(parse_games(html))

    games = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE")
    games = games.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    subset = ["game_id"] if "game_id" in games.columns else (["game_key"] if "game_key" in games.columns else None)
    if subset:
        games = games.drop_duplicates(subset=subset).reset_index(drop=True)

    out_csv = OUT_DIR / "games.csv"
    games.to_csv(out_csv, index=False)
    logging.info("saved %d games -> %s", len(games), out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_year", type=int, default=2019,
                    help="Start season end-year (e.g., 2019 means 2018–19).")
    ap.add_argument("--to", dest="to_year", type=int, default=2025,
                    help="End season end-year (inclusive).")
    ap.add_argument("--seasons", type=int, nargs="+",
                    help="Explicit list of season end-years (overrides --from/--to).")
    args = ap.parse_args()

    years = args.seasons if args.seasons else years_span(args.from_year, args.to_year)
    main(years)