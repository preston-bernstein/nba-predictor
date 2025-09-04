from pathlib import Path
import argparse
import logging
import pandas as pd

from .br_client import fetch_season_html
from .br_parse import parse_games

OUT_DIR = Path("data_cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def main(seasons: list[int]):
    frames = []
    for yr in seasons:
        logging.info("fetching season %d", yr)
        html = fetch_season_html(yr)
        frames.append(parse_games(html))

    games = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE")
    out_csv = OUT_DIR / "games.csv"
    games.to_csv(out_csv, index=False)
    logging.info("saved %d games -> %s", len(games), out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2024, 2025],
        help="Season end years (e.g. 2024 for the 2023-24 season)."
    )
    args = parser.parse_args()
    main(args.seasons)
