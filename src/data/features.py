import logging

import pandas as pd

from src import config

from .transform import build_features_df  # <- the pure transformer

IN_PATH = config.GAMES
OUT_PATH = config.FEATS


def build_features() -> None:
    games = pd.read_csv(IN_PATH, parse_dates=["GAME_DATE"])
    feats = build_features_df(games)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_PATH, index=False)
    logging.info("saved %d rows -> %s", len(feats), OUT_PATH)


def _main() -> None:
    build_features()


if __name__ == "__main__":  # pragma: no cover
    _main()
