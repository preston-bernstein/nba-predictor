from __future__ import annotations

import argparse
import sys

from src import config
from src.model.trainer import Trainer, TrainSettings


def main(models: list[str] | None = None) -> None:
    models = models or ["logreg"]  # keep default behavior
    trainer = Trainer(
        feats_path=config.FEATS,
        art_dir=config.ART_DIR,
        settings=TrainSettings(min_features=2, test_frac=0.25),
    )

    metrics = trainer.run(model_names=models)
    sys.stdout.write(f"Saved models in -> {trainer.art_dir}\n")
    sys.stdout.write(f"Metrics -> {metrics}\n")


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],  # default keeps current behavior
        help="One or more: logreg rf",
    )
    args = ap.parse_args(argv)
    main(args.models)


if __name__ == "__main__":
    cli()
