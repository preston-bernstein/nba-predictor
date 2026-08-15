from __future__ import annotations

import argparse

from src import config
from src.model.trainer import Trainer


def main(models: list[str] | None = None) -> None:
    models = models or ["logreg"]  # keep default behavior
    trainer = Trainer(
        feats_path=config.FEATS,
        art_dir=config.ART_DIR,
        min_features=2,
        test_frac=0.25,
    )

    metrics = trainer.run(model_names=models)
    print("Saved models in ->", trainer.art_dir)
    print("Metrics ->", metrics)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],  # default keeps current behavior
        help="One or more: logreg rf",
    )
    args = ap.parse_args()
    main(args.models)
