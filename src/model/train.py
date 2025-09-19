from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.model.trainer import Trainer


def main(models: list[str] | None = None) -> None:
    models = models or ["logreg"]  # keep default behavior
    trainer = Trainer(
        feats_path=Path(config.FEATS),
        art_dir=Path(config.ART_DIR),
        pref_features=("delta_off", "delta_def", "delta_rest", "delta_elo"),
        min_features=2,
        test_frac=0.25,
    )

    metrics = trainer.run(model_names=models)
    print("Saved models in ->", trainer.art_dir)
    print("Metrics ->", metrics)


if __name__ == "__main__": # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],  # default keeps current behavior
        help="One or more: logreg rf",
    )
    args = ap.parse_args()
    main(args.models)
