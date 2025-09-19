from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _determinism():
    """Make tests reproducible by fixing hash seed and numpy RNG."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        import numpy as np

        np.random.seed(0)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_fs(tmp_path_factory):
    """Point config paths to a temporary directory so tests never touch real data/artifacts."""
    base = tmp_path_factory.mktemp("run")
    os.environ["NBA_DATA_DIR"] = str(base / "data_cache")
    os.environ["NBA_ART_DIR"] = str(base / "artifacts")
    (base / "data_cache").mkdir(parents=True, exist_ok=True)
    (base / "artifacts").mkdir(parents=True, exist_ok=True)
    yield
