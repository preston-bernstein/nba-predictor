import pytest

from src.model.models import REGISTRY, get_models


def test_get_models_happy_path():
    pairs = get_models(["logreg", "rf"])
    names = [n for n, _ in pairs]
    assert names == ["logreg", "rf"]
    # Instances are constructed
    assert all(hasattr(m, "fit") for _, m in pairs)


def test_get_models_unknown_raises():
    with pytest.raises(ValueError):
        get_models(["nope"])


def test_registry_has_expected_keys():
    assert "logreg" in REGISTRY
    assert "rf" in REGISTRY
