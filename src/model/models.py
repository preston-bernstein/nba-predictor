"""Model factory/registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# A simple registry so it's easy to add/swap models later.
REGISTRY = {
    "logreg": lambda: LogisticRegression(max_iter=1000, n_jobs=None),
    "rf": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}


def get_models(requested: Iterable[str]) -> list[tuple[str, BaseEstimator]]:
    """
    Return (name, instance) pairs for the requested model identifiers.
    Raises ValueError on unknown names. Keeps requested order.
    """
    out: list[tuple[str, BaseEstimator]] = []
    for name in requested:
        factory: Callable[[], BaseEstimator] | None = REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"unknown model '{name}'. available: {sorted(REGISTRY)}")
        out.append((name, factory()))
    return out
