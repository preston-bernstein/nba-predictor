"""setup() must work when the private fleet_logging package is not installed."""

from __future__ import annotations

import builtins
import importlib
import logging
import sys

import pytest


def test_setup_falls_back_to_stdlib_when_fleet_logging_missing(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "fleet_logging" or name.startswith("fleet_logging."):
            raise ModuleNotFoundError("No module named 'fleet_logging'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "fleet_logging", raising=False)
    monkeypatch.delitem(sys.modules, "src.utils.logging", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    try:
        mod = importlib.import_module("src.utils.logging")
    except ImportError:
        pytest.fail("src.utils.logging must import when the optional logging package is absent")
    mod.setup()

    assert logging.getLogger().level == logging.WARNING


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    root.setLevel(level)
    root.handlers[:] = handlers
