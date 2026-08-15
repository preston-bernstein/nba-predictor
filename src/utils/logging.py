from __future__ import annotations

import logging
import os

from fleet_logging import configure_logging


def setup() -> None:
    """
    Logging setup: delegates to fleet_logging.configure_logging (home-infra
    CONVENTIONS.md §18 fleet JSON logging contract) instead of hand-rolling
    logging.basicConfig() plus a bespoke JSON handler here.

    Level is still controlled via LOG_LEVEL (default INFO; fixes this
    function's prior lookup key, which was "LOG LEVEL" with a space and so
    never actually matched the env var anyone would set).

    Note: fleet-logging always emits its canonical JSON line, on every call
    -- it has no separate "pretty console locally" mode. This module used to
    branch on APP_ENV=prod to print a human-readable format outside of prod;
    that branch is dropped rather than forced back in, since one canonical
    JSON line everywhere (matching the rest of the fleet) is fleet-logging's
    whole point, not an oversight.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    configure_logging("nba-predictor", level=level)
