from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, TextIO


def setup() -> None:
    """
    Logging Setup:
    - pretty console locally
    - JSON lines in prod (APP_ENV=prod)
    - level via LOG_LEVEL (default INFO)
    """
    level = os.getenv("LOG LEVEL", "INFO").upper()
    prod = os.getenv("APP_ENV") == "prod"

    # Base Config
    logging.basicConfig(level=level, stream=sys.stdout)

    if prod:

        class JsonHandler(logging.StreamHandler[TextIO]):
            def emit(self, record: logging.LogRecord) -> None:
                msg: dict[str, Any] = {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    msg["exc_info"] = logging.Formatter().formatException(record.exc_info)
                sys.stdout.write(json.dumps(msg) + "\n")

        root = logging.getLogger()
        handler: logging.Handler = JsonHandler()
        root.handlers = [handler]
    else:
        fmt = "[%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(fmt)
        for h in logging.getLogger().handlers:
            h.setFormatter(formatter)
