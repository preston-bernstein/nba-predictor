from __future__ import annotations

from mangum import Mangum

from .app import create_app

handler = Mangum(create_app())
