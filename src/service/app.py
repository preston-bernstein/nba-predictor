from fastapi import FastAPI

from ..utils.logging import setup as setup_logging
from .errors import register_handlers
from .routes import router


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="nba-predictor", version="0.1.0")
    app.include_router(router, prefix="/v1")
    register_handlers(app)
    return app


app = create_app()
