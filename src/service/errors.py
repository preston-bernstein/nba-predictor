from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

__all__ = ["unprocessable", "bad_request", "not_found"]


def unprocessable(detail: str) -> HTTPException:
    """422 Unprocessable Entity (validation/domain/history errors)."""
    return HTTPException(status_code=422, detail=detail)


def bad_request(detail: str) -> HTTPException:
    """400 Bad Request."""
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def not_found(detail: str) -> HTTPException:
    """404 Not Found."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def register_handlers(app: FastAPI) -> None:
    """global exception handlers"""

    @app.exception_handler(ValueError)
    async def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        # Domain/validation errors thrown inside code
        logger.warning("ValueError handled as 422: %s", exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
        # for dict lookups on team codes, etc
        logger.warning("KeyError handled as 400: %s", exc)
        return JSONResponse(status_code=400, content={"detail": f"Missing key: {exc}"})
