from __future__ import annotations

from fastapi import HTTPException, status

__all__ = ["unprocessable", "bad_request", "not_found"]


def unprocessable(detail: str) -> HTTPException:
    """422 Unprocessable Entity (validation/domain/history errors)."""
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def bad_request(detail: str) -> HTTPException:
    """400 Bad Request."""
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def not_found(detail: str) -> HTTPException:
    """404 Not Found."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
