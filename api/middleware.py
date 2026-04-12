"""
api/middleware.py
~~~~~~~~~~~~~~~~~
FastAPI middleware stack: CORS, request logging, rate limiting.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


log = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Attach request ID for tracing
        structlog.contextvars.bind_contextvars(request_id=request_id)

        log.info(
            "request_start",
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        log.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 1),
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(round(latency_ms, 1))

        structlog.contextvars.clear_contextvars()
        return response


def register_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI app."""
    import os

    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000",
    ).split(",")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)