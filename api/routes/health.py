"""
api/routes/health.py
~~~~~~~~~~~~~~~~~~~~
Health check endpoints.
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter
from api.models import HealthResponse

router = APIRouter(tags=["Health"])

_start_time = time.time()
APP_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    """Returns service health status and dependency connectivity."""
    services = {
        "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing_key",
        "tavily": "configured" if os.getenv("TAVILY_API_KEY") else "missing_key",
        "langsmith": "enabled" if os.getenv("LANGCHAIN_TRACING_V2") == "true" else "disabled",
        "redis": "configured" if os.getenv("REDIS_URL") else "disabled",
        "n8n": "configured" if os.getenv("N8N_WEBHOOK_URL") else "disabled",
    }

    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        services=services,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/", include_in_schema=False)
async def root():
    return {"message": "Multi-Agent Research Assistant API", "docs": "/docs"}