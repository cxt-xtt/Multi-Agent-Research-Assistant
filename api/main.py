"""
api/main.py
~~~~~~~~~~~
FastAPI application entrypoint for the Multi-Agent Research Assistant.

Mounts:
  - /research  — Research pipeline endpoints
  - /health    — Health check
  - /          — Static frontend dashboard
  - /docs      — Swagger UI (auto-generated)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.health import router as health_router
from api.middleware import register_middleware
from utils.logger import configure_logging

configure_logging(os.getenv("LOG_LEVEL", "INFO"))

from api.routes.research import router as research_router

import structlog
log = structlog.get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hooks."""
    log.info(
        "app_startup",
        version="1.0.0",
        debug=os.getenv("DEBUG", "false"),
        langsmith=os.getenv("LANGCHAIN_TRACING_V2", "false"),
    )
    yield
    log.info("app_shutdown")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description=(
        "A production-grade multi-agent research pipeline powered by LangGraph, "
        "CrewAI, and GPT-4o. Automates web search, summarization, and fact-checking "
        "in a coordinated agent workflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Research", "description": "Research pipeline endpoints."},
        {"name": "Health", "description": "Service health and status."},
    ],
)

# ── Middleware ─────────────────────────────────────────────────────────────────
register_middleware(app)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(research_router, prefix="/api")

# ── Static Frontend ────────────────────────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard():
        return FileResponse(os.path.join(_static_dir, "index.html"))


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )