"""
api/routes/research.py
~~~~~~~~~~~~~~~~~~~~~~
FastAPI routes for the research pipeline endpoints.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Depends, status

from api.models import (
    ResearchRequest,
    ResearchResponse,
    FeedbackRequest,
    ErrorResponse,
)
from workflows.pipeline import ResearchPipeline
from utils.tracing import log_feedback

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])

_pipeline: ResearchPipeline | None = None


def get_pipeline() -> ResearchPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ResearchPipeline()
    return _pipeline


@router.post(
    "/",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a research query",
    description=(
        "Submit a research question to the multi-agent pipeline. "
        "Returns a structured research report with summary, sources, and fact-check."
    ),
    responses={
        200: {"description": "Research report successfully generated."},
        422: {"description": "Validation error."},
        500: {"model": ErrorResponse, "description": "Internal pipeline error."},
    },
)
async def run_research(
    request: ResearchRequest,
    pipeline: ResearchPipeline = Depends(get_pipeline),
) -> ResearchResponse:
    """Run the full multi-agent research pipeline for the given query."""
    log.info("research_request", query=request.query[:100], depth=request.search_depth)

    try:
        report = await pipeline.run(
            query=request.query,
            search_depth=request.search_depth.value,
            use_cache=request.use_cache,
            notify_n8n=request.notify_n8n,
        )
    except Exception as exc:
        log.error("research_endpoint_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(exc)}",
        )

    if not report:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline returned an empty report.",
        )

    # Rename _from_cache -> from_cache
    report["from_cache"] = report.pop("_from_cache", False)

    return ResearchResponse(**report)


@router.post(
    "/feedback",
    status_code=status.HTTP_200_OK,
    summary="Submit feedback for a research run",
    description="Log user feedback (accuracy, helpfulness) for a run to LangSmith.",
)
async def submit_feedback(request: FeedbackRequest) -> dict:
    """Log feedback for a pipeline run to LangSmith."""
    success = log_feedback(
        run_id=request.run_id,
        key=request.key,
        score=request.score,
        comment=request.comment,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangSmith feedback service is unavailable.",
        )
    return {"status": "ok", "message": "Feedback logged successfully."}