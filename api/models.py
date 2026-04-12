"""
api/models.py
~~~~~~~~~~~~~
Pydantic request and response models for the FastAPI layer.
"""

from __future__ import annotations

from typing import Optional, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class SearchDepth(str, Enum):
    basic = "basic"
    advanced = "advanced"


class Verdict(str, Enum):
    verified = "VERIFIED"
    mostly_verified = "MOSTLY_VERIFIED"
    uncertain = "UNCERTAIN"
    disputed = "DISPUTED"


class PipelineStatus(str, Enum):
    completed = "completed"
    partial = "partial"
    failed = "failed"


# ── Request Models ─────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Request body for POST /research"""
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Research question or topic.",
        examples=["What are the latest advances in quantum computing?"],
    )
    search_depth: SearchDepth = Field(
        default=SearchDepth.advanced,
        description="Tavily search depth. 'advanced' is slower but higher quality.",
    )
    use_cache: bool = Field(
        default=True,
        description="Return cached result if available (within TTL).",
    )
    notify_n8n: bool = Field(
        default=True,
        description="Fire n8n webhook on pipeline completion.",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What are the latest advances in quantum computing?",
            "search_depth": "advanced",
            "use_cache": True,
        }
    }}


class FeedbackRequest(BaseModel):
    """Request body for POST /research/{run_id}/feedback"""
    run_id: str = Field(..., description="LangSmith run ID.")
    key: str = Field(
        ...,
        description="Feedback dimension (e.g., 'accuracy', 'helpfulness').",
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1.")
    comment: Optional[str] = Field(None, description="Optional text comment.")


# ── Response Models ────────────────────────────────────────────────────────────

class SourceItem(BaseModel):
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None


class EntitySet(BaseModel):
    people: list[str] = []
    organizations: list[str] = []
    dates: list[str] = []
    statistics: list[str] = []


class SummaryResponse(BaseModel):
    summary: str = ""
    key_points: list[str] = []
    key_entities: Optional[EntitySet] = None
    sources_used: list[int] = []
    confidence: float = 0.0


class ClaimResponse(BaseModel):
    claim: str = ""
    confidence: float = 0.0
    status: str = "PARTIAL"
    supporting_sources: list[int] = []
    note: str = ""


class FactCheckResponse(BaseModel):
    overall_confidence: float = 0.0
    verdict: Verdict = Verdict.uncertain
    verdict_emoji: str = "❓"
    claims: list[ClaimResponse] = []
    contradictions: list[str] = []
    unverified_claims: list[str] = []
    fact_check_summary: str = ""


class NodeTimings(BaseModel):
    search: Optional[float] = None
    summarize: Optional[float] = None
    fact_check: Optional[float] = None


class ResearchResponse(BaseModel):
    """Full response from the research pipeline."""
    query: str = ""
    status: PipelineStatus = PipelineStatus.completed
    started_at: str = ""
    completed_at: Optional[str] = None
    total_latency_ms: float = 0.0
    node_timings_ms: Optional[NodeTimings] = None
    errors: list[str] = []

    # Core outputs
    direct_answer: Optional[str] = None
    sources: list[SourceItem] = []
    summary: Optional[SummaryResponse] = None
    fact_check: Optional[FactCheckResponse] = None

    # Meta
    overall_confidence: float = 0.0
    from_cache: bool = False

    model_config = {"populate_by_name": True, "extra": "ignore"}


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None