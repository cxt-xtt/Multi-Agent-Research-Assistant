"""
tests/test_api.py
~~~~~~~~~~~~~~~~~
Integration tests for the FastAPI endpoints.
Uses httpx AsyncClient with app dependency overrides.
"""
 
from __future__ import annotations
 
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
 
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
 
 
# ── Fixtures ──────────────────────────────────────────────────────────────────
 
MOCK_REPORT = {
    "query": "test query",
    "status": "completed",
    "started_at": "2024-01-15T10:00:00+00:00",
    "completed_at": "2024-01-15T10:00:12+00:00",
    "total_latency_ms": 12000.0,
    "node_timings_ms": {"search": 2000, "summarize": 5000, "fact_check": 4000},
    "errors": [],
    "direct_answer": "Test direct answer",
    "sources": [
        {
            "title": "Test Source",
            "url": "https://example.com",
            "content": "Test content here.",
            "score": 0.92,
            "published_date": "2024-01-10",
        }
    ],
    "summary": {
        "summary": "This is a test summary paragraph.",
        "key_points": ["Key point 1", "Key point 2"],
        "key_entities": {"people": [], "organizations": ["MIT"]},
        "sources_used": [1],
        "confidence": 0.87,
    },
    "fact_check": {
        "overall_confidence": 0.85,
        "verdict": "VERIFIED",
        "verdict_emoji": "✅",
        "claims": [
            {
                "claim": "Test claim",
                "confidence": 0.9,
                "status": "SUPPORTED",
                "supporting_sources": [1],
                "note": "Confirmed by source 1",
            }
        ],
        "contradictions": [],
        "unverified_claims": [],
        "fact_check_summary": "All claims verified by sources.",
    },
    "overall_confidence": 0.86,
    "_from_cache": False,
}
 
 
@pytest.fixture
def mock_pipeline():
    """Return a mock pipeline that returns MOCK_REPORT."""
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=MOCK_REPORT)
    return pipeline
 
 
@pytest.fixture
def app(mock_pipeline):
    """Create the FastAPI app with pipeline dependency override."""
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("TAVILY_API_KEY", "test-key")
 
    from api.main import app as fastapi_app
    from api.routes.research import get_pipeline
 
    fastapi_app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()
 
 
@pytest.fixture
def client(app):
    return TestClient(app)
 
 
# ── Health Endpoint Tests ──────────────────────────────────────────────────────
 
class TestHealthEndpoint:
 
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
 
    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert "uptime_seconds" in data
 
    def test_health_status_healthy(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"
 
    def test_health_services_include_required_keys(self, client):
        resp = client.get("/health")
        services = resp.json()["services"]
        for key in ["openai", "tavily", "langsmith", "redis", "n8n"]:
            assert key in services
 
 
# ── Research Endpoint Tests ────────────────────────────────────────────────────
 
class TestResearchEndpoint:
 
    def test_post_research_success(self, client, mock_pipeline):
        resp = client.post(
            "/api/research/",
            json={"query": "What are the latest AI advances?"},
        )
        assert resp.status_code == 200
 
    def test_post_research_response_has_summary(self, client):
        resp = client.post(
            "/api/research/",
            json={"query": "Latest quantum computing research"},
        )
        data = resp.json()
        assert "summary" in data
        assert data["summary"]["summary"] != ""
 
    def test_post_research_response_has_fact_check(self, client):
        resp = client.post(
            "/api/research/",
            json={"query": "Latest quantum computing research"},
        )
        data = resp.json()
        assert "fact_check" in data
        assert data["fact_check"]["verdict"] == "VERIFIED"
 
    def test_post_research_response_has_sources(self, client):
        resp = client.post(
            "/api/research/",
            json={"query": "AI research"},
        )
        data = resp.json()
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
 
    def test_post_research_respects_search_depth(self, client, mock_pipeline):
        client.post(
            "/api/research/",
            json={"query": "test", "search_depth": "basic"},
        )
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs.get("search_depth") == "basic"
 
    def test_post_research_invalid_query_too_short(self, client):
        resp = client.post("/api/research/", json={"query": "ab"})
        assert resp.status_code == 422
 
    def test_post_research_missing_query(self, client):
        resp = client.post("/api/research/", json={})
        assert resp.status_code == 422
 
    def test_post_research_invalid_search_depth(self, client):
        resp = client.post(
            "/api/research/",
            json={"query": "valid query", "search_depth": "ultra"},
        )
        assert resp.status_code == 422
 
    def test_post_research_pipeline_error_returns_500(self, client, mock_pipeline):
        mock_pipeline.run = AsyncMock(side_effect=RuntimeError("Pipeline crashed"))
        resp = client.post("/api/research/", json={"query": "crash test"})
        assert resp.status_code == 500
 
    def test_post_research_strips_whitespace_from_query(self, client, mock_pipeline):
        client.post("/api/research/", json={"query": "  spaces around  "})
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs.get("query") == "spaces around"
 
    def test_post_research_default_use_cache_true(self, client, mock_pipeline):
        client.post("/api/research/", json={"query": "test query here"})
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs.get("use_cache") is True
 
    def test_post_research_cache_disabled(self, client, mock_pipeline):
        client.post(
            "/api/research/",
            json={"query": "test query here", "use_cache": False},
        )
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs.get("use_cache") is False
 
 
# ── Feedback Endpoint Tests ────────────────────────────────────────────────────
 
class TestFeedbackEndpoint:
 
    @patch("api.routes.research.log_feedback", return_value=True)
    def test_feedback_success(self, mock_feedback, client):
        resp = client.post(
            "/api/research/feedback",
            json={
                "run_id": "abc123",
                "key": "accuracy",
                "score": 0.9,
                "comment": "Very accurate!",
            },
        )
        assert resp.status_code == 200
 
    @patch("api.routes.research.log_feedback", return_value=False)
    def test_feedback_langsmith_unavailable(self, mock_feedback, client):
        resp = client.post(
            "/api/research/feedback",
            json={"run_id": "abc123", "key": "accuracy", "score": 0.5},
        )
        assert resp.status_code == 503
 
    def test_feedback_score_out_of_range(self, client):
        resp = client.post(
            "/api/research/feedback",
            json={"run_id": "abc123", "key": "accuracy", "score": 1.5},
        )
        assert resp.status_code == 422
 
    def test_feedback_negative_score_invalid(self, client):
        resp = client.post(
            "/api/research/feedback",
            json={"run_id": "abc123", "key": "accuracy", "score": -0.1},
        )
        assert resp.status_code == 422