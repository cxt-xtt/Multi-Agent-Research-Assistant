"""
tests/test_workflows.py
~~~~~~~~~~~~~~~~~~~~~~~
Tests for the LangGraph pipeline, caching, and pipeline orchestration.
"""
 
from __future__ import annotations
 
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
 
from agents.search_agent import SearchOutput, SearchResult
from agents.summarizer_agent import SummaryOutput
from agents.fact_checker_agent import FactCheckOutput
from agents.base_agent import AgentResult
 
 
# ── Fixtures ──────────────────────────────────────────────────────────────────
 
@pytest.fixture
def mock_search_output():
    return SearchOutput(
        query="test query",
        results=[
            SearchResult(
                title="Test Article",
                url="https://test.com/article",
                content="Some test content about the topic.",
                score=0.88,
            )
        ],
        answer="Direct answer here.",
        total_found=1,
    )
 
 
@pytest.fixture
def mock_summary_output():
    return SummaryOutput(
        summary="A comprehensive summary of the topic.",
        key_points=["Point A", "Point B", "Point C"],
        key_entities={"organizations": ["TestOrg"]},
        sources_used=[1],
        confidence=0.85,
    )
 
 
@pytest.fixture
def mock_fact_check_output():
    return FactCheckOutput(
        overall_confidence=0.82,
        verdict="VERIFIED",
        claims=[],
        contradictions=[],
        unverified_claims=[],
        fact_check_summary="Claims are well-supported.",
    )
 
 
# ── LangGraph Node Tests ───────────────────────────────────────────────────────
 
class TestGraphNodes:
 
    def test_search_node_success(self, mock_search_output):
        from workflows.graph import make_search_node
 
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            agent_name="SearchAgent",
            output=mock_search_output,
        )
 
        node = make_search_node(mock_agent)
        state = {
            "query": "test",
            "search_depth": "advanced",
            "errors": [],
            "node_timings": {},
        }
        result = node(state)
 
        assert result["search_output"] is mock_search_output
        assert result["errors"] == []
        assert "search" in result["node_timings"]
 
    def test_search_node_failure_adds_error(self):
        from workflows.graph import make_search_node
 
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            agent_name="SearchAgent",
            output=None,
            error="API timeout",
        )
 
        node = make_search_node(mock_agent)
        state = {
            "query": "test",
            "search_depth": "basic",
            "errors": [],
            "node_timings": {},
        }
        result = node(state)
 
        assert result["search_output"] is None
        assert len(result["errors"]) == 1
        assert "API timeout" in result["errors"][0]
 
    def test_summarize_node_success(self, mock_search_output, mock_summary_output):
        from workflows.graph import make_summarize_node
 
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            agent_name="SummarizerAgent",
            output=mock_summary_output,
        )
 
        node = make_summarize_node(mock_agent)
        state = {
            "query": "test",
            "search_output": mock_search_output,
            "errors": [],
            "node_timings": {},
        }
        result = node(state)
 
        assert result["summary_output"] is mock_summary_output
        assert "summarize" in result["node_timings"]
 
    def test_fact_check_node_success(self, mock_search_output, mock_summary_output, mock_fact_check_output):
        from workflows.graph import make_fact_check_node
 
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            agent_name="FactCheckerAgent",
            output=mock_fact_check_output,
        )
 
        node = make_fact_check_node(mock_agent)
        state = {
            "query": "test",
            "search_output": mock_search_output,
            "summary_output": mock_summary_output,
            "errors": [],
            "node_timings": {},
        }
        result = node(state)
 
        assert result["fact_check_output"] is mock_fact_check_output
 
    def test_report_node_assembles_full_report(
        self, mock_search_output, mock_summary_output, mock_fact_check_output
    ):
        from workflows.graph import build_report_node as report_node
 
        state = {
            "query": "test query",
            "search_output": mock_search_output,
            "summary_output": mock_summary_output,
            "fact_check_output": mock_fact_check_output,
            "errors": [],
            "started_at": "2024-01-15T10:00:00+00:00",
            "node_timings": {"search": 2000, "summarize": 4000, "fact_check": 3000},
        }
        result = report_node(state)
 
        assert "report" in result
        report = result["report"]
        assert report["query"] == "test query"
        assert report["status"] == "completed"
        assert len(report["sources"]) == 1
        assert report["summary"] is not None
        assert report["fact_check"] is not None
        assert report["overall_confidence"] > 0
 
    def test_report_node_partial_status_on_errors(self, mock_search_output):
        from workflows.graph import build_report_node as report_node
 
        state = {
            "query": "failing query",
            "search_output": mock_search_output,
            "summary_output": None,
            "fact_check_output": None,
            "errors": ["SummarizerAgent: timeout"],
            "started_at": "2024-01-15T10:00:00+00:00",
            "node_timings": {"search": 1500},
        }
        result = report_node(state)
        assert result["report"]["status"] == "partial"
 
 
# ── Conditional Edge Tests ─────────────────────────────────────────────────────
 
class TestConditionalEdges:
 
    def test_should_continue_after_search_with_results(self, mock_search_output):
        from workflows.graph import should_continue_after_search
        state = {"search_output": mock_search_output}
        assert should_continue_after_search(state) == "summarize"
 
    def test_should_skip_after_search_no_results(self):
        from workflows.graph import should_continue_after_search
        empty_output = SearchOutput(query="test", results=[], total_found=0)
        state = {"search_output": empty_output}
        assert should_continue_after_search(state) == "build_report"
 
    def test_should_skip_after_search_none_output(self):
        from workflows.graph import should_continue_after_search
        state = {"search_output": None}
        assert should_continue_after_search(state) == "build_report"
 
    def test_should_continue_to_fact_check_with_summary(self, mock_summary_output):
        from workflows.graph import should_continue_after_summarize
        state = {"summary_output": mock_summary_output}
        assert should_continue_after_summarize(state) == "fact_check"
 
    def test_should_skip_fact_check_without_summary(self):
        from workflows.graph import should_continue_after_summarize
        state = {"summary_output": None}
        assert should_continue_after_summarize(state) == "build_report"
 
 
# ── Cache Tests ────────────────────────────────────────────────────────────────
 
class TestCacheClient:
 
    @pytest.mark.asyncio
    async def test_fallback_set_and_get(self):
        from utils.cache import CacheClient
        cache = CacheClient()
        cache._use_fallback = True
 
        await cache.set("test:key", {"value": 42})
        result = await cache.get("test:key")
        assert result == {"value": 42}
 
    @pytest.mark.asyncio
    async def test_fallback_returns_none_for_missing(self):
        from utils.cache import CacheClient
        cache = CacheClient()
        cache._use_fallback = True
 
        result = await cache.get("nonexistent:key")
        assert result is None
 
    @pytest.mark.asyncio
    async def test_fallback_delete(self):
        from utils.cache import CacheClient
        cache = CacheClient()
        cache._use_fallback = True
 
        await cache.set("to_delete", "value")
        await cache.delete("to_delete")
        result = await cache.get("to_delete")
        assert result is None
 
    @pytest.mark.asyncio
    async def test_fallback_exists(self):
        from utils.cache import CacheClient
        cache = CacheClient()
        cache._use_fallback = True
 
        await cache.set("exists_key", "hello")
        assert await cache.exists("exists_key") is True
        assert await cache.exists("missing_key") is False
 
    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        from workflows.pipeline import ResearchPipeline
        key1 = ResearchPipeline._cache_key("quantum computing", "advanced")
        key2 = ResearchPipeline._cache_key("quantum computing", "advanced")
        key3 = ResearchPipeline._cache_key("different query", "advanced")
 
        assert key1 == key2          # Same inputs → same key
        assert key1 != key3          # Different input → different key
        assert key1.startswith("research:")
 
 
# ── Pipeline Integration Tests ─────────────────────────────────────────────────
 
class TestResearchPipeline:
 
    @pytest.mark.asyncio
    async def test_pipeline_returns_cache_on_hit(self):
        from workflows.pipeline import ResearchPipeline
        from utils.cache import CacheClient
 
        cached_report = {"query": "cached", "status": "completed", "_from_cache": False}
 
        mock_cache = MagicMock(spec=CacheClient)
        mock_cache.get = AsyncMock(return_value=cached_report)
        mock_cache.set = AsyncMock()
 
        pipeline = ResearchPipeline.__new__(ResearchPipeline)
        pipeline.cache = mock_cache
        pipeline.enable_cache = True
        pipeline.enable_n8n = False
 
        # Patch the graph to ensure it's not called
        pipeline._graph = MagicMock()
 
        result = await pipeline.run("cached query", use_cache=True)
        assert result["_from_cache"] is True
        pipeline._graph.invoke.assert_not_called()
 
    @pytest.mark.asyncio
    async def test_pipeline_skips_cache_when_disabled(self):
        from workflows.pipeline import ResearchPipeline
        from utils.cache import CacheClient
 
        mock_cache = MagicMock(spec=CacheClient)
        mock_cache.get = AsyncMock(return_value={"cached": True})
        mock_cache.set = AsyncMock()
 
        mock_report = {
            "query": "test", "status": "completed",
            "total_latency_ms": 5000, "_from_cache": False,
        }
 
        pipeline = ResearchPipeline.__new__(ResearchPipeline)
        pipeline.cache = mock_cache
        pipeline.enable_cache = True
        pipeline.enable_n8n = False
        pipeline._graph = MagicMock()
        pipeline._graph.invoke = MagicMock(return_value={"report": mock_report})
 
        result = await pipeline.run("test query", use_cache=False)
        mock_cache.get.assert_not_called()