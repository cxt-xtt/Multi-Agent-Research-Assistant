"""
tests/test_agents.py
~~~~~~~~~~~~~~~~~~~~
Unit tests for Search, Summarizer, and FactChecker agents.
Uses mocking to avoid real API calls in CI.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from agents.base_agent import AgentResult
from agents.search_agent import SearchAgent, SearchOutput, SearchResult
from agents.summarizer_agent import SummarizerAgent, SummaryOutput
from agents.fact_checker_agent import FactCheckerAgent, FactCheckOutput, ClaimResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_search_result():
    return SearchResult(
        title="Quantum Computing Breakthrough",
        url="https://example.com/quantum",
        content="Researchers at MIT have achieved quantum supremacy with 1000 qubits.",
        score=0.92,
        published_date="2024-01-15",
    )


@pytest.fixture
def mock_search_output(mock_search_result):
    return SearchOutput(
        query="latest quantum computing advances",
        results=[mock_search_result],
        answer="Quantum computing has seen major breakthroughs in qubit counts.",
        total_found=1,
    )


@pytest.fixture
def mock_summary_output():
    return SummaryOutput(
        summary="Quantum computing has advanced significantly, with MIT achieving 1000-qubit systems.",
        key_points=[
            "MIT achieved 1000-qubit quantum supremacy",
            "Error correction has improved by 40%",
        ],
        key_entities={
            "organizations": ["MIT"],
            "statistics": ["1000 qubits", "40% improvement"],
        },
        sources_used=[1],
        confidence=0.88,
    )


# ── SearchAgent Tests ──────────────────────────────────────────────────────────

class TestSearchAgent:

    def test_parse_results_deduplicates(self):
        agent = SearchAgent(api_key="test")
        raw = [
            {"url": "https://a.com", "title": "A", "content": "content a", "score": 0.9},
            {"url": "https://a.com", "title": "A Duplicate", "content": "dup", "score": 0.8},
            {"url": "https://b.com", "title": "B", "content": "content b", "score": 0.7},
        ]
        results = agent._parse_results(raw)
        urls = [r.url for r in results]
        assert len(urls) == 2
        assert urls.count("https://a.com") == 1

    def test_parse_results_filters_low_score(self):
        agent = SearchAgent(api_key="test", min_score=0.5)
        raw = [
            {"url": "https://high.com", "title": "High", "content": "good", "score": 0.9},
            {"url": "https://low.com", "title": "Low", "content": "poor", "score": 0.2},
        ]
        results = agent._parse_results(raw)
        assert len(results) == 1
        assert results[0].url == "https://high.com"

    def test_parse_results_sorted_by_score(self):
        agent = SearchAgent(api_key="test")
        raw = [
            {"url": "https://b.com", "title": "B", "content": "b", "score": 0.7},
            {"url": "https://a.com", "title": "A", "content": "a", "score": 0.95},
        ]
        results = agent._parse_results(raw)
        assert results[0].score > results[1].score

    @patch("agents.search_agent.TavilyClient")
    def test_run_success(self, mock_tavily_cls, mock_search_result):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "url": mock_search_result.url,
                    "title": mock_search_result.title,
                    "content": mock_search_result.content,
                    "score": mock_search_result.score,
                }
            ],
            "answer": "Test answer",
        }
        mock_tavily_cls.return_value = mock_client

        agent = SearchAgent(api_key="test-key")
        result = agent.run("test query")

        assert result.success
        assert isinstance(result.output, SearchOutput)
        assert result.output.total_found == 1
        assert result.output.answer == "Test answer"

    @patch("agents.search_agent.TavilyClient")
    def test_run_api_failure_returns_error_result(self, mock_tavily_cls):
        mock_client = MagicMock()
        mock_client.search.side_effect = ConnectionError("API down")
        mock_tavily_cls.return_value = mock_client

        agent = SearchAgent(api_key="test-key")
        result = agent.run("test query")

        assert not result.success
        assert result.error is not None
        assert "API down" in result.error or result.output is None


# ── SummarizerAgent Tests ──────────────────────────────────────────────────────

class TestSummarizerAgent:

    def test_parse_response_valid_json(self):
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.log = MagicMock()

        import json
        valid_json = json.dumps({
            "summary": "Test summary",
            "key_points": ["Point 1", "Point 2"],
            "key_entities": {"people": [], "organizations": ["MIT"]},
            "sources_used": [1, 2],
            "confidence": 0.87,
        })

        output = agent._parse_response(valid_json)
        assert output.summary == "Test summary"
        assert len(output.key_points) == 2
        assert output.confidence == pytest.approx(0.87)

    def test_parse_response_strips_markdown_fences(self):
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.log = MagicMock()

        wrapped = '```json\n{"summary": "ok", "key_points": [], "key_entities": {}, "sources_used": [], "confidence": 0.5}\n```'
        output = agent._parse_response(wrapped)
        assert output.summary == "ok"

    def test_parse_response_fallback_on_invalid_json(self):
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.log = MagicMock()

        output = agent._parse_response("This is not JSON at all.")
        assert "not JSON" in output.summary or output.confidence == pytest.approx(0.4)

    def test_run_no_search_output_returns_error(self):
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.log = MagicMock()
        agent.name = "SummarizerAgent"

        result = agent._run("test query", search_output=None)
        assert not result.success
        assert result.error is not None

    def test_build_sources_block(self, mock_search_output):
        agent = SummarizerAgent.__new__(SummarizerAgent)
        agent.log = MagicMock()

        block = agent._build_sources_block(mock_search_output)
        assert "[Source 1]" in block
        assert "Quantum Computing" in block
        assert "Direct Answer" in block


# ── FactCheckerAgent Tests ─────────────────────────────────────────────────────

class TestFactCheckerAgent:

    def test_parse_response_valid_json(self):
        agent = FactCheckerAgent.__new__(FactCheckerAgent)
        agent.log = MagicMock()

        import json
        valid_json = json.dumps({
            "overall_confidence": 0.82,
            "verdict": "VERIFIED",
            "claims": [
                {
                    "claim": "MIT achieved 1000 qubits",
                    "confidence": 0.9,
                    "status": "SUPPORTED",
                    "supporting_sources": [1],
                    "note": "Directly stated in source 1",
                }
            ],
            "contradictions": [],
            "unverified_claims": [],
            "fact_check_summary": "All claims verified.",
        })

        output = agent._parse_response(valid_json)
        assert output.verdict == "VERIFIED"
        assert output.overall_confidence == pytest.approx(0.82)
        assert len(output.claims) == 1
        assert output.claims[0].status == "SUPPORTED"

    def test_verdict_emoji_mapping(self):
        for verdict, emoji in [
            ("VERIFIED", "✅"),
            ("MOSTLY_VERIFIED", "🟡"),
            ("UNCERTAIN", "⚠️"),
            ("DISPUTED", "❌"),
        ]:
            output = FactCheckOutput(overall_confidence=0.5, verdict=verdict)
            assert output.verdict_emoji == emoji

    def test_run_missing_inputs_returns_error(self):
        agent = FactCheckerAgent.__new__(FactCheckerAgent)
        agent.log = MagicMock()
        agent.name = "FactCheckerAgent"

        result = agent._run("query", summary_output=None, search_output=None)
        assert not result.success

    def test_parse_response_fallback(self):
        agent = FactCheckerAgent.__new__(FactCheckerAgent)
        agent.log = MagicMock()

        output = agent._parse_response("not valid json {{{")
        assert output.verdict == "UNCERTAIN"
        assert output.overall_confidence == pytest.approx(0.5)


# ── AgentResult Tests ──────────────────────────────────────────────────────────

class TestAgentResult:

    def test_success_flag_when_no_error(self):
        result = AgentResult(agent_name="test", output={"data": 1})
        assert result.success is True

    def test_success_false_when_error(self):
        result = AgentResult(agent_name="test", output=None, error="Something failed")
        assert result.success is False

    def test_to_dict_structure(self):
        result = AgentResult(
            agent_name="SearchAgent",
            output={"query": "test"},
            metadata={"depth": "advanced"},
            latency_ms=123.4,
        )
        d = result.to_dict()
        assert d["agent_name"] == "SearchAgent"
        assert d["latency_ms"] == pytest.approx(123.4, rel=0.01)
        assert d["success"] is True
        assert "metadata" in d