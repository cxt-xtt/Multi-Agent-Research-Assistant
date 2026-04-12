"""
agents/search_agent.py
~~~~~~~~~~~~~~~~~~~~~~
Web Search Agent powered by Tavily Search API.

Performs real-time web search, deduplicates results, and returns
structured source objects with relevance scores for downstream agents.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from dataclasses import dataclass, field

import structlog
from langsmith import traceable
from tavily import TavilyClient

from agents.base_agent import BaseAgent, AgentResult

log = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result from Tavily."""
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None
    published_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": round(self.score, 4),
            "published_date": self.published_date,
        }


@dataclass
class SearchOutput:
    """Aggregated output from a search operation."""
    query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: Optional[str] = None  # Tavily's direct answer if available
    total_found: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "total_found": self.total_found,
            "results": [r.to_dict() for r in self.results],
        }

    def get_combined_content(self, max_chars: int = 8000) -> str:
        """Return concatenated content from all results, capped at max_chars."""
        parts = []
        total = 0
        for r in self.results:
            chunk = f"[Source: {r.title}]\n{r.content}\n"
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n---\n".join(parts)


class SearchAgent(BaseAgent):
    """
    Web Search Agent using Tavily Search API.

    Features:
    - Real-time web search with configurable depth
    - Automatic deduplication of URLs
    - Relevance score filtering
    - Structured output for downstream agents
    """

    name = "SearchAgent"
    description = "Performs real-time web searches and returns ranked, structured results."

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "advanced",
        max_results: int = 8,
        min_score: float = 0.3,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.client = TavilyClient(api_key=api_key or os.getenv("TAVILY_API_KEY"))
        self.search_depth = search_depth  # "basic" or "advanced"
        self.max_results = max_results
        self.min_score = min_score

    @traceable(name="SearchAgent._run", run_type="retriever")
    def _run(self, query: str, **kwargs) -> AgentResult:
        """
        Execute a web search for the given query.

        Args:
            query: Research question or topic.
            search_depth: Override default depth ("basic" | "advanced").
            max_results: Override default result count.

        Returns:
            AgentResult with SearchOutput as the output payload.
        """
        depth = kwargs.get("search_depth", self.search_depth)
        max_r = kwargs.get("max_results", self.max_results)

        self.log.info("search_start", query=query, depth=depth, max_results=max_r)

        # Call Tavily API
        raw = self.client.search(
            query=query,
            search_depth=depth,
            max_results=max_r,
            include_answer=True,
            include_raw_content=False,
        )

        # Parse & filter results
        results = self._parse_results(raw.get("results", []))

        output = SearchOutput(
            query=query,
            results=results,
            answer=raw.get("answer"),
            total_found=len(results),
        )

        self.log.info(
            "search_complete",
            total_found=output.total_found,
            has_direct_answer=bool(output.answer),
        )

        return AgentResult(
            agent_name=self.name,
            output=output,
            metadata={
                "search_depth": depth,
                "max_results": max_r,
                "total_returned": output.total_found,
                "has_direct_answer": bool(output.answer),
            },
        )

    def _parse_results(self, raw_results: list[dict]) -> list[SearchResult]:
        """Parse Tavily raw results into SearchResult objects, filtered and deduplicated."""
        seen_urls: set[str] = set()
        parsed: list[SearchResult] = []

        for r in raw_results:
            url = r.get("url", "")
            score = float(r.get("score", 0.0))

            # Deduplicate and filter low-relevance results
            if url in seen_urls or score < self.min_score:
                continue

            seen_urls.add(url)
            parsed.append(
                SearchResult(
                    title=r.get("title", "Untitled"),
                    url=url,
                    content=r.get("content", ""),
                    score=score,
                    published_date=r.get("published_date"),
                )
            )

        # Sort by relevance score descending
        parsed.sort(key=lambda x: x.score, reverse=True)
        return parsed