"""
workflows/pipeline.py
~~~~~~~~~~~~~~~~~~~~~
End-to-end orchestration layer.

Provides a unified `ResearchPipeline` class that wraps the LangGraph
state machine, handles caching, triggers n8n webhooks, and publishes
results to downstream systems.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from agents.search_agent import SearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.fact_checker_agent import FactCheckerAgent
from workflows.graph import build_research_graph, ResearchState
from utils.tracing import get_tracer
from utils.cache import CacheClient
from utils.logger import get_logger

log = get_logger(__name__)


class ResearchPipeline:
    """
    End-to-end Multi-Agent Research Pipeline.

    Flow:
      1. Check Redis cache for existing result
      2. Run LangGraph state machine (Search → Summarize → Fact-Check → Report)
      3. Store result in cache
      4. Fire n8n webhook with report (optional)
      5. Return structured research report

    Usage:
        pipeline = ResearchPipeline()
        report = await pipeline.run("Latest advances in quantum computing")
    """

    def __init__(
        self,
        search_agent: Optional[SearchAgent] = None,
        summarizer_agent: Optional[SummarizerAgent] = None,
        fact_checker_agent: Optional[FactCheckerAgent] = None,
        cache: Optional[CacheClient] = None,
        enable_n8n: bool = True,
        enable_cache: bool = True,
    ):
        self.search_agent = search_agent or SearchAgent()
        self.summarizer_agent = summarizer_agent or SummarizerAgent()
        self.fact_checker_agent = fact_checker_agent or FactCheckerAgent()
        self.cache = cache or CacheClient()
        self.enable_n8n = enable_n8n and bool(os.getenv("N8N_WEBHOOK_URL"))
        self.enable_cache = enable_cache

        # Build compiled LangGraph
        self._graph = build_research_graph(
            search_agent=self.search_agent,
            summarizer_agent=self.summarizer_agent,
            fact_checker_agent=self.fact_checker_agent,
        )

        log.info(
            "pipeline_initialized",
            n8n_enabled=self.enable_n8n,
            cache_enabled=self.enable_cache,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        search_depth: str = "advanced",
        use_cache: bool = True,
        notify_n8n: bool = True,
    ) -> dict:
        """
        Execute the full research pipeline for the given query.

        Args:
            query: Research question or topic string.
            search_depth: "basic" or "advanced" (Tavily depth).
            use_cache: Whether to check/write Redis cache.
            notify_n8n: Whether to fire n8n webhook on completion.

        Returns:
            Structured research report dictionary.
        """
        cache_key = self._cache_key(query, search_depth)

        # ── 1. Cache check ────────────────────────────────────────────────────
        if self.enable_cache and use_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                log.info("pipeline_cache_hit", query=query)
                cached["_from_cache"] = True
                return cached

        # ── 2. Build initial state ────────────────────────────────────────────
        initial_state: ResearchState = {
            "query": query,
            "search_depth": search_depth,
            "search_output": None,
            "summary_output": None,
            "fact_check_output": None,
            "report": None,
            "errors": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "total_latency_ms": 0.0,
            "node_timings": {},
        }

        log.info("pipeline_run_start", query=query, depth=search_depth)

        # ── 3. Run LangGraph (in thread pool to avoid blocking) ───────────────
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: self._graph.invoke(initial_state),
        )

        report = final_state.get("report", {})
        report["_from_cache"] = False

        log.info(
            "pipeline_run_complete",
            latency_ms=report.get("total_latency_ms"),
            status=report.get("status"),
            confidence=report.get("overall_confidence"),
        )

        # ── 4. Write cache ────────────────────────────────────────────────────
        if self.enable_cache and report.get("status") == "completed":
            await self.cache.set(cache_key, report)

        # ── 5. Fire n8n webhook ───────────────────────────────────────────────
        if self.enable_n8n and notify_n8n:
            asyncio.create_task(self._notify_n8n(report))

        return report

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str, depth: str) -> str:
        digest = hashlib.sha256(f"{query}:{depth}".encode()).hexdigest()[:16]
        return f"research:{digest}"

    async def _notify_n8n(self, report: dict) -> None:
        """POST the research report to the n8n webhook (non-blocking)."""
        url = os.getenv("N8N_WEBHOOK_URL", "")
        secret = os.getenv("N8N_WEBHOOK_SECRET", "")

        if not url:
            return

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json=report,
                    headers={"X-Webhook-Secret": secret},
                )
                log.info("n8n_webhook_sent", status_code=resp.status_code)
        except Exception as exc:
            log.warning("n8n_webhook_failed", error=str(exc))