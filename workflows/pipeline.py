# -*- coding: utf-8 -*-
"""
workflows/pipeline.py
~~~~~~~~~~~~~~~~~~~~~
End-to-end orchestration layer.

Provides a unified ResearchPipeline class that wraps the LangGraph
state machine, handles caching, triggers n8n webhooks, and publishes
results to downstream systems.

V2: Added user context (conversation history) and knowledge base (ChromaDB RAG).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from agents.search_agent import SearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.fact_checker_agent import FactCheckerAgent
from workflows.graph import build_research_graph, ResearchState
from utils.tracing import get_tracer
from utils.cache import CacheClient
from utils.logger import get_logger
from utils.conversation import get_context_block, save_turn
from utils.knowledge_agent import kb as knowledge_base

log = get_logger(__name__)


class ResearchPipeline:
    """
    End-to-end Multi-Agent Research Pipeline.

    Flow:
      1. Inject user context (conversation history + knowledge base)
      2. Check Redis cache for existing result
      3. Run LangGraph state machine (Search -> Summarize -> Fact-Check -> Report)
      4. Store result in cache
      5. Save conversation turn for future context
      6. Fire n8n webhook with report (optional)
      7. Return structured research report

    Usage:
        pipeline = ResearchPipeline()
        report = await pipeline.run("Latest advances in quantum computing", user_id="alice")
    """

    def __init__(
        self,
        search_agent: Optional[SearchAgent] = None,
        summarizer_agent: Optional[SummarizerAgent] = None,
        fact_checker_agent: Optional[FactCheckerAgent] = None,
        cache: Optional[CacheClient] = None,
        enable_n8n: bool = True,
        enable_cache: bool = True,
        enable_knowledge_base: bool = True,
        enable_user_context: bool = True,
    ):
        self.search_agent = search_agent or SearchAgent()
        self.summarizer_agent = summarizer_agent or SummarizerAgent()
        self.fact_checker_agent = fact_checker_agent or FactCheckerAgent()
        self.cache = cache or CacheClient()
        self.enable_n8n = enable_n8n and bool(os.getenv("N8N_WEBHOOK_URL"))
        self.enable_cache = enable_cache
        self.enable_kb = enable_knowledge_base
        self.enable_context = enable_user_context

        self._graph = build_research_graph(
            search_agent=self.search_agent,
            summarizer_agent=self.summarizer_agent,
            fact_checker_agent=self.fact_checker_agent,
        )

        log.info(
            "pipeline_initialized",
            n8n_enabled=self.enable_n8n,
            cache_enabled=self.enable_cache,
            kb_enabled=self.enable_kb,
            context_enabled=self.enable_context,
        )

    async def run(
        self,
        query: str,
        user_id: str = "default",
        search_depth: str = "advanced",
        use_cache: bool = True,
        notify_n8n: bool = True,
    ) -> dict:
        """Execute the full research pipeline for the given query."""

        context_parts = []

        if self.enable_context:
            history = get_context_block(user_id)
            if history:
                context_parts.append(history)

        if self.enable_kb:
            try:
                kb_results = knowledge_base.search(user_id, query, top_k=3)
                if kb_results:
                    context_parts.append(
                        "*** User Knowledge Base ***\n" + "\n---\n".join(kb_results)
                    )
            except Exception as e:
                log.warning("kb_search_failed", error=str(e))

        enhanced_query = (
            "\n\n".join(context_parts) + f"\n\nCurrent query: {query}"
            if context_parts
            else query
        )

        cache_key = self._cache_key(query, search_depth)

        if self.enable_cache and use_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                log.info("pipeline_cache_hit", query=query, user_id=user_id)
                cached["_from_cache"] = True
                return cached

        initial_state: ResearchState = {
            "query": enhanced_query,
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

        log.info(
            "pipeline_run_start",
            query=query,
            user_id=user_id,
            depth=search_depth,
            has_context=bool(context_parts),
        )

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
            user_id=user_id,
        )

        if self.enable_context:
            summary_obj = report.get("summary") or {}
            answer_text = (
                report.get("direct_answer")
                or summary_obj.get("summary", "")
                or str(report)[:200]
            )
            save_turn(user_id, query, answer_text)

        if self.enable_cache and report.get("status") == "completed":
            await self.cache.set(cache_key, report)

        if self.enable_n8n and notify_n8n:
            asyncio.create_task(self._notify_n8n(report))

        return report

    @staticmethod
    def _cache_key(query: str, depth: str) -> str:
        digest = hashlib.sha256(f"{query}:{depth}".encode()).hexdigest()[:16]
        return f"research:{digest}"

    async def _notify_n8n(self, report: dict) -> None:
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
