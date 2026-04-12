"""
workflows/graph.py
~~~~~~~~~~~~~~~~~~
LangGraph state machine for the Multi-Agent Research Pipeline.
 
  [START]
     │
     ▼
  search_node  ──► (no results?) ──► [END with error]
     │
     ▼
  summarize_node
     │
     ▼
  fact_check_node
     │
     ▼
  build_report_node
     │
     ▼
  [END]
"""
 
from __future__ import annotations
 
import time
from typing import TypedDict, Optional
from datetime import datetime, timezone
 
import structlog
from langgraph.graph import StateGraph, END, START
 
from agents.search_agent import SearchAgent, SearchOutput
from agents.summarizer_agent import SummarizerAgent, SummaryOutput
from agents.fact_checker_agent import FactCheckerAgent, FactCheckOutput
 
log = structlog.get_logger(__name__)
 
 
# ── Pipeline State ────────────────────────────────────────────────────────────
 
class ResearchState(TypedDict):
    """Shared state passed between all graph nodes."""
    query: str
    search_depth: str
    search_output: Optional[SearchOutput]
    summary_output: Optional[SummaryOutput]
    fact_check_output: Optional[FactCheckOutput]
    report: Optional[dict]
    errors: list[str]
    started_at: str
    completed_at: Optional[str]
    total_latency_ms: float
    node_timings: dict[str, float]
 
 
# ── Node Implementations ──────────────────────────────────────────────────────
 
def make_search_node(agent: SearchAgent):
    def search_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        log.info("graph_node_search", query=state["query"])
 
        result = agent.run(
            query=state["query"],
            search_depth=state.get("search_depth", "advanced"),
        )
 
        timings = dict(state.get("node_timings", {}))
        timings["search"] = (time.perf_counter() - t0) * 1000
        errors = list(state.get("errors", []))
 
        if not result.success:
            errors.append(f"SearchAgent: {result.error}")
            return {"search_output": None, "errors": errors, "node_timings": timings}
 
        return {
            "search_output": result.output,
            "errors": errors,
            "node_timings": timings,
        }
    return search_node
 
 
def make_summarize_node(agent: SummarizerAgent):
    def summarize_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        log.info("graph_node_summarize")
 
        result = agent.run(
            query=state["query"],
            search_output=state["search_output"],
        )
 
        timings = dict(state.get("node_timings", {}))
        timings["summarize"] = (time.perf_counter() - t0) * 1000
        errors = list(state.get("errors", []))
 
        if not result.success:
            errors.append(f"SummarizerAgent: {result.error}")
            return {"summary_output": None, "errors": errors, "node_timings": timings}
 
        return {
            "summary_output": result.output,
            "errors": errors,
            "node_timings": timings,
        }
    return summarize_node
 
 
def make_fact_check_node(agent: FactCheckerAgent):
    def fact_check_node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        log.info("graph_node_fact_check")
 
        result = agent.run(
            query=state["query"],
            summary_output=state["summary_output"],
            search_output=state["search_output"],
        )
 
        timings = dict(state.get("node_timings", {}))
        timings["fact_check"] = (time.perf_counter() - t0) * 1000
        errors = list(state.get("errors", []))
 
        if not result.success:
            errors.append(f"FactCheckerAgent: {result.error}")
            return {"fact_check_output": None, "errors": errors, "node_timings": timings}
 
        return {
            "fact_check_output": result.output,
            "errors": errors,
            "node_timings": timings,
        }
    return fact_check_node
 
 
def build_report_node(state: ResearchState) -> dict:
    """Assembles all agent outputs into the final research report."""
    log.info("graph_node_build_report")
 
    completed_at = datetime.now(timezone.utc).isoformat()
    node_timings = state.get("node_timings", {})
    total_ms = sum(node_timings.values())
 
    search: Optional[SearchOutput] = state.get("search_output")
    summary: Optional[SummaryOutput] = state.get("summary_output")
    fact_check: Optional[FactCheckOutput] = state.get("fact_check_output")
 
    report = {
        "query": state["query"],
        "status": "completed" if not state.get("errors") else "partial",
        "started_at": state["started_at"],
        "completed_at": completed_at,
        "total_latency_ms": round(total_ms, 1),
        "node_timings_ms": {k: round(v, 1) for k, v in node_timings.items()},
        "errors": state.get("errors", []),
        "sources": [r.to_dict() for r in search.results] if search else [],
        "direct_answer": search.answer if search else None,
        "summary": summary.to_dict() if summary else None,
        "fact_check": fact_check.to_dict() if fact_check else None,
        "overall_confidence": _compute_overall_confidence(summary, fact_check),
    }
 
    return {
        "report": report,
        "completed_at": completed_at,
        "total_latency_ms": total_ms,
    }
 
 
# ── Conditional Edges ─────────────────────────────────────────────────────────
 
def should_continue_after_search(state: ResearchState) -> str:
    if not state.get("search_output") or not state["search_output"].results:
        log.warning("graph_skip_no_search_results")
        return "build_report"
    return "summarize"
 
 
def should_continue_after_summarize(state: ResearchState) -> str:
    if not state.get("summary_output"):
        return "build_report"
    return "fact_check"
 
 
# ── Graph Builder ─────────────────────────────────────────────────────────────
 
def build_research_graph(
    search_agent: Optional[SearchAgent] = None,
    summarizer_agent: Optional[SummarizerAgent] = None,
    fact_checker_agent: Optional[FactCheckerAgent] = None,
):
    """Build and compile the LangGraph research pipeline."""
    search_agent = search_agent or SearchAgent()
    summarizer_agent = summarizer_agent or SummarizerAgent()
    fact_checker_agent = fact_checker_agent or FactCheckerAgent()
 
    graph = StateGraph(ResearchState)
 
    # Add nodes — note: node names must NOT clash with ResearchState field names
    graph.add_node("search", make_search_node(search_agent))
    graph.add_node("summarize", make_summarize_node(summarizer_agent))
    graph.add_node("fact_check", make_fact_check_node(fact_checker_agent))
    graph.add_node("build_report", build_report_node)   # renamed from "report"
 
    # Edges
    graph.add_edge(START, "search")
 
    graph.add_conditional_edges(
        "search",
        should_continue_after_search,
        {"summarize": "summarize", "build_report": "build_report"},
    )
 
    graph.add_conditional_edges(
        "summarize",
        should_continue_after_summarize,
        {"fact_check": "fact_check", "build_report": "build_report"},
    )
 
    graph.add_edge("fact_check", "build_report")
    graph.add_edge("build_report", END)
 
    return graph.compile()
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _compute_overall_confidence(
    summary: Optional[SummaryOutput],
    fact_check: Optional[FactCheckOutput],
) -> float:
    scores = []
    if summary:
        scores.append(summary.confidence)
    if fact_check:
        scores.append(fact_check.overall_confidence)
    return round(sum(scores) / len(scores), 3) if scores else 0.0