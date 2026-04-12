"""
agents/base_agent.py
~~~~~~~~~~~~~~~~~~~~
Abstract base class for all research agents.
Provides common interface, retry logic, and LangSmith tracing hooks.
"""
 
from __future__ import annotations
 
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
 
import structlog
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
 
log = structlog.get_logger(__name__)
 
 
class AgentResult:
    """Standardised result container returned by every agent."""
 
    def __init__(
        self,
        agent_name: str,
        output: Any,
        metadata: Optional[dict] = None,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
    ):
        self.agent_name = agent_name
        self.output = output
        self.metadata = metadata or {}
        self.error = error
        self.latency_ms = latency_ms
        self.success = error is None
 
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "output": self.output,
            "metadata": self.metadata,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
        }
 
    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} AgentResult(agent={self.agent_name}, latency={self.latency_ms:.0f}ms)"
 
 
class BaseAgent(ABC):
    """
    Abstract base class for all LLM-powered research agents.
 
    Subclasses must implement `_run(query, **kwargs) -> AgentResult`.
    The public `run()` method wraps execution with:
      - Structured logging
      - LangSmith tracing (via @traceable decorator)
      - Exponential-backoff retry on transient failures
      - Latency measurement
    """
 
    # Override in subclasses
    name: str = "BaseAgent"
    description: str = "Abstract research agent"
    max_retries: int = 3
 
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log = structlog.get_logger(self.name)
 
    # ── Public interface ─────────────────────────────────────────
 
    def run(self, query: str, **kwargs) -> AgentResult:
        """
        Execute the agent with retry logic and timing.
        This is the primary method callers should use.
        """
        self.log.info("agent_start", query=query[:120])
        start = time.perf_counter()
 
        try:
            result = self._run_with_retry(query, **kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            self.log.info(
                "agent_complete",
                latency_ms=round(result.latency_ms, 1),
                success=result.success,
            )
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self.log.error("agent_failed", error=str(exc), latency_ms=round(latency_ms, 1))
            return AgentResult(
                agent_name=self.name,
                output=None,
                error=str(exc),
                latency_ms=latency_ms,
            )
 
    # ── Internal helpers ─────────────────────────────────────────
 
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _run_with_retry(self, query: str, **kwargs) -> AgentResult:
        return self._run(query, **kwargs)
 
    @abstractmethod
    def _run(self, query: str, **kwargs) -> AgentResult:
        """
        Core agent logic — must be implemented by each subclass.
 
        Args:
            query: The research question or topic string.
            **kwargs: Agent-specific options.
 
        Returns:
            AgentResult with output and metadata.
        """
        ...
 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"