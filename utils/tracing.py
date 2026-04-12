"""
utils/tracing.py
~~~~~~~~~~~~~~~~
LangSmith integration for pipeline tracing and observability.

Every research pipeline run is automatically traced with:
  - Full agent execution tree
  - Token usage per node
  - Latency breakdown
  - Intermediate inputs/outputs for debugging
"""

from __future__ import annotations

import os
from typing import Optional, Callable, Any
from functools import wraps

import structlog
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

log = structlog.get_logger(__name__)

# Singleton LangSmith client
_client: Optional[Client] = None


def get_tracer() -> Optional[Client]:
    """Return the singleton LangSmith client, initializing if needed."""
    global _client

    if not os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        return None

    if _client is None:
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            log.warning("langsmith_no_api_key")
            return None

        _client = Client(api_key=api_key)
        log.info(
            "langsmith_initialized",
            project=os.getenv("LANGCHAIN_PROJECT", "default"),
        )

    return _client


def get_current_run_id() -> Optional[str]:
    """Return the current LangSmith run ID for linking traces."""
    run_tree = get_current_run_tree()
    if run_tree:
        return str(run_tree.id)
    return None


def trace_pipeline(name: str, metadata: Optional[dict] = None):
    """
    Decorator to wrap a function in a named LangSmith trace.

    Usage:
        @trace_pipeline(name="research_pipeline")
        async def run_research(query: str): ...
    """
    def decorator(func: Callable) -> Callable:
        @traceable(name=name, metadata=metadata or {})
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def log_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: Optional[str] = None,
) -> bool:
    """
    Log user feedback for a pipeline run to LangSmith.

    Args:
        run_id: The LangSmith run ID.
        key: Feedback key (e.g., "accuracy", "helpfulness").
        score: Score between 0.0 and 1.0.
        comment: Optional text comment.

    Returns:
        True if feedback was logged successfully.
    """
    client = get_tracer()
    if not client:
        return False

    try:
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        log.info("langsmith_feedback_logged", run_id=run_id, key=key, score=score)
        return True
    except Exception as exc:
        log.warning("langsmith_feedback_failed", error=str(exc))
        return False


def get_run_url(run_id: str) -> Optional[str]:
    """Return the LangSmith UI URL for a given run ID."""
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_id}"