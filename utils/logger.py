"""
utils/logger.py
~~~~~~~~~~~~~~~
Structured logging setup using structlog + rich for pretty console output.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with pretty console output in dev, JSON in production."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    is_dev = os.getenv("DEBUG", "false").lower() == "true"

    if is_dev:
        # Pretty colored output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production log aggregation
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to route through structlog
    logging.basicConfig(
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(message)s",
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)


# Auto-configure on import
configure_logging(os.getenv("LOG_LEVEL", "INFO"))