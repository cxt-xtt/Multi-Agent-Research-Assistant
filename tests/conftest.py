"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared pytest fixtures and configuration.
Adds the project root to sys.path so all modules are importable.
"""
import sys
import os
 
# Ensure project root is on the path regardless of where pytest is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
# Set dummy env vars so modules can be imported without real keys
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langsmith-key")
os.environ.setdefault("LANGCHAIN_PROJECT", "test-project")
os.environ.setdefault("SECRET_KEY", "test-secret-key-32-chars-minimum!")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
 