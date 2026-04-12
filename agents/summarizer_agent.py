"""
agents/summarizer_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Summarization Agent powered by OpenAI GPT-4o.

Takes raw search results and produces structured, concise research
summaries with key entities, themes, and citations.
"""

from __future__ import annotations

import os
from typing import Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable

from agents.base_agent import BaseAgent, AgentResult
from agents.search_agent import SearchOutput

# ── System prompt ────────────────────────────────────────────────────────────

SUMMARIZER_SYSTEM_PROMPT = """You are an expert research summarizer. Your job is to synthesize \
information from multiple web sources into a clear, structured research summary.

GUIDELINES:
- Be factual, concise, and well-organized
- Cite sources inline using [Source N] notation
- Extract key entities: people, organizations, dates, numbers
- Highlight the most important findings prominently
- Note any conflicting information between sources
- Do NOT fabricate information not present in the sources
- Maintain a neutral, academic tone

OUTPUT FORMAT (respond in this exact JSON structure):
{
  "summary": "<2-4 paragraph comprehensive summary>",
  "key_points": ["<point 1>", "<point 2>", ...],
  "key_entities": {
    "people": ["name1", "name2"],
    "organizations": ["org1", "org2"],
    "dates": ["date1", "date2"],
    "statistics": ["stat1", "stat2"]
  },
  "sources_used": [<source index numbers used>],
  "confidence": <float 0.0-1.0 reflecting source quality and coverage>
}"""


@dataclass
class SummaryOutput:
    """Structured output from the SummarizerAgent."""
    summary: str
    key_points: list[str] = field(default_factory=list)
    key_entities: dict = field(default_factory=dict)
    sources_used: list[int] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "key_entities": self.key_entities,
            "sources_used": self.sources_used,
            "confidence": round(self.confidence, 3),
        }


class SummarizerAgent(BaseAgent):
    """
    Summarization Agent using GPT-4o.

    Features:
    - Structured JSON output with key entities
    - Source citation tracking
    - Confidence scoring based on source quality
    - Multi-document synthesis
    """

    name = "SummarizerAgent"
    description = "Synthesizes search results into structured research summaries using GPT-4o."

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.llm = ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model = model

    @traceable(name="SummarizerAgent._run", run_type="llm")
    def _run(self, query: str, **kwargs) -> AgentResult:
        """
        Summarize search results for the given query.

        Args:
            query: The original research question.
            search_output: SearchOutput object from SearchAgent (passed via kwargs).
            extra_context: Optional additional text context.

        Returns:
            AgentResult with SummaryOutput as the output payload.
        """
        search_output: Optional[SearchOutput] = kwargs.get("search_output")
        extra_context: str = kwargs.get("extra_context", "")

        if not search_output or not search_output.results:
            return AgentResult(
                agent_name=self.name,
                output=None,
                error="No search results provided to summarize.",
            )

        # Build source-numbered content block
        sources_block = self._build_sources_block(search_output)

        user_message = f"""RESEARCH QUERY: {query}

SOURCES:
{sources_block}

{f'ADDITIONAL CONTEXT: {extra_context}' if extra_context else ''}

Please synthesize the above sources into a structured research summary following the JSON format."""

        self.log.info("summarize_start", query=query, num_sources=len(search_output.results))

        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        raw_text = response.content.strip()

        # Parse JSON response
        summary_output = self._parse_response(raw_text)

        self.log.info(
            "summarize_complete",
            num_key_points=len(summary_output.key_points),
            confidence=summary_output.confidence,
        )

        return AgentResult(
            agent_name=self.name,
            output=summary_output,
            metadata={
                "model": self.model,
                "num_sources": len(search_output.results),
                "confidence": summary_output.confidence,
                "usage": {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if response.usage_metadata else 0,
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0) if response.usage_metadata else 0,
                },
            },
        )

    def _build_sources_block(self, search_output: SearchOutput) -> str:
        """Format search results into a numbered source block for the prompt."""
        parts = []

        # Include Tavily's direct answer if available
        if search_output.answer:
            parts.append(f"[Direct Answer]: {search_output.answer}\n")

        for i, result in enumerate(search_output.results, 1):
            parts.append(
                f"[Source {i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"Relevance: {result.score:.2f}\n"
                f"Content: {result.content[:1200]}\n"
            )

        return "\n---\n".join(parts)

    def _parse_response(self, raw_text: str) -> SummaryOutput:
        """Parse the LLM JSON response into a SummaryOutput object."""
        import json, re

        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("```").strip()

        try:
            data = json.loads(clean)
            return SummaryOutput(
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                key_entities=data.get("key_entities", {}),
                sources_used=data.get("sources_used", []),
                confidence=float(data.get("confidence", 0.5)),
                raw_response=raw_text,
            )
        except (json.JSONDecodeError, ValueError):
            self.log.warning("json_parse_failed", raw_preview=raw_text[:200])
            # Fallback: treat entire response as plain summary
            return SummaryOutput(
                summary=raw_text,
                confidence=0.4,
                raw_response=raw_text,
            )