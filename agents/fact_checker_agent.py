"""
agents/fact_checker_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fact-Checking Agent powered by GPT-4o.

Cross-references claims from the summary against source material,
assigns per-claim confidence scores, and flags contradictions.
"""

from __future__ import annotations

import os
import json
import re
from typing import Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable

from agents.base_agent import BaseAgent, AgentResult
from agents.search_agent import SearchOutput
from agents.summarizer_agent import SummaryOutput

# ── System prompt ────────────────────────────────────────────────────────────

FACT_CHECKER_SYSTEM_PROMPT = """You are an expert fact-checker and critical analyst. Your job is to \
evaluate claims from a research summary against the original source material.

For each key claim:
1. Verify it is supported by at least one source
2. Assign a confidence score (0.0–1.0)
3. Note any contradictions or caveats between sources
4. Flag anything that appears to be hallucinated or unsupported

CONFIDENCE SCALE:
  1.0 — Confirmed by multiple independent sources
  0.8 — Confirmed by one reliable source
  0.6 — Partially supported / inferred
  0.4 — Weakly supported / ambiguous
  0.2 — Contradicted by sources or unverifiable
  0.0 — Not found in sources / likely hallucinated

Respond ONLY with a valid JSON object in this exact schema:
{
  "overall_confidence": <float 0.0-1.0>,
  "verdict": "VERIFIED" | "MOSTLY_VERIFIED" | "UNCERTAIN" | "DISPUTED",
  "claims": [
    {
      "claim": "<extracted claim text>",
      "confidence": <float 0.0-1.0>,
      "status": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED",
      "supporting_sources": [<source indices>],
      "note": "<brief explanation>"
    }
  ],
  "contradictions": ["<description of contradiction 1>", ...],
  "unverified_claims": ["<claim that could not be verified>", ...],
  "fact_check_summary": "<2-3 sentence overall assessment>"
}"""


@dataclass
class ClaimResult:
    """Result for a single claim evaluation."""
    claim: str
    confidence: float
    status: str  # SUPPORTED | PARTIAL | UNSUPPORTED | CONTRADICTED
    supporting_sources: list[int]
    note: str

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "supporting_sources": self.supporting_sources,
            "note": self.note,
        }


@dataclass
class FactCheckOutput:
    """Full fact-check result for a research summary."""
    overall_confidence: float
    verdict: str  # VERIFIED | MOSTLY_VERIFIED | UNCERTAIN | DISPUTED
    claims: list[ClaimResult] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    fact_check_summary: str = ""

    @property
    def verdict_emoji(self) -> str:
        mapping = {
            "VERIFIED": "✅",
            "MOSTLY_VERIFIED": "🟡",
            "UNCERTAIN": "⚠️",
            "DISPUTED": "❌",
        }
        return mapping.get(self.verdict, "❓")

    def to_dict(self) -> dict:
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "verdict": self.verdict,
            "verdict_emoji": self.verdict_emoji,
            "claims": [c.to_dict() for c in self.claims],
            "contradictions": self.contradictions,
            "unverified_claims": self.unverified_claims,
            "fact_check_summary": self.fact_check_summary,
        }


class FactCheckerAgent(BaseAgent):
    """
    Fact-Checking Agent using GPT-4o.

    Features:
    - Per-claim confidence scoring
    - Contradiction detection across sources
    - Hallucination flagging
    - Overall verdict: VERIFIED | MOSTLY_VERIFIED | UNCERTAIN | DISPUTED
    """

    name = "FactCheckerAgent"
    description = "Cross-references summary claims against sources and assigns confidence scores."

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
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

    @traceable(name="FactCheckerAgent._run", run_type="llm")
    def _run(self, query: str, **kwargs) -> AgentResult:
        """
        Fact-check the summary against the original search results.

        Args:
            query: The original research question.
            summary_output: SummaryOutput from SummarizerAgent (via kwargs).
            search_output: SearchOutput from SearchAgent (via kwargs).

        Returns:
            AgentResult with FactCheckOutput as the output payload.
        """
        summary_output: Optional[SummaryOutput] = kwargs.get("summary_output")
        search_output: Optional[SearchOutput] = kwargs.get("search_output")

        if not summary_output or not summary_output.summary:
            return AgentResult(
                agent_name=self.name,
                output=None,
                error="No summary provided to fact-check.",
            )

        if not search_output or not search_output.results:
            return AgentResult(
                agent_name=self.name,
                output=None,
                error="No source material provided for fact-checking.",
            )

        sources_block = self._build_sources_block(search_output)
        key_points_block = "\n".join(f"- {p}" for p in summary_output.key_points)

        user_message = f"""RESEARCH QUERY: {query}

SUMMARY TO FACT-CHECK:
{summary_output.summary}

KEY CLAIMS TO VERIFY:
{key_points_block}

ORIGINAL SOURCES (ground truth):
{sources_block}

Please fact-check each claim against the sources and return the structured JSON response."""

        self.log.info(
            "fact_check_start",
            query=query,
            num_claims=len(summary_output.key_points),
            num_sources=len(search_output.results),
        )

        messages = [
            SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        raw_text = response.content.strip()

        fact_check_output = self._parse_response(raw_text)

        self.log.info(
            "fact_check_complete",
            verdict=fact_check_output.verdict,
            overall_confidence=fact_check_output.overall_confidence,
            num_contradictions=len(fact_check_output.contradictions),
        )

        return AgentResult(
            agent_name=self.name,
            output=fact_check_output,
            metadata={
                "model": self.model,
                "verdict": fact_check_output.verdict,
                "overall_confidence": fact_check_output.overall_confidence,
                "num_claims_checked": len(fact_check_output.claims),
                "num_contradictions": len(fact_check_output.contradictions),
            },
        )

    def _build_sources_block(self, search_output: SearchOutput) -> str:
        parts = []
        for i, result in enumerate(search_output.results, 1):
            parts.append(
                f"[Source {i}] {result.title}\n"
                f"{result.content[:1000]}"
            )
        return "\n---\n".join(parts)

    def _parse_response(self, raw_text: str) -> FactCheckOutput:
        clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("```").strip()

        try:
            data = json.loads(clean)
            claims = [
                ClaimResult(
                    claim=c.get("claim", ""),
                    confidence=float(c.get("confidence", 0.5)),
                    status=c.get("status", "PARTIAL"),
                    supporting_sources=c.get("supporting_sources", []),
                    note=c.get("note", ""),
                )
                for c in data.get("claims", [])
            ]
            return FactCheckOutput(
                overall_confidence=float(data.get("overall_confidence", 0.5)),
                verdict=data.get("verdict", "UNCERTAIN"),
                claims=claims,
                contradictions=data.get("contradictions", []),
                unverified_claims=data.get("unverified_claims", []),
                fact_check_summary=data.get("fact_check_summary", ""),
            )
        except (json.JSONDecodeError, ValueError):
            self.log.warning("json_parse_failed", raw_preview=raw_text[:200])
            return FactCheckOutput(
                overall_confidence=0.5,
                verdict="UNCERTAIN",
                fact_check_summary=raw_text[:500],
            )