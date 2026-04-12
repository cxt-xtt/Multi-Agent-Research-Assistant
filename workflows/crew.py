"""
workflows/crew.py
~~~~~~~~~~~~~~~~~
CrewAI crew definition for parallel/coordinated agent execution.

The ResearchCrew orchestrates three specialized agents:
  - ResearcherAgent   : Gathers information via web search
  - AnalystAgent      : Synthesizes and summarizes findings
  - FactCheckerAgent  : Validates claims and assigns confidence scores

Tasks are executed sequentially with shared context, allowing each
agent to build on the prior agent's output.
"""

from __future__ import annotations

import os
from typing import Optional

from crewai import Agent, Crew, Task, Process
from crewai_tools import TavilySearchResults
from langchain_openai import ChatOpenAI

import structlog

log = structlog.get_logger(__name__)


def build_research_crew(
    query: str,
    verbose: bool = False,
    max_iter: int = 5,
) -> tuple[Crew, list[Task]]:
    """
    Build a CrewAI research crew for the given query.

    Args:
        query: The research question or topic.
        verbose: Enable verbose CrewAI output.
        max_iter: Maximum iterations per agent.

    Returns:
        Tuple of (Crew instance, list of Task objects).
    """
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.2,
    )

    # ── Tools ────────────────────────────────────────────────────────────────

    search_tool = TavilySearchResults(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=8,
    )

    # ── Agents ───────────────────────────────────────────────────────────────

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            "Conduct comprehensive web research to gather accurate, up-to-date information "
            f"about: {query}. Find diverse, high-quality sources."
        ),
        backstory=(
            "You are an expert researcher with 15 years of experience in investigative journalism "
            "and academic research. You excel at finding relevant, credible sources and extracting "
            "the most important information quickly."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Research Synthesis Specialist",
        goal=(
            "Analyze the gathered research data and produce a clear, comprehensive, "
            "well-structured summary with key insights and findings."
        ),
        backstory=(
            "You are a skilled analyst who specializes in synthesizing complex information "
            "from multiple sources into clear, actionable summaries. You always cite your "
            "sources and highlight the most important insights."
        ),
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    fact_checker = Agent(
        role="Fact-Checking Expert",
        goal=(
            "Critically evaluate all claims in the research summary, verify them against "
            "source material, and provide a confidence-scored fact-check report."
        ),
        backstory=(
            "You are a meticulous fact-checker with expertise in identifying misinformation, "
            "verifying claims against primary sources, and quantifying confidence levels. "
            "You are known for your rigorous, unbiased approach."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
    )

    # ── Tasks ────────────────────────────────────────────────────────────────

    task_research = Task(
        description=(
            f"Conduct thorough web research on the following topic:\n\n"
            f"QUERY: {query}\n\n"
            "Instructions:\n"
            "1. Search for 6–10 high-quality, relevant web sources\n"
            "2. Extract key facts, statistics, quotes, and insights\n"
            "3. Note the publication date and credibility of each source\n"
            "4. Identify any conflicting information across sources\n"
            "5. Compile a structured list of findings with source URLs"
        ),
        expected_output=(
            "A structured research document containing:\n"
            "- List of sources (title, URL, relevance score)\n"
            "- Key facts and findings extracted from each source\n"
            "- Any conflicts or contradictions found\n"
            "- Raw content excerpts for important claims"
        ),
        agent=researcher,
    )

    task_summarize = Task(
        description=(
            "Using the research gathered by the Research Analyst, create a comprehensive "
            "research summary:\n\n"
            "Instructions:\n"
            "1. Synthesize information from all provided sources\n"
            "2. Write a 3–5 paragraph executive summary\n"
            "3. Extract 5–10 key bullet points\n"
            "4. Identify important entities (people, organizations, dates, numbers)\n"
            "5. Note any gaps or areas needing further research"
        ),
        expected_output=(
            "A polished research summary including:\n"
            "- Executive summary (3–5 paragraphs)\n"
            "- Bullet-point key findings\n"
            "- Entity list (people, orgs, dates, stats)\n"
            "- Source citations inline\n"
            "- Confidence level (0.0–1.0)"
        ),
        agent=analyst,
        context=[task_research],
    )

    task_fact_check = Task(
        description=(
            "Critically fact-check the research summary produced by the Synthesis Specialist:\n\n"
            "Instructions:\n"
            "1. Extract each major claim from the summary\n"
            "2. Verify each claim against the original source material\n"
            "3. Assign a confidence score (0.0–1.0) to each claim\n"
            "4. Flag any unsupported or contradicted claims\n"
            "5. Provide an overall verdict: VERIFIED / MOSTLY_VERIFIED / UNCERTAIN / DISPUTED"
        ),
        expected_output=(
            "A fact-check report including:\n"
            "- Per-claim confidence scores\n"
            "- Status: SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED\n"
            "- List of contradictions found\n"
            "- List of unverifiable claims\n"
            "- Overall confidence score and verdict\n"
            "- 2–3 sentence fact-check assessment"
        ),
        agent=fact_checker,
        context=[task_research, task_summarize],
    )

    # ── Crew ─────────────────────────────────────────────────────────────────

    crew = Crew(
        agents=[researcher, analyst, fact_checker],
        tasks=[task_research, task_summarize, task_fact_check],
        process=Process.sequential,  # Tasks run in order with shared context
        verbose=verbose,
        memory=False,  # Disable persistent memory for stateless API use
    )

    return crew, [task_research, task_summarize, task_fact_check]


def run_crew(query: str, verbose: bool = False) -> dict:
    """
    Run the full CrewAI research crew for the given query.

    Args:
        query: Research question or topic.
        verbose: Enable verbose output.

    Returns:
        Dictionary with crew output and metadata.
    """
    log.info("crew_run_start", query=query)

    crew, tasks = build_research_crew(query=query, verbose=verbose)

    try:
        result = crew.kickoff()
        log.info("crew_run_complete")

        return {
            "success": True,
            "query": query,
            "output": str(result),
            "usage": result.usage_metrics if hasattr(result, "usage_metrics") else {},
        }
    except Exception as exc:
        log.error("crew_run_failed", error=str(exc))
        return {
            "success": False,
            "query": query,
            "output": None,
            "error": str(exc),
        }