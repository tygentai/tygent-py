"""Example of using CrewAI for comprehensive market analysis.

This script reuses the prompts from the Google ADK market analysis example but
implements the workflow using CrewAI agents and tasks. The resulting crew is
then accelerated via Tygent's ``accelerate_crew`` helper.

Requires the ``crewai`` package and a supported LLM backend. Install with
``pip install crewai`` and configure the necessary environment variables for
your LLM provider (for example, ``OPENAI_API_KEY`` for OpenAI models).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    from crewai import Agent, Crew, Task
except Exception as e:  # pragma: no cover - optional dependency
    Agent = Crew = Task = None  # type: ignore
    IMPORT_ERROR = str(e)

from tygent.integrations.crewai import accelerate_crew

# ---------------------------------------------------------------------------
# Prompts reused from the Google ADK market analysis example
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "You are a strategic analyst preparing a comprehensive market intelligence "
    "report for executive leadership. Research emerging trends, competitive "
    "threats, customer behavior patterns, and market opportunities to guide "
    "major business decisions and investment strategies."
)

BASE_PROMPT = (
    "{task}: Analyze market opportunities, competitive landscape, customer trends, "
    "and regulatory environment across multiple industries to inform strategic "
    "business decisions."
)


# ---------------------------------------------------------------------------
# Crew planning and construction
# ---------------------------------------------------------------------------

# A static plan mirroring the output of the Google ADK example. In a real
# application this could be produced dynamically by an LLM.
MARKET_ANALYSIS_PLAN: Dict[str, Dict[str, Any]] = {
    "industry_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Industry research focusing on market size, growth rates, and key players",
        ),
        "deps": [],
    },
    "competitor_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Competitive analysis identifying major competitors and their strengths",
        ),
        "deps": [],
    },
    "customer_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Customer behavior analysis detailing needs and purchasing patterns",
        ),
        "deps": [],
    },
    "regulatory_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Regulatory environment review summarizing relevant policies",
        ),
        "deps": [],
    },
    "trend_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Trend analysis highlighting emerging opportunities",
        ),
        "deps": [],
    },
    "risk_assessment": {
        "prompt": BASE_PROMPT.format(
            task="Risk assessment evaluating threats from market, competitors, and regulations",
        ),
        "deps": ["industry_analysis", "competitor_analysis", "regulatory_analysis"],
    },
    "validation": {
        "prompt": BASE_PROMPT.format(
            task="Validate findings for consistency and reliability",
        ),
        "deps": ["risk_assessment"],
    },
    "executive_summary": {
        "prompt": BASE_PROMPT.format(
            task="Executive summary synthesizing insights and strategic recommendations",
        ),
        "deps": [
            "industry_analysis",
            "competitor_analysis",
            "customer_analysis",
            "regulatory_analysis",
            "trend_analysis",
            "validation",
        ],
    },
}


def build_crew(plan: Dict[str, Dict[str, Any]]) -> Crew:
    """Construct and return a CrewAI crew from the provided plan."""

    if Agent is None or Crew is None or Task is None:  # pragma: no cover - sanity
        raise ImportError(f"CrewAI is not installed: {IMPORT_ERROR}")

    analyst = Agent(
        role="Market Analyst",
        goal="Provide comprehensive market intelligence",
        backstory=INSTRUCTION,
    )

    tasks: Dict[str, Task] = {}
    for name, node in plan.items():
        deps: List[Task] = [tasks[d] for d in node.get("deps", [])]
        task = Task(description=node["prompt"], agent=analyst, context=deps)
        task.id = name
        task.dependencies = deps  # type: ignore[attr-defined]
        tasks[name] = task

    crew = Crew(agents=[analyst], tasks=list(tasks.values()), verbose=True)
    return crew


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


async def main() -> None:
    """Execute the market analysis crew with and without acceleration."""

    if Agent is None:
        print(f"CrewAI integration unavailable: {IMPORT_ERROR}")
        return

    print("=== CrewAI Market Analysis Example ===\n")
    crew = build_crew(MARKET_ANALYSIS_PLAN)

    print("=== Standard Execution ===")
    start = time.perf_counter()
    standard_output = crew.kickoff()
    standard_time = time.perf_counter() - start
    print(str(standard_output)[:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    integration = accelerate_crew(crew).optimize({"maxParallelCalls": 4})
    start = time.perf_counter()
    accel_results = await integration.execute({})
    accel_time = time.perf_counter() - start
    print(str(accel_results["executive_summary"])[:500])
    print(f"\nAccelerated execution time: {accel_time:.2f} seconds")
    if standard_time > accel_time:
        improvement = ((standard_time - accel_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    asyncio.run(main())
