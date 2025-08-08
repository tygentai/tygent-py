"""Example of using Google ADK for comprehensive market analysis.

Requires the ``google-adk`` and ``google-genai`` packages. Install them with:
``pip install google-adk google-genai`` and configure authentication using
either an API key (``GOOGLE_API_KEY``) or Google Cloud service account
credentials (``GOOGLE_APPLICATION_CREDENTIALS``, ``GOOGLE_CLOUD_PROJECT``, and
``GOOGLE_CLOUD_LOCATION``).

This script builds a directed acyclic graph (DAG) representing a multi-step
market intelligence research workflow using Google's Agent Development Kit
(ADK). The DAG mirrors the prompt-based plan described in the documentation and
includes validation and synthesis stages before producing an executive
summary.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

from tygent.accelerate import accelerate

try:  # pragma: no cover - optional dependency
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency
    print("This example requires the google-adk and google-genai packages.")
    print("Install them with: pip install google-adk google-genai")
    raise SystemExit(1)

if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print(
        "Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS before running this example."
    )
    raise SystemExit(1)


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

PLAN: Dict[str, Dict[str, Any]] = {
    "industry_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Analyze industry trends and market dynamics"
        ),
        "deps": [],
    },
    "competitive_intelligence": {
        "prompt": BASE_PROMPT.format(
            task="Research competitor strategies and positioning"
        ),
        "deps": [],
    },
    "customer_research": {
        "prompt": BASE_PROMPT.format(task="Analyze customer behavior and preferences"),
        "deps": [],
    },
    "regulatory_review": {
        "prompt": BASE_PROMPT.format(
            task="Review regulatory environment and compliance requirements"
        ),
        "deps": [],
    },
    "trend_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Identify emerging market trends and opportunities"
        ),
        "deps": [],
    },
    "expert_insights": {
        "prompt": BASE_PROMPT.format(
            task="Gather expert opinions and industry analysis"
        ),
        "deps": [],
    },
    "market_data": {
        "prompt": BASE_PROMPT.format(
            task="Process market size, growth, and forecast data"
        ),
        "deps": [],
    },
    "risk_assessment": {
        "prompt": BASE_PROMPT.format(task="Assess market risks and business threats"),
        "deps": [],
    },
    "cross_validation": {
        "prompt": BASE_PROMPT.format(
            task="Cross-validate findings across research sources"
        )
        + " Sources: {industry_analysis}, {competitive_intelligence}, {customer_research}, {regulatory_review}",
        "deps": [
            "industry_analysis",
            "competitive_intelligence",
            "customer_research",
            "regulatory_review",
        ],
    },
    "fact_verification": {
        "prompt": BASE_PROMPT.format(task="Verify strategic insights and market claims")
        + " Inputs: {trend_analysis}, {expert_insights}, {market_data}, {risk_assessment}",
        "deps": ["trend_analysis", "expert_insights", "market_data", "risk_assessment"],
    },
    "credibility_assessment": {
        "prompt": BASE_PROMPT.format(task="Assess source credibility")
        + " References: {cross_validation}, {fact_verification}",
        "deps": ["cross_validation", "fact_verification"],
    },
    "quality_check": {
        "prompt": BASE_PROMPT.format(task="Quality control and data validation")
        + " Review: {credibility_assessment}",
        "deps": ["credibility_assessment"],
    },
    "strategic_analysis": {
        "prompt": BASE_PROMPT.format(
            task="Synthesize market intelligence into strategic insights"
        )
        + " Data: {quality_check}",
        "deps": ["quality_check"],
    },
    "opportunity_identification": {
        "prompt": BASE_PROMPT.format(
            task="Identify strategic opportunities and recommendations"
        )
        + " Data: {quality_check}",
        "deps": ["quality_check"],
    },
    "executive_summary": {
        "prompt": BASE_PROMPT.format(
            task="Compile executive market intelligence report"
        )
        + " Inputs: {strategic_analysis}, {opportunity_identification}",
        "deps": ["strategic_analysis", "opportunity_identification"],
    },
}


async def _create_runner() -> InMemoryRunner:
    agent = LlmAgent(
        name="analyst",
        model="gemini-1.5-flash",
        instruction=INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7, max_output_tokens=150
        ),
    )
    runner = InMemoryRunner(agent)
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id="user", session_id="session"
    )
    return runner


def _extract(events: List[Any]) -> str:
    """Return the text content from a list of ADK events."""

    if not events:
        return ""
    event = events[0]
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", [])
    if parts:
        return parts[0].text
    return str(event)


async def _call_runner(runner: InMemoryRunner, name: str, prompt: str) -> str:
    events: List[Any] = []
    usage = None
    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        events.append(event)
        if usage is None:
            usage = getattr(event, "usage_metadata", None)
    text = _extract(events)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        response_tokens = getattr(usage, "candidates_token_count", None)
        logger.info(
            "%s: %s input tokens, %s output tokens",
            name,
            prompt_tokens,
            response_tokens,
        )
    else:
        logger.info("%s: token counts unavailable", name)
    return text


async def execute_plan(runner: InMemoryRunner) -> Dict[str, str]:
    """Execute the predefined DAG and return node outputs."""

    results: Dict[str, str] = {}
    pending = {name: set(node.get("deps", [])) for name, node in PLAN.items()}
    while len(results) < len(PLAN):
        ready = [
            name
            for name, deps in pending.items()
            if name not in results and deps.issubset(results.keys())
        ]
        tasks = {
            name: asyncio.create_task(
                _call_runner(runner, name, PLAN[name]["prompt"].format(**results))
            )
            for name in ready
        }
        outputs = await asyncio.gather(*tasks.values())
        for name, text in zip(tasks.keys(), outputs):
            results[name] = text
    return results


async def main() -> None:
    """Execute the market analysis DAG with and without acceleration."""
    logging.basicConfig(level=logging.INFO)

    print("=== Google ADK Market Intelligence Example ===\n")
    runner = await _create_runner()

    print("=== Standard Execution ===")
    start = asyncio.get_event_loop().time()
    results = await execute_plan(runner)
    standard_time = asyncio.get_event_loop().time() - start
    print("Executive Summary:\n")
    print(results["executive_summary"][:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    accelerate(runner)
    start = asyncio.get_event_loop().time()
    accel_results = await execute_plan(runner)
    accel_time = asyncio.get_event_loop().time() - start
    print("Executive Summary:\n")
    print(accel_results["executive_summary"][:500])
    print(f"\nAccelerated execution time: {accel_time:.2f} seconds")
    if standard_time > accel_time:
        improvement = ((standard_time - accel_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    asyncio.run(main())
