"""Example of using Tygent with Google ADK for comprehensive market analysis.

Requires the `google-adk` and `google-genai` packages. Install them with:
`pip install google-adk google-genai` and configure authentication using either
an API key (`GOOGLE_API_KEY`) or Google Cloud service account credentials
(`GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, and
`GOOGLE_CLOUD_LOCATION`).

This script builds a directed acyclic graph (DAG) representing a multi-step
market intelligence research workflow. It uses Google's Agent Development Kit
(ADK) with an in-memory runner to execute each node in the workflow. The DAG
structure mirrors the prompt-based plan described in the documentation and
includes validation and synthesis stages before producing an executive summary.
"""

import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv

from tygent.accelerate import accelerate

load_dotenv()

try:  # pragma: no cover - optional dependency
    import google.genai as genai
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from tygent.integrations.google_adk import GoogleADKIntegration
except Exception:  # pragma: no cover - optional dependency
    print("This example requires the google-adk and google-genai packages.")
    print("Install them with: pip install google-adk google-genai")
    raise SystemExit(1)


API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    # Fall back to default credentials for Vertex AI / service account auth
    genai.configure()


INSTRUCTION = (
    "You are a strategic analyst preparing a comprehensive market intelligence "
    "report for executive leadership. Research emerging trends, competitive "
    "threats, customer behavior patterns, and market opportunities to guide "
    "major business decisions and investment strategies."
)


async def _create_integration() -> GoogleADKIntegration:
    """Set up the Google ADK integration with an LLM agent."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    agent = LlmAgent(
        name="analyst",
        model=model,
        instruction=INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7, max_output_tokens=150
        ),
    )
    runner = InMemoryRunner(agent)
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id="user", session_id="session"
    )
    integration = GoogleADKIntegration(runner)
    integration.optimize({"maxParallelCalls": 8})
    return integration


BASE_PROMPT = (
    "{task}: Analyze market opportunities, competitive landscape, customer trends, "
    "and regulatory environment across multiple industries to inform strategic "
    "business decisions."
)


def _build_dag(integration: GoogleADKIntegration) -> None:
    """Construct the DAG described in the market analysis prompt."""

    # Parallel Layer 1
    integration.add_node(
        "industry_analysis",
        BASE_PROMPT.format(task="Analyze industry trends and market dynamics"),
    )
    integration.add_node(
        "competitive_intelligence",
        BASE_PROMPT.format(task="Research competitor strategies and positioning"),
    )
    integration.add_node(
        "customer_research",
        BASE_PROMPT.format(task="Analyze customer behavior and preferences"),
    )
    integration.add_node(
        "regulatory_review",
        BASE_PROMPT.format(
            task="Review regulatory environment and compliance requirements"
        ),
    )
    integration.add_node(
        "trend_analysis",
        BASE_PROMPT.format(task="Identify emerging market trends and opportunities"),
    )
    integration.add_node(
        "expert_insights",
        BASE_PROMPT.format(task="Gather expert opinions and industry analysis"),
    )
    integration.add_node(
        "market_data",
        BASE_PROMPT.format(task="Process market size, growth, and forecast data"),
    )
    integration.add_node(
        "risk_assessment",
        BASE_PROMPT.format(task="Assess market risks and business threats"),
    )

    # Parallel Layer 2
    integration.add_node(
        "cross_validation",
        BASE_PROMPT.format(task="Cross-validate findings across research sources")
        + " Sources: {industry_analysis}, {competitive_intelligence}, {customer_research}, {regulatory_review}",
        dependencies=[
            "industry_analysis",
            "competitive_intelligence",
            "customer_research",
            "regulatory_review",
        ],
    )
    integration.add_node(
        "fact_verification",
        BASE_PROMPT.format(task="Verify strategic insights and market claims")
        + " Inputs: {trend_analysis}, {expert_insights}, {market_data}, {risk_assessment}",
        dependencies=[
            "trend_analysis",
            "expert_insights",
            "market_data",
            "risk_assessment",
        ],
    )

    # Sequential Layer 3
    integration.add_node(
        "credibility_assessment",
        BASE_PROMPT.format(task="Assess source credibility")
        + " References: {cross_validation}, {fact_verification}",
        dependencies=["cross_validation", "fact_verification"],
    )
    integration.add_node(
        "quality_check",
        BASE_PROMPT.format(task="Quality control and data validation")
        + " Review: {credibility_assessment}",
        dependencies=["credibility_assessment"],
    )

    # Parallel Layer 4
    integration.add_node(
        "strategic_analysis",
        BASE_PROMPT.format(
            task="Synthesize market intelligence into strategic insights"
        )
        + " Data: {quality_check}",
        dependencies=["quality_check"],
    )
    integration.add_node(
        "opportunity_identification",
        BASE_PROMPT.format(task="Identify strategic opportunities and recommendations")
        + " Data: {quality_check}",
        dependencies=["quality_check"],
    )

    # Final Layer 5
    integration.add_node(
        "executive_summary",
        BASE_PROMPT.format(task="Compile executive market intelligence report")
        + " Inputs: {strategic_analysis}, {opportunity_identification}",
        dependencies=["strategic_analysis", "opportunity_identification"],
    )


def _extract(events: Any) -> str:
    """Return the text content from a list of ADK events."""

    if not events:
        return ""
    event = events[0]
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", [])
    if parts:
        return parts[0].text
    return str(event)


async def main() -> None:
    """Execute the market analysis DAG with and without acceleration."""

    print("=== Google ADK Market Intelligence Example ===\n")
    integration = await _create_integration()
    _build_dag(integration)

    print("=== Standard Execution ===")
    start = asyncio.get_event_loop().time()
    results: Dict[str, Any] = await integration.execute({})
    standard_time = asyncio.get_event_loop().time() - start
    summary = _extract(results["executive_summary"])
    print("Executive Summary:\n")
    print(summary[:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    accelerate(integration.runner)
    start = asyncio.get_event_loop().time()
    accel_results: Dict[str, Any] = await integration.execute({})
    accel_time = asyncio.get_event_loop().time() - start
    accel_summary = _extract(accel_results["executive_summary"])
    print("Executive Summary:\n")
    print(accel_summary[:500])
    print(f"\nAccelerated execution time: {accel_time:.2f} seconds")
    if standard_time > accel_time:
        improvement = ((standard_time - accel_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":
    asyncio.run(main())
