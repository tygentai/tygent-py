"""Example of using Google ADK for comprehensive market analysis.

Requires the ``google-adk`` and ``google-genai`` packages. Install them with:
``pip install google-adk google-genai`` and configure authentication using
either an API key (``GOOGLE_API_KEY``) or Google Cloud service account
credentials (``GOOGLE_APPLICATION_CREDENTIALS``, ``GOOGLE_CLOUD_PROJECT``, and
``GOOGLE_CLOUD_LOCATION``).

This script builds a directed acyclic graph (DAG) representing a multi-step
market intelligence research workflow using Google's Agent Development Kit
(ADK). Instead of using a hard-coded plan, the model is prompted to generate
the plan as JSON which is then optimized and executed via Tygent. The workflow
includes validation and synthesis stages before producing an executive
summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
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


PLAN_GENERATION_PROMPT = (
    "Generate a comprehensive market intelligence research plan as a JSON object representing a DAG. "
    "The JSON should map step names to objects containing 'prompt' and 'deps' (a list of dependencies). "
    'Use the prompt template "'
    + BASE_PROMPT
    + '" filling in an appropriate task for each step. '
    "Cover industry, competitor, customer, regulatory, trend, expert insight, market data, risk, validation, and synthesis "
    "phases, culminating in an executive_summary. Each prompt should be concise and suitable for direct model invocation. "
    "Return only valid JSON."
)


def _parse_plan_json(text: str) -> Dict[str, Any]:
    """Parse JSON that may be wrapped in Markdown code fences."""

    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - example fallback
        raise ValueError(f"Failed to parse plan JSON: {text}") from exc


async def build_plan(
    runner: InMemoryRunner, log_usage: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Ask the model to produce a DAG plan and return it as a dictionary."""

    text = await _call_runner(runner, "plan_builder", PLAN_GENERATION_PROMPT, log_usage)
    return _parse_plan_json(text)


async def _create_runner() -> InMemoryRunner:
    agent = LlmAgent(
        name="analyst",
        model="gemini-2.5-pro",
        instruction=INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7, max_output_tokens=64000
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


async def _call_runner(
    runner: InMemoryRunner, name: str, prompt: str, log_usage: bool = False
) -> str:
    start = asyncio.get_event_loop().time()
    if log_usage:
        logger.info("Starting %s", name)
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
    duration = asyncio.get_event_loop().time() - start
    text = _extract(events)
    if log_usage:
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            logger.info(
                "%s finished in %.2fs: %s input tokens, %s output tokens",
                name,
                duration,
                prompt_tokens,
                response_tokens,
            )
        else:
            logger.info(
                "%s finished in %.2fs: token counts unavailable", name, duration
            )
    return text


async def execute_plan(
    plan: Dict[str, Dict[str, Any]],
    runner: InMemoryRunner,
    log_usage: bool = False,
) -> Dict[str, str]:
    """Execute the provided DAG sequentially in defined order."""

    results: Dict[str, str] = {}
    for name, node in plan.items():
        results[name] = await _call_runner(
            runner,
            name,
            node["prompt"].format(**results),
            log_usage,
        )
    return results


def build_tygent_plan(
    plan: Dict[str, Dict[str, Any]],
    runner: InMemoryRunner,
    log_usage: bool,
) -> Dict[str, Any]:
    """Build a Tygent-compatible plan from the provided DAG."""

    steps: List[Dict[str, Any]] = []
    for name, node in plan.items():
        prompt = node["prompt"]

        async def step(inputs: Dict[str, str], name=name, prompt=prompt):
            return await _call_runner(
                runner,
                name,
                prompt.format(**inputs),
                log_usage,
            )

        steps.append({"name": name, "func": step, "dependencies": node["deps"]})

    return {"name": "market_intelligence", "steps": steps}


async def main(log_usage: bool = False) -> None:
    """Execute the market analysis DAG with and without acceleration."""
    logging.basicConfig(
        level=logging.INFO if log_usage else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=== Google ADK Market Intelligence Example ===\n")
    runner = await _create_runner()
    plan = await build_plan(runner, log_usage)

    print("=== Standard Execution ===")
    start = asyncio.get_event_loop().time()
    results = await execute_plan(plan, runner, log_usage)
    standard_time = asyncio.get_event_loop().time() - start
    print("Executive Summary:\n")
    print(results["executive_summary"][:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    accelerated_plan = accelerate(build_tygent_plan(plan, runner, log_usage))
    start = asyncio.get_event_loop().time()
    accel_raw = await accelerated_plan({})
    accel_results = accel_raw["results"]
    accel_time = asyncio.get_event_loop().time() - start
    print("Executive Summary:\n")
    print(accel_results["executive_summary"][:500])
    print(f"\nAccelerated execution time: {accel_time:.2f} seconds")
    if standard_time > accel_time:
        improvement = ((standard_time - accel_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    parser = argparse.ArgumentParser(
        description="Google ADK Market Intelligence Example"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable per-node logging of execution time and token usage",
    )
    args = parser.parse_args()
    asyncio.run(main(log_usage=args.log))
