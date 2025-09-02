"""Example of using LangChain for comprehensive market analysis.

Requires the ``langchain`` and ``langchain-google-genai`` packages. Install
them with:
``pip install langchain langchain-google-genai`` and set the
``GOOGLE_API_KEY`` environment variable.

This script builds a directed acyclic graph (DAG) representing a multi-step
market intelligence research workflow. Instead of a hard-coded plan, the model
is prompted to generate the plan as JSON which is then optimized and executed
via Tygent. The workflow includes validation and synthesis stages before
producing an executive summary.
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
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    print("This example requires the langchain and langchain-google-genai packages.")
    print("Install them with: pip install langchain langchain-google-genai")
    raise SystemExit(1)

if not os.getenv("GOOGLE_API_KEY"):
    print("Set the GOOGLE_API_KEY environment variable before running this example.")
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
    "Respond with valid JSON only—without markdown or additional text."
)


async def _call_llm(
    llm: ChatGoogleGenerativeAI, name: str, prompt: str, log_usage: bool = False
) -> str:
    """Invoke the LLM and return the text response."""

    start = asyncio.get_event_loop().time()
    if log_usage:
        logger.info("Starting %s", name)
    message = await llm.ainvoke(prompt)
    text = message.content if isinstance(message.content, str) else str(message.content)
    duration = asyncio.get_event_loop().time() - start
    if log_usage:
        logger.info("%s finished in %.2fs", name, duration)
    return text


def _parse_plan_json(text: str) -> Dict[str, Dict[str, Any]]:
    """Return a plan dictionary from raw model output."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Failed to parse plan JSON: {text}")


async def build_plan(
    llm: ChatGoogleGenerativeAI, log_usage: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Ask the model to produce a DAG plan and return it as a dictionary."""

    text = await _call_llm(llm, "plan_builder", PLAN_GENERATION_PROMPT, log_usage)
    return _parse_plan_json(text)


async def execute_plan(
    plan: Dict[str, Dict[str, Any]],
    llm: ChatGoogleGenerativeAI,
    log_usage: bool = False,
) -> Dict[str, str]:
    """Execute the provided DAG sequentially in defined order."""

    results: Dict[str, str] = {}
    for name, node in plan.items():
        results[name] = await _call_llm(
            llm,
            name,
            node["prompt"].format(**results),
            log_usage,
        )
    return results


async def main(log_usage: bool = False) -> None:
    """Execute the market analysis DAG with and without acceleration."""

    logging.basicConfig(
        level=logging.INFO if log_usage else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=== LangChain Market Intelligence Example ===\n")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
    plan = await build_plan(llm, log_usage)

    print("=== Standard Execution ===")
    start = asyncio.get_event_loop().time()
    results = await execute_plan(plan, llm, log_usage)
    standard_time = asyncio.get_event_loop().time() - start
    print("Executive Summary:\n")
    print(results["executive_summary"][:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    steps: List[Dict[str, Any]] = []
    for name, node in plan.items():
        prompt = node["prompt"]

        async def step(inputs: Dict[str, str], name=name, prompt=prompt):
            return await _call_llm(
                llm,
                name,
                prompt.format(**inputs),
                log_usage,
            )

        steps.append({"name": name, "func": step, "dependencies": node["deps"]})

    accelerated_plan = accelerate({"name": "market_intelligence", "steps": steps})
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
        description="LangChain Market Intelligence Example",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable per-node logging of execution time",
    )
    args = parser.parse_args()
    asyncio.run(main(log_usage=args.log))
