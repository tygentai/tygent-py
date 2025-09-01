"""LangGraph example generating a market analysis report with Gemini 2.5 Pro.

This script mirrors the Google ADK market analysis workflow but uses
LangGraph for orchestration. The model is prompted to build a DAG plan in
JSON, which is then executed either directly with LangGraph or accelerated via
Tygent's scheduler. Execution time for both paths is measured.

Requires ``langgraph``, ``langchain`` and ``langchain-google-genai`` packages:

``pip install langgraph langchain langchain-google-genai google-generativeai``

Set the ``GOOGLE_API_KEY`` environment variable before running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List

from tygent.accelerate import accelerate

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

try:  # pragma: no cover - optional dependency
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency
    print(
        "This example requires the langgraph, langchain, and langchain-google-genai packages."
    )
    print(
        "Install them with: pip install langgraph langchain langchain-google-genai google-generativeai"
    )
    raise SystemExit(1)

if not os.getenv("GOOGLE_API_KEY"):
    print("Set GOOGLE_API_KEY before running this example.")
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


def _parse_plan_json(text: str) -> Dict[str, Dict[str, Any]]:
    """Return a plan dictionary from raw model output."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Failed to parse plan JSON: {text}")


def _call_llm(llm: ChatGoogleGenerativeAI, name: str, prompt: str, log: bool) -> str:
    """Invoke the model and optionally log usage."""

    start = time.perf_counter()
    if log:
        logging.info("Starting %s", name)
    resp = llm.invoke(prompt)
    duration = time.perf_counter() - start
    text = resp.content
    if log:
        usage = (
            resp.response_metadata.get("token_usage", {})
            if hasattr(resp, "response_metadata")
            else {}
        )
        prompt_tokens = usage.get("prompt_tokens") or usage.get("promptTokenCount")
        response_tokens = usage.get("completion_tokens") or usage.get(
            "candidatesTokenCount"
        )
        if prompt_tokens is not None:
            logging.info(
                "%s finished in %.2fs: %s input tokens, %s output tokens",
                name,
                duration,
                prompt_tokens,
                response_tokens,
            )
        else:
            logging.info(
                "%s finished in %.2fs: token counts unavailable", name, duration
            )
    return text


def build_plan(llm: ChatGoogleGenerativeAI, log: bool) -> Dict[str, Dict[str, Any]]:
    """Ask the model to produce a DAG plan."""

    text = _call_llm(llm, "plan_builder", PLAN_GENERATION_PROMPT, log)
    return _parse_plan_json(text)


def execute_with_langgraph(
    plan: Dict[str, Dict[str, Any]],
    llm: ChatGoogleGenerativeAI,
    log: bool,
) -> Dict[str, str]:
    """Execute the plan using LangGraph and return node results."""

    workflow = StateGraph(dict)
    for name, node in plan.items():
        prompt = node["prompt"]

        def node_fn(state: Dict[str, str], name=name, prompt=prompt):
            return {name: _call_llm(llm, name, prompt.format(**state), log)}

        workflow.add_node(name, node_fn)

    roots = [n for n, v in plan.items() if not v["deps"]]
    workflow.add_node("start", lambda state: {})
    workflow.set_entry_point("start")
    for root in roots:
        workflow.add_edge("start", root)

    for name, node in plan.items():
        for dep in node["deps"]:
            workflow.add_edge(dep, name)

    leaves = {
        name for name in plan if not any(name in node["deps"] for node in plan.values())
    }
    for leaf in leaves:
        workflow.add_edge(leaf, END)

    app = workflow.compile()
    return app.invoke({})


def build_tygent_plan(
    plan: Dict[str, Dict[str, Any]],
    llm: ChatGoogleGenerativeAI,
    log: bool,
) -> Dict[str, Any]:
    """Return a Tygent plan equivalent to the provided DAG."""

    steps: List[Dict[str, Any]] = []
    for name, node in plan.items():
        prompt = node["prompt"]

        async def step(inputs: Dict[str, str], name=name, prompt=prompt):
            return _call_llm(llm, name, prompt.format(**inputs), log)

        steps.append({"name": name, "func": step, "dependencies": node["deps"]})

    return {"name": "market_intelligence", "steps": steps}


async def main(log_usage: bool = False) -> None:
    """Run the market analysis using LangGraph and Tygent."""

    logging.basicConfig(
        level=logging.INFO if log_usage else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", temperature=0.7, max_output_tokens=64000
    )

    print("=== LangGraph Google Market Intelligence Example ===\n")
    plan = build_plan(llm, log_usage)

    print("=== Standard Execution ===")
    start = time.perf_counter()
    results = execute_with_langgraph(plan, llm, log_usage)
    standard_time = time.perf_counter() - start
    print("Executive Summary:\n")
    print(results.get("executive_summary", "")[:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    accelerated = accelerate(build_tygent_plan(plan, llm, log_usage))
    start = time.perf_counter()
    accel_raw = await accelerated({})
    accel_results = accel_raw["results"]
    accel_time = time.perf_counter() - start
    print("Executive Summary:\n")
    print(accel_results.get("executive_summary", "")[:500])
    print(f"\nAccelerated execution time: {accel_time:.2f} seconds")
    if standard_time > accel_time:
        improvement = ((standard_time - accel_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    parser = argparse.ArgumentParser(
        description="LangGraph Google Market Intelligence Example"
    )
    parser.add_argument(
        "--log", action="store_true", help="Enable per-node execution logging"
    )
    args = parser.parse_args()
    asyncio.run(main(log_usage=args.log))
