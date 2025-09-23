"""CrewAI counterpart to the LangChain market intelligence example.

This script mirrors the dynamic market research workflow from
``examples/langchain_market_analysis.py`` but maps the generated plan onto a
CrewAI-style task graph. It first executes the workflow sequentially using the
Crew agent's ``execute_task`` method before running an accelerated version via
Tygent's CrewAI integration.

Dependencies
------------
- ``crewai`` (``pip install crewai``)
- ``google-generativeai`` (``pip install google-generativeai``)
- ``python-dotenv`` (optional, for ``.env`` loading)
- A ``GOOGLE_API_KEY`` environment variable for Gemini access
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import re
from types import MethodType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

try:  # pragma: no cover - optional dependency
    from crewai import Agent, Crew, Process, Task
except Exception:  # pragma: no cover - optional dependency
    print("This example requires the crewai package. Install it with: pip install crewai")
    raise SystemExit(1)

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    print("This example requires the google-generativeai package. Install it with: pip install google-generativeai")
    raise SystemExit(1)

from tygent.integrations.crewai import accelerate_crew

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Set the GOOGLE_API_KEY environment variable before running this example.")
    raise SystemExit(1)

genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


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
    model: Any, name: str, prompt: str, log_usage: bool = False
) -> str:
    """Invoke Gemini directly and return the text response."""

    loop = asyncio.get_event_loop()
    start = loop.time()
    if log_usage:
        logger.info("Starting %s", name)

    try:
        if hasattr(model, "generate_content_async"):
            response = await model.generate_content_async(prompt)
        else:
            response = await loop.run_in_executor(None, model.generate_content, prompt)
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.exception("%s failed: %s", name, exc)
        raise

    text = getattr(response, "text", None)
    if not text:
        parts: List[str] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                piece = getattr(part, "text", None)
                if piece:
                    parts.append(piece)
        if not parts and hasattr(response, "content"):
            parts.append(str(response.content))
        text = "\n".join(parts)

    duration = loop.time() - start
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


async def build_plan(model: Any, log_usage: bool = False) -> Dict[str, Dict[str, Any]]:
    """Ask the model to produce a DAG plan and return it as a dictionary."""

    text = await _call_llm(model, "plan_builder", PLAN_GENERATION_PROMPT, log_usage)
    return _parse_plan_json(text)


def _filter_kwargs(params: Mapping[str, inspect.Parameter], candidates: Mapping[str, Any]) -> Dict[str, Any]:
    """Return kwargs supported by a callable signature."""

    return {key: value for key, value in candidates.items() if key in params}


def _task_name(task: Task) -> str:
    """Infer a stable task identifier."""

    for attr in ("id", "name", "task_id"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    return f"task_{id(task)}"


def _topological_order(plan: Mapping[str, Dict[str, Any]]) -> List[str]:
    """Return a safe execution order respecting declared dependencies."""

    indegree: Dict[str, int] = {name: 0 for name in plan}
    adjacency: Dict[str, List[str]] = {name: [] for name in plan}

    for name, node in plan.items():
        for dep in node.get("deps", []) or []:
            if dep in plan:
                indegree[name] += 1
                adjacency.setdefault(dep, []).append(name)

    queue: List[str] = [name for name, count in indegree.items() if count == 0]
    order: List[str] = []

    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in adjacency.get(current, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(order) != len(plan):  # Fallback to declared order if cycle detected
        return list(plan.keys())
    return order


def _create_agent(model: Any, log_usage: bool) -> Agent:
    """Instantiate a CrewAI agent and patch it with an async executor."""

    agent_sig = inspect.signature(Agent)
    base_kwargs = {
        "role": "Strategic Market Analyst",
        "goal": "Transform fragmented research into executive-ready market intelligence.",
        "backstory": (
            "A veteran analyst with deep expertise in competitive research, regulatory mapping, "
            "and customer insight synthesis across global markets."
        ),
        "allow_delegation": False,
        "verbose": log_usage,
    }
    kwargs = _filter_kwargs(agent_sig.parameters, base_kwargs)
    agent = Agent(**kwargs)

    object.__setattr__(agent, "_tygent_model", model)
    object.__setattr__(agent, "_tygent_log_usage", log_usage)

    async def _execute_task(self: Agent, task: Task, context: MutableMapping[str, Any] | None = None) -> Any:
        data: Dict[str, Any] = {}
        if context:
            data.update(context)
        prompt_template = getattr(task, "prompt_template", getattr(task, "description", ""))
        try:
            prompt = prompt_template.format(**data)
        except Exception:
            prompt = prompt_template
        return await _call_llm(
            getattr(self, "_tygent_model"),
            _task_name(task),
            prompt,
            bool(getattr(self, "_tygent_log_usage", False)),
        )

    object.__setattr__(agent, "execute_task", MethodType(_execute_task, agent))
    if not getattr(agent, "name", None):
        object.__setattr__(agent, "name", "market_intelligence_agent")
    return agent


def _create_tasks(plan: Mapping[str, Dict[str, Any]], agent: Agent) -> Dict[str, Task]:
    """Convert the JSON plan into CrewAI ``Task`` objects."""

    task_sig = inspect.signature(Task)
    tasks: Dict[str, Task] = {}

    for name, node in plan.items():
        base_kwargs = {
            "description": node["prompt"],
            "agent": agent,
            "expected_output": "Concise research findings ready for executive synthesis.",
            "async_execution": False,
            "name": name,
        }
        kwargs = _filter_kwargs(task_sig.parameters, base_kwargs)
        task = Task(**kwargs)
        object.__setattr__(task, "id", name)
        object.__setattr__(task, "prompt_template", node["prompt"])
        object.__setattr__(task, "agent", agent)
        tasks[name] = task

    for name, node in plan.items():
        dependencies = [tasks[dep] for dep in node.get("deps", []) if dep in tasks]
        task = tasks[name]
        object.__setattr__(task, "dependencies", dependencies)
        if hasattr(task, "context"):
            object.__setattr__(task, "context", dependencies)

    return tasks


def _create_crew(tasks: Mapping[str, Task], order: Iterable[str], agent: Agent, log_usage: bool) -> Crew:
    """Instantiate a Crew container for the generated tasks."""

    crew_sig = inspect.signature(Crew)
    task_list = [tasks[name] for name in order]
    base_kwargs = {
        "agents": [agent],
        "tasks": task_list,
        "name": "market_intelligence",
        "process": Process.sequential,
        "verbose": log_usage,
    }
    kwargs = _filter_kwargs(crew_sig.parameters, base_kwargs)
    crew = Crew(**kwargs)
    if not getattr(crew, "tasks", None):
        object.__setattr__(crew, "tasks", task_list)
    if not getattr(crew, "agents", None):
        object.__setattr__(crew, "agents", [agent])
    return crew


def _build_context(
    inputs: Mapping[str, Any],
    prior_results: Mapping[str, Any],
    task: Task,
) -> Dict[str, Any]:
    """Merge base inputs with dependency outputs for prompt templating."""

    context: Dict[str, Any] = dict(inputs)
    for dep in getattr(task, "dependencies", []) or []:
        dep_name = _task_name(dep)
        if dep_name in prior_results:
            value = prior_results[dep_name]
            context[dep_name] = value
            if isinstance(value, Mapping):
                for key, val in value.items():
                    context.setdefault(str(key), val)
            else:
                context.setdefault(f"{dep_name}_text", value)
    for key, value in prior_results.items():
        if key not in context:
            context[key] = value
    context.pop(_task_name(task), None)
    return context


async def execute_plan_sequential(
    order: Iterable[str],
    tasks: Mapping[str, Task],
    agent: Agent,
    inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Execute the plan sequentially using the Crew agent."""

    results: Dict[str, Any] = dict(inputs)
    for name in order:
        task = tasks[name]
        context = _build_context(inputs, results, task)
        result = await agent.execute_task(task, context)
        results[name] = result
    return results


async def main(log_usage: bool = False) -> None:
    """Execute the market analysis workflow sequentially and with acceleration."""

    logging.basicConfig(
        level=logging.INFO if log_usage else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=== CrewAI Market Intelligence Example ===\n")
    model = genai.GenerativeModel(MODEL_NAME)

    plan = await build_plan(model, log_usage)
    order = _topological_order(plan)

    agent = _create_agent(model, log_usage)
    tasks = _create_tasks(plan, agent)
    crew = _create_crew(tasks, order, agent, log_usage)

    base_inputs = {"instruction": INSTRUCTION}

    print("=== Standard Execution ===")
    start = asyncio.get_event_loop().time()
    sequential_results = await execute_plan_sequential(order, tasks, agent, base_inputs)
    standard_time = asyncio.get_event_loop().time() - start
    summary = sequential_results.get("executive_summary", "No executive summary produced.")
    print("Executive Summary:\n")
    print(str(summary)[:500])
    print(f"\nStandard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    accelerated = accelerate_crew(crew)
    start = asyncio.get_event_loop().time()
    accelerated_raw = await accelerated.execute(base_inputs)
    accel_results = accelerated_raw.get("results", {})
    accel_summary = accel_results.get("executive_summary", "No executive summary produced.")
    accel_time = asyncio.get_event_loop().time() - start
    print("Executive Summary:\n")
    print(str(accel_summary)[:500])
    print(f"\nAccelerated execution time: {accel_time:.2f} seconds")
    if standard_time > accel_time > 0:
        improvement = ((standard_time - accel_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    parser = argparse.ArgumentParser(
        description="CrewAI Market Intelligence Example",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable per-node logging of execution time",
    )
    args = parser.parse_args()
    asyncio.run(main(log_usage=args.log))
