"""Synthetic planning agents used for benchmarking Tygent conversions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List


@dataclass
class StepSpec:
    """Description for a single reasoning step."""

    name: str
    delay: float
    tokens: int
    dependencies: Iterable[str]
    builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    critical: bool = False
    func: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]] = field(init=False)

    def __post_init__(self) -> None:
        async def _func(inputs: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(self.delay)
            payload = dict(self.builder(inputs))
            payload.setdefault("tokens", self.tokens)
            return payload

        self.func = _func

    @property
    def dependencies_list(self) -> List[str]:
        return list(self.dependencies)

    def as_plan_entry(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "name": self.name,
            "func": self.func,
            "token_cost": self.tokens,
        }
        if self.dependencies:
            entry["dependencies"] = list(self.dependencies)
        if self.critical:
            entry["critical"] = True
        return entry


class BasePlanningAgent:
    """Common helpers shared by all synthetic planning agents."""

    def __init__(self, name: str, steps: List[StepSpec]):
        self._name = name
        self._steps = steps

    def workflow(self) -> Dict[str, Any]:
        """Return a plan dictionary consumable by :func:`tygent.accelerate`."""

        return {
            "name": self._name,
            "steps": [step.as_plan_entry() for step in self._steps],
        }

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sequentially execute the plan while tracking token usage."""

        results: Dict[str, Dict[str, Any]] = {}
        total_tokens = 0

        for step in self._steps:
            dependency_payload: Dict[str, Any] = {}
            for dep in step.dependencies_list:
                dep_result = results[dep]
                if isinstance(dep_result, dict):
                    dependency_payload.update(dep_result)
                else:
                    dependency_payload[dep] = dep_result

            combined_inputs = {**inputs, **dependency_payload}
            output = await step.func(combined_inputs)
            results[step.name] = output
            total_tokens += int(output.get("tokens", step.tokens))

        return {"results": results, "tokens_used": total_tokens}


def _outline_builder(framework: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def builder(inputs: Dict[str, Any]) -> Dict[str, Any]:
        issue = inputs["issue"]
        tier = inputs["customer_profile"]["tier"]
        return {
            "outline": (
                f"{framework} reasoning plan for '{issue}' with priority "
                f"support to the {tier} tier customer."
            ),
            "focus_points": [
                "Acknowledge the delay",
                "Reassure about next steps",
                "Offer proactive follow up",
            ],
        }

    return builder


def _policy_builder(framework: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def builder(inputs: Dict[str, Any]) -> Dict[str, Any]:
        outline = inputs["outline"]
        return {
            "policy_context": (
                f"{framework} policy lookup referencing outline: {outline}."
            ),
            "next_steps": [
                "Refund eligibility check",
                "Expedited shipping availability",
            ],
        }

    return builder


def _history_builder(framework: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def builder(inputs: Dict[str, Any]) -> Dict[str, Any]:
        profile = inputs["customer_profile"]
        return {
            "history_notes": (
                f"{framework} summarized history with {profile['previous_orders']} orders."
            ),
            "loyalty_signal": (
                "high" if profile["tier"].lower() == "gold" else "standard"
            ),
        }

    return builder


def _draft_builder(framework: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def builder(inputs: Dict[str, Any]) -> Dict[str, Any]:
        policy = inputs["policy_context"]
        history = inputs["history_notes"]
        return {
            "draft_reply": (
                f"{framework} draft combining policy insight '{policy}' "
                f"with customer context '{history}'."
            ),
        }

    return builder


def _review_builder(framework: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def builder(inputs: Dict[str, Any]) -> Dict[str, Any]:
        draft = inputs["draft_reply"]
        return {
            "final_reply": f"{framework} QA approved: {draft}",
        }

    return builder


def _build_reasoning_steps(framework: str) -> List[StepSpec]:
    return [
        StepSpec(
            name="plan_outline",
            delay=0.05,
            tokens=120,
            dependencies=[],
            builder=_outline_builder(framework),
            critical=True,
        ),
        StepSpec(
            name="policy_retrieval",
            delay=0.12,
            tokens=220,
            dependencies=["plan_outline"],
            builder=_policy_builder(framework),
        ),
        StepSpec(
            name="history_analysis",
            delay=0.12,
            tokens=180,
            dependencies=["plan_outline"],
            builder=_history_builder(framework),
        ),
        StepSpec(
            name="compose_reply",
            delay=0.08,
            tokens=200,
            dependencies=["policy_retrieval", "history_analysis"],
            builder=_draft_builder(framework),
        ),
        StepSpec(
            name="quality_review",
            delay=0.04,
            tokens=60,
            dependencies=["compose_reply"],
            builder=_review_builder(framework),
        ),
    ]


class OpenAIReasoningAgent(BasePlanningAgent):
    def __init__(self) -> None:
        super().__init__("openai_sdk_support", _build_reasoning_steps("OpenAI SDK"))


class LangGraphReasoningAgent(BasePlanningAgent):
    def __init__(self) -> None:
        super().__init__("langgraph_support", _build_reasoning_steps("LangGraph"))


class CrewAIReasoningAgent(BasePlanningAgent):
    def __init__(self) -> None:
        super().__init__("crewai_support", _build_reasoning_steps("CrewAI"))


class NemoReasoningAgent(BasePlanningAgent):
    def __init__(self) -> None:
        super().__init__("nemo_support", _build_reasoning_steps("NeMo"))
