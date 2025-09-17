"""Benchmark ensuring conversion to Tygent improves latency without extra tokens."""

import asyncio
import time
from typing import Dict, List, Tuple, Type

import pytest

from tygent import accelerate
from tygent.testing.planning_examples import (
    CrewAIReasoningAgent,
    LangGraphReasoningAgent,
    NemoReasoningAgent,
    OpenAIReasoningAgent,
)

AGENT_CLASSES: List[Type] = [
    OpenAIReasoningAgent,
    LangGraphReasoningAgent,
    CrewAIReasoningAgent,
    NemoReasoningAgent,
]

CUSTOMER_SCENARIO: Dict[str, object] = {
    "issue": "Shipment delayed past the promised arrival window",
    "customer_profile": {"tier": "Gold", "previous_orders": 7},
}


def _run_baseline(agent, inputs: Dict[str, object]) -> Tuple[Dict[str, object], float]:
    start = time.perf_counter()
    result = asyncio.run(agent.execute(inputs))
    duration = time.perf_counter() - start
    return result, duration


def _run_accelerated(
    agent, inputs: Dict[str, object]
) -> Tuple[Dict[str, object], float]:
    accelerated = accelerate(agent)
    start = time.perf_counter()
    result = asyncio.run(accelerated.execute(inputs))
    duration = time.perf_counter() - start
    return result, duration


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_conversion_speedup_and_token_parity(agent_cls):
    agent = agent_cls()

    baseline_result, baseline_latency = _run_baseline(agent, CUSTOMER_SCENARIO)
    assert "tokens_used" in baseline_result
    assert baseline_latency > 0

    accelerated_result, accelerated_latency = _run_accelerated(agent, CUSTOMER_SCENARIO)

    # Ensure outputs match exactly after conversion.
    assert accelerated_result["results"] == baseline_result["results"]

    baseline_tokens = int(baseline_result["tokens_used"])
    accelerated_tokens = sum(
        step.get("tokens", 0) for step in accelerated_result["results"].values()
    )

    assert accelerated_tokens == baseline_tokens

    # Parallel execution should be noticeably faster thanks to Tygent.
    assert accelerated_latency < baseline_latency
    assert accelerated_latency <= baseline_latency * 0.8
