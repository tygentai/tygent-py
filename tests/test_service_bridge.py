from __future__ import annotations

import asyncio

from tygent.service_bridge import (
    DEFAULT_LLM_RUNTIME,
    ServicePlanBuilder,
    execute_service_plan,
)


def test_service_plan_builder_and_execution() -> None:
    calls = []

    async def mock_runtime(prompt: str, metadata, inputs):
        calls.append({"prompt": prompt, "metadata": dict(metadata), "inputs": inputs})
        return {"text": prompt.upper()}

    DEFAULT_LLM_RUNTIME.register("mock-provider", mock_runtime)

    payload = {
        "name": "ingested",
        "steps": [
            {
                "name": "discover",
                "kind": "llm",
                "prompt": "Research {topic}",
                "dependencies": [],
                "metadata": {"provider": "mock-provider"},
                "links": ["https://docs"],
            },
            {
                "name": "summary",
                "kind": "llm",
                "prompt": "Summarize {discover[result][text]}",
                "dependencies": ["discover"],
                "metadata": {"provider": "mock-provider", "token_estimate": 42},
            },
        ],
        "prefetch": {"links": ["https://docs"]},
    }

    builder = ServicePlanBuilder()
    service_plan = builder.build(payload)
    result = asyncio.run(execute_service_plan(service_plan, {"topic": "latency"}))

    assert "discover" in result["results"]
    assert "summary" in result["results"]
    # Ensure runtime executed with rendered prompt
    discover_call = calls[0]
    assert discover_call["prompt"] == "Research latency"
    # Prefetch results should be passed forward
    assert "prefetch" in discover_call["inputs"]
    assert discover_call["inputs"]["prefetch"]["https://docs"] == "prefetched"
    # Token estimate propagated
    assert service_plan.plan["steps"][1]["token_cost"] == 42
