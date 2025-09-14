import asyncio
import time

from tygent import accelerate


async def step_a(inputs):
    await asyncio.sleep(0.1)
    return {"a": 1}


async def step_b(inputs):
    await asyncio.sleep(0.1)
    return {"b": 1}


plan = {
    "name": "perf",
    "steps": [
        {"name": "a", "func": step_a},
        {"name": "b", "func": step_b},
    ],
}


async def run_sequential():
    await step_a({})
    await step_b({})


def test_accelerated_plan_faster_than_sequential():
    start = time.perf_counter()
    asyncio.run(run_sequential())
    sequential = time.perf_counter() - start

    accelerated = accelerate(plan)
    start = time.perf_counter()
    asyncio.run(accelerated({}))
    parallel = time.perf_counter() - start

    assert parallel < sequential
