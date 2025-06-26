import asyncio
from statistics import mean

from examples.dag_example import run_scheduler, run_sequential


async def benchmark(runs: int = 3) -> None:
    seq_times = []
    sched_times = []
    for _ in range(runs):
        seq_times.append(await run_sequential())
        sched_times.append(await run_scheduler())
    print(f"Sequential mean: {mean(seq_times):.3f}s")
    print(f"Scheduler mean: {mean(sched_times):.3f}s")


if __name__ == "__main__":
    asyncio.run(benchmark())
