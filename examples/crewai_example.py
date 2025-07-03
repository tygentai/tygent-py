"""
Example of using Tygent with CrewAI.

This script demonstrates how to leverage the `example_crewai_acceleration`
function provided by the Tygent CrewAI integration module. It simply imports
and executes the helper to showcase how CrewAI workflows can be accelerated.
"""

import asyncio

try:
    from tygent.integrations.crewai import example_crewai_acceleration
except Exception as e:  # pragma: no cover - optional dependency
    example_crewai_acceleration = None
    IMPORT_ERROR = str(e)


if __name__ == "__main__":
    if example_crewai_acceleration is None:
        print(f"CrewAI integration unavailable: {IMPORT_ERROR}")
    else:
        asyncio.run(example_crewai_acceleration())
