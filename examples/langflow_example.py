"""
Example of using Tygent with Langflow.

This script calls the `example_langflow_acceleration` helper from the
Tygent Langflow integration to illustrate automatic acceleration of a
simple Langflow workflow.
"""

import asyncio
import sys

# Use local Tygent source if available
sys.path.append("./tygent-py")

try:
    from tygent.integrations.langflow import example_langflow_acceleration
except Exception as e:  # pragma: no cover
    example_langflow_acceleration = None
    IMPORT_ERROR = str(e)


if __name__ == "__main__":
    if example_langflow_acceleration is None:
        print(f"Langflow integration unavailable: {IMPORT_ERROR}")
    else:
        asyncio.run(example_langflow_acceleration())
