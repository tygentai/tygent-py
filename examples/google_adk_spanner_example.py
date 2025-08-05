"""Example using Google ADK with and without Tygent acceleration.

This script creates a simple Google ADK agent that answers the query:
"Describe the Spanner paper by Google assuming I am a 10th grader, a junior undergraduate, and a first year graduate student."
It then executes the query twice – once using standard execution and once
with Tygent acceleration applied – printing only the execution times and
overall speedup.
"""

import asyncio
import os
import time

from dotenv import load_dotenv

# Load environment variables from a .env file if present. The file should
# include settings like GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and
# GOOGLE_APPLICATION_CREDENTIALS as shown in the ADK samples repository.
load_dotenv()

try:
    from google import genai
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except ImportError:
    print("This example requires the google-adk and google-genai packages.")
    print("Install them with: pip install google-adk google-genai")
    raise

from tygent import accelerate

# Ensure required environment variables for Vertex AI are set
required_env_vars = [
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
]
missing = [var for var in required_env_vars if not os.getenv(var)]
if missing:
    print("Missing required environment variables: " + ", ".join(missing))
    raise SystemExit

# Configure Google GenAI client using application default credentials
client = genai.Client()

# Create a simple LLM agent capable of explaining topics at different levels
model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-pro")
tag = LlmAgent(
    name="spanner_explainer",
    model=model_name,
    instruction=(
        "You explain technical topics for different levels of expertise."
        " Provide clear and concise responses."
    ),
)

QUERY = (
    "Describe the Spanner paper by Google assuming I am a 10th grader, "
    "a junior undergraduate, and a first year graduate student."
)


async def run_query(runner: InMemoryRunner) -> None:
    """Execute the query with the provided runner."""
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id="user", session_id="session"
    )
    content = types.Content(role="user", parts=[types.Part(text=QUERY)])
    async for _ in runner.run_async(
        user_id="user", session_id="session", new_message=content
    ):
        pass


async def main():
    # Create runners up front so initialization time isn't measured
    baseline_runner = InMemoryRunner(tag)
    accelerated_runner = InMemoryRunner(tag)
    accelerate(accelerated_runner)

    start = time.time()
    await run_query(baseline_runner)
    without_tygent = time.time() - start

    start = time.time()
    await run_query(accelerated_runner)
    with_tygent = time.time() - start

    print(f"Without Tygent: {without_tygent:.2f}s")
    print(f"With Tygent: {with_tygent:.2f}s")
    if with_tygent:
        print(f"Acceleration: {without_tygent / with_tygent:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
