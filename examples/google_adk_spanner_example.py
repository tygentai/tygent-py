"""Example using Google ADK with and without Tygent acceleration.

This script creates a simple Google ADK agent that answers the query:
"Describe the Spanner paper by Google assuming I am a 10th grader, a junior undergraduate, and a first year graduate student."
It then executes the query twice – once using standard execution and once
with Tygent acceleration applied.
"""

import asyncio
import os
import time

from dotenv import load_dotenv

# Load environment variables (GOOGLE_API_KEY) from a .env file if present
load_dotenv()

try:
    from google import genai
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import Runner
    from google.genai import types
except ImportError:
    print("This example requires the google-adk and google-genai packages.")
    print("Install them with: pip install google-adk google-genai")
    raise

from tygent.integrations.google_adk import patch

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("GOOGLE_API_KEY environment variable not set.")
    print("Get an API key from https://makersuite.google.com/app/apikey")
    raise SystemExit

# Configure Google GenAI client
client = genai.Client(api_key=API_KEY)

# Create a simple LLM agent capable of explaining topics at different levels
tag = LlmAgent(
    name="spanner_explainer",
    model="gemini-1.5-flash",
    instruction=(
        "You explain technical topics for different levels of expertise."
        " Provide clear and concise responses."
    ),
)

# Create a runner and session
runner = Runner(tag)
runner.session_service.create_session_sync(
    app_name=runner.app_name, user_id="user", session_id="session"
)

QUERY = (
    "Describe the Spanner paper by Google assuming I am a 10th grader, "
    "a junior undergraduate, and a first year graduate student."
)


def format_events(events):
    if not events:
        return "No response"
    event = events[-1]
    parts = getattr(event.content, "parts", [])
    if parts and hasattr(parts[0], "text"):
        return parts[0].text
    return str(event)


async def run_without_tygent():
    content = types.Content(role="user", parts=[types.Part(text=QUERY)])
    events = []
    async for event in runner.run_async(
        user_id="user", session_id="session", new_message=content
    ):
        events.append(event)
    return events


async def run_with_tygent():
    patch()  # Patch Runner.run_async to use Tygent scheduler
    content = types.Content(role="user", parts=[types.Part(text=QUERY)])
    events = []
    async for event in runner.run_async(
        user_id="user", session_id="session", new_message=content
    ):
        events.append(event)
    return events


async def main():
    print("=== Without Tygent Acceleration ===")
    start = time.time()
    events = await run_without_tygent()
    print(format_events(events))
    print(f"Execution time: {time.time() - start:.2f}s\n")

    print("=== With Tygent Acceleration ===")
    start = time.time()
    events = await run_with_tygent()
    print(format_events(events))
    print(f"Execution time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
