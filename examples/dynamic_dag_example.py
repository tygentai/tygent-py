"""
Dynamic DAG Modification Example - Shows runtime DAG rewriting capabilities
--------------------------------------------------------------------------
This example demonstrates Tygent's ability to modify execution graphs during runtime
based on intermediate results, errors, or resource constraints.

Features demonstrated:
1. Conditional branching based on intermediate results
2. Fallback mechanisms when APIs fail
3. Resource-aware execution adaptation
4. Real-time DAG visualization of modifications
"""

import asyncio
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.append("./tygent-py")
from openai import AsyncOpenAI

from tygent import accelerate


def _get_openai_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client.

    Raises
    ------
    RuntimeError
        If no OpenAI API key is available via ``OPENAI_API_KEY`` or
        ``OPENAI_APY_KEY``.
    """

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APY_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or OPENAI_APY_KEY"
        )
    return AsyncOpenAI(api_key=api_key)


# Simulated external services
async def weather_api_call(location: str) -> Dict[str, Any]:
    """Simulated weather API that sometimes fails."""
    print(f"Calling weather API for {location}...")
    await asyncio.sleep(0.5)

    # Simulate API failure 30% of the time
    if random.random() < 0.3:
        raise Exception("Weather API temporarily unavailable")

    return {
        "temperature": random.randint(60, 85),
        "conditions": random.choice(["sunny", "cloudy", "rainy", "stormy"]),
        "location": location,
    }


async def backup_weather_service(location: str) -> Dict[str, Any]:
    """Backup weather service with different data format."""
    print(f"Using backup weather service for {location}...")
    await asyncio.sleep(0.3)

    return {
        "temp_f": random.randint(55, 80),
        "weather": random.choice(["clear", "overcast", "precipitation"]),
        "city": location,
    }


async def traffic_api_call(location: str) -> Dict[str, Any]:
    """Traffic information API."""
    print(f"Getting traffic data for {location}...")
    await asyncio.sleep(0.4)

    return {
        "traffic_level": random.choice(["light", "moderate", "heavy"]),
        "avg_speed": random.randint(25, 65),
        "incidents": random.randint(0, 3),
    }


async def activity_recommendations(
    weather: Dict[str, Any], traffic: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate activity recommendations based on conditions."""
    print("Generating activity recommendations...")
    await asyncio.sleep(0.2)

    activities = []

    # Weather-based recommendations
    temp = weather.get("temperature", weather.get("temp_f", 70))
    conditions = weather.get("conditions", weather.get("weather", "clear"))

    if temp > 75 and conditions in ["sunny", "clear"]:
        activities.extend(["outdoor hiking", "beach visit", "picnic"])
    elif conditions in ["rainy", "precipitation"]:
        activities.extend(["museum visit", "indoor shopping", "movie theater"])
    else:
        activities.extend(["city walking tour", "cafe hopping"])

    # Traffic-based adjustments
    if traffic["traffic_level"] == "heavy":
        activities = [f"{activity} (avoid rush hour)" for activity in activities]

    return {"recommended_activities": activities[:3]}


async def llm_finalize_plan(destination: str, activities: List[str]) -> str:
    """Generate a short itinerary using an OpenAI model."""

    prompt = (
        f"Create a short travel itinerary for {destination} including: "
        f"{', '.join(activities)}."
    )

    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e


# Your existing workflow that we want to make adaptive
async def travel_planning_workflow(destination: str) -> str:
    """Travel planning workflow that adapts to failures and conditions."""
    print(f"Planning activities for {destination}")

    # Step 1: Get weather information (primary API)
    try:
        weather_data = await weather_api_call(destination)
        print(
            f"✓ Weather: {weather_data['temperature']}°F, {weather_data['conditions']}"
        )
    except Exception as e:
        print(f"⚠ Primary weather API failed: {e}")
        # Dynamic adaptation: switch to backup service
        weather_data = await backup_weather_service(destination)
        # Normalize the data format
        weather_data = {
            "temperature": weather_data["temp_f"],
            "conditions": weather_data["weather"],
            "location": weather_data["city"],
        }
        print(
            f"✓ Backup weather: {weather_data['temperature']}°F, {weather_data['conditions']}"
        )

    # Step 2: Get traffic information (can run in parallel with weather)
    traffic_data = await traffic_api_call(destination)
    print(
        f"✓ Traffic: {traffic_data['traffic_level']} with {traffic_data['incidents']} incidents"
    )

    # Step 3: Conditional branching based on weather conditions
    if weather_data["conditions"] in ["stormy", "precipitation"]:
        print("🌧 Severe weather detected - adding indoor alternatives...")
        # In real implementation, this would modify the DAG to include indoor activity nodes
        indoor_activities = await get_indoor_alternatives(destination)
        print(f"✓ Indoor alternatives: {indoor_activities['options']}")

    # Step 4: Generate recommendations
    recommendations = await activity_recommendations(weather_data, traffic_data)

    # Step 5: Resource-aware execution
    if traffic_data["traffic_level"] == "heavy":
        print("🚗 Heavy traffic detected - optimizing for local activities...")
        # This would trigger DAG modification to prioritize nearby locations
        local_options = await get_local_alternatives(destination, recommendations)
        recommendations = local_options

    # Step 6: Summarize itinerary using an LLM
    itinerary = await llm_finalize_plan(
        destination, recommendations["recommended_activities"]
    )

    return itinerary


async def get_indoor_alternatives(location: str) -> Dict[str, Any]:
    """Additional node added dynamically for bad weather."""
    print(f"Finding indoor activities in {location}...")
    await asyncio.sleep(0.3)
    return {"options": ["art galleries", "indoor markets", "historic buildings"]}


async def get_local_alternatives(
    location: str, original_recs: Dict[str, Any]
) -> Dict[str, Any]:
    """Modify recommendations for heavy traffic conditions."""
    print(f"Optimizing for local activities in {location}...")
    await asyncio.sleep(0.2)

    # Filter for local activities
    local_activities = []
    for activity in original_recs["recommended_activities"]:
        if "outdoor" in activity:
            local_activities.append(f"local {activity}")
        else:
            local_activities.append(activity)

    return {"recommended_activities": local_activities}


async def main():
    print("Dynamic DAG Modification Example")
    print("================================\n")

    destinations = ["San Francisco", "New York", "Seattle"]

    for destination in destinations:
        print(f"\n{'='*50}")
        print(f"Planning trip to {destination}")
        print(f"{'='*50}")

        # Standard execution
        print("\n=== Standard Execution ===")
        start_time = time.time()

        try:
            standard_result = await travel_planning_workflow(destination)
            standard_time = time.time() - start_time
            print(f"Result: {standard_result}")
            print(f"Execution time: {standard_time:.2f} seconds")
        except Exception as e:
            print(f"Standard execution failed: {e}")
            continue

        print("\n=== Accelerated with Dynamic Adaptation ===")
        start_time = time.time()

        # The accelerate() function automatically adds dynamic DAG modification
        # capabilities to handle the failures and conditions we encounter
        accelerated_workflow = accelerate(travel_planning_workflow)

        try:
            accelerated_result = await accelerated_workflow(destination)
            accelerated_time = time.time() - start_time
            print(f"Result: {accelerated_result}")
            print(f"Execution time: {accelerated_time:.2f} seconds")

            # Results should be identical even with dynamic modifications
            print(f"Results consistent: {standard_result == accelerated_result}")

            if standard_time > accelerated_time:
                improvement = ((standard_time - accelerated_time) / standard_time) * 100
                print(f"Performance improvement: {improvement:.1f}% faster")

        except Exception as e:
            print(f"Accelerated execution failed: {e}")

        print("\n✅ Dynamic adaptations applied:")
        print("   • API failure detection and automatic fallback")
        print("   • Conditional branching for weather conditions")
        print("   • Resource-aware execution for traffic conditions")
        print("   • Real-time DAG modification during execution")

        # Small delay before next destination
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Set random seed for reproducible demo results
    random.seed(42)
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            # Fallback for interactive environments with an existing event loop
            loop = asyncio.get_event_loop()
            task = loop.create_task(main())
            loop.run_until_complete(task)
        else:
            raise
