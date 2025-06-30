"""
Example of using Tygent with Google AI/Gemini models.

This example demonstrates how to integrate Tygent with Google's Gemini
models to optimize execution of multi-step workflows.
"""

import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This example requires the google-generativeai package
try:
    import google.generativeai as genai

    from tygent.integrations.google_ai import (
        GoogleAIBatchProcessor,
        GoogleAIIntegration,
    )
except ImportError:
    print("This example requires the google-generativeai package.")
    print("Install it with: pip install google-generativeai")
    exit(1)

# Configure Google AI with API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("GOOGLE_API_KEY environment variable not set.")
    print("Get an API key from https://makersuite.google.com/app/apikey")
    exit(1)

genai.configure(api_key=API_KEY)


async def main():
    """Run the Google AI integration example."""
    # Configure the Gemini model
    model = genai.GenerativeModel("gemini-pro")

    print("=== Tygent Google AI Integration Example ===")
    print("Creating a travel planning assistant with optimized execution...\n")

    # Create a Google AI integration with Tygent
    google_ai = GoogleAIIntegration(model)

    # Define optimization constraints
    google_ai.optimize(
        {
            "maxParallelCalls": 2,
            "maxExecutionTime": 30000,  # 30 seconds
            "priorityNodes": ["weather_info"],  # Prioritize weather info
        }
    )

    # Add nodes to the execution DAG
    google_ai.addNode(
        name="destination_analysis",
        promptTemplate="Analyze {destination} as a travel destination. "
        + "Provide key highlights, best time to visit, and any travel warnings.",
        dependencies=[],
    )

    google_ai.addNode(
        name="weather_info",
        promptTemplate="What's the typical weather in {destination} during {month}?",
        dependencies=[],
    )

    google_ai.addNode(
        name="activity_suggestions",
        promptTemplate="Suggest 5 must-do activities in {destination} during {month}, "
        + "taking into account the weather conditions: {weather_info}",
        dependencies=["weather_info"],
    )

    google_ai.addNode(
        name="accommodation_suggestions",
        promptTemplate="Recommend 3 types of accommodations in {destination} "
        + "suitable for a {duration} day trip in {month}.",
        dependencies=[],
    )

    google_ai.addNode(
        name="travel_plan",
        promptTemplate="Create a {duration} day travel itinerary for {destination} in {month}. "
        + "Include these destination highlights: {destination_analysis} "
        + "Include these activities: {activity_suggestions} "
        + "Include these accommodations: {accommodation_suggestions}",
        dependencies=[
            "destination_analysis",
            "activity_suggestions",
            "accommodation_suggestions",
        ],
    )

    # Execute the DAG with inputs
    inputs = {"destination": "Kyoto, Japan", "month": "April", "duration": 5}

    print(f"Planning a trip to {inputs['destination']} in {inputs['month']}...")

    # Run the optimized execution
    start_time = asyncio.get_event_loop().time()
    results = await google_ai.execute(inputs)
    end_time = asyncio.get_event_loop().time()

    # Display the results
    print("\n=== Travel Plan Generated ===")
    print(results["travel_plan"][:1000] + "...\n")

    print(f"Execution completed in {end_time - start_time:.2f} seconds")
    print(f"Number of nodes executed: {len(results)}")

    # Demonstrate batch processing
    print("\n=== Demonstrating Batch Processing ===")

    # Create a batch processor
    batch_processor = GoogleAIBatchProcessor(
        model=model, batchSize=2, maxConcurrentBatches=2
    )

    # Define batch processing function
    async def process_city(city: str, model: Any) -> Dict[str, str]:
        response = await model.generate_content(
            f"What are the top 3 attractions in {city}?"
        )
        return {"city": city, "attractions": response.text}

    # Process a batch of cities
    cities = ["Paris", "New York", "Tokyo", "Rome", "Sydney"]
    print(f"Processing information for {len(cities)} cities in optimized batches...")

    start_time = asyncio.get_event_loop().time()
    city_results = await batch_processor.process(cities, process_city)
    end_time = asyncio.get_event_loop().time()

    print("\n=== Batch Processing Results ===")
    for result in city_results:
        print(f"\n{result['city']}:")
        print(f"{result['attractions'][:150]}...")

    print(f"\nBatch processing completed in {end_time - start_time:.2f} seconds")
    print(
        "With standard sequential processing, this would have taken significantly longer."
    )


if __name__ == "__main__":
    asyncio.run(main())
