"""
Example usage of the Tygent Python package - Simple Accelerate Pattern
Shows how to use Tygent's accelerate() function for drop-in optimization.
"""

import asyncio
import os
import sys

sys.path.append("./tygent-py")
from tygent import accelerate

# Set your API key - in production use environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and set your API key


async def search_data(query):
    """Example search function."""
    print(f"Searching for: {query}")
    # In real implementation, this would call a search API
    await asyncio.sleep(0.5)  # Simulate API call
    return f"Search results for '{query}'"


async def get_weather(location):
    """Example weather function."""
    print(f"Getting weather for: {location}")
    # In real implementation, this would call a weather API
    await asyncio.sleep(0.3)  # Simulate API call
    return {"temperature": 72, "conditions": "Sunny", "location": location}


async def analyze_data(search_results, weather_data):
    """Example analysis function."""
    print("Analyzing combined data...")
    await asyncio.sleep(0.2)  # Simulate processing
    return f"Analysis: {search_results} combined with weather {weather_data}"


# Your existing workflow function - no changes needed
async def my_existing_workflow():
    """Existing workflow that you want to accelerate."""
    print("Starting workflow...")

    # These calls normally run sequentially
    search_results = await search_data("artificial intelligence advancements")
    weather_data = await get_weather("New York")
    analysis = await analyze_data(search_results, weather_data)

    print(f"Final result: {analysis}")
    return analysis


async def main():
    print("=== Standard Execution ===")
    start_time = asyncio.get_event_loop().time()

    # Run your existing workflow normally
    result1 = await my_existing_workflow()

    standard_time = asyncio.get_event_loop().time() - start_time
    print(f"Standard execution time: {standard_time:.2f} seconds\n")

    print("=== Accelerated Execution ===")
    start_time = asyncio.get_event_loop().time()

    # Only change: wrap your existing function with accelerate()
    accelerated_workflow = accelerate(my_existing_workflow)
    result2 = await accelerated_workflow()

    accelerated_time = asyncio.get_event_loop().time() - start_time
    print(f"Accelerated execution time: {accelerated_time:.2f} seconds")

    # Results should be identical
    print(f"\nResults match: {result1 == result2}")

    if standard_time > accelerated_time:
        improvement = ((standard_time - accelerated_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")


if __name__ == "__main__":
    asyncio.run(main())
