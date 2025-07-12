"""
Example usage of the Tygent Python package - Simple Accelerate Pattern
Demonstrates real agents that fetch data from public APIs and analyze it
with OpenAI. Shows how to use Tygent's accelerate() function for drop-in
optimization.
"""

import asyncio
import os
import sys
from urllib.parse import quote, quote_plus

import aiohttp
from openai import AsyncOpenAI

sys.path.append("./tygent-py")
from tygent import accelerate
from tygent.agent import Agent

# Set your API key - in production use environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and set your API key


def _get_openai_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client using the ``OPENAI_API_KEY`` env var."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)


class SearchAgent(Agent):
    def __init__(self) -> None:
        super().__init__("search_agent")

    async def execute(self, inputs):
        query = inputs.get("query", "")
        if not query:
            raise ValueError("query must be provided")

        print(f"Searching Wikipedia for '{query}'...")
        async with aiohttp.ClientSession() as session:
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            }
            async with session.get(
                "https://en.wikipedia.org/w/api.php", params=search_params
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Wikipedia search error ({resp.status}): {text}"
                    )
                search_data = await resp.json()
                search_results = search_data.get("query", {}).get("search", [])
                if not search_results:
                    raise RuntimeError(f"No Wikipedia results for '{query}'")
                title = search_results[0]["title"]

            encoded = quote(title.replace(" ", "_"), safe="")
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/" f"{encoded}"
            )
            async with session.get(summary_url) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Wikipedia API error ({resp.status}): {text}")
                summary_data = await resp.json()
                return {
                    "summary": summary_data.get("extract", ""),
                    "title": summary_data.get("title", title),
                }


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__("weather_agent")

    async def _geocode(self, session: aiohttp.ClientSession, location: str):
        params = {"name": location, "count": 1}
        async with session.get(
            "https://geocoding-api.open-meteo.com/v1/search", params=params
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"Geocoding failed ({resp.status}): {text}")
            data = await resp.json()
            results = data.get("results")
            if not results:
                raise RuntimeError(f"Location '{location}' not found")
            return float(results[0]["latitude"]), float(results[0]["longitude"])

    async def _fetch_weather(
        self, session: aiohttp.ClientSession, lat: float, lon: float
    ):
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "timezone": "UTC",
        }
        async with session.get(
            "https://api.open-meteo.com/v1/forecast", params=params
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"Weather API error ({resp.status}): {text}")
            data = await resp.json()
            return data.get("current_weather", {})

    @staticmethod
    def _code_to_text(code: int) -> str:
        mapping = {
            0: "clear",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "fog",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            80: "rain showers",
            95: "thunderstorm",
        }
        return mapping.get(code, f"code {code}")

    async def execute(self, inputs):
        location = inputs.get("location", "")
        if not location:
            raise ValueError("location must be provided")

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                lat, lon = await self._geocode(session, location)
                weather_raw = await self._fetch_weather(session, lat, lon)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise RuntimeError(
                    f"Network error during weather lookup: {exc}"
                ) from exc

        code = int(weather_raw.get("weathercode", 0))
        temp_c = float(weather_raw.get("temperature", 0.0))
        return {
            "temperature": temp_c * 9 / 5 + 32,
            "conditions": self._code_to_text(code),
            "location": location,
        }


class AnalysisAgent(Agent):
    def __init__(self) -> None:
        super().__init__("analysis_agent")

    async def execute(self, inputs):
        search_summary = inputs.get("search_summary", "")
        weather = inputs.get("weather", {})
        print("Analyzing data via OpenAI...")
        client = _get_openai_client()
        prompt = (
            "Given the following search summary and weather data, "
            "provide a short analysis. "
            f"Search: {search_summary}. Weather: {weather}."
        )
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


# Your existing workflow function - no changes needed
async def my_existing_workflow():
    """Existing workflow that you want to accelerate."""
    print("Starting workflow...")

    search_agent = SearchAgent()
    weather_agent = WeatherAgent()
    analysis_agent = AnalysisAgent()

    # These calls normally run sequentially
    search = await search_agent.execute(
        {"query": "artificial intelligence advancements"}
    )
    weather = await weather_agent.execute({"location": "New York"})
    analysis = await analysis_agent.execute(
        {"search_summary": search["summary"], "weather": weather}
    )

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
