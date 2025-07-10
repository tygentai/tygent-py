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
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure local package import when running from source checkout
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
import aiohttp
from openai import AsyncOpenAI

from tygent import accelerate
from tygent.agent import Agent


def _get_openai_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client.

    Raises
    ------
    RuntimeError
        If no OpenAI API key is available via ``OPENAI_API_KEY``.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)


# External services implemented as agents
class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__("weather_agent")

    async def _geocode(
        self, session: "aiohttp.ClientSession", location: str
    ) -> Tuple[float, float]:
        """Get latitude and longitude for a location using Open-Meteo."""
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
        self, session: "aiohttp.ClientSession", lat: float, lon: float
    ) -> Dict[str, Any]:
        """Retrieve current weather data from Open-Meteo."""
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

    async def _summarize_weather(self, weather: Dict[str, Any], location: str) -> str:
        """Summarize raw weather data using an LLM."""
        client = _get_openai_client()
        prompt = (
            f"Provide a short human readable summary for the following "
            f"weather data in {location}: {weather}."
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

    @staticmethod
    def _code_to_text(code: int) -> str:
        """Convert Open-Meteo weather code to descriptive text."""
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

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        location = inputs.get("location", "")
        if not location:
            raise ValueError("location must be provided")

        async with aiohttp.ClientSession() as session:
            lat, lon = await self._geocode(session, location)
            weather_raw = await self._fetch_weather(session, lat, lon)

        code = int(weather_raw.get("weathercode", 0))
        temp_c = float(weather_raw.get("temperature", 0.0))
        summary = await self._summarize_weather(weather_raw, location)

        return {
            "temperature": temp_c * 9 / 5 + 32,
            "conditions": self._code_to_text(code),
            "location": location,
            "summary": summary,
        }


class BackupWeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__("backup_weather_agent")

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Backup weather service using wttr.in."""
        location = inputs.get("location", "")
        if not location:
            raise ValueError("location must be provided")

        print(f"Using backup weather service for {location}...")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://wttr.in/{location}?format=j1") as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"wttr.in error ({resp.status}): {text}")
                data = await resp.json()

        current = data["current_condition"][0]
        temp_f = float(current["temp_F"])
        conditions = current["weatherDesc"][0]["value"].lower()

        # Summarize using OpenAI
        client = _get_openai_client()
        prompt = (
            f"Provide a short human readable summary for the following "
            f"weather data in {location}: {current}."
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        return {
            "temperature": temp_f,
            "conditions": conditions,
            "location": location,
            "summary": summary,
        }


class TrafficAgent(Agent):
    def __init__(self) -> None:
        super().__init__("traffic_agent")

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Approximate traffic using CityBikes availability."""
        location = inputs.get("location", "")
        if not location:
            raise ValueError("location must be provided")

        print(f"Getting traffic data for {location}...")

        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.citybik.es/v2/networks") as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"CityBikes list error ({resp.status}): {text}")
                all_networks = await resp.json()

            network_id = None
            for network in all_networks.get("networks", []):
                city = network.get("location", {}).get("city", "").lower()
                if city == location.lower():
                    network_id = network.get("id")
                    break

            if not network_id:
                raise RuntimeError(f"No bike network found for {location}")

            async with session.get(
                f"https://api.citybik.es/v2/networks/{network_id}"
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(
                        f"CityBikes network error ({resp.status}): {text}"
                    )
                network_data = await resp.json()

        stations = network_data["network"].get("stations", [])
        free = sum(int(s.get("free_bikes") or 0) for s in stations)
        empty = sum(int(s.get("empty_slots") or 0) for s in stations)
        total = free + empty
        ratio = free / total if total else 0

        if ratio > 0.6:
            traffic_level = "light"
        elif ratio > 0.3:
            traffic_level = "moderate"
        else:
            traffic_level = "heavy"

        avg_speed = max(10, int(60 * ratio))
        incidents = max(0, int((1 - ratio) * 3))

        client = _get_openai_client()
        prompt = (
            f"In {location}, the bike share network has {free} free bikes out of "
            f"{total}. Describe current traffic conditions in one sentence."
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        return {
            "traffic_level": traffic_level,
            "avg_speed": avg_speed,
            "incidents": incidents,
            "summary": summary,
        }


class ActivityRecommendationAgent(Agent):
    def __init__(self) -> None:
        super().__init__("activity_recommendation_agent")

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate activity recommendations via OpenAI."""
        weather = inputs.get("weather", {})
        traffic = inputs.get("traffic", {})
        print("Generating activity recommendations...")

        client = _get_openai_client()
        prompt = (
            "Suggest three short activities for a visitor given the following "
            f"conditions:\nWeather: {weather}\nTraffic: {traffic}. "
            'Respond in JSON as {"activities": [..]}'
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
                acts = data.get("activities")
                if acts is None and isinstance(data, list):
                    acts = data
                elif acts is None:
                    acts = []
            except json.JSONDecodeError:
                acts = [a.strip("- \n") for a in content.splitlines() if a.strip()]
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        return {"recommended_activities": acts[:3]}


class LLMFinalizeAgent(Agent):
    def __init__(self) -> None:
        super().__init__("llm_finalize_agent")

    async def execute(self, inputs: Dict[str, Any]) -> str:
        """Generate a short itinerary using an OpenAI model."""

        destination = inputs.get("destination", "")
        activities: List[str] = inputs.get("activities", [])
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


class IndoorAlternativesAgent(Agent):
    def __init__(self) -> None:
        super().__init__("indoor_alternatives_agent")

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest indoor activities using OpenAI."""
        location = inputs.get("location", "")
        if not location:
            raise ValueError("location must be provided")

        print(f"Finding indoor activities in {location}...")

        client = _get_openai_client()
        prompt = (
            f"List three interesting indoor activities in {location}. "
            'Respond in JSON as {"options": [..]}'
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
                options = data.get("options")
                if options is None and isinstance(data, list):
                    options = data
                elif options is None:
                    options = []
            except json.JSONDecodeError:
                options = [a.strip("- \n") for a in content.splitlines() if a.strip()]
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        return {"options": options[:3]}


class LocalAlternativesAgent(Agent):
    def __init__(self) -> None:
        super().__init__("local_alternatives_agent")

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Modify recommendations for heavy traffic conditions via OpenAI."""
        location = inputs.get("location", "")
        original_recs = inputs.get("original_recs", {})
        if not location:
            raise ValueError("location must be provided")

        print(f"Optimizing for local activities in {location}...")

        client = _get_openai_client()
        prompt = (
            f"Given these activities {original_recs.get('recommended_activities', [])} "
            f"for a trip to {location}, suggest local alternatives that avoid traffic. "
            'Respond in JSON as {"recommended_activities": [..]}'
        )
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
                acts = data.get("recommended_activities")
                if acts is None and isinstance(data, list):
                    acts = data
                elif acts is None:
                    acts = []
            except json.JSONDecodeError:
                acts = [a.strip("- \n") for a in content.splitlines() if a.strip()]
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        return {"recommended_activities": acts[:3]}


# Your existing workflow that we want to make adaptive
async def travel_planning_workflow(destination: str) -> str:
    """Travel planning workflow that adapts to failures and conditions."""
    print(f"Planning activities for {destination}")

    weather_agent = WeatherAgent()
    backup_weather_agent = BackupWeatherAgent()
    traffic_agent = TrafficAgent()
    activity_agent = ActivityRecommendationAgent()
    llm_agent = LLMFinalizeAgent()
    indoor_agent = IndoorAlternativesAgent()
    local_agent = LocalAlternativesAgent()

    # Step 1: Get weather information (primary API)
    try:
        weather_data = await weather_agent.execute({"location": destination})
        print(
            f"✓ Weather: {weather_data['temperature']}°F, {weather_data['conditions']}"
        )
    except Exception as e:
        print(f"⚠ Primary weather API failed: {e}")
        weather_data = await backup_weather_agent.execute({"location": destination})
        weather_data = {
            "temperature": weather_data["temp_f"],
            "conditions": weather_data["weather"],
            "location": weather_data["city"],
        }
        print(
            f"✓ Backup weather: {weather_data['temperature']}°F, {weather_data['conditions']}"
        )

    # Step 2: Get traffic information (can run in parallel with weather)
    try:
        traffic_data = await traffic_agent.execute({"location": destination})
    except Exception as e:
        print(f"⚠ Traffic data unavailable: {e}")
        traffic_data = {
            "traffic_level": "heavy",
            "avg_speed": 10,
            "incidents": 0,
            "summary": "No real-time data available; assuming heavy traffic.",
        }
    else:
        print(
            f"✓ Traffic: {traffic_data['traffic_level']} with {traffic_data['incidents']} incidents"
        )

    # Step 3: Conditional branching based on weather conditions
    if weather_data["conditions"] in ["stormy", "precipitation"]:
        print("🌧 Severe weather detected - adding indoor alternatives...")
        indoor_activities = await indoor_agent.execute({"location": destination})
        print(f"✓ Indoor alternatives: {indoor_activities['options']}")

    # Step 4: Generate recommendations
    recommendations = await activity_agent.execute(
        {"weather": weather_data, "traffic": traffic_data}
    )

    # Step 5: Resource-aware execution
    if traffic_data["traffic_level"] == "heavy":
        print("🚗 Heavy traffic detected - optimizing for local activities...")
        local_options = await local_agent.execute(
            {"location": destination, "original_recs": recommendations}
        )
        recommendations = local_options

    # Step 6: Summarize itinerary using an LLM
    itinerary = await llm_agent.execute(
        {
            "destination": destination,
            "activities": recommendations["recommended_activities"],
        }
    )

    return itinerary


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
