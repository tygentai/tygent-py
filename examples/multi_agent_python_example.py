"""
Example of using Tygent's multi-agent capabilities in Python.

This example demonstrates how to:
1. Create multiple agents with different roles
2. Use MultiAgentManager to coordinate agent execution
3. Execute agents in parallel for improved performance
"""

import asyncio
import os
import sys

sys.path.append("./tygent-py")

from tygent import MultiAgentManager


# Agent classes that work with MultiAgentManager
class ResearcherAgent:
    async def execute(self, inputs):
        query = inputs.get("query", "unknown query")
        print(f"Researcher analyzing: {query}")
        return {"analysis": f"Research findings for: {query}", "confidence": 0.85}


class CriticAgent:
    async def execute(self, inputs):
        query = inputs.get("query", "unknown query")
        print(f"Critic reviewing: {query}")
        return {
            "critique": f"Critical analysis of: {query}",
            "suggestions": ["Consider bias", "Verify sources"],
        }


class SynthesizerAgent:
    async def execute(self, inputs):
        query = inputs.get("query", "unknown query")
        print(f"Synthesizer combining insights")
        return {
            "synthesis": f"Combined insights from research and critique",
            "final_recommendation": "Proceed with caution",
        }


async def main():
    """Run the multi-agent example."""
    print("Tygent Multi-Agent Example")
    print("==========================")

    # Create a multi-agent manager
    manager = MultiAgentManager("research_team")

    # Create agent instances
    researcher = ResearcherAgent()
    critic = CriticAgent()
    synthesizer = SynthesizerAgent()

    # Add agents to the manager
    manager.add_agent("researcher", researcher)
    manager.add_agent("critic", critic)
    manager.add_agent("synthesizer", synthesizer)
    print("Added agents: researcher, critic, synthesizer")

    # Execute the multi-agent workflow
    print(f"\nExecuting multi-agent workflow...")

    result = await manager.execute(
        {"query": "What are the potential benefits and risks of quantum computing?"}
    )

    # Display results
    print("\n=== Multi-Agent Results ===")
    for agent_name, output in result.items():
        print(f"{agent_name}: {output}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
