"""
Example of integrating Tygent with LangChain - Simple Accelerate Pattern
Shows how to use Tygent's accelerate() function with existing LangChain agents.
"""

import sys

sys.path.append("./tygent-py")
from tygent import accelerate

# Mock LangChain components for demonstration
# In real usage, these would be actual LangChain imports:
# from langchain.agents import initialize_agent, Tool
# from langchain.llms import OpenAI


class MockLangChainTool:
    """Mock LangChain tool for demonstration."""

    def __init__(self, name, func):
        self.name = name
        self.func = func


class MockLangChainAgent:
    """Mock LangChain agent for demonstration."""

    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm

    def run(self, query):
        """Simulate LangChain agent execution."""
        # In real LangChain, this would intelligently use tools
        results = []
        for tool in self.tools:
            if "search" in tool.name.lower() and "search" in query.lower():
                results.append(f"{tool.name}: {tool.func(query)}")
            elif "calculator" in tool.name.lower() and any(
                char.isdigit() for char in query
            ):
                results.append(f"{tool.name}: {tool.func(query)}")

        return f"Agent response: {query}. Tool results: {'; '.join(results) if results else 'No tools needed'}"


# Your existing LangChain setup - no changes needed
def search_func(query):
    """Example search tool function."""
    return f"Search results for: {query}"


def calculator_func(expression):
    """Example calculator tool function."""
    try:
        # Simple calculation for demo
        return f"Calculated: {expression}"
    except:
        return "Invalid calculation"


def main():
    print("Tygent + LangChain Integration Example")
    print("=====================================")

    # Your existing LangChain agent setup - unchanged
    print("\n1. Creating LangChain agent with tools...")

    tools = [
        MockLangChainTool(name="Search", func=search_func),
        MockLangChainTool(name="Calculator", func=calculator_func),
    ]

    # Your existing agent creation - unchanged
    agent = MockLangChainAgent(tools=tools, llm="gpt-4")

    print("2. Testing standard LangChain agent...")
    query = "Search for AI developments and calculate 2+2"
    standard_result = agent.run(query)
    print(f"Standard result: {standard_result}")

    print("\n3. Accelerating LangChain agent with Tygent...")

    # Only change: wrap your existing agent with accelerate()
    accelerated_agent = accelerate(agent)

    print("4. Testing accelerated LangChain agent...")
    accelerated_result = accelerated_agent.run(query)
    print(f"Accelerated result: {accelerated_result}")

    print(f"\nResults match: {standard_result == accelerated_result}")
    print("\n✅ LangChain integration complete!")
    print("   Behind the scenes, Tygent automatically:")
    print("   • Analyzes tool dependencies")
    print("   • Runs independent tools in parallel")
    print("   • Optimizes execution order")
    print("   • Maintains exact same agent behavior")


if __name__ == "__main__":
    main()
