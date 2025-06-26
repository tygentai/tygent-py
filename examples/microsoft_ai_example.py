"""
Example of using Tygent with Microsoft AI services.

This example demonstrates how to integrate Tygent with Microsoft's Azure OpenAI
Service to optimize execution of multi-step workflows.
"""

import asyncio
import os
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This example requires the azure-openai package
try:
    from azure.openai import AsyncAzureOpenAI, AsyncOpenAI

    from tygent.integrations.microsoft_ai import (
        MicrosoftAIIntegration,
        SemanticKernelOptimizer,
    )
except ImportError:
    print("This example requires the azure-openai package.")
    print("Install it with: pip install azure-openai")
    exit(1)

# Check for environment variables
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    print(
        "Please set the AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables."
    )
    print("You can get these from the Azure portal.")
    exit(1)


async def main():
    """Run the Microsoft AI integration example."""
    # Initialize the Azure OpenAI client
    client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-12-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    print("=== Tygent Microsoft AI Integration Example ===")
    print("Creating a market research analysis workflow with optimized execution...\n")

    # Create a Microsoft AI integration with Tygent
    azure_ai = MicrosoftAIIntegration(client, DEPLOYMENT_NAME)

    # Define optimization constraints
    azure_ai.optimize(
        {
            "maxParallelCalls": 3,
            "maxExecutionTime": 30000,  # 30 seconds
            "priorityNodes": ["market_trends"],  # Prioritize market trends
        }
    )

    # Add nodes to the execution DAG
    azure_ai.addNode(
        name="market_overview",
        promptTemplate="Provide a high-level overview of the {industry} industry in {region} in 2025.",
        dependencies=[],
    )

    azure_ai.addNode(
        name="market_trends",
        promptTemplate="What are the top 5 emerging trends in the {industry} industry in {region} for 2025?",
        dependencies=[],
    )

    azure_ai.addNode(
        name="competitor_analysis",
        promptTemplate="Identify and analyze the top 3 competitors in the {industry} industry in {region}.",
        dependencies=[],
    )

    azure_ai.addNode(
        name="regulatory_landscape",
        promptTemplate="Summarize the key regulatory considerations for the {industry} industry in {region}.",
        dependencies=[],
    )

    azure_ai.addNode(
        name="growth_opportunities",
        promptTemplate=(
            "Based on the following information:\n"
            + "Market overview: {market_overview}\n"
            + "Emerging trends: {market_trends}\n"
            + "Competitor analysis: {competitor_analysis}\n"
            + "Regulatory landscape: {regulatory_landscape}\n\n"
            + "Identify 3-5 high-potential growth opportunities for a new entrant in the {industry} industry in {region}."
        ),
        dependencies=[
            "market_overview",
            "market_trends",
            "competitor_analysis",
            "regulatory_landscape",
        ],
    )

    azure_ai.addNode(
        name="entry_strategy",
        promptTemplate=(
            "Create a market entry strategy outline for the {industry} industry in {region}, "
            + "focusing on these growth opportunities: {growth_opportunities}"
        ),
        dependencies=["growth_opportunities"],
    )

    # Execute the DAG with inputs
    inputs = {"industry": "renewable energy", "region": "Southeast Asia"}

    print(f"Analyzing the {inputs['industry']} industry in {inputs['region']}...")

    # Run the optimized execution
    start_time = asyncio.get_event_loop().time()
    results = await azure_ai.execute(inputs)
    end_time = asyncio.get_event_loop().time()

    # Display the results
    print("\n=== Market Entry Strategy ===")
    print(results["entry_strategy"][:1000] + "...\n")

    print(f"Execution completed in {end_time - start_time:.2f} seconds")
    print(f"Number of nodes executed: {len(results)}")

    # Demonstrate Semantic Kernel integration
    print("\n=== Would you like to see Semantic Kernel integration? ===")
    print("Note: This requires Semantic Kernel to be installed.")
    print("You can install it with: pip install semantic-kernel")
    print(
        "(Example implementation would integrate Tygent's optimization with Semantic Kernel plugins)"
    )


async def semantic_kernel_example():
    """
    Example implementation of Semantic Kernel integration with Tygent.

    Note: This is a placeholder implementation that would require the semantic-kernel package
    to be installed to actually run.
    """
    # This is a skeleton for what the implementation would look like

    try:
        import semantic_kernel as sk
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    except ImportError:
        print("Semantic Kernel is not installed. Skipping this example.")
        return

    # Initialize a Semantic Kernel instance
    kernel = sk.Kernel()

    # Add Azure OpenAI service
    kernel.add_chat_service(
        "chat-gpt4",
        AzureChatCompletion(
            deployment_name=DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        ),
    )

    # Create a plugin
    plugin = kernel.create_semantic_function(
        """{{$input}}
        Analyze the sentiment of the above text and classify it as positive, negative, or neutral.
        Provide a brief explanation for your classification.
        """,
        function_name="sentiment_analysis",
        plugin_name="TextAnalysis",
    )

    # Create Tygent optimizer for Semantic Kernel
    sk_optimizer = SemanticKernelOptimizer(kernel)

    # Register the plugin
    sk_optimizer.registerPlugin(plugin)

    # Create an optimized plan
    sk_optimizer.createPlan("Analyze sentiment and provide recommendations")

    # Optimize with constraints
    sk_optimizer.optimize({"maxParallelCalls": 2, "maxExecutionTime": 10000})

    # Execute the optimized plan
    results = await sk_optimizer.execute(
        {
            "input": "The new product launch exceeded our expectations with record sales, "
            + "though some customers reported minor usability issues."
        }
    )

    print("\n=== Semantic Kernel Results ===")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
