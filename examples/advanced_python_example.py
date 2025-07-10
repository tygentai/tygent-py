"""
Advanced Example: Customer Support Agent with Tygent - Simple Accelerate Pattern
--------------------------------------------------------------------------------
This example demonstrates how to accelerate an existing customer support workflow
using Tygent's accelerate() function for automatic parallel optimization.

The support agent workflow can:
1. Analyze customer questions
2. Search a knowledge base
3. Check customer purchase history
4. Generate personalized responses
5. Recommend related products

Tygent automatically identifies and parallelizes independent operations
like knowledge base search and customer history lookup.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure the local package is used when running from the source checkout
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from tygent import accelerate

# Set your API key - in production use environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and set your API key

# Simulated database
KNOWLEDGE_BASE = {
    "product_return": "Products can be returned within 30 days with receipt for a full refund.",
    "shipping_time": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "account_reset": "You can reset your password by clicking 'Forgot Password' on the login page.",
    "product_warranty": "Our products come with a 1-year limited warranty covering manufacturing defects.",
}

CUSTOMER_DATABASE = {
    "user123": {
        "name": "Jane Smith",
        "purchases": [
            {
                "product": "Wireless Headphones",
                "date": "2025-02-15",
                "order_id": "ORD-7891",
            },
            {"product": "Smart Speaker", "date": "2025-04-10", "order_id": "ORD-8567"},
        ],
        "subscription": "Premium",
        "account_created": "2024-12-01",
    },
    "user456": {
        "name": "John Doe",
        "purchases": [
            {"product": "Smartphone", "date": "2025-03-05", "order_id": "ORD-3245"},
        ],
        "subscription": "Basic",
        "account_created": "2025-01-15",
    },
}

PRODUCT_RECOMMENDATIONS = {
    "Wireless Headphones": ["Headphone Case", "Bluetooth Adapter", "Extended Warranty"],
    "Smart Speaker": ["Smart Bulbs", "Voice Remote", "Speaker Stand"],
    "Smartphone": ["Phone Case", "Screen Protector", "Wireless Charger"],
}


# Tool functions for our agent
async def analyze_question(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the customer question to determine intent and keywords."""
    question = inputs.get("question", "")
    print(f"Analyzing question: {question}")

    # In a real implementation, this would use an LLM or classifier
    # For demo purposes, using simple keyword matching
    keywords = []
    intent = "general"

    if "return" in question.lower() or "refund" in question.lower():
        keywords.append("return")
        keywords.append("refund")
        intent = "product_return"
    elif "shipping" in question.lower() or "delivery" in question.lower():
        keywords.append("shipping")
        keywords.append("delivery")
        intent = "shipping_time"
    elif (
        "password" in question.lower()
        or "login" in question.lower()
        or "reset" in question.lower()
    ):
        keywords.append("password")
        keywords.append("account")
        intent = "account_reset"
    elif "warranty" in question.lower() or "broken" in question.lower():
        keywords.append("warranty")
        keywords.append("repair")
        intent = "product_warranty"

    # Add a simulated delay to represent real analysis time
    await asyncio.sleep(0.5)

    return {"intent": intent, "keywords": keywords, "confidence": 0.85}


async def search_knowledge_base(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Search for relevant information in the knowledge base."""
    intent = inputs.get("intent", "general")
    keywords = inputs.get("keywords", [])

    print(f"Searching knowledge base for intent: {intent}, keywords: {keywords}")

    # In a real implementation, this would use a vector search or database query
    # For demo purposes, using direct lookup based on intent
    knowledge_base_result = KNOWLEDGE_BASE.get(intent, "No specific information found.")

    # Add a simulated delay to represent real database query time
    await asyncio.sleep(0.7)

    return {
        "knowledge_result": knowledge_base_result,
        "sources": [f"knowledge_base:{intent}"],
    }


async def get_customer_history(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve customer purchase history and account information."""
    user_id = inputs.get("user_id", "")

    print(f"Getting customer history for user: {user_id}")

    # In a real implementation, this would query a customer database
    # For demo purposes, using a mock database lookup
    customer_info = CUSTOMER_DATABASE.get(user_id, {})

    # Add a simulated delay to represent real database query time
    await asyncio.sleep(0.8)

    if not customer_info:
        return {"error": "Customer not found"}

    return {
        "customer_name": customer_info.get("name", ""),
        "purchase_history": customer_info.get("purchases", []),
        "subscription_tier": customer_info.get("subscription", ""),
        "account_age": "5 months",  # In a real system, this would be calculated
    }


async def generate_product_recommendations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate product recommendations based on purchase history."""
    purchases = inputs.get("purchase_history", [])

    recommendations = []
    for purchase in purchases:
        product = purchase.get("product", "")
        if product in PRODUCT_RECOMMENDATIONS:
            recommendations.extend(PRODUCT_RECOMMENDATIONS[product])

    # Deduplicate recommendations
    recommendations = list(set(recommendations))

    # Add a simulated delay
    await asyncio.sleep(0.3)

    return {"recommended_products": recommendations[:3]}  # Top 3 recommendations


async def generate_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a personalized response to the customer question."""
    question = inputs.get("question", "")
    knowledge_result = inputs.get("knowledge_result", "")
    customer_name = inputs.get("customer_name", "")
    subscription_tier = inputs.get("subscription_tier", "")
    recommended_products = inputs.get("recommended_products", [])

    # In a real implementation, this would use an LLM with a prompt
    # For demo purposes, using template-based generation

    # Personalized greeting
    response = f"Hello {customer_name}, thanks for contacting our support team.\n\n"

    # Answer to question
    response += f"Regarding your question about '{question}':\n{knowledge_result}\n\n"

    # Subscription tier message
    if subscription_tier == "Premium":
        response += "As a Premium member, you have access to our priority support line at 1-800-555-HELP.\n\n"

    # Product recommendations
    if recommended_products:
        response += (
            "Based on your previous purchases, you might also be interested in:\n"
        )
        for product in recommended_products:
            response += f"- {product}\n"

    # Add a simulated delay to represent generation time
    await asyncio.sleep(0.5)

    return {
        "response_text": response,
        "response_sentiment": "helpful",
        "response_length": len(response),
    }


# Your existing customer support workflow - no changes needed
async def customer_support_workflow(question: str, user_id: str) -> str:
    """Complete customer support workflow that processes a customer question."""
    print(f"Processing customer query: '{question}' for user: {user_id}")

    # Step 1: Analyze the question
    analysis = await analyze_question({"question": question})

    # Step 2: Search knowledge base (depends on analysis)
    knowledge = await search_knowledge_base(
        {"intent": analysis["intent"], "keywords": analysis["keywords"]}
    )

    # Step 3: Get customer history (independent of analysis)
    customer_info = await get_customer_history({"user_id": user_id})

    # Step 4: Generate recommendations (depends on customer history)
    recommendations = await generate_product_recommendations(
        {"purchase_history": customer_info.get("purchase_history", [])}
    )

    # Step 5: Generate final response (depends on all previous steps)
    response = await generate_response(
        {
            "question": question,
            "knowledge_result": knowledge["knowledge_result"],
            "customer_name": customer_info.get("customer_name", ""),
            "subscription_tier": customer_info.get("subscription_tier", ""),
            "recommended_products": recommendations["recommended_products"],
        }
    )

    return response["response_text"]


async def main():
    print("Advanced Customer Support Agent with Tygent")
    print("===========================================\n")

    # Define our customer query scenario
    customer_query = "Can I return the headphones I bought last month?"
    customer_id = "user123"

    print(f"Customer query: '{customer_query}'")
    print(f"Customer ID: {customer_id}\n")

    print("=== Standard Execution ===")
    start_time = time.time()

    # Run your existing workflow normally
    standard_response = await customer_support_workflow(customer_query, customer_id)

    standard_time = time.time() - start_time
    print(f"Standard execution time: {standard_time:.2f} seconds")
    print(f"Response: {standard_response[:100]}...\n")

    print("=== Accelerated Execution ===")
    start_time = time.time()

    # Only change: wrap your existing workflow with accelerate()
    accelerated_workflow = accelerate(customer_support_workflow)
    accelerated_response = await accelerated_workflow(customer_query, customer_id)

    accelerated_time = time.time() - start_time
    print(f"Accelerated execution time: {accelerated_time:.2f} seconds")
    print(f"Response: {accelerated_response[:100]}...")

    # Results should be identical
    print(f"\nResults match: {standard_response == accelerated_response}")

    if standard_time > accelerated_time:
        improvement = ((standard_time - accelerated_time) / standard_time) * 100
        print(f"Performance improvement: {improvement:.1f}% faster")

    print("\n✅ Behind the scenes, Tygent automatically:")
    print("   • Identified that knowledge search and customer lookup are independent")
    print("   • Ran those operations in parallel")
    print("   • Maintained the correct dependency order for final response")
    print("   • Delivered identical results with improved performance")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            loop = asyncio.get_event_loop()
            task = loop.create_task(main())
            loop.run_until_complete(task)
        else:
            raise
