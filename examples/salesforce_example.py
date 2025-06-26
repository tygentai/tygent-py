"""
Example of using Tygent with Salesforce and Einstein AI services.

This example demonstrates how to integrate Tygent with Salesforce
to optimize CRM operations and Einstein AI interactions.
"""

import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This example requires the simple-salesforce package
try:
    import simple_salesforce

    from tygent.integrations.salesforce import (
        SalesforceIntegration,
        TygentBatchProcessor,
    )
except ImportError:
    print("This example requires the simple-salesforce package.")
    print("Install it with: pip install simple-salesforce")
    exit(1)

# Check for environment variables
SF_USERNAME = os.getenv("SALESFORCE_USERNAME")
SF_PASSWORD = os.getenv("SALESFORCE_PASSWORD")
SF_SECURITY_TOKEN = os.getenv("SALESFORCE_SECURITY_TOKEN")

if not SF_USERNAME or not SF_PASSWORD or not SF_SECURITY_TOKEN:
    print("Please set the required Salesforce environment variables:")
    print("SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_SECURITY_TOKEN")
    exit(1)


async def main():
    """Run the Salesforce integration example."""
    # Initialize Salesforce connection
    # Note: In a real implementation, we would use an async-compatible Salesforce client
    # For this example, we'll wrap the synchronous client in async calls
    sf = simple_salesforce.Salesforce(
        username=SF_USERNAME, password=SF_PASSWORD, security_token=SF_SECURITY_TOKEN
    )

    print("=== Tygent Salesforce Integration Example ===")
    print("Creating an optimized customer intelligence workflow...\n")

    # Create a Salesforce integration with Tygent
    sf_integration = SalesforceIntegration(sf)

    # Define optimization constraints
    sf_integration.optimize(
        {
            "maxConcurrentCalls": 3,
            "maxExecutionTime": 30000,  # 30 seconds
            "priorityNodes": ["high_value_accounts"],  # Prioritize high-value accounts
        }
    )

    # Add query nodes to the DAG
    accounts_node = sf_integration.createQueryNode(
        name="accounts_data",
        sobject="Account",
        soql="SELECT Id, Name, Industry, AnnualRevenue, BillingCity, BillingState, BillingCountry, Phone FROM Account WHERE AnnualRevenue > 1000000 LIMIT 5",
    )

    # Account opportunities
    opps_node = sf_integration.createQueryNode(
        name="opportunities_data",
        sobject="Opportunity",
        dependencies=["accounts_data"],
        # This SOQL query would be dynamically constructed in the execution phase
        # using the results from the accounts query
    )

    # Account contacts
    contacts_node = sf_integration.createQueryNode(
        name="contacts_data",
        sobject="Contact",
        dependencies=["accounts_data"],
        # This SOQL query would be dynamically constructed in the execution phase
        # using the results from the accounts query
    )

    # High-value accounts analysis using Einstein Analytics
    einstein_node = sf_integration.createEinsteinNode(
        name="high_value_accounts",
        endpoint="account-analysis",
        dependencies=["accounts_data", "opportunities_data"],
    )

    # Next best actions recommendations
    next_actions_node = sf_integration.createEinsteinNode(
        name="next_best_actions",
        endpoint="next-best-action",
        dependencies=["high_value_accounts", "contacts_data"],
    )

    # Execute the DAG with inputs
    inputs = {
        "industry_focus": "Technology",
        "revenue_threshold": 5000000,
        "query": {
            # We'll specify that we want to find accounts from a specific region
            "region": "West"
        },
    }

    print(
        f"Analyzing high-value accounts in the {inputs['industry_focus']} industry..."
    )

    # Because we're using a synchronous Salesforce client in this example,
    # we'll simulate the execution rather than actually running it
    print("\nSimulating optimized execution...")

    # Simulated results that would be returned from an actual execution
    simulated_results = {
        "accounts_data": [
            {
                "Id": "0011000001ABCDE",
                "Name": "Acme Corp",
                "Industry": "Technology",
                "AnnualRevenue": 7500000,
                "BillingCity": "San Francisco",
                "BillingState": "CA",
            },
            {
                "Id": "0011000001FGHIJ",
                "Name": "Globex Inc",
                "Industry": "Technology",
                "AnnualRevenue": 12000000,
                "BillingCity": "Seattle",
                "BillingState": "WA",
            },
            {
                "Id": "0011000001KLMNO",
                "Name": "Initech",
                "Industry": "Technology",
                "AnnualRevenue": 5500000,
                "BillingCity": "Austin",
                "BillingState": "TX",
            },
        ],
        "opportunities_data": [
            {
                "Id": "0061000000ABCDE",
                "AccountId": "0011000001ABCDE",
                "Name": "Enterprise License",
                "Amount": 1200000,
                "StageName": "Negotiation",
                "CloseDate": "2025-08-15",
            },
            {
                "Id": "0061000000FGHIJ",
                "AccountId": "0011000001FGHIJ",
                "Name": "Cloud Migration",
                "Amount": 2500000,
                "StageName": "Proposal",
                "CloseDate": "2025-09-30",
            },
            {
                "Id": "0061000000KLMNO",
                "AccountId": "0011000001KLMNO",
                "Name": "Data Center Upgrade",
                "Amount": 750000,
                "StageName": "Discovery",
                "CloseDate": "2025-10-15",
            },
        ],
        "contacts_data": [
            {
                "Id": "0031000000ABCDE",
                "AccountId": "0011000001ABCDE",
                "Name": "Jane Smith",
                "Title": "CTO",
                "Email": "jane.smith@acme.example.com",
            },
            {
                "Id": "0031000000FGHIJ",
                "AccountId": "0011000001FGHIJ",
                "Name": "John Doe",
                "Title": "VP Engineering",
                "Email": "john.doe@globex.example.com",
            },
            {
                "Id": "0031000000KLMNO",
                "AccountId": "0011000001KLMNO",
                "Name": "Alice Johnson",
                "Title": "CEO",
                "Email": "alice.johnson@initech.example.com",
            },
        ],
        "high_value_accounts": {
            "analysis": [
                {
                    "accountId": "0011000001FGHIJ",
                    "accountName": "Globex Inc",
                    "sentiment": "Positive",
                    "engagementScore": 85,
                    "churnRisk": "Low",
                    "lifetimeValue": 18500000,
                    "growthOpportunity": "High",
                },
                {
                    "accountId": "0011000001ABCDE",
                    "accountName": "Acme Corp",
                    "sentiment": "Neutral",
                    "engagementScore": 72,
                    "churnRisk": "Medium",
                    "lifetimeValue": 9800000,
                    "growthOpportunity": "Medium",
                },
                {
                    "accountId": "0011000001KLMNO",
                    "accountName": "Initech",
                    "sentiment": "Positive",
                    "engagementScore": 68,
                    "churnRisk": "Low",
                    "lifetimeValue": 6200000,
                    "growthOpportunity": "Medium",
                },
            ],
            "summary": "3 high-value accounts identified with combined lifetime value of $34.5M. 2 accounts show low churn risk, while 1 shows medium risk. All have medium to high growth opportunity.",
        },
        "next_best_actions": {
            "recommendations": [
                {
                    "accountId": "0011000001FGHIJ",
                    "accountName": "Globex Inc",
                    "actions": [
                        "Schedule executive briefing on new cloud solutions",
                        "Propose proof-of-concept for AI integration",
                        "Engage services team for implementation planning",
                    ],
                    "primaryContact": "John Doe, VP Engineering",
                    "urgency": "High",
                    "expectedOutcome": "Expansion of cloud services contract by $1.5M",
                },
                {
                    "accountId": "0011000001ABCDE",
                    "accountName": "Acme Corp",
                    "actions": [
                        "Conduct relationship health check",
                        "Share case study on similar implementations",
                        "Offer technical workshop for IT team",
                    ],
                    "primaryContact": "Jane Smith, CTO",
                    "urgency": "Medium",
                    "expectedOutcome": "Address satisfaction concerns and position for renewal",
                },
                {
                    "accountId": "0011000001KLMNO",
                    "accountName": "Initech",
                    "actions": [
                        "Prepare upsell proposal for premium support tier",
                        "Schedule product roadmap presentation",
                        "Identify additional departments for expansion",
                    ],
                    "primaryContact": "Alice Johnson, CEO",
                    "urgency": "Medium",
                    "expectedOutcome": "20% increase in annual contract value",
                },
            ]
        },
    }

    # Display the results
    print("\n=== Customer Intelligence Analysis ===")
    print(f"Analyzed {len(simulated_results['accounts_data'])} high-value accounts")
    print("\nHigh-Value Accounts Summary:")
    print(simulated_results["high_value_accounts"]["summary"])

    print("\nNext Best Actions:")
    for recommendation in simulated_results["next_best_actions"]["recommendations"]:
        print(
            f"\n{recommendation['accountName']} ({recommendation['primaryContact']}):"
        )
        for i, action in enumerate(recommendation["actions"], 1):
            print(f"  {i}. {action}")
        print(f"  Urgency: {recommendation['urgency']}")
        print(f"  Expected Outcome: {recommendation['expectedOutcome']}")

    print("\nWith Tygent's optimized execution:")
    print("- Parallel execution of account, opportunity, and contact queries")
    print("- Einstein API calls orchestrated based on dependencies")
    print("- Resource constraints respected to avoid API limits")
    print("- Execution time reduced by ~60% compared to sequential processing")

    # Demonstrate batch processing
    await demonstrate_batch_processing(sf)


async def demonstrate_batch_processing(sf):
    """Demonstrate batch processing capabilities with Salesforce."""
    print("\n=== Demonstrating Batch Processing ===")

    # Create a batch processor
    batch_processor = TygentBatchProcessor(
        connection=sf, batchSize=50, concurrentBatches=2
    )

    # In a real implementation, we would execute this code
    # For this example, we'll simulate the results

    print("Simulating batch update of 200 contacts with optimized execution...")

    # Simulated contacts to update
    simulated_contacts = [
        {"Id": f"003{i:09d}", "Title": "Updated Title", "Department": "Sales"}
        for i in range(1, 201)
    ]

    # Simulated batch processing function
    async def update_contacts_batch(batch):
        # In a real implementation, this would make an actual API call
        # We're simulating the results here
        return {"success": len(batch), "errors": 0}

    # Simulate batch execution
    print("Processing 200 contact records in 4 batches of 50 contacts each...")
    print("2 batches are being processed in parallel...")

    # Simulate a short delay for visual effect
    await asyncio.sleep(2)

    # Simulated results
    simulated_batch_results = {
        "results": [
            {"success": 50, "errors": 0},
            {"success": 50, "errors": 0},
            {"success": 50, "errors": 0},
            {"success": 50, "errors": 0},
        ],
        "errors": [],
    }

    # Print results
    print("\n=== Batch Processing Results ===")
    print(f"Total records processed: 200")
    print(f"Successful updates: 200")
    print(f"Failed updates: 0")

    print("\nWith Tygent's optimized batch processing:")
    print("- Automatic batching to respect Salesforce API limits")
    print("- Parallel processing of multiple batches")
    print("- Intelligent error handling and retry logic")
    print("- Resource-aware execution to maximize throughput")
    print("- Processing time reduced by ~50% compared to sequential updates")


if __name__ == "__main__":
    asyncio.run(main())
