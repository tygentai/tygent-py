"""Customer support planning example using Tygent.

This script demonstrates how an existing workflow can generate a structured
plan from JSON-style guidelines, draft a customer response, and compare
sequential execution with Tygent's accelerated scheduler.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure the local package is available when running from a checkout
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tygent import accelerate

# ---------------------------------------------------------------------------
# Customer data & JSON-style guidelines the responder will loosely follow
# ---------------------------------------------------------------------------

CUSTOMER_MESSAGE = (
    "Hey team, I ordered a walnut desk two weeks ago (order ORD-7341) and it still "
    "hasn't arrived. I'm getting nervous because I needed it for a client setup. "
    "Can you tell me what's going on and whether there's anything you can do to help?"
)

BASE_CUSTOMER_INPUTS: Dict[str, Any] = {
    "customer_message": CUSTOMER_MESSAGE,
    "customer_name": "Jordan",
    "customer_id": "CUST-1042",
    "order_id": "ORD-7341",
}

SUPPORT_GUIDELINES: Dict[str, Any] = {
    "tone": "Warm, empathetic, and action-oriented",
    "structure": [
        "greeting",
        "acknowledgement",
        "status_update",
        "resolution",
        "closing",
    ],
    "language": {
        "use_plain_language": True,
        "contractions": True,
        "avoid_jargon": True,
    },
    "policies": {
        "shipping_delay_threshold_days": 7,
        "make_good_offer": "apply a courtesy expedited shipping credit",
        "escalation_path": "loop in logistics specialist if shipment is lost",
    },
    "closing": "Warm regards,\nAurora Furnishings Support",
}


# ---------------------------------------------------------------------------
# Tool functions used in the plan
# ---------------------------------------------------------------------------


async def analyze_customer_message(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate intent and sentiment analysis of the customer's message."""

    await asyncio.sleep(0.4)  # Simulate latency from a language model
    message = inputs.get("customer_message", "").lower()

    issue = (
        "delayed shipment"
        if "hasn't arrived" in message or "late" in message
        else "general"
    )
    sentiment = "frustrated" if "nervous" in message or "help" in message else "neutral"

    analysis = {
        "issue": issue,
        "sentiment": sentiment,
        "urgency": "high" if "two weeks" in message else "medium",
        "must_do": ["acknowledge delay", "share status", "offer make-good"],
    }

    return {"analysis": analysis}


async def fetch_order_status(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Pretend to call an order management system for the latest status."""

    await asyncio.sleep(0.6)  # Simulate network/database round trip
    order_id = inputs.get("order_id", "unknown")

    order_details = {
        "order_id": order_id,
        "status": "In transit",
        "latest_checkpoint": "Departed regional hub early this morning",
        "estimated_delivery": "two business days",
        "remedy": "expedited replacement shipment if the parcel stalls again",
    }

    return {"order_details": order_details}


async def load_guidelines(_: Dict[str, Any]) -> Dict[str, Any]:
    """Return the JSON-style guidelines used by quality review."""

    await asyncio.sleep(0.1)  # Simulate loading policy content
    return {"guidelines": SUPPORT_GUIDELINES}


async def design_response_plan(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a structured plan combining analysis with policy guidance."""

    await asyncio.sleep(0.25)
    analysis = inputs["analysis"]
    guidelines = inputs["guidelines"]

    plan: List[Dict[str, str]] = [
        {
            "stage": "greeting",
            "action": "Address the customer by name with a calm, confident tone.",
        },
        {
            "stage": "acknowledgement",
            "action": (
                f"Recognize the {analysis['issue']} and validate the customer's "
                "frustration about waiting."
            ),
        },
        {
            "stage": "status_update",
            "action": "Share the latest tracking insight and set expectations for next steps.",
        },
        {
            "stage": "resolution",
            "action": (
                "Explain the proactive step that aligns with the make-good policy: "
                f"{guidelines['policies']['make_good_offer']}."
            ),
        },
        {
            "stage": "closing",
            "action": "Reassure continued monitoring and invite the customer to reply if needed.",
        },
    ]

    return {"response_plan": plan}


async def compose_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Draft the actual customer response by loosely following the plan."""

    await asyncio.sleep(0.35)
    plan = inputs["response_plan"]
    guidelines = inputs["guidelines"]
    order = inputs["order_details"]
    analysis = inputs["analysis"]
    customer_name = inputs.get("customer_name", "there")

    paragraphs = [
        f"Hi {customer_name},",
        (
            "Thank you for reaching out about your desk order. Waiting this long is "
            f"understandably frustrating, and I'm sorry for the {analysis['issue']}."
        ),
        (
            f"I just checked order {order['order_id']} and it is currently {order['status'].lower()}. "
            f"The latest update shows it {order['latest_checkpoint'].lower()}, so it should arrive within {order['estimated_delivery']}."
        ),
        (
            "To make sure you are covered, I've set up a flag so we can "
            f"{order['remedy']}—that way you're not left waiting if the package stalls again."
        ),
        (
            "I'll keep an eye on this and send another update tomorrow. If anything changes or you have more "
            "questions, just hit reply and I'll jump back in."
        ),
    ]

    closing = guidelines.get("closing")
    if closing:
        paragraphs.append(closing)

    wrapped: List[str] = []
    for paragraph in paragraphs[:-1] if closing else paragraphs:
        wrapped.append(textwrap.fill(paragraph, width=90))
    if closing:
        wrapped.append(closing)

    draft = "\n\n".join(wrapped)

    return {"draft": draft, "applied_structure": [step["stage"] for step in plan]}


async def review_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a lightweight quality review against the guidelines."""

    await asyncio.sleep(0.15)
    draft = inputs["draft"]
    guidelines = inputs["guidelines"]

    structure_followed = inputs.get("applied_structure", []) == guidelines["structure"]
    review_notes = {
        "tone": "empathetic" if "sorry" in draft.lower() else "needs empathy",
        "structure_followed": structure_followed,
        "policy_reference": guidelines["policies"]["make_good_offer"],
    }

    return {"final_response": draft, "review_notes": review_notes}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def print_plan(plan: List[Dict[str, str]]) -> None:
    """Pretty-print the generated response plan."""

    print("Plan Outline:")
    for idx, step in enumerate(plan, start=1):
        print(f"  {idx}. {step['stage'].title()}: {step['action']}")


def print_response(response: str) -> None:
    """Print the final customer-facing response."""

    print("\nCustomer Response:\n")
    print(response)


# ---------------------------------------------------------------------------
# Sequential vs accelerated execution
# ---------------------------------------------------------------------------


async def run_standard_workflow(
    base_inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """Execute the workflow sequentially without Tygent acceleration."""

    print("=== Standard Execution ===")
    start = asyncio.get_event_loop().time()
    combined: Dict[str, Any] = dict(base_inputs)

    for step in (
        analyze_customer_message,
        fetch_order_status,
        load_guidelines,
        design_response_plan,
        compose_response,
        review_response,
    ):
        result = await step(combined)
        combined.update(result)

    duration = asyncio.get_event_loop().time() - start

    print()
    print_plan(combined["response_plan"])
    print_response(combined["final_response"])
    print(f"\nStandard execution time: {duration:.2f} seconds\n")

    return combined, duration


CUSTOMER_SUPPORT_PLAN: Dict[str, Any] = {
    "name": "customer_support_follow_up",
    "steps": [
        {
            "name": "analyze_customer_message",
            "func": analyze_customer_message,
            "critical": True,
        },
        {"name": "fetch_order_status", "func": fetch_order_status},
        {"name": "load_guidelines", "func": load_guidelines},
        {
            "name": "design_response_plan",
            "func": design_response_plan,
            "dependencies": ["analyze_customer_message", "load_guidelines"],
            "critical": True,
        },
        {
            "name": "compose_response",
            "func": compose_response,
            "dependencies": [
                "design_response_plan",
                "fetch_order_status",
                "analyze_customer_message",
            ],
        },
        {
            "name": "review_response",
            "func": review_response,
            "dependencies": ["compose_response", "load_guidelines"],
        },
    ],
}


async def run_accelerated_workflow(
    base_inputs: Dict[str, Any],
    expected_plan: List[Dict[str, str]],
    expected_response: str,
) -> Tuple[Dict[str, Any], float]:
    """Execute the workflow through Tygent's scheduler."""

    print("=== Accelerated Execution ===")
    accelerated_plan = accelerate(CUSTOMER_SUPPORT_PLAN)
    start = asyncio.get_event_loop().time()
    raw_results = await accelerated_plan(base_inputs)
    duration = asyncio.get_event_loop().time() - start

    results = raw_results["results"]
    plan = results["design_response_plan"]["response_plan"]
    final_response = results["review_response"]["final_response"]

    print()
    print_plan(plan)
    print_response(final_response)
    print(f"\nAccelerated execution time: {duration:.2f} seconds")
    print(f"Plans match sequential output: {plan == expected_plan}")
    print(f"Responses match sequential output: {final_response == expected_response}\n")

    return {"response_plan": plan, "final_response": final_response}, duration


async def main() -> None:
    """Run both execution modes and compare their durations."""

    standard_outputs, standard_time = await run_standard_workflow(BASE_CUSTOMER_INPUTS)
    accelerated_outputs, accelerated_time = await run_accelerated_workflow(
        BASE_CUSTOMER_INPUTS,
        standard_outputs["response_plan"],
        standard_outputs["final_response"],
    )

    if standard_time > accelerated_time:
        improvement = ((standard_time - accelerated_time) / standard_time) * 100
        print(f"Acceleration improvement: {improvement:.1f}% faster")
    else:
        print("Acceleration produced similar timing to sequential execution.")

    assert (
        standard_outputs["final_response"] == accelerated_outputs["final_response"]
    ), "Accelerated workflow should match the standard response"


if __name__ == "__main__":
    asyncio.run(main())
