"""Example of using LangChain with Google's Gemini 2.5 Pro for market analysis.

This script demonstrates a simple single-task workflow using LangChain.
The model is prompted to produce a comprehensive market intelligence report
for executive leadership. The same chain can be accelerated with Tygent
for comparison.

Requires the ``langchain-google-genai`` package. Install it with:
``pip install langchain langchain-google-genai google-generativeai``
and configure authentication using ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import os
import time

from tygent.accelerate import accelerate

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

try:  # pragma: no cover - optional dependency
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    print(
        "This example requires the langchain, langchain-google-genai, and google-generativeai packages."
    )
    print(
        "Install them with: pip install langchain langchain-google-genai google-generativeai"
    )
    raise SystemExit(1)

if not os.getenv("GOOGLE_API_KEY"):
    print("Set GOOGLE_API_KEY before running this example.")
    raise SystemExit(1)

INSTRUCTION = (
    "You are a strategic analyst preparing a comprehensive market intelligence "
    "report for executive leadership. Research emerging trends, competitive "
    "threats, customer behavior patterns, and market opportunities to guide "
    "major business decisions and investment strategies."
)

BASE_PROMPT = (
    "{task}: Analyze market opportunities, competitive landscape, customer trends, "
    "and regulatory environment across multiple industries to inform strategic "
    "business decisions."
)


def build_chain():
    """Create the LangChain workflow."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        max_output_tokens=64000,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INSTRUCTION),
            ("human", BASE_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def main(task: str) -> None:
    """Run the market analysis example."""

    chain = build_chain()
    print("=== LangChain Google Market Intelligence Example ===\n")

    print("=== Standard Execution ===")
    start = time.perf_counter()
    standard = chain.invoke({"task": task})
    standard_time = time.perf_counter() - start
    print(standard[:500])
    print(f"\nStandard execution took {standard_time:.2f}s")

    print("\n=== Accelerated Execution ===")
    accelerated_chain = accelerate(chain)
    start = time.perf_counter()
    accelerated = accelerated_chain.invoke({"task": task})
    accelerated_time = time.perf_counter() - start
    print(accelerated[:500])
    print(f"\nAccelerated execution took {accelerated_time:.2f}s")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    parser = argparse.ArgumentParser(
        description="LangChain Google Market Intelligence Example"
    )
    parser.add_argument(
        "--task",
        default="Prepare an executive market intelligence summary",
        help="Task description for the market analysis",
    )
    args = parser.parse_args()
    main(task=args.task)
