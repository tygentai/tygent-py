"""
Agent implementation for Tygent.
"""

import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


class Agent:
    """
    Agent for orchestrating execution of tasks.
    """

    def __init__(self, name: str):
        """
        Initialize an agent.

        Args:
            name: The name of the agent
        """
        self.name = name

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the agent with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            The result of the agent execution
        """
        raise NotImplementedError("Subclasses must implement execute()")


class OpenAIAgent(Agent):
    """Agent that delegates tasks to an OpenAI model."""

    def __init__(self, name: str, model: str = "gpt-3.5-turbo") -> None:
        super().__init__(name)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APY_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY or OPENAI_APY_KEY environment variable must be set"
            )
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def execute(self, inputs: Dict[str, Any]) -> str:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            raise ValueError("inputs must include a list of 'messages'")

        response = await self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content
