"""
Agent implementation for Tygent.
"""

from typing import Dict, List, Any, Optional


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
