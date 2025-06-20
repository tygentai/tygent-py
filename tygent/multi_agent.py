"""
Multi-agent implementation for Tygent.
"""

from typing import Dict, List, Any, Optional, TypedDict
import asyncio
import time
import uuid


class Message(TypedDict):
    """Message passed between agents."""

    id: str
    from_agent: str
    to_agent: str
    content: Any
    timestamp: float


class CommunicationBus:
    """
    Communication bus for multi-agent messaging.
    """

    def __init__(self):
        """
        Initialize a communication bus.
        """
        self.messages = []

    async def send(self, sender: str, recipient: str, content: Any) -> None:
        """
        Send a message from one agent to another.

        Args:
            sender: The name of the sending agent
            recipient: The name of the receiving agent
            content: The message content
        """
        message = {
            "from_agent": sender,
            "to_agent": recipient,
            "content": content,
            "timestamp": asyncio.get_event_loop().time(),
        }
        self.messages.append(message)

    async def receive(
        self, recipient: str, since: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Receive messages for the specified agent.

        Args:
            recipient: The name of the receiving agent
            since: Optional timestamp to filter messages

        Returns:
            List of messages for the agent
        """
        if since is None:
            return [m for m in self.messages if m["to_agent"] == recipient]
        else:
            return [
                m
                for m in self.messages
                if m["to_agent"] == recipient and m["timestamp"] > since
            ]


class MultiAgentManager:
    """
    Manager for orchestrating multiple agents.
    """

    def __init__(self, name: str):
        """
        Initialize a multi-agent manager.

        Args:
            name: The name of the manager
        """
        self.name = name
        self.agents = {}
        self.communication_bus = CommunicationBus()

    def add_agent(self, agent_name: str, agent: Any) -> None:
        """
        Add an agent to the manager.

        Args:
            agent_name: The name of the agent
            agent: The agent instance
        """
        self.agents[agent_name] = agent

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all agents with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            Dictionary mapping agent names to their outputs
        """
        # Execute agents in parallel
        tasks = {
            agent_name: asyncio.create_task(agent.execute(inputs))
            for agent_name, agent in self.agents.items()
        }

        # Wait for all agents to complete
        results = {}
        for agent_name, task in tasks.items():
            try:
                results[agent_name] = await task
            except Exception as e:
                results[agent_name] = {"error": str(e)}

        return results
