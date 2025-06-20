"""
Fixed tests for the multi-agent module that match the actual implementation.
"""

import unittest
from unittest.mock import AsyncMock
import asyncio
import time
import sys
import os
import uuid

# Add the parent directory to the path so we can import tygent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tygent.multi_agent import Message, CommunicationBus, MultiAgentManager


class TestMessage(unittest.TestCase):
    """Tests for the Message TypedDict."""

    def test_message_creation(self):
        """Test that messages can be created with the correct structure."""
        message: Message = {
            "id": str(uuid.uuid4()),
            "from_agent": "agent1",
            "to_agent": "agent2",
            "content": "Hello, agent2!",
            "timestamp": time.time(),
        }

        self.assertEqual(message["from_agent"], "agent1")
        self.assertEqual(message["to_agent"], "agent2")
        self.assertEqual(message["content"], "Hello, agent2!")
        self.assertTrue("id" in message)
        self.assertTrue("timestamp" in message)
        self.assertIsInstance(message["timestamp"], float)

    def test_message_with_complex_content(self):
        """Test that messages can contain complex content."""
        complex_content = {
            "text": "Complex message",
            "data": [1, 2, 3],
            "metadata": {"priority": "high"},
        }

        message: Message = {
            "id": "test-id-123",
            "from_agent": "sender",
            "to_agent": "receiver",
            "content": complex_content,
            "timestamp": 1234567890.0,
        }

        self.assertEqual(message["content"]["text"], "Complex message")
        self.assertEqual(message["content"]["data"], [1, 2, 3])
        self.assertEqual(message["content"]["metadata"]["priority"], "high")


class TestCommunicationBus(unittest.TestCase):
    """Tests for the CommunicationBus class."""

    def setUp(self):
        """Set up test fixtures."""
        self.bus = CommunicationBus()

    def test_bus_initialization(self):
        """Test that communication bus initializes correctly."""
        self.assertIsInstance(self.bus.messages, list)
        self.assertEqual(len(self.bus.messages), 0)

    def test_send_message(self):
        """Test sending messages through the bus."""

        # Use asyncio.run for async test
        async def run_test():
            await self.bus.send("agent1", "agent2", "Hello!")

            self.assertEqual(len(self.bus.messages), 1)
            message = self.bus.messages[0]
            self.assertEqual(message["from_agent"], "agent1")
            self.assertEqual(message["to_agent"], "agent2")
            self.assertEqual(message["content"], "Hello!")
            self.assertTrue("timestamp" in message)

        asyncio.run(run_test())

    def test_receive_messages(self):
        """Test receiving messages from the bus."""

        async def run_test():
            # Send some messages
            await self.bus.send("agent1", "agent2", "Message 1")
            await self.bus.send("agent3", "agent2", "Message 2")
            await self.bus.send("agent1", "agent3", "Message 3")

            # Receive messages for agent2
            messages = await self.bus.receive("agent2")

            # Should receive 2 messages directed to agent2
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["content"], "Message 1")
            self.assertEqual(messages[1]["content"], "Message 2")

        asyncio.run(run_test())

    def test_receive_with_timestamp_filter(self):
        """Test receiving messages with timestamp filtering."""

        async def run_test():
            # Send a message
            await self.bus.send("agent1", "agent2", "Old message")

            # Get current time using event loop time (matching implementation)
            cutoff_time = asyncio.get_event_loop().time()

            # Wait a bit and send another message
            await asyncio.sleep(0.01)
            await self.bus.send("agent1", "agent2", "New message")

            # Receive messages since cutoff time
            recent_messages = await self.bus.receive("agent2", since=cutoff_time)

            # Should only get the new message
            self.assertEqual(len(recent_messages), 1)
            self.assertEqual(recent_messages[0]["content"], "New message")

        asyncio.run(run_test())


class TestMultiAgentManager(unittest.TestCase):
    """Tests for the MultiAgentManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MultiAgentManager("test_manager")

    def test_manager_initialization(self):
        """Test that manager initializes correctly."""
        self.assertIsInstance(self.manager.agents, dict)
        self.assertIsInstance(self.manager.communication_bus, CommunicationBus)
        self.assertEqual(len(self.manager.agents), 0)

    def test_add_agent(self):
        """Test adding agents to the manager."""
        # Start with no agents registered
        self.assertEqual(len(self.manager.agents), 0)

        # Add a dummy agent and verify it is tracked
        self.manager.add_agent("dummy", object())
        self.assertIn("dummy", self.manager.agents)
        self.assertIsInstance(self.manager.agents, dict)

    def test_manager_basic_functionality(self):
        """Test basic manager functionality."""
        # Test that manager can maintain state
        self.assertIsInstance(self.manager.agents, dict)
        self.assertIsInstance(self.manager.communication_bus, CommunicationBus)

        # Test that communication bus is working
        async def run_test():
            await self.manager.communication_bus.send("agent1", "agent2", "Hello")
            messages = await self.manager.communication_bus.receive("agent2")
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["content"], "Hello")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
