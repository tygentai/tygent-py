"""
Multi-agent support for Tygent.

This module provides classes for managing multiple agents working together:
- MultiAgentManager: Creates and manages multiple agents
- CommunicationBus: Handles inter-agent communication
- Message: Represents messages passed between agents
"""

import time
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Tuple

from .agent import Agent
from .dag import DAG
from .nodes import BaseNode, ToolNode, LLMNode

class Message:
    """
    Represents a message passed between agents.
    """
    
    def __init__(self, from_agent: str, to_agent: str, content: str, message_type: str = "standard"):
        """
        Initialize a message.
        
        Args:
            from_agent: ID of the agent sending the message
            to_agent: ID of the agent receiving the message
            content: The message content
            message_type: Type of message ("standard", "request", "response", "broadcast")
        """
        self.id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.content = content
        self.timestamp = time.time()
        self.message_type = message_type
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "content": self.content,
            "timestamp": self.timestamp,
            "type": self.message_type
        }


class CommunicationBus:
    """
    Handles communication between agents.
    """
    
    def __init__(self):
        """
        Initialize the communication bus.
        """
        self.message_queue: List[Message] = []
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        
    def publish(self, message: Message) -> None:
        """
        Publish a message to the bus.
        
        Args:
            message: The message to publish
        """
        self.message_queue.append(message)
        
        # Deliver to specific agent if targeted
        if message.to_agent in self.subscribers:
            for callback in self.subscribers[message.to_agent]:
                callback(message)
                
        # Deliver to broadcast subscribers if it's a broadcast
        if message.message_type == "broadcast" and "*" in self.subscribers:
            for callback in self.subscribers["*"]:
                callback(message)
                
    def subscribe(self, agent_id: str, callback: Callable[[Message], None]) -> None:
        """
        Subscribe to messages.
        
        Args:
            agent_id: The agent ID to subscribe for, or "*" for all broadcasts
            callback: Function to call when a message is received
        """
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        
    def get_messages(self, agent_id: Optional[str] = None, limit: Optional[int] = None) -> List[Message]:
        """
        Get messages from the queue.
        
        Args:
            agent_id: Optional agent ID to filter messages for
            limit: Optional limit on number of messages to return
            
        Returns:
            List of messages
        """
        if agent_id is None:
            messages = self.message_queue
        else:
            messages = [
                msg for msg in self.message_queue 
                if msg.to_agent == agent_id or 
                (msg.message_type == "broadcast" and msg.from_agent != agent_id)
            ]
            
        if limit is not None:
            return messages[-limit:]
        return messages


class AgentRole:
    """
    Defines a role for an agent in a multi-agent system.
    """
    
    def __init__(self, name: str, description: str, system_prompt: str):
        """
        Initialize an agent role.
        
        Args:
            name: Role name (e.g., "Researcher", "Critic")
            description: Role description
            system_prompt: System prompt for the agent in this role
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt


class OptimizationSettings:
    """
    Settings for optimizing multi-agent interactions.
    """
    
    def __init__(self, 
                 batch_messages: bool = False, 
                 parallel_thinking: bool = True,
                 shared_memory: bool = True,
                 early_stop_threshold: float = 0.0):
        """
        Initialize optimization settings.
        
        Args:
            batch_messages: Whether to batch messages between agents
            parallel_thinking: Whether agents can think in parallel
            shared_memory: Whether agents share memory
            early_stop_threshold: Threshold for early stopping (0.0 = disabled)
        """
        self.batch_messages = batch_messages
        self.parallel_thinking = parallel_thinking
        self.shared_memory = shared_memory
        self.early_stop_threshold = early_stop_threshold


class MultiAgentManager:
    """
    Manages multiple agents working together.
    """
    
    def __init__(self, planning_model: str = "gpt-4o"):
        """
        Initialize a multi-agent manager.
        
        Args:
            planning_model: LLM model to use for planning
        """
        self.agents: Dict[str, Agent] = {}
        self.agent_roles: Dict[str, AgentRole] = {}
        self.communication_bus = CommunicationBus()
        self.planning_model = planning_model
        
    def add_agent(self, agent_id: str, role: AgentRole) -> Agent:
        """
        Add an agent with a specific role.
        
        Args:
            agent_id: The ID for the agent
            role: The role for the agent
            
        Returns:
            The created agent
        """
        agent = Agent(
            name=f"{role.name} ({agent_id})",
            planning_enabled=True,
            planning_model=self.planning_model
        )
        
        self.agents[agent_id] = agent
        self.agent_roles[agent_id] = role
        
        # Set up message handling
        self.communication_bus.subscribe(agent_id, self._handle_message)
        
        return agent
    
    def _handle_message(self, message: Message) -> None:
        """
        Handle a message received by an agent.
        
        Args:
            message: The message received
        """
        # In a full implementation, this would update the agent's context
        # and potentially trigger responses
        pass
    
    def create_conversation_dag(self, 
                               query: str, 
                               optimization_settings: Optional[OptimizationSettings] = None) -> DAG:
        """
        Create a DAG representing the conversation flow.
        
        Args:
            query: The initial query/topic
            optimization_settings: Optional optimization settings
            
        Returns:
            A DAG representing the conversation
        """
        if optimization_settings is None:
            optimization_settings = OptimizationSettings()
            
        dag = DAG(f"conversation_{str(uuid.uuid4())[:8]}")
        
        # Create input node for the query
        input_node = ToolNode("input", lambda x: {"query": query})
        dag.add_node(input_node)
        
        # Create agent nodes
        agent_nodes: Dict[str, LLMNode] = {}
        for agent_id, role in self.agent_roles.items():
            # Create a function that will process this agent's turn
            def create_agent_function(agent_id):
                async def agent_function(inputs):
                    agent = self.agents[agent_id]
                    role = self.agent_roles[agent_id]
                    
                    # Construct prompt with context from inputs
                    query = inputs.get("query", "")
                    context = inputs.get("context", "")
                    
                    prompt = f"""
                    You are acting as {role.name}. {role.description}
                    
                    {role.system_prompt}
                    
                    Here is the question or topic:
                    {query}
                    
                    {f"Here is additional context: {context}" if context else ""}
                    """
                    
                    # In a real implementation, this would use the agent to generate a response
                    # For now, we'll just return a placeholder
                    return {
                        "agent_id": agent_id,
                        "response": f"Response from {role.name} about {query}",
                        "timestamp": time.time()
                    }
                return agent_function
            
            # Create the node for this agent
            agent_node = LLMNode(
                f"agent_{agent_id}", 
                self.planning_model,
                f"Generate response for {role.name}"
            )
            agent_node.process = create_agent_function(agent_id)
            agent_nodes[agent_id] = agent_node
            dag.add_node(agent_node)
        
        # Create output/aggregation node
        output_node = ToolNode("output", lambda x: x)
        dag.add_node(output_node)
        
        # Create edges based on optimization settings
        if optimization_settings.parallel_thinking:
            # Connect input to all agents in parallel
            for agent_id in self.agent_roles:
                dag.add_edge("input", f"agent_{agent_id}")
                
            # Connect all agents to output
            for agent_id in self.agent_roles:
                dag.add_edge(f"agent_{agent_id}", "output")
        else:
            # Connect in sequence
            prev_node_id = "input"
            for agent_id in self.agent_roles:
                dag.add_edge(prev_node_id, f"agent_{agent_id}")
                prev_node_id = f"agent_{agent_id}"
            
            # Connect last agent to output
            dag.add_edge(prev_node_id, "output")
        
        return dag
    
    async def execute_conversation(self, 
                                 query: str, 
                                 optimization_settings: Optional[OptimizationSettings] = None) -> Dict[str, Any]:
        """
        Execute a multi-agent conversation.
        
        Args:
            query: The initial query/topic
            optimization_settings: Optional optimization settings
            
        Returns:
            The results of the conversation
        """
        # Create DAG for the conversation
        dag = self.create_conversation_dag(query, optimization_settings)
        
        # Execute the DAG
        results = await self._execute_dag(dag, {"query": query})
        
        return results
    
    async def _execute_dag(self, dag: DAG, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a DAG for multi-agent conversation.
        
        Args:
            dag: The DAG to execute
            initial_inputs: Initial inputs for the DAG
            
        Returns:
            The results of executing the DAG
        """
        # Topologically sort nodes
        node_order = dag.get_topological_order()
        
        # Results for each node
        node_results: Dict[str, Any] = {}
        
        # Start with input node
        node_results["input"] = initial_inputs
        
        # Process nodes in order
        for node_id in node_order:
            if node_id == "input":
                continue  # Already processed
                
            node = dag.nodes[node_id]
            
            # Get inputs for this node from previous nodes
            inputs = dag.get_node_inputs(node_id, node_results)
            
            # Execute the node
            if hasattr(node, "process") and callable(node.process):
                result = await node.process(inputs)
                node_results[node_id] = result
        
        return node_results
    
    def find_critical_path(self, dag: DAG) -> List[str]:
        """
        Find the critical path in the DAG.
        
        Args:
            dag: The DAG to analyze
            
        Returns:
            List of node IDs in the critical path
        """
        # For simplicity, we'll estimate each agent node takes 1 time unit
        node_durations = {
            node_id: 1.0 if node_id.startswith("agent_") else 0.1
            for node_id in dag.nodes
        }
        
        # Topological sort
        topo_order = dag.get_topological_order()
        
        # Earliest completion time for each node
        earliest_completion: Dict[str, float] = {}
        
        # Predecessor on critical path
        predecessor: Dict[str, Optional[str]] = {}
        
        # Initialize earliest completion times for all nodes
        for node_id in dag.nodes:
            # Start nodes have their own duration as earliest completion time
            if not any(node_id in dag.edges.get(pred_id, []) for pred_id in dag.nodes):
                earliest_completion[node_id] = node_durations[node_id]
                predecessor[node_id] = None
            else:
                earliest_completion[node_id] = 0.0
                predecessor[node_id] = None
        
        # Compute earliest completion times following topological order
        for node_id in topo_order:
            duration = node_durations[node_id]
            
            # Find all incoming edges
            for pred_id in dag.nodes:
                if node_id in dag.edges.get(pred_id, []):
                    pred_time = earliest_completion[pred_id]
                    if pred_time + duration > earliest_completion[node_id]:
                        earliest_completion[node_id] = pred_time + duration
                        predecessor[node_id] = pred_id
        
        # Find the end node with the maximum completion time
        end_nodes = [
            node_id for node_id in dag.nodes
            if not dag.edges.get(node_id, [])
        ]
        
        if not end_nodes:
            return ["input"]  # Default for test cases with no end node
        
        end_node = max(end_nodes, key=lambda x: earliest_completion[x])
        
        # Reconstruct critical path
        critical_path = []
        current = end_node
        
        while current is not None:
            critical_path.append(current)
            current = predecessor[current]
        
        # Reverse to get start-to-end order
        return list(reversed(critical_path))