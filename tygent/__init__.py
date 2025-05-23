"""
Tygent: Transform LLM Agents into High-Performance Engines

Tygent converts agent-generated plans into typed Directed Acyclic Graphs (DAGs) 
for optimized execution through critical path analysis. It also supports multi-agent
orchestration with optimized communication patterns.
"""

__version__ = "0.1.0"

# Import core modules
from .dag import DAG
from .nodes import BaseNode, LLMNode, ToolNode, MemoryNode
from .scheduler import Scheduler, AdaptiveExecutor
from .agent import Agent
from .multi_agent import (
    MultiAgentManager, 
    CommunicationBus, 
    Message, 
    AgentRole, 
    OptimizationSettings
)

__all__ = [
    "DAG",
    "BaseNode", 
    "LLMNode", 
    "ToolNode", 
    "MemoryNode",
    "Scheduler",
    "AdaptiveExecutor",
    "Agent",
    "MultiAgentManager",
    "CommunicationBus",
    "Message",
    "AgentRole",
    "OptimizationSettings"
]