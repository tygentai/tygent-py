"""
Tygent: Transform LLM Agents into High-Performance Engines

Tygent converts agent-generated plans into typed Directed Acyclic Graphs (DAGs) 
for optimized execution through critical path analysis.
"""

__version__ = "0.1.0"

# Import core modules
from .dag import DAG
from .nodes import BaseNode, LLMNode, ToolNode, MemoryNode
from .scheduler import Scheduler, AdaptiveExecutor
from .agent import Agent

__all__ = [
    "DAG",
    "BaseNode", 
    "LLMNode", 
    "ToolNode", 
    "MemoryNode",
    "Scheduler",
    "AdaptiveExecutor",
    "Agent"
]