"""
Tygent: Transform LLM Agents into High-Performance Engines

Tygent converts agent-generated plans into typed Directed Acyclic Graphs (DAGs)
for optimized execution through critical path analysis. It also supports multi-agent
orchestration with optimized communication patterns.
"""

__version__ = "0.1.0"

# Import core modules
from .dag import DAG
from .nodes import Node, LLMNode, ToolNode, BaseNode
from .scheduler import Scheduler
from .agent import Agent
from .multi_agent import MultiAgentManager, CommunicationBus, Message
from .accelerate import accelerate
from .adaptive_executor import (
    AdaptiveExecutor,
    RewriteRule,
    create_fallback_rule,
    create_conditional_branch_rule,
    create_resource_adaptation_rule,
)

__all__ = [
    "DAG",
    "Node",
    "LLMNode",
    "ToolNode",
    "BaseNode",
    "Scheduler",
    "Agent",
    "MultiAgentManager",
    "CommunicationBus",
    "Message",
    "accelerate",
    "AdaptiveExecutor",
    "RewriteRule",
    "create_fallback_rule",
    "create_conditional_branch_rule",
    "create_resource_adaptation_rule",
]
