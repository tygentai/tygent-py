"""
Tygent: Transform LLM Agents into High-Performance Engines

Tygent converts agent-generated plans into typed Directed Acyclic Graphs (DAGs)
for optimized execution through critical path analysis. It also supports multi-agent
orchestration with optimized communication patterns.
"""

__version__ = "0.4.0"

from .accelerate import accelerate
from .adaptive_executor import (
    AdaptiveExecutor,
    RewriteRule,
    create_conditional_branch_rule,
    create_fallback_rule,
    create_resource_adaptation_rule,
)
from .agent import Agent, OpenAIAgent
from .audit import audit_dag, audit_plan, audit_plans

# Import core modules
from .dag import DAG
from .multi_agent import CommunicationBus, Message, MultiAgentManager
from .nodes import BaseNode, LLMNode, Node, ToolNode
from .patch import install
from .plan_parser import parse_plan, parse_plans
from .scheduler import Scheduler, StopExecution

__all__ = [
    "DAG",
    "Node",
    "LLMNode",
    "ToolNode",
    "BaseNode",
    "Scheduler",
    "StopExecution",
    "Agent",
    "OpenAIAgent",
    "MultiAgentManager",
    "CommunicationBus",
    "Message",
    "accelerate",
    "AdaptiveExecutor",
    "RewriteRule",
    "create_fallback_rule",
    "create_conditional_branch_rule",
    "create_resource_adaptation_rule",
    "parse_plan",
    "parse_plans",
    "audit_dag",
    "audit_plan",
    "audit_plans",
    "install",
]
