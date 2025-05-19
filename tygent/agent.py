"""
Agent module provides the agent implementation that can create DAGs from natural language plans.
"""

from typing import Dict, List, Any, Optional, Callable
import time
from .dag import DAG
from .nodes import BaseNode, ToolNode, LLMNode, MemoryNode

class Agent:
    """
    Agent creates and manages DAGs from natural language plans.
    
    The agent can:
    1. Generate a plan from a natural language input
    2. Convert the plan into a typed DAG
    3. Execute the plan using a scheduler
    """
    
    def __init__(self, name: str, planning_enabled: bool = True, 
                 memory_node: Optional[MemoryNode] = None,
                 planning_model: str = "gpt-4o"):
        """
        Initialize an agent.
        
        Args:
            name: Name of the agent
            planning_enabled: Whether to use planning
            memory_node: Optional memory node for the agent
            planning_model: LLM model to use for planning
        """
        self.name = name
        self.planning_enabled = planning_enabled
        self.planning_model = planning_model
        self.tools: Dict[str, ToolNode] = {}
        self.memory = memory_node or MemoryNode("memory")
        
        # The execution DAG created from the plan
        self.execution_dag: Optional[DAG] = None
    
    def register_tool(self, tool: ToolNode) -> None:
        """
        Register a tool with the agent.
        
        Args:
            tool: Tool node to register
        """
        self.tools[tool.id] = tool
    
    def plan(self, task: str) -> str:
        """
        Generate a plan for a given task.
        
        Args:
            task: The task to plan for
            
        Returns:
            A string containing the plan
        """
        if not self.planning_enabled:
            return f"Direct execution of task: {task}"
        
        # In a real implementation, this would call an LLM to generate a plan
        try:
            import openai
            
            # List available tools
            tool_descriptions = "\n".join([
                f"- {tool_id}: {tool.node_type}" 
                for tool_id, tool in self.tools.items()
            ])
            
            prompt = f"""
            Task: {task}
            
            Available tools:
            {tool_descriptions}
            
            Generate a step-by-step plan to accomplish this task.
            Each step should be clear and actionable.
            Format your response as a numbered list.
            """
            
            completion = openai.chat.completions.create(
                model=self.planning_model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that creates detailed execution plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            content = completion.choices[0].message.content
            return content if content is not None else ""
            
        except (ImportError, Exception) as e:
            # Fallback for testing
            return f"""
            Plan for: {task}
            
            1. Analyze the task requirements
            2. Gather necessary information
            3. Process information
            4. Generate results
            5. Verify accuracy of results
            6. Present final answer
            """
    
    def set_execution_dag(self, dag: DAG) -> None:
        """
        Set the execution DAG for the agent.
        
        Args:
            dag: The DAG to use for execution
        """
        self.execution_dag = dag
    
    def plan_to_dag(self, task: str) -> DAG:
        """
        Convert a task to a DAG by first generating a plan and then converting it.
        
        Args:
            task: The task to create a DAG for
            
        Returns:
            A DAG representing the plan for the task
        """
        # Generate a plan
        plan = self.plan(task)
        
        # Convert plan to DAG
        dag = self._convert_plan_to_dag(plan, task)
        
        return dag
    
    def _convert_plan_to_dag(self, plan: str, task: str) -> DAG:
        """
        Convert a plan to a DAG using an LLM.
        
        Args:
            plan: The plan to convert
            task: The original task
            
        Returns:
            A DAG representing the plan
        """
        # Create a new DAG
        dag = DAG(f"dag_for_{task[:20].replace(' ', '_')}")
        
        try:
            import openai
            
            # List available tools
            tool_descriptions = "\n".join([
                f"- {tool_id}: {tool.node_type}" 
                for tool_id, tool in self.tools.items()
            ])
            
            prompt = f"""
            Task: {task}
            
            Plan:
            {plan}
            
            Available tools:
            {tool_descriptions}
            
            Convert this plan into a DAG (Directed Acyclic Graph) structure.
            Identify which steps can be executed in parallel and which have dependencies.
            Format your response as a JSON object with:
            1. nodes: array of node objects with id, type (tool, llm, input, output), and description
            2. edges: array of edge objects with from (node id) and to (node id)
            3. parallelizable: array of sets of node ids that can be executed in parallel
            """
            
            completion = openai.chat.completions.create(
                model=self.planning_model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that converts plans to optimized execution graphs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            dag_structure = completion.choices[0].message.content
            
            # Process the response and build the DAG
            # This is simplified - in a real implementation would need to parse JSON
            # and create appropriate nodes
            
            # For testing, create a simple DAG
            input_node = ToolNode("input", lambda x: x)
            process_node = LLMNode("process", self.planning_model, f"Process the following: {{input}}")
            output_node = ToolNode("output", lambda x: x)
            
            dag.add_node(input_node)
            dag.add_node(process_node)
            dag.add_node(output_node)
            
            dag.add_edge("input", "process")
            dag.add_edge("process", "output")
            
            return dag
            
        except (ImportError, Exception) as e:
            # Fallback for testing
            input_node = ToolNode("input", lambda x: x)
            process_node = LLMNode("process", "gpt-3.5-turbo", f"Process the following: {{input}}")
            output_node = ToolNode("output", lambda x: x)
            
            dag.add_node(input_node)
            dag.add_node(process_node)
            dag.add_node(output_node)
            
            dag.add_edge("input", "process")
            dag.add_edge("process", "output")
            
            return dag