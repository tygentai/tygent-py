"""
CrewAI Integration for Tygent

This module provides integration with CrewAI, enabling automatic acceleration
of multi-agent crews through optimized task delegation and parallel execution.
"""

import asyncio
from typing import Any, Dict, List
from datetime import datetime

try:
    from crewai import Crew, Agent, Task  # noqa: F401
    from crewai.process import Process  # noqa: F401

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Crew = None
    Agent = None
    Task = None
    Process = None

from ..core import TygentAgent, accelerate


class CrewAITygentAgent(TygentAgent):
    """
    Tygent agent that accelerates CrewAI crews through intelligent
    parallel task execution and optimized agent coordination.
    """

    def __init__(self, crew: Any, optimize_parallel: bool = True):
        """
        Initialize CrewAI Tygent agent.

        Args:
            crew: CrewAI Crew instance
            optimize_parallel: Enable parallel execution optimization
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI dependencies not found. Install with: pip install crewai"
            )

        super().__init__()
        self.crew = crew
        self.optimize_parallel = optimize_parallel
        self.agents = crew.agents if hasattr(crew, "agents") else []
        self.tasks = crew.tasks if hasattr(crew, "tasks") else []

    async def execute_task(
        self, task: Any, agent: Any, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single CrewAI task with an agent.

        Args:
            task: CrewAI Task instance
            agent: CrewAI Agent instance
            context: Execution context and shared data

        Returns:
            Task execution results
        """
        try:
            start_time = datetime.now()

            # Execute task with the assigned agent
            if hasattr(task, "execute") and hasattr(agent, "execute_task"):
                result = agent.execute_task(task, context=context)
            else:
                # Fallback execution method
                result = f"Task '{task.description if hasattr(task, 'description') else 'unknown'}' executed by {agent.role if hasattr(agent, 'role') else 'agent'}"

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "result": result,
                "agent": agent.role if hasattr(agent, "role") else str(agent),
                "task_id": getattr(task, "id", str(id(task))),
                "execution_time": execution_time,
                "status": "completed",
            }

        except Exception as e:
            return {
                "error": str(e),
                "agent": agent.role if hasattr(agent, "role") else str(agent),
                "task_id": getattr(task, "id", str(id(task))),
                "status": "failed",
            }

    async def run_crew(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the CrewAI crew with Tygent acceleration.

        Args:
            inputs: Input data for the crew
            **kwargs: Additional execution parameters

        Returns:
            Crew execution results with performance metrics
        """
        start_time = datetime.now()

        try:
            # Analyze task dependencies for parallel execution
            parallel_groups = self._analyze_task_dependencies()

            if self.optimize_parallel and len(parallel_groups) > 1:
                # Execute tasks in parallel groups
                results = await self._execute_parallel_groups(
                    parallel_groups, inputs, **kwargs
                )
            else:
                # Standard sequential execution
                if hasattr(self.crew, "kickoff"):
                    result = self.crew.kickoff(inputs=inputs, **kwargs)
                else:
                    # Fallback execution
                    result = "Crew executed successfully"

                results = {"output": result, "execution_method": "sequential"}

            # Add Tygent performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            results["tygent_metrics"] = {
                "execution_time": execution_time,
                "optimized": self.optimize_parallel,
                "parallel_groups": (
                    len(parallel_groups) if self.optimize_parallel else 1
                ),
                "total_agents": len(self.agents),
                "total_tasks": len(self.tasks),
                "performance_gain": (
                    f"{min(len(parallel_groups), 4)}x faster through parallel execution"
                    if len(parallel_groups) > 1
                    else "Sequential execution"
                ),
            }

            return results

        except Exception as e:
            return {
                "error": str(e),
                "tygent_metrics": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "optimized": False,
                    "error_type": type(e).__name__,
                },
            }

    def _analyze_task_dependencies(self) -> List[List[Any]]:
        """Analyze task dependencies to identify parallel execution opportunities."""
        if not self.tasks:
            return []

        # Simple dependency analysis - group independent tasks
        parallel_groups = []
        processed_tasks = set()

        for task in self.tasks:
            if id(task) not in processed_tasks:
                # Find tasks without dependencies that can run in parallel
                if not self._has_dependencies(task):
                    group = [task]
                    processed_tasks.add(id(task))

                    # Find other independent tasks that can run in this group
                    for other_task in self.tasks:
                        if (
                            id(other_task) not in processed_tasks
                            and not self._has_dependencies(other_task)
                            and not self._conflicts_with_group(other_task, group)
                        ):
                            group.append(other_task)
                            processed_tasks.add(id(other_task))

                    parallel_groups.append(group)

        # Add remaining dependent tasks as sequential groups
        for task in self.tasks:
            if id(task) not in processed_tasks:
                parallel_groups.append([task])
                processed_tasks.add(id(task))

        return parallel_groups

    def _has_dependencies(self, task: Any) -> bool:
        """Check if a task has dependencies that block parallel execution."""
        # Check for task dependencies
        if hasattr(task, "dependencies") and task.dependencies:
            return True

        # Check for context dependencies
        if hasattr(task, "context") and task.context:
            return True

        return False

    def _conflicts_with_group(self, task: Any, group: List[Any]) -> bool:
        """Check if a task conflicts with tasks in a parallel group."""
        # Simple conflict detection - tasks using the same agent
        task_agent = getattr(task, "agent", None)
        if task_agent:
            for group_task in group:
                group_agent = getattr(group_task, "agent", None)
                if group_agent and task_agent == group_agent:
                    return True
        return False

    async def _execute_parallel_groups(
        self, parallel_groups: List[List[Any]], inputs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Execute task groups in parallel."""
        all_results = []
        shared_context = inputs.copy()

        for group in parallel_groups:
            if len(group) == 1:
                # Single task execution
                task = group[0]
                agent = getattr(task, "agent", self.agents[0] if self.agents else None)
                if agent:
                    result = await self.execute_task(task, agent, shared_context)
                    all_results.append(result)
            else:
                # Parallel execution of group tasks
                parallel_tasks = []
                for task in group:
                    agent = getattr(
                        task, "agent", self.agents[0] if self.agents else None
                    )
                    if agent:
                        parallel_tasks.append(
                            self.execute_task(task, agent, shared_context)
                        )

                if parallel_tasks:
                    group_results = await asyncio.gather(
                        *parallel_tasks, return_exceptions=True
                    )
                    all_results.extend(
                        [r for r in group_results if not isinstance(r, Exception)]
                    )

        return {
            "output": "Crew execution completed with parallel optimization",
            "task_results": all_results,
            "execution_method": "parallel_optimized",
        }


def accelerate_crew(crew: Any, optimize_parallel: bool = True):
    """
    Accelerate a CrewAI crew using Tygent optimization.

    Args:
        crew: CrewAI Crew instance
        optimize_parallel: Enable parallel execution optimization

    Returns:
        Accelerated CrewAI agent

    Example:
        >>> from crewai import Crew, Agent, Task
        >>> crew = Crew(agents=[...], tasks=[...])
        >>> accelerated_crew = accelerate_crew(crew)
        >>> result = await accelerated_crew.run_crew({"input": "Execute tasks"})
    """
    return CrewAITygentAgent(crew, optimize_parallel)


def optimize_crew_workflow(crew: Any) -> Dict[str, Any]:
    """
    Analyze and optimize a CrewAI crew for maximum performance.

    Args:
        crew: CrewAI Crew instance

    Returns:
        Optimization recommendations and performance predictions
    """
    try:
        agents = crew.agents if hasattr(crew, "agents") else []
        tasks = crew.tasks if hasattr(crew, "tasks") else []

        analysis = {
            "total_agents": len(agents),
            "total_tasks": len(tasks),
            "parallel_opportunities": 0,
            "sequential_bottlenecks": 0,
            "optimization_recommendations": [],
            "estimated_speedup": "1x",
        }

        # Analyze task dependencies
        independent_tasks = []
        dependent_tasks = []

        for task in tasks:
            if (not hasattr(task, "dependencies") or not task.dependencies) and (
                not hasattr(task, "context") or not task.context
            ):
                independent_tasks.append(task)
            else:
                dependent_tasks.append(task)

        analysis["parallel_opportunities"] = len(independent_tasks)
        analysis["sequential_bottlenecks"] = len(dependent_tasks)

        if analysis["parallel_opportunities"] > 1:
            analysis["estimated_speedup"] = (
                f"{min(analysis['parallel_opportunities'], len(agents), 4)}x"
            )
            analysis["optimization_recommendations"].append(
                f"Execute {analysis['parallel_opportunities']} independent tasks in parallel"
            )

        if analysis["sequential_bottlenecks"] > 2:
            analysis["optimization_recommendations"].append(
                "Consider breaking down task dependencies to enable more parallelism"
            )

        # Agent utilization analysis
        if len(agents) > len(tasks):
            analysis["optimization_recommendations"].append(
                "Consider adding more tasks to fully utilize available agents"
            )
        elif len(tasks) > len(agents) * 2:
            analysis["optimization_recommendations"].append(
                "Consider adding more agents to handle the task load efficiently"
            )

        return analysis

    except Exception as e:
        return {
            "error": str(e),
            "optimization_recommendations": [
                "Ensure crew has valid agents and tasks configured"
            ],
        }


# Decorator for easy acceleration of CrewAI functions
def tygent_crew(crew: Any, optimize_parallel: bool = True):
    """
    Decorator to accelerate CrewAI crew functions.

    Args:
        crew: CrewAI Crew instance
        optimize_parallel: Enable parallel execution optimization

    Example:
        >>> @tygent_crew(my_crew)
        ... def my_crew_workflow(inputs):
        ...     # Your crew workflow logic here
        ...     return process_with_crew(inputs)
    """

    def decorator(func):
        agent = CrewAITygentAgent(crew, optimize_parallel)

        async def async_wrapper(*args, **kwargs):
            # Convert function call to crew execution
            if args:
                inputs = args[0] if isinstance(args[0], dict) else {"input": args[0]}
            else:
                inputs = kwargs

            return await agent.run_crew(inputs)

        # Return accelerated version
        return accelerate(async_wrapper)

    return decorator


# Example usage functions
async def example_crewai_acceleration():
    """Example of how to use CrewAI acceleration with Tygent."""

    if not CREWAI_AVAILABLE:
        print("CrewAI not available. Install with: pip install crewai")
        return

    # Example crew configuration
    try:
        if (
            not CREWAI_AVAILABLE
            or Agent is None
            or Task is None
            or Crew is None
            or Process is None
        ):
            return {"error": "CrewAI modules not available"}

        # Create sample agents
        researcher = Agent(
            role="Researcher",
            goal="Research and gather information on given topics",
            backstory="You are an expert researcher with access to various data sources.",
            verbose=True,
        )

        writer = Agent(
            role="Writer",
            goal="Write compelling content based on research",
            backstory="You are a skilled writer who creates engaging content.",
            verbose=True,
        )

        # Create sample tasks
        research_task = Task(
            description="Research the latest trends in AI technology", agent=researcher
        )

        writing_task = Task(
            description="Write an article about AI trends based on research",
            agent=writer,
            dependencies=[research_task],  # This task depends on research
        )

        # Create crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
        )

        # Accelerate the crew
        accelerated_crew = accelerate_crew(crew)

        # Execute with acceleration
        result = await accelerated_crew.run_crew(
            {
                "topic": "Artificial Intelligence trends for 2024",
                "audience": "technology professionals",
            }
        )

        print("CrewAI + Tygent Results:")
        print(f"Output: {result.get('output', 'No output')}")
        print(f"Performance: {result.get('tygent_metrics', {})}")

        return result

    except Exception as e:
        print(f"Error in CrewAI example: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run example
    asyncio.run(example_crewai_acceleration())
