"""
Adaptive Executor with dynamic DAG modification capabilities.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
from .dag import DAG
from .scheduler import Scheduler


class RewriteRule:
    """Rule for dynamic DAG modification during execution."""

    def __init__(
        self,
        trigger: Callable[[Dict[str, Any]], bool],
        action: Callable[[DAG, Dict[str, Any]], DAG],
        name: str = "unnamed_rule",
    ):
        """
        Initialize a rewrite rule.

        Args:
            trigger: Function that returns True when rule should activate
            action: Function that modifies the DAG and returns the new DAG
            name: Human-readable name for the rule
        """
        self.trigger = trigger
        self.action = action
        self.name = name


class AdaptiveExecutor:
    """
    Executor that can dynamically modify DAGs during runtime based on intermediate results.
    """

    def __init__(
        self,
        base_dag: DAG,
        rewrite_rules: List[RewriteRule] = None,
        max_modifications: int = 5,
    ):
        """
        Initialize adaptive executor.

        Args:
            base_dag: Initial DAG structure
            rewrite_rules: List of rules for dynamic modification
            max_modifications: Maximum number of DAG modifications per execution
        """
        self.base_dag = base_dag
        self.rewrite_rules = rewrite_rules or []
        self.max_modifications = max_modifications
        # Initialize the scheduler with the base DAG so it is ready to execute
        # immediately when no modifications are required.
        self.scheduler = Scheduler(self.base_dag)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute DAG with dynamic modification capabilities.

        Args:
            inputs: Initial execution inputs

        Returns:
            Execution results including modification history
        """
        current_dag = self.base_dag.copy()
        modification_count = 0
        modification_history = []
        execution_state = {"inputs": inputs}

        while modification_count < self.max_modifications:
            # Execute current DAG
            try:
                result = await self.scheduler.execute(
                    current_dag, execution_state.get("inputs", inputs)
                )
                execution_state.update(result)

                # Check if any rewrite rules should trigger
                triggered_rules = []
                for rule in self.rewrite_rules:
                    try:
                        if rule.trigger(execution_state):
                            triggered_rules.append(rule)
                    except Exception as e:
                        print(f"Error evaluating rule {rule.name}: {e}")
                        continue

                # If no rules triggered, execution is complete
                if not triggered_rules:
                    break

                # Apply the first triggered rule
                rule_to_apply = triggered_rules[0]
                try:
                    new_dag = rule_to_apply.action(current_dag, execution_state)
                    modification_history.append(
                        {
                            "rule_name": rule_to_apply.name,
                            "modification_count": modification_count + 1,
                            "trigger_state": execution_state.copy(),
                        }
                    )
                    current_dag = new_dag
                    modification_count += 1

                    print(f"Applied rule '{rule_to_apply.name}' - DAG modified")

                except Exception as e:
                    print(f"Error applying rule {rule_to_apply.name}: {e}")
                    break

            except Exception as e:
                print(f"Execution error: {e}")
                break

        # Return final results with modification history
        return {
            **execution_state,
            "modification_history": modification_history,
            "final_dag": current_dag,
            "total_modifications": modification_count,
        }


# Common rewrite rule factories
def create_fallback_rule(
    error_condition: Callable[[Dict[str, Any]], bool],
    fallback_node_creator: Callable[[DAG, Dict[str, Any]], Any],
    rule_name: str = "fallback_rule",
) -> RewriteRule:
    """
    Create a fallback rule that adds alternative execution paths on errors.

    Args:
        error_condition: Function to detect error conditions
        fallback_node_creator: Function to create fallback nodes
        rule_name: Name for the rule

    Returns:
        RewriteRule for fallback handling
    """

    def add_fallback_action(dag: DAG, state: Dict[str, Any]) -> DAG:
        # Create new DAG with fallback path
        new_dag = dag.copy()
        fallback_node = fallback_node_creator(new_dag, state)
        new_dag.add_node(fallback_node)
        return new_dag

    return RewriteRule(
        trigger=error_condition, action=add_fallback_action, name=rule_name
    )


def create_conditional_branch_rule(
    condition: Callable[[Dict[str, Any]], bool],
    branch_creator: Callable[[DAG, Dict[str, Any]], Any],
    rule_name: str = "conditional_branch",
) -> RewriteRule:
    """
    Create a rule that adds conditional execution branches.

    Args:
        condition: Function to evaluate branching condition
        branch_creator: Function to create new branch nodes
        rule_name: Name for the rule

    Returns:
        RewriteRule for conditional branching
    """

    def add_branch_action(dag: DAG, state: Dict[str, Any]) -> DAG:
        new_dag = dag.copy()
        branch_node = branch_creator(new_dag, state)
        new_dag.add_node(branch_node)
        return new_dag

    return RewriteRule(trigger=condition, action=add_branch_action, name=rule_name)


def create_resource_adaptation_rule(
    resource_checker: Callable[[Dict[str, Any]], bool],
    adaptation_action: Callable[[DAG, Dict[str, Any]], DAG],
    rule_name: str = "resource_adaptation",
) -> RewriteRule:
    """
    Create a rule that adapts execution based on available resources.

    Args:
        resource_checker: Function to check resource availability
        adaptation_action: Function to adapt the DAG
        rule_name: Name for the rule

    Returns:
        RewriteRule for resource adaptation
    """

    return RewriteRule(
        trigger=resource_checker, action=adaptation_action, name=rule_name
    )
