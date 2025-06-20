"""
Scheduler for executing DAGs in Tygent.
"""

import asyncio
from typing import Dict, List, Any, Optional
from tygent.dag import DAG


class Scheduler:
    """
    Scheduler for executing DAGs.
    """

    def __init__(self, dag: DAG):
        """
        Initialize a scheduler.

        Args:
            dag: The DAG to schedule
        """
        self.dag = dag
        self.max_parallel_nodes = 4
        self.max_execution_time = 60000  # milliseconds
        self.priority_nodes = []

    def configure(
        self,
        max_parallel_nodes: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        priority_nodes: Optional[List[str]] = None,
    ) -> None:
        """Configure scheduler parameters.

        This helper existed in the test expectations. Older versions of the
        library lacked it, causing an ``AttributeError`` when integrations tried
        to call ``scheduler.configure``.  The method simply updates the internal
        attributes if values are provided.
        """

        if max_parallel_nodes is not None:
            self.max_parallel_nodes = max_parallel_nodes
        if max_execution_time is not None:
            self.max_execution_time = max_execution_time
        if priority_nodes is not None:
            self.priority_nodes = priority_nodes

    async def execute(
        self,
        dag_or_inputs: Any,
        maybe_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a DAG with the given inputs.

        The scheduler historically accepted only a single ``inputs`` dictionary
        and operated on the DAG passed during initialization.  Integrations in
        this repository, however, call ``execute`` with the DAG as the first
        argument followed by the inputs (and optionally a context dictionary).
        To remain backward compatible this method now accepts both calling
        conventions.

        Args:
            dag_or_inputs: Either a :class:`~tygent.dag.DAG` instance or the
                inputs dictionary when using the legacy signature.
            maybe_inputs: Inputs dictionary when ``dag_or_inputs`` is a DAG, or
                an optional context when using the legacy signature.
            context: Optional context passed when using the newer calling
                convention.

        Returns:
            Dictionary mapping node names to their outputs
        """
        if isinstance(dag_or_inputs, DAG):
            dag = dag_or_inputs
            inputs = maybe_inputs or {}
        else:
            dag = self.dag
            inputs = dag_or_inputs or {}
            context = maybe_inputs if context is None else context

        # Get the execution order
        execution_order = dag.getTopologicalOrder()

        # Prioritize nodes
        if self.priority_nodes:
            # Move priority nodes to the beginning of the list if they are in the execution order
            for node_name in reversed(self.priority_nodes):
                if node_name in execution_order:
                    execution_order.remove(node_name)
                    execution_order.insert(0, node_name)

        # Store node outputs
        node_outputs: Dict[str, Any] = {}

        # Nodes ready for execution
        ready_nodes: List[str] = []

        # Nodes waiting for dependencies
        waiting_nodes: Dict[str, List[str]] = {}

        # Initialize ready and waiting nodes
        for node_name in execution_order:
            node = dag.getNode(node_name)
            if not node:
                continue

            if not node.dependencies:
                # No dependencies, can execute immediately
                ready_nodes.append(node_name)
            else:
                # Has dependencies, must wait
                waiting_nodes[node_name] = list(node.dependencies)

        # Process nodes until all are executed
        while ready_nodes or waiting_nodes:
            # Execute nodes in parallel
            current_batch = ready_nodes[: self.max_parallel_nodes]
            ready_nodes = ready_nodes[self.max_parallel_nodes :]

            if not current_batch:
                # No nodes ready, check if we're deadlocked
                if waiting_nodes:
                    # Get all executed nodes
                    executed = set(node_outputs.keys())
                    # Check if any waiting node can be unblocked
                    for node_name, deps in list(waiting_nodes.items()):
                        # Remove dependencies that have been executed
                        waiting_nodes[node_name] = [
                            d for d in deps if d not in executed
                        ]
                        # If all dependencies are executed, move to ready
                        if not waiting_nodes[node_name]:
                            ready_nodes.append(node_name)
                            del waiting_nodes[node_name]

                    # If no nodes were unblocked, we're deadlocked
                    if not ready_nodes:
                        raise ValueError(
                            f"Deadlock detected in DAG execution. Waiting nodes: {waiting_nodes}"
                        )
                else:
                    # No waiting nodes either, we're done
                    break

                # Continue to next iteration
                continue

            # Create tasks for all nodes in the current batch
            tasks = []
            for node_name in current_batch:
                node = dag.getNode(node_name)
                if not node:
                    continue

                # Create a task for executing the node
                # Find dependencies from node_outputs
                dependency_outputs = {
                    dep: node_outputs[dep]
                    for dep in node.dependencies
                    if dep in node_outputs
                }

                # Create task with combined inputs and dependency outputs
                tasks.append(self._execute_node(dag, node, inputs, dependency_outputs))

            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks)

            # Store results
            for node_name, result in zip(current_batch, results):
                node_outputs[node_name] = result

            # Update waiting nodes
            for node_name, deps in list(waiting_nodes.items()):
                # Remove dependencies that have been executed
                waiting_nodes[node_name] = [d for d in deps if d not in node_outputs]
                # If all dependencies are executed, move to ready
                if not waiting_nodes[node_name]:
                    ready_nodes.append(node_name)
                    del waiting_nodes[node_name]

        # Format the results as expected by the tests
        return {"results": node_outputs}

    async def _execute_node(
        self,
        dag: DAG,
        node: Any,
        inputs: Dict[str, Any],
        dependency_outputs: Dict[str, Any],
    ) -> Any:
        """
        Execute a node with the given inputs and dependency outputs.

        Args:
            node: The node to execute
            inputs: Dictionary of input values
            dependency_outputs: Dictionary of outputs from dependency nodes

        Returns:
            The result of the node execution
        """
        # Combine inputs with mapped fields from dependencies
        node_inputs = inputs.copy()

        # Apply mappings from edge metadata to input fields
        for dep_name, dep_output in dependency_outputs.items():
            # Check if we have a mapping for this dependency
            if (
                dep_name in dag.edge_mappings
                and node.name in dag.edge_mappings[dep_name]
            ):
                mapping = dag.edge_mappings[dep_name][node.name]

                # Apply the mapping to the node inputs
                for source_field, target_field in mapping.items():
                    if source_field in dep_output:
                        node_inputs[target_field] = dep_output[source_field]
            else:
                # No mapping, include all fields
                node_inputs.update(dep_output)

        # Set timeout based on max_execution_time
        try:
            # Convert milliseconds to seconds for asyncio
            timeout = self.max_execution_time / 1000.0

            # Execute with timeout
            result = await asyncio.wait_for(node.execute(node_inputs), timeout=timeout)
            return result

        except asyncio.TimeoutError:
            # Handle timeout
            raise TimeoutError(
                f"Node {node.name} execution timed out after {self.max_execution_time}ms"
            )
        except Exception as e:
            # Handle other exceptions
            raise RuntimeError(f"Error executing node {node.name}: {str(e)}")
