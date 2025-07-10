"""
Scheduler for executing DAGs in Tygent.
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional


class StopExecution(Exception):
    """Raised by hooks to halt further DAG execution."""

    pass


from tygent.dag import DAG


class Scheduler:
    """
    Scheduler for executing DAGs.
    """

    def __init__(
        self,
        dag: DAG,
        audit_file: Optional[str] = None,
        hooks: Optional[List[Callable[..., Optional[bool]]]] = None,
    ):
        """
        Initialize a scheduler.

        Args:
            dag: The DAG to schedule
            audit_file: Optional path to write audit logs
            hooks: Optional list of callables invoked during node execution
        """
        self.dag = dag
        self.audit_file = audit_file
        self.hooks = hooks or []
        self._stop = False
        self.max_parallel_nodes = 4
        self.max_execution_time = 60000  # milliseconds
        self.priority_nodes = []
        self.token_budget: Optional[int] = None
        self.tokens_used = 0
        self.requests_per_minute: Optional[int] = None
        self._request_times: List[float] = []

    def configure(
        self,
        max_parallel_nodes: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        priority_nodes: Optional[List[str]] = None,
        token_budget: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        audit_file: Optional[str] = None,
        hooks: Optional[List[Callable[..., Optional[bool]]]] = None,
    ) -> None:
        """Configure scheduler parameters.

        This helper existed in the test expectations. Older versions of the
        library lacked it, causing an ``AttributeError`` when integrations tried
        to call ``scheduler.configure``.  The method simply updates the internal
        attributes if values are provided.

        Parameters
        ----------
        max_parallel_nodes : int, optional
            Maximum number of nodes that may run concurrently.
        max_execution_time : int, optional
            Timeout for each node in milliseconds.
        priority_nodes : list of str, optional
            Names of nodes that should run before others.
        token_budget : int, optional
            Maximum allowed token usage for this execution.
        requests_per_minute : int, optional
            Limit on how many nodes may start per 60s window.
        hooks : list of callables, optional
            Functions invoked before and after node execution. Returning ``False``
            or raising :class:`StopExecution` halts further processing.
        """

        if max_parallel_nodes is not None:
            self.max_parallel_nodes = max_parallel_nodes
        if max_execution_time is not None:
            self.max_execution_time = max_execution_time
        if priority_nodes is not None:
            self.priority_nodes = priority_nodes
        if token_budget is not None:
            self.token_budget = token_budget
        if requests_per_minute is not None:
            self.requests_per_minute = requests_per_minute
        if audit_file is not None:
            self.audit_file = audit_file
        if hooks is not None:
            self.hooks = hooks

    async def _run_hooks(
        self,
        stage: str,
        node: Any,
        inputs: Dict[str, Any],
        output: Any,
    ) -> None:
        """Execute hook functions for a given stage."""

        for hook in self.hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    cont = await hook(
                        stage=stage,
                        node=node,
                        inputs=inputs,
                        output=output,
                        scheduler=self,
                    )
                else:
                    cont = hook(
                        stage=stage,
                        node=node,
                        inputs=inputs,
                        output=output,
                        scheduler=self,
                    )
                if cont is False:
                    self._stop = True
            except StopExecution:
                self._stop = True
            except Exception:
                continue
            if self._stop:
                break

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

        # Reset per-run counters
        self.tokens_used = 0
        self._request_times = []

        # Get the execution order
        execution_order = dag.getTopologicalOrder()

        # Pre-compute critical path lengths for latency-aware scheduling
        try:
            critical_path = dag.compute_critical_path()
        except Exception:
            critical_path = {name: 0.0 for name in execution_order}

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
            # Prioritize nodes by critical-path latency
            ready_nodes.sort(key=lambda n: critical_path.get(n, 0), reverse=True)

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
            try:
                results = await asyncio.gather(*tasks)
            except StopExecution:
                return {"results": node_outputs}

            # Store results
            for node_name, result in zip(current_batch, results):
                node_outputs[node_name] = result

            if self._stop:
                break

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

        # Enforce token budget if configured
        if self.token_budget is not None:
            cost = getattr(node, "token_cost", 0)
            if self.tokens_used + cost > self.token_budget:
                raise RuntimeError(f"Token budget exceeded when executing {node.name}")
            self.tokens_used += cost

        # Rate limiting based on start times
        if self.requests_per_minute is not None:
            now = time.monotonic()
            self._request_times = [t for t in self._request_times if now - t < 60]
            if len(self._request_times) >= self.requests_per_minute:
                wait_for = 60 - (now - self._request_times[0])
                if wait_for > 0:
                    await asyncio.sleep(wait_for)
                now = time.monotonic()
                self._request_times = [t for t in self._request_times if now - t < 60]
            self._request_times.append(now)

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

        await self._run_hooks("before_execute", node, node_inputs, None)
        if self._stop:
            raise StopExecution()

        start = time.time()
        status = "success"
        result: Any = None

        # Set timeout based on max_execution_time
        try:
            # Convert milliseconds to seconds for asyncio
            timeout = self.max_execution_time / 1000.0

            # Execute with timeout
            result = await asyncio.wait_for(node.execute(node_inputs), timeout=timeout)
            await self._run_hooks("after_execute", node, node_inputs, result)
            return result

        except asyncio.TimeoutError:
            # Handle timeout
            status = "timeout"
            raise TimeoutError(
                f"Node {node.name} execution timed out after {self.max_execution_time}ms"
            )
        except Exception as e:
            status = f"error: {e}"
            # Handle other exceptions
            raise RuntimeError(f"Error executing node {node.name}: {str(e)}")
        finally:
            if self.audit_file:
                entry = {
                    "node": node.name,
                    "start": start,
                    "end": time.time(),
                    "status": status,
                    "inputs": node_inputs,
                    "output": result,
                }
                try:
                    with open(self.audit_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception:
                    pass
            await self._run_hooks("audit", node, node_inputs, result)
            # Hook may request stopping further execution
