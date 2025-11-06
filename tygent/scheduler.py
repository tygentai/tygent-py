"""
Scheduler for executing DAGs in Tygent.
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence


class StopExecution(Exception):
    """Raised by hooks to halt further DAG execution."""

    pass


from tygent.dag import DAG
from tygent.session import InMemorySessionStore, NodeContext, SessionStore


class TerminationPolicy:
    """Policy that decides whether a cyclic component should continue iterating."""

    def should_continue(
        self,
        iteration: int,
        previous_outputs: Dict[str, Any],
        current_outputs: Dict[str, Any],
    ) -> bool:
        """
        Return True to run another iteration.

        Parameters
        ----------
        iteration:
            Current iteration number (1-based).
        previous_outputs:
            Outputs from the prior iteration. Empty on the first pass.
        current_outputs:
            Outputs produced during the current iteration.
        """

        raise NotImplementedError


class SingleIterationTermination(TerminationPolicy):
    """Default termination policy: run the component exactly once."""

    def should_continue(
        self,
        iteration: int,
        previous_outputs: Dict[str, Any],
        current_outputs: Dict[str, Any],
    ) -> bool:
        return False


class FixedPointTermination(TerminationPolicy):
    """Continue iterating until outputs stabilise or the iteration cap is reached."""

    def __init__(
        self,
        *,
        max_iterations: int = 10,
        comparator: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.comparator = comparator or (lambda a, b: a == b)

    def should_continue(
        self,
        iteration: int,
        previous_outputs: Dict[str, Any],
        current_outputs: Dict[str, Any],
    ) -> bool:
        if iteration >= self.max_iterations:
            return False
        for key, current in current_outputs.items():
            if key not in previous_outputs:
                return True
            if not self.comparator(previous_outputs[key], current):
                return True
        return False


class Scheduler:
    """
    Scheduler for executing DAGs.
    """

    def __init__(
        self,
        dag: DAG,
        audit_file: Optional[str] = None,
        hooks: Optional[List[Callable[..., Optional[bool]]]] = None,
        session_store: Optional[SessionStore] = None,
        termination_policies: Optional[Dict[Sequence[str], TerminationPolicy]] = None,
    ):
        """
        Initialize a scheduler.

        Args:
            dag: The DAG to schedule
            audit_file: Optional path to write audit logs
            hooks: Optional list of callables invoked during node execution
            session_store: Storage backend used to persist intermediate and
                cross-run state.
            termination_policies: Optional mapping of node name collections to
                termination policies applied when their component forms a cycle.
        """
        self.dag = dag
        self.audit_file = audit_file
        self.hooks = hooks or []
        self.session_store = session_store or InMemorySessionStore()
        self._stop = False
        self.max_parallel_nodes = 4
        self.max_execution_time = 60000  # milliseconds
        self.priority_nodes = []
        self.token_budget: Optional[int] = None
        self.tokens_used = 0
        self.requests_per_minute: Optional[int] = None
        self._request_times: List[float] = []
        self._termination_policies: Dict[frozenset[str], TerminationPolicy] = {}
        if termination_policies:
            for nodes, policy in termination_policies.items():
                self.register_termination_policy(nodes, policy)

    def configure(
        self,
        max_parallel_nodes: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        priority_nodes: Optional[List[str]] = None,
        token_budget: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        audit_file: Optional[str] = None,
        hooks: Optional[List[Callable[..., Optional[bool]]]] = None,
        session_store: Optional[SessionStore] = None,
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
        session_store : SessionStore, optional
            Persisted state backend to use for subsequent executions.
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
        if session_store is not None:
            self.session_store = session_store

    def register_termination_policy(
        self,
        nodes: Sequence[str],
        policy: TerminationPolicy,
    ) -> None:
        """
        Register a termination policy for the component containing ``nodes``.

        Parameters
        ----------
        nodes:
            Iterable of node names that identify the cyclic component.
        policy:
            TerminationPolicy implementation controlling iteration.
        """

        key = frozenset(nodes)
        if not key:
            raise ValueError("nodes argument must contain at least one node name")
        self._termination_policies[key] = policy

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

        # Get the execution order. Cycles fall back to insertion order.
        try:
            execution_order = dag.getTopologicalOrder()
        except ValueError:
            execution_order = list(dag.nodes.keys())

        components = dag.get_strongly_connected_components()
        self._apply_cycle_policies(dag)

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
        priority_node_set = set(self.priority_nodes or [])

        def _priority_tuple(node_name: str) -> tuple:
            node = dag.getNode(node_name)
            metadata = getattr(node, "metadata", {}) if node else {}
            tags_value = metadata.get("tags", [])
            if isinstance(tags_value, list):
                tags = tags_value
            elif tags_value:
                tags = [tags_value]
            else:
                tags = []
            level = metadata.get("level", 0)
            is_redundancy = "redundancy" in tags or bool(metadata.get("generated"))
            is_critical = (
                node_name in priority_node_set
                or "critical" in tags
                or metadata.get("is_critical") is True
            )
            cp_score = critical_path.get(node_name, 0)
            try:
                level_value = float(level)
            except (TypeError, ValueError):
                level_value = 0.0
            return (
                0 if is_critical else 1,
                0 if not is_redundancy else 1,
                -cp_score,
                level_value,
                node_name,
            )

        while ready_nodes or waiting_nodes:
            # Prioritize nodes using criticality, redundancy, and graph metrics
            ready_nodes.sort(key=_priority_tuple)

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
                    if not ready_nodes and waiting_nodes:
                        unresolved = {
                            dep
                            for deps in waiting_nodes.values()
                            for dep in deps
                            if dep not in waiting_nodes and dep not in node_outputs
                        }
                        if unresolved:
                            raise ValueError(
                                "Deadlock detected due to unresolved dependencies: "
                                f"{sorted(unresolved)}"
                            )

                        handled_cycle = False
                        for comp_index, component in enumerate(components):
                            if not set(component).issubset(waiting_nodes.keys()):
                                continue
                            if len(component) == 1 and not dag.has_self_loop(
                                component[0]
                            ):
                                continue

                            await self._execute_cyclic_component(
                                dag,
                                component,
                                inputs,
                                node_outputs,
                            )
                            for node_name in component:
                                waiting_nodes.pop(node_name, None)
                            handled_cycle = True

                        if handled_cycle:
                            for node_name, deps in list(waiting_nodes.items()):
                                waiting_nodes[node_name] = [
                                    d for d in deps if d not in node_outputs
                                ]
                                if not waiting_nodes[node_name]:
                                    ready_nodes.append(node_name)
                                    del waiting_nodes[node_name]
                            # Re-evaluate scheduling after handling cycles
                            continue

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
        *,
        context: Optional[NodeContext] = None,
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
            # Normalize outputs so simple types map to their dependency name
            if not isinstance(dep_output, dict):
                dep_output = {dep_name: dep_output}

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

            # Expose dependency outputs under their step name for prompt templating
            node_inputs.setdefault(dep_name, dep_output)

        context = context or NodeContext(
            node_name=node.name,
            session=self.session_store,
            metadata=dict(getattr(node, "metadata", {}) or {}),
        )

        await node.prepare(context)

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
            result = await asyncio.wait_for(
                node.run(node_inputs, context), timeout=timeout
            )
            await self._run_hooks("after_execute", node, node_inputs, result)
            await node.finalize(context, result)
            if not context.state_saved:
                context.save_state(result)
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

    async def _execute_cyclic_component(
        self,
        dag: DAG,
        component_nodes: Sequence[str],
        inputs: Dict[str, Any],
        node_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a strongly connected component until its termination policy halts.
        """

        component_set = set(component_nodes)
        policy = self._get_termination_policy(component_nodes)
        memory: Dict[str, Any] = {}
        _missing = object()
        for name in component_nodes:
            if name in node_outputs:
                memory[name] = node_outputs[name]
            else:
                stored = self.session_store.get_node_state(name, _missing)
                if stored is not _missing:
                    memory[name] = stored

        def _dependency_payload(
            deps_inside: List[str],
            deps_outside: List[str],
            current_outputs: Dict[str, Any],
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {}
            for dep in deps_inside:
                if dep in current_outputs:
                    payload[dep] = current_outputs[dep]
                elif dep in memory:
                    payload[dep] = memory[dep]
            payload.update({dep: node_outputs[dep] for dep in deps_outside})
            return payload

        iteration = 0
        while True:
            iteration += 1
            current_iteration_outputs: Dict[str, Any] = {}
            pending = list(component_nodes)

            while pending:
                progress = False
                for node_name in list(pending):
                    node = dag.getNode(node_name)
                    if not node:
                        pending.remove(node_name)
                        continue

                    deps_inside = [
                        dep for dep in node.dependencies if dep in component_set
                    ]
                    deps_outside = [
                        dep for dep in node.dependencies if dep not in component_set
                    ]

                    missing_outside = [
                        dep for dep in deps_outside if dep not in node_outputs
                    ]
                    if missing_outside:
                        raise ValueError(
                            f"External dependencies {missing_outside} unresolved for node {node_name}"
                        )

                    if all(dep in current_iteration_outputs for dep in deps_inside):
                        dependency_outputs = _dependency_payload(
                            deps_inside,
                            deps_outside,
                            current_iteration_outputs,
                        )
                    else:
                        continue

                    node_context = NodeContext(
                        node_name=node_name,
                        session=self.session_store,
                        iteration=iteration,
                        metadata=dict(getattr(node, "metadata", {}) or {}),
                    )
                    result = await self._execute_node(
                        dag,
                        node,
                        inputs,
                        dependency_outputs,
                        context=node_context,
                    )
                    current_iteration_outputs[node_name] = result
                    pending.remove(node_name)
                    progress = True

                if not progress and pending:
                    # Break the cycle using memoised values for any remaining deps
                    node_name = pending[0]
                    node = dag.getNode(node_name)
                    if not node:
                        pending.remove(node_name)
                        continue

                    deps_inside = [
                        dep for dep in node.dependencies if dep in component_set
                    ]
                    deps_outside = [
                        dep for dep in node.dependencies if dep not in component_set
                    ]

                    missing_outside = [
                        dep for dep in deps_outside if dep not in node_outputs
                    ]
                    if missing_outside:
                        raise ValueError(
                            f"External dependencies {missing_outside} unresolved for node {node_name}"
                        )

                    dependency_outputs = _dependency_payload(
                        deps_inside,
                        deps_outside,
                        current_iteration_outputs,
                    )

                    node_context = NodeContext(
                        node_name=node_name,
                        session=self.session_store,
                        iteration=iteration,
                        metadata=dict(getattr(node, "metadata", {}) or {}),
                    )
                    result = await self._execute_node(
                        dag,
                        node,
                        inputs,
                        dependency_outputs,
                        context=node_context,
                    )
                    current_iteration_outputs[node_name] = result
                    pending.remove(node_name)

            previous_outputs = memory.copy()
            memory.update(current_iteration_outputs)
            node_outputs.update(current_iteration_outputs)

            if not policy.should_continue(
                iteration, previous_outputs, current_iteration_outputs
            ):
                return current_iteration_outputs

    def _apply_cycle_policies(self, dag: DAG) -> None:
        """Register termination policies declared on the DAG metadata."""

        metadata = getattr(dag, "metadata", {}) or {}
        policies = metadata.get("cycle_policies", {})
        if not isinstance(policies, dict):
            return

        for nodes_tuple, spec in policies.items():
            try:
                nodes = list(nodes_tuple)
            except TypeError:
                continue
            key = frozenset(nodes)
            if key in self._termination_policies and not (spec or {}).get(
                "override", False
            ):
                continue
            policy = self._policy_from_spec(spec or {})
            self._termination_policies[key] = policy

    def _policy_from_spec(self, spec: Dict[str, Any]) -> TerminationPolicy:
        """Instantiate a termination policy from a plan specification."""

        policy_type = str(spec.get("type", "single_pass")).lower()
        if policy_type in {"single", "single_pass", "once"}:
            return SingleIterationTermination()
        if policy_type in {"fixed_point", "fixed", "stabilise", "stabilize"}:
            max_iterations = spec.get("max_iterations")
            try:
                max_iter_value = (
                    int(max_iterations) if max_iterations is not None else 10
                )
            except (TypeError, ValueError):
                max_iter_value = 10
            return FixedPointTermination(max_iterations=max_iter_value)
        return SingleIterationTermination()

    def _get_termination_policy(
        self, component_nodes: Sequence[str]
    ) -> TerminationPolicy:
        """Return a termination policy for the component or the default policy."""

        key = frozenset(component_nodes)
        policy = self._termination_policies.get(key)
        if policy:
            return policy
        return SingleIterationTermination()
