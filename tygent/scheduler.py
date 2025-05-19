"""
Scheduler module provides execution engines for running DAGs efficiently.
"""

import asyncio
from typing import Dict, List, Any, Set, Optional, Tuple

from .dag import DAG

class Scheduler:
    """
    Scheduler executes a DAG by scheduling nodes in topological order.
    """
    
    def __init__(self, dag: DAG, max_workers: int = 10):
        """
        Initialize a scheduler.
        
        Args:
            dag: The DAG to execute
            max_workers: Maximum number of parallel workers (default: 10)
        """
        self.dag = dag
        self.max_workers = max_workers
    
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the DAG with the given input.
        
        Args:
            input_data: Input data for the DAG
            
        Returns:
            Results from the DAG execution
        """
        # Initialize with the input data
        node_outputs: Dict[str, Any] = {"input": {"data": input_data}}
        executed_nodes: Set[str] = set()
        
        # Get nodes in topological order
        topo_order = self.dag.get_topological_order()
        
        # Track execution times
        execution_times: Dict[str, float] = {}
        import time
        start_time = time.time()
        
        for node_id in topo_order:
            node = self.dag.nodes[node_id]
            
            # Get inputs for this node
            inputs = self.dag.get_node_inputs(node_id, node_outputs)
            
            # Execute the node
            node_start_time = time.time()
            try:
                result = await node.execute(inputs)
                # Store the result
                node_outputs[node_id] = result
            except Exception as e:
                node_outputs[node_id] = {"error": str(e)}
            node_end_time = time.time()
            
            # Record execution time and mark as executed
            execution_times[node_id] = node_end_time - node_start_time  # seconds
            executed_nodes.add(node_id)
        
        end_time = time.time()
        total_time = end_time - start_time  # seconds
        
        # Build and return a comprehensive result
        return {
            "results": node_outputs,
            "execution_times": execution_times,
            "total_time": total_time,
            "executed_nodes": list(executed_nodes)
        }


class AdaptiveExecutor:
    """
    Advanced executor that utilizes parallelism and adapts execution based on conditions.
    """
    
    def __init__(self, dag: DAG, max_workers: int = 10):
        """
        Initialize an adaptive executor.
        
        Args:
            dag: The DAG to execute
            max_workers: Maximum number of parallel workers (default: 10)
        """
        self.dag = dag
        self.max_workers = max_workers
    
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute the DAG with parallel execution where possible.
        
        Args:
            input_data: Input data for the DAG
            
        Returns:
            Results from the DAG execution
        """
        # Initialize with the input data
        node_outputs: Dict[str, Any] = {"input": {"data": input_data}}
        executed_nodes: Set[str] = set()
        
        # Get dependency count for each node
        dependencies: Dict[str, int] = {}
        for node_id in self.dag.nodes:
            dependencies[node_id] = 0
        
        for from_id, to_list in self.dag.edges.items():
            for to_id in to_list:
                dependencies[to_id] = dependencies.get(to_id, 0) + 1
        
        # Nodes with no dependencies can be executed right away
        ready_nodes = [node_id for node_id, dep_count in dependencies.items() if dep_count == 0]
        
        # Track execution times
        execution_times: Dict[str, float] = {}
        import time
        start_time = time.time()
        
        # Continue until all nodes are processed
        while ready_nodes:
            # Process nodes in parallel, limited by max_workers
            batch_size = min(len(ready_nodes), self.max_workers)
            batch = ready_nodes[:batch_size]
            ready_nodes = ready_nodes[batch_size:]
            
            async def process_node(node_id: str) -> None:
                node = self.dag.nodes[node_id]
                
                # Get inputs for this node
                inputs = self.dag.get_node_inputs(node_id, node_outputs)
                
                # Execute the node
                node_start_time = time.time()
                try:
                    result = await node.execute(inputs)
                    # Store the result
                    node_outputs[node_id] = result
                except Exception as e:
                    node_outputs[node_id] = {"error": str(e)}
                node_end_time = time.time()
                
                # Record execution time and mark as executed
                execution_times[node_id] = node_end_time - node_start_time  # seconds
                executed_nodes.add(node_id)
                
                # Update dependencies and find newly ready nodes
                for to_id in self.dag.edges.get(node_id, []):
                    dependencies[to_id] -= 1
                    if dependencies[to_id] == 0:
                        ready_nodes.append(to_id)
            
            # Create tasks for batch processing
            tasks = [process_node(node_id) for node_id in batch]
            
            # Wait for all tasks in this batch to complete
            await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time  # seconds
        
        # Build and return a comprehensive result
        return {
            "results": node_outputs,
            "execution_times": execution_times,
            "total_time": total_time,
            "executed_nodes": list(executed_nodes)
        }