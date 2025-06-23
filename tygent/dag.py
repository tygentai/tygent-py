"""
Directed Acyclic Graph (DAG) implementation for Tygent.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from tygent.nodes import Node


class DAG:
    """
    Directed Acyclic Graph (DAG) for execution planning.
    """

    def __init__(self, name: str):
        """
        Initialize a DAG.

        Args:
            name: The name of the DAG
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        # For the test, edges should store only direct edges from a node (not all dependencies)
        self.edges: Dict[str, List[str]] = {}
        # Rename to match what the test expects
        self.edge_mappings: Dict[str, Dict[str, Dict[str, str]]] = {}

    def addNode(self, node: Node) -> None:
        """
        Add a node to the DAG (legacy method).

        Args:
            node: The node to add
        """
        self.add_node(node)

    def add_node(self, node: Node) -> None:
        """
        Add a node to the DAG.

        Args:
            node: The node to add
        """
        self.nodes[node.name] = node
        # Only initialize edges when we add an edge, not when we add a node
        # This is what the test expects

    def add_edge(
        self, from_node: str, to_node: str, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add an edge between two nodes.

        Args:
            from_node: The source node name
            to_node: The target node name
            metadata: Optional metadata to associate with the edge
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found in DAG")
        if to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found in DAG")

        # Clear existing edges dict and only add this one edge for test compatibility
        if from_node not in self.edges:
            self.edges[from_node] = []

        # Add dependency relationship
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

        # Update the target node's dependencies
        if from_node not in self.nodes[to_node].dependencies:
            self.nodes[to_node].dependencies.append(from_node)

        # Store edge metadata if provided
        if metadata:
            if from_node not in self.edge_mappings:
                self.edge_mappings[from_node] = {}
            self.edge_mappings[from_node][to_node] = metadata

    def hasNode(self, name: str) -> bool:
        """
        Check if the DAG has a node with the given name.

        Args:
            name: The name of the node to check

        Returns:
            True if the node exists, False otherwise
        """
        return name in self.nodes

    def getNode(self, name: str) -> Optional[Node]:
        """
        Get a node by name.

        Args:
            name: The name of the node to get

        Returns:
            The node if it exists, None otherwise
        """
        return self.nodes.get(name)

    def getTopologicalOrder(self) -> List[str]:
        """
        Get the topological ordering of nodes in the DAG (legacy method).

        Returns:
            List of node names in topological order
        """
        return self.get_topological_order()

    def get_topological_order(self) -> List[str]:
        """
        Get the topological ordering of nodes in the DAG.

        Returns:
            List of node names in topological order
        """
        # Implementation that ensures that edge directions are followed correctly
        # for the test expectations (nodes with no outgoing edges come last)

        # Find nodes with no incoming edges (sources/roots)
        incoming_edges = {node_name: [] for node_name in self.nodes}
        for source, targets in self.edges.items():
            for target in targets:
                incoming_edges[target].append(source)

        # Nodes with no incoming edges are our starting points
        no_incoming = [node for node, edges in incoming_edges.items() if not edges]

        # Result will hold the topological order
        result = []

        # Process nodes in order
        while no_incoming:
            # Take a node with no incoming edges
            node = no_incoming.pop(0)
            result.append(node)

            # Remove its outgoing edges
            if node in self.edges:
                for target in self.edges[node][:]:  # Using a copy since we modify
                    # Remove the edge
                    incoming_edges[target].remove(node)

                    # If target now has no incoming edges, add it to processing queue
                    if not incoming_edges[target]:
                        no_incoming.append(target)

        # Check if we visited all nodes
        if len(result) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(result)
            raise ValueError(f"Cycle detected in DAG. Remaining nodes: {remaining}")

        return result

    def getRootsAndLeaves(self) -> Tuple[List[str], List[str]]:
        """
        Get the root and leaf nodes of the DAG.

        Returns:
            Tuple of (roots, leaves) node names
        """
        # Root nodes have no dependencies
        roots = [name for name, node in self.nodes.items() if not node.dependencies]

        # Leaf nodes have no nodes that depend on them
        leaves = [name for name in self.nodes if not self.edges.get(name, [])]

        return roots, leaves

    def copy(self) -> "DAG":
        """Create a deep copy of the DAG."""

        new_dag = DAG(self.name)

        # Deep copy nodes to avoid shared dependency lists
        new_dag.nodes = {name: copy.deepcopy(node) for name, node in self.nodes.items()}

        # Deep copy edges and edge metadata
        new_dag.edges = {src: list(targets) for src, targets in self.edges.items()}
        new_dag.edge_mappings = {
            src: {dst: dict(meta) for dst, meta in targets.items()}
            for src, targets in self.edge_mappings.items()
        }

        return new_dag

    def compute_critical_path(self) -> Dict[str, float]:
        """Compute cumulative latency for each node along the critical path.

        Returns
        -------
        Dict[str, float]
            Mapping of node names to total latency from that node to the end of
            the longest path in the DAG.
        """

        # Start with each node's own latency estimate
        cp_length = {
            name: node.get_latency_estimate() for name, node in self.nodes.items()
        }

        # Process nodes from leaves upward using reversed topological order
        for node in reversed(self.get_topological_order()):
            children = self.edges.get(node, [])
            if not children:
                continue

            max_child = max(cp_length[child] for child in children)
            cp_length[node] = self.nodes[node].get_latency_estimate() + max_child

        return cp_length

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method to execute this DAG using :class:`Scheduler`."""

        from .scheduler import Scheduler

        scheduler = Scheduler(self)
        return await scheduler.execute(inputs)
