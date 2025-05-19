"""
DAG module provides the core DAG (Directed Acyclic Graph) implementation for Tygent.
"""

import uuid
from typing import Dict, List, Callable, Any, Set, Optional, Tuple

from .nodes import BaseNode

# Type for edge mappings (source field -> destination field)
EdgeMapping = Dict[str, str]

# Type for condition functions
ConditionFunction = Callable[[Dict[str, Any]], bool]

class DAG:
    """
    Directed Acyclic Graph that represents a workflow of computation nodes.
    
    DAGs are created by LLMs using the plan an agent generates for its actions.
    """
    
    def __init__(self, name: str):
        """
        Create a new DAG.
        
        Args:
            name: The name of the DAG
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.conditional_edges: Dict[str, Dict[str, ConditionFunction]] = {}
        self.edge_mappings: Dict[str, Dict[str, EdgeMapping]] = {}
    
    def add_node(self, node: BaseNode) -> None:
        """
        Add a node to the DAG.
        
        Args:
            node: The node to add
        
        Raises:
            ValueError: If a node with the same ID already exists
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists in the DAG")
        
        self.nodes[node.id] = node
        
        if node.id not in self.edges:
            self.edges[node.id] = []
    
    def add_edge(self, from_node_id: str, to_node_id: str, mapping: Optional[EdgeMapping] = None) -> None:
        """
        Add a directed edge between two nodes.
        
        Args:
            from_node_id: The source node ID
            to_node_id: The target node ID
            mapping: Optional mapping of output fields from source to input fields of target
        
        Raises:
            ValueError: If either node does not exist
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} does not exist in the DAG")
        
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} does not exist in the DAG")
        
        if from_node_id not in self.edges:
            self.edges[from_node_id] = []
        
        if to_node_id not in self.edges[from_node_id]:
            self.edges[from_node_id].append(to_node_id)
        
        if mapping:
            if from_node_id not in self.edge_mappings:
                self.edge_mappings[from_node_id] = {}
            
            self.edge_mappings[from_node_id][to_node_id] = mapping
    
    def add_conditional_edge(self, 
                            from_node_id: str, 
                            to_node_id: str, 
                            condition: ConditionFunction, 
                            mapping: Optional[EdgeMapping] = None) -> None:
        """
        Add a conditional edge between two nodes.
        
        Args:
            from_node_id: The source node ID
            to_node_id: The target node ID
            condition: Function that evaluates if the edge should be traversed
            mapping: Optional mapping of output fields from source to input fields of target
        """
        # First add a normal edge
        self.add_edge(from_node_id, to_node_id, mapping)
        
        # Then add the condition
        if from_node_id not in self.conditional_edges:
            self.conditional_edges[from_node_id] = {}
        
        self.conditional_edges[from_node_id][to_node_id] = condition
    
    def get_topological_order(self) -> List[str]:
        """
        Return a valid topological ordering of the nodes.
        
        Returns:
            A list of node IDs in topological order
        
        Raises:
            ValueError: If the graph contains a cycle
        """
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []
        
        def visit(node_id: str) -> None:
            if node_id in temp_visited:
                raise ValueError(f"DAG contains a cycle including node {node_id}")
            
            if node_id not in visited:
                temp_visited.add(node_id)
                
                neighbors = self.edges.get(node_id, [])
                for neighbor in neighbors:
                    visit(neighbor)
                
                temp_visited.remove(node_id)
                visited.add(node_id)
                order.append(node_id)
        
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)
        
        # Reverse to get correct topological order
        return list(reversed(order))
    
    def get_node_inputs(self, node_id: str, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map outputs from previous nodes to inputs for a specific node.
        
        Args:
            node_id: The ID of the node to get inputs for
            outputs: The current outputs from all nodes
            
        Returns:
            A dictionary of inputs for the specified node
        """
        inputs: Dict[str, Any] = {}
        
        # Find all edges that point to this node
        for from_node_id, to_nodes in self.edges.items():
            if node_id in to_nodes:
                # Check if there's a condition that prevents this edge
                if (from_node_id in self.conditional_edges and 
                    node_id in self.conditional_edges[from_node_id]):
                    if not self.conditional_edges[from_node_id][node_id](outputs):
                        continue  # Skip this edge if condition is not met
                
                # Check if there's a mapping for this edge
                mapping = None
                if from_node_id in self.edge_mappings:
                    mapping = self.edge_mappings[from_node_id].get(node_id)
                
                if mapping:
                    # Apply the mapping
                    for src_key, dst_key in mapping.items():
                        if from_node_id in outputs and src_key in outputs[from_node_id]:
                            inputs[dst_key] = outputs[from_node_id][src_key]
                elif from_node_id in outputs:
                    # No mapping, just forward all outputs
                    inputs.update(outputs[from_node_id])
        
        return inputs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DAG to a dictionary representation.
        
        Returns:
            A dictionary representation of the DAG
        """
        return {
            'id': self.id,
            'name': self.name,
            'nodes': {id: node.to_dict() for id, node in self.nodes.items()},
            'edges': self.edges,
            # Note: Can't easily serialize conditional edges or mappings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DAG':
        """
        Create a DAG from a dictionary representation.
        
        Args:
            data: Dictionary containing the DAG specification
            
        Returns:
            A reconstructed DAG
        """
        dag = cls(data['name'])
        dag.id = data['id']
        
        # Reconstruct edges
        dag.edges = data['edges']
        
        # Note: The nodes would need to be reconstructed with their proper types
        # This would require factories for each node type
        
        return dag