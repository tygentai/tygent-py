"""
Base node classes for Tygent.
"""

from typing import Dict, List, Any, Optional, Callable, Union


class BaseNode:
    """Base class for all nodes in Tygent."""

    def __init__(self, name: str):
        """
        Initialize a base node.

        Args:
            name: The name of the node
        """
        self.name = name
        self.dependencies: List[str] = []

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the node with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            The result of the node execution
        """
        raise NotImplementedError("Subclasses must implement execute()")


class Node(BaseNode):
    """Base node class for execution in the DAG."""

    def __init__(self, name: str):
        """
        Initialize a node.

        Args:
            name: The name of the node
        """
        super().__init__(name)

    def setDependencies(self, dependencies: List[str]) -> None:
        """
        Set the dependencies of this node.

        Args:
            dependencies: List of node names this node depends on
        """
        self.dependencies = dependencies

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the node with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            The result of the node execution
        """
        raise NotImplementedError("Subclasses must implement execute()")


class LLMNode(Node):
    """Base class for LLM nodes."""

    def __init__(
        self, name: str, model: Optional[Any] = None, prompt_template: str = ""
    ):
        """
        Initialize an LLM node.

        Args:
            name: The name of the node
            model: LLM model instance
            prompt_template: Template string for the prompt
        """
        super().__init__(name)
        self.model = model
        self.prompt_template = prompt_template


class ToolNode(Node):
    """Tool node for executing functions."""

    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Any]):
        """
        Initialize a tool node.

        Args:
            name: The name of the node
            func: The function to execute
        """
        super().__init__(name)
        self.func = func

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the tool node with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            The result of the function execution
        """
        try:
            result = self.func(inputs)

            # Handle both synchronous and asynchronous functions
            if hasattr(result, "__await__"):
                return await result

            return result
        except Exception as e:
            print(f"Error executing tool {self.name}: {e}")
            raise
