"""
Base node classes for Tygent.
"""

from typing import Any, Callable, Dict, List, Optional, Union


def default_llm_latency_model(node: "LLMNode") -> float:
    """Very rough latency estimate for LLM calls."""

    return 0.5 + 0.01 * getattr(node, "token_cost", 0)


def default_tool_latency_model(node: "ToolNode") -> float:
    """Default latency for local tool calls."""

    return 0.1


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

    def __init__(
        self,
        name: str,
        token_cost: int = 0,
        latency_estimate: Optional[float] = None,
        latency_model: Optional[Callable[["Node"], float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a node.

        Parameters
        ----------
        name : str
            The name of the node.
        token_cost : int, optional
            Estimated token cost for executing this node.
        latency_estimate : float, optional
            Fixed latency estimate in seconds. If ``None`` a ``latency_model``
            may be used to compute the value on demand.
        latency_model : callable, optional
            Function that accepts the node instance and returns an estimated
            latency in seconds.
        metadata : dict, optional
            Arbitrary metadata for the node.
        """
        super().__init__(name)
        self.token_cost = token_cost
        self.latency_estimate = latency_estimate
        self.latency_model = latency_model
        self.metadata = metadata or {}

    def get_latency_estimate(self) -> float:
        """Return the estimated latency for this node."""

        if self.latency_estimate is not None:
            return float(self.latency_estimate)
        if self.latency_model:
            try:
                return float(self.latency_model(self))
            except Exception:
                return 0.0
        return 0.0

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
        self,
        name: str,
        model: Optional[Any] = None,
        prompt_template: str = "",
        token_cost: int = 0,
        latency_estimate: Optional[float] = None,
        latency_model: Optional[Callable[["LLMNode"], float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an LLM node.

        Args:
            name: The name of the node
            model: LLM model instance
            prompt_template: Template string for the prompt
        """
        if latency_model is None:
            latency_model = default_llm_latency_model

        super().__init__(
            name,
            token_cost=token_cost,
            latency_estimate=latency_estimate,
            latency_model=latency_model,
            metadata=metadata,
        )
        self.model = model
        self.prompt_template = prompt_template


class ToolNode(Node):
    """Tool node for executing functions."""

    def __init__(
        self,
        name: str,
        func: Callable[[Dict[str, Any]], Any],
        token_cost: int = 0,
        latency_estimate: Optional[float] = None,
        latency_model: Optional[Callable[["ToolNode"], float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a tool node.

        Args:
            name: The name of the node
            func: The function to execute
        """
        if latency_model is None:
            latency_model = default_tool_latency_model

        super().__init__(
            name,
            token_cost=token_cost,
            latency_estimate=latency_estimate,
            latency_model=latency_model,
            metadata=metadata,
        )
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
