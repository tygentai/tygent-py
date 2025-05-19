"""
Nodes module provides the different types of nodes that can be used in a Tygent DAG.
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Optional, Callable, List, Union

NodeType = str  # 'base', 'llm', 'tool', 'memory', 'input', 'output'

class BaseNode(ABC):
    """
    Base class for all node types in the Tygent system.
    """
    
    def __init__(
        self, 
        id: str, 
        node_type: NodeType,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        expected_latency: Optional[float] = None
    ):
        """
        Initialize a base node.
        
        Args:
            id: Unique identifier for the node
            node_type: Type of the node
            input_schema: Expected schema for node inputs
            output_schema: Expected schema for node outputs
            expected_latency: Expected execution time (seconds)
        """
        self.id = id
        self.node_type = node_type
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.expected_latency = expected_latency
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node functionality.
        
        Args:
            inputs: Input values for the node
            
        Returns:
            Output values from the node execution
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input values against the schema.
        
        Args:
            inputs: Input values to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If required inputs are missing or have wrong type
        """
        if not self.input_schema:
            return True
        
        for key, expected_type in self.input_schema.items():
            if key not in inputs:
                raise ValueError(f"Required input '{key}' not provided")
            
            # Basic type checking - should be enhanced with proper schema validation
            actual_type = type(inputs[key])
            if expected_type != actual_type and expected_type != Any:
                raise TypeError(f"Input '{key}' has type {actual_type}, expected {expected_type}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary representation.
        
        Returns:
            A dictionary representing the node
        """
        return {
            'id': self.id,
            'type': self.node_type,
            'expected_latency': self.expected_latency
        }


class LLMNode(BaseNode):
    """
    Node that represents an LLM call in the workflow.
    """
    
    def __init__(
        self,
        id: str,
        model: str,
        prompt_template: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        expected_latency: Optional[float] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize an LLM node.
        
        Args:
            id: Unique identifier for the node
            model: The LLM model to use
            prompt_template: The template for constructing prompts
            input_schema: Expected schema for node inputs
            output_schema: Expected schema for node outputs
            expected_latency: Expected execution time (seconds)
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum tokens to generate
        """
        super().__init__(id, 'llm', input_schema, output_schema, expected_latency)
        self.model = model
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_client = None
        
        # Try to initialize OpenAI client if available
        try:
            import openai
            import os
            if os.environ.get("OPENAI_API_KEY"):
                self.openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except (ImportError, Exception):
            pass
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an LLM call.
        
        Args:
            inputs: Input values for the LLM
            
        Returns:
            Output from the LLM
        """
        self.validate_inputs(inputs)
        
        # Format the prompt template with the inputs
        prompt = self.prompt_template
        for key, value in inputs.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
        
        # Use the OpenAI client if available
        if self.openai_client:
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                content = completion.choices[0].message.content
                return {"response": content if content is not None else ""}
            except Exception as e:
                return {"error": f"Error calling OpenAI API: {str(e)}"}
        
        # Simulate a response for testing purposes
        return {"response": f"Simulated LLM response to prompt: {prompt[:50]}..."}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert LLM node to dictionary representation.
        
        Returns:
            A dictionary representing the LLM node
        """
        base_dict = super().to_dict()
        return {
            **base_dict,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


class ToolNode(BaseNode):
    """
    Node that represents a tool or function call in the workflow.
    """
    
    def __init__(
        self,
        id: str,
        tool_fn: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        expected_latency: Optional[float] = None
    ):
        """
        Initialize a tool node.
        
        Args:
            id: Unique identifier for the node
            tool_fn: The function to execute
            input_schema: Expected schema for node inputs
            output_schema: Expected schema for node outputs
            expected_latency: Expected execution time (seconds)
        """
        super().__init__(id, 'tool', input_schema, output_schema, expected_latency)
        self.tool_fn = tool_fn
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool function.
        
        Args:
            inputs: Input values for the tool
            
        Returns:
            Output from the tool execution
        """
        self.validate_inputs(inputs)
        
        try:
            # Call the tool function with the inputs
            result = await self.tool_fn(inputs)
            
            # If the result is already a dict, return it
            if result is not None and isinstance(result, dict):
                return result
            
            # If it's a simple value, wrap it
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tool node to dictionary representation.
        
        Returns:
            A dictionary representing the tool node
        """
        base_dict = super().to_dict()
        return {
            **base_dict,
            'function_name': self.tool_fn.__name__ if hasattr(self.tool_fn, '__name__') else 'anonymous'
        }


class MemoryNode(BaseNode):
    """
    Node that represents a memory operation in the workflow.
    """
    
    def __init__(
        self,
        id: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        expected_latency: float = 0.1
    ):
        """
        Initialize a memory node.
        
        Args:
            id: Unique identifier for the node
            input_schema: Expected schema for node inputs
            output_schema: Expected schema for node outputs
            expected_latency: Expected execution time (seconds)
        """
        super().__init__(id, 'memory', input_schema, output_schema, expected_latency)
        self.memory: Dict[str, Any] = {}
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a memory operation.
        
        Args:
            inputs: Input values for the memory operation
            
        Returns:
            Current state of the memory
        """
        self.validate_inputs(inputs)
        
        # Special operations for memory
        operation = inputs.get('operation', 'store')
        
        if operation == 'store':
            # Store key-value pairs
            for key, value in inputs.items():
                if key != 'operation':
                    self.memory[key] = value
        elif operation == 'retrieve':
            # Retrieve specific keys
            result: Dict[str, Any] = {}
            keys = inputs.get('keys', list(self.memory.keys()))
            for key in keys:
                if key in self.memory:
                    result[key] = self.memory[key]
            return result
        elif operation == 'clear':
            # Clear specific keys or all memory
            keys = inputs.get('keys', list(self.memory.keys()))
            for key in keys:
                if key in self.memory:
                    del self.memory[key]
        
        # Return the current state of memory
        return dict(self.memory)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory node to dictionary representation.
        
        Returns:
            A dictionary representing the memory node
        """
        base_dict = super().to_dict()
        return {
            **base_dict,
            'memory_size': len(self.memory)
        }