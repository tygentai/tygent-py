# Tygent Python Package

Transform LLM Agents into High-Performance Engines with DAG optimization.

## Installation

```bash
pip install tygent
```

## Overview

Tygent converts agent-generated plans into typed Directed Acyclic Graphs (DAGs) for optimized execution through critical path analysis. This enables parallel execution of independent tasks and more efficient use of resources.

## Key Features

- **DAG Optimization**: Transform sequential plans into parallel execution graphs
- **Typed Execution**: Strong typing for inputs and outputs between nodes
- **Critical Path Analysis**: Identify and optimize the critical execution path
- **Constraint-Aware Scheduling**: Schedule tasks based on resource constraints
- **Dynamic Runtime Adaptation**: Adapt execution based on intermediate results

## Quick Start

```python
from tygent import DAG, ToolNode, LLMNode, Scheduler

# Create a DAG for your workflow
dag = DAG("my_workflow")

# Define tool functions
async def search_data(inputs):
    # Implementation
    return {"results": f"Search results for {inputs.get('query')}"}

async def extract_info(inputs):
    # Implementation
    return {"extracted": f"Extracted from {inputs.get('results')}"}

# Add nodes to the DAG
dag.add_node(ToolNode("search", search_data))
dag.add_node(ToolNode("extract", extract_info))
dag.add_node(LLMNode(
    "analyze", 
    model="gpt-4o",
    prompt_template="Analyze this data: {extracted}"
))

# Define execution flow with dependencies
dag.add_edge("search", "extract")
dag.add_edge("extract", "analyze")

# Create a scheduler to execute the DAG
scheduler = Scheduler(dag)

# Execute the workflow
import asyncio
async def run():
    result = await scheduler.execute({"query": "What is the latest news about AI?"})
    print(result)

asyncio.run(run())
```

## Documentation

For detailed documentation and more examples, visit [tygent.ai](https://tygent.ai/docs) or check out the [examples repository](https://github.com/tygent-ai/tygent-examples).
