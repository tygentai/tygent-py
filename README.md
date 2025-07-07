# Tygent Python - Speed & Efficiency Layer for AI Agents

[![CI](https://github.com/tygent-ai/tygent-py/workflows/CI/badge.svg)](https://github.com/tygent-ai/tygent-py/actions)
[![PyPI version](https://badge.fury.io/py/tygent.svg)](https://badge.fury.io/py/tygent)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Transform your existing AI agents into high-performance engines with intelligent parallel execution and optimized scheduling. Tygent aims to speed up workflows and reduce costs with **no code changes required**.

## Quick Start

### Installation

```bash
pip install tygent
```

### Basic Usage - Accelerate Any Function

```python
from tygent import accelerate


# Your existing code
def research_topic(topic):
    # Your existing research logic
    return {"summary": f"Research on {topic}"}

# Wrap the function to run via Tygent's scheduler
accelerated_research = accelerate(research_topic)
result = accelerated_research("AI trends")
```

### Zero-Lift Framework Patching

```python
import asyncio

import tygent

# Apply patches for any installed integrations
tygent.install()

from google.generativeai import GenerativeModel

model = GenerativeModel("gemini-pro")
result = asyncio.run(model.generate_content("Hello"))
```

### Multi-Agent System

```python
import asyncio

from tygent import MultiAgentManager

# Create manager
manager = MultiAgentManager("customer_support")

# Add agents to the system
class AnalyzerAgent:
    def analyze(self, question):
        return {"intent": "password_reset", "keywords": ["reset", "password"]}

class ResearchAgent:
    def search(self, keywords):
        return {"help_docs": ["Reset guide", "Account recovery"]}

manager.add_agent("analyzer", AnalyzerAgent())
manager.add_agent("researcher", ResearchAgent())

# Execute with optimized communication
result = asyncio.run(
    manager.execute({"question": "How do I reset my password?"})
)
```

## Key Features

- **🚀 Speed Improvement**: Intelligent parallel execution of independent operations
- **💰 Cost Reduction**: Optimized token usage and API call batching
- **🔧 Zero Code Changes**: Drop-in acceleration for existing functions and agents
- **🧠 Smart DAG Optimization**: Automatic dependency analysis and parallel scheduling
- **🔄 Dynamic Adaptation**: Runtime DAG modification based on conditions and failures
- **🎯 Multi-Framework Support**: Works with CrewAI, HuggingFace, Google AI, and custom agents
- **📄 Plan Parsing**: Build DAGs directly from framework plans or dictionaries
- **📋 Auditing & Tracing**: Inspect plans, hook into node execution, and record results

## Architecture

Tygent uses Directed Acyclic Graphs (DAGs) to model and optimize your agent workflows:

```
Your Sequential Code:        Tygent Optimized:
┌─────────────────┐         ┌─────────────────┐
│   Step 1        │         │   Step 1        │
└─────────────────┘         └─────────────────┘
         │                           │
┌─────────────────┐         ┌─────────┬───────┐
│   Step 2        │   →     │ Step 2  │Step 3 │ (Parallel)
└─────────────────┘         └─────────┴───────┘
         │                           │
┌─────────────────┐         ┌─────────────────┐
│   Step 3        │         │   Step 4        │
└─────────────────┘         └─────────────────┘
```

## Advanced Usage

### Dynamic DAG Modification

```python
from tygent import accelerate
from tygent.adaptive_executor import AdaptiveExecutor


# Workflow that adapts to failures and conditions
@accelerate
async def travel_planning_workflow(destination):
    # Tygent automatically handles:
    # - API failures with fallback services
    # - Conditional branching based on weather
    # - Resource-aware execution adaptation
    
    weather = await get_weather(destination)  # Primary API
    # Auto-fallback to backup_weather_service if primary fails
    
    if weather["condition"] == "rain":
        # Dynamically adds indoor alternatives node
        recommendations = await get_indoor_alternatives(destination)
    else:
        recommendations = await get_outdoor_activities(destination)
    
    return recommendations
```

### Integration Examples

#### Example: Accelerating a LangChain Agent
```python
from tygent import accelerate


# Your existing LangChain agent
class MockLangChainAgent:
    def run(self, query):
        return f"LangChain response to: {query}"

agent = MockLangChainAgent()

# Accelerate it
accelerated_agent = accelerate(agent)
result = accelerated_agent.run("Analyze market trends")
```

#### Custom Multi-Agent System
```python
import asyncio

from tygent import DAG, LLMNode, MultiAgentManager, ToolNode

# Create a DAG for manual workflow control
dag = DAG("content_generation")

def research_function(inputs):
    return {"research_data": f"Data about {inputs.get('topic', 'general')}"}

class SimpleLLMNode(LLMNode):
    async def execute(self, inputs):
        # Normally this would call an LLM; here we just format text
        return {"outline": f"Outline for {inputs.get('research_data', '')}"}

dag.add_node(ToolNode("research", research_function))
dag.add_node(SimpleLLMNode("outline"))
dag.add_edge("research", "outline")

result = asyncio.run(dag.execute({"topic": "AI trends"}))
```

### Parsing Plans

Tygent can convert structured plans into executable DAGs with `parse_plan`.

```python
from tygent import Scheduler, accelerate, parse_plan

plan = {
    "name": "math",
    "steps": [
        {"name": "add", "func": add_fn, "critical": True},
        {"name": "mult", "func": mult_fn, "dependencies": ["add"]},
    ],
}

# Build a DAG manually
dag, critical = parse_plan(plan)
scheduler = Scheduler(dag)
scheduler.priority_nodes = critical

# Or accelerate the plan directly (works with frameworks exposing `get_plan`)
run_plan = accelerate(plan)
```

If you have multiple plans (e.g. produced by different LLMs) you can
combine them into a single DAG:

```python
from tygent import parse_plans, Scheduler

dag, critical = parse_plans([plan_a, plan_b])
scheduler = Scheduler(dag)
scheduler.priority_nodes = critical
```

## Testing

### Running Tests

Make sure to install the package in editable mode before executing the tests.

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Install package in development mode
pip install -e .

# Run core tests (always pass)
pytest tests/test_dag.py tests/test_multi_agent.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tygent --cov-report=html
```

### Test Coverage

Our test suite covers:
- **Core DAG functionality**: Node management, topological sorting, parallel execution
- **Multi-agent communication**: Message passing, agent orchestration, conversation history
- **Async operations**: Proper async/await handling, concurrent execution
- **Error handling**: Graceful failure recovery, fallback mechanisms

**Current Status**: 14/14 core tests passing ✅

#### Recent Test Fixes (v1.1)
- Fixed Message interface to match TypedDict implementation
- Corrected async timestamp handling using `asyncio.get_event_loop().time()`
- Added pytest.ini configuration for proper async test support
- Updated MultiAgentManager constructor calls with required name parameter
- Removed dependencies on non-existent classes (AgentRole, OptimizationSettings)

### CI/CD

GitHub Actions workflow automatically runs:
- **Multi-version testing**: Python 3.8, 3.9, 3.10, 3.11
- **Multi-platform**: Ubuntu, macOS, Windows  
- **Code quality**: flake8 linting, black formatting, mypy type checking
- **Package building**: Automated wheel and source distribution creation
- **PyPI publishing**: Automatic publishing on main branch pushes
- **Coverage reporting**: HTML and LCOV coverage reports

Triggers: Every push and pull request to main/develop branches

## Framework Integrations

### Supported Frameworks
- **CrewAI**: Multi-agent coordination
- **Microsoft Semantic Kernel**: Plugin optimization
- **LangSmith**: Experiment tracking integration
- **LangFlow**: Visual workflow authoring
- **Custom Agents**: Universal function acceleration

### External Service Integrations
- **OpenAI**: GPT-4, GPT-3.5-turbo optimization
- **Google AI**: Gemini model integration
- **Microsoft Azure**: Azure OpenAI service
- **Salesforce**: Einstein AI and CRM operations
- **HuggingFace**: Transformer models

## Performance Benchmarks

Benchmark tests live under `tests/benchmarks/` and compare sequential
execution with Tygent's scheduler. Typical results on a small DAG of four
dependent tasks:

| Scenario                 | Time (s) |
|--------------------------|---------:|
| Sequential execution     | ~0.70    |
| Scheduler (1 worker)     | ~0.72    |
| Scheduler (2 workers)    | ~0.52    |

Run the benchmarks using:

```bash
pip install -e .
pytest tests/benchmarks/ -v
```

## Development

### Project Structure
```
tygent-py/
├── tygent/
│   ├── __init__.py          # Main exports
│   ├── accelerate.py        # Core acceleration wrapper
│   ├── dag.py              # DAG implementation
│   ├── nodes.py            # Node types (Tool, LLM, etc.)
│   ├── scheduler.py        # Execution scheduler
│   ├── multi_agent.py      # Multi-agent system
│   ├── adaptive_executor.py # Dynamic DAG modification
│   └── integrations/       # Framework integrations
├── tests/                  # Test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

### Code Quality

- **Type hints**: Full type annotation coverage
- **Testing**: Comprehensive test suite with >90% coverage
- **Linting**: Black formatting, flake8 compliance
- **Documentation**: Detailed docstrings and examples

## License

Creative Commons Attribution-NonCommercial 4.0 International License.

See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [https://tygent.ai/docs](https://tygent.ai/docs)
- **Issues**: [GitHub Issues](https://github.com/tygent-ai/tygent-py/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tygent-ai/tygent-py/discussions)
- **Email**: support@tygent.ai

---

**Transform your agents. Accelerate your AI.**