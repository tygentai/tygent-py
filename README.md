# Tygent (Python)

Tygent reshapes unstructured LLM agent plans into structured execution blueprints so downstream tools know which context to fetch and when to run each step. Typed execution graphs[^dag] are the core structure the runtime emits, giving you explicit dependencies, metadata, and prefetch hints that the scheduler can consume for reliable, optimised execution.

## Highlights
- **Structured planner** – normalise free-form plans or framework payloads into typed steps with dependencies, tags, and prefetch directives (`parse_plan`, `ServicePlanBuilder`), exporting context fabric descriptors compatible with Recontext.
- **Context-aware execution** – structured graph metadata[^dag] flows into the scheduler so you can prioritise critical nodes, enforce token budgets, rate-limit API calls, and capture audit traces.
- **Drop-in acceleration** – wrap callables, agents, or plan dictionaries with `tygent.accelerate` to obtain an executor that understands the structured representation.
- **Adaptive workflows** – mutate the structured plan at runtime with `AdaptiveExecutor` rewrite rules for fallbacks, conditional branches, or resource-aware tuning.
- **Multi-agent runtime** – coordinate independent agents through `MultiAgentManager` and a shared `CommunicationBus` while preserving structured plan metadata.
- **Framework patches** – call `tygent.install()` to enable runtime helpers for integrations in `tygent.integrations.*`.
- **Service bridge + CLI** – the `tyapi` package ships an aiohttp service and CLI that convert third-party plans into the structured format, surface prefetch hints, and benchmark sequential vs accelerated runs.

## Installation

```bash
pip install tygent
```

Development installs (tests, tyapi service, docs tooling) expect Python 3.8+ and the optional extras listed in `pyproject.toml`.

## Quick tour

The snippets below show how Tygent promotes loosely described plans into explicit structures that drive context prefetching and execution control.

### 1. Accelerate a structured plan

```python
import asyncio
from tygent import accelerate

plan = {
    "name": "market_research",
    "steps": [
        {"name": "collect", "func": lambda inputs: {"sources": inputs["query"]}},
        {
            "name": "summarize",
            "func": lambda inputs: f"Summary: {inputs['collect']['sources']}",
            "dependencies": ["collect"],
            "critical": True,
        },
    ],
}


async def main() -> None:
    execute_plan = accelerate(plan)
    result = await execute_plan({"query": "AI funding"})
    print(result["results"]["summarize"])


asyncio.run(main())
```

Passing a plan dictionary produces a scheduler-backed executor that preserves metadata (critical steps, edge mappings, etc.) and returns the scheduler output structure—one of the structured formats Tygent can derive from loosely specified plans.

### 2. Drop-in acceleration for existing code

```python
from tygent import accelerate

@accelerate
def fetch_profile(user_id: str) -> dict:
    # Your existing implementation
    return {"user": user_id}

profile = fetch_profile("abc123")
```

`accelerate` unwraps sync or async callables and inspects attached plans when available (e.g. LangChain, Google ADK runners, OpenAI Assistants). When the framework exposes a plan or workflow, Tygent parses it into the structured graph[^dag] and schedules it using the built-in executor.

### 3. Build the structured graph directly

```python
import asyncio
from tygent import DAG, ToolNode, Scheduler

dag = DAG("demo")
dag.add_node(ToolNode("search", lambda inputs: {"hits": ["url"]}))
dag.add_node(ToolNode("summarize", lambda inputs: f"Summary of {inputs['search']['hits']}"))
dag.add_edge("search", "summarize")

scheduler = Scheduler(dag)
result = asyncio.run(scheduler.execute({"query": "latest research"}))
print(result["results"]["summarize"])  # -> "Summary of ['url']"
```

The scheduler exposes token budgeting, request throttling, audit hooks, and critical path prioritisation through `Scheduler.configure`.

### 4. Adaptive execution

```python
import asyncio
from tygent import AdaptiveExecutor, create_fallback_rule
from tygent import ToolNode, DAG

# Base graph
base = DAG("workflow")
base.add_node(ToolNode("primary", lambda inputs: 1 / inputs.get("divisor", 1)))

executor = AdaptiveExecutor(
    base,
    rewrite_rules=[
        create_fallback_rule(
            error_condition=lambda state: "error" in state.get("results", {}).get("primary", {}),
            fallback_node_creator=lambda dag, state: ToolNode("fallback", lambda _inputs: 1),
            rule_name="fallback_on_error",
        )
    ],
)


async def main() -> None:
    outputs = await executor.execute({"divisor": 0})
    print(outputs["results"].keys())


asyncio.run(main())
```

Rewrite rules receive intermediate state and can inject new nodes or branches before the scheduler re-runs the structured graph[^dag].

### 5. Coordinate multiple agents

```python
import asyncio
from tygent import MultiAgentManager

manager = MultiAgentManager("support")


class Analyzer:
    async def execute(self, inputs):
        return {"keywords": inputs["question"].split()}


class Retrieval:
    async def execute(self, inputs):
        return {"docs": ["reset-guide.md"]}


manager.add_agent("analyzer", Analyzer())
manager.add_agent("retrieval", Retrieval())


async def main() -> None:
    result = await manager.execute({"question": "How do I reset my password?"})
    print(result)


asyncio.run(main())
```

The manager runs agents concurrently and uses `CommunicationBus` for message passing when agents opt in.

## Service bridge and SaaS example

The Python repository bundles a mini SaaS-style planner under `tyapi/`:

- `ServicePlanBuilder` converts JSON specs (e.g. from the service) into executable plans by templating prompts, tagging critical nodes, and wiring redundancy hints.
- `execute_service_plan` prefetches referenced links (via `prefetch_many`) and executes the resulting structured graph[^dag] with optional parallelism limits.
- `tyapi.service.cli` provides commands to register accounts, issue API keys, configure ingestors, and run an aiohttp server that exposes `/v1/plan/convert` and `/v1/plan/benchmark`.

Run the service locally:

```bash
# Install in editable mode for development
pip install -e .[dev]

# Register an account and start the server
python -m tyapi.service.cli register --name "Acme" --email "ops@example.com"
python -m tyapi.service.cli serve --port 8080
```

The accompanying web UI (served from `tyapi/frontend/`) lets you paste framework-specific plans, choose redundancy settings, and compare sequential vs accelerated execution latencies.

## Examples & integrations

A collection of runnable samples lives in `examples/`:

- `advanced_python_example.py` – end-to-end structured graph[^dag] creation and execution
- `dynamic_dag_example.py` – AdaptiveExecutor rewrite rules in action on the structured graph[^dag]
- `langchain_integration.py` – working with popular agent frameworks
- `crewai_market_analysis.py`, `google_adk_market_analysis.py` – integration-specific accelerators

Call `tygent.install()` to load integration patches (Anthropic, Google AI, HuggingFace, Microsoft AI, Salesforce) before instantiating their SDK clients.

## Editor extensions

Tygent ships helper extensions for embedding the structured planner inside popular IDEs:

- **VS Code** (`vscode-extension/`) – the *Tygent: Enable Agent* command inserts `tygent.install()` and required imports into the active Python file, making it easy to convert agents in place.
- **Cursor** (`cursor-extension/`) – mirrors the VS Code command with a Cursor-specific *Tygent: Enable Agent (Cursor)* action so Cursor users can patch working files without leaving the editor.

Build either extension with `npm run compile` inside the respective folder, then use VS Code’s `Extension Development Host` or Cursor’s extension loader to test install the generated package.

## Testing

```bash
pip install -e .[dev]
pytest tests -q
```

Targeted suites exist for the tyapi service (`pytest tyapi/tests -q`) and core structured graph behaviour[^dag] (`pytest tests/test_dag.py`). The repository uses `pytest-asyncio` for async flows; see `pytest.ini` for configuration.

## Project layout

```
tygent/
│   accelerate.py      # drop-in wrappers and framework adapters
│   scheduler.py       # execution engine with hooks & budgets
│   adaptive_executor.py
│   multi_agent.py
│   service_bridge.py
│   integrations/      # opt-in SDK patches
└── tyapi/             # SaaS planner service and CLI
```

Additional tooling (editor extensions, docs) lives under `cursor-extension/`, `vscode-extension/`, and `docs/`.

[^dag]: Tygent materialises plans as typed directed acyclic graphs (DAGs) so dependencies, context-prefetch hints, and critical paths stay explicit for the execution engine and Recontext-compatible context fabric.

---

Need help? Open a GitHub issue or reach out at support@tygent.ai.
