"""
Example of integrating Tygent with LangGraph.
"""

import asyncio
import sys

sys.path.append("./tygent-py")
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

import tygent as tg


def main():
    """Run the LangGraph integration example."""

    print("\nTygent + LangGraph Integration Example")
    print("======================================\n")

    # Create a LangGraph workflow
    print("Creating a LangGraph workflow...")
    workflow = StateGraph("research_workflow")

    # Define states
    def search_state(state):
        """Perform search operation."""
        print(
            f"  - Executing search state with query: {state.get('query', 'No query')}"
        )
        return {"search_results": "Simulated search results for renewable energy"}

    def synthesize_state(state):
        """Synthesize findings."""
        print(
            f"  - Executing synthesize state with results: {state.get('search_results', 'No results')}"
        )
        return {"synthesis": "Synthesized information about renewable energy trends"}

    # Add states to the graph
    print("Adding states to the LangGraph workflow...")
    workflow.add_node("search", search_state)
    workflow.add_node("synthesize", synthesize_state)

    # Add edges
    print("Adding edges between states...")
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", END)

    print("\nConverting LangGraph workflow to Tygent DAG...")

    # Convert LangGraph workflow to Tygent DAG
    def langgraph_to_tygent(workflow):
        """Convert a LangGraph workflow to a Tygent DAG."""
        dag_name = getattr(workflow, "name", "langgraph_workflow")
        dag = tg.DAG(dag_name)

        # Convert LangGraph nodes to Tygent nodes
        for node_name in workflow.nodes:
            if node_name == END:
                continue

            state_spec = workflow.nodes[node_name]
            tool_fn = getattr(state_spec, "runnable", state_spec)

            async def wrapper(inputs, fn=tool_fn):
                if hasattr(fn, "invoke"):
                    result = fn.invoke(inputs)
                    if hasattr(result, "__await__"):
                        result = await result
                    return result
                if hasattr(fn, "__call__"):
                    result = fn(inputs)
                    if hasattr(result, "__await__"):
                        return await result
                    return result

            node = tg.ToolNode(node_name, wrapper)
            dag.add_node(node)

        # Add edges based on LangGraph transitions
        for source, target in workflow.edges:
            if source == END or target == END:
                continue

            dag.add_edge(source, target)

        return dag

    # Convert and execute
    tygent_dag = langgraph_to_tygent(workflow)

    print("Executing the converted DAG with Tygent's scheduler...")
    scheduler = tg.Scheduler(tygent_dag)
    result = asyncio.run(scheduler.execute({"query": "renewable energy"}))

    print("\nResults:")
    for node_id, outputs in result.items():
        if node_id != "__inputs":
            print(f"  - Node: {node_id}")
            for key, value in outputs.items():
                print(f"    - {key}: {value}")

    print("\nDAG Execution completed successfully.")


if __name__ == "__main__":
    main()
