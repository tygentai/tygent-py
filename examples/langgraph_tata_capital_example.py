"""LangGraph multi-agent example using OpenAI ChatGPT and Tygent.

This example demonstrates how to:
1. Build a LangGraph workflow from a JSON agent specification
2. Use OpenAI's ChatGPT model for each agent
3. Run the workflow with and without Tygent to compare latency and token usage

To run this example you will need to install the required optional
dependencies and set an OpenAI API key:

```
pip install langgraph langchain-openai
export OPENAI_API_KEY="your-key"
```
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

try:  # pragma: no cover - optional dependencies
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependencies
    print("This example requires the langgraph and langchain-openai packages.")
    print("Install them with: pip install langgraph langchain-openai")
    raise SystemExit(1)

import tygent as tg

# JSON configuration for the Tata Capital Home Loan agents
TATA_CONFIG_JSON = """
{
  "name": "Tata Capital Home Loan V24",
  "agents": {
    "callback_agent": {
      "name": "Callback Agent",
      "metadata": {},
      "description": "Handles callback instructions from customer, reschedule requests, customer busy, customer driving, customer in a meeting or not interested to talk",
      "system_prompt": "You are the callback specialist agent. You are called if the customer is not free to talk or wants a callback or wants to reschedule the call or is busy currently.\nYour responsibilities:\n1. Warmly acknowledge that you can reach out to the customer again later.\n2. Ask for a time when the customer is free to talk again. \n3. Note down the callback date and time within business hours\n4. Thank the customer and politely close the call\nExample. \"Thank you for your time, I was talking from Tata Capital, have a good day.\"\n\nEnsure that you generate the response on the basis of the previous complete conversation context.",
      "knowledge_base": [
        "The business hours of the agents are from Monday to Friday, 9am to 5pm. Only If the customer is asking to call outside of business hours, the agent should politely ask the customer for acceptable callback time.",
        "Agent should never explicitly ask for callback time within business hours, only nudge if the customer sharing time outside of business hours."
      ]
    },
    "fallback_agent": {
      "name": "Fallback Agent",
      "metadata": {},
      "description": "Handles general and open non home loan queries, customer service requests, customer abuses and escalations. clarifications or non home loan related edge cases",
      "system_prompt": "You are the general support specialist. Your responsibilities:\n1. Handle general banking queries not covered by other agents\n2. Provide clarifications and additional help\n3. Manage edge cases and unexpected customer requests\n4. Ensure smooth conversation flow and customer satisfaction\n\n**APPROACH:**\n- Listen carefully to customer needs\n- Provide helpful and accurate information\n- Route back to appropriate specialists when needed\n- Maintain professional and supportive tone, never engage in arguments or intents other than home loan.\n\nEnsure that you generate the response on the basis of the previous complete conversation context.",
      "knowledge_base": [
        "The customer's outstanding loan amount is Rupees ${current_pos}.",
        "The customer's EMI left are ${balance_tenure}.",
        "The customer's current interest rate is ${current_offered_rate}%",
        "The customer can find further information related to service issues on the Tata Capital Housing Finance Website.",
        "If the customer expresses urgent issue resolution, the agent can also suggest sharing the details in an EMail to customercare@tatacapital.com.",
        "The agent is not authorised to send any documents to the customer",
        "The agent will only use the knowledge_base to answer customer queries"
      ]
    },
    "greeting_agent": {
      "name": "Greeting Agent",
      "metadata": {},
      "description": "Only handles initial greetings, customer identity verification, queries about Tata Capital or Agent name, customer name",
      "system_prompt": "\nYou are the greeting specialist agent Neha, from Tata Capital Housing Finance Limited.\nAt the start of the call, your responsibilities:\n1. Warmly greet customers and introduce yourself as Neha from Tata Capital Housing Finance Limited and confirm customer name.\n2. Verify the customer's identity. Greeting message is already done to customer. Just gather customer's response to that.\n2a. If customer's identity is not confirmed, politely close the call.\n2b. Reponse like \"à¤¹à¤¾à¤ à¤¬à¤¤à¤¾à¤\" is identity confirmation. Go ahead if customer's identiy is confirmed. Do not generate post verification lines like 'à¤à¤ªà¤à¥ à¤ªà¤¹à¤à¤¾à¤¨ à¤à¥ à¤ªà¥à¤·à¥à¤à¤¿ à¤¹à¥ à¤à¤ à¤¹à¥'. Mark verificaion DONE.\nHandoff to supervisor and retention_agent\n\nIf the greeting_agent is called during the conversation, your responsibilities:\n1. Politely answer customer queries about agent name, customer name, Tata Capital or intent of call\n2. Identity Confirmation must only be done at the start of conversation, never in the middle.\n\nIf the customer is asking who does the agent want to talk to, politely do the customer identity check again by asking if the agent is talking to ${Voiced_Customer_Name}.\nEnsure that you generate the response on the basis of the previous complete conversation context.",
      "knowledge_base": [
        "The customer's name is ${Voiced_Customer_Name}",
        "The customer's current interest rate is ${current_offered_rate}%",
        "The customer's outstanding loan amount is Rupees ${current_pos}.",
        "The customer's EMI left are ${balance_tenure}.",
        "The agent should never reveal the intent of the call if right party confirmation failed. Agent should politely close the call."
      ]
    },
    "terminal_agent": {
      "name": "Closing Agent",
      "metadata": {},
      "description": "Handles Call closing if customer is satisfied with service or customer agrees to top-up loan request, or customer wants to escalate or speak to supervisor, or customer has shared callback date and time, or customer denies any help in foreclosure, or customer does not have any issue with the loan, other agents specifically call the terminal_agent",
      "system_prompt": "You are the call closing specialist. Your responsibilities:\n1. Reply to the customer based on customer's last response politely and expertly.\n2. Politely thank the customer for customer's time, and close the call.\nEnglish: \"Thank you for your time, have a good day.\"\nHindi: \"à¤à¤ªà¤à¥ à¤¸à¤®à¤¯ à¤à¥ à¤²à¤¿à¤ à¤§à¤¨à¥à¤¯à¤µà¤¾à¤¦, à¤à¤ªà¤à¤¾ à¤¦à¤¿à¤¨ à¤¶à¥à¤­ à¤¹à¥à¥¤\"",
      "knowledge_base": [
        "If the party at other end of call is not ${Voiced_Customer_Name}, the agent should politely close the call by thanking the party at the end. The agent should never reveal the intent of the call",
        "If the party at other end of call is relative of ${Voiced_Customer_Name}, the agent should politely close the call by mentioning that the agent will callback later. The agent should never reveal the intent of the call"
      ]
    },
    "retention_agent": {
      "name": "Retention Agent",
      "metadata": {},
      "description": "customer not giving or disclosing reasons of SOA queries, handles top-up loan requests, customer asking for balance transfer (BT), customer asking for foreclosure or closure or list of documents (LOD) to close, or selling property or self funding",
      "system_prompt": "You are a generalist retention agent with the aim to retain customer from foreclosing or balance transfer and help customer. Never offer to share foreclosure steps. You are called when customer shares general reason for requesting SOA documents or customer is not specifying the reason of foreclosure.\nYou are an empathetic and persistent agent, actively listen to customer and identify customer concerns.\nYour responsibilities:\n1. Politely ask if the customer has any query or issue with existing loan because the customer requested for SOA documents. EXAMPLES \"à¤®à¥à¤à¤¨à¥ à¤à¤ªà¤à¥ current home loan à¤à¥ à¤¬à¤¾à¤°à¥ à¤®à¥à¤ à¤¬à¤¾à¤¤ à¤à¤°à¤¨à¥ à¤à¥ à¤²à¤¿à¤ à¤à¥à¤² à¤à¤¿à¤¯à¤¾ à¤¹à¥à¥¤ Can you tell if you have any concern regarding your loan?\"\n2. Warmly acknowledge if customer is unhappy and assure that Tata Capital will help its valuable customer.\n3. If the customer is asking for Balance transfer, ask further on specific reason for balance transfer.\n4. If the customer mentions about top-up loan, excitedly share that Tata Capital can provide further loan on top of existing loan.\n4a. Gather expected top up loan amount and share that you have raised ticket and the team will reach out within 2 days.\n5. If the customer has given reason for SOA request which is not knowledge_base, politely thank the customer and close the call.\n6. If the customer is not disclosing any reason for SOA request, politely thank the customer and close the conversation, call terminal_agent\n6. If customer is asking for steps to foreclose or pay off the loan, ask the customer why customer wants to close the loan\n8. If the customer is providing sale of property or paying off loan through self fund, use knowledge_base, politely close the call.\n9. Never offer to share any documents, OTP or information not in knowledge_base with the customer, you are not authorised to do that.\nNever reveal that you are negotiating with customer, strictly no words like negotiate.\n\nEnsure that you generate the response on the basis of the previous complete conversation context.",
      "knowledge_base": [
        "The customer's current interest rate is ${current_offered_rate}%",
        "The customer's outstanding loan amount is Rupees ${current_pos}.",
        "The customer's EMI left are ${balance_tenure}.",
        "Customer may foreclose loan if he wants to transfer remaining balance to another bank or customer is selling property or customer is paying remaining due amount.",
        "Customer inputs like 'I will do full payment', 'I have the money to pay it off', 'I have managed funds to pay the loan' signals that customer wants to foreclose the loan through self funding. Tata Capital wants to retain the customers. The customer can use excessive funds for personal investments to earn returns instead of foreclosing, customer can also partially pay from self funds instead of foreclosing the loan.",
        "If customer is selling property, agent should tell the customer that someone will reach out to customer within 2 days, politely close the call. Call 'terminal_agent'.",
        "Customer may request for balance transfer if another bank is giving better rate of interest or another bank is giving top up loan amount",
        "Tata Capital also provides option for top-up loan, the agent can check for expected amount and raise internally for team to get back within 2 days.",
        "If the customer expresses urgent issue resolution, the agent can also suggest sharing the details in an EMail to customercare@tatacapital.com.",
        "Terms like FC (foreclosure) and LOD (List of Documents) mean that the customer wants to foreclose the loan and the agent should further ask the reason."
      ]
    }
  },
  "flow_id": "tata_capital_home_loan_supervisor",
  "description": "Dynamic multi-agent supervisor architecture for Tata Capital Housing Finance Ltd with specialized agents"
}
"""


def build_graph(config: Dict[str, Any]) -> StateGraph:
    """Create a LangGraph workflow from the JSON configuration."""

    graph = StateGraph("tata_capital_home_loan")

    for agent_name, agent_conf in config["agents"].items():
        system_prompt = agent_conf["system_prompt"]
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        async def agent_fn(state: Dict[str, Any], llm=llm, prompt=system_prompt):
            message = state.get("message", "")
            full_prompt = f"{prompt}\nCustomer: {message}" if message else prompt
            response = await llm.ainvoke(full_prompt)
            usage = getattr(response, "response_metadata", {}).get("token_usage", {})
            next_message = response.content
            tokens = state.get("token_counts", [])
            tokens.append(usage)
            return {"message": next_message, "token_counts": tokens}

        graph.add_node(agent_name, agent_fn)

    # Simple sequential flow for demonstration
    graph.add_edge("greeting_agent", "retention_agent")
    graph.add_edge("retention_agent", "terminal_agent")
    graph.add_edge("terminal_agent", END)
    graph.set_entry_point("greeting_agent")
    return graph


def langgraph_to_tygent(graph: StateGraph) -> tg.DAG:
    """Convert a LangGraph workflow to a Tygent DAG."""

    dag_name = getattr(graph, "name", "langgraph_workflow")
    dag = tg.DAG(dag_name)

    for node_name in graph.nodes:
        if node_name == END:
            continue

        state_spec = graph.nodes[node_name]
        tool_fn = getattr(state_spec, "runnable", state_spec)

        async def wrapper(inputs: Dict[str, Any], fn=tool_fn):
            result = await fn(inputs)
            return result

        dag.add_node(tg.ToolNode(node_name, wrapper))

    for source, target in graph.edges:
        if source == END or target == END:
            continue
        dag.add_edge(source, target)

    return dag


async def run_without_tygent(workflow: Any, message: str) -> Dict[str, Any]:
    start = time.perf_counter()
    state = await workflow.ainvoke({"message": message, "token_counts": []})
    duration = time.perf_counter() - start
    tokens = sum(t.get("total_tokens", 0) for t in state.get("token_counts", []))
    print(f"Standard execution time: {duration:.2f}s | Tokens: {tokens}")
    return state


async def run_with_tygent(dag: tg.DAG, message: str) -> Dict[str, Any]:
    scheduler = tg.Scheduler(dag)
    start = time.perf_counter()
    outputs = await scheduler.execute({"message": message, "token_counts": []})
    duration = time.perf_counter() - start
    final_state = outputs.get("terminal_agent", {})
    tokens = sum(t.get("total_tokens", 0) for t in final_state.get("token_counts", []))
    print(f"Tygent execution time: {duration:.2f}s | Tokens: {tokens}")
    return final_state


async def main() -> None:
    config = json.loads(TATA_CONFIG_JSON)
    graph = build_graph(config)
    workflow = graph.compile()
    dag = langgraph_to_tygent(graph)

    customer_message = "I am considering transferring my home loan to another bank."

    print("Running workflow without Tygent...")
    standard_state = await run_without_tygent(workflow, customer_message)

    print("\nRunning workflow with Tygent...")
    tygent_state = await run_with_tygent(dag, customer_message)

    print("\nFinal responses:")
    print(f"Standard: {standard_state.get('message')}")
    print(f"Tygent:   {tygent_state.get('message')}")


if __name__ == "__main__":
    asyncio.run(main())
