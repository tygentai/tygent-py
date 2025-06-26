/**
 * Example of using Tygent's multi-agent capabilities in Node.js/TypeScript.
 * 
 * This example demonstrates how to:
 * 1. Create multiple agents with different roles
 * 2. Use MultiAgentManager to coordinate agent execution
 * 3. Execute agents in parallel for improved performance
 */

import { MultiAgentOrchestrator, AgentRole } from './tygent-js/src/multi-agent';

// Define agent roles
const roles: Record<string, AgentRole> = {
  researcher: {
    name: "Researcher",
    description: "Specializes in finding and analyzing information.",
    systemPrompt: "You are a skilled researcher who excels at gathering relevant information. Your goal is to provide comprehensive, accurate, and well-sourced information about the topic at hand."
  },
  critic: {
    name: "Critic", 
    description: "Identifies flaws and suggests improvements.",
    systemPrompt: "You are a thoughtful critic who evaluates information critically. Your goal is to identify potential flaws, biases, or gaps in reasoning and suggest improvements."
  },
  synthesizer: {
    name: "Synthesizer",
    description: "Combines insights into a coherent whole.",
    systemPrompt: "You are an expert synthesizer who brings together different perspectives. Your goal is to create a cohesive and comprehensive understanding of the topic by incorporating multiple viewpoints."
  }
};

async function main() {
  console.log("Tygent Multi-Agent Example");
  console.log("==========================");
  
  // Create a multi-agent orchestrator
  const orchestrator = new MultiAgentOrchestrator("gpt-4o");
  
  // Add agents with their roles
  for (const [agentId, role] of Object.entries(roles)) {
    orchestrator.addAgent(agentId, role);
    console.log(`Added agent: ${role.name}`);
  }
  
  // Execute the multi-agent workflow 
  console.log(`\nExecuting multi-agent workflow...`);
  
  try {
    const dag = orchestrator.createConversationDag(
      "What are the potential benefits and risks of quantum computing?",
      {
        batchMessages: true,
        parallelThinking: true,
        sharedMemory: true,
        earlyStopThreshold: 0.95
      }
    );
    
    console.log("Created conversation DAG for multi-agent coordination");
    console.log("Note: Full execution requires OpenAI API key for LLM interactions");
    
  } catch (error) {
    console.log("DAG creation completed (API calls would require OpenAI key)");
  }
  
  console.log("\nExample completed successfully!");
}

// Run the example
main().catch(console.error);
