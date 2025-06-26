/**
 * Example of using Tygent with Microsoft AI services in Node.js
 * 
 * This example demonstrates how to integrate Tygent with Microsoft's Azure OpenAI 
 * Service to optimize execution of multi-step workflows.
 */
// @ts-nocheck
let dotenv: any;
try {
  dotenv = require('dotenv');
} catch (e) {
  console.log('dotenv package not available');
}
let MicrosoftAIIntegration: any;
let SemanticKernelOptimizer: any;
try {
  ({ MicrosoftAIIntegration, SemanticKernelOptimizer } = require('./tygent-js/src/integrations/microsoft-ai'));
} catch (e) {
  console.log('Microsoft AI integration unavailable:', e.message);
  process.exit(0);
}

// Load environment variables
if (dotenv && dotenv.config) {
  dotenv.config();
}

// This example requires the @azure/openai package
let OpenAIClient: any;
try {
  const { OpenAIClient, AzureKeyCredential } = require('@azure/openai');
  OpenAIClient = { OpenAIClient, AzureKeyCredential };
} catch (error) {
  console.error("This example requires the @azure/openai package.");
  console.error("Install it with: npm install @azure/openai");
  process.exit(0);
}

// Check for environment variables
const AZURE_OPENAI_KEY = process.env.AZURE_OPENAI_KEY;
const AZURE_OPENAI_ENDPOINT = process.env.AZURE_OPENAI_ENDPOINT;
const DEPLOYMENT_NAME = process.env.AZURE_OPENAI_DEPLOYMENT_NAME || "gpt-4";

if (!AZURE_OPENAI_KEY || !AZURE_OPENAI_ENDPOINT) {
  console.error("Please set the AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables.");
  console.error("You can get these from the Azure portal.");
  process.exit(0);
}

async function main() {
  // Initialize the Azure OpenAI client
  const client = new OpenAIClient.OpenAIClient(
    AZURE_OPENAI_ENDPOINT,
    new OpenAIClient.AzureKeyCredential(AZURE_OPENAI_KEY)
  );
  
  console.log("=== Tygent Microsoft AI Integration Example ===");
  console.log("Creating a market research analysis workflow with optimized execution...\n");
  
  // Create a Microsoft AI integration with Tygent
  const azureAI = new MicrosoftAIIntegration(client, DEPLOYMENT_NAME);
  
  // Define optimization constraints
  azureAI.optimize({
    maxParallelCalls: 3,
    maxExecutionTime: 30000,  // 30 seconds
    priorityNodes: ["market_trends"]  // Prioritize market trends
  });
  
  // Add nodes to the execution DAG
  azureAI.addNode(
    "market_overview",
    "Provide a high-level overview of the {industry} industry in {region} in 2025.",
    []
  );
  
  azureAI.addNode(
    "market_trends",
    "What are the top 5 emerging trends in the {industry} industry in {region} for 2025?",
    []
  );
  
  azureAI.addNode(
    "competitor_analysis",
    "Identify and analyze the top 3 competitors in the {industry} industry in {region}.",
    []
  );
  
  azureAI.addNode(
    "regulatory_landscape",
    "Summarize the key regulatory considerations for the {industry} industry in {region}.",
    []
  );
  
  azureAI.addNode(
    "growth_opportunities",
    "Based on the following information:\n" +
    "Market overview: {market_overview}\n" +
    "Emerging trends: {market_trends}\n" +
    "Competitor analysis: {competitor_analysis}\n" +
    "Regulatory landscape: {regulatory_landscape}\n\n" +
    "Identify 3-5 high-potential growth opportunities for a new entrant in the {industry} industry in {region}.",
    ["market_overview", "market_trends", "competitor_analysis", "regulatory_landscape"]
  );
  
  azureAI.addNode(
    "entry_strategy",
    "Create a market entry strategy outline for the {industry} industry in {region}, " +
    "focusing on these growth opportunities: {growth_opportunities}",
    ["growth_opportunities"]
  );
  
  // Execute the DAG with inputs
  const inputs = {
    industry: "renewable energy",
    region: "Southeast Asia"
  };
  
  console.log(`Analyzing the ${inputs.industry} industry in ${inputs.region}...`);
  
  // Run the optimized execution
  const startTime = Date.now();
  const results = await azureAI.execute(inputs);
  const endTime = Date.now();
  
  // Display the results
  console.log("\n=== Market Entry Strategy ===");
  console.log(results.entry_strategy.substring(0, 1000) + "...\n");
  
  console.log(`Execution completed in ${(endTime - startTime) / 1000} seconds`);
  console.log(`Number of nodes executed: ${Object.keys(results).length}`);
  
  // Demonstrate Semantic Kernel integration
  console.log("\n=== Would you like to see Semantic Kernel integration? ===");
  console.log("Note: This requires Semantic Kernel to be installed.");
  console.log("You can install it with: npm install @microsoft/semantic-kernel");
  console.log("(Example implementation would integrate Tygent's optimization with Semantic Kernel plugins)");
}

/**
 * Example implementation of Semantic Kernel integration with Tygent.
 * 
 * Note: This is a placeholder implementation that would require the
 * @microsoft/semantic-kernel package to be installed to actually run.
 */
async function semanticKernelExample() {
  try {
    // This require would normally fail unless semantic-kernel is installed
    const semanticKernel = require('@microsoft/semantic-kernel');
    
    // Initialize a Semantic Kernel instance
    const kernel = new semanticKernel.Kernel();
    
    // Add Azure OpenAI service
    const azureOpenAIService = new semanticKernel.AzureOpenAIChatCompletion({
      deploymentName: DEPLOYMENT_NAME,
      endpoint: AZURE_OPENAI_ENDPOINT,
      apiKey: AZURE_OPENAI_KEY
    });
    
    kernel.addService(azureOpenAIService);
    
    // Create Tygent optimizer for Semantic Kernel
    const skOptimizer = new SemanticKernelOptimizer(kernel);
    
    // Create a plugin
    // This is a simplified example of what might be done
    const pluginFunctions = {
      sentimentAnalysis: async (input: string) => {
        // In a real implementation, this would use the kernel
        return `The sentiment of "${input}" is positive.`;
      }
    };
    
    // Register the plugin
    skOptimizer.registerPlugin(pluginFunctions, "TextAnalysis");
    
    // Create an optimized plan
    skOptimizer.createPlan("Analyze sentiment and provide recommendations");
    
    // Optimize with constraints
    skOptimizer.optimize({
      maxParallelCalls: 2,
      maxExecutionTime: 10000
    });
    
    // Execute the optimized plan
    const results = await skOptimizer.execute({
      input: "The new product launch exceeded our expectations with record sales, " +
             "though some customers reported minor usability issues."
    });
    
    console.log("\n=== Semantic Kernel Results ===");
    console.log(results);
    
  } catch (error) {
    console.log("Semantic Kernel is not installed. Skipping this example.");
  }
}

// Run the example
main().catch(error => {
  console.error("Error running example:", error);
});
