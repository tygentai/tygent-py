/**
 * Example of using Tygent with Google AI/Gemini models in Node.js
 * 
 * This example demonstrates how to integrate Tygent with Google's Gemini
 * models to optimize execution of multi-step workflows.
 */
let dotenv: any;
try {
  dotenv = require('dotenv');
} catch (e) {
  console.log('dotenv package not available');
}
import { GoogleAIIntegration, GoogleAIBatchProcessor } from './tygent-js/src/integrations/google-ai';

// Load environment variables if dotenv is available
if (dotenv && dotenv.config) {
  dotenv.config();
}

// This example requires the @google/generative-ai package
let GoogleGenerativeAI: any;
try {
  const { GoogleGenerativeAI: GoogleAI } = require('@google/generative-ai');
  GoogleGenerativeAI = GoogleAI;
} catch (error) {
  console.error("This example requires the @google/generative-ai package.");
  console.error("Install it with: npm install @google/generative-ai");
  process.exit(0);
}

// Check for API key
const API_KEY = process.env.GOOGLE_API_KEY;
if (!API_KEY) {
  console.error("GOOGLE_API_KEY environment variable not set.");
  console.error("Get an API key from https://makersuite.google.com/app/apikey");
  process.exit(1);
}

// Initialize the Google AI API
const genAI = new GoogleGenerativeAI(API_KEY);

async function main() {
  // Configure the Gemini model
  const model = genAI.getGenerativeModel({ model: 'gemini-pro' });
  
  console.log("=== Tygent Google AI Integration Example ===");
  console.log("Creating a travel planning assistant with optimized execution...\n");
  
  // Create a Google AI integration with Tygent
  const googleAI = new GoogleAIIntegration(model);
  
  // Define optimization constraints
  googleAI.optimize({
    maxParallelCalls: 2,
    maxExecutionTime: 30000,  // 30 seconds
    priorityNodes: ["weather_info"]  // Prioritize weather info
  });
  
  // Add nodes to the execution DAG
  googleAI.addNode(
    "destination_analysis",
    "Analyze {destination} as a travel destination. " +
    "Provide key highlights, best time to visit, and any travel warnings.",
    []
  );
  
  googleAI.addNode(
    "weather_info",
    "What's the typical weather in {destination} during {month}?",
    []
  );
  
  googleAI.addNode(
    "activity_suggestions",
    "Suggest 5 must-do activities in {destination} during {month}, " +
    "taking into account the weather conditions: {weather_info}",
    ["weather_info"]
  );
  
  googleAI.addNode(
    "accommodation_suggestions",
    "Recommend 3 types of accommodations in {destination} " +
    "suitable for a {duration} day trip in {month}.",
    []
  );
  
  googleAI.addNode(
    "travel_plan",
    "Create a {duration} day travel itinerary for {destination} in {month}. " +
    "Include these destination highlights: {destination_analysis} " +
    "Include these activities: {activity_suggestions} " +
    "Include these accommodations: {accommodation_suggestions}",
    ["destination_analysis", "activity_suggestions", "accommodation_suggestions"]
  );
  
  // Execute the DAG with inputs
  const inputs = {
    destination: "Kyoto, Japan",
    month: "April",
    duration: 5
  };
  
  console.log(`Planning a trip to ${inputs.destination} in ${inputs.month}...`);
  
  // Run the optimized execution
  const startTime = Date.now();
  const results = await googleAI.execute(inputs);
  const endTime = Date.now();
  
  // Display the results
  console.log("\n=== Travel Plan Generated ===");
  console.log(results.travel_plan.substring(0, 1000) + "...\n");
  
  console.log(`Execution completed in ${(endTime - startTime) / 1000} seconds`);
  console.log(`Number of nodes executed: ${Object.keys(results).length}`);
  
  // Demonstrate batch processing
  console.log("\n=== Demonstrating Batch Processing ===");
  
  // Create a batch processor
  const batchProcessor = new GoogleAIBatchProcessor(
    model,
    2,  // batchSize
    2   // maxConcurrentBatches
  );
  
  // Define batch processing function
  async function processCity(city: string, model: any) {
    const response = await model.generateContent(`What are the top 3 attractions in ${city}?`);
    return {
      city,
      attractions: response.response.text()
    };
  }
  
  // Process a batch of cities
  const cities = ["Paris", "New York", "Tokyo", "Rome", "Sydney"];
  console.log(`Processing information for ${cities.length} cities in optimized batches...`);
  
  const batchStartTime = Date.now();
  const cityResults = await batchProcessor.process(cities, processCity);
  const batchEndTime = Date.now();
  
  console.log("\n=== Batch Processing Results ===");
  for (const result of cityResults) {
    console.log(`\n${result.city}:`);
    console.log(`${result.attractions.substring(0, 150)}...`);
  }
  
  console.log(`\nBatch processing completed in ${(batchEndTime - batchStartTime) / 1000} seconds`);
  console.log("With standard sequential processing, this would have taken significantly longer.");
}

// Run the example
main().catch(error => {
  console.error("Error running example:", error);
});
