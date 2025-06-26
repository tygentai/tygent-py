/**
 * Example usage of the Tygent Node.js package - Simple Accelerate Pattern
 * Shows how to use Tygent's accelerate() function for drop-in optimization.
 */

import { accelerate } from './tygent-js/src/accelerate';

// Set your API key in environment variables
// process.env.OPENAI_API_KEY = 'your-api-key'; // Uncomment and set your API key

/**
 * Example search function
 */
async function searchData(query: string): Promise<string> {
  console.log(`Searching for: ${query}`);
  // In a real implementation, this would call a search API
  await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call
  return `Search results for '${query}'`;
}

/**
 * Example weather function
 */
async function getWeather(location: string): Promise<any> {
  console.log(`Getting weather for: ${location}`);
  // In a real implementation, this would call a weather API
  await new Promise(resolve => setTimeout(resolve, 300)); // Simulate API call
  return { 
    temperature: 72, 
    conditions: 'Sunny', 
    location 
  };
}

/**
 * Example analysis function
 */
async function analyzeData(searchResults: string, weatherData: any): Promise<string> {
  console.log('Analyzing combined data...');
  await new Promise(resolve => setTimeout(resolve, 200)); // Simulate processing
  return `Analysis: ${searchResults} combined with weather ${JSON.stringify(weatherData)}`;
}

/**
 * Your existing workflow function - no changes needed
 */
async function myExistingWorkflow(): Promise<string> {
  console.log('Starting workflow...');
  
  // These calls normally run sequentially
  const searchResults = await searchData('artificial intelligence advancements');
  const weatherData = await getWeather('New York');
  const analysis = await analyzeData(searchResults, weatherData);
  
  console.log(`Final result: ${analysis}`);
  return analysis;
}

/**
 * Main execution function
 */
async function main() {
  console.log('=== Standard Execution ===');
  const startTime1 = Date.now();
  
  // Run your existing workflow normally
  const result1 = await myExistingWorkflow();
  
  const standardTime = (Date.now() - startTime1) / 1000;
  console.log(`Standard execution time: ${standardTime.toFixed(2)} seconds\n`);
  
  console.log('=== Accelerated Execution ===');
  const startTime2 = Date.now();
  
  // Only change: wrap your existing function with accelerate()
  const acceleratedWorkflow = accelerate(myExistingWorkflow);
  const result2 = await acceleratedWorkflow();
  
  const acceleratedTime = (Date.now() - startTime2) / 1000;
  console.log(`Accelerated execution time: ${acceleratedTime.toFixed(2)} seconds`);
  
  // Results should be identical
  console.log(`\nResults match: ${result1 === result2}`);
  
  if (standardTime > acceleratedTime) {
    const improvement = ((standardTime - acceleratedTime) / standardTime) * 100;
    console.log(`Performance improvement: ${improvement.toFixed(1)}% faster`);
  }
}

// Run the example
main().catch(console.error);
