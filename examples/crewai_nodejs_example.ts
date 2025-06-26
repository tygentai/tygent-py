/**
 * Example of using Tygent with CrewAI in Node.js/TypeScript.
 *
 * This script invokes the `exampleCrewAIAcceleration` helper from the Tygent
 * CrewAI integration to showcase accelerated CrewAI workflows.
*/
// @ts-nocheck
let exampleCrewAIAcceleration: any;
try {
  ({ exampleCrewAIAcceleration } = require('./tygent-js/src/integrations/crewai'));
} catch (e) {
  console.log('CrewAI integration unavailable:', e.message);
  process.exit(0);
}

async function main() {
  await exampleCrewAIAcceleration();
}

main().catch(console.error);
