/**
 * Example of using Tygent with Langflow in Node.js/TypeScript.
 *
 * This script executes the `exampleLangflowAcceleration` helper from the
 * Langflow integration to demonstrate how Tygent accelerates Langflow
 * workflows.
*/
// @ts-nocheck
let exampleLangflowAcceleration: any;
try {
  ({ exampleLangflowAcceleration } = require('./tygent-js/src/integrations/langflow'));
} catch (e) {
  console.log('Langflow integration unavailable:', e.message);
  process.exit(0);
}

async function main() {
  await exampleLangflowAcceleration();
}

main().catch(console.error);
