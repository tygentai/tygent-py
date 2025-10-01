# Tygent Editor Extensions

Give your existing Python agents a one-click upgrade. The Tygent extensions for Visual Studio Code and Cursor insert the `tygent.install()` bootstrap into any open agent file, so you can immediately benefit from Tygent's parallel execution without rewriting your workflows.

## Prerequisites
- **Tygent** installed in the Python environment that runs your agents: `pip install tygent`
- **Visual Studio Code 1.75+** for the VS Code extension and/or the **Cursor IDE** (built on VS Code)
- **Node.js 16+** and **npm** to build a local VSIX package from this repository

Clone the repository so you can package the extensions locally:

```bash
git clone https://github.com/tygent-ai/tygent-py.git
cd tygent-py
```

---

## Visual Studio Code extension

### Build the extension package
1. Install dependencies and compile the TypeScript sources:
   ```bash
   cd vscode-extension
   npm install
   npm run compile
   ```
2. Package the extension (install `@vscode/vsce` if you do not have it yet):
   ```bash
   npx @vscode/vsce package
   ```
   The command produces a file similar to `tygent-agent-converter-0.0.1.vsix` that you can install in VS Code.

### Install in VS Code
1. Launch VS Code and open the Extensions view.
2. Choose the **…** menu ➜ **Install from VSIX…**, then select the generated `.vsix` file.
3. Restart VS Code if prompted.

### Convert a Python agent
1. Open the Python file that defines your agent.
2. Run **Tygent: Enable Agent** from the Command Palette.
3. The extension inserts `import tygent` and a `tygent.install([...])` call preloaded with the default integrations (Google AI, Anthropic, Hugging Face, Microsoft AI, Salesforce, Claude Code, Gemini CLI, OpenAI Codex) at the top of the file. If the file starts with a shebang (`#!`) the snippet is placed right after it, and the command quietly exits if the file already calls `tygent.install(` so you never get duplicate imports.【F:vscode-extension/src/extension.ts†L1-L38】
4. Save the file and run your agent as usual—Tygent now accelerates downstream framework calls automatically.

### Roll back
If you want to revert the change, simply remove the inserted `import tygent` statement and the generated `tygent.install([...])` call. Re-running the command after manual edits is safe; the extension re-applies the snippet only when it is missing.【F:vscode-extension/src/extension.ts†L1-L38】

---

## Cursor extension

Cursor shares the VS Code extension architecture, so you can install a packaged VSIX using the same steps.

### Build the extension package
```bash
cd cursor-extension
npm install
npm run compile
npx @vscode/vsce package
```

### Install in Cursor
1. Open Cursor ➜ click your profile ➜ **Settings** ➜ **Extensions** ➜ **Install from VSIX**.
2. Pick the generated `tygent-cursor-agent-converter-0.0.1.vsix` file.
3. Reload Cursor when prompted.

### Convert a Python agent in Cursor
1. Open any Python file you want to accelerate.
2. Run **Tygent: Enable Agent (Cursor)** from the Command Palette.
3. The command confirms the document is a Python file, adds the import only when it is missing, honours shebang headers, and surfaces errors if the editor cannot be modified. It now inserts the same pre-populated `tygent.install([...])` list used by the VS Code extension so Claude Code, Gemini CLI, and OpenAI Codex planners are patched automatically.【F:cursor-extension/src/extension.ts†L1-L53】
4. Save the file; your agent now boots with Tygent every time it runs inside Cursor.

---

## Validate the performance boost
- Run the quick smoke test to see a parallel plan beat a sequential baseline:
  ```bash
  pytest tests/test_extension_performance.py
  ```
  The test accelerates a two-step async workflow and asserts the Tygent version is faster than the naive implementation.【F:tests/test_extension_performance.py†L1-L41】
- For a deeper look, execute the benchmark suite:
  ```bash
  pytest tests/benchmarks/test_agent_conversion_benchmark.py
  ```
  It converts synthetic OpenAI, LangGraph, CrewAI, and NeMo style agents to Tygent, then compares latency and token usage to the original sequential runs.【F:tests/benchmarks/test_agent_conversion_benchmark.py†L1-L69】【F:tygent/testing/planning_examples.py†L1-L105】

---

## Troubleshooting
- **Command unavailable** – ensure the extension is enabled and you are running the correct command (`Tygent: Enable Agent` in VS Code or `Tygent: Enable Agent (Cursor)` in Cursor).
- **Nothing happens** – check whether the file already contains `tygent.install(`; the extensions intentionally avoid inserting duplicate snippets.【F:vscode-extension/src/extension.ts†L1-L38】【F:cursor-extension/src/extension.ts†L1-L53】
- **Wrong file type** – the Cursor command warns when the active document is not a Python file. Switch back to your agent module and re-run the command.【F:cursor-extension/src/extension.ts†L16-L48】

Need more help? File an issue in this repository with details about your editor version and the Python agent you are converting.
