import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

SKIP_REQUIREMENTS = {
    "google_ai_example.py": "GOOGLE_API_KEY",
    "microsoft_ai_example.py": "AZURE_OPENAI_KEY",
    "salesforce_example.py": "SALESFORCE_USERNAME",
}

OPTIONAL_MODULES = {"langgraph_integration.py": "langchain"}

results = []

for example in sorted(EXAMPLES_DIR.glob("*.py")):
    env_var = SKIP_REQUIREMENTS.get(example.name)
    if env_var and env_var not in os.environ:
        print(f"Skipping {example.name} (missing {env_var})")
        results.append((example.name, None, False, None))
        continue

    module_name = OPTIONAL_MODULES.get(example.name)
    if module_name:
        from importlib.util import find_spec

        if find_spec(module_name) is None:
            print(f"Skipping {example.name} (missing {module_name})")
            results.append((example.name, None, False, None))
            continue

    print(f"Running {example.name}...")
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(example)], capture_output=True, text=True
    )
    end = time.perf_counter()
    duration = end - start
    output = proc.stdout + proc.stderr
    success = proc.returncode == 0
    improvement = None
    m = re.search(r"Performance improvement: ([0-9.]+)%", output)
    if m:
        improvement = float(m.group(1))
    results.append((example.name, duration, success, improvement))
    if not success:
        print(output)

print("\nBenchmark results:")
for name, duration, success, improvement in results:
    status = "ok" if success else "fail"
    line = f"{name}: {status}, {duration:.2f}s" if duration else f"{name}: skipped"
    if improvement is not None:
        line += f", {improvement:.1f}% faster"
    print(line)

failed = any(
    duration is not None and not success for _, duration, success, _ in results
)
if failed:
    sys.exit(1)
