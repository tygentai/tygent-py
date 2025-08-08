import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"

SKIP_REQUIREMENTS = {
    "google_ai_example.py": ["GOOGLE_API_KEY"],
    "google_adk_market_analysis.py": [
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ],
    "microsoft_ai_example.py": ["AZURE_OPENAI_KEY"],
    "salesforce_example.py": ["SALESFORCE_USERNAME"],
}

OPTIONAL_MODULES = {
    "langgraph_integration.py": ["langgraph", "langchain", "langchain_community"]
}

results = []

for example in sorted(EXAMPLES_DIR.glob("*.py")):
    env_vars = SKIP_REQUIREMENTS.get(example.name)
    if env_vars:
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        if not any(v in os.environ for v in env_vars):
            missing = " or ".join(env_vars)
            print(f"Skipping {example.name} (missing {missing})")
            results.append((example.name, None, False, None))
            continue

    module_names = OPTIONAL_MODULES.get(example.name)
    if module_names:
        from importlib.util import find_spec

        if isinstance(module_names, str):
            module_names = [module_names]

        missing = [m for m in module_names if find_spec(m) is None]
        if missing:
            print(
                f"Installing missing modules for {example.name}: {', '.join(missing)}"
            )
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", *missing]
                )
            except subprocess.CalledProcessError:
                print(
                    f"Skipping {example.name} (failed to install {', '.join(missing)})"
                )
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
