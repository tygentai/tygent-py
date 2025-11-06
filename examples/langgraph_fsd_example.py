"""
LangGraph Functional Spec Writer (FSD) Example
----------------------------------------------
Demonstrates how Tygent executes a LangGraph-style workflow with cyclic subgraphs,
interactive checkpoints, persistent session state, and LLM-backed nodes.

The demo runs the same plan twice—first with a hand-written sequential loop (no Tygent),
then with the Tygent scheduler—so you can compare latency and output quality. A third
pass reuses the persisted session to show faster restarts. Reviewer feedback is automated
for the demo: LLM prompts are instructed to mimic a human reviewer who approves on the
second revision once comments are addressed.

Requirements
------------
- ``pip install openai``
- ``export OPENAI_API_KEY=...`` (or configure via environment on your platform)
"""

import asyncio
import difflib
import json
import os
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

# Ensure the local package is used when running from the source checkout
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "This example requires the 'openai' package. Install it via `pip install openai`."
    ) from exc

from tygent.plan_parser import parse_plan
from tygent.scheduler import Scheduler
from tygent.session import InMemorySessionStore

CURRENT_CONTEXT = ""
_ASYNC_CLIENT: Optional[AsyncOpenAI] = None
TOKEN_USAGE: Dict[str, Dict[str, int]] = defaultdict(
    lambda: {"prompt": 0, "completion": 0, "total": 0}
)
ITERATION_COUNTERS: Dict[str, int] = defaultdict(int)


def _load_dotenv(env_path: Path) -> None:
    """Populate os.environ with variables from a .env file if present."""

    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT_DIR / ".env")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client."""

    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export it before running the demo."
            )
        _ASYNC_CLIENT = AsyncOpenAI(api_key=api_key)
    return _ASYNC_CLIENT


def _extract_json(text: str) -> Dict[str, Any]:
    """Try to parse JSON from the LLM response."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse JSON: {text}") from err
        raise ValueError(f"Failed to parse JSON: {text}")


async def call_llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call the OpenAI Chat Completions API expecting a JSON object."""

    client = get_openai_client()
    response = await client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    usage = getattr(response, "usage", None)
    if usage:
        context_key = CURRENT_CONTEXT or "global"
        bucket = TOKEN_USAGE[context_key]
        bucket["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
        bucket["completion"] += getattr(usage, "completion_tokens", 0) or 0
        bucket["total"] += getattr(usage, "total_tokens", 0) or 0
    return _extract_json(content)


def set_context(name: str) -> None:
    """Set a logging prefix used by helper nodes."""

    global CURRENT_CONTEXT
    CURRENT_CONTEXT = name


def _log(message: str) -> None:
    """Context-aware logger for node executions."""

    prefix = f"[{CURRENT_CONTEXT}] " if CURRENT_CONTEXT else ""
    print(f"{prefix}{message}")


async def ingest_requirements(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the raw product brief into structured requirements."""

    brief = inputs.get("brief", "").strip()
    if not brief:
        raise ValueError("A product brief is required to start the workflow.")

    lines = [line.strip() for line in brief.splitlines() if line.strip()]
    title = lines[0].lstrip("# ").strip() if lines else "Untitled Product"
    goals = [line for line in lines if line.lower().startswith("goal")]

    await asyncio.sleep(0.05)
    return {
        "title": title,
        "summary": lines[1:] if len(lines) > 1 else [],
        "goals": goals or ["Goal: improve decision-making latency by 35%"],
        "owner": inputs.get("requester", "product.design@acme.dev"),
    }


def render_markdown(
    requirements: Dict[str, Any], revision: int, notes: List[str]
) -> str:
    """Render a Markdown document for the current revision."""

    body = textwrap.dedent(
        f"""
        # Functional Specification – {requirements.get('title', 'Untitled')}

        **Revision:** {revision}
        **Prepared by:** {requirements.get('owner', 'Unknown')}

        ## Summary
        {' '.join(requirements.get('summary', ['No summary provided.']))}

        ## Goals
        {chr(10).join(f'- {goal}' for goal in requirements.get('goals', []))}

        ## Architecture Overview
        - Drafts a modular service that streams insights to internal tools.
        - Includes analytics dashboards and alerting hooks.

        ## Review Notes
        {chr(10).join(f'- {note}' for note in notes) if notes else 'No open items.'}
        """
    ).strip()
    return body


async def draft_spec(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update the functional spec draft via the LLM."""

    requirements = inputs.get("ingest_requirements", {})
    feedback = inputs.get("apply_feedback", {})

    next_revision = int(feedback.get("next_revision", 1))
    revision = max(next_revision, 1)
    outstanding = feedback.get(
        "notes",
        [
            "Add non-functional requirements covering latency budgets.",
            "Document integration touchpoints with CRM systems.",
        ],
    )

    prompt = textwrap.dedent(
        f"""
        You are a staff product designer drafting a functional specification (FSD).
        Produce a JSON object with keys:
          - draft_markdown: markdown draft of the specification
          - revision_summary: bullet summary of changes made this revision
          - outstanding_items: list of open issues reviewers should check

        Rules:
          * Always respect the supplied requirements and goals.
          * If outstanding reviewer notes are provided, incorporate them and remove
            them from outstanding_items when addressed.
          * For the first revision (revision == 1), leave at least one outstanding item
            so reviewers can request updates.
        """
    ).strip()

    user = textwrap.dedent(
        f"""
        requirements: {json.dumps(requirements, indent=2)}
        revision: {revision}
        outstanding_reviewer_notes: {json.dumps(outstanding, indent=2)}
        """
    ).strip()

    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )

    draft_markdown = response.get("draft_markdown") or render_markdown(
        requirements, revision, outstanding if revision == 1 else []
    )
    outstanding_items = response.get("outstanding_items") or (
        outstanding if revision == 1 else []
    )

    status = "approved" if feedback.get("approved") else "in_review"
    context_key = CURRENT_CONTEXT or "global"
    ITERATION_COUNTERS[context_key] += 1
    _log(
        f"draft_spec: revision={revision} status={status} open_items={len(outstanding_items)}"
    )
    return {
        "revision": revision,
        "status": status,
        "content": draft_markdown,
        "revision_summary": response.get("revision_summary", ""),
        "open_items": outstanding_items,
    }


async def request_signoff(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Reviewer checkpoint that delegates the approval decision to the LLM."""

    draft = inputs.get("draft_spec", {})
    revision = int(draft.get("revision", 1))
    open_items = draft.get("open_items", [])
    summary = draft.get("revision_summary", "")

    prompt = textwrap.dedent(
        """
        You are the lead reviewer for a functional specification. Respond with JSON
        containing:
          - approved: boolean (true only if the draft is ready to publish)
          - comments: list of action items that must be addressed before approval
          - rationale: short explanation (string)

        Rules:
          * If revision < 2, always set approved to false and leave actionable comments.
          * If outstanding open_items are provided, ensure they are resolved before approving.
          * Once the draft addresses all feedback (no open items and revision >= 2), approve it.
        """
    ).strip()

    user = textwrap.dedent(
        f"""
        revision: {revision}
        open_items: {json.dumps(open_items, indent=2)}
        revision_summary: {summary}
        """
    ).strip()

    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )

    approved = bool(response.get("approved"))
    comments = response.get("comments") or []
    _log(f"collect_feedback: revision={revision} approved={approved}")
    return {
        "approved": approved,
        "comments": comments,
        "reviewed_revision": revision,
        "transcript": response.get("rationale", ""),
    }


async def apply_feedback(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge reviewer feedback and determine the next revision number."""

    draft = inputs.get("draft_spec", {})
    review = inputs.get("collect_feedback", {})
    current_revision = int(draft.get("revision", 1))
    approved = bool(review.get("approved"))

    notes = review.get("comments", [])
    next_revision = current_revision if approved else current_revision + 1

    await asyncio.sleep(0.05)
    _log(
        f"apply_feedback: revision={current_revision} approved={approved} next_revision={next_revision}"
    )
    return {
        "approved": approved,
        "notes": notes,
        "next_revision": next_revision,
        "revision_completed": current_revision,
    }


async def store_spec(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Store the final spec (no-op that returns a summary in this demo)."""

    draft = inputs.get("draft_spec", {})
    feedback = inputs.get("apply_feedback", {})
    status = "published" if feedback.get("approved") else "pending_approval"

    await asyncio.sleep(0.05)
    _log(f"publish_spec: status={status} revision={draft.get('revision')}")
    return {
        "status": status,
        "revision": draft.get("revision"),
        "excerpt": draft.get("content", "")[:200] + "...",
        "notes": feedback.get("notes", []),
    }


PLAN_DEFINITION: Dict[str, Any] = {
    "name": "langgraph_fsd",
    "steps": [
        {
            "name": "ingest_requirements",
            "func": ingest_requirements,
            "session": {"persist": True},
        },
        {
            "name": "draft_spec",
            "func": draft_spec,
            "dependencies": ["ingest_requirements", "apply_feedback"],
            "loop": {
                "group": "fsd_iteration",
                "termination": {"type": "fixed_point", "max_iterations": 2},
            },
            "interactive": True,
            "session": {"persist": True},
        },
        {
            "name": "collect_feedback",
            "func": request_signoff,
            "dependencies": ["draft_spec"],
            "loop": {"group": "fsd_iteration"},
            "interactive": True,
        },
        {
            "name": "apply_feedback",
            "func": apply_feedback,
            "dependencies": ["draft_spec", "collect_feedback"],
            "loop": {"group": "fsd_iteration"},
        },
        {
            "name": "publish_spec",
            "func": store_spec,
            "dependencies": ["draft_spec", "apply_feedback"],
            "critical": True,
        },
    ],
}


def describe_plan(plan: Dict[str, Any]) -> None:
    """Print a quick explanation of loops, interactive nodes, and persisted state."""

    loop_groups: Dict[str, Dict[str, Any]] = {}
    interactive_nodes: List[str] = []
    persistent_nodes: List[str] = []

    for step in plan.get("steps", []):
        loop = step.get("loop") or step.get("cycle")
        if loop:
            group = loop.get("group", step["name"])
            entry = loop_groups.setdefault(
                group,
                {
                    "nodes": [],
                    "termination": loop.get("termination", {"type": "single"}),
                },
            )
            entry["nodes"].append(step["name"])
            if loop.get("termination"):
                entry["termination"] = loop["termination"]
        if step.get("interactive"):
            interactive_nodes.append(step["name"])
        if step.get("session"):
            persistent_nodes.append(step["name"])

    print("Plan analysis:")
    if loop_groups:
        print("  • Cyclic components detected:")
        for group, meta in loop_groups.items():
            nodes = ", ".join(meta["nodes"])
            termination = meta["termination"]
            print(
                f"    - Group '{group}' cycles through [{nodes}] with termination policy {termination}"
            )
    if interactive_nodes:
        print(
            f"  • Interactive nodes (human-in-the-loop hooks): {', '.join(interactive_nodes)}"
        )
    if persistent_nodes:
        print(f"  • Nodes that persist session state: {', '.join(persistent_nodes)}")


async def run_baseline_sequential(
    brief: str, requester: str, max_iterations: int = 5
) -> Dict[str, Any]:
    """Sequential LangGraph-style execution without Tygent orchestration."""

    context = "baseline"
    set_context(context)
    TOKEN_USAGE[context] = {"prompt": 0, "completion": 0, "total": 0}
    ITERATION_COUNTERS[context] = 0
    base_inputs = {"brief": brief, "requester": requester}
    requirements = await ingest_requirements(base_inputs)
    state: Dict[str, Any] = {"ingest_requirements": requirements}
    feedback_state: Dict[str, Any] = {
        "approved": False,
        "notes": [],
        "next_revision": 1,
    }
    history: List[Dict[str, Any]] = []
    start = perf_counter()

    while len(history) < max_iterations:
        iteration_inputs = {
            **base_inputs,
            "ingest_requirements": state["ingest_requirements"],
            "apply_feedback": feedback_state,
        }
        draft = await draft_spec(iteration_inputs)
        review = await request_signoff({"draft_spec": draft})
        feedback_state = await apply_feedback(
            {"draft_spec": draft, "collect_feedback": review}
        )
        state.update(
            {
                "draft_spec": draft,
                "collect_feedback": review,
                "apply_feedback": feedback_state,
            }
        )
        history.append(
            {
                "draft": draft,
                "review": review,
                "feedback": feedback_state,
            }
        )
        if feedback_state.get("approved"):
            break

    publish = await store_spec(
        {"draft_spec": state["draft_spec"], "apply_feedback": feedback_state}
    )
    elapsed = perf_counter() - start
    set_context("")
    return {
        "elapsed": elapsed,
        "publish": publish,
        "history": history,
        "session": state,
        "tokens": TOKEN_USAGE[context],
        "iterations": ITERATION_COUNTERS[context],
    }


async def run_tygent_orchestrated(
    brief: str, requester: str, context: str
) -> Dict[str, Any]:
    """Execute the workflow through Tygent's scheduler."""

    dag, critical = parse_plan(PLAN_DEFINITION)
    session = InMemorySessionStore()
    scheduler = Scheduler(dag, session_store=session)
    scheduler.priority_nodes = critical

    set_context(context)
    TOKEN_USAGE[context] = {"prompt": 0, "completion": 0, "total": 0}
    ITERATION_COUNTERS[context] = 0
    start = perf_counter()
    results = await scheduler.execute({"brief": brief, "requester": requester})
    elapsed = perf_counter() - start
    set_context("")

    publish = results["results"]["publish_spec"]
    return {
        "elapsed": elapsed,
        "publish": publish,
        "results": results,
        "session": session,
        "scheduler": scheduler,
        "tokens": TOKEN_USAGE[context],
        "iterations": ITERATION_COUNTERS[context],
    }


async def rerun_with_persisted_state(
    scheduler: Scheduler, inputs: Dict[str, Any], context: str
) -> Tuple[float, Dict[str, Any], Dict[str, int], int]:
    """Re-execute using the same scheduler to highlight session persistence."""

    set_context(context)
    TOKEN_USAGE[context] = {"prompt": 0, "completion": 0, "total": 0}
    ITERATION_COUNTERS[context] = 0
    start = perf_counter()
    results = await scheduler.execute(inputs)
    elapsed = perf_counter() - start
    set_context("")
    return (
        elapsed,
        results["results"]["publish_spec"],
        TOKEN_USAGE[context],
        ITERATION_COUNTERS[context],
    )


def compare_outputs(
    baseline_publish: Dict[str, Any], tygent_publish: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute simple comparison metrics for the generated specification."""

    baseline_content = baseline_publish.get("excerpt", "")
    tygent_content = tygent_publish.get("excerpt", "")
    ratio = difflib.SequenceMatcher(None, baseline_content, tygent_content).ratio()
    return {
        "similarity": ratio,
        "baseline_revision": baseline_publish.get("revision"),
        "tygent_revision": tygent_publish.get("revision"),
        "baseline_status": baseline_publish.get("status"),
        "tygent_status": tygent_publish.get("status"),
    }


async def run_demo() -> None:
    """Run the demo with and without Tygent and report the results."""

    product_brief = textwrap.dedent(
        """
        # GraphInsights
        Goal: surface explainable metrics for operations leadership.
        Goal: reduce time-to-diagnosis for incident commanders.
        The platform ingests telemetry and renders pre-built decision flows.
        """
    ).strip()

    print("\n=== LangGraph FSD Demo ===")
    print(
        "Human feedback is automated for this demo—Tygent treats the reviewer steps as interactive nodes."
    )
    describe_plan(PLAN_DEFINITION)

    print("\n--- Baseline sequential execution (no Tygent) ---")
    baseline_result = await run_baseline_sequential(product_brief, "fsd@acme.dev")
    print(
        f"Baseline latency: {baseline_result['elapsed']*1000:.1f} ms, "
        f"iterations: {baseline_result['iterations']}, "
        f"status: {baseline_result['publish']['status']}"
    )

    print("\n--- Tygent orchestrated execution ---")
    tygent_context = "tygent-first"
    tygent_result = await run_tygent_orchestrated(
        product_brief, "fsd@acme.dev", tygent_context
    )
    print(
        f"Tygent latency: {tygent_result['elapsed']*1000:.1f} ms, "
        f"status: {tygent_result['publish']['status']}"
    )
    session_snapshot = tygent_result["session"].snapshot()
    print(
        "Persisted session keys (cross-run state):",
        list(session_snapshot.keys()),
    )

    print("\n--- Demonstrating persistent session state ---")
    rerun_context = "tygent-rerun"
    repeat_latency, repeat_publish, rerun_tokens, rerun_iterations = (
        await rerun_with_persisted_state(
            tygent_result["scheduler"],
            {"brief": product_brief, "requester": "fsd@acme.dev"},
            rerun_context,
        )
    )
    print(
        f"Re-run latency with existing session: {repeat_latency*1000:.1f} ms "
        f"(status: {repeat_publish['status']}, revision: {repeat_publish['revision']})"
    )

    print("\n--- Output comparison ---")
    comparison = compare_outputs(baseline_result["publish"], tygent_result["publish"])
    print(
        f"Content similarity: {comparison['similarity']*100:.2f}% "
        f"(baseline revision {comparison['baseline_revision']} / status {comparison['baseline_status']}, "
        f"Tygent revision {comparison['tygent_revision']} / status {comparison['tygent_status']})"
    )
    print("Excerpt preview (Tygent):\n", tygent_result["publish"]["excerpt"])

    latency_delta = baseline_result["elapsed"] - tygent_result["elapsed"]
    rows = [
        {
            "mode": "Baseline (sequential)",
            "latency": baseline_result["elapsed"] * 1000,
            "iterations": baseline_result["iterations"],
            "status": baseline_result["publish"]["status"],
            "tokens": baseline_result["tokens"]["total"],
        },
        {
            "mode": "Tygent (first run)",
            "latency": tygent_result["elapsed"] * 1000,
            "iterations": tygent_result["iterations"],
            "status": tygent_result["publish"]["status"],
            "tokens": tygent_result["tokens"]["total"],
        },
        {
            "mode": "Tygent (re-run)",
            "latency": repeat_latency * 1000,
            "iterations": rerun_iterations,
            "status": repeat_publish["status"],
            "tokens": rerun_tokens["total"],
        },
    ]

    headers = ["Execution Mode", "Latency (ms)", "Iterations", "Status", "Tokens"]
    col_widths = [
        max(len(headers[0]), *(len(row["mode"]) for row in rows)),
        len(headers[1]),
        len(headers[2]),
        max(len(headers[3]), *(len(row["status"]) for row in rows)),
        len(headers[4]),
    ]

    def _format_row(row: List[str], widths: List[int]) -> str:
        return " | ".join(col.ljust(width) for col, width in zip(row, widths))

    print("\n=== Summary Table ===")
    print(_format_row(headers, col_widths))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in rows:
        formatted = [
            row["mode"],
            f"{row['latency']:.1f}",
            str(row["iterations"]),
            row["status"],
            str(row["tokens"]),
        ]
        print(_format_row(formatted, col_widths))

    print("\n=== Summary ===")
    print("• Cycles are handled automatically via the fixed-point termination policy.")
    print("• Interactive review nodes run with mocked human feedback for automation.")
    print("• Session state persists between runs, enabling faster replays.")
    print(
        f"• Latency improvement on first pass: {latency_delta*1000:.1f} ms "
        "(positive values mean Tygent was faster)."
    )
    print(
        f"• Second Tygent run reused cached state and completed in {repeat_latency*1000:.1f} ms."
    )
    print(
        f"• Token usage (prompt+completion): baseline {baseline_result['tokens']['total']}, "
        f"Tygent first run {tygent_result['tokens']['total']}, rerun {rerun_tokens['total']}."
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
