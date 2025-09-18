"""
ADK → Tygent (multi-task) with scheduling analysis and side-by-side printing (1:1 transform)

What you get:
- Build JSON DAG plans (SPEC) via Google ADK (Gemini) for up to 100 tasks.
- Supports --basic mode using a minimal prompt template.
- Tygent plan is a 1:1 transform of SPEC (same step names, count, order).
- Tygent steps INCLUDE the original PROMPTS so they print on the right.
- Graph analysis summarized for scheduling:
    * Critical path + length
    * Topological "waves" (levels) → max parallelism
    * Sources & sinks
    * Bottlenecks (high fan-out) and joins (high fan-in)
    * Basic counts (nodes/edges)

Side-by-side printing:
- Left: SPEC (prompt, deps, links)
- Right: Tygent (prompt, deps, links, critical flag, level), plus a scheduling summary up top.

Install:
  pip install google-adk google-genai python-dotenv

Env (one of):
  GOOGLE_API_KEY
  or
  GOOGLE_APPLICATION_CREDENTIALS (+ GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import textwrap
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# ----------------- Optional: load .env -----------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------- Google ADK / GenAI -----------------
try:
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except Exception:
    print("This script requires google-adk and google-genai.")
    print("Install with: pip install google-adk google-genai python-dotenv")
    raise SystemExit(1)

if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS before running.")
    raise SystemExit(1)

# ----------------- Prompting -----------------
INSTRUCTION = (
    "You are an expert orchestration planner that outputs clean JSON DAGs "
    "for multi-step AI agent workflows."
)

BASE_PROMPT = (
    "{task}: Analyze opportunities, constraints, data sources, risks, and validation steps "
    "to produce a concise, actionable deliverable."
)

# Detailed template
PLAN_GENERATION_PROMPT_TEMPLATE = (
    "Generate a comprehensive agent workflow plan as a JSON object representing a DAG for the task:\n"
    "TASK: {task}\n\n"
    "The JSON must map step names to objects with:\n"
    "  - 'prompt': a concise instruction (may embed URLs if needed)\n"
    "  - 'deps': a list of dependencies (step names)\n"
    "  - 'links' (optional): a list of external URLs required for this step\n\n"
    f"Use this prompt template per step: \"{BASE_PROMPT}\" with an appropriate task.\n"
    "Cover discovery, analysis, data gathering, validation, synthesis, and an 'executive_summary'.\n"
    "Respond with valid JSON ONLY (no markdown or extra text)."
)

# Basic template (requested)
BASIC_PLAN_GENERATION_PROMPT_TEMPLATE = (
    "Generate a comprehensive agent workflow plan as a JSON object:\n"
    "TASK: {task}\n\n"
    "Respond with valid JSON ONLY (no markdown or extra text)."
)

# ----------------- Runner helpers -----------------
async def _create_runner() -> InMemoryRunner:
    agent = LlmAgent(
        name="planner",
        model="gemini-2.5-pro",
        instruction=INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.5, max_output_tokens=64000
        ),
    )
    runner = InMemoryRunner(agent)
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id="user", session_id="session"
    )
    return runner

def _extract(events: List[Any]) -> str:
    if not events:
        return ""
    event = events[0]
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", [])
    if parts:
        return parts[0].text
    return str(event)

async def _call_runner(runner: InMemoryRunner, name: str, prompt: str) -> str:
    events: List[Any] = []
    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        events.append(event)
    return _extract(events)

def _parse_plan_json(text: str) -> Dict[str, Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Failed to parse plan JSON: {text}")

async def build_plan_for_task(
    runner: InMemoryRunner,
    task: str,
    basic: bool = False,
) -> Dict[str, Dict[str, Any]]:
    template = BASIC_PLAN_GENERATION_PROMPT_TEMPLATE if basic else PLAN_GENERATION_PROMPT_TEMPLATE
    prompt = template.format(task=task)
    raw = await _call_runner(runner, f"plan:{task[:40]}", prompt)
    return _parse_plan_json(raw)

# ----------------- DAG / analysis utilities -----------------
class CycleError(RuntimeError):
    pass

def _topo_sort(nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    indeg = {n: 0 for n in nodes}
    graph = defaultdict(list)
    for n, v in nodes.items():
        for d in v.get("deps", []):
            if d not in nodes:
                raise KeyError(f"Unknown dependency '{d}' for node '{n}'")
            graph[d].append(n)
            indeg[n] += 1
    q = deque([n for n, deg in indeg.items() if deg == 0])
    order: List[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(nodes):
        raise CycleError("Cycle detected in plan DAG")
    return order

def _children_map(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    children = defaultdict(list)
    for n, v in nodes.items():
        for d in v.get("deps", []):
            children[d].append(n)
    return children

def _longest_path(nodes: Dict[str, Dict[str, Any]], sink: str | None = None) -> List[str]:
    order = _topo_sort(nodes)
    children = _children_map(nodes)
    dist = {n: 0 for n in nodes}
    prev = {n: None for n in nodes}
    for u in order:
        for v in children[u]:
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1
                prev[v] = u
    target = sink if sink in nodes else max(dist, key=dist.get)
    path: List[str] = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def _levelize(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    Compute earliest-start level (wave) for each node:
    level[n] = 0 if no deps else 1 + max(level[d] for d in deps)
    """
    order = _topo_sort(nodes)
    level: Dict[str, int] = {}
    for n in order:
        deps = nodes[n].get("deps", [])
        if not deps:
            level[n] = 0
        else:
            level[n] = 1 + max(level[d] for d in deps)
    return level

def _graph_analysis(nodes: Dict[str, Dict[str, Any]], prefer_sink: str = "executive_summary") -> Dict[str, Any]:
    topo = _topo_sort(nodes)
    children = _children_map(nodes)
    level = _levelize(nodes)
    sources = [n for n, v in nodes.items() if not v.get("deps")]
    sinks = [n for n in nodes if not children[n]]

    # degrees
    in_deg = {n: len(nodes[n].get("deps", [])) for n in nodes}
    out_deg = {n: len(children[n]) for n in nodes}

    # widest wave
    wave_counts = defaultdict(int)
    for n, lv in level.items():
        wave_counts[lv] += 1
    max_parallelism = max(wave_counts.values()) if wave_counts else 1
    max_parallel_waves = [lv for lv, c in wave_counts.items() if c == max_parallelism]

    # critical path
    cp = _longest_path(nodes, sink=prefer_sink)
    cp_len = len(cp)

    # joins (high fan-in) & bottlenecks (high fan-out)
    top_joins = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:3]
    top_bottlenecks = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:3]

    # edges count
    edge_count = sum(in_deg.values())

    # waves list (levels -> nodes) - useful to schedule
    waves: List[List[str]] = []
    if wave_counts:
        max_lv = max(level.values())
        for lv in range(max_lv + 1):
            group = [n for n, l in level.items() if l == lv]
            group.sort()
            waves.append(group)

    return {
        "topo_order": topo,
        "levels": level,
        "waves": waves,
        "sources": sources,
        "sinks": sinks,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "edge_count": edge_count,
        "critical_path": cp,
        "critical_path_length": cp_len,
        "max_parallelism": max_parallelism,
        "max_parallel_waves": max_parallel_waves,
        "top_joins": top_joins,
        "top_bottlenecks": top_bottlenecks,
    }

# ----------------- Links helpers -----------------
_URL_RE = re.compile(r'(?i)\b((?:https?://|www\.)[^\s<>"\'\)\]]+)')

def extract_links_from_text(text: str | None) -> List[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    norm = []
    for u in urls:
        if u.lower().startswith("www."):
            norm.append("http://" + u)
        else:
            norm.append(u)
    seen, out = set(), []
    for u in norm:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def merge_links(spec_links: Any, prompt_links: List[str]) -> List[str]:
    merged: List[str] = []
    if isinstance(spec_links, list):
        for v in spec_links:
            if isinstance(v, str):
                merged.append(v)
    merged.extend(prompt_links)
    seen, out = set(), []
    for u in merged:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# ----------------- Tygent transform (1:1 base steps, includes prompts) -----------------
def build_tygent_transform_with_prompts_and_analysis(
    spec: Dict[str, Dict[str, Any]],
    redundancy_mode: str = "inline",  # "inline" | "steps" | "off"
) -> Dict[str, Any]:
    """
    Transform SPEC into a Tygent plan with the SAME base steps (names, count, order).
    - Each Tygent step includes: name, dependencies, prompt, links, is_critical, tags, level.
    - Redundancy kept inline by default (metadata) to avoid step count differences.
    - Graph analysis computed for scheduling summary.
    """
    # Validate + analyze
    analysis = _graph_analysis(spec, prefer_sink="executive_summary")
    critical_set: Set[str] = set(analysis["critical_path"])
    levels: Dict[str, int] = analysis["levels"]

    steps: List[Dict[str, Any]] = []
    redundancy_map: Dict[str, Dict[str, Any]] = {}

    # Build base steps (1:1 with SPEC) including prompts and level
    for name, node in spec.items():
        prompt = node.get("prompt", "")
        deps = node.get("deps", [])
        links = merge_links(node.get("links", []), extract_links_from_text(prompt))
        level = levels.get(name, 0)

        # inline redundancy definition (metadata only)
        needs_check = name in {"validation_and_triangulation", "synthesis_and_strategic_recommendations", "executive_summary"} or bool(deps)
        redundancy = None
        if needs_check:
            redundancy = {
                "name": f"{name}__redundancy",
                "depends_on": [name],
                "prompt_template": (
                    "You are a consistency checker. Review the output of '{name}' and its inputs:\n"
                    "Inputs (by step): {inputs_json}\n"
                    "Output ({name}): {output}\n\n"
                    "Task: Identify contradictions, missing justifications, or unsupported claims in the output "
                    "relative to the inputs. Respond with 'OK' if consistent; otherwise list issues as bullets."
                ),
            }
            redundancy_map[name] = redundancy

        steps.append({
            "name": name,
            "dependencies": deps,
            "prompt": prompt,          # <-- include prompt for printing/scheduling context
            "links": links,
            "level": level,            # <-- earliest-start wave (0-based)
            "is_critical": (name in critical_set),
            "tags": (["critical"] if name in critical_set else []),
            "redundancy": redundancy if redundancy_mode != "off" else None,
        })

    # Optionally materialize redundancy checks as extra nodes (adds steps)
    if redundancy_mode == "steps":
        materialized: List[Dict[str, Any]] = []
        for s in steps:
            materialized.append(s)
            if s.get("redundancy"):
                rc = s["redundancy"]
                materialized.append({
                    "name": rc["name"],
                    "dependencies": rc["depends_on"],
                    "prompt": rc["prompt_template"],  # helpful if you do want to view it
                    "links": [],
                    "level": None,
                    "is_critical": False,
                    "tags": ["redundancy"],
                    "redundancy": None,
                })
        steps = materialized

    return {
        "name": "tygent_plan",
        "steps": steps,
        "meta": {
            "redundancy_mode": redundancy_mode,
            **analysis,  # includes critical_path, levels, waves, degrees, etc.
        },
    }

# ----------------- Side-by-side printing (with prompts + scheduling summary) -----------------
LEFT_COL, RIGHT_COL, GAP = 62, 62, 4

def _wrap_lines(text: str, width: int) -> List[str]:
    return textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False) or [""]

def lines_for_spec(spec: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for i, (name, node) in enumerate(spec.items(), start=1):
        prompt = node.get("prompt", "")
        deps = node.get("deps", [])
        links = merge_links(node.get("links", []), extract_links_from_text(prompt))
        lines.append(f"{i}. {name}")
        if prompt:
            lines.append(f"   - Prompt: {prompt}")
        lines.append(f"   - Depends on: {', '.join(deps) if deps else 'None'}")
        if links:
            lines.append("   - Links:")
            for url in links:
                lines.append(f"       • {url}")
        lines.append("")
    return lines

def _scheduling_summary_lines(meta: Dict[str, Any], max_waves_to_show: int = 4) -> List[str]:
    lines: List[str] = []
    cp = meta.get("critical_path", [])
    cp_len = meta.get("critical_path_length", 0)
    nodes_count = len(meta.get("levels", {}))
    edges_count = meta.get("edge_count", 0)
    max_parallel = meta.get("max_parallelism", 1)
    max_parallel_waves = meta.get("max_parallel_waves", [])
    sources = meta.get("sources", [])
    sinks = meta.get("sinks", [])
    top_joins = meta.get("top_joins", [])
    top_bottlenecks = meta.get("top_bottlenecks", [])
    waves = meta.get("waves", [])

    lines.append("=== Scheduling Summary ===")
    lines.append(f"Nodes: {nodes_count}  |  Edges: {edges_count}")
    if cp:
        lines.append("Critical Path: " + " → ".join(cp) + f"  (length {cp_len})")
    lines.append(f"Max parallelism: {max_parallel}  at waves {max_parallel_waves}")
    if sources:
        lines.append("Sources: " + ", ".join(sources))
    if sinks:
        lines.append("Sinks: " + ", ".join(sinks))

    if top_joins:
        display = [f"{n} (in={deg})" for n, deg in top_joins if deg > 0]
        if display:
            lines.append("Top joins (fan-in): " + ", ".join(display))
    if top_bottlenecks:
        display = [f"{n} (out={deg})" for n, deg in top_bottlenecks if deg > 0]
        if display:
            lines.append("Top bottlenecks (fan-out): " + ", ".join(display))

    if waves:
        lines.append("")
        lines.append("First waves (parallel groups):")
        for i, group in enumerate(waves[:max_waves_to_show]):
            lines.append(f"  Wave {i}: " + (", ".join(group) if group else "—"))
    lines.append("")  # trailing blank
    return lines

def lines_for_tygent_aligned(spec: Dict[str, Dict[str, Any]], tygent: Dict[str, Any]) -> List[str]:
    """
    Render Tygent lines in the SAME order as the SPEC keys, with prompts and scheduling fields.
    """
    meta = tygent.get("meta", {})
    mode = meta.get("redundancy_mode", "inline")
    cp = meta.get("critical_path", [])
    critical_set = set(cp)

    # index tygent steps by name
    idx = {s["name"]: s for s in tygent.get("steps", [])}

    lines: List[str] = []
    # Scheduling summary at top
    lines.extend(_scheduling_summary_lines(meta))

    # base steps in spec order
    for i, name in enumerate(spec.keys(), start=1):
        s = idx.get(name, {"dependencies": [], "tags": [], "links": [], "prompt": "", "level": None})
        deps = s.get("dependencies", [])
        links = s.get("links", [])
        tags = s.get("tags", [])
        prompt = s.get("prompt", "")
        level = s.get("level", None)
        badge = " [CRITICAL]" if name in critical_set else ""
        lines.append(f"{i}. {name}{badge}")
        if prompt:
            lines.append(f"   - Prompt: {prompt}")
        lines.append(f"   - Depends on: {', '.join(deps) if deps else 'None'}")
        if level is not None:
            lines.append(f"   - Level (earliest wave): {level}")
        if links:
            lines.append("   - Links:")
            for url in links:
                lines.append(f"       • {url}")
        if tags:
            lines.append(f"   - Tags: {', '.join(tags)}")

        # inline redundancy view
        red = s.get("redundancy")
        if mode == "inline" and red:
            lines.append("   - Redundancy check (inline):")
            lines.append(f"       • name: {red['name']}")
            lines.append(f"       • depends_on: {', '.join(red['depends_on'])}")
        lines.append("")

        # if redundancy materialized as steps, show the child node immediately after
        if mode == "steps":
            rc_name = f"{name}__redundancy"
            rc = idx.get(rc_name)
            if rc:
                lines.append(f"    ↳ {rc_name}")
                lines.append(f"       - Depends on: {', '.join(rc.get('dependencies', [])) or 'None'}")
                lines.append(f"       - Tags: {', '.join(rc.get('tags', [])) or 'None'}")
                lines.append("")

    return lines

def print_side_by_side(title: str, left: List[str], right: List[str]) -> None:
    left_wrapped = [ln for line in left for ln in _wrap_lines(line, LEFT_COL)]
    right_wrapped = [rn for line in right for rn in _wrap_lines(line, RIGHT_COL)]
    print("=" * (LEFT_COL + RIGHT_COL + GAP))
    print(title.center(LEFT_COL + RIGHT_COL + GAP))
    print("=" * (LEFT_COL + RIGHT_COL + GAP))
    maxlen = max(len(left_wrapped), len(right_wrapped))
    for i in range(maxlen):
        l = left_wrapped[i] if i < len(left_wrapped) else ""
        r = right_wrapped[i] if i < len(right_wrapped) else ""
        print(f"{l:<{LEFT_COL}}{' ' * GAP}{r:<{RIGHT_COL}}")
    print()

# ----------------- 100 tasks -----------------
TASKS = [
    "Market intelligence for EV charging expansion",
    "Competitor teardown for a fintech B2B payments API",
    "Customer segmentation for a D2C wellness brand",
    "Regulatory scan for healthcare data sharing",
    "Trend analysis for on-device AI assistants",
    "Expert interview synthesis for enterprise MLOps",
    "Market data aggregation for SMB payroll tools",
    "Risk assessment for LLM rollouts in contact centers",
    "Validation and triangulation for retail demand forecasts",
    "Synthesis & exec summary for an AI note-taking app",
    "Churn reduction playbook for a SaaS analytics product",
    "Pricing strategy for a developer tooling startup",
    "Geo expansion analysis for a food delivery service",
    "Vendor selection for a vector database migration",
    "Data quality audit for recommendation pipelines",
    "Security posture review for multi-tenant SaaS",
    "Cloud cost optimization for training pipelines",
    "Partner ecosystem mapping for a BI platform",
    "Feature adoption analysis for a CRM plugin",
    "Go-to-market plan for a cybersecurity agent",
    "Sales enablement content plan for LLM platform",
    "RFP response strategy for a gov cloud contract",
    "M&A thesis for data observability companies",
    "Open-source community growth plan for SDK",
    "Privacy impact assessment for adtech signals",
    "Localization roadmap for a productivity suite",
    "App store optimization for a learning app",
    "Retention cohort deep-dive for a finance app",
    "Activation funnel mapping for a dev platform",
    "Enterprise readiness checklist for GenAI app",
    "Incident response runbook for AI outages",
    "Benchmark design for retrieval augmentation",
    "Human-in-the-loop QA for AI summarization",
    "Risk registry for AI content generation",
    "SLAs & SLOs definition for inference APIs",
    "Data roadmap for personalization in ecommerce",
    "AB test plan for onboarding experience",
    "Compliance checklist for SOC 2 Type II",
    "GDPR/CCPA audit for analytics pipeline",
    "Latency reduction plan for agent workflows",
    "Observability plan for multi-agent systems",
    "Vendor comparison for GPU cloud providers",
    "Synthetic data strategy for edge cases",
    "Bias and fairness assessment for hiring model",
    "Content moderation policy for UGC platform",
    "Data retention policy for AI artifacts",
    "Prompt library governance for enterprise",
    "CI/CD pipeline for LLM prompt changes",
    "Guardrail strategy for finance assistants",
    "PII redaction pipeline for support tickets",
    "Zero-trust design for agent tool access",
    "Knowledge graph build for customer 360",
    "Cold-start strategy for marketplace search",
    "Telemetry schema for agent performance",
    "Causal inference plan for pricing changes",
    "Playbook for migrating to function-calling",
    "Eval harness design for reasoning tasks",
    "Shadow deployment plan for new models",
    "On-device vs server inference tradeoff study",
    "Long-context retrieval cost–benefit analysis",
    "Session memory policy for assistants",
    "Red team plan for prompt injection",
    "Access control model for tools & actions",
    "API quota policy for partner ecosystem",
    "LLM vendor diversification strategy",
    "Fallback routing for degraded models",
    "Cache strategy for frequent prompts",
    "Prefetching design for multi-step plans",
    "Throughput scaling plan for batch jobs",
    "Sustainability analysis for energy usage",
    "TAM/SAM/SOM for workflow automation",
    "Design partner program for pilots",
    "RROI calculator for agent acceleration",
    "Content ops workflow for knowledge bases",
    "Change management plan for AI rollout",
    "User research plan for pro users",
    "Instrumentation plan for funnel analytics",
    "Security review of tool invocation paths",
    "SLI definitions for grounding quality",
    "Blueprint for MCP endpoint exposure",
    "SRE playbook for vector DB incidents",
    "Hybrid search (BM25+embed) evaluation",
    "Data contract policy for event streams",
    "Privacy sandbox impact on attribution",
    "Feature flag strategy for agent tools",
    "Autoscaling policy for inference gateways",
    "Fail-open vs fail-closed decision log",
    "Postmortem template for AI regressions",
    "Runtimes comparison for fine-tuning",
    "Key results tree for agent KPIs",
    "User trust framework for AI explainability",
    "Procurement checklist for LLM services",
    "Legal review workflow for generated content",
    "Observability SLOs for tool latency",
    "Access review cadence for tool scopes",
    "Incident drills schedule for AI systems",
    "Data residency plan for multi-region ops",
    "Knowledge distillation plan for cost savings",
    "Experimentation guardrails for revenue impact",
    "Self-serve analytics enablement plan",
    "Security assessment for browser automation",
    "Model eval battery for reasoning depth",
    "Red team hunt for jailbreak surface",
]

# ----------------- CLI main -----------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADK → Tygent (multi-task) 1:1 transform with scheduling analysis, side-by-side"
    )
    parser.add_argument("--basic", action="store_true", help="Use the basic plan generation prompt")
    parser.add_argument("--max", type=int, default=100, help="Max number of tasks to process (<=100)")
    parser.add_argument("--redundancy", choices=["inline", "steps", "off"], default="inline",
                        help="How to represent redundancy checks in Tygent (default: inline)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    runner = await _create_runner()
    total = max(1, min(args.max, len(TASKS)))

    for idx, task in enumerate(TASKS[:total], start=1):
        # Header
        width = LEFT_COL + RIGHT_COL + GAP
        print("\n" + "#" * width)
        header = f"[{idx}/{total}] {task}  (redundancy={args.redundancy}, basic={args.basic})"
        print(header.center(width))
        print("#" * width + "\n")

        # 1) Build SPEC
        try:
            spec = await build_plan_for_task(runner, task, basic=args.basic)
        except Exception as e:
            print(f"Failed to build plan for task: {task}\nError: {e}\n")
            continue

        # 2) Tygent transform (1:1, includes prompts + analysis)
        try:
            tygent = build_tygent_transform_with_prompts_and_analysis(spec, redundancy_mode=args.redundancy)
        except Exception as e:
            print(f"Failed to transform plan for task: {task}\nError: {e}\n")
            continue

        # Assert 1:1 on base steps when redundancy is inline/off
        if args.redundancy != "steps":
            spec_set = set(spec.keys())
            ty_set = set(s["name"] for s in tygent["steps"])
            if spec_set != ty_set:
                print("WARNING: Base step sets differ between SPEC and Tygent.")
                print("SPEC only:", sorted(spec_set - ty_set))
                print("Tygent only:", sorted(ty_set - spec_set))

        # 3) Side-by-side print (prompts on both sides + scheduling summary on right)
        left_lines = lines_for_spec(spec)
        right_lines = lines_for_tygent_aligned(spec, tygent)
        title = "SPEC (left) vs Tygent (right)"
        print_side_by_side(title, left_lines, right_lines)

if __name__ == "__main__":
    asyncio.run(main())