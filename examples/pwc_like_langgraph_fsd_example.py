"""
LangGraph Functional Spec Writer (FSD) Example
----------------------------------------------
Demonstrates how Tygent executes a LangGraph-style workflow with cyclic subgraphs,
interactive checkpoints, persistent session state, parallel artifact generation, and
LLM-backed reviewer loops.

The demo runs the same plan twice—first with a hand-written sequential loop (no Tygent),
then with the Tygent scheduler—to compare latency, iterations, token usage, and output
quality. Because the plan includes parallelizable steps (three independent reviewers and
two artifact generators), the scheduled run can overlap network-bound LLM calls. A third
pass reuses the persisted session cache to show faster restarts. Reviewer feedback is
automated for the demo via role-specific prompts that mimic human reviewers.

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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

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
TOKEN_USAGE: Dict[str, Dict[str, int]] = {}
ITERATION_COUNTERS: Dict[str, int] = {}
NODE_TIMINGS: Dict[str, Dict[str, List[float]]] = {}


def _init_context_metrics(context: str) -> None:
    TOKEN_USAGE[context] = {"prompt": 0, "completion": 0, "total": 0}
    ITERATION_COUNTERS[context] = 0
    NODE_TIMINGS[context] = defaultdict(list)


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
        bucket = TOKEN_USAGE.setdefault(
            context_key, {"prompt": 0, "completion": 0, "total": 0}
        )
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


def _record_duration(node_name: str, start: float) -> None:
    ctx = CURRENT_CONTEXT or "global"
    timings = NODE_TIMINGS.setdefault(ctx, defaultdict(list))
    timings[node_name].append((perf_counter() - start) * 1000.0)  # store ms


async def ingest_requirements(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the raw product brief into structured requirements."""

    start = perf_counter()
    brief = inputs.get("brief", "").strip()
    if not brief:
        raise ValueError("A product brief is required to start the workflow.")

    lines = [line.strip() for line in brief.splitlines() if line.strip()]
    title = lines[0].lstrip("# ").strip() if lines else "Untitled Product"
    goals = [line for line in lines if line.lower().startswith("goal")]

    await asyncio.sleep(0.05)
    result = {
        "title": title,
        "summary": lines[1:] if len(lines) > 1 else [],
        "goals": goals or ["Goal: improve decision-making latency by 35%"],
        "owner": inputs.get("requester", "product.design@acme.dev"),
    }
    _record_duration("ingest_requirements", start)
    return result


async def flatten_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize inbound payloads into a consistent dictionary."""

    start = perf_counter()
    payload = {
        "correlation_id": inputs.get("correlation_id", "corr-001"),
        "requested_sections": inputs.get("requested_sections")
        or [
            "Executive Summary",
            "Business Goals",
            "Technical Scope",
            "Security",
        ],
    }
    await asyncio.sleep(0.01)
    result = {"payload": payload}
    _record_duration("flatten_payload", start)
    return result


async def get_mapping_values(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Derive mapping values used downstream for data enrichment."""

    start = perf_counter()
    payload = inputs.get("flatten_payload", {}).get("payload", {})
    mapping = {
        "account_id": inputs.get("account_id", "acme-enterprise"),
        "requested_sections": payload.get("requested_sections", []),
    }
    await asyncio.sleep(0.02)
    result = {"mapping": mapping}
    _record_duration("get_mapping_values", start)
    return result


async def get_adhoc_ids(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch adhoc identifiers (tickets, docs) to include in the spec."""

    start = perf_counter()
    mapping = inputs.get("get_mapping_values", {}).get("mapping", {})
    requested = mapping.get("requested_sections", [])
    ids = [f"TKT-{1000 + idx}" for idx, _ in enumerate(requested, start=1)]
    await asyncio.sleep(0.02)
    result = {"adhoc_ids": ids}
    _record_duration("get_adhoc_ids", start)
    return result


async def kb_extrapolation_base(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate knowledge base nuggets from existing sources."""

    start = perf_counter()
    mapping = inputs.get("get_mapping_values", {}).get("mapping", {})
    ids = inputs.get("get_adhoc_ids", {}).get("adhoc_ids", [])

    prompt = textwrap.dedent(
        """
        You are an enterprise knowledge curator. Produce JSON with:
          - kb_summary: bullet list of facts relevant to the document request
          - source_ids: echo the source identifiers you used
        """
    ).strip()
    user = json.dumps(
        {
            "requested_sections": mapping.get("requested_sections", []),
            "source_ids": ids,
        },
        indent=2,
    )
    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    result = {
        "kb_summary": response.get("kb_summary", []),
        "source_ids": response.get("source_ids", ids),
    }
    _record_duration("kb_extrapolation_base", start)
    return result


async def intent_classification(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the intent and priority of the FSD request."""

    start = perf_counter()
    kb = inputs.get("kb_extrapolation_base", {})
    prompt = textwrap.dedent(
        """
        You are classifying the intent of an FSD request. Return JSON with:
          - intent: one of ["net-new", "enhancement", "compliance"]
          - priority: one of ["low", "medium", "high"]
          - justification: short text
        """
    ).strip()
    user = json.dumps(kb, indent=2)
    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    result = {
        "intent": response.get("intent", "net-new"),
        "priority": response.get("priority", "medium"),
        "justification": response.get("justification", ""),
    }
    _record_duration("intent_classification", start)
    return result


SECTION_TITLES = [
    "Executive Summary",
    "Business Goals",
    "Non-Functional Requirements",
    "System Overview",
    "User Journeys",
    "Data Flow",
    "Integration Points",
    "Security & Compliance",
    "Privacy Considerations",
    "Scalability Plan",
    "Failover Strategy",
    "Monitoring & Observability",
    "Automation & Tooling",
    "Release Plan",
    "Adoption Strategy",
    "Change Management",
    "Support Model",
    "KPIs & Metrics",
    "Risks & Mitigations",
    "Open Questions",
    "Appendix",
]


async def _generate_section(
    section: str, context: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    prompt = textwrap.dedent(
        f"""
        You are composing the "{section}" section of a functional specification.
        Return JSON with:
          - heading: string
          - content: markdown paragraph(s)
          - followups: list of questions still outstanding
        """
    ).strip()
    user = json.dumps(context, indent=2)
    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    return section, {
        "heading": response.get("heading", section),
        "content": response.get("content", ""),
        "followups": response.get("followups", []),
    }


def _make_section_node(node_name: str, section_title: str):
    async def _node(inputs: Dict[str, Any]) -> Dict[str, Any]:
        start = perf_counter()
        intent = inputs.get("intent_classification", {})
        kb = inputs.get("kb_extrapolation_base", {})
        fallback = inputs.get("fallback_function", {})
        context = {
            "intent": intent,
            "kb_summary": kb.get("kb_summary", []),
            "notes": fallback.get("notes", []),
            "revision": int(fallback.get("next_revision", 1)),
        }
        _, payload = await _generate_section(section_title, context)
        payload["section"] = section_title
        _record_duration(node_name, start)
        return payload

    return _node


SECTION_NODE_FUNCTIONS: Dict[
    str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
] = {
    f"section_chunk_{idx:02d}": _make_section_node(f"section_chunk_{idx:02d}", title)
    for idx, title in enumerate(SECTION_TITLES)
}


async def context_retrieval(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge per-section context nodes into a consolidated structure."""

    start = perf_counter()
    sections: Dict[str, Dict[str, Any]] = {}
    open_items = set()
    for node_name, title in zip(SECTION_NODE_FUNCTIONS.keys(), SECTION_TITLES):
        payload = inputs.get(node_name, {})
        section_title = payload.get("section", title)
        sections[section_title] = {
            "heading": payload.get("heading", section_title),
            "content": payload.get("content", ""),
            "followups": payload.get("followups", []),
        }
        open_items.update(payload.get("followups", []))

    fallback = inputs.get("fallback_function", {})
    open_items.update(fallback.get("notes", []))
    result = {
        "sections": sections,
        "open_items": sorted(open_items),
        "revision": int(fallback.get("next_revision", 1)),
    }
    _record_duration("context_retrieval", start)
    return result


async def prompt_updation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the composite prompt for reviewers and subsequent drafting."""

    start = perf_counter()
    context = inputs.get("context_retrieval", {})
    prompt = textwrap.dedent(
        """
        You are updating the master prompt for the drafting agent. Respond with JSON:
          - prompt: string with instructions for the drafter
          - checklist: list of items to double-check
        """
    ).strip()
    user = json.dumps(context, indent=2)
    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    result = {
        "prompt": response.get("prompt", ""),
        "checklist": response.get("checklist", []),
        "open_items": context.get("open_items", []),
        "revision": context.get("revision", 1),
    }
    _record_duration("prompt_updation", start)
    return result


async def human_feedback(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a human reviewer providing approval or further comments."""

    start = perf_counter()
    prompt_state = inputs.get("prompt_updation", {})
    revision = int(prompt_state.get("revision", 1))
    outstanding = prompt_state.get("open_items", []) + prompt_state.get("checklist", [])

    system = textwrap.dedent(
        """
        You are acting as the human reviewer. Respond with JSON:
          - approved: boolean
          - comments: list of actionable feedback
          - summary: short review note
        Default to not approving on the first revision when outstanding items exist.
        Approve once outstanding items are cleared or if revision >= 3.
        """
    ).strip()
    user = json.dumps(
        {
            "revision": revision,
            "outstanding": outstanding,
            "prompt": prompt_state.get("prompt", ""),
        },
        indent=2,
    )
    response = await call_llm_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    approved = bool(response.get("approved"))
    comments = response.get("comments", [])
    _log(f"human_feedback: approved={approved} comments={len(comments)}")
    result = {
        "approved": approved,
        "comments": comments,
        "summary": response.get("summary", ""),
        "revision": revision,
    }
    _record_duration("human_feedback", start)
    return result


async def fallback_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback logic that synthesises reviewer feedback for the next pass."""

    start = perf_counter()
    consolidated = inputs.get("consolidate_feedback", {})
    revision = int(consolidated.get("revision_completed", 1))

    if consolidated.get("approved"):
        result = {
            "approved": True,
            "notes": [],
            "next_revision": revision,
        }
    else:
        combined = consolidated.get("notes", [])
        notes = sorted(set(combined))
        result = {
            "approved": False,
            "notes": notes,
            "next_revision": revision + 1,
        }
    _record_duration("fallback_function", start)
    return result


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

    start = perf_counter()
    requirements = inputs.get("ingest_requirements", {})
    prompt_state = inputs.get("prompt_updation", {})
    feedback = inputs.get("consolidate_feedback") or {}

    next_revision = int(feedback.get("next_revision", 1))
    revision = max(next_revision, 1)
    outstanding = (
        feedback.get("notes")
        or prompt_state.get("open_items")
        or [
            "Add non-functional requirements covering latency budgets.",
            "Document integration touchpoints with CRM systems.",
        ]
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
        prompt: {json.dumps(prompt_state, indent=2)}
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
    result = {
        "revision": revision,
        "status": status,
        "content": draft_markdown,
        "revision_summary": response.get("revision_summary", ""),
        "open_items": outstanding_items,
    }
    _record_duration("draft_spec", start)
    return result


async def generate_architecture(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Produce an architecture outline for the current draft."""

    start = perf_counter()
    draft = inputs.get("draft_spec", {})
    prompt = textwrap.dedent(
        """
        You are the architecture lead on a product team. Given a functional specification
        draft, summarise the system architecture. Respond with JSON containing:
          - architecture_summary: paragraph describing the target architecture
          - components: list of major components with owner/team
          - integration_points: list of key integrations (name + purpose)
        """
    ).strip()
    user = json.dumps(
        {
            "draft_markdown": draft.get("content", ""),
            "revision": draft.get("revision"),
        },
        indent=2,
    )

    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    components = response.get("components", [])
    _log(f"generate_architecture: components={len(components)}")
    result = {
        "architecture_summary": response.get("architecture_summary", ""),
        "components": components,
        "integration_points": response.get("integration_points", []),
    }
    _record_duration("generate_architecture", start)
    return result


async def generate_risk_register(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a lightweight risk register with mitigations."""

    start = perf_counter()
    draft = inputs.get("draft_spec", {})
    prompt = textwrap.dedent(
        """
        You are the security and compliance lead reviewing a functional specification.
        Return a JSON object with:
          - risks: list of {area, description, severity, mitigation}
          - overall_assessment: short narrative summary
        """
    ).strip()
    user = json.dumps(
        {
            "draft_markdown": draft.get("content", ""),
            "revision": draft.get("revision"),
            "open_items": draft.get("open_items", []),
        },
        indent=2,
    )
    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    risks = response.get("risks", [])
    _log(f"generate_risk_register: risks={len(risks)}")
    result = {
        "risks": risks,
        "overall_assessment": response.get("overall_assessment", ""),
    }
    _record_duration("generate_risk_register", start)
    return result


async def _review_role(
    role: str,
    mandate: str,
    draft: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared reviewer helper that returns approval decisions."""

    start = perf_counter()
    prompt = textwrap.dedent(
        f"""
        You are the {role} reviewer on a product specification. Respond with JSON:
          - approved: boolean
          - comments: list of actionable feedback strings
          - rationale: short explanation
        Your mandate: {mandate}
        Approve only when the draft addresses your concerns and no blocking issues remain.
        """
    ).strip()
    user = json.dumps(
        {
            "draft_markdown": draft.get("content", ""),
            "revision": draft.get("revision"),
            "open_items": draft.get("open_items", []),
        },
        indent=2,
    )

    response = await call_llm_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )
    approved = bool(response.get("approved"))
    comments = response.get("comments") or []
    _log(f"{role} review: approved={approved} comments={len(comments)}")
    result = {
        "approved": approved,
        "comments": comments,
        "rationale": response.get("rationale", ""),
        "role": role,
    }
    _record_duration(f"review_{role.lower()}", start)
    return result


async def review_product(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return await _review_role(
        role="Product",
        mandate="Ensure the draft covers user problems, scope, and success metrics.",
        draft=inputs.get("draft_spec", {}),
    )


async def review_engineering(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return await _review_role(
        role="Engineering",
        mandate="Verify feasibility, estimate complexity, and identify technical risks.",
        draft=inputs.get("draft_spec", {}),
    )


async def review_security(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return await _review_role(
        role="Security",
        mandate="Evaluate compliance, data protection, and abuse prevention requirements.",
        draft=inputs.get("draft_spec", {}),
    )


async def consolidate_feedback(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Combine multi-role reviewer feedback and decide the next revision."""

    start = perf_counter()
    draft = inputs.get("draft_spec", {})
    revision = int(draft.get("revision", 1))
    reviews = {
        "product": inputs.get("review_product", {}),
        "engineering": inputs.get("review_engineering", {}),
        "security": inputs.get("review_security", {}),
    }
    human = inputs.get("human_feedback", {})

    notes: List[str] = []
    approvals: List[bool] = []
    review_matrix: Dict[str, Any] = {}
    for role, payload in reviews.items():
        approvals.append(bool(payload.get("approved")))
        notes.extend(payload.get("comments", []))
        review_matrix[role] = {
            "approved": bool(payload.get("approved")),
            "comments": payload.get("comments", []),
            "rationale": payload.get("rationale", ""),
        }

    review_matrix["human"] = {
        "approved": bool(human.get("approved")),
        "comments": human.get("comments", []),
        "rationale": human.get("summary", ""),
    }
    approvals.append(bool(human.get("approved")))
    notes.extend(human.get("comments", []))

    outstanding = draft.get("open_items", [])
    approved = all(approvals) and not outstanding and not notes
    next_revision = revision if approved else revision + 1

    _log(
        f"consolidate_feedback: approved={approved} notes={len(notes)} next_revision={next_revision}"
    )
    result = {
        "approved": approved,
        "notes": notes or outstanding,
        "next_revision": next_revision,
        "review_matrix": review_matrix,
        "revision_completed": revision,
    }
    _record_duration("consolidate_feedback", start)
    return result


async def write_to_docx(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the final document content."""

    start = perf_counter()
    draft = inputs.get("draft_spec", {})
    sections = inputs.get("context_retrieval", {}).get("sections", {})
    architecture = inputs.get("generate_architecture", {})
    risks = inputs.get("generate_risk_register", {})
    feedback = inputs.get("consolidate_feedback", {})

    lines = [draft.get("content", "")]
    for section_info in sections.values():
        lines.append(f"## {section_info.get('heading')}\n{section_info.get('content')}")
    lines.append("## Architecture Overview")
    lines.append(architecture.get("architecture_summary", ""))
    lines.append("## Risk Register")
    for risk in risks.get("risks", []):
        lines.append(
            f"- ({risk.get('severity','medium')}) {risk.get('area','General')}: "
            f"{risk.get('description','')} — Mitigation: {risk.get('mitigation','')}"
        )
    await asyncio.sleep(0.05)
    result = {
        "document": "\n\n".join(lines),
        "approved": feedback.get("approved", False),
        "revision": feedback.get("revision_completed", draft.get("revision")),
    }
    _record_duration("write_to_docx", start)
    return result


async def master_variable_dump(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Collect final runtime state for auditing."""

    start = perf_counter()
    doc = inputs.get("publish_spec", {})
    summary = {
        "revision": doc.get("revision"),
        "approved": doc.get("status") == "published",
        "length": len(doc.get("document", "")),
    }
    await asyncio.sleep(0.01)
    _record_duration("master_variable_dump", start)
    return summary


async def store_spec(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the assembled specification and return metadata."""

    start = perf_counter()
    document = inputs.get("write_to_docx", {})
    architecture = inputs.get("generate_architecture", {})
    risk_register = inputs.get("generate_risk_register", {})
    feedback = inputs.get("consolidate_feedback", {})
    status = "published" if feedback.get("approved") else "pending_approval"

    await asyncio.sleep(0.02)
    _log(f"publish_spec: status={status} revision={document.get('revision', 'n/a')}")
    result = {
        "status": status,
        "revision": document.get("revision"),
        "excerpt": (document.get("document", "") or "")[:200] + "...",
        "notes": feedback.get("notes", []),
        "architecture": architecture,
        "risk_register": risk_register,
        "document": document.get("document", ""),
    }
    _record_duration("publish_spec", start)
    return result


_BASE_PLAN_STEPS: List[Dict[str, Any]] = [
    {
        "name": "flatten_payload",
        "func": flatten_payload,
        "session": {"persist": True},
    },
    {
        "name": "get_mapping_values",
        "func": get_mapping_values,
        "dependencies": ["flatten_payload"],
        "session": {"persist": True},
    },
    {
        "name": "get_adhoc_ids",
        "func": get_adhoc_ids,
        "dependencies": ["get_mapping_values"],
        "session": {"persist": True},
    },
    {
        "name": "ingest_requirements",
        "func": ingest_requirements,
        "dependencies": ["flatten_payload"],
        "session": {"persist": True},
    },
    {
        "name": "kb_extrapolation_base",
        "func": kb_extrapolation_base,
        "dependencies": ["get_mapping_values", "get_adhoc_ids"],
        "session": {"persist": True},
    },
    {
        "name": "intent_classification",
        "func": intent_classification,
        "dependencies": ["kb_extrapolation_base"],
        "session": {"persist": True},
    },
    {
        "name": "fallback_function",
        "func": fallback_function,
        "dependencies": ["consolidate_feedback"],
        "loop": {"group": "fsd_iteration"},
        "interactive": True,
    },
]

_SECTION_PLAN_STEPS: List[Dict[str, Any]] = [
    {
        "name": node_name,
        "func": func,
        "dependencies": [
            "intent_classification",
            "kb_extrapolation_base",
            "fallback_function",
        ],
        "loop": {"group": "fsd_iteration"},
        "session": {"persist": True},
    }
    for node_name, func in SECTION_NODE_FUNCTIONS.items()
]

_CONTEXT_DEPENDENCIES = list(SECTION_NODE_FUNCTIONS.keys()) + ["fallback_function"]

_REMAINING_PLAN_STEPS: List[Dict[str, Any]] = [
    {
        "name": "context_retrieval",
        "func": context_retrieval,
        "dependencies": _CONTEXT_DEPENDENCIES,
        "loop": {"group": "fsd_iteration"},
        "session": {"persist": True},
    },
    {
        "name": "prompt_updation",
        "func": prompt_updation,
        "dependencies": ["context_retrieval"],
        "loop": {"group": "fsd_iteration"},
        "interactive": True,
    },
    {
        "name": "human_feedback",
        "func": human_feedback,
        "dependencies": ["prompt_updation", "draft_spec"],
        "loop": {"group": "fsd_iteration"},
        "interactive": True,
    },
    {
        "name": "draft_spec",
        "func": draft_spec,
        "dependencies": [
            "ingest_requirements",
            "prompt_updation",
        ],
        "loop": {
            "group": "fsd_iteration",
            "termination": {"type": "fixed_point", "max_iterations": 2},
        },
        "interactive": True,
        "session": {"persist": True},
    },
    {
        "name": "generate_architecture",
        "func": generate_architecture,
        "dependencies": ["draft_spec"],
        "loop": {"group": "fsd_iteration"},
        "session": {"persist": True},
    },
    {
        "name": "generate_risk_register",
        "func": generate_risk_register,
        "dependencies": ["draft_spec"],
        "loop": {"group": "fsd_iteration"},
        "session": {"persist": True},
    },
    {
        "name": "review_product",
        "func": review_product,
        "dependencies": ["draft_spec"],
        "loop": {"group": "fsd_iteration"},
        "interactive": True,
    },
    {
        "name": "review_engineering",
        "func": review_engineering,
        "dependencies": ["draft_spec"],
        "loop": {"group": "fsd_iteration"},
        "interactive": True,
    },
    {
        "name": "review_security",
        "func": review_security,
        "dependencies": ["draft_spec"],
        "loop": {"group": "fsd_iteration"},
        "interactive": True,
    },
    {
        "name": "consolidate_feedback",
        "func": consolidate_feedback,
        "dependencies": [
            "draft_spec",
            "review_product",
            "review_engineering",
            "review_security",
            "human_feedback",
        ],
        "loop": {"group": "fsd_iteration"},
    },
    {
        "name": "write_to_docx",
        "func": write_to_docx,
        "dependencies": [
            "draft_spec",
            "context_retrieval",
            "generate_architecture",
            "generate_risk_register",
            "consolidate_feedback",
        ],
        "critical": True,
    },
    {
        "name": "publish_spec",
        "func": store_spec,
        "dependencies": [
            "write_to_docx",
            "generate_architecture",
            "generate_risk_register",
            "consolidate_feedback",
        ],
        "critical": True,
    },
    {
        "name": "master_variable_dump",
        "func": master_variable_dump,
        "dependencies": ["publish_spec"],
        "critical": True,
    },
]

PLAN_DEFINITION: Dict[str, Any] = {
    "name": "langgraph_fsd",
    "steps": _BASE_PLAN_STEPS + _SECTION_PLAN_STEPS + _REMAINING_PLAN_STEPS,
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
    _init_context_metrics(context)
    base_inputs = {"brief": brief, "requester": requester}
    requirements = await ingest_requirements(base_inputs)
    state: Dict[str, Any] = {"ingest_requirements": requirements}
    feedback_state: Dict[str, Any] = {
        "approved": False,
        "notes": [
            "Add data-retention policy details.",
            "Clarify success metrics for adoption.",
        ],
        "next_revision": 1,
        "revision_completed": 0,
    }
    state["fallback_function"] = feedback_state
    state["consolidate_feedback"] = feedback_state
    state["write_to_docx"] = {"document": "", "approved": False, "revision": 0}
    flatten = await flatten_payload(base_inputs)
    mapping_values = await get_mapping_values(
        {**base_inputs, "flatten_payload": flatten}
    )
    adhoc_ids = await get_adhoc_ids(
        {**base_inputs, "get_mapping_values": mapping_values}
    )
    kb = await kb_extrapolation_base(
        {
            **base_inputs,
            "get_mapping_values": mapping_values,
            "get_adhoc_ids": adhoc_ids,
        }
    )
    intent = await intent_classification({"kb_extrapolation_base": kb})
    state.update(
        {
            "flatten_payload": flatten,
            "get_mapping_values": mapping_values,
            "get_adhoc_ids": adhoc_ids,
            "kb_extrapolation_base": kb,
            "intent_classification": intent,
        }
    )
    history: List[Dict[str, Any]] = []
    start = perf_counter()

    while len(history) < max_iterations:
        iteration_inputs = {
            **base_inputs,
            "flatten_payload": state["flatten_payload"],
            "get_mapping_values": state["get_mapping_values"],
            "get_adhoc_ids": state["get_adhoc_ids"],
            "kb_extrapolation_base": state["kb_extrapolation_base"],
            "intent_classification": state["intent_classification"],
            "ingest_requirements": state["ingest_requirements"],
            "consolidate_feedback": feedback_state,
        }
        section_inputs = {}
        for node_name, node_fn in SECTION_NODE_FUNCTIONS.items():
            section_inputs[node_name] = await node_fn(
                {
                    "intent_classification": state["intent_classification"],
                    "kb_extrapolation_base": state["kb_extrapolation_base"],
                    "fallback_function": feedback_state,
                }
            )
        ctx = await context_retrieval(
            {**section_inputs, "fallback_function": feedback_state}
        )
        prompt_state = await prompt_updation({"context_retrieval": ctx})
        draft = await draft_spec({**iteration_inputs, "prompt_updation": prompt_state})
        architecture = await generate_architecture({"draft_spec": draft})
        risk_register = await generate_risk_register({"draft_spec": draft})
        product_review = await review_product({"draft_spec": draft})
        engineering_review = await review_engineering({"draft_spec": draft})
        security_review = await review_security({"draft_spec": draft})
        human = await human_feedback(
            {"prompt_updation": prompt_state, "draft_spec": draft}
        )
        consolidated = await consolidate_feedback(
            {
                "draft_spec": draft,
                "review_product": product_review,
                "review_engineering": engineering_review,
                "review_security": security_review,
                "human_feedback": human,
            }
        )
        doc_output = await write_to_docx(
            {
                "draft_spec": draft,
                "context_retrieval": ctx,
                "generate_architecture": architecture,
                "generate_risk_register": risk_register,
                "consolidate_feedback": consolidated,
            }
        )
        feedback_state = await fallback_function({"consolidate_feedback": consolidated})
        state.update(
            {
                "context_retrieval": ctx,
                "prompt_updation": prompt_state,
                "human_feedback": human,
                "consolidate_feedback": consolidated,
                "fallback_function": feedback_state,
                "draft_spec": draft,
                "write_to_docx": doc_output,
                "generate_architecture": architecture,
                "generate_risk_register": risk_register,
                "review_product": product_review,
                "review_engineering": engineering_review,
                "review_security": security_review,
            }
        )
        history.append(
            {
                "context_retrieval": ctx,
                "prompt_updation": prompt_state,
                "human_feedback": human,
                "consolidate_feedback": consolidated,
                "fallback_function": feedback_state,
                "draft": draft,
                "write_to_docx": doc_output,
                "architecture": architecture,
                "risk_register": risk_register,
                "product_review": product_review,
                "engineering_review": engineering_review,
                "security_review": security_review,
            }
        )
        if feedback_state.get("approved"):
            break

    publish = await store_spec(
        {
            "draft_spec": state["draft_spec"],
            "context_retrieval": state["context_retrieval"],
            "generate_architecture": state["generate_architecture"],
            "generate_risk_register": state["generate_risk_register"],
            "write_to_docx": state["write_to_docx"],
            "consolidate_feedback": state["consolidate_feedback"],
        }
    )
    dump = await master_variable_dump({"publish_spec": publish})
    elapsed = perf_counter() - start
    set_context("")
    return {
        "elapsed": elapsed,
        "publish": publish,
        "history": history,
        "session": state,
        "tokens": TOKEN_USAGE.get(context, {"prompt": 0, "completion": 0, "total": 0}),
        "iterations": ITERATION_COUNTERS.get(context, 0),
        "dump": dump,
        "timings": {k: list(v) for k, v in NODE_TIMINGS.get(context, {}).items()},
    }


async def run_tygent_orchestrated(
    brief: str, requester: str, context: str
) -> Dict[str, Any]:
    """Execute the workflow through Tygent's scheduler."""

    dag, critical = parse_plan(PLAN_DEFINITION)
    session = InMemorySessionStore()
    scheduler = Scheduler(dag, session_store=session)
    scheduler.priority_nodes = critical
    scheduler.configure(max_parallel_nodes=10, max_execution_time=300000)
    session.set_node_state(
        "consolidate_feedback",
        {
            "approved": False,
            "notes": [
                "Add data-retention policy details.",
                "Clarify success metrics for adoption.",
            ],
            "next_revision": 1,
            "revision_completed": 0,
        },
    )

    set_context(context)
    _init_context_metrics(context)
    start = perf_counter()
    results = await scheduler.execute({"brief": brief, "requester": requester})
    elapsed = perf_counter() - start
    set_context("")

    publish = results["results"].get("publish_spec", {})
    return {
        "elapsed": elapsed,
        "publish": publish,
        "results": results,
        "session": session,
        "scheduler": scheduler,
        "tokens": TOKEN_USAGE.get(context, {"prompt": 0, "completion": 0, "total": 0}),
        "iterations": ITERATION_COUNTERS.get(context, 0),
        "timings": {k: list(v) for k, v in NODE_TIMINGS.get(context, {}).items()},
    }


async def rerun_with_persisted_state(
    scheduler: Scheduler, inputs: Dict[str, Any], context: str
) -> Tuple[float, Dict[str, Any], Dict[str, int], int]:
    """Re-execute using the same scheduler to highlight session persistence."""

    set_context(context)
    _init_context_metrics(context)
    start = perf_counter()
    results = await scheduler.execute(inputs)
    elapsed = perf_counter() - start
    set_context("")
    return (
        elapsed,
        results["results"].get("publish_spec", {}),
        TOKEN_USAGE.get(context, {"prompt": 0, "completion": 0, "total": 0}),
        ITERATION_COUNTERS.get(context, 0),
        {k: list(v) for k, v in NODE_TIMINGS.get(context, {}).items()},
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
    base_status = baseline_result["publish"].get("status", "unknown")
    print(
        f"Baseline latency: {baseline_result['elapsed']*1000:.1f} ms, "
        f"iterations: {baseline_result['iterations']}, "
        f"status: {base_status}"
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
    (
        repeat_latency,
        repeat_publish,
        rerun_tokens,
        rerun_iterations,
        rerun_timings,
    ) = await rerun_with_persisted_state(
        tygent_result["scheduler"],
        {"brief": product_brief, "requester": "fsd@acme.dev"},
        rerun_context,
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
    arch_summary = tygent_result["publish"]["architecture"]
    risk_summary = tygent_result["publish"]["risk_register"]
    print(
        f"Architecture components: {len(arch_summary.get('components', []))}, "
        f"Risks identified: {len(risk_summary.get('risks', []))}"
    )

    latency_delta = baseline_result["elapsed"] - tygent_result["elapsed"]
    baseline_timings = baseline_result.get("timings", {})
    tygent_timings = tygent_result.get("timings", {})
    rows = [
        {
            "mode": "Baseline (sequential)",
            "latency": baseline_result["elapsed"] * 1000,
            "iterations": baseline_result["iterations"],
            "status": base_status,
            "tokens": baseline_result["tokens"]["total"],
        },
        {
            "mode": "Tygent (first run)",
            "latency": tygent_result["elapsed"] * 1000,
            "iterations": tygent_result["iterations"],
            "status": publish_result.get("status", "unknown"),
            "tokens": tygent_result["tokens"]["total"],
        },
        {
            "mode": "Tygent (re-run)",
            "latency": repeat_latency * 1000,
            "iterations": rerun_iterations,
            "status": repeat_publish.get("status", "unknown"),
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

    def _summarize_timings(label: str, timings: Dict[str, List[float]]) -> None:
        print(f"\n{label} node timings (ms):")
        if not timings:
            print("  (no timing data captured)")
            return
        header = ["Node", "Avg", "Max", "Count"]
        widths = [
            max(len(header[0]), *(len(name) for name in timings)),
            len(header[1]),
            len(header[2]),
            len(header[3]),
        ]
        print(_format_row(header, widths))
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))
        for node, samples in sorted(timings.items()):
            avg = sum(samples) / len(samples)
            mx = max(samples)
            row = [
                node,
                f"{avg:.1f}",
                f"{mx:.1f}",
                str(len(samples)),
            ]
            print(_format_row(row, widths))

    _summarize_timings("Baseline", baseline_timings)
    _summarize_timings("Tygent (first run)", tygent_timings)
    _summarize_timings("Tygent (re-run)", rerun_timings)

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
    print(
        "• Tygent overlapped three reviewer calls and two artifact generators per revision, "
        "showcasing parallel execution within the cyclic loop."
    )
    print(
        f"• Final document length (chars): baseline {len(baseline_result['publish'].get('document',''))}, "
        f"Tygent first run {len(tygent_result['publish'].get('document',''))}, "
        f"rerun {len(repeat_publish.get('document',''))}."
    )
    print(
        "• PDF parity: pipeline includes flatten/get_mapping/get_adhoc/context_section_* nodes "
        "with scheduler configured to max_parallel_nodes=10 and 5-minute timeouts."
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
