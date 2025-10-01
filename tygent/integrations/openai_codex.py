"""OpenAI Codex planning integration for Tygent.

OpenAI's historical Codex planning endpoints emitted lightweight workflow
representations with node metadata. The structure varies slightly across
clients, but generally includes a list of steps with prompts and dependency
information. This module adapts those payloads into
:class:`~tygent.service_bridge.ServicePlan` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

from ..service_bridge import ServicePlan, ServicePlanBuilder


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, Mapping) and "id" in item:
                name = item.get("id")
                if isinstance(name, str):
                    items.append(name)
            elif isinstance(item, Mapping) and "name" in item:
                name = item.get("name")
                if isinstance(name, str):
                    items.append(name)
        return items
    return []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "critical"}
    return bool(value)


def _extract_nodes(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    workflow = payload.get("workflow")
    if isinstance(workflow, Mapping):
        nodes = workflow.get("nodes") or workflow.get("steps")
        if isinstance(nodes, Sequence):
            return nodes  # type: ignore[return-value]
    steps = payload.get("steps") or payload.get("actions")
    if isinstance(steps, Sequence):
        return steps  # type: ignore[return-value]
    raise ValueError("OpenAI Codex payload missing step definitions")


@dataclass
class OpenAICodexPlanAdapter:
    """Convert OpenAI Codex workflow payloads into ``ServicePlan`` objects."""

    payload: Mapping[str, Any]
    builder: ServicePlanBuilder = ServicePlanBuilder()

    def to_service_plan(self) -> ServicePlan:
        nodes = _extract_nodes(self.payload)
        plan_name = (
            self.payload.get("name")
            or (self.payload.get("workflow") or {}).get("name")
            or "openai_codex_plan"
        )

        plan_payload: MutableMapping[str, Any] = {"name": str(plan_name), "steps": []}

        for node in nodes:
            if not isinstance(node, Mapping):
                raise TypeError("OpenAI Codex nodes must be mappings")
            plan_payload["steps"].append(self._format_step(node))

        links = self._prefetch_links()
        if links:
            plan_payload["prefetch"] = {"links": links}

        return self.builder.build(plan_payload)

    # ------------------------------------------------------------------
    def _format_step(self, node: Mapping[str, Any]) -> Mapping[str, Any]:
        name = node.get("name") or node.get("id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("OpenAI Codex node missing 'name'")

        prompt = (
            node.get("prompt")
            or node.get("instruction")
            or node.get("code")
            or node.get("description")
        )
        prompt_str = str(prompt) if prompt is not None else ""

        dependencies = _as_list(
            node.get("dependencies")
            or node.get("deps")
            or node.get("parents")
            or node.get("requires")
        )

        metadata = {
            **_as_mapping(node.get("metadata")),
        }
        provider = node.get("provider") or metadata.get("provider") or "openai"
        metadata.setdefault("provider", provider)
        metadata.setdefault("prompt", prompt_str)
        metadata.setdefault("kind", node.get("kind", "llm"))
        metadata.setdefault("framework", "openai_codex")
        level = node.get("level") or node.get("stage")
        if level is not None and "level" not in metadata:
            metadata["level"] = level

        tags = set(_as_list(node.get("tags")) + _as_list(metadata.get("tags")))
        tags.add("openai-codex")

        links = _as_list(node.get("links") or node.get("resources") or node.get("urls"))

        token_estimate = (
            node.get("token_estimate")
            or node.get("tokens")
            or metadata.get("token_estimate")
        )
        if token_estimate is not None and "token_estimate" not in metadata:
            metadata["token_estimate"] = token_estimate

        is_critical = _as_bool(
            node.get("critical")
            or node.get("is_critical")
            or metadata.get("is_critical")
        )

        return {
            "name": name,
            "kind": metadata.get("kind", "llm"),
            "prompt": prompt_str,
            "dependencies": dependencies,
            "metadata": metadata,
            "tags": sorted(tags),
            "links": links,
            "is_critical": is_critical,
            "token_estimate": token_estimate,
        }

    # ------------------------------------------------------------------
    def _prefetch_links(self) -> List[str]:
        links: List[str] = []
        links.extend(_as_list(self.payload.get("prefetch_links")))
        workflow = self.payload.get("workflow")
        if isinstance(workflow, Mapping):
            links.extend(_as_list(workflow.get("prefetch_links")))
            prefetch = workflow.get("prefetch")
            if isinstance(prefetch, Mapping):
                links.extend(_as_list(prefetch.get("links")))
        links.extend(_as_list(self.payload.get("links")))
        links.extend(_as_list(self.payload.get("resources")))
        return list(dict.fromkeys(links))


def patch() -> None:
    """Attach ``to_tygent_service_plan`` to Codex planners if present."""

    try:
        import openai.codex.planning as codex_planning  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    planner_cls = getattr(codex_planning, "Planner", None)
    if planner_cls is None or hasattr(planner_cls, "to_tygent_service_plan"):
        return

    def to_tygent_service_plan(self, payload: Mapping[str, Any]) -> ServicePlan:
        adapter = OpenAICodexPlanAdapter(payload)
        return adapter.to_service_plan()

    setattr(planner_cls, "to_tygent_service_plan", to_tygent_service_plan)


__all__ = ["OpenAICodexPlanAdapter", "patch"]
