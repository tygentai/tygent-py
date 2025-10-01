"""Gemini CLI planning integration.

The Gemini CLI emits structured execution plans when invoked with planning
flags. Those plans typically resemble::

    {
        "plan": {
            "steps": [
                {
                    "name": "collect",
                    "instruction": "Call Gemini on query",
                    "deps": [],
                    "provider": "google",
                },
                ...
            ],
            "links": ["https://storage"]
        }
    }

This module converts the CLI payloads into :class:`~tygent.service_bridge.ServicePlan`
instances so Tygent can execute them with scheduling awareness. As with other
integrations, a ``patch`` helper is exposed to add a convenience method to the
CLI planner when the optional dependency is present.
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
            elif isinstance(item, Mapping) and "url" in item:
                url = item.get("url")
                if isinstance(url, str):
                    items.append(url)
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


def _extract_plan(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = payload.get("plan")
    if isinstance(plan, Mapping):
        return plan
    return payload


@dataclass
class GeminiCLIPlanAdapter:
    """Convert Gemini CLI plan payloads into ``ServicePlan`` objects."""

    payload: Mapping[str, Any]
    builder: ServicePlanBuilder = ServicePlanBuilder()

    def to_service_plan(self) -> ServicePlan:
        plan_section = _extract_plan(self.payload)
        steps = plan_section.get("steps") or plan_section.get("tasks") or []
        if not isinstance(steps, Sequence):
            raise ValueError("Gemini CLI payload requires a sequence of steps")

        name = (
            plan_section.get("name")
            or self.payload.get("name")
            or self.payload.get("task")
            or "gemini_cli_plan"
        )

        plan_payload: MutableMapping[str, Any] = {"name": str(name), "steps": []}

        for step in steps:
            if not isinstance(step, Mapping):
                raise TypeError("Gemini CLI step entries must be mappings")
            plan_payload["steps"].append(self._format_step(step))

        links = self._prefetch_links(plan_section)
        if links:
            plan_payload["prefetch"] = {"links": links}

        return self.builder.build(plan_payload)

    # ------------------------------------------------------------------
    def _format_step(self, step: Mapping[str, Any]) -> Mapping[str, Any]:
        name = step.get("name") or step.get("id") or step.get("title")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Gemini CLI step missing 'name'")

        prompt = (
            step.get("prompt")
            or step.get("instruction")
            or step.get("description")
            or step.get("command")
        )
        prompt_str = str(prompt) if prompt is not None else ""

        dependencies = _as_list(step.get("dependencies") or step.get("deps"))

        metadata = {
            **_as_mapping(step.get("metadata")),
        }
        provider = step.get("provider") or metadata.get("provider") or "google"
        metadata.setdefault("provider", provider)
        metadata.setdefault("prompt", prompt_str)
        metadata.setdefault("kind", step.get("kind", "llm"))
        metadata.setdefault("framework", "gemini_cli")
        level = step.get("level") or step.get("stage")
        if level is not None and "level" not in metadata:
            metadata["level"] = level

        tags = set(_as_list(step.get("tags")) + _as_list(metadata.get("tags")))
        tags.add("gemini-cli")

        links = _as_list(step.get("links") or step.get("resources") or step.get("urls"))

        token_estimate = (
            step.get("token_estimate")
            or step.get("tokens")
            or metadata.get("token_estimate")
        )
        if token_estimate is not None and "token_estimate" not in metadata:
            metadata["token_estimate"] = token_estimate

        is_critical = _as_bool(
            step.get("critical")
            or step.get("is_critical")
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
    def _prefetch_links(self, plan_section: Mapping[str, Any]) -> List[str]:
        links: List[str] = []
        links.extend(_as_list(plan_section.get("prefetch_links")))
        links.extend(_as_list(plan_section.get("links")))
        prefetch = plan_section.get("prefetch")
        if isinstance(prefetch, Mapping):
            links.extend(_as_list(prefetch.get("links")))
        attachments = plan_section.get("attachments")
        links.extend(_as_list(attachments))
        return list(dict.fromkeys(links))


def patch() -> None:
    """Attach a helper to ``gemini_cli`` planners if available."""

    try:
        import gemini_cli.planner as gemini_planner  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    planner_cls = getattr(gemini_planner, "Planner", None)
    if planner_cls is None or hasattr(planner_cls, "to_tygent_service_plan"):
        return

    def to_tygent_service_plan(self, payload: Mapping[str, Any]) -> ServicePlan:
        adapter = GeminiCLIPlanAdapter(payload)
        return adapter.to_service_plan()

    setattr(planner_cls, "to_tygent_service_plan", to_tygent_service_plan)


__all__ = ["GeminiCLIPlanAdapter", "patch"]
