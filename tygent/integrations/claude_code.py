"""Claude Code planning integration for Tygent.

The Claude Code editor exposes high-level planning payloads before executing
revision commands. These payloads typically resemble the following structure::

    {
        "plan_id": "session-123",
        "tasks": [
            {
                "id": "plan_outline",
                "prompt": "Summarise repo state",
                "deps": [],
                "metadata": {"provider": "anthropic", "is_critical": true},
                "links": ["https://docs"],
            },
            ...
        ],
        "resources": ["https://docs"],
    }

This module normalises such payloads into :class:`~tygent.service_bridge.ServicePlan`
objects so they can be executed by Tygent's scheduler. It also exposes a
``patch`` helper that attaches a ``to_tygent_service_plan`` method to
``claude_code`` planners when the optional dependency is available.
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


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "critical"}
    return bool(value)


@dataclass
class ClaudeCodePlanAdapter:
    """Convert Claude Code planning payloads into ``ServicePlan`` objects."""

    payload: Mapping[str, Any]
    builder: ServicePlanBuilder = ServicePlanBuilder()

    def to_service_plan(self) -> ServicePlan:
        """Return a ServicePlan ready for execution by the scheduler."""

        plan_payload: MutableMapping[str, Any] = {
            "name": str(
                self.payload.get("plan_id")
                or self.payload.get("session_id")
                or self.payload.get("name")
                or "claude_code_plan"
            ),
            "steps": [],
        }

        steps = self.payload.get("tasks") or self.payload.get("steps") or []
        if not isinstance(steps, Sequence):
            raise ValueError("Claude Code payload requires a sequence of tasks")

        for entry in steps:
            if not isinstance(entry, Mapping):
                raise TypeError("Each Claude Code task must be a mapping")
            plan_payload["steps"].append(self._format_step(entry))

        prefetch_links = list(self._gather_prefetch_links())
        if prefetch_links:
            plan_payload["prefetch"] = {"links": prefetch_links}

        return self.builder.build(plan_payload)

    # ------------------------------------------------------------------
    def _format_step(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        name = (
            task.get("name") or task.get("id") or task.get("step") or task.get("title")
        )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Claude Code task missing 'name' or 'id'")

        prompt = (
            task.get("prompt") or task.get("instruction") or task.get("description")
        )
        prompt_str = str(prompt) if prompt is not None else ""

        dependencies = _as_list(
            task.get("dependencies") or task.get("deps") or task.get("requires")
        )

        metadata = {
            **_as_mapping(task.get("metadata")),
        }
        provider = task.get("provider") or metadata.get("provider") or "anthropic"
        metadata.setdefault("provider", provider)
        metadata.setdefault("framework", "claude_code")
        metadata.setdefault("prompt", prompt_str)
        metadata.setdefault("kind", task.get("kind", "llm"))
        level = task.get("level") or task.get("stage")
        if level is not None and "level" not in metadata:
            metadata["level"] = level

        tags = set(_as_list(task.get("tags")) + _as_list(metadata.get("tags")))
        tags.add("claude-code")

        links = _as_list(task.get("links") or task.get("resources") or task.get("urls"))

        token_estimate = (
            task.get("token_estimate")
            or task.get("tokens")
            or metadata.get("token_estimate")
        )
        if token_estimate is not None and "token_estimate" not in metadata:
            metadata["token_estimate"] = token_estimate

        is_critical = _coerce_bool(
            task.get("critical")
            or task.get("is_critical")
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
    def _gather_prefetch_links(self) -> List[str]:
        links: List[str] = []
        links.extend(_as_list(self.payload.get("prefetch_links")))
        prefetch = self.payload.get("prefetch")
        if isinstance(prefetch, Mapping):
            links.extend(_as_list(prefetch.get("links")))
        links.extend(_as_list(self.payload.get("resources")))
        links.extend(_as_list(self.payload.get("attachments")))
        return list(dict.fromkeys(links))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def patch() -> None:
    """Attach ``to_tygent_service_plan`` to Claude Code planners if present."""

    try:
        import claude_code.planner as claude_planner  # type: ignore
    except Exception:  # pragma: no cover - optional dependency absent
        return

    planner_cls = getattr(claude_planner, "Planner", None)
    if planner_cls is None or hasattr(planner_cls, "to_tygent_service_plan"):
        return

    def to_tygent_service_plan(self, payload: Mapping[str, Any]) -> ServicePlan:
        adapter = ClaudeCodePlanAdapter(payload)
        return adapter.to_service_plan()

    setattr(planner_cls, "to_tygent_service_plan", to_tygent_service_plan)


__all__ = ["ClaudeCodePlanAdapter", "patch"]
