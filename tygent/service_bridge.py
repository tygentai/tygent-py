from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional

from .prefetch import prefetch_many

PromptHandler = Callable[[str, Dict[str, Any], Dict[str, Any]], Awaitable[Any]]


class LLMRuntimeRegistry:
    """Registry that maps provider identifiers to async callables."""

    def __init__(self) -> None:
        self._handlers: Dict[str, PromptHandler] = {}

    def register(self, provider: str, handler: PromptHandler) -> None:
        self._handlers[provider] = handler

    async def call(
        self,
        provider: str,
        prompt: str,
        *,
        step_metadata: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> Any:
        try:
            handler = self._handlers[provider]
        except KeyError as exc:
            raise RuntimeError(
                f"No runtime registered for provider '{provider}'"
            ) from exc
        result = handler(prompt, step_metadata, dict(inputs))
        if asyncio.iscoroutine(result):
            return await result
        return result


DEFAULT_LLM_RUNTIME = LLMRuntimeRegistry()


async def _echo_runtime(
    prompt: str, metadata: Mapping[str, Any], inputs: Dict[str, Any]
) -> Dict[str, Any]:
    return {"prompt": prompt, "metadata": dict(metadata), "inputs": inputs}


DEFAULT_LLM_RUNTIME.register("echo", _echo_runtime)


@dataclass
class ServicePlan:
    plan: Dict[str, Any]
    prefetch_links: List[str]
    raw: Dict[str, Any]


class ServicePlanBuilder:
    """Construct executable plans from service payloads."""

    def __init__(self, registry: Optional[LLMRuntimeRegistry] = None) -> None:
        self.registry = registry or DEFAULT_LLM_RUNTIME

    def build(self, payload: Mapping[str, Any]) -> ServicePlan:
        steps_payload = payload.get("steps")
        if not isinstance(steps_payload, list):
            raise ValueError("Service payload missing 'steps' list")

        steps: List[Dict[str, Any]] = []
        for step in steps_payload:
            name = step.get("name")
            if not isinstance(name, str):
                raise ValueError("Each step requires a string 'name'")
            prompt = step.get("prompt", "")
            kind = step.get("kind", "llm")
            dependencies = step.get("dependencies", [])
            metadata = step.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"Step {name} metadata must be a mapping")
            links = step.get("links", [])

            tags = list(step.get("tags", []))
            merged_metadata = dict(metadata)
            if merged_metadata.get("tags"):
                existing = list(merged_metadata.get("tags"))
                for tag in tags:
                    if tag not in existing:
                        existing.append(tag)
                tags = existing
            if tags:
                merged_metadata["tags"] = tags
            level = step.get("level")
            if level is not None and "level" not in merged_metadata:
                merged_metadata["level"] = level
            merged_metadata.setdefault("links", list(links))
            merged_metadata.setdefault("prompt", prompt)
            merged_metadata.setdefault("kind", kind)
            merged_metadata.setdefault("is_critical", step.get("is_critical", False))

            func = self._build_step_function(
                name, prompt, kind, merged_metadata, list(links)
            )
            plan_entry: Dict[str, Any] = {
                "name": name,
                "func": func,
                "dependencies": list(dependencies),
                "critical": bool(step.get("is_critical", False)),
                "metadata": dict(merged_metadata),
            }
            token_estimate = step.get("metadata", {}).get("token_estimate")
            if token_estimate is not None:
                try:
                    plan_entry["token_cost"] = int(token_estimate)
                except (TypeError, ValueError):
                    plan_entry["token_cost"] = 0
            steps.append(plan_entry)

        plan_dict = {
            "name": payload.get("name", "tygent_service_plan"),
            "steps": steps,
        }
        prefetch_links = []
        seen = set()
        for url in payload.get("prefetch", {}).get("links", []) or []:
            if url not in seen:
                seen.add(url)
                prefetch_links.append(url)
        return ServicePlan(
            plan=plan_dict, prefetch_links=prefetch_links, raw=dict(payload)
        )

    def _build_step_function(
        self,
        name: str,
        prompt_template: str,
        kind: str,
        metadata: Mapping[str, Any],
        links: List[str],
    ) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        provider = str(metadata.get("provider", "echo"))

        delay_value = metadata.get("simulated_duration")
        if delay_value is None:
            delay_value = metadata.get("latency_estimate")
        if delay_value is None:
            delay_value = metadata.get("estimated_duration")
        try:
            delay_seconds = max(float(delay_value), 0.0)
        except (TypeError, ValueError):
            delay_seconds = 0.01 if delay_value is None else 0.0
        if delay_value is None and delay_seconds == 0.0:
            delay_seconds = 0.01

        async def _runner(inputs: Dict[str, Any]) -> Dict[str, Any]:
            rendered_prompt = _render_prompt(prompt_template, inputs)
            payload: Dict[str, Any] = {
                "step": name,
                "prompt": rendered_prompt,
                "inputs": inputs,
                "links": links,
                "kind": kind,
                "metadata": dict(metadata),
            }
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            if kind == "llm":
                result = await self.registry.call(
                    provider,
                    rendered_prompt,
                    step_metadata=metadata,
                    inputs=inputs,
                )
                payload["result"] = result
            else:
                payload["result"] = {"echo": rendered_prompt}
            return payload

        return _runner


def _render_prompt(template: str, inputs: Mapping[str, Any]) -> str:
    if not template:
        return ""
    context = _FormatContext(inputs)
    try:
        return template.format_map(context)
    except KeyError:
        return template


class _FormatContext(dict):
    def __init__(self, inputs: Mapping[str, Any]) -> None:
        super().__init__()
        self._inputs = inputs

    def __missing__(self, key: str) -> str:
        value = _lookup(self._inputs, key)
        if value is None:
            return ""
        return value  # allow formatter to continue traversing nested keys

    def __getitem__(self, key: str) -> str:  # type: ignore[override]
        value = _lookup(self._inputs, key)
        if value is None:
            raise KeyError(key)
        return value


def _lookup(source: Mapping[str, Any], path: str) -> Any:
    current: Any = source
    token = ""
    index = 0
    length = len(path)

    def _advance(part: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get(part)
        if isinstance(value, list):
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(value):
                    return value[idx]
        return None

    while index < length:
        char = path[index]
        if char == ".":
            if token:
                current = _advance(token, current)
                if current is None:
                    return None
                token = ""
            index += 1
        elif char == "[":
            if token:
                current = _advance(token, current)
                if current is None:
                    return None
                token = ""
            index += 1
            closing = path.find("]", index)
            if closing == -1:
                return None
            key = path[index:closing]
            current = _advance(key, current)
            if current is None:
                return None
            index = closing + 1
        else:
            token += char
            index += 1

    if token:
        current = _advance(token, current)
    return current


async def execute_service_plan(
    service_plan: ServicePlan,
    inputs: Dict[str, Any],
    *,
    registry: Optional[LLMRuntimeRegistry] = None,
    max_parallel_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Prefetch plan resources, construct a scheduler, and execute."""

    from .plan_parser import parse_plan
    from .scheduler import Scheduler

    prefetch_results = {}
    if service_plan.prefetch_links:
        prefetch_results = await prefetch_many(service_plan.prefetch_links)
    enriched_inputs = dict(inputs)
    if prefetch_results:
        enriched_inputs.setdefault("prefetch", prefetch_results)

    dag, critical = parse_plan(service_plan.plan)
    scheduler = Scheduler(dag)
    if max_parallel_nodes is not None:
        scheduler.max_parallel_nodes = max_parallel_nodes
    scheduler.priority_nodes = critical
    return await scheduler.execute(enriched_inputs)
