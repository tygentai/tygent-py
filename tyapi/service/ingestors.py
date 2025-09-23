from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Type


class IngestError(ValueError):
    """Raised when a framework plan cannot be ingested."""


@dataclass
class StepPayload:
    name: str
    kind: str
    prompt: str
    deps: Iterable[str]
    links: Iterable[str]
    metadata: Dict[str, Any]

    def to_spec(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "deps": list(self.deps),
            "links": list(self.links),
            "metadata": dict(self.metadata),
        }


class BasePlanIngestor:
    """Convert framework-specific plans to canonical step specifications."""

    name: str = "generic"

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})

    def ingest(self, plan: Mapping[str, Any], *, options: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "config": self.config}


class GenericPlanIngestor(BasePlanIngestor):
    """Accepts plans already expressed in the canonical format."""

    name = "generic"

    def ingest(self, plan: Mapping[str, Any], *, options: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            raise IngestError("Expecting 'steps' list in plan payload")
        spec: Dict[str, Dict[str, Any]] = {}
        for raw in steps:
            if not isinstance(raw, Mapping):
                raise IngestError("Plan steps must be objects")
            name = raw.get("name")
            if not isinstance(name, str):
                raise IngestError("Each step requires a string 'name'")
            kind = str(raw.get("kind", "llm"))
            prompt = str(raw.get("prompt", ""))
            deps = raw.get("deps", [])
            if isinstance(deps, str):
                raise IngestError(f"Step {name} has invalid string 'deps'")
            if not isinstance(deps, Iterable):
                raise IngestError(f"Step {name} has invalid 'deps'")
            links = raw.get("links", [])
            if isinstance(links, str):
                links = [links]
            elif not isinstance(links, Iterable):
                raise IngestError(f"Step {name} has invalid 'links'")
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise IngestError(f"Step {name} has invalid 'metadata'")
            payload = StepPayload(
                name=name,
                kind=kind,
                prompt=prompt,
                deps=list(deps),
                links=list(links),
                metadata=dict(metadata),
            )
            if name in spec:
                raise IngestError(f"Duplicate step name: {name}")
            spec[name] = payload.to_spec()
        if not spec:
            raise IngestError("Plan contains no steps")
        return spec


class LangChainIngestor(GenericPlanIngestor):
    name = "langchain"


class CrewAIIngestor(GenericPlanIngestor):
    name = "crewai"


class PlanIngestorRegistry:
    def __init__(self) -> None:
        self._ingestors: Dict[str, Type[BasePlanIngestor]] = {
            GenericPlanIngestor.name: GenericPlanIngestor,
            LangChainIngestor.name: LangChainIngestor,
            CrewAIIngestor.name: CrewAIIngestor,
        }

    def register(self, ingestor_cls: Type[BasePlanIngestor]) -> None:
        self._ingestors[ingestor_cls.name] = ingestor_cls

    def create(self, name: str, config: Optional[Mapping[str, Any]] = None) -> BasePlanIngestor:
        try:
            ingestor_cls = self._ingestors[name]
        except KeyError as exc:
            raise KeyError(f"Unknown ingestor: {name}") from exc
        return ingestor_cls(config)

    def describe(self) -> Dict[str, Any]:
        return {name: cls.__doc__ or "" for name, cls in self._ingestors.items()}


DEFAULT_INGESTOR_REGISTRY = PlanIngestorRegistry()
