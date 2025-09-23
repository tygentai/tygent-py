from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .ingestors import DEFAULT_INGESTOR_REGISTRY, BasePlanIngestor, PlanIngestorRegistry
from .state import AccountRecord, ServiceState
from .transform import PlanTransformer


class PlanConversionService:
    """High-level orchestrator that powers the SaaS plan conversion API."""

    def __init__(
        self,
        state: ServiceState,
        ingestor_registry: PlanIngestorRegistry = DEFAULT_INGESTOR_REGISTRY,
    ) -> None:
        self._state = state
        self._ingestors = ingestor_registry

    async def convert(
        self,
        account: AccountRecord,
        *,
        plan_payload: Optional[Mapping[str, Any]] = None,
        ingestor_payload: Optional[Mapping[str, Any]] = None,
        spec: Optional[Dict[str, Dict[str, Any]]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = dict(options or {})
        redundancy_mode = options.get("redundancy_mode", "inline")

        if spec is None:
            ingestor = self._resolve_ingestor(account, ingestor_payload)
            if plan_payload is None:
                raise ValueError("'plan' payload required when spec is not provided")
            spec = ingestor.ingest(plan_payload, options=options)
        else:
            ingestor = None

        transformer = PlanTransformer(redundancy_mode=redundancy_mode)
        tygent_plan = transformer.transform(spec)

        return {
            "ingestor": ingestor.describe() if ingestor else None,
            "options": options,
            "tygent_plan": tygent_plan,
            "prefetch": tygent_plan.get("prefetch"),
            "spec": spec,
        }

    def _resolve_ingestor(
        self,
        account: AccountRecord,
        payload: Optional[Mapping[str, Any]],
    ) -> BasePlanIngestor:
        config = self._extract_config(account.ingestor_config, payload)
        name = config.get("name", "generic")
        ingestor_config = config.get("config") or {}
        return self._ingestors.create(name, ingestor_config)

    @staticmethod
    def _extract_config(
        account_config: Mapping[str, Any], payload: Optional[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if account_config:
            merged.update(account_config)
        if payload:
            merged.update(payload)
        return merged
