from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from tygent.service_bridge import ServicePlanBuilder, execute_service_plan

from .auth import AuthError, AuthManager
from .service import PlanConversionService
from .state import ServiceState, default_state_path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


def create_app(state_path: Optional[Path] = None) -> web.Application:
    state_file = state_path or default_state_path()
    state = ServiceState(state_file)
    auth = AuthManager(state)
    service = PlanConversionService(state)

    app = web.Application()

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def catalog(_: web.Request) -> web.Response:
        payload = {
            "ingestors": service._ingestors.describe(),  # type: ignore[attr-defined]
        }
        return web.json_response(payload)

    async def convert(request: web.Request) -> web.Response:
        try:
            account = auth.authenticate(request.headers.get("X-Tygent-Key"))
        except AuthError as exc:
            raise web.HTTPUnauthorized(reason=str(exc)) from exc

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason="Invalid JSON body") from exc

        spec = body.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise web.HTTPBadRequest(reason="spec must be a JSON object if provided")

        plan_payload = body.get("plan")
        if plan_payload is not None and not isinstance(plan_payload, dict):
            raise web.HTTPBadRequest(reason="plan must be an object when provided")
        ingestor_payload = body.get("ingestor")
        options = body.get("options")

        try:
            result = await service.convert(
                account,
                plan_payload=plan_payload,
                ingestor_payload=ingestor_payload,
                spec=spec,
                options=options,
            )
        except (KeyError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        return web.json_response(result)

    async def benchmark(request: web.Request) -> web.Response:
        try:
            account = auth.authenticate(request.headers.get("X-Tygent-Key"))
        except AuthError as exc:
            raise web.HTTPUnauthorized(reason=str(exc)) from exc

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason="Invalid JSON body") from exc

        spec = body.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise web.HTTPBadRequest(reason="spec must be a JSON object if provided")

        plan_payload = body.get("plan")
        if plan_payload is not None and not isinstance(plan_payload, dict):
            raise web.HTTPBadRequest(reason="plan must be an object when provided")
        ingestor_payload = body.get("ingestor")
        options = body.get("options")
        inputs_payload = body.get("inputs") or {}
        if inputs_payload and not isinstance(inputs_payload, dict):
            raise web.HTTPBadRequest(reason="inputs must be an object when provided")

        try:
            conversion = await service.convert(
                account,
                plan_payload=plan_payload,
                ingestor_payload=ingestor_payload,
                spec=spec,
                options=options,
            )
        except (KeyError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        tygent_plan = conversion.get("tygent_plan")
        if not isinstance(tygent_plan, dict):
            raise web.HTTPBadRequest(reason="Unable to build Tygent plan from payload")

        builder = ServicePlanBuilder()

        try:
            service_plan = builder.build(tygent_plan)
        except Exception as exc:  # pragma: no cover - defensive
            raise web.HTTPBadRequest(reason=f"Failed to build execution plan: {exc}") from exc

        inputs: Dict[str, Any] = dict(inputs_payload)

        start = time.perf_counter()
        baseline_output = await execute_service_plan(
            service_plan,
            dict(inputs),
            max_parallel_nodes=1,
        )
        baseline_duration_ms = round((time.perf_counter() - start) * 1000, 3)

        start = time.perf_counter()
        accelerated_output = await execute_service_plan(
            builder.build(tygent_plan),
            dict(inputs),
        )
        accelerated_duration_ms = round((time.perf_counter() - start) * 1000, 3)

        step_count = len(tygent_plan.get("steps", [])) if isinstance(tygent_plan.get("steps"), list) else None

        execution_summary = {
            "baseline": {
                "duration_ms": baseline_duration_ms,
                "step_count": step_count,
                "results": baseline_output.get("results"),
                "tokens_used": baseline_output.get("tokens_used"),
            },
            "tygent": {
                "duration_ms": accelerated_duration_ms,
                "step_count": step_count,
                "results": accelerated_output.get("results"),
                "tokens_used": accelerated_output.get("tokens_used"),
            },
        }

        payload = {
            "conversion": conversion,
            "execution": execution_summary,
        }
        return web.json_response(payload)

    async def register_account(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason="Invalid JSON body") from exc

        name = body.get("name")
        email = body.get("email")
        label = body.get("label") or "default"
        if not isinstance(name, str) or not name.strip():
            raise web.HTTPBadRequest(reason="Missing account name")
        if not isinstance(email, str) or not email.strip():
            raise web.HTTPBadRequest(reason="Missing account email")

        account = state.register_account(name=name.strip(), email=email.strip())
        api_key = state.create_api_key(account.account_id, label=label)

        payload = {
            "account": account.to_dict(),
            "api_key": api_key,
        }
        return web.json_response(payload, status=201)

    async def list_accounts(_: web.Request) -> web.Response:
        accounts = [record.to_dict() for record in state.list_accounts()]
        return web.json_response({"accounts": accounts})

    async def generate_key(request: web.Request) -> web.Response:
        account_id = request.match_info.get("account_id")
        if not account_id:
            raise web.HTTPBadRequest(reason="Missing account_id")

        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}

        label = body.get("label") or "default"

        try:
            api_key = state.create_api_key(account_id, label=label)
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc

        return web.json_response(
            {
                "account_id": account_id,
                "label": label,
                "api_key": api_key,
            },
            status=201,
        )

    async def index(_: web.Request) -> web.StreamResponse:
        if FRONTEND_INDEX.exists():
            return web.FileResponse(FRONTEND_INDEX)
        return web.Response(text="Tygent service console unavailable", content_type="text/plain")

    app.router.add_get("/health", health)
    app.router.add_get("/v1/catalog", catalog)
    app.router.add_post("/v1/plan/convert", convert)
    app.router.add_post("/v1/plan/benchmark", benchmark)
    app.router.add_get("/v1/accounts", list_accounts)
    app.router.add_post("/v1/accounts/register", register_account)
    app.router.add_post("/v1/accounts/{account_id}/keys", generate_key)
    app.router.add_get("/", index)
    if FRONTEND_DIR.exists():
        app.router.add_static("/assets/", FRONTEND_DIR)

    return app


__all__ = ["create_app"]
