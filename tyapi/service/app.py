from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from .auth import AuthError, AuthManager
from .service import PlanConversionService
from .state import ServiceState, default_state_path


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

    app.router.add_get("/health", health)
    app.router.add_get("/v1/catalog", catalog)
    app.router.add_post("/v1/plan/convert", convert)

    return app


__all__ = ["create_app"]
