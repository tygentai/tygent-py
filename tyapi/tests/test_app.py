from __future__ import annotations

import asyncio
import pytest
from aiohttp.test_utils import TestClient, TestServer

from tyapi.service.app import create_app
from tyapi.service.auth import AuthError, AuthManager
from tyapi.service.service import PlanConversionService
from tyapi.service.state import ServiceState


@pytest.fixture
def service_bundle(tmp_path):
    state = ServiceState(tmp_path / "state.json")
    account = state.register_account("Acme", "ops@acme.test")
    api_key = state.create_api_key(account.account_id)
    service = PlanConversionService(state)
    auth = AuthManager(state)
    return account, api_key, service, auth


def test_plan_convert_with_plan_payload(service_bundle) -> None:
    account, _, service, _ = service_bundle
    plan_payload = {
        "steps": [
            {"name": "discover", "prompt": "Find", "deps": []},
            {
                "name": "deliver",
                "prompt": "Compose",
                "deps": ["discover"],
                "links": ["https://example"],
            },
        ]
    }

    result = asyncio.run(
        service.convert(
            account,
            plan_payload=plan_payload,
            options={"redundancy_mode": "inline"},
        )
    )

    assert result["ingestor"]["name"] == "generic"
    assert result["tygent_plan"]["steps"]
    assert result["prefetch"]["links"]
    assert result["spec"]["discover"]["prompt"] == "Find"


def test_plan_convert_with_spec_only(service_bundle) -> None:
    account, _, service, _ = service_bundle
    spec = {
        "discover": {"prompt": "Find data", "deps": []},
        "deliver": {"prompt": "Compose", "deps": ["discover"]},
    }

    result = asyncio.run(service.convert(account, spec=spec))

    assert {step["name"] for step in result["tygent_plan"]["steps"]} >= {"discover", "deliver"}
    assert result["prefetch"]
    assert result["spec"]["discover"]["prompt"] == "Find data"


def test_auth_manager(service_bundle) -> None:
    _, api_key, _, auth = service_bundle
    assert auth.verify(api_key) is True
    with pytest.raises(AuthError):
        auth.authenticate("invalid")


def test_convert_requires_plan_or_spec(service_bundle) -> None:
    account, _, service, _ = service_bundle
    with pytest.raises(ValueError):
        asyncio.run(service.convert(account))

def test_account_registration_and_key_routes(tmp_path) -> None:
    async def scenario() -> None:
        app = create_app(tmp_path / "state.json")
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                register_resp = await client.post(
                    "/v1/accounts/register",
                    json={"name": "Beta", "email": "beta@example.com", "label": "default"},
                )
                assert register_resp.status == 201
                register_payload = await register_resp.json()
                account_id = register_payload["account"]["account_id"]
                first_key = register_payload["api_key"]
                assert first_key

                list_resp = await client.get("/v1/accounts")
                assert list_resp.status == 200
                list_payload = await list_resp.json()
                account_ids = [record["account_id"] for record in list_payload["accounts"]]
                assert account_id in account_ids

                new_key_resp = await client.post(
                    f"/v1/accounts/{account_id}/keys", json={"label": "secondary"}
                )
                assert new_key_resp.status == 201
                new_key_payload = await new_key_resp.json()
                assert new_key_payload["api_key"] and new_key_payload["api_key"] != first_key

    asyncio.run(scenario())


def test_generate_key_unknown_account(tmp_path) -> None:
    async def scenario() -> None:
        app = create_app(tmp_path / "state.json")
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                resp = await client.post("/v1/accounts/missing/keys", json={})
                assert resp.status == 404

    asyncio.run(scenario())


def test_plan_benchmark_endpoint(tmp_path) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "state.json"
        state = ServiceState(state_path)
        account = state.register_account("Gamma", "gamma@example.com")
        api_key = state.create_api_key(account.account_id)

        app = create_app(state_path)
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                payload = {
                    "plan": {
                        "steps": [
                            {
                                "name": "discover",
                                "prompt": "Scan sources",
                                "deps": [],
                                "metadata": {"simulated_duration": 0.03},
                            },
                            {
                                "name": "collect",
                                "prompt": "Collect documents",
                                "deps": [],
                                "metadata": {"simulated_duration": 0.03},
                            },
                            {
                                "name": "synthesize",
                                "prompt": "Summarize",
                                "deps": ["discover", "collect"],
                                "metadata": {"simulated_duration": 0.03},
                            },
                        ]
                    },
                    "options": {"redundancy_mode": "inline"},
                }

                resp = await client.post(
                    "/v1/plan/benchmark",
                    headers={"X-Tygent-Key": api_key},
                    json=payload,
                )
                assert resp.status == 200
                data = await resp.json()
                assert data["conversion"]["tygent_plan"]["steps"]
                baseline = data["execution"]["baseline"]
                accelerated = data["execution"]["tygent"]
                assert baseline["duration_ms"] >= accelerated["duration_ms"]
                assert baseline["step_count"] == 3

    asyncio.run(scenario())
