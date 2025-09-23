from __future__ import annotations

import asyncio
import pytest

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


def test_plan_convert_with_spec_only(service_bundle) -> None:
    account, _, service, _ = service_bundle
    spec = {
        "discover": {"prompt": "Find data", "deps": []},
        "deliver": {"prompt": "Compose", "deps": ["discover"]},
    }

    result = asyncio.run(service.convert(account, spec=spec))

    assert {step["name"] for step in result["tygent_plan"]["steps"]} >= {"discover", "deliver"}
    assert result["prefetch"]


def test_auth_manager(service_bundle) -> None:
    _, api_key, _, auth = service_bundle
    assert auth.verify(api_key) is True
    with pytest.raises(AuthError):
        auth.authenticate("invalid")


def test_convert_requires_plan_or_spec(service_bundle) -> None:
    account, _, service, _ = service_bundle
    with pytest.raises(ValueError):
        asyncio.run(service.convert(account))
