from __future__ import annotations

from pathlib import Path

import pytest

from tyapi.service.auth import AuthError, AuthManager
from tyapi.service.state import ServiceState


@pytest.fixture
def state(tmp_path: Path) -> ServiceState:
    return ServiceState(tmp_path / "state.json")


def test_register_and_key_generation(state: ServiceState) -> None:
    account = state.register_account("Acme", "ops@acme.test")
    api_key = state.create_api_key(account.account_id, label="primary")

    state2 = ServiceState(state.path)
    auth = AuthManager(state2)
    authenticated = auth.authenticate(api_key)
    assert authenticated.account_id == account.account_id


def test_authenticate_missing_key(state: ServiceState) -> None:
    auth = AuthManager(state)
    with pytest.raises(AuthError):
        auth.authenticate(None)


def test_ingestor_configuration(state: ServiceState) -> None:
    account = state.register_account("Beta", "beta@test")
    state.set_ingestor_config(
        account.account_id, {"name": "generic", "config": {"prefix": "beta"}}
    )
    reloaded = ServiceState(state.path)
    stored = reloaded.get_account(account.account_id)
    assert stored
    assert stored.ingestor_config["config"]["prefix"] == "beta"
