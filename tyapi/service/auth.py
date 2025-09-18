from __future__ import annotations

from typing import Optional

from .state import AccountRecord, ServiceState


class AuthError(RuntimeError):
    """Raised when Tygent API key authentication fails."""


class AuthManager:
    def __init__(self, state: ServiceState) -> None:
        self._state = state

    def authenticate(self, api_key: Optional[str]) -> AccountRecord:
        if not api_key:
            raise AuthError("Missing X-Tygent-Key header")
        account = self._state.resolve_api_key(api_key)
        if not account:
            raise AuthError("Invalid Tygent API key")
        return account

    def verify(self, api_key: str) -> bool:
        return self._state.resolve_api_key(api_key) is not None
