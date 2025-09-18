from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utcnow_iso() -> str:
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ApiKeyRecord:
    hash: str
    label: str
    created_at: str


@dataclass
class AccountRecord:
    account_id: str
    name: str
    email: str
    created_at: str
    ingestor_config: Dict[str, Any] = field(default_factory=dict)
    api_keys: List[ApiKeyRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at,
            "ingestor_config": self.ingestor_config,
            "api_keys": [
                {"hash": key.hash, "label": key.label, "created_at": key.created_at}
                for key in self.api_keys
            ],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AccountRecord":
        api_keys = [
            ApiKeyRecord(hash=rec["hash"], label=rec.get("label", ""), created_at=rec["created_at"])
            for rec in data.get("api_keys", [])
        ]
        return AccountRecord(
            account_id=data["account_id"],
            name=data.get("name", ""),
            email=data.get("email", ""),
            created_at=data.get("created_at", _utcnow_iso()),
            ingestor_config=data.get("ingestor_config", {}),
            api_keys=api_keys,
        )


class ServiceState:
    """File-backed store for SaaS configuration, accounts, and API keys."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._accounts: Dict[str, AccountRecord] = {}
        self._load()

    # -------------------------- persistence --------------------------
    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        accounts = raw.get("accounts", [])
        for account_data in accounts:
            account = AccountRecord.from_dict(account_data)
            self._accounts[account.account_id] = account

    def _flush(self) -> None:
        self._ensure_parent()
        payload = {"accounts": [acc.to_dict() for acc in self._accounts.values()]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # -------------------------- account ops --------------------------
    def register_account(self, name: str, email: str) -> AccountRecord:
        account_id = f"acct_{uuid.uuid4().hex[:12]}"
        record = AccountRecord(
            account_id=account_id,
            name=name,
            email=email,
            created_at=_utcnow_iso(),
        )
        self._accounts[account_id] = record
        self._flush()
        return record

    def list_accounts(self) -> List[AccountRecord]:
        return list(self._accounts.values())

    def get_account(self, account_id: str) -> Optional[AccountRecord]:
        return self._accounts.get(account_id)

    # -------------------------- provider config --------------------------
    def set_ingestor_config(self, account_id: str, config: Dict[str, Any]) -> None:
        account = self._require_account(account_id)
        account.ingestor_config = config
        self._flush()

    # -------------------------- API keys --------------------------
    def create_api_key(self, account_id: str, label: str = "default") -> str:
        account = self._require_account(account_id)
        key = secrets.token_urlsafe(32)
        account.api_keys.append(
            ApiKeyRecord(hash=_hash_key(key), label=label, created_at=_utcnow_iso())
        )
        self._flush()
        return key

    def revoke_api_key(self, account_id: str, label: str) -> bool:
        account = self._require_account(account_id)
        before = len(account.api_keys)
        account.api_keys = [key for key in account.api_keys if key.label != label]
        changed = len(account.api_keys) != before
        if changed:
            self._flush()
        return changed

    def resolve_api_key(self, api_key: str) -> Optional[AccountRecord]:
        hashed = _hash_key(api_key)
        for account in self._accounts.values():
            if any(rec.hash == hashed for rec in account.api_keys):
                return account
        return None

    # -------------------------- internal helpers --------------------------
    def _require_account(self, account_id: str) -> AccountRecord:
        account = self._accounts.get(account_id)
        if not account:
            raise KeyError(f"Unknown account_id: {account_id}")
        return account


def default_state_path() -> Path:
    """Resolve the default location for the service state file."""

    env_value = os.getenv("TYGENT_SERVICE_STATE")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path(__file__).resolve().parent / "service_state.json").resolve()
