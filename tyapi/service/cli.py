from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from .app import create_app
from .ingestors import DEFAULT_INGESTOR_REGISTRY
from .state import ServiceState, default_state_path


def _load_state(path: Optional[str]) -> ServiceState:
    state_path = Path(path).expanduser().resolve() if path else default_state_path()
    return ServiceState(state_path)


def _prompt(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value:
            return value
        print("Value is required. Please try again.")


def _parse_json_config(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON config: {exc}")


def _do_register(args: argparse.Namespace) -> None:
    state = _load_state(args.state)
    name = args.name or _prompt("Account name")
    email = args.email or _prompt("Contact email")
    record = state.register_account(name=name, email=email)
    print("Registered account:")
    print(json.dumps(record.to_dict(), indent=2))


def _do_configure_ingestor(args: argparse.Namespace) -> None:
    state = _load_state(args.state)
    account_id = args.account or _prompt("Account id")
    ingestor_name = args.name or _prompt("Ingestor name", default="generic")
    raw_config = args.config or _prompt("Ingestor config (JSON)", default="{}")
    config = {"name": ingestor_name, "config": _parse_json_config(raw_config)}
    state.set_ingestor_config(account_id, config)
    print(f"Updated ingestor configuration for {account_id} -> {ingestor_name}")


def _do_generate_key(args: argparse.Namespace) -> None:
    state = _load_state(args.state)
    account_id = args.account or _prompt("Account id")
    label = args.label or _prompt("Key label", default="default")
    api_key = state.create_api_key(account_id, label=label)
    print("Generated Tygent API key (store securely, it will not be shown again):")
    print(api_key)


def _do_list_accounts(args: argparse.Namespace) -> None:
    state = _load_state(args.state)
    accounts = [record.to_dict() for record in state.list_accounts()]
    print(json.dumps(accounts, indent=2))


def _do_catalog(_: argparse.Namespace) -> None:
    payload = {
        "ingestors": DEFAULT_INGESTOR_REGISTRY.describe(),
    }
    print(json.dumps(payload, indent=2))


def _do_serve(args: argparse.Namespace) -> None:
    state_path = Path(args.state).expanduser().resolve() if args.state else None
    app = create_app(state_path)

    port = args.port
    print(f"Starting Tygent SaaS planner service on http://127.0.0.1:{port}")
    web.run_app(app, port=port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tygent SaaS planner CLI")
    parser.add_argument("--state", help="Path to service state file")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="Register a new tenant account")
    register.add_argument("--name")
    register.add_argument("--email")
    register.set_defaults(func=_do_register)

    ingest = sub.add_parser("configure-ingestor", help="Set default plan ingestor for an account")
    ingest.add_argument("--account")
    ingest.add_argument("--name")
    ingest.add_argument("--config")
    ingest.set_defaults(func=_do_configure_ingestor)

    keygen = sub.add_parser("generate-key", help="Create a new API key for an account")
    keygen.add_argument("--account")
    keygen.add_argument("--label")
    keygen.set_defaults(func=_do_generate_key)

    list_accounts = sub.add_parser("list-accounts", help="List registered accounts and metadata")
    list_accounts.set_defaults(func=_do_list_accounts)

    catalog = sub.add_parser("catalog", help="Show available plan ingestors")
    catalog.set_defaults(func=_do_catalog)

    serve = sub.add_parser("serve", help="Run the local SaaS planner service")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=_do_serve)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
