"""
Session management primitives for Tygent executions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class SessionStore:
    """Abstract storage for persisting execution state across runs."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())

    def get(self, key: str, default: Any = None) -> Any:
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_node_state(self, node_name: str, default: Any = None) -> Any:
        return self.get(f"nodes.{node_name}", default)

    def set_node_state(self, node_name: str, value: Any) -> None:
        self.set(f"nodes.{node_name}", value)

    def delete_node_state(self, node_name: str) -> None:
        self.delete(f"nodes.{node_name}")


class InMemorySessionStore(SessionStore):
    """Simple dictionary-backed session store."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        super().__init__(session_id=session_id)
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._data)


@dataclass
class NodeContext:
    """Runtime context provided to nodes during execution."""

    node_name: str
    session: SessionStore
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _state_saved: bool = field(default=False, init=False)

    def load_state(self, default: Any = None) -> Any:
        """Return persisted state for the node, if available."""

        return self.session.get_node_state(self.node_name, default)

    def save_state(self, value: Any) -> None:
        """Persist state for the node."""

        self.session.set_node_state(self.node_name, value)
        self._state_saved = True

    @property
    def state_saved(self) -> bool:
        """Whether ``save_state`` has been called during this execution."""

        return self._state_saved
