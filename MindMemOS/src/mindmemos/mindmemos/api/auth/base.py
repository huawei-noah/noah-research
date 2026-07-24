"""Authentication provider contracts for memory API requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ResolvedIdentity:
    """Resolved standalone memory API identity.

    Purpose: Carry the project scope authorized by one API key.
    Used in: ``api.deps.get_request_context`` when building ``MemoryRequestContext``.
    """

    key_id: str
    project_id: str
    memory_algorithm: str
    scopes: list[str] = field(default_factory=list)
    account_id: str | None = None
    request_id: str | None = None
    user_override_config: dict[str, Any] | None = None
    project_override_config: dict[str, Any] | None = None


class AuthProvider(Protocol):
    """Resolve an Authorization credential into a memory API identity."""

    def resolve_api_key(self, api_key: str) -> ResolvedIdentity: ...
