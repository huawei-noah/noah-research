"""Memory algorithm selection for public API requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..errors import AuthenticationError

MemoryAlgorithm = Literal["vanilla", "schema"]


@dataclass(frozen=True)
class MemoryAlgorithmBinding:
    add_pipeline: str
    search_pipeline: str


MEMORY_ALGORITHM_REGISTRY: dict[str, MemoryAlgorithmBinding] = {
    "vanilla": MemoryAlgorithmBinding(add_pipeline="vanilla_add", search_pipeline="vanilla"),
    "schema": MemoryAlgorithmBinding(add_pipeline="schema_add", search_pipeline="schema"),
}


def resolve_memory_algorithm(value: str | None) -> MemoryAlgorithm:
    """Validate and return a configured memory algorithm."""

    if not value:
        raise AuthenticationError("memory_algorithm is required", code="auth.memory_algorithm_missing")
    if value not in MEMORY_ALGORITHM_REGISTRY:
        raise AuthenticationError(
            f"unsupported memory_algorithm: {value}",
            code="auth.memory_algorithm_unsupported",
        )
    return value  # type: ignore[return-value]


def binding_for_memory_algorithm(value: str) -> MemoryAlgorithmBinding:
    """Return add/search pipeline names for a memory algorithm."""

    algorithm = resolve_memory_algorithm(value)
    return MEMORY_ALGORITHM_REGISTRY[algorithm]
