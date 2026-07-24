"""Kafka message serialization and consumed-message wrappers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def serialize_value(value: Any) -> bytes:
    """Serialize a Kafka message value to bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json().encode("utf-8")
    return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")


@dataclass
class ConsumedMessage:
    """Message view passed to business handlers."""

    topic: str
    partition: int
    offset: int
    key: str | None
    value: bytes
    headers: dict[str, str] = field(default_factory=dict)
    timestamp_ms: int | None = None

    def json(self) -> Any:
        """Parse the consumed message value as UTF-8 JSON."""
        return json.loads(self.value.decode("utf-8"))

    def text(self) -> str:
        """Parse the consumed message value as UTF-8 text."""
        return self.value.decode("utf-8")
