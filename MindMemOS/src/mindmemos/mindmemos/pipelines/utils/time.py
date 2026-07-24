"""Shared time formatting helpers for pipeline response DTOs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def format_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return format_datetime(value)


def format_source_timestamp(memory: Any) -> str | None:
    if memory.validate_from is not None:
        return format_datetime(memory.validate_from)
    millis = memory.metadata.get("source_timestamp_ms")
    if not isinstance(millis, int | float):
        return None
    return format_datetime(datetime.fromtimestamp(millis / 1000, tz=UTC))


def format_memory_event_time(memory: Any, *, fallback_to_source_timestamp: bool = False) -> str | None:
    resolved = resolved_event_datetime(memory.metadata)
    if resolved is not None:
        return format_datetime(resolved)
    if fallback_to_source_timestamp:
        return format_source_timestamp(memory)
    return format_optional_datetime(memory.validate_from)


def resolved_event_datetime(metadata: dict) -> datetime | None:
    value = metadata.get("resolved_event_datetime")
    parsed = _parse_datetime_value(value)
    if parsed is not None:
        return parsed

    value = metadata.get("resolved_event_date")
    parsed = _parse_date_value(value)
    if parsed is not None:
        return parsed

    value = metadata.get("resolved_event_range")
    if isinstance(value, list) and value:
        return _parse_date_value(value[0])
    return None


def _parse_datetime_value(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if "T" not in normalized and len(normalized) == 19:
        normalized = normalized.replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _parse_date_value(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return None
