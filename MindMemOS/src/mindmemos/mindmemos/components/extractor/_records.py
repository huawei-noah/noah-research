"""Operators that turn durable add records into episode extraction inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from ...logging import get_logger
from ...typing import MemoryRequestContext

logger = get_logger(__name__)
_STANDARD_DIALOGUE_ROLES = {"user", "assistant", "system", "tool"}
_NON_DIALOGUE_ENTRY_LABELS = {"text", "url", "file", "unknown"}


class AddRecordLike(Protocol):
    add_record_id: str
    payload: dict[str, Any]


def to_chunker_entries(records: list[AddRecordLike]) -> list[dict[str, Any]]:
    """Convert add-buffer records into chunker entries.

    Args:
        records: Durable add-buffer records.

    Returns:
        Entry dicts containing content, speaker, timestamp, and add_record_id.
    """
    return [
        {
            "content": content,
            "speaker": speaker,
            "timestamp": timestamp,
            "add_record_id": record.add_record_id,
        }
        for record in records
        for content, speaker, timestamp in [_record_message_fields(record)]
    ]


def force_generation(records: list[AddRecordLike]) -> bool:
    """Return whether the last record forces episode generation.

    Args:
        records: Add-buffer records.

    Returns:
        True when the last record payload has ``force_generation=True``.
    """
    return bool(records and records[-1].payload.get("force_generation"))


def to_conversation_text(records: list[AddRecordLike]) -> str:
    """Format add-buffer records as indexed conversation text for extraction.

    Args:
        records: Add-buffer records.

    Returns:
        Conversation text such as ``0. 2024-01-01 user: hello``.
    """
    lines: list[str] = []
    for index, record in enumerate(records):
        content, speaker, timestamp = _record_message_fields(record)
        if content.strip():
            lines.append(f"{index}. {timestamp} {_conversation_speaker_label(speaker)}: {content}")
    return "\n".join(lines)


def records_datetime(records: list[AddRecordLike]) -> datetime:
    """Extract the conversation event time from record payloads.

    Args:
        records: Add-buffer records.

    Returns:
        Parsed UTC datetime, or the current time when no record has a valid timestamp.
    """
    for record in records:
        parsed = _event_datetime(record.payload) or _added_datetime(record.payload)
        if parsed is not None:
            return parsed
    return datetime.now(UTC)


def records_added_datetime(records: list[AddRecordLike]) -> datetime:
    """Return the first server-side add time from the record list."""

    for record in records:
        parsed = _added_datetime(record.payload)
        if parsed is not None:
            return parsed
    return datetime.now(UTC)


def metadata(records: list[AddRecordLike]) -> dict[str, Any]:
    """Collect add-buffer metadata for memory writes.

    Args:
        records: Add-buffer records.

    Returns:
        Metadata containing add_record_ids and per-record metadata.
    """
    return {
        "add_record_ids": [record.add_record_id for record in records],
        "record_metadata": [record.payload.get("metadata") or {} for record in records],
    }


def context(records: list[AddRecordLike], fallback: MemoryRequestContext) -> MemoryRequestContext:
    """Restore request context from add-buffer records.

    Args:
        records: Add-buffer records.
        fallback: Context to use when records cannot be parsed.

    Returns:
        Restored memory request context.
    """
    for record in records:
        payload = record.payload
        if not payload:
            continue
        try:
            return MemoryRequestContext.model_validate(
                {
                    "request_id": payload.get("request_id") or fallback.request_id,
                    "account_id": payload.get("account_id") or fallback.account_id,
                    "project_id": payload.get("project_id") or fallback.project_id,
                    "api_key_uuid": payload.get("api_key_uuid") or fallback.api_key_uuid,
                    "user_id": payload.get("user_id") or fallback.user_id,
                    "app_id": payload.get("app_id") or fallback.app_id,
                    "session_id": payload.get("session_id") or fallback.session_id,
                    "agent_id": payload.get("agent_id") or fallback.agent_id,
                }
            )
        except Exception:
            logger.warning("failed to restore context from add buffer record; using drain context", exc_info=True)
            break
    return fallback


def dialogue_timestamp(value: datetime) -> str:
    """Format a conversation timestamp for prompts.

    Args:
        value: UTC datetime.

    Returns:
        A string such as ``2024-01-15 10:30:00 (Monday)``.
    """
    return f"{value.strftime('%Y-%m-%d %H:%M:%S')} ({value.strftime('%A')})"


def _record_message_fields(record: AddRecordLike) -> tuple[str, str, str]:
    messages = record.payload.get("messages") or []
    message = messages[0] if messages and isinstance(messages[0], dict) else {}
    if "role" in message and "content" in message:
        return str(message.get("content") or ""), str(message.get("role") or "unknown"), _payload_time(record.payload)
    if "text" in message:
        return str(message.get("text") or ""), "text", _payload_time(record.payload)
    if "url" in message:
        return str(message.get("url") or ""), "url", _payload_time(record.payload)
    if "file_name" in message or "file_path" in message:
        content = f"{message.get('file_name') or 'file'} ({message.get('file_path') or ''})"
        return content, "file", _payload_time(record.payload)
    return "", "unknown", _payload_time(record.payload)


def _conversation_speaker_label(speaker: str) -> str:
    normalized = speaker.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _STANDARD_DIALOGUE_ROLES | _NON_DIALOGUE_ENTRY_LABELS:
        return normalized
    return f"speaker={speaker}"


def _payload_time(payload: dict[str, Any]) -> str:
    parsed = _event_datetime(payload) or _added_datetime(payload)
    if parsed is not None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _event_datetime(payload: dict[str, Any]) -> datetime | None:
    for key in ("event_timestamp_ms", "timestamp"):
        raw_ts = payload.get(key)
        if isinstance(raw_ts, int) and raw_ts > 0:
            return _timestamp_from_ms(raw_ts)
    message_time = _message_timestamp(payload)
    if message_time is not None:
        return message_time
    return _parse_timestamp(payload.get("event_time"))


def _added_datetime(payload: dict[str, Any]) -> datetime | None:
    for key in ("added_at", "buffered_at", "request_submitted_at"):
        parsed = _parse_timestamp(payload.get(key))
        if parsed is not None:
            return parsed
    raw_ts = payload.get("added_timestamp_ms")
    if isinstance(raw_ts, int) and raw_ts > 0:
        return _timestamp_from_ms(raw_ts)
    return None


def _message_timestamp(payload: dict[str, Any]) -> datetime | None:
    messages = payload.get("messages") or []
    message = messages[0] if messages and isinstance(messages[0], dict) else {}
    raw_ts = message.get("timestamp")
    if isinstance(raw_ts, int) and raw_ts > 0:
        return _timestamp_from_ms(raw_ts)
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str):
        try:
            return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            try:
                return _as_utc(datetime.strptime(value, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                return None
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _timestamp_from_ms(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC)
