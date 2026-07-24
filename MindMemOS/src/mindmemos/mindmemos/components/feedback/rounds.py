"""Conversation round compaction for feedback analysis."""

from __future__ import annotations

from typing import Any


class FeedbackRoundCompactor:
    """Compact raw add-record messages into user-query and final-answer pairs."""

    def compact(self, raw_messages: list[Any]) -> list[dict[str, Any]]:
        messages = [raw for raw in raw_messages if isinstance(raw, dict)]
        user_query = _first_content_message(messages, role="user")
        assistant_summary = _last_content_message(messages, role="assistant")
        return [message for message in (user_query, assistant_summary) if message is not None]


def _first_content_message(messages: list[dict[str, Any]], *, role: str) -> dict[str, Any] | None:
    for message in messages:
        if message.get("role") == role and _has_text_content(message):
            return _compact_message(message)
    return None


def _last_content_message(messages: list[dict[str, Any]], *, role: str) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == role and _has_text_content(message):
            return _compact_message(message)
    return None


def _has_text_content(message: dict[str, Any]) -> bool:
    return isinstance(message.get("content"), str) and bool(message["content"].strip())


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    compact = {"role": message.get("role"), "content": message.get("content")}
    if "timestamp" in message:
        compact["timestamp"] = message["timestamp"]
    return compact
