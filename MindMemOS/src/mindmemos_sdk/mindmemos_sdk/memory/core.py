"""Shared request/response core for sync and async memory clients."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..errors import MindMemOSSDKError
from ..transport import Envelope
from .models import (
    AddMode,
    AddResult,
    FeedbackMode,
    GetResult,
    MemorySearchHit,
    Message,
    SearchResult,
    SearchStrategy,
    StatusResult,
    build_add_body,
    build_delete_body,
    build_dreaming_body,
    build_feedback_body,
    build_get_body,
    build_search_body,
    build_update_body,
)

T = TypeVar("T")


@dataclass(frozen=True)
class MemoryDefaults:
    user_id: str | None = None
    app_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class MemoryRequest(Generic[T]):
    path: str
    body: dict[str, Any]
    parse: Callable[[Envelope], T]


class MemoryCore:
    """Build memory API requests and parse envelopes."""

    def __init__(self, defaults: MemoryDefaults | None = None) -> None:
        self._defaults = defaults or MemoryDefaults()

    def _resolve_user_id(self, user_id: str | None) -> str:
        resolved = user_id or self._defaults.user_id
        if not resolved:
            raise MindMemOSSDKError(
                "user_id is required: pass user_id=... or configure a default via `mindmemos auth`."
            )
        return resolved

    def add(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        user_id: str | None = None,
        mode: AddMode = "sync",
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_context: list[Any] | None = None,
        score: float | None = None,
        task_id: str | None = None,
    ) -> MemoryRequest[AddResult]:
        if not messages:
            raise MindMemOSSDKError("`messages` must be a non-empty list.")
        return MemoryRequest(
            path="/v1/memory/add",
            body=build_add_body(
                user_id=self._resolve_user_id(user_id),
                messages=messages,
                mode=mode,
                app_id=app_id or self._defaults.app_id,
                agent_id=agent_id or self._defaults.agent_id,
                session_id=session_id or self._defaults.session_id,
                metadata=metadata,
                skill_context=skill_context,
                score=score,
                task_id=task_id,
            ),
            parse=parse_add_result,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = 10,
        user_id: str | None = None,
        search_strategy: SearchStrategy = "fast",
        rerank: bool = False,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRequest[SearchResult]:
        return MemoryRequest(
            path="/v1/memory/search",
            body=build_search_body(
                user_id=self._resolve_user_id(user_id),
                query=query,
                top_k=top_k,
                search_strategy=search_strategy,
                rerank=rerank,
                score_threshold=score_threshold,
                filters=filters,
                app_id=app_id or self._defaults.app_id,
                agent_id=agent_id or self._defaults.agent_id,
                session_id=session_id or self._defaults.session_id,
            ),
            parse=parse_search_result,
        )

    def get(
        self,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> MemoryRequest[GetResult]:
        return MemoryRequest(
            path="/v1/memory/get",
            body=build_get_body(filters=filters, top_k=top_k),
            parse=parse_get_result,
        )

    def update(self, memory_id: str, content: str) -> MemoryRequest[StatusResult]:
        return MemoryRequest(
            path="/v1/memory/update",
            body=build_update_body(memory_id=memory_id, content=content),
            parse=parse_status_result,
        )

    def delete(self, memory_id: str) -> MemoryRequest[StatusResult]:
        return MemoryRequest(
            path="/v1/memory/delete",
            body=build_delete_body(memory_id=memory_id),
            parse=parse_status_result,
        )

    def feedback(
        self,
        *,
        feedback: str | None = None,
        mode: FeedbackMode | None = None,
        messages: list[Message | dict[str, Any]] | None = None,
        recalled_memories: list[MemorySearchHit | dict[str, Any]] | None = None,
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRequest[StatusResult]:
        return MemoryRequest(
            path="/v1/memory/feedback",
            body=build_feedback_body(
                feedback=feedback,
                mode=mode,
                messages=messages,
                recalled_memories=recalled_memories,
                user_id=user_id or self._defaults.user_id,
                app_id=app_id or self._defaults.app_id,
                agent_id=agent_id or self._defaults.agent_id,
                session_id=session_id or self._defaults.session_id,
            ),
            parse=parse_status_result,
        )

    def dreaming(
        self,
        *,
        mode: AddMode = "async",
        user_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRequest[StatusResult]:
        return MemoryRequest(
            path="/v1/memory/dreaming",
            body=build_dreaming_body(
                mode=mode,
                user_id=user_id or self._defaults.user_id,
                app_id=app_id or self._defaults.app_id,
                agent_id=agent_id or self._defaults.agent_id,
                session_id=session_id or self._defaults.session_id,
            ),
            parse=parse_status_result,
        )


def parse_add_result(envelope: Envelope) -> AddResult:
    data = envelope.data or {}
    return AddResult(
        code=envelope.code,
        request_id=envelope.request_id,
        memories=data.get("memories", []),
    )


def parse_search_result(envelope: Envelope) -> SearchResult:
    data = envelope.data or {}
    return SearchResult(
        request_id=envelope.request_id,
        memories=data.get("memories", []),
    )


def parse_get_result(envelope: Envelope) -> GetResult:
    data = envelope.data or {}
    return GetResult(
        request_id=envelope.request_id,
        memories=data.get("memories", []),
    )


def parse_status_result(envelope: Envelope) -> StatusResult:
    return StatusResult(
        code=envelope.code,
        request_id=envelope.request_id,
        message=envelope.message,
    )
