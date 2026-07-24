"""Request/response models for the ``/v1/memory/*`` resource.

These mirror the backend HTTP contract (``mindmemos.api.schemas``) but stay in the
SDK namespace so callers depend on the SDK, not the server package. Response models
use ``extra="ignore"`` so new server fields never break older SDK versions.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

AddMode = Literal["sync", "async"]
FeedbackMode = Literal["sync", "async"]
SearchStrategy = Literal["fast", "agentic"]


class TextMessage(BaseModel):
    """A free-form text snippet to add as memory.

    Mirrors :class:`mindmemos.typing.memory.TextMessage`.
    """

    text: str


class DialogueMessage(BaseModel):
    """One conversational turn to add as memory.

    Mirrors :class:`mindmemos.typing.memory.DialogueMessage`.
    """

    role: str
    content: str
    timestamp: int | None = None
    """Dialogue timestamp in 13-digit milliseconds. Leave unset when unavailable."""


class UrlMessage(BaseModel):
    """A URL reference to add as memory.

    Mirrors :class:`mindmemos.typing.memory.UrlMessage`.
    """

    url: str


class FileMessage(BaseModel):
    """An uploaded or referenced file attachment to add as memory.

    Mirrors :class:`mindmemos.typing.memory.FileMessage`. ``file_type`` may be left
    empty; the server derives it from ``file_path``.
    """

    file_name: str
    file_path: str
    file_type: str = ""


# Union of all message types accepted by ``POST /v1/memory/add``, matching the
# server-side ``AddRequest.messages`` contract.
Message = Union[DialogueMessage, UrlMessage, FileMessage, TextMessage]


class MemoryAddItem(BaseModel):
    """One memory produced by an add operation."""

    model_config = ConfigDict(extra="ignore")

    operation: str
    content: str
    memory_id: str | None = None
    mem_type: str | None = None
    confidence: float | None = None
    related_memory_ids: list[str] = Field(default_factory=list)
    graph_edge_count: int = 0


class AddResult(BaseModel):
    """Typed result of ``MemoryClient.add``.

    ``code`` carries the envelope outcome (``ok`` for sync, ``queued`` for async).
    ``memories`` is empty in async mode.
    """

    model_config = ConfigDict(extra="ignore")

    code: str = "ok"
    request_id: str | None = None
    memories: list[MemoryAddItem] = Field(default_factory=list)


class MemoryLineage(BaseModel):
    """Lineage metadata populated by vanilla search."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["current", "archived"] = "current"
    derived_from_memory_ids: list[str] = Field(default_factory=list)
    """Ancestor memory IDs, newest first."""
    derived_to_memory_ids: list[str] = Field(default_factory=list)


class MemorySearchHit(BaseModel):
    """One memory returned by a search/get query."""

    model_config = ConfigDict(extra="ignore")

    id: str
    memory: str
    memory_type: str = "fact"
    last_update_at: str | None = None
    event_time: str | None = None
    source_timestamp: str | None = None
    lineage: MemoryLineage | None = None


class SearchResult(BaseModel):
    """Typed result of ``MemoryClient.search``."""

    model_config = ConfigDict(extra="ignore")

    request_id: str | None = None
    memories: list[MemorySearchHit] = Field(default_factory=list)


class GetResult(BaseModel):
    """Typed result of ``MemoryClient.get``.

    Same shape as :class:`SearchResult`: a list of memories scoped to the current
    project. Unlike search, ``get`` takes no query; it lists or filters memories.
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str | None = None
    memories: list[MemorySearchHit] = Field(default_factory=list)


class StatusResult(BaseModel):
    """Typed result of the status-only memory ops (update / delete / feedback /
    dreaming).

    These endpoints return no domain data: the envelope carries only ``code`` and
    an optional ``message``. The transport already turns a non-success envelope
    into :class:`~mindmemos_sdk.errors.ApiError`, so a returned ``StatusResult``
    means the server accepted the operation; ``code`` is normally ``"ok"`` and
    ``message`` is usually empty.
    """

    model_config = ConfigDict(extra="ignore")

    code: str = "ok"
    request_id: str | None = None
    message: str = ""


def serialize_messages(messages: list[Message | dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize typed or raw message objects for request bodies."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, BaseModel):
            result.append(msg.model_dump())
        else:
            result.append(msg)
    return result


def build_add_body(
    *,
    user_id: str,
    messages: list[Message | dict[str, Any]],
    mode: AddMode = "sync",
    app_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    skill_context: list[BaseModel | dict[str, Any]] | None = None,
    score: float | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Build a memory add request body without empty optional fields."""
    body: dict[str, Any] = {"user_id": user_id, "messages": serialize_messages(messages), "mode": mode}
    if app_id:
        body["app_id"] = app_id
    if agent_id:
        body["agent_id"] = agent_id
    if session_id:
        body["session_id"] = session_id
    if metadata:
        body["metadata"] = metadata
    if skill_context:
        body["skill_context"] = [
            item.model_dump(mode="json", exclude_none=True) if isinstance(item, BaseModel) else item
            for item in skill_context
        ]
    if score is not None:
        body["score"] = score
    if task_id is not None:
        body["task_id"] = task_id
    return body


def build_search_body(
    *,
    user_id: str,
    query: str,
    top_k: int | None = 10,
    search_strategy: SearchStrategy = "fast",
    rerank: bool = False,
    score_threshold: float | None = None,
    filters: dict[str, Any] | None = None,
    app_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build a memory search request body without empty optional fields."""
    if not isinstance(rerank, bool):
        raise TypeError("rerank must be a bool")
    body: dict[str, Any] = {
        "user_id": user_id,
        "query": query,
        "search_strategy": search_strategy,
    }
    if top_k is not None:
        body["top_k"] = top_k
    body["rerank"] = rerank
    if score_threshold is not None:
        body["score_threshold"] = score_threshold
    if app_id:
        body["app_id"] = app_id
    if agent_id:
        body["agent_id"] = agent_id
    if session_id:
        body["session_id"] = session_id
    if filters:
        body["filters"] = filters
    return body


def build_get_body(
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Build a memory get request body without empty optional fields."""
    body: dict[str, Any] = {}
    if filters:
        body["filters"] = filters
    if top_k is not None:
        body["top_k"] = top_k
    return body


def build_update_body(*, memory_id: str, content: str) -> dict[str, Any]:
    """Build a memory update request body."""
    return {"memory_id": memory_id, "content": content}


def build_delete_body(*, memory_id: str) -> dict[str, Any]:
    """Build a memory delete request body."""
    return {"memory_id": memory_id}


def build_dreaming_body(
    *,
    mode: AddMode = "async",
    user_id: str | None = None,
    app_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build a memory dreaming request body without empty optional fields."""
    body: dict[str, Any] = {"mode": mode}
    if user_id:
        body["user_id"] = user_id
    if app_id:
        body["app_id"] = app_id
    if agent_id:
        body["agent_id"] = agent_id
    if session_id:
        body["session_id"] = session_id
    return body


def build_feedback_body(
    *,
    feedback: str | None = None,
    mode: FeedbackMode | None = None,
    messages: list[Message | dict[str, Any]] | None = None,
    recalled_memories: list[MemorySearchHit | dict[str, Any]] | None = None,
    user_id: str | None = None,
    app_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build a memory feedback request body."""
    body: dict[str, Any] = {}
    if user_id:
        body["user_id"] = user_id
    if app_id:
        body["app_id"] = app_id
    if agent_id:
        body["agent_id"] = agent_id
    if session_id:
        body["session_id"] = session_id
    if mode is not None:
        body["mode"] = mode
    if feedback is not None:
        body["feedback"] = feedback
    if messages is not None:
        body["messages"] = serialize_messages(messages)
    if recalled_memories is not None:
        body["recalled_memories"] = [
            item.model_dump() if isinstance(item, BaseModel) else item for item in recalled_memories
        ]
    return body
