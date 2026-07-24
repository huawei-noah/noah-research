"""Activity-log reading DTOs shared by dreaming / feedback pipelines.

This module only describes the *result* of reading the ``add_record_v1`` and
``search_record_v1`` audit collections. It must not carry dreaming / feedback
algorithm decisions (e.g. whether a memory should be consolidated): the
``RecentActivityCollector`` component is a read-only producer of raw material,
and downstream pipelines apply their own judgment on top of these structures.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .memory import (
    DialogueMessage,
    FileMessage,
    TextMessage,
    UrlMessage,
)

ActivityMessage = DialogueMessage | UrlMessage | FileMessage | TextMessage


class ActivityScope(BaseModel):
    """Purpose: Describe the project-scoped slice of activity logs to collect.

    Used in: ``RecentActivityCollector`` input. ``project_id`` is mandatory and
    force-injected into the Qdrant filter (hard isolation); the optional fields
    further narrow the scope to a user / session / agent / app.
    """

    project_id: str
    account_id: str | None = None
    api_key_uuid: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None


class ActivityRecordContext(BaseModel):
    """Purpose: Carry per-record audit identity for grouping and isolation.

    Used in: ``SearchActivityEvent`` and conversation grouping. Mirrors the
    context fields persisted on each add / search record payload.
    """

    project_id: str
    request_id: str | None = None
    account_id: str | None = None
    api_key_uuid: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None


class MemoryRef(BaseModel):
    """Purpose: Lightweight reference to a memory recalled by one search request.

    Used in: ``SearchActivityEvent.recalled_memories`` and
    ``RecentActivityBundle.recalled_memories``. This is a redundant pointer
    stored on the search record; it does NOT hydrate ``memory_item_v1``.
    Downstream code that needs a full ``Memory`` re-fetches it by ``memory_id``.
    """

    memory_id: str
    content: str | None = None
    search_record_id: str
    rank: int | None = None
    score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WrittenMemoryRef(BaseModel):
    """Purpose: Session-level reference to a memory written / updated by add.

    Used in: ``ConversationActivity.written_memories`` and
    ``RecentActivityBundle.written_memories``. The collector only guarantees the
    session-level set relation ("this group of sessions wrote / updated these
    memories"); it does not align a specific memory to a specific add message.
    """

    memory_id: str
    content: str | None = None
    operation: str | None = None
    add_record_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    user_id: str | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    payloads: list[dict[str, Any]] = Field(default_factory=list)


class SearchActivityEvent(BaseModel):
    """Purpose: Preserve the query -> recalled-memory alignment of one search.

    Used in: ``ConversationActivity.search_events`` and
    ``RecentActivityBundle.search_events``. Unlike add records, a search
    ``query`` and its ``recalled_memories`` are the input and output of the same
    retrieval request and must stay one-to-one aligned.
    """

    search_record_id: str
    context: ActivityRecordContext
    occurred_at: datetime | None = None
    query: str
    filters: dict[str, Any] | None = None
    top_k: int | None = None
    deep_search: bool | None = None
    status: str | None = None
    recalled_memories: list[MemoryRef] = Field(default_factory=list)


class AddActivityEvent(BaseModel):
    """Purpose: Preserve one add-record event inside an activity conversation.

    Used in: feedback pipelines that need add-record identity, compactable
    messages, and write payloads after the shared collector has grouped recent
    activity by conversation.
    """

    add_record_id: str
    context: ActivityRecordContext
    occurred_at: datetime | None = None
    completed_at: datetime | None = None
    status: str | None = None
    messages: list[ActivityMessage] = Field(default_factory=list)
    memory_payloads: list[dict[str, Any]] = Field(default_factory=list)


class ConversationActivity(BaseModel):
    """Purpose: One conversation worth of activity, grouped by ``session_id``.

    Used in: ``RecentActivityBundle.conversations``. Records without a
    ``session_id`` degrade to "one record == one conversation". By convention a
    search usually precedes its add within the same session, so the initial
    recall comes from ``search_events`` and the post-conversation writes come
    from ``written_memories``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None
    occurred_at: datetime | None = None
    last_occurred_at: datetime | None = None
    add_record_ids: list[str] = Field(default_factory=list)
    search_record_ids: list[str] = Field(default_factory=list)
    add_events: list[AddActivityEvent] = Field(default_factory=list)
    messages: list[ActivityMessage] = Field(default_factory=list)
    search_events: list[SearchActivityEvent] = Field(default_factory=list)
    written_memories: list[WrittenMemoryRef] = Field(default_factory=list)
    written_memory_ids: list[str] = Field(default_factory=list)
    feedback_add_record_ids: list[str] = Field(default_factory=list)
    dreaming_add_record_ids: list[str] = Field(default_factory=list)


class RecentActivityBundle(BaseModel):
    """Purpose: Sole output of ``RecentActivityCollector``.

    Used in: dreaming / feedback pipelines. Carries the session aggregation
    layer (``conversations``) plus the search event layer and globally
    de-duplicated memory references for the collected time window.
    """

    window_start: datetime
    window_end: datetime
    scope: ActivityScope
    conversations: list[ConversationActivity] = Field(default_factory=list)
    search_events: list[SearchActivityEvent] = Field(default_factory=list)
    recalled_memories: list[MemoryRef] = Field(default_factory=list)
    written_memories: list[WrittenMemoryRef] = Field(default_factory=list)
