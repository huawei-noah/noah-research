"""Pipeline I/O contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from .memory import (
    AddMode,
    DialogueMessage,
    FileMessage,
    MemoryOperation,
    MemoryType,
    TextMessage,
    UrlMessage,
)

ServiceResultStatus = Literal["ok", "error", "queued"]
SearchPipelineStrategy = Literal["default", "vanilla", "schema"]


def _utc_millis() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def utc_datetime_from_millis(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC)


def utc_millis_from_datetime(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.astimezone(UTC).timestamp() * 1000)


def last_message_timestamp(messages: list[DialogueMessage | UrlMessage | FileMessage | TextMessage]) -> int | None:
    latest: int | None = None
    for message in messages:
        timestamp = getattr(message, "timestamp", None)
        if isinstance(timestamp, int) and timestamp > 0:
            latest = timestamp
    return latest


class MemoryAddEventItem(BaseModel):
    operation: MemoryOperation
    """Memory operation type."""

    content: str
    """Memory content."""

    memory_id: str | None = None
    """Memory ID."""

    mem_type: MemoryType | None = None
    """Memory type."""

    memory_type: MemoryType | str | None = None
    """Standard displayed memory type."""

    confidence: float | None = None
    """Extraction confidence."""

    related_memory_ids: list[str] = Field(default_factory=list)
    """Related memory IDs."""

    graph_edge_count: int = 0
    """Number of graph edges produced by this memory operation."""

    @model_validator(mode="after")
    def _fill_memory_type(self) -> "MemoryAddEventItem":
        """Keep the display field aligned with the stored memory type."""

        if self.memory_type is None and self.mem_type is not None:
            self.memory_type = self.mem_type
        return self


class MemoryLineage(BaseModel):
    """Lineage metadata for a returned memory."""

    role: Literal["current", "archived"] = "current"
    """Whether this returned memory is the current version or an archived version."""

    derived_from_memory_ids: list[str] = Field(default_factory=list)
    """Ancestor memory IDs reached through ``DERIVED_FROM`` graph edges, newest first."""

    derived_to_memory_ids: list[str] = Field(default_factory=list)
    """Current memory IDs that descend from this archived memory."""


class MemorySearchItem(BaseModel):
    """Memory item returned by search and read APIs."""

    id: str
    """Memory ID."""

    memory: str
    """Memory content."""

    memory_type: MemoryType | str = "fact"
    """Standard displayed memory type."""

    last_update_at: str
    """Latest create/update time formatted as %Y-%m-%d %H:%M:%S."""

    event_time: str | None = None
    """Business event time formatted as %Y-%m-%d %H:%M:%S; distinct from DB update time."""

    source_timestamp: str | None = None
    """Original source time formatted as %Y-%m-%d %H:%M:%S for temporal QA."""

    lineage: MemoryLineage | None = None
    """Version lineage metadata populated by vanilla search."""


class AddPipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    messages: list[DialogueMessage | UrlMessage | FileMessage | TextMessage] = Field(default_factory=list)
    """Input messages; dialogue, URL, file, and plain text messages may be mixed."""

    event_timestamp_ms: int = Field(
        default_factory=_utc_millis,
        validation_alias=AliasChoices("event_timestamp_ms", "event_timestamp", "timestamp"),
        serialization_alias="event_timestamp_ms",
    )
    """Request-level event time in UTC milliseconds.

    This field is kept as the legacy request-level occurrence timestamp. System
    add/queue time is generated at the persistence boundary instead.
    """

    mode: AddMode = Field(default="sync")
    """Add execution mode: sync or async."""

    force_generation: bool = False
    """Force buffered messages to be split and generated immediately when supported."""

    metadata: dict = Field(default_factory=dict)
    """Business extension metadata."""

    @property
    def timestamp(self) -> int:
        return self.event_timestamp_ms

    @property
    def timestamp_utc(self) -> datetime:
        return utc_datetime_from_millis(self.event_timestamp_ms)

    @property
    def event_timestamp(self) -> int:
        if "event_timestamp_ms" in self.model_fields_set:
            return self.event_timestamp_ms
        return last_message_timestamp(self.messages) or self.event_timestamp_ms

    @property
    def event_timestamp_utc(self) -> datetime:
        return utc_datetime_from_millis(self.event_timestamp)


class AddPipelineSyncResult(BaseModel):
    status: ServiceResultStatus
    """Service completion status."""

    memories: list[MemoryAddEventItem] = Field(default_factory=list)
    """Generated memory events."""


class AddPipelineAsyncResult(BaseModel):
    status: ServiceResultStatus = "queued"
    """Service completion status."""

    memories: list[MemoryAddEventItem] = Field(default_factory=list)
    """Always empty in async mode; workers produce the actual write results."""


class SearchPipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    """Search query."""

    filters: dict[str, Any] | None = None
    """Public filter DSL parsed into an internal SearchFilter."""

    top_k: int | None = Field(default=10, ge=1)
    """Maximum number of memories to return; None returns all final candidates."""

    search_pipeline: SearchPipelineStrategy = "default"
    """Internal search engine key used by search_pipeline before optional agentic orchestration."""

    rerank: bool = False
    """Whether to rerank final candidates before applying top_k."""

    score_threshold: float | None = None
    """Minimum rerank relevance score (0–1). Only effective when rerank=True."""

    agentic: bool = False
    """Whether to wrap the selected search pipeline in multi-round orchestration."""

    max_rounds: int = 3
    """Request-level maximum agentic rounds; ignored when agentic is false."""

    include_patches: bool = True
    """Deprecated compatibility flag; vanilla search owns archived-version lineage recall."""


class SearchPipelineResult(BaseModel):
    status: ServiceResultStatus
    """Service completion status."""

    memories: list[MemorySearchItem]
    """Returned memories."""


class GetPipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: dict[str, Any] | None = None
    """Public filter DSL; None returns project-scoped memories."""

    top_k: int | None = None
    """Maximum number of memories to return; None uses the reader default."""


class GetPipelineResult(BaseModel):
    status: ServiceResultStatus
    """Service completion status."""

    memories: list[MemorySearchItem]
    """Returned memories."""

    message: str | None = None
    """Failure reason when get cannot be completed."""


class DeletePipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(alias="memory_id")
    """Memory ID."""


class DeletePipelineResult(BaseModel):
    status: ServiceResultStatus
    """Delete status."""

    message: str | None = None
    """Failure reason when delete cannot be completed."""


class UpdatePipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(alias="memory_id")
    """Memory ID."""

    content: str
    """New content for the target memory."""


class UpdatePipelineResult(BaseModel):
    status: ServiceResultStatus
    """Update status."""

    message: str | None = None
    """Failure reason when the update cannot be applied."""


class FeedbackPipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    feedback: str | None = None
    """User feedback text; if None, analyze recent add records from the database."""

    messages: list[DialogueMessage | UrlMessage | FileMessage | TextMessage] = Field(default_factory=list)
    """Full conversation context for explicit feedback, supplied by the caller."""

    recalled_memories: list[MemorySearchItem] = Field(default_factory=list)
    """Memories actually recalled in the explicit feedback round, supplied by the caller."""

    mode: Literal["sync", "async"] = "sync"
    """Feedback execution mode; async queues work for a Kafka worker."""


class FeedbackAddAction(BaseModel):
    action: Literal["add"] = "add"
    """Create a new memory from durable feedback."""

    result_memory_id: str | None = None
    """Memory id after action execution when the add pipeline can provide it."""

    after_content: str
    """New memory content to add."""

    reason: str | None = None
    """Reason for the action."""

    status: ServiceResultStatus = "ok"
    """Execution status for this action."""


class FeedbackUpdateAction(BaseModel):
    action: Literal["update"] = "update"
    """Update an existing memory with corrected or supplemented content."""

    target_memory_id: str
    """Existing memory id to update."""

    result_memory_id: str | None = None
    """Memory id after action execution; equals target_memory_id for in-place updates."""

    before_content: str
    """Content before the change."""

    after_content: str
    """Content after the change."""

    reason: str | None = None
    """Reason for the action."""

    status: ServiceResultStatus = "ok"
    """Execution status for this action."""


class FeedbackDeleteAction(BaseModel):
    action: Literal["delete"] = "delete"
    """Delete an existing stale or wrong memory."""

    target_memory_id: str
    """Existing memory id to delete."""

    result_memory_id: str | None = None
    """Deleted memory id after action execution."""

    before_content: str
    """Content before the deletion."""

    reason: str | None = None
    """Reason for the action."""

    status: ServiceResultStatus = "ok"
    """Execution status for this action."""


class FeedbackNoopAction(BaseModel):
    action: Literal["noop"] = "noop"
    """Record that no memory mutation is needed."""

    target_memory_id: str | None = None
    """Existing memory id that was inspected, if any."""

    before_content: str | None = None
    """Existing memory content that was inspected, if any."""

    reason: str | None = None
    """Reason no mutation is needed."""

    status: ServiceResultStatus = "ok"
    """Execution status for this action."""


FeedbackActionResult = Annotated[
    FeedbackAddAction | FeedbackUpdateAction | FeedbackDeleteAction | FeedbackNoopAction,
    Field(discriminator="action"),
]


class FeedbackPipelineResult(BaseModel):
    status: ServiceResultStatus
    """Service completion status."""

    message: str | None = None
    """Feedback message."""

    actions: list[FeedbackActionResult] = Field(default_factory=list)
    """Executed feedback action results."""


class DreamingPipelineInput(BaseModel):
    """Dreaming pipeline request payload."""

    mode: AddMode = Field(default="async")
    """Execution mode. ``async`` queues work; ``sync`` runs consolidation inline."""


class DreamingPipelineResult(BaseModel):
    status: ServiceResultStatus
    """Service completion status."""

    message: str | None = None
    """Dreaming message."""
