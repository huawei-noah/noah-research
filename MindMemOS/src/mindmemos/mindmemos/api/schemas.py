"""Shared HTTP API schemas.

Two distinct identity sources exist:

- **Security/isolation context** (``account_id`` / ``project_id`` /
  ``api_key_uuid`` / ``scopes`` / ``request_id``) is resolved from the bearer
  credential by ``api.deps`` and carried in
  :class:`mindmemos.typing.memory.MemoryRequestContext`.
- **Actor identity** (``user_id`` / ``app_id`` / ``session_id`` / ``agent_id``)
  is supplied in the JSON body on endpoints that need it
  (add, search, feedback, dreaming), and
  ``api.mappers`` merges those values into ``MemoryRequestContext``.

Every endpoint has its HTTP request model here, even the actor-less ones
(get / delete / update / feedback / dreaming), so the route → service boundary
is uniform: routes always receive an ``api.schemas`` request model and the
service always converts it to a pipeline input via ``api.mappers``.

The request models here intentionally do NOT inherit the pipeline input DTOs;
the conversion is explicit in ``api.mappers``.
"""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from ..typing import (
    AddMode,
    DialogueMessage,
    FileMessage,
    MemoryAddEventItem,
    MemorySearchItem,
    SkillContext,
    TextMessage,
    UrlMessage,
)

T = TypeVar("T")
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ResolvedKey(BaseModel):
    """Information after resolved api key from local/account system"""

    account_id: NonEmptyStr
    project_id: NonEmptyStr
    api_key_uuid: NonEmptyStr
    memory_algorithm: NonEmptyStr
    scopes: list[str] | None = None
    user_override_config: dict | None = None
    project_override_config: dict | None = None


class AuthContext(BaseModel):
    """Security / isolation context resolved from the bearer credential.

    Produced by ``api.deps`` (``get_request_context`` /
    ``get_internal_request_context``) and checked by ``require_scopes``. It
    carries only trustworthy, request-constant fields — **no actor identity**.
    Actor fields (user_id / app_id / session_id / agent_id) arrive in request
    bodies on add/search/feedback/dreaming and are merged into
    :class:`mindmemos.typing.memory.MemoryRequestContext` by
    ``api.mappers.to_memory_request_context``.
    """

    request_id: NonEmptyStr
    account_id: NonEmptyStr
    project_id: NonEmptyStr
    api_key_uuid: NonEmptyStr
    memory_algorithm: NonEmptyStr
    scopes: list[str] = Field(default_factory=list)


# Actor identity: ``user_id`` is required for add/search/feedback and optional
# for dreaming. It is supplied in the request body. The Qdrant payload schema
# indexes all four fields as plain KEYWORD values with no storage-level not-null
# constraint.


class ActorIdentityRequest(BaseModel):
    """Actor identity supplied by request bodies that scope memory operations."""

    user_id: NonEmptyStr | None = None
    """Business actor user ID. Required by add/search/feedback service methods."""

    app_id: NonEmptyStr | None = None

    session_id: NonEmptyStr | None = None

    agent_id: NonEmptyStr | None = None


class AddRequest(ActorIdentityRequest):
    """HTTP body for ``POST /v1/memory/add``.

    Mirrors the public add fields mapped into
    :class:`mindmemos.typing.service.AddPipelineInput` plus actor identity.
    Manager-only schema-add controls are intentionally excluded. Converted by
    ``api.mappers.to_add_pipeline_input``; actor fields are merged into the
    request context by ``api.mappers.to_memory_request_context``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    messages: list[DialogueMessage | UrlMessage | FileMessage | TextMessage] = Field(min_length=1)
    """Message list supporting dialogue, URL, file, and plain text messages."""

    mode: AddMode = Field(default="sync")
    """Add mode: sync or async."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Business extension metadata."""

    skill_context: list[SkillContext] | None = None
    """Skill references hit in this turn, excluding full text, used for trace binding.

    This is optional for legacy callers. It is consumed only by the service layer
    for skill binding and is not passed into add pipeline input.
    """

    score: float | None = None
    """Optional score for this add trajectory.

    This evaluates uploaded skill trajectories. It is not passed into add pipeline
    input and is written only as a trace annotation in ``add_record_v1`` for skill
    evolution and rollout algorithms.
    """

    task_id: NonEmptyStr | None = None
    """Optional task identifier for the trajectory.

        During multi-run rollouts, multiple trajectories for one task share the same
        ``task_id``. It is not passed into add pipeline input and is written only as
        a trace annotation in ``add_record_v1``.
    """

    @field_validator("messages")
    @classmethod
    def _messages_have_content(cls, messages):
        validate_messages_have_content(messages)
        return messages


class SearchRequest(ActorIdentityRequest):
    """HTTP body for ``POST /v1/memory/search``.

    Mirrors :class:`mindmemos.typing.service.SearchPipelineInput` plus actor
    identity. Converted by
    ``api.mappers.to_search_pipeline_input``; actor fields are merged into the
    request context by ``api.mappers.to_memory_request_context``.
    """

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyStr
    """Search query."""

    filters: dict[str, Any] | None = None
    """Custom structured filter DSL parsed by ``mappers.api.parse_search_dsl``."""

    top_k: int | None = Field(default=10, ge=1)
    """Number of final memories to return. None returns all final candidates."""

    search_strategy: Literal["fast", "agentic"] = "fast"
    """Public search mode: fast single-pass search or agentic multi-round search."""

    rerank: bool = False
    """Whether to rerank final results."""

    score_threshold: float | None = Field(default=None, ge=0, le=1)
    """Minimum rerank relevance score (0–1). Only effective when rerank=True."""

    max_rounds: int = Field(default=3, ge=1)
    """Maximum agentic rounds. Ignored when search_strategy is fast."""


class GetRequest(BaseModel):
    """HTTP body for ``POST /v1/memory/get``.

    Mirrors :class:`mindmemos.typing.service.GetPipelineInput`. No actor identity.
    Converted by ``api.mappers.to_get_pipeline_input``.
    """

    model_config = ConfigDict(extra="forbid")

    filters: dict[str, Any] | None = None
    """Custom structured filter DSL parsed by ``mappers.api.parse_search_dsl``."""

    top_k: int | None = Field(default=None, ge=1)
    """Maximum memories to return. None uses the reader default page size."""


class DeleteRequest(BaseModel):
    """HTTP body for ``POST /v1/memory/delete``.

    Mirrors :class:`mindmemos.typing.service.DeletePipelineInput`. No actor identity.
    Converted by ``api.mappers.to_delete_pipeline_input``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: NonEmptyStr = Field(alias="memory_id")
    """Memory ID"""


class UpdateRequest(BaseModel):
    """HTTP body for ``POST /v1/memory/update``.

    Mirrors :class:`mindmemos.typing.service.UpdatePipelineInput`. No actor identity.
    Converted by ``api.mappers.to_update_pipeline_input``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: NonEmptyStr = Field(alias="memory_id")
    """Memory ID"""

    content: NonEmptyStr
    """Replacement content for the specified memory id."""


class FeedbackRequest(ActorIdentityRequest):
    """HTTP body for ``POST /v1/memory/feedback``.

    Mirrors :class:`mindmemos.typing.service.FeedbackPipelineInput` plus actor
    identity. Converted by ``api.mappers.to_feedback_pipeline_input``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    feedback: NonEmptyStr | None = None
    """User feedback text. When None, the pipeline analyzes recent add records itself."""

    messages: list[DialogueMessage | UrlMessage | FileMessage | TextMessage] = Field(default_factory=list)
    """Full dialogue context for explicit feedback. May be empty for implicit feedback."""

    recalled_memories: list[MemorySearchItem] = Field(default_factory=list)
    """Memories retrieved during explicit feedback. May be empty; the planner decides whether to search."""

    mode: Literal["sync", "async"] = "sync"
    """Feedback execution mode; sync runs inline, async queues a Kafka task."""

    @field_validator("messages")
    @classmethod
    def _messages_have_content(cls, messages):
        validate_messages_have_content(messages)
        return messages


class DreamingRequest(ActorIdentityRequest):
    """HTTP body for ``POST /v1/memory/dreaming``.

    Mirrors :class:`mindmemos.typing.service.DreamingPipelineInput` plus actor
    identity. Converted by ``api.mappers.to_dreaming_pipeline_input``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: AddMode = Field(default="async")
    """Execution mode. ``async`` queues Kafka work; ``sync`` runs consolidation inline."""


class AddData(BaseModel):
    """``data`` payload for ``POST /v1/memory/add``.

    Domain data only. The pipeline's ``status`` (ok / queued / error) is lifted
    to ``ApiResponse.code`` by ``api.mappers``; async mode returns an empty
    ``memories`` list with envelope ``code="queued"``.
    """

    memories: list[MemoryAddEventItem] = Field(default_factory=list)


class MemoryListData(BaseModel):
    """``data`` payload for ``POST /v1/memory/search`` and ``/get``.

    Domain data only; ``status`` / ``message`` are lifted to the envelope.
    """

    memories: list[MemorySearchItem] = Field(default_factory=list)


class ApiResponse(BaseModel, Generic[T]):
    """Unified response envelope.

    ``code`` carries the business outcome (``"ok"`` on success, ``"queued"`` for
    async add, ``"error"`` for soft failures, or an exception code from the
    global handler). ``data`` carries domain data only — pipeline status/message
    are flattened into ``code`` / ``message`` at the HTTP boundary.
    """

    code: str = "ok"
    message: str = ""
    request_id: str | None = None
    data: T | None = None


def validate_messages_have_content(messages) -> None:
    for index, message in enumerate(messages):
        text = (
            getattr(message, "content", None)
            or getattr(message, "text", None)
            or getattr(message, "url", None)
            or getattr(message, "file_path", None)
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"messages[{index}] must contain non-empty content")
