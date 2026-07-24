"""Adapters between HTTP request models and pipeline contracts.

Actor identity (user_id / app_id / session_id / agent_id) arrives from request
bodies on add/search/feedback/dreaming. These helpers split each request model
into:

1. the pure business pipeline input (``mindmemos.typing.service``), and
2. a :class:`~mindmemos.typing.memory.MemoryRequestContext` assembled from the
   security-only :class:`~mindmemos.api.schemas.AuthContext` plus actor fields
   (``to_memory_request_context``).

Routes stay thin; the split happens in the service layer. Request models do not
inherit the pipeline input DTOs, so the field mapping is made explicit here.
"""

from __future__ import annotations

from ..config import get_config
from ..errors import BadRequestError
from ..typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    DeletePipelineInput,
    DeletePipelineResult,
    DreamingPipelineInput,
    DreamingPipelineResult,
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    GetPipelineInput,
    GetPipelineResult,
    MemoryRequestContext,
    SearchPipelineInput,
    SearchPipelineResult,
    UpdatePipelineInput,
    UpdatePipelineResult,
)
from .schemas import (
    AddData,
    AddRequest,
    ApiResponse,
    AuthContext,
    DeleteRequest,
    DreamingRequest,
    FeedbackRequest,
    GetRequest,
    MemoryListData,
    SearchRequest,
    UpdateRequest,
)

# Actor identity fields carried by request models but owned by MemoryRequestContext.
_ACTOR_FIELDS = ("user_id", "app_id", "session_id", "agent_id")
_SKILL_FIELDS = ("skill_context", "score", "task_id")


def to_add_pipeline_input(req: AddRequest) -> AddPipelineInput:
    """Build pure add pipeline input from a public add request.

    Args:
        req: Public add request model.

    Returns:
        Add pipeline input with actor fields and trace annotations removed.
    """

    return AddPipelineInput.model_validate(
        req.model_dump(
            by_alias=True,
            exclude={*_ACTOR_FIELDS, *_SKILL_FIELDS},
            exclude_none=True,
        )
    )


def to_search_pipeline_input(req: SearchRequest, *, search_pipeline: str) -> SearchPipelineInput:
    """Build pure search pipeline input from a public search request."""

    _validate_request_top_k(req.top_k)
    data = req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS) | {"search_strategy"})
    data["search_pipeline"] = search_pipeline
    data["agentic"] = req.search_strategy == "agentic"
    return SearchPipelineInput.model_validate(data)


def _validate_request_top_k(top_k: int | None) -> None:
    if top_k is None:
        return
    top_k_max = get_config().algo_config.search.request_top_k_max
    if top_k > top_k_max:
        raise BadRequestError(
            f"top_k must be <= {top_k_max}; value={top_k}",
            code="search.top_k_too_large",
        )


def to_get_pipeline_input(req: GetRequest) -> GetPipelineInput:
    """Build get pipeline input from a public get request."""

    return GetPipelineInput.model_validate(req.model_dump(by_alias=True))


def to_delete_pipeline_input(req: DeleteRequest) -> DeletePipelineInput:
    """Build delete pipeline input from a public delete request."""

    return DeletePipelineInput.model_validate(req.model_dump(by_alias=True))


def to_update_pipeline_input(req: UpdateRequest) -> UpdatePipelineInput:
    """Build update pipeline input from a public update request."""

    return UpdatePipelineInput.model_validate(req.model_dump(by_alias=True))


def to_feedback_pipeline_input(req: FeedbackRequest) -> FeedbackPipelineInput:
    """Build feedback pipeline input from a public feedback request."""

    return FeedbackPipelineInput.model_validate(
        req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS))
    )


def to_dreaming_pipeline_input(req: DreamingRequest) -> DreamingPipelineInput:
    """Build dreaming pipeline input from a public dreaming request."""

    return DreamingPipelineInput.model_validate(req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS)))


def to_memory_request_context(
    auth: AuthContext,
    identity: AddRequest | SearchRequest | FeedbackRequest | DreamingRequest | None = None,
    *,
    require_user_id: bool = False,
) -> MemoryRequestContext:
    """Build pipeline request context from auth and optional actor identity.

    Args:
        auth: Security context resolved by API dependencies.
        identity: Optional request model carrying actor identity fields.
        require_user_id: Whether the merged actor identity must include ``user_id``.

    Returns:
        The resolved memory request context.
    """

    resolved_actor = {field: getattr(identity, field, None) for field in _ACTOR_FIELDS} if identity is not None else {}
    if require_user_id and not resolved_actor.get("user_id"):
        raise BadRequestError("user_id is required in request body", code="missing_user_id")
    return MemoryRequestContext(
        request_id=auth.request_id,
        account_id=auth.account_id,
        project_id=auth.project_id,
        api_key_uuid=auth.api_key_uuid,
        memory_algorithm=auth.memory_algorithm,
        scopes=auth.scopes,
        **resolved_actor,
    )


# Flatten the pipeline's ``status`` / ``message`` into the ``ApiResponse``
# envelope (``code`` / ``message``) so ``data`` carries domain data only and the
# success signal is not duplicated across two layers.


def to_add_api_response(
    result: AddPipelineSyncResult | AddPipelineAsyncResult,
    request_id: str | None,
) -> ApiResponse[AddData]:
    """Convert an add pipeline result into an HTTP response envelope."""

    return ApiResponse[AddData](
        code=result.status,
        request_id=request_id,
        data=AddData(memories=result.memories),
    )


def to_memory_list_api_response(
    result: SearchPipelineResult | GetPipelineResult,
    request_id: str | None,
) -> ApiResponse[MemoryListData]:
    """Convert a search or get pipeline result into an HTTP response envelope."""

    return ApiResponse[MemoryListData](
        code=result.status,
        message=getattr(result, "message", None) or "",
        request_id=request_id,
        data=MemoryListData(memories=result.memories),
    )


def to_status_api_response(
    result: DeletePipelineResult | UpdatePipelineResult | FeedbackPipelineResult | DreamingPipelineResult,
    request_id: str | None,
) -> ApiResponse[None]:
    """Convert a status-only pipeline result into an HTTP response envelope."""

    return ApiResponse[None](
        code=result.status,
        message=getattr(result, "message", None) or "",
        request_id=request_id,
        data=None,
    )
