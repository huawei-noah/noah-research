"""Memory HTTP business logic."""

from typing import Literal
from uuid import uuid4

from ...config import get_config
from ...logging import get_logger, traced
from ...pipelines import create_pipeline
from ...pipelines.add import AddPipeline
from ...pipelines.delete import DefaultDeletePipeline, DeletePipeline
from ...pipelines.dreaming import DreamingPipeline
from ...pipelines.feedback import FeedbackPipeline
from ...pipelines.get import DefaultGetPipeline, GetPipeline
from ...pipelines.memory_db import MemoryOperationRecorder, suppress_recording_errors, utcnow
from ...pipelines.search import SearchPipeline
from ...pipelines.skill import SkillVersionStore, get_skill_version_store
from ...pipelines.update import DefaultUpdatePipeline, UpdatePipeline
from ...typing import (
    AddPipelineAsyncResult,
    AddPipelineSyncResult,
    DeletePipelineResult,
    DreamingPipelineResult,
    FeedbackPipelineResult,
    GetPipelineResult,
    SearchPipelineResult,
    SkillBinding,
    SkillContext,
    UpdatePipelineResult,
)
from ..algorithm import binding_for_memory_algorithm
from ..deps import annotate_request_trace
from ..mappers import (
    to_add_pipeline_input,
    to_delete_pipeline_input,
    to_dreaming_pipeline_input,
    to_feedback_pipeline_input,
    to_get_pipeline_input,
    to_memory_request_context,
    to_search_pipeline_input,
    to_update_pipeline_input,
)
from ..schemas import (
    AddRequest,
    AuthContext,
    DeleteRequest,
    DreamingRequest,
    FeedbackRequest,
    GetRequest,
    SearchRequest,
    UpdateRequest,
)

logger = get_logger(__name__)

PipelineKind = Literal["add", "search", "get", "delete", "update", "feedback", "dreaming"]
SEARCH_PIPELINE_NAME = "search_pipeline"


class MemoryService:
    """Stateless facade routing memory endpoints to their pipelines."""

    def __init__(
        self,
        *,
        get_pipeline: GetPipeline | None = None,
        add_pipeline: AddPipeline | None = None,
        search_pipeline: SearchPipeline | None = None,
        delete_pipeline: DeletePipeline | None = None,
        update_pipeline: UpdatePipeline | None = None,
        feedback_pipeline: FeedbackPipeline | None = None,
        dreaming_pipeline: DreamingPipeline | None = None,
        add_pipeline_name: str | None = None,
        search_pipeline_name: str | None = None,
        get_pipeline_name: str | None = None,
        delete_pipeline_name: str | None = None,
        update_pipeline_name: str | None = None,
        feedback_pipeline_name: str | None = None,
        dreaming_pipeline_name: str | None = None,
        operation_recorder: MemoryOperationRecorder | None = None,
        skill_store: SkillVersionStore | None = None,
    ) -> None:
        self._add = add_pipeline
        if search_pipeline is None and search_pipeline_name is None:
            search_pipeline_name = SEARCH_PIPELINE_NAME
        self._search = search_pipeline
        self._get = get_pipeline if get_pipeline is not None else (None if get_pipeline_name else DefaultGetPipeline())
        self._delete = (
            delete_pipeline
            if delete_pipeline is not None
            else (None if delete_pipeline_name else DefaultDeletePipeline())
        )
        self._update = (
            update_pipeline
            if update_pipeline is not None
            else (None if update_pipeline_name else DefaultUpdatePipeline())
        )
        self._feedback = feedback_pipeline
        self._dreaming = dreaming_pipeline
        self._pipeline_names: dict[str, tuple[PipelineKind, str | None]] = {
            "_add": ("add", add_pipeline_name),
            "_search": ("search", search_pipeline_name),
            "_get": ("get", get_pipeline_name),
            "_delete": ("delete", delete_pipeline_name),
            "_update": ("update", update_pipeline_name),
            "_feedback": ("feedback", feedback_pipeline_name),
            "_dreaming": ("dreaming", dreaming_pipeline_name),
        }
        self._recorder = operation_recorder or MemoryOperationRecorder()
        self._skill_store = skill_store if skill_store is not None else get_skill_version_store()
        self._algorithm_add_pipelines: dict[str, AddPipeline] = {}

    def _pipeline(self, attr: str):
        pipeline = getattr(self, attr)
        if pipeline is not None:
            return pipeline
        pipeline_type, pipeline_name = self._pipeline_names[attr]
        if pipeline_name is None:
            return None
        pipeline = create_pipeline(type=pipeline_type, name=pipeline_name)
        setattr(self, attr, pipeline)
        return pipeline

    def _add_pipeline_for_algorithm(self, memory_algorithm: str) -> tuple[AddPipeline | None, str | None]:
        pipeline_name = binding_for_memory_algorithm(memory_algorithm).add_pipeline
        pipeline = self._algorithm_add_pipelines.get(pipeline_name)
        if pipeline is None:
            pipeline = create_pipeline(type="add", name=pipeline_name)
            self._algorithm_add_pipelines[pipeline_name] = pipeline
        return pipeline, pipeline_name

    def _add_pipeline_for_auth(self, auth: AuthContext) -> tuple[AddPipeline | None, str | None]:
        if self._add is not None:
            return self._add, self._pipeline_names["_add"][1]
        return self._add_pipeline_for_algorithm(auth.memory_algorithm)

    @traced("memory_service.add")
    async def add(
        self,
        auth: AuthContext,
        request: AddRequest,
    ) -> AddPipelineSyncResult | AddPipelineAsyncResult:
        """Run the add pipeline according to the requested mode."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline, _add_pipeline_name = self._add_pipeline_for_auth(auth)
        if pipeline is None:
            raise NotImplementedError("add pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        payload = to_add_pipeline_input(request)
        add_record_id = str(uuid4())
        request_submitted_at = utcnow()
        skill_bindings = await self._bind_skill_context(ctx.project_id, add_record_id, request.skill_context)
        try:
            if payload.mode == "async":
                record_metadata = {
                    "request_submitted_at": request_submitted_at.isoformat(),
                    "skill_bindings": [binding.model_dump(mode="json") for binding in skill_bindings or []],
                    "score": request.score,
                    "task_id": request.task_id,
                }
                await suppress_recording_errors(
                    self._recorder.record_add_input(
                        payload,
                        ctx=ctx,
                        request_submitted_at=request_submitted_at,
                        add_record_id=add_record_id,
                        status="queued",
                        skill_bindings=skill_bindings,
                        score=request.score,
                        task_id=request.task_id,
                    ),
                    operation="add",
                )
                return await pipeline.add_async(
                    payload,
                    ctx,
                    add_record_id=add_record_id,
                    record_metadata=record_metadata,
                )
            await suppress_recording_errors(
                self._recorder.record_add_input(
                    payload,
                    ctx=ctx,
                    request_submitted_at=request_submitted_at,
                    add_record_id=add_record_id,
                    status="processing",
                    skill_bindings=skill_bindings,
                    score=request.score,
                    task_id=request.task_id,
                ),
                operation="add",
            )
            return await pipeline.add_sync(payload, ctx, add_record_id=add_record_id)
        except Exception as exc:
            await suppress_recording_errors(
                self._recorder.mark_add_failed(ctx, add_record_id, str(exc)),
                operation="add",
            )
            raise

    async def _bind_skill_context(
        self,
        project_id: str,
        add_record_id: str,
        skill_context: list[SkillContext] | None,
    ) -> list[SkillBinding] | None:
        """Resolve ``skill_context`` to trace bindings, swallowing all failures."""

        if self._skill_store is None or not skill_context:
            return None
        try:
            return await self._skill_store.bind_skill_context(
                project_id=project_id,
                add_record_id=add_record_id,
                skill_context=skill_context,
            )
        except Exception as exc:  # noqa: BLE001 - binding must never block add
            logger.warning("skill_context binding failed", error=str(exc), add_record_id=add_record_id)
            return None

    @traced("memory_service.search")
    async def search(
        self,
        auth: AuthContext,
        request: SearchRequest,
    ) -> SearchPipelineResult:
        """Run the search pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        binding = binding_for_memory_algorithm(auth.memory_algorithm)
        payload = to_search_pipeline_input(request, search_pipeline=binding.search_pipeline)
        pipeline = self._pipeline("_search")
        if pipeline is None:
            raise NotImplementedError("search pipeline implementation is not wired yet")
        request_submitted_at = utcnow()
        try:
            result = await pipeline.search(payload, ctx)
        except Exception:
            task_completed_at = utcnow()
            await suppress_recording_errors(
                self._recorder.record_search(
                    payload,
                    None,
                    ctx=ctx,
                    request_submitted_at=request_submitted_at,
                    task_completed_at=task_completed_at,
                ),
                operation="search",
            )
            raise
        task_completed_at = utcnow()
        await suppress_recording_errors(
            self._recorder.record_search(
                payload,
                result,
                ctx=ctx,
                request_submitted_at=request_submitted_at,
                task_completed_at=task_completed_at,
            ),
            operation="search",
        )
        return result

    @traced("memory_service.get")
    async def get(self, auth: AuthContext, request: GetRequest) -> GetPipelineResult:
        """Run the get pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_get")
        if pipeline is None:
            raise NotImplementedError("get pipeline implementation is not wired yet")
        return await pipeline.get(to_get_pipeline_input(request), to_memory_request_context(auth))


    @traced("memory_service.delete")
    async def delete(self, auth: AuthContext, request: DeleteRequest) -> DeletePipelineResult:
        """Run the delete pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_delete")
        if pipeline is None:
            raise NotImplementedError("delete pipeline implementation is not wired yet")
        return await pipeline.delete(to_delete_pipeline_input(request), to_memory_request_context(auth))

    @traced("memory_service.update")
    async def update(self, auth: AuthContext, request: UpdateRequest) -> UpdatePipelineResult:
        """Run the update pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_update")
        if pipeline is None:
            raise NotImplementedError("update pipeline implementation is not wired yet")
        return await pipeline.update(to_update_pipeline_input(request), to_memory_request_context(auth))

    @traced("memory_service.feedback")
    async def feedback(self, auth: AuthContext, request: FeedbackRequest) -> FeedbackPipelineResult:
        """Run the feedback pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_feedback")
        if pipeline is None:
            raise NotImplementedError("feedback pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        payload = to_feedback_pipeline_input(request)
        if payload.mode == "async":
            return await pipeline.feedback_async(payload, ctx)
        return await pipeline.feedback_sync(payload, ctx)

    @traced("memory_service.dream")
    async def dream(self, auth: AuthContext, request: DreamingRequest) -> DreamingPipelineResult:
        """Run the dreaming pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_dreaming")
        if pipeline is None:
            raise NotImplementedError("dreaming pipeline implementation is not wired yet")
        payload = to_dreaming_pipeline_input(request)
        ctx = to_memory_request_context(auth, request)
        if payload.mode == "sync":
            return await pipeline.dream_sync(payload, ctx)
        return await pipeline.dream(payload, ctx)


_service: MemoryService | None = None
_service_key: tuple[str, ...] | None = None


def get_memory_service() -> MemoryService:
    """Process-global service singleton, used as a FastAPI dependency."""

    global _service, _service_key
    cfg = get_config().pipelines
    get_pipeline = cfg["get"]
    delete_pipeline = cfg["delete"]
    update_pipeline = cfg["update"]
    feedback_pipeline = cfg["feedback"]
    dreaming_pipeline = cfg["dreaming"]
    service_key = (
        get_pipeline,
        delete_pipeline,
        update_pipeline,
        feedback_pipeline,
        dreaming_pipeline,
    )
    if _service is None or _service_key != service_key:
        _service = MemoryService(
            get_pipeline_name=get_pipeline,
            delete_pipeline_name=delete_pipeline,
            update_pipeline_name=update_pipeline,
            feedback_pipeline_name=feedback_pipeline,
            dreaming_pipeline_name=dreaming_pipeline,
        )
        _service_key = service_key
    return _service
