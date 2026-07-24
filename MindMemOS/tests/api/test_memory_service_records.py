from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import mindmemos.api.mappers as api_mappers
import pytest
from mindmemos.api.schemas import AddRequest, AuthContext, DreamingRequest, FeedbackRequest, SearchRequest
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.config.algo.search import SearchConfig
from mindmemos.errors import BadRequestError
from mindmemos.typing.memory import DialogueMessage, MemoryRequestContext
from mindmemos.typing.service import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    DreamingPipelineInput,
    DreamingPipelineResult,
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    MemoryAddEventItem,
    MemorySearchItem,
    SearchPipelineInput,
    SearchPipelineResult,
)
from mindmemos.typing.skill import SkillBinding, SkillContext


def make_context(memory_algorithm: str = "schema") -> AuthContext:
    return AuthContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        memory_algorithm=memory_algorithm,
    )


@pytest.fixture(autouse=True)
def search_config(monkeypatch) -> None:
    config = SearchConfig()

    def fake_get_config():
        return type("Cfg", (), {"algo_config": type("Algo", (), {"search": config})()})()

    monkeypatch.setattr(api_mappers, "get_config", fake_get_config)
    monkeypatch.setitem(
        MemoryService.search.__globals__, "to_search_pipeline_input", api_mappers.to_search_pipeline_input
    )


class FakeAddPipeline:
    def __init__(self) -> None:
        self.sync_calls: list[dict] = []

    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        self.sync_calls.append(
            {
                "inp": inp,
                "ctx": context,
                "add_record_id": add_record_id,
            }
        )
        return AddPipelineSyncResult(
            status="ok",
            memories=[MemoryAddEventItem(operation="add", content=inp.messages[0].content)],
        )

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ):
        raise AssertionError("unexpected async add")

    async def has_pending(self, ctx: MemoryRequestContext) -> bool:
        return False


class FailingAddPipeline(FakeAddPipeline):
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        raise RuntimeError("add failed")


class FakeAsyncAddPipeline(FakeAddPipeline):
    def __init__(self) -> None:
        self.async_calls: list[dict] = []

    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        raise AssertionError("unexpected sync add")

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict | None = None,
    ) -> AddPipelineAsyncResult:
        self.async_calls.append(
            {"inp": inp, "ctx": context, "add_record_id": add_record_id, "record_metadata": record_metadata}
        )
        return AddPipelineAsyncResult(status="queued")


class FakeSearchPipeline:
    def __init__(self, label: str = "default") -> None:
        self.label = label

    async def search(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> SearchPipelineResult:
        return SearchPipelineResult(
            status="ok",
            memories=[
                MemorySearchItem(
                    id="mem-1",
                    memory=f"{self.label}:{inp.search_pipeline}:{inp.agentic}:{inp.query}",
                    last_update_at="2026-05-28 00:00:00",
                )
            ],
        )


class FailingSearchPipeline(FakeSearchPipeline):
    async def search(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> SearchPipelineResult:
        raise RuntimeError("search failed")


class FakeFeedbackPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, FeedbackPipelineInput, MemoryRequestContext]] = []

    async def feedback_sync(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        self.calls.append(("sync", inp, context))
        return FeedbackPipelineResult(status="ok", message="done")

    async def feedback_async(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        self.calls.append(("async", inp, context))
        return FeedbackPipelineResult(status="queued", message="queued")


class FakeDreamingPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, DreamingPipelineInput, MemoryRequestContext]] = []

    async def dream(self, inp: DreamingPipelineInput, context: MemoryRequestContext) -> DreamingPipelineResult:
        self.calls.append(("async", inp, context))
        return DreamingPipelineResult(status="queued", message="queued")

    async def dream_sync(self, inp: DreamingPipelineInput, context: MemoryRequestContext) -> DreamingPipelineResult:
        self.calls.append(("sync", inp, context))
        return DreamingPipelineResult(status="ok", message="done")


@dataclass
class FakeRecorder:
    add_calls: list[dict] = field(default_factory=list)
    failed_calls: list[dict] = field(default_factory=list)
    search_calls: list[dict] = field(default_factory=list)

    async def record_add_input(
        self,
        inp,
        *,
        ctx,
        request_submitted_at,
        add_record_id,
        status,
        skill_bindings=None,
        score=None,
        task_id=None,
    ) -> str:
        self.add_calls.append(
            {
                "add_record_id": add_record_id,
                "inp": inp,
                "ctx": ctx,
                "request_submitted_at": request_submitted_at,
                "status": status,
                "skill_bindings": skill_bindings,
                "score": score,
                "task_id": task_id,
            }
        )
        return add_record_id

    async def mark_add_failed(self, ctx, add_record_id, error) -> None:
        self.failed_calls.append({"ctx": ctx, "add_record_id": add_record_id, "error": error})

    async def record_search(self, inp, result, *, ctx, request_submitted_at, task_completed_at) -> None:
        self.search_calls.append(
            {
                "inp": inp,
                "result": result,
                "ctx": ctx,
                "request_submitted_at": request_submitted_at,
                "task_completed_at": task_completed_at,
            }
        )


class UnusedPipeline:
    pass


@dataclass
class FakeSkillStore:
    calls: list[dict] = field(default_factory=list)
    raise_on_bind: bool = False

    async def bind_skill_context(self, *, project_id, add_record_id, skill_context):
        self.calls.append({"project_id": project_id, "add_record_id": add_record_id, "skill_context": skill_context})
        if self.raise_on_bind:
            raise RuntimeError("boom")
        return [
            SkillBinding(name=sc.name, content_hash=sc.content_hash, base_version_id=sc.base_version_id)
            for sc in skill_context
        ]


def make_service(**kwargs) -> MemoryService:
    return MemoryService(
        get_pipeline=UnusedPipeline(),
        delete_pipeline=UnusedPipeline(),
        update_pipeline=UnusedPipeline(),
        **kwargs,
    )


def add_request() -> AddRequest:
    return AddRequest(
        user_id="u1",
        messages=[DialogueMessage(role="user", content="remember me", timestamp=1770000000000)],
    )


@pytest.mark.asyncio
async def test_memory_service_records_add_result_for_vanilla_add_pipeline() -> None:
    recorder = FakeRecorder()
    pipeline = FakeAddPipeline()
    service = make_service(
        add_pipeline=pipeline,
        add_pipeline_name="vanilla_add",
        operation_recorder=recorder,
    )

    result = await service.add(make_context(), add_request())

    assert result.status == "ok"
    assert len(recorder.add_calls) == 1
    call = recorder.add_calls[0]
    assert call["status"] == "processing"
    # The same add_record_id flows into the pipeline so it can write the output back.
    assert pipeline.sync_calls[0]["add_record_id"] == call["add_record_id"]


@pytest.mark.asyncio
async def test_memory_service_records_add_result_for_non_vanilla_add_pipeline() -> None:
    recorder = FakeRecorder()
    pipeline = FakeAddPipeline()
    service = make_service(
        add_pipeline=pipeline,
        add_pipeline_name="default_add",
        operation_recorder=recorder,
    )

    result = await service.add(make_context(), add_request())

    assert result.status == "ok"
    assert len(recorder.add_calls) == 1
    call = recorder.add_calls[0]
    assert call["status"] == "processing"
    assert isinstance(call["request_submitted_at"], datetime)
    assert pipeline.sync_calls[0]["add_record_id"] == call["add_record_id"]


@pytest.mark.asyncio
async def test_memory_service_records_async_result_for_vanilla_add_pipeline() -> None:
    recorder = FakeRecorder()
    pipeline = FakeAsyncAddPipeline()
    service = make_service(
        add_pipeline=pipeline,
        add_pipeline_name="vanilla_add",
        operation_recorder=recorder,
    )
    request = add_request().model_copy(update={"mode": "async"})

    result = await service.add(make_context(), request)

    assert result.status == "queued"
    assert len(recorder.add_calls) == 1
    assert recorder.add_calls[0]["status"] == "queued"
    assert len(pipeline.async_calls) == 1
    assert pipeline.async_calls[0]["add_record_id"] == recorder.add_calls[0]["add_record_id"]
    assert pipeline.async_calls[0]["record_metadata"]["request_submitted_at"]


@pytest.mark.asyncio
async def test_memory_service_records_async_result_for_default_add_pipeline() -> None:
    recorder = FakeRecorder()
    pipeline = FakeAsyncAddPipeline()
    service = make_service(
        add_pipeline=pipeline,
        add_pipeline_name="default_add",
        operation_recorder=recorder,
    )
    request = add_request().model_copy(update={"mode": "async"})

    result = await service.add(make_context(), request)

    assert result.status == "queued"
    assert len(recorder.add_calls) == 1
    assert recorder.add_calls[0]["status"] == "queued"
    assert pipeline.async_calls[0]["ctx"].project_id == "proj-1"
    assert pipeline.async_calls[0]["add_record_id"] == recorder.add_calls[0]["add_record_id"]


@pytest.mark.asyncio
async def test_memory_service_records_add_result_for_schema_add_pipeline() -> None:
    recorder = FakeRecorder()
    pipeline = FakeAddPipeline()
    service = make_service(
        add_pipeline=pipeline,
        add_pipeline_name="schema_add",
        operation_recorder=recorder,
    )

    result = await service.add(make_context(), add_request())

    assert result.status == "ok"
    assert len(recorder.add_calls) == 1
    call = recorder.add_calls[0]
    assert call["status"] == "processing"
    assert pipeline.sync_calls[0]["add_record_id"] == call["add_record_id"]


@pytest.mark.asyncio
async def test_memory_service_records_failed_add_for_vanilla_add_pipeline() -> None:
    recorder = FakeRecorder()
    service = make_service(
        add_pipeline=FailingAddPipeline(),
        add_pipeline_name="vanilla_add",
        operation_recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="add failed"):
        await service.add(make_context(), add_request())

    # Input is recorded up front; the failure is patched onto the same record.
    assert len(recorder.add_calls) == 1
    assert len(recorder.failed_calls) == 1
    assert recorder.failed_calls[0]["add_record_id"] == recorder.add_calls[0]["add_record_id"]


@pytest.mark.asyncio
async def test_add_request_actor_identity_merged_into_context_and_stripped_from_input() -> None:
    recorder = FakeRecorder()
    service = make_service(
        add_pipeline=FakeAddPipeline(),
        add_pipeline_name="default_add",
        operation_recorder=recorder,
    )

    await service.add(
        make_context(),
        AddRequest(
            user_id="caller-user",
            session_id="caller-session",
            agent_id="caller-agent",
            app_id="caller-app",
            messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)],
        ),
    )

    call = recorder.add_calls[0]
    assert call["ctx"].user_id == "caller-user"
    assert call["ctx"].session_id == "caller-session"
    assert call["ctx"].agent_id == "caller-agent"
    assert call["ctx"].app_id == "caller-app"
    assert isinstance(call["inp"], AddPipelineInput)
    assert not hasattr(call["inp"], "user_id")


@pytest.mark.asyncio
async def test_add_request_missing_user_id_mentions_request_body() -> None:
    service = make_service(add_pipeline=FakeAddPipeline())

    with pytest.raises(BadRequestError, match="user_id is required in request body"):
        await service.add(
            make_context(),
            AddRequest(messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)]),
        )


@pytest.mark.asyncio
async def test_add_request_missing_optional_actor_fields_stay_empty() -> None:
    recorder = FakeRecorder()
    service = make_service(add_pipeline=FakeAddPipeline(), operation_recorder=recorder)

    # Only the required user_id is supplied; the optional actor fields
    # (app_id / session_id / agent_id) must stay empty on the assembled context.
    await service.add(
        make_context(),
        AddRequest(
            user_id="caller-user", messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)]
        ),
    )

    ctx = recorder.add_calls[0]["ctx"]
    assert ctx.user_id == "caller-user"
    assert ctx.app_id is None
    assert ctx.session_id is None
    assert ctx.agent_id is None


@pytest.mark.asyncio
async def test_add_binds_skill_context_and_passes_bindings_to_recorder() -> None:
    recorder = FakeRecorder()
    skill_store = FakeSkillStore()
    service = make_service(add_pipeline=FakeAddPipeline(), operation_recorder=recorder, skill_store=skill_store)

    await service.add(
        make_context(),
        AddRequest(
            user_id="u1",
            messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)],
            skill_context=[SkillContext(name="prd-writer", content_hash="h1", base_version_id="")],
        ),
    )

    # The store was driven with the trace id that the recorder also received.
    assert len(skill_store.calls) == 1
    call = recorder.add_calls[0]
    assert skill_store.calls[0]["add_record_id"] == call["add_record_id"]
    assert call["add_record_id"]
    assert [b.name for b in call["skill_bindings"]] == ["prd-writer"]
    # skill_context never leaks into the pipeline input contract.
    assert not hasattr(call["inp"], "skill_context")


@pytest.mark.asyncio
async def test_vanilla_add_records_skill_bindings_at_service() -> None:
    recorder = FakeRecorder()
    skill_store = FakeSkillStore()
    pipeline = FakeAddPipeline()
    service = make_service(
        add_pipeline=pipeline,
        add_pipeline_name="vanilla_add",
        operation_recorder=recorder,
        skill_store=skill_store,
    )

    result = await service.add(
        make_context("vanilla"),
        AddRequest(
            user_id="u1",
            messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)],
            skill_context=[SkillContext(name="prd-writer", content_hash="h1", base_version_id="")],
            score=1.0,
            task_id="case-1",
        ),
    )

    assert result.status == "ok"
    assert len(recorder.add_calls) == 1
    assert len(skill_store.calls) == 1
    call = recorder.add_calls[0]
    assert call["add_record_id"] == skill_store.calls[0]["add_record_id"]
    assert call["add_record_id"]
    assert [binding.name for binding in call["skill_bindings"]] == ["prd-writer"]
    assert call["score"] == 1.0
    assert call["task_id"] == "case-1"
    assert len(pipeline.sync_calls) == 1


@pytest.mark.asyncio
async def test_add_without_skill_context_passes_no_bindings() -> None:
    recorder = FakeRecorder()
    skill_store = FakeSkillStore()
    service = make_service(add_pipeline=FakeAddPipeline(), operation_recorder=recorder, skill_store=skill_store)

    await service.add(
        make_context(),
        AddRequest(user_id="u1", messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)]),
    )

    assert skill_store.calls == []
    assert recorder.add_calls[0]["skill_bindings"] is None


@pytest.mark.asyncio
async def test_add_skill_binding_failure_never_blocks_add() -> None:
    recorder = FakeRecorder()
    skill_store = FakeSkillStore(raise_on_bind=True)
    service = make_service(add_pipeline=FakeAddPipeline(), operation_recorder=recorder, skill_store=skill_store)

    result = await service.add(
        make_context(),
        AddRequest(
            user_id="u1",
            messages=[DialogueMessage(role="user", content="hi", timestamp=1770000000000)],
            skill_context=[SkillContext(name="prd-writer", content_hash="h1", base_version_id="")],
        ),
    )

    # Add still succeeds and is still recorded, just without skill bindings.
    assert result.status == "ok"
    assert recorder.add_calls[0]["skill_bindings"] is None


@pytest.mark.asyncio
async def test_memory_service_records_search_result_for_regular_search_pipeline() -> None:
    recorder = FakeRecorder()
    service = make_service(
        search_pipeline=FakeSearchPipeline(),
        search_pipeline_name="search_pipeline",
        operation_recorder=recorder,
    )

    result = await service.search(make_context(), SearchRequest(user_id="u1", query="Qdrant"))

    assert result.status == "ok"
    assert len(recorder.search_calls) == 1
    call = recorder.search_calls[0]
    assert call["result"] is result
    assert isinstance(call["request_submitted_at"], datetime)
    assert isinstance(call["task_completed_at"], datetime)
    assert call["task_completed_at"] >= call["request_submitted_at"]


@pytest.mark.asyncio
async def test_search_request_actor_identity_merged_into_context_and_stripped_from_input() -> None:
    recorder = FakeRecorder()
    service = make_service(
        search_pipeline=FakeSearchPipeline(),
        search_pipeline_name="search_pipeline",
        operation_recorder=recorder,
    )

    await service.search(
        make_context(),
        SearchRequest(
            user_id="caller-user",
            session_id="caller-session",
            app_id="caller-app",
            agent_id="caller-agent",
            query="Qdrant",
        ),
    )

    call = recorder.search_calls[0]
    assert call["ctx"].user_id == "caller-user"
    assert call["ctx"].session_id == "caller-session"
    assert call["ctx"].app_id == "caller-app"
    assert call["ctx"].agent_id == "caller-agent"
    assert isinstance(call["inp"], SearchPipelineInput)
    assert not hasattr(call["inp"], "user_id")


@pytest.mark.asyncio
async def test_memory_service_records_agentic_search_result() -> None:
    recorder = FakeRecorder()
    service = make_service(
        search_pipeline=FakeSearchPipeline("search"),
        search_pipeline_name="search_pipeline",
        operation_recorder=recorder,
    )

    result = await service.search(
        make_context("schema"), SearchRequest(user_id="u1", query="Qdrant", search_strategy="agentic")
    )

    assert result.status == "ok"
    assert result.memories[0].memory == "search:schema:True:Qdrant"
    assert len(recorder.search_calls) == 1
    call = recorder.search_calls[0]
    assert call["result"] is result
    assert call["inp"].search_pipeline == "schema"
    assert call["inp"].agentic is True


@pytest.mark.asyncio
async def test_memory_service_records_failed_search() -> None:
    recorder = FakeRecorder()
    service = make_service(
        search_pipeline=FailingSearchPipeline(),
        search_pipeline_name="search_pipeline",
        operation_recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="search failed"):
        await service.search(make_context("schema"), SearchRequest(user_id="u1", query="Qdrant"))

    assert len(recorder.search_calls) == 1
    call = recorder.search_calls[0]
    assert call["inp"].search_pipeline == "schema"
    assert call["result"] is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_request_rejects_out_of_bounds_top_k(top_k: int) -> None:
    with pytest.raises(ValueError):
        SearchRequest(user_id="u1", query="Qdrant", top_k=top_k)


def test_search_request_allows_positive_top_k_and_none_before_configured_cap() -> None:
    assert SearchRequest(user_id="u1", query="Qdrant", top_k=101).top_k == 101
    assert SearchRequest(user_id="u1", query="Qdrant", top_k=None).top_k is None


@pytest.mark.asyncio
async def test_memory_service_rejects_search_top_k_above_configured_user_limit(monkeypatch) -> None:
    search_config = SearchConfig(request_top_k_max=7)

    def fake_get_config():
        return type("Cfg", (), {"algo_config": type("Algo", (), {"search": search_config})()})()

    monkeypatch.setattr(api_mappers, "get_config", fake_get_config)
    monkeypatch.setitem(
        MemoryService.search.__globals__, "to_search_pipeline_input", api_mappers.to_search_pipeline_input
    )
    service = make_service(
        search_pipeline=FakeSearchPipeline(),
        search_pipeline_name="search_pipeline",
        operation_recorder=FakeRecorder(),
    )

    with pytest.raises(BadRequestError, match="top_k must be <= 7"):
        await service.search(make_context(), SearchRequest(user_id="u1", query="Qdrant", top_k=8))


@pytest.mark.asyncio
async def test_memory_service_dream_defaults_to_async_mode() -> None:
    pipeline = FakeDreamingPipeline()
    service = make_service(
        dreaming_pipeline=pipeline,
        dreaming_pipeline_name="default_dreaming",
        operation_recorder=FakeRecorder(),
    )

    result = await service.dream(
        make_context(),
        DreamingRequest(user_id="dreamer-1", app_id="app", agent_id="agent", session_id="session"),
    )

    assert result.status == "queued"
    assert [(name, inp.mode) for name, inp, _ctx in pipeline.calls] == [("async", "async")]
    _name, _inp, ctx = pipeline.calls[0]
    assert ctx.user_id == "dreamer-1"
    assert ctx.app_id == "app"
    assert ctx.agent_id == "agent"
    assert ctx.session_id == "session"


@pytest.mark.asyncio
async def test_memory_service_dream_sync_mode_calls_sync_pipeline() -> None:
    pipeline = FakeDreamingPipeline()
    service = make_service(
        dreaming_pipeline=pipeline,
        dreaming_pipeline_name="default_dreaming",
        operation_recorder=FakeRecorder(),
    )

    result = await service.dream(make_context(), DreamingRequest(mode="sync"))

    assert result.status == "ok"
    assert [(name, inp.mode) for name, inp, _ctx in pipeline.calls] == [("sync", "sync")]


@pytest.mark.asyncio
async def test_memory_service_routes_regular_and_agentic_to_same_search_pipeline() -> None:
    service = make_service(
        search_pipeline=FakeSearchPipeline("search"),
        operation_recorder=FakeRecorder(),
    )

    shallow = await service.search(
        make_context("vanilla"), SearchRequest(user_id="u1", query="Qdrant", search_strategy="fast")
    )
    deep = await service.search(
        make_context("schema"), SearchRequest(user_id="u1", query="Qdrant", search_strategy="agentic")
    )

    assert shallow.memories[0].memory == "search:vanilla:False:Qdrant"
    assert deep.memories[0].memory == "search:schema:True:Qdrant"


@pytest.mark.asyncio
async def test_memory_service_feedback_uses_actor_context() -> None:
    pipeline = FakeFeedbackPipeline()
    service = make_service(feedback_pipeline=pipeline, operation_recorder=FakeRecorder())

    result = await service.feedback(
        make_context(),
        FeedbackRequest(user_id="persona-1", app_id="app", agent_id="agent", session_id="session", mode="sync"),
    )

    assert result.status == "ok"
    mode, inp, ctx = pipeline.calls[0]
    assert mode == "sync"
    assert inp.mode == "sync"
    assert ctx.user_id == "persona-1"
    assert ctx.app_id == "app"
    assert ctx.agent_id == "agent"
    assert ctx.session_id == "session"


@pytest.mark.asyncio
async def test_memory_service_feedback_requires_user_id() -> None:
    service = make_service(feedback_pipeline=FakeFeedbackPipeline(), operation_recorder=FakeRecorder())

    with pytest.raises(BadRequestError, match="user_id"):
        await service.feedback(make_context(), FeedbackRequest(mode="sync"))
