import pytest
from mindmemos.typing.memory import MemoryRequestContext, MemoryView
from mindmemos.typing.service import UpdatePipelineInput

from mindmemos.pipelines.update import DefaultUpdatePipeline
from mindmemos.typing import MemoryDbMutationResult, MemoryDbUpdateCommand, MemoryDbWriteResult


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeReader:
    def __init__(self, memory: MemoryView | None) -> None:
        self.memory = memory
        self.get_calls = []

    async def get_memory(self, ctx: MemoryRequestContext, memory_id: str) -> MemoryView | None:
        self.get_calls.append((ctx.project_id, memory_id))
        return self.memory


class FakeWriter:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, MemoryDbUpdateCommand]] = []

    async def apply_mutation_plan(self, ctx: MemoryRequestContext, plan) -> MemoryDbWriteResult:
        mutations = []
        for command in plan.memory_updates:
            self.update_calls.append((ctx.project_id, command))
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
        return MemoryDbWriteResult(mutations=mutations)

    async def update_memory(self, ctx: MemoryRequestContext, command: MemoryDbUpdateCommand) -> object:
        self.update_calls.append((ctx.project_id, command))
        return object()


@pytest.mark.asyncio
async def test_update_patches_existing_memory() -> None:
    memory = MemoryView(
        memory_id="mem-1",
        project_id="proj-1",
        content="old content",
        mem_type="fact",
        status="active",
        property_name="preference",
        metadata={"entity_name": "User"},
    )
    reader = FakeReader(memory)
    writer = FakeWriter()
    pipeline = DefaultUpdatePipeline(db_reader=reader, db_writer=writer)

    result = await pipeline.update(UpdatePipelineInput(memory_id="mem-1", content="new content"), make_context())

    assert reader.get_calls == [("proj-1", "mem-1")]
    assert len(writer.update_calls) == 1
    project_id, command = writer.update_calls[0]
    assert project_id == "proj-1"
    assert command.memory_id == "mem-1"
    assert command.content == "new content"
    assert command.embedding is None
    assert command.bm25_indices is None
    assert result.status == "ok"
    assert result.message is None


@pytest.mark.asyncio
async def test_update_returns_error_when_memory_is_missing() -> None:
    reader = FakeReader(None)
    writer = FakeWriter()
    pipeline = DefaultUpdatePipeline(db_reader=reader, db_writer=writer)

    result = await pipeline.update(UpdatePipelineInput(memory_id="missing", content="new content"), make_context())

    assert writer.update_calls == []
    assert result.status == "error"
    assert result.message == "memory not found: missing"


@pytest.mark.asyncio
async def test_update_returns_error_when_memory_is_not_active() -> None:
    memory = MemoryView(memory_id="mem-1", project_id="proj-1", content="old", mem_type="fact", status="archived")
    reader = FakeReader(memory)
    writer = FakeWriter()
    pipeline = DefaultUpdatePipeline(db_reader=reader, db_writer=writer)

    result = await pipeline.update(UpdatePipelineInput(memory_id="mem-1", content="new content"), make_context())

    assert writer.update_calls == []
    assert result.status == "error"
    assert result.message == "memory is not active (status=archived): mem-1"


@pytest.mark.asyncio
async def test_update_returns_error_when_content_is_blank() -> None:
    memory = MemoryView(memory_id="mem-1", project_id="proj-1", content="old", mem_type="fact", status="active")
    reader = FakeReader(memory)
    writer = FakeWriter()
    pipeline = DefaultUpdatePipeline(db_reader=reader, db_writer=writer)

    result = await pipeline.update(UpdatePipelineInput(memory_id="mem-1", content="   "), make_context())

    assert writer.update_calls == []
    assert result.status == "error"
    assert result.message == "content is empty"
