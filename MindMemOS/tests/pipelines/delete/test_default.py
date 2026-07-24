import pytest
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import DeletePipelineInput

from mindmemos.pipelines.delete import DefaultDeletePipeline
from mindmemos.typing import MemoryDbDeleteCommand, MemoryDbMutationResult, MemoryDbWriteResult


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeMutationResult:
    def __init__(self, *, changed: bool) -> None:
        self.changed = changed


class FakeReader:
    pass


class FakeWriter:
    def __init__(self, *, changed: bool) -> None:
        self.changed = changed
        self.calls = []

    async def apply_mutation_plan(self, ctx: MemoryRequestContext, plan) -> MemoryDbWriteResult:
        mutations = []
        for command in plan.memory_deletes:
            self.calls.append((ctx.project_id, command.memory_id, command.hard, command.consistency))
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=self.changed))
        return MemoryDbWriteResult(mutations=mutations)

    async def delete_memory(self, ctx: MemoryRequestContext, command: MemoryDbDeleteCommand) -> FakeMutationResult:
        self.calls.append((ctx.project_id, command.memory_id, command.hard, command.consistency))
        return FakeMutationResult(changed=self.changed)


@pytest.mark.asyncio
async def test_delete_archives_existing_memory() -> None:
    writer = FakeWriter(changed=True)
    pipeline = DefaultDeletePipeline(db_reader=FakeReader(), db_writer=writer)

    result = await pipeline.delete(DeletePipelineInput(memory_id="mem-1"), make_context())

    assert writer.calls == [("proj-1", "mem-1", False, "strong")]
    assert result.status == "ok"
    assert result.message is None


@pytest.mark.asyncio
async def test_delete_returns_error_when_memory_is_missing() -> None:
    pipeline = DefaultDeletePipeline(db_reader=FakeReader(), db_writer=FakeWriter(changed=False))

    result = await pipeline.delete(DeletePipelineInput(memory_id="missing"), make_context())

    assert result.status == "error"
    assert result.message == "memory not found: missing"
