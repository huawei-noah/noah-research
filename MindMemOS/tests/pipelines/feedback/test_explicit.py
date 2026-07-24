import pytest
from types import SimpleNamespace

from mindmemos.components.feedback import DefaultExplicitFeedbackPlanner, FeedbackMemorySearchDecision
from mindmemos.pipelines.feedback.default import DefaultFeedbackPipeline
from mindmemos.pipelines.feedback.executor import FeedbackActionExecutor
from mindmemos.pipelines.feedback.explicit import ExplicitFeedbackHandler
from mindmemos.typing.llm import ChatResponse, EmbeddingResponse
from mindmemos.typing.memory import DialogueMessage, MemoryRequestContext, MemoryView
from mindmemos.typing.memory_db import MemoryDbMutationResult, MemoryDbWriteResult
from mindmemos.typing.service import (
    AddPipelineInput,
    AddPipelineSyncResult,
    DeletePipelineInput,
    DeletePipelineResult,
    FeedbackActionResult,
    FeedbackAddAction,
    FeedbackDeleteAction,
    FeedbackPipelineInput,
    FeedbackUpdateAction,
    MemorySearchItem,
    SearchPipelineInput,
    SearchPipelineResult,
    UpdatePipelineInput,
    UpdatePipelineResult,
)

from mindmemos.components.feedback import DefaultExplicitFeedbackPlanner, FeedbackMemorySearchDecision


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeNeo4jStore:
    """Mock Neo4j store."""

    def __init__(self) -> None:
        self.patches: list[dict] = []

    async def run_read(self, query: str, **params: object) -> list[dict]:
        return []

class FakePlanner:
    def __init__(self, *, decision: FeedbackMemorySearchDecision | None = None) -> None:
        self.decision = decision or FeedbackMemorySearchDecision(need_search=False)
        self.planned_input: FeedbackPipelineInput | None = None

    async def decide_memory_search(self, inp: FeedbackPipelineInput):
        return self.decision

    async def plan(self, inp: FeedbackPipelineInput):
        self.planned_input = inp
        return [
            FeedbackUpdateAction(
                target_memory_id=inp.recalled_memories[0].id,
                before_content=inp.recalled_memories[0].memory,
                after_content="User uses uv.",
                reason="user corrected package manager",
            )
        ]


class FakeExecutor:
    async def execute(self, actions: list[FeedbackActionResult], context: MemoryRequestContext):
        return actions


class FakeLlmClient:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.messages = []
        self.tasks = []

    async def chat(self, task, messages, format_parser=None, **kwargs):
        content = self.contents.pop(0)
        self.tasks.append(task)
        self.messages.append(messages)
        parsed = format_parser(content) if format_parser else None
        return ChatResponse(finish_reason="stop", content=content, parsed=parsed)


class FakeSearchPipeline:
    def __init__(self, memories: list[MemorySearchItem]) -> None:
        self.memories = memories
        self.input: SearchPipelineInput | None = None
        self.context: MemoryRequestContext | None = None

    async def search(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> SearchPipelineResult:
        self.input = inp
        self.context = context
        return SearchPipelineResult(status="ok", memories=self.memories)


class FakeAddPipeline:
    def __init__(self) -> None:
        self.input: AddPipelineInput | None = None
        self.context: MemoryRequestContext | None = None

    async def add_sync(self, inp: AddPipelineInput, context: MemoryRequestContext) -> AddPipelineSyncResult:
        self.input = inp
        self.context = context
        return AddPipelineSyncResult(status="ok", memories=[])


class FakeUpdatePipeline:
    def __init__(self) -> None:
        self.input: UpdatePipelineInput | None = None
        self.context: MemoryRequestContext | None = None

    async def update(self, inp: UpdatePipelineInput, context: MemoryRequestContext) -> UpdatePipelineResult:
        self.input = inp
        self.context = context
        return UpdatePipelineResult(status="ok")


class FakeDeletePipeline:
    def __init__(self) -> None:
        self.input: DeletePipelineInput | None = None
        self.context: MemoryRequestContext | None = None

    async def delete(self, inp: DeletePipelineInput, context: MemoryRequestContext) -> DeletePipelineResult:
        self.input = inp
        self.context = context
        return DeletePipelineResult(status="ok")


class FakeDbReader:
    def __init__(self) -> None:
        self.context = None
        self.memory = MemoryView(
            memory_id="mem-1",
            project_id="proj-1",
            content="User uses conda.",
            mem_type="fact",
            status="active",
        )
        self._clients = SimpleNamespace(neo4j=FakeNeo4jStore())

    async def get_memory(self, context: MemoryRequestContext, memory_id: str):
        self.context = context
        if memory_id != self.memory.memory_id:
            return None
        return self.memory


class FakeDbWriter:
    def __init__(self) -> None:
        self.context = None
        self.plan = None
        self.consistency = None
        self.update_command = None
        self.update_commands: list = []
        self.delete_command = None
        self.mutation_plans = []
        self._clients = SimpleNamespace(neo4j=FakeNeo4jStore())

    async def apply_mutation_plan(self, context: MemoryRequestContext, plan, *, consistency="fast"):
        self.context = context
        self.mutation_plans.append(plan)
        self.consistency = consistency
        write_plan = plan.to_write_plan()
        if write_plan.memories:
            self.plan = write_plan
        if plan.memory_updates:
            cmd = plan.memory_updates[0]
            self.update_command = cmd
            self.update_commands.append(cmd)
        if plan.memory_deletes:
            self.delete_command = plan.memory_deletes[0]
        mutations = []
        for command in plan.memory_updates:
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
        for command in plan.memory_deletes:
            mutations.append(MemoryDbMutationResult(memory_id=command.memory_id, changed=True))
        # Collect memory_ids from writes
        memory_ids = [cmd.memory.memory_id for cmd in plan.memory_writes]
        return MemoryDbWriteResult(memory_ids=memory_ids, mutations=mutations)

    async def write(self, context: MemoryRequestContext, plan, *, consistency="fast"):
        self.context = context
        self.plan = plan
        self.consistency = consistency
        return MemoryDbWriteResult(memory_ids=[plan.memories[0].memory_id])

    async def update_memory(self, context: MemoryRequestContext, command):
        self.context = context
        self.update_command = command
        return MemoryDbMutationResult(memory_id="mem-1-new")

    async def delete_memory(self, context: MemoryRequestContext, command):
        self.context = context
        self.delete_command = command
        return MemoryDbMutationResult(memory_id=command.memory_id)


class FakeEmbedClient:
    async def embed(self, task, text):
        return EmbeddingResponse(embeddings=[[0.1, 0.2, 0.3]])


class FakeTextPreprocessor:
    def preprocess_text(self, text, include_entities=False):
        class Result:
            tokens = ["user", "prefers", "uv"]

        return Result()


class FakeSparseEncoder:
    def encode_document(self, tokens):
        class Result:
            indices = [1, 2, 3]
            values = [1.0, 0.5, 0.25]

        return Result()


@pytest.mark.asyncio
async def test_explicit_feedback_requires_messages_context() -> None:
    pipeline = DefaultFeedbackPipeline()

    result = await pipeline.feedback(FeedbackPipelineInput(feedback="wrong memory"), make_context())

    assert result.status == "error"
    assert result.message == "explicit feedback requires messages context"


@pytest.mark.asyncio
async def test_explicit_feedback_uses_llm_planner() -> None:
    llm_client = FakeLlmClient("""{"need_search": false, "query": null}""")
    planner = DefaultExplicitFeedbackPlanner(llm_client=llm_client)
    decision = await planner.decide_memory_search(
        FeedbackPipelineInput(
            feedback="not conda, uv",
            messages=[DialogueMessage(role="user", content="not conda, uv", timestamp=1770000000000)],
            recalled_memories=[
                MemorySearchItem(id="mem-1", memory="User uses conda.", last_update_at="2026-06-01 00:00:00")
            ],
        )
    )

    assert decision.need_search is False
    assert llm_client.tasks == ["feedback.explicit.search_decision"]
    assert llm_client.messages[0][0]["role"] == "system"


@pytest.mark.asyncio
async def test_explicit_feedback_uses_llm_planner_for_actions() -> None:
    llm_client = FakeLlmClient(
        """{"need_search": false, "query": null}""",
        """
        {
          "actions": [
            {
              "action": "update",
              "target_memory_id": "mem-1",
              "before_content": "User uses conda.",
              "after_content": "User uses uv.",
              "reason": "user corrected package manager"
            }
          ]
        }
        """,
    )
    pipeline = DefaultFeedbackPipeline(
        explicit_handler=ExplicitFeedbackHandler(
            planner=DefaultExplicitFeedbackPlanner(llm_client=llm_client),
            executor=FakeExecutor(),
        )
    )

    result = await pipeline.feedback(
        FeedbackPipelineInput(
            feedback="not conda, uv",
            messages=[DialogueMessage(role="user", content="not conda, uv", timestamp=1770000000000)],
            recalled_memories=[
                MemorySearchItem(id="mem-1", memory="User uses conda.", last_update_at="2026-06-01 00:00:00")
            ],
        ),
        make_context(),
    )

    assert result.status == "ok"
    assert len(result.actions) == 1
    assert result.actions[0].action == "update"
    assert result.actions[0].target_memory_id == "mem-1"
    assert result.actions[0].before_content == "User uses conda."
    assert result.actions[0].after_content == "User uses uv."
    assert llm_client.tasks == ["feedback.explicit.search_decision", "feedback.explicit.plan"]
    assert llm_client.messages[1][0]["role"] == "system"


@pytest.mark.asyncio
async def test_explicit_feedback_uses_injected_planner() -> None:
    pipeline = DefaultFeedbackPipeline(
        explicit_handler=ExplicitFeedbackHandler(planner=FakePlanner(), executor=FakeExecutor())
    )

    result = await pipeline.feedback(
        FeedbackPipelineInput(
            feedback="not conda, uv",
            messages=[DialogueMessage(role="user", content="not conda, uv", timestamp=1770000000000)],
            recalled_memories=[
                MemorySearchItem(id="mem-1", memory="User uses conda.", last_update_at="2026-06-01 00:00:00")
            ],
        ),
        make_context(),
    )

    assert result.status == "ok"
    assert result.actions[0].action == "update"
    assert result.actions[0].after_content == "User uses uv."


@pytest.mark.asyncio
async def test_explicit_feedback_searches_once_when_planner_requests_more_memory() -> None:
    planner = FakePlanner(decision=FeedbackMemorySearchDecision(need_search=True, query="python package manager uv"))
    search_pipeline = FakeSearchPipeline(
        [
            MemorySearchItem(id="mem-1", memory="User uses conda.", last_update_at="2026-06-01 00:00:00"),
            MemorySearchItem(
                id="mem-2", memory="User prefers uv for Python packages.", last_update_at="2026-06-02 00:00:00"
            ),
        ]
    )
    pipeline = DefaultFeedbackPipeline(
        explicit_handler=ExplicitFeedbackHandler(
            planner=planner,
            executor=FakeExecutor(),
            search_pipeline=search_pipeline,
        )
    )
    context = make_context()

    result = await pipeline.feedback(
        FeedbackPipelineInput(
            feedback="not conda, uv",
            messages=[DialogueMessage(role="user", content="not conda, uv", timestamp=1770000000000)],
            recalled_memories=[
                MemorySearchItem(id="mem-1", memory="User uses conda.", last_update_at="2026-06-01 00:00:00")
            ],
        ),
        context,
    )

    assert result.status == "ok"
    assert search_pipeline.input == SearchPipelineInput(query="python package manager uv", search_pipeline="vanilla")
    assert search_pipeline.context == context
    assert planner.planned_input is not None
    assert [memory.id for memory in planner.planned_input.recalled_memories] == ["mem-1", "mem-2"]


@pytest.mark.asyncio
async def test_feedback_executor_runs_add_update_and_delete_actions() -> None:
    db_reader = FakeDbReader()
    db_writer = FakeDbWriter()
    executor = FeedbackActionExecutor(
        db_reader=db_reader,
        db_writer=db_writer,
        embed_client=FakeEmbedClient(),
        text_preprocessor=FakeTextPreprocessor(),
        sparse_encoder=FakeSparseEncoder(),
    )
    context = make_context()

    results = await executor.execute(
        [
            FeedbackAddAction(after_content="User prefers uv."),
            FeedbackUpdateAction(
                target_memory_id="mem-1",
                before_content="User uses conda.",
                after_content="User uses uv.",
            ),
            FeedbackDeleteAction(target_memory_id="mem-2", before_content="User uses conda."),
        ],
        context,
    )

    assert [result.status for result in results] == ["ok", "ok", "ok"]
    write_plans = [plan.to_write_plan() for plan in db_writer.mutation_plans if plan.to_write_plan().memories]
    # The add action and the update action both create new memories
    all_memories = [memory for plan in write_plans for memory in plan.memories]
    add_memory = [m for m in all_memories if m.content == "User prefers uv."]
    update_memory = [m for m in all_memories if m.content == "User uses uv."]
    assert len(add_memory) == 1, "add should create one memory"
    assert len(update_memory) == 1, "update should create one new version memory"
    assert db_writer.consistency == "strong"
    # Update command now archives the old memory (no content change)
    assert db_writer.update_commands[0].memory_id == "mem-1"
    assert db_writer.update_commands[0].status == "archived"
    assert db_writer.update_commands[0].reason == "feedback_update"
    # Delete command unchanged
    assert db_writer.delete_command.memory_id == "mem-2"
    assert db_writer.delete_command.reason == "feedback_delete"
    # Update now returns the new memory_id (not "mem-1-new")
    assert results[1].result_memory_id != "mem-1"
    assert results[2].result_memory_id == "mem-2"
    # No more patch_memory_id
    assert not hasattr(results[1], "patch_memory_id") or results[1].patch_memory_id is None
    assert not hasattr(results[2], "patch_memory_id") or results[2].patch_memory_id is None

    # Verify DERIVED_FROM relationship is in the write plan
    update_write_plan = [p for p in write_plans if any(m.content == "User uses uv." for m in p.memories)][0]
    assert len(update_write_plan.relationships) == 1
    assert update_write_plan.relationships[0].rel_type == "DERIVED_FROM"
    assert update_write_plan.relationships[0].source.node_id == update_memory[0].memory_id
    assert update_write_plan.relationships[0].target.node_id == "mem-1"
