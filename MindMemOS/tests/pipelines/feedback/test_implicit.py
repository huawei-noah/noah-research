from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.infra.db.models import QdrantRecord
from mindmemos.pipelines.feedback.default import DefaultFeedbackPipeline
from mindmemos.pipelines.feedback.implicit import ImplicitFeedbackHandler, ImplicitFeedbackRecordCollector
from mindmemos.typing.algo import (
    ImplicitFeedbackRound,
    ImplicitFeedbackSessionMaterial,
    ImplicitFeedbackSignal,
    ImplicitFeedbackSignalResult,
    SupplementalSearchQuery,
)
from mindmemos.typing.llm import ChatResponse
from mindmemos.typing.memory import MemoryRequestContext, MemoryView, SearchFilter
from mindmemos.typing.service import (
    FeedbackActionResult,
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    FeedbackUpdateAction,
    MemorySearchItem,
    SearchPipelineInput,
    SearchPipelineResult,
)

from mindmemos.components.feedback import ImplicitFeedbackActionPlanner, ImplicitFeedbackSignalDetector


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeQdrant:
    def __init__(self) -> None:
        self.add_calls = 0
        self.search_calls = 0
        self.add_filters = []
        self.search_filters = []
        self.updated_add_records: list[tuple[str, str, dict]] = []

    async def scroll_add_records(self, project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        self.add_calls += 1
        self.add_filters.append(filter_)
        records = [
            QdrantRecord(
                point_id="add-1",
                payload={
                    "account_id": "acc-1",
                    "project_id": project_id,
                    "api_key_uuid": "key-1",
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "request_id": "request-add-1",
                    "status": "ok",
                    "request_submitted_at": "2026-06-01T00:00:00+00:00",
                    "buffer_sequence": 1,
                    "messages": [
                        {"role": "user", "content": "你要说得详细一点", "timestamp": 1770000000000},
                    ],
                    "memories": [{"operation": "add", "content": "User prefers detailed answers."}],
                },
            ),
            QdrantRecord(
                point_id="add-2",
                payload={
                    "account_id": "acc-1",
                    "project_id": project_id,
                    "api_key_uuid": "key-1",
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "request_id": "request-add-1",
                    "status": "ok",
                    "request_submitted_at": "2026-06-01T00:00:01+00:00",
                    "buffer_sequence": 2,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": None,
                            "timestamp": "2026-06-01T09:00:23.170Z",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "name": "Read",
                                    "input": {"file_path": "output.log", "limit": 50},
                                    "result": {"content": "File does not exist.", "is_error": True},
                                }
                            ],
                        },
                        {"role": "assistant", "content": "好的，我会更详细。", "timestamp": 1770000001000},
                    ],
                    "memories": [],
                },
            ),
        ]
        records = _filter_fake_qdrant_records(records, filter_)
        return records[:limit], None

    async def update_add_record_payload(self, project_id, add_record_id, payload):
        self.updated_add_records.append((project_id, add_record_id, payload))

    async def scroll_search_records(self, project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        self.search_calls += 1
        self.search_filters.append(filter_)
        return [
            QdrantRecord(
                point_id="search-1",
                payload={
                    "account_id": "acc-1",
                    "project_id": project_id,
                    "api_key_uuid": "key-1",
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "status": "ok",
                    "request_submitted_at": "2026-06-01T00:00:00+00:00",
                    "query": "你要说得详细一点",
                    "memories": [
                        {
                            "id": "mem-1",
                            "memory": "User likes concise answers.",
                            "last_update_at": "2026-05-31 00:00:00",
                        }
                    ],
                },
            ),
            QdrantRecord(
                point_id="search-2",
                payload={
                    "account_id": "acc-1",
                    "project_id": project_id,
                    "api_key_uuid": "key-1",
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "status": "ok",
                    "request_submitted_at": "2026-06-01T00:01:00+00:00",
                    "query": "下次回答也详细点",
                    "memories": [
                        {
                            "id": "mem-1",
                            "memory": "User likes concise answers.",
                            "last_update_at": "2026-05-31 00:00:00",
                        }
                    ],
                },
            ),
        ], None


class FakeDatabaseClients:
    def __init__(self) -> None:
        self.qdrant = FakeQdrant()
        self.neo4j = SimpleNamespace()


def _filter_fake_qdrant_records(records: list[QdrantRecord], filter_) -> list[QdrantRecord]:
    if filter_ is None:
        return records
    must_not = getattr(filter_, "must_not", None) or []
    return [record for record in records if not any(_fake_condition_matches(record.payload, cond) for cond in must_not)]


def _fake_condition_matches(payload: dict, condition) -> bool:
    key = getattr(condition, "key", None)
    match = getattr(condition, "match", None)
    if key is None or match is None:
        return False
    return payload.get(key) == getattr(match, "value", None)


class FakeMemoryReader:
    def __init__(self, *, clients: FakeDatabaseClients | None = None, memories: list[MemoryView] | None = None) -> None:
        self.clients = clients
        self.filters: SearchFilter | None = None
        self.memories = memories

    async def list_memories(self, ctx: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        self.filters = filters
        memories = self.memories
        if memories is None:
            memories = [
                MemoryView(
                    memory_id="mem-added-1",
                    project_id=ctx.project_id,
                    content="User prefers detailed answers.",
                    mem_type="fact",
                    status="active",
                    request_id="request-add-1",
                    session_id=ctx.session_id,
                    created_at=datetime(2026, 6, 1, tzinfo=UTC),
                )
            ]
        return memories, None

    async def list_add_records(self, ctx: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        if self.clients is None:
            return [], None
        return await self.clients.qdrant.scroll_add_records(ctx.project_id, filter_=filters, limit=limit, cursor=cursor)

    async def list_search_records(self, ctx: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        if self.clients is None:
            return [], None
        return await self.clients.qdrant.scroll_search_records(
            ctx.project_id, filter_=filters, limit=limit, cursor=cursor
        )


class FakeQueryRewriter:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rewrite(self, original_query: str) -> SupplementalSearchQuery:
        self.queries.append(original_query)
        return SupplementalSearchQuery(query=f"memory preference {original_query}")


class FakeSearchPipeline:
    def __init__(self) -> None:
        self.inputs: list[SearchPipelineInput] = []

    async def search(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> SearchPipelineResult:
        self.inputs.append(inp)
        return SearchPipelineResult(
            status="ok",
            memories=[
                MemorySearchItem(
                    id=f"supplemental-{len(self.inputs)}",
                    memory=f"Supplemental memory for {inp.query}",
                    last_update_at="2026-06-01 00:00:00",
                )
            ],
        )


class FakeMemoryWriter:
    def __init__(self, *, clients: FakeDatabaseClients | None = None) -> None:
        self.clients = clients

    async def patch_add_record(self, ctx: MemoryRequestContext, add_record_id: str, payload: dict):
        if self.clients is not None:
            await self.clients.qdrant.update_add_record_payload(ctx.project_id, add_record_id, payload)


class FakeLlmClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None
        self.task = None

    async def chat(self, task, messages, format_parser=None, **kwargs):
        self.task = task
        self.messages = messages
        parsed = format_parser(self.content) if format_parser else None
        return ChatResponse(finish_reason="stop", content=self.content, parsed=parsed)


class FakeHandler:
    async def run(self, inp: FeedbackPipelineInput, context: MemoryRequestContext):
        return FeedbackPipelineResult(status="ok", message="fake implicit handler")


class FakeCollector:
    async def collect(self, context: MemoryRequestContext):
        return [
            ImplicitFeedbackSessionMaterial(session_id="session-1"),
            ImplicitFeedbackSessionMaterial(session_id="session-2"),
        ]


class FakeSignalDetector:
    def __init__(self) -> None:
        self.session_ids: list[str] = []

    async def detect(self, material):
        self.session_ids.append(material.session_id)
        if material.session_id == "session-1":
            return ImplicitFeedbackSignalResult(
                signals=[
                    ImplicitFeedbackSignal(
                        round_index=0,
                        category="long_term",
                        reason="User asks for more detail as a durable preference.",
                    ),
                    ImplicitFeedbackSignal(
                        round_index=0,
                        category="task_temporary",
                        reason="The current answer was too brief.",
                    ),
                ]
            )
        return ImplicitFeedbackSignalResult()


class FakeActionPlanner:
    def __init__(self) -> None:
        self.calls = []

    async def plan(self, *, round_, signals, memories):
        self.calls.append((round_, signals, memories))
        return [
            FeedbackUpdateAction(
                target_memory_id="mem-1",
                before_content="User likes concise answers.",
                after_content="User prefers detailed answers.",
            )
        ]


class FakeExecutor:
    def __init__(self) -> None:
        self.actions = []

    async def execute(self, actions: list[FeedbackActionResult], context: MemoryRequestContext):
        self.actions.extend(actions)
        return actions


@pytest.mark.asyncio
async def test_implicit_collector_groups_session_messages_and_dedupes_memories() -> None:
    clients = FakeDatabaseClients()
    memory_reader = FakeMemoryReader(clients=clients)
    query_rewriter = FakeQueryRewriter()
    search_pipeline = FakeSearchPipeline()
    sessions = await ImplicitFeedbackRecordCollector(
        memory_reader=memory_reader,
        memory_writer=FakeMemoryWriter(clients=clients),
        query_rewriter=query_rewriter,
        search_pipeline=search_pipeline,
    ).collect(make_context())

    assert len(sessions) == 1
    material = sessions[0]
    assert material.session_id == "session-1"
    assert [message.get("content") for message in material.messages] == [
        "你要说得详细一点",
        "好的，我会更详细。",
    ]
    assert all("tool_calls" not in message for message in material.messages)
    assert len(material.rounds) == 1
    assert material.source_add_record_ids == ["add-1", "add-2"]
    assert [message.get("content") for message in material.rounds[0].messages] == [
        "你要说得详细一点",
        "好的，我会更详细。",
    ]
    assert [memory.id for memory in material.memories] == ["mem-added-1", "mem-1", "supplemental-1", "supplemental-2"]
    assert [memory.memory for memory in material.memories] == [
        "User prefers detailed answers.",
        "User likes concise answers.",
        "Supplemental memory for memory preference 你要说得详细一点",
        "Supplemental memory for memory preference 下次回答也详细点",
    ]
    assert memory_reader.filters is not None
    assert query_rewriter.queries == ["你要说得详细一点", "下次回答也详细点"]
    assert [item.query for item in search_pipeline.inputs] == [
        "memory preference 你要说得详细一点",
        "memory preference 下次回答也详细点",
    ]
    assert [item.search_pipeline for item in search_pipeline.inputs] == ["vanilla", "vanilla"]


def _filter_keys(filter_) -> set[str]:
    return {condition.key for condition in filter_.must}


@pytest.mark.asyncio
async def test_implicit_collector_defaults_to_user_scope() -> None:
    clients = FakeDatabaseClients()

    await ImplicitFeedbackRecordCollector(
        memory_reader=FakeMemoryReader(clients=clients),
        memory_writer=FakeMemoryWriter(clients=clients),
        query_rewriter=FakeQueryRewriter(),
        search_pipeline=FakeSearchPipeline(),
    ).collect(make_context())

    assert "session_id" not in _filter_keys(clients.qdrant.add_filters[0])
    assert "session_id" not in _filter_keys(clients.qdrant.search_filters[0])


@pytest.mark.asyncio
async def test_implicit_collector_can_filter_one_session() -> None:
    clients = FakeDatabaseClients()

    await ImplicitFeedbackRecordCollector(
        memory_reader=FakeMemoryReader(clients=clients),
        memory_writer=FakeMemoryWriter(clients=clients),
        query_rewriter=FakeQueryRewriter(),
        search_pipeline=FakeSearchPipeline(),
    ).collect(make_context(), scope="session")

    assert "session_id" in _filter_keys(clients.qdrant.add_filters[0])
    assert "session_id" in _filter_keys(clients.qdrant.search_filters[0])


@pytest.mark.asyncio
async def test_implicit_collector_drops_add_record_memory_without_real_memory_id() -> None:
    clients = FakeDatabaseClients()

    sessions = await ImplicitFeedbackRecordCollector(
        memory_reader=FakeMemoryReader(clients=clients, memories=[]),
        memory_writer=FakeMemoryWriter(clients=clients),
        query_rewriter=FakeQueryRewriter(),
        search_pipeline=FakeSearchPipeline(),
    ).collect(make_context())

    assert len(sessions) == 1
    assert [memory.id for memory in sessions[0].memories] == ["mem-1", "supplemental-1", "supplemental-2"]


@pytest.mark.asyncio
async def test_implicit_collector_skips_feedback_processed_add_records() -> None:
    clients = FakeDatabaseClients()
    add_records, cursor = await clients.qdrant.scroll_add_records("proj-1")
    for record in add_records:
        record.payload["feedback_processed"] = True

    async def scroll_processed_add_records(project_id, *, filter_=None, limit=50, cursor=None, order_by=None):
        return _filter_fake_qdrant_records(add_records, filter_)[:limit], cursor

    clients.qdrant.scroll_add_records = scroll_processed_add_records

    sessions = await ImplicitFeedbackRecordCollector(
        memory_reader=FakeMemoryReader(clients=clients),
        memory_writer=FakeMemoryWriter(clients=clients),
        query_rewriter=FakeQueryRewriter(),
        search_pipeline=FakeSearchPipeline(),
    ).collect(make_context())

    assert sessions == []


@pytest.mark.asyncio
async def test_implicit_signal_detector_uses_rounds_without_memories() -> None:
    clients = FakeDatabaseClients()
    sessions = await ImplicitFeedbackRecordCollector(
        memory_reader=FakeMemoryReader(clients=clients),
        memory_writer=FakeMemoryWriter(clients=clients),
        query_rewriter=FakeQueryRewriter(),
        search_pipeline=FakeSearchPipeline(),
    ).collect(make_context())
    llm_client = FakeLlmClient(
        """
            {
              "signals": [
                {
                  "round_index": 0,
                  "category": "long_term",
                  "reason": "用户说你要说得详细一点"
                }
              ]
        }
        """
    )

    result = await ImplicitFeedbackSignalDetector(llm_client=llm_client).detect(sessions[0])

    assert len(result.signals) == 1
    assert result.signals[0].round_index == 0
    assert result.signals[0].category == "long_term"
    assert llm_client.task == "feedback.implicit.detect_signals"
    assert "memories" not in llm_client.messages[1]["content"]
    assert "rounds" in llm_client.messages[1]["content"]


@pytest.mark.asyncio
async def test_implicit_action_planner_passes_feedback_categories_to_llm() -> None:
    llm_client = FakeLlmClient('{"actions":[]}')
    planner = ImplicitFeedbackActionPlanner(llm_client=llm_client)

    actions = await planner.plan(
        round_=ImplicitFeedbackRound(
            messages=[
                {"role": "user", "content": "这个项目里 review 要更严格看测试"},
                {"role": "assistant", "content": "知道了。"},
            ]
        ),
        signals=[
            ImplicitFeedbackSignal(
                round_index=0,
                category="scenario_specific",
                reason="反馈限定在当前项目 review 场景",
            ),
            ImplicitFeedbackSignal(
                round_index=0,
                category="long_term",
                reason="用户要求记住一个通用代码风格偏好",
            ),
        ],
        memories=[],
    )

    assert actions == []
    assert llm_client.task == "feedback.implicit.plan_actions"
    assert '"category": "scenario_specific"' in llm_client.messages[1]["content"]
    assert '"category": "long_term"' in llm_client.messages[1]["content"]
    assert '"signals"' in llm_client.messages[1]["content"]


@pytest.mark.asyncio
async def test_implicit_handler_collects_sessions_and_detects_signals() -> None:
    class MarkingCollector(FakeCollector):
        def __init__(self) -> None:
            self.marked: list[str] = []

        async def collect(self, context: MemoryRequestContext):
            return [
                ImplicitFeedbackSessionMaterial(
                    session_id="session-1",
                    rounds=[
                        {
                            "messages": [
                                {"role": "user", "content": "你要说得详细一点"},
                                {"role": "assistant", "content": "好的，我会更详细。"},
                            ]
                        }
                    ],
                    memories=[
                        {
                            "id": "mem-1",
                            "memory": "User likes concise answers.",
                            "last_update_at": "2026-06-01 00:00:00",
                        }
                    ],
                    source_add_record_ids=["add-1"],
                ),
                ImplicitFeedbackSessionMaterial(session_id="session-2", source_add_record_ids=["add-2"]),
            ]

        async def mark_feedback_processed(self, context: MemoryRequestContext, add_record_ids: list[str]):
            self.marked.extend(add_record_ids)

    collector = MarkingCollector()
    signal_detector = FakeSignalDetector()
    action_planner = FakeActionPlanner()
    executor = FakeExecutor()
    handler = ImplicitFeedbackHandler(
        collector=collector,
        signal_detector=signal_detector,
        action_planner=action_planner,
        executor=executor,
    )

    result = await handler.run(FeedbackPipelineInput(), make_context())

    assert result.status == "ok"
    assert result.message == "processed 2 implicit feedback signals in 2 sessions"
    assert signal_detector.session_ids == ["session-1", "session-2"]
    assert collector.marked == ["add-1", "add-2"]
    assert len(result.actions) == 1
    assert result.actions[0].action == "update"
    assert len(action_planner.calls) == 1
    assert action_planner.calls[0][0].messages[0]["content"] == "你要说得详细一点"
    assert [signal.category for signal in action_planner.calls[0][1]] == ["long_term", "task_temporary"]
    assert action_planner.calls[0][2][0].id == "mem-1"
    assert executor.actions == result.actions


@pytest.mark.asyncio
async def test_implicit_handler_groups_multiple_signals_by_round() -> None:
    class Collector:
        def __init__(self) -> None:
            self.marked: list[str] = []

        async def collect(self, context: MemoryRequestContext):
            return [
                ImplicitFeedbackSessionMaterial(
                    session_id="session-1",
                    rounds=[
                        ImplicitFeedbackRound(
                            messages=[
                                {"role": "user", "content": "这本书我听说过，但我不喜欢恐怖、血腥题材。"},
                                {"role": "assistant", "content": "我换一本。"},
                            ]
                        ),
                        ImplicitFeedbackRound(
                            messages=[
                                {"role": "user", "content": "短途飞行两小时内能看完吗？"},
                                {"role": "assistant", "content": "我推荐短篇。"},
                            ]
                        ),
                    ],
                    memories=[
                        MemorySearchItem(
                            id="mem-1",
                            memory="User was recommended a horror short story.",
                            last_update_at="2026-06-01 00:00:00",
                        )
                    ],
                    source_add_record_ids=["add-1"],
                )
            ]

        async def mark_feedback_processed(self, context: MemoryRequestContext, add_record_ids: list[str]):
            self.marked.extend(add_record_ids)

    class Detector:
        async def detect(self, material):
            return ImplicitFeedbackSignalResult(
                signals=[
                    ImplicitFeedbackSignal(
                        round_index=0,
                        category="task_temporary",
                        reason="Change the current recommendation.",
                    ),
                    ImplicitFeedbackSignal(
                        round_index=0,
                        category="long_term",
                        reason="The user dislikes horror and bloody books.",
                    ),
                    ImplicitFeedbackSignal(
                        round_index=1,
                        category="task_temporary",
                        reason="The two-hour limit is specific to the current short flight.",
                    ),
                    ImplicitFeedbackSignal(
                        round_index=99,
                        category="long_term",
                        reason="Invalid index should be ignored.",
                    ),
                ]
            )

    collector = Collector()
    action_planner = FakeActionPlanner()
    executor = FakeExecutor()
    handler = ImplicitFeedbackHandler(
        collector=collector,
        signal_detector=Detector(),
        action_planner=action_planner,
        executor=executor,
    )

    result = await handler.run(FeedbackPipelineInput(), make_context())

    assert result.message == "processed 4 implicit feedback signals in 1 sessions"
    assert len(action_planner.calls) == 2
    assert [signal.category for signal in action_planner.calls[0][1]] == ["task_temporary", "long_term"]
    assert [signal.category for signal in action_planner.calls[1][1]] == ["task_temporary"]
    assert collector.marked == ["add-1"]


@pytest.mark.asyncio
async def test_default_pipeline_uses_implicit_handler() -> None:
    pipeline = DefaultFeedbackPipeline(implicit_handler=FakeHandler())

    result = await pipeline.feedback(FeedbackPipelineInput(), make_context())

    assert result.status == "ok"
    assert result.message == "fake implicit handler"
