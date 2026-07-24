import json
from types import SimpleNamespace

import mindmemos.workers.memory_add as memory_add
import pytest
from mindmemos.typing.service import AddPipelineSyncResult, MemoryAddEventItem

from mindmemos.infra.kafka import ConsumedMessage


def make_message(body: dict) -> ConsumedMessage:
    return ConsumedMessage(
        topic=memory_add.TOPIC,
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(body).encode("utf-8"),
    )


class FakeRecorder:
    def __init__(self) -> None:
        self.processing_calls = []
        self.completed_calls = []
        self.failed_calls = []

    async def mark_add_processing(self, ctx, add_record_id):
        self.processing_calls.append((ctx, add_record_id))

    async def record_add_input(self, inp, *, ctx, request_submitted_at, add_record_id, status, **kwargs):
        self.processing_calls.append((ctx, add_record_id))

    async def mark_add_completed(self, ctx, add_record_id, result):
        self.completed_calls.append((ctx, add_record_id, result))

    async def mark_add_failed(self, ctx, add_record_id, error):
        self.failed_calls.append((ctx, add_record_id, error))


@pytest.mark.asyncio
async def test_memory_add_worker_uses_algorithm_bound_add_pipeline_sync_entry(monkeypatch) -> None:
    calls = []
    pipeline_names = []

    class ConfiguredPipeline:
        async def add_sync(self, inp, context, *, add_record_id=None):
            calls.append(SimpleNamespace(inp=inp, context=context, add_record_id=add_record_id))
            return SimpleNamespace(memories=[])

    monkeypatch.setattr(
        memory_add,
        "create_pipeline",
        lambda *, type, name: pipeline_names.append(name) or ConfiguredPipeline(),
    )

    await memory_add.handle_memory_add(
        make_message(
            {
                "context": {
                    "request_id": "req-1",
                    "account_id": "acc-1",
                    "project_id": "proj-1",
                    "api_key_uuid": "key-1",
                    "memory_algorithm": "vanilla",
                    "user_id": "user-1",
                    "session_id": "session-1",
                },
                "input": {
                    "messages": [{"text": "Kafka async add"}],
                    "mode": "async",
                },
            }
        )
    )

    assert len(calls) == 1
    assert pipeline_names == ["vanilla_add"]
    assert calls[0].context.request_id == "req-1"
    # AddPipelineInput.timestamp defaults to wall-clock millis, so compare the
    # meaningful fields instead of full-object equality.
    assert calls[0].inp.mode == "sync"
    assert [m.text for m in calls[0].inp.messages] == ["Kafka async add"]


@pytest.mark.asyncio
async def test_memory_add_worker_marks_processing_and_forwards_add_record_id(monkeypatch) -> None:
    recorder = FakeRecorder()
    forwarded: list = []

    class ConfiguredPipeline:
        async def add_sync(self, inp, context, *, add_record_id=None):
            forwarded.append(add_record_id)
            return AddPipelineSyncResult(
                status="ok",
                memories=[MemoryAddEventItem(operation="add", content="remembered", memory_id="mem-1")],
            )

    monkeypatch.setattr(
        memory_add,
        "create_pipeline",
        lambda *, type, name: ConfiguredPipeline(),
    )
    monkeypatch.setattr(memory_add, "MemoryOperationRecorder", lambda: recorder)

    await memory_add.handle_memory_add(
        make_message(
            {
                "add_record_id": "add-rec-1",
                "context": {
                    "request_id": "req-1",
                    "account_id": "acc-1",
                    "project_id": "proj-1",
                    "api_key_uuid": "key-1",
                    "memory_algorithm": "vanilla",
                    "user_id": "user-1",
                    "session_id": "session-1",
                },
                "input": {
                    "messages": [{"text": "Kafka async add"}],
                    "mode": "async",
                },
            }
        )
    )

    assert [(ctx.project_id, add_record_id) for ctx, add_record_id in recorder.processing_calls] == [
        ("proj-1", "add-rec-1")
    ]
    # The worker forwards the id to add_sync, which owns the output write-back, so
    # the worker no longer double-writes via mark_add_completed.
    assert forwarded == ["add-rec-1"]
    assert recorder.completed_calls == []
    assert recorder.failed_calls == []


@pytest.mark.asyncio
async def test_memory_add_worker_re_raises_pipeline_exception(monkeypatch) -> None:
    """When add_sync raises, the worker re-raises so the consumer can retry/DLQ."""

    class FailingPipeline:
        async def add_sync(self, inp, context, *, add_record_id=None):
            raise RuntimeError("pipeline db error")

    monkeypatch.setattr(
        memory_add,
        "create_pipeline",
        lambda *, type, name: FailingPipeline(),
    )

    with pytest.raises(RuntimeError, match="pipeline db error"):
        await memory_add.handle_memory_add(
            make_message(
                {
                    "context": {
                        "request_id": "req-fail",
                        "account_id": "acc-1",
                        "project_id": "proj-1",
                        "api_key_uuid": "key-1",
                        "memory_algorithm": "vanilla",
                        "user_id": "user-1",
                        "session_id": "session-1",
                    },
                    "input": {
                        "messages": [{"text": "This will fail."}],
                        "mode": "sync",
                    },
                }
            )
        )


@pytest.mark.asyncio
async def test_memory_add_worker_patches_async_add_record_on_failure(monkeypatch) -> None:
    recorder = FakeRecorder()

    class FailingPipeline:
        async def add_sync(self, inp, context, *, add_record_id=None):
            raise RuntimeError("pipeline db error")

    monkeypatch.setattr(
        memory_add,
        "create_pipeline",
        lambda *, type, name: FailingPipeline(),
    )
    monkeypatch.setattr(memory_add, "MemoryOperationRecorder", lambda: recorder)

    with pytest.raises(RuntimeError, match="pipeline db error"):
        await memory_add.handle_memory_add(
            make_message(
                {
                    "add_record_id": "add-rec-fail",
                    "context": {
                        "request_id": "req-fail",
                        "account_id": "acc-1",
                        "project_id": "proj-1",
                        "api_key_uuid": "key-1",
                        "memory_algorithm": "vanilla",
                        "user_id": "user-1",
                        "session_id": "session-1",
                    },
                    "input": {
                        "messages": [{"text": "This will fail."}],
                        "mode": "sync",
                    },
                }
            )
        )

    assert [(ctx.project_id, add_record_id) for ctx, add_record_id in recorder.processing_calls] == [
        ("proj-1", "add-rec-fail")
    ]
    assert [(ctx.project_id, add_record_id, error) for ctx, add_record_id, error in recorder.failed_calls] == [
        ("proj-1", "add-rec-fail", "pipeline db error")
    ]
    assert recorder.completed_calls == []
