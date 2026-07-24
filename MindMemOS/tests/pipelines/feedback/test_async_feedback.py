from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from mindmemos.pipelines.feedback.default import MEMORY_FEEDBACK_TOPIC, DefaultFeedbackPipeline
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import FeedbackPipelineInput, FeedbackPipelineResult

from mindmemos.infra.kafka import ConsumedMessage
from mindmemos.workers import memory_feedback


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


@dataclass
class CapturedProducer:
    topic: str | None = None
    value: dict | None = None
    dispatch_key: str | None = None

    async def send(self, topic: str, value: dict, *, dispatch_key: str | None = None, **kwargs) -> None:
        self.topic = topic
        self.value = value
        self.dispatch_key = dispatch_key


@pytest.mark.asyncio
async def test_default_feedback_async_queues_kafka_task(monkeypatch) -> None:
    producer = CapturedProducer()
    monkeypatch.setattr("mindmemos.pipelines.feedback.default.get_producer", lambda: producer)
    pipeline = DefaultFeedbackPipeline(explicit_handler=object(), implicit_handler=object())

    result = await pipeline.feedback_async(FeedbackPipelineInput(feedback="queue this", mode="async"), make_context())

    assert result.status == "queued"
    assert result.message == "feedback queued"
    assert producer.topic == MEMORY_FEEDBACK_TOPIC
    assert producer.dispatch_key == "proj-1:user-1"
    assert producer.value is not None
    assert producer.value["context"]["project_id"] == "proj-1"
    assert producer.value["input"]["feedback"] == "queue this"
    assert producer.value["input"]["mode"] == "async"


class FakeWorkerPipeline:
    def __init__(self) -> None:
        self.input: FeedbackPipelineInput | None = None
        self.context: MemoryRequestContext | None = None

    async def feedback_sync(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        self.input = inp
        self.context = context
        return FeedbackPipelineResult(status="ok")


@pytest.mark.asyncio
async def test_feedback_worker_executes_queued_task_as_sync(monkeypatch) -> None:
    pipeline = FakeWorkerPipeline()
    monkeypatch.setattr(
        memory_feedback,
        "get_config",
        lambda: SimpleNamespace(pipelines=SimpleNamespace(feedback="default_feedback")),
    )
    monkeypatch.setattr(memory_feedback, "create_pipeline", lambda **kwargs: pipeline)

    msg = ConsumedMessage(
        topic=MEMORY_FEEDBACK_TOPIC,
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(
            {
                "context": make_context().model_dump(mode="json"),
                "input": FeedbackPipelineInput(feedback="queued feedback", mode="async").model_dump(mode="json"),
            }
        ).encode("utf-8"),
    )

    await memory_feedback.handle_memory_feedback(msg)

    assert pipeline.context is not None
    assert pipeline.context.project_id == "proj-1"
    assert pipeline.input is not None
    assert pipeline.input.feedback == "queued feedback"
    assert pipeline.input.mode == "sync"
