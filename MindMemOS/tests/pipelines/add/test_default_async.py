"""Tests for add_async Kafka publishing in add pipelines."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import mindmemos.pipelines.add.vanilla.vanilla_add as vanilla_mod
import pytest
from mindmemos.pipelines.add.default import DefaultAddPipeline
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import AddPipelineInput

from mindmemos.config import TextProcessingConfig, VanillaAddConfig
from mindmemos.pipelines.add.vanilla import VanillaAddPipeline


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-async-1",
        account_id="acc-async-1",
        project_id="proj-async-1",
        api_key_uuid="key-async-1",
        memory_algorithm="vanilla",
        user_id="user-async-1",
        session_id="session-async-1",
    )


def make_pipeline() -> VanillaAddPipeline:
    return VanillaAddPipeline(
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        consistency="fast",
        vanilla_add_config=VanillaAddConfig(),
        llm_client=None,
        embed_client=None,
    )


def make_default_pipeline() -> DefaultAddPipeline:
    return DefaultAddPipeline(
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        consistency="fast",
    )


@pytest.mark.asyncio
async def test_add_async_publishes_to_kafka(monkeypatch) -> None:
    """Verify add_async serializes input+context and publishes to memory.add topic."""
    mock_producer = AsyncMock()
    monkeypatch.setattr("mindmemos.infra.kafka.get_producer", lambda: mock_producer)
    monkeypatch.setattr(
        vanilla_mod,
        "get_config",
        lambda: SimpleNamespace(kafka=SimpleNamespace(enabled=True)),
    )

    pipeline = make_pipeline()
    inp = AddPipelineInput(messages=[{"text": "Kafka async add"}])
    ctx = make_context()

    result = await pipeline.add_async(inp, ctx)

    assert result.status == "queued"
    assert result.memories == []

    # Producer.send called with correct arguments
    mock_producer.send.assert_awaited_once()
    call = mock_producer.send.call_args
    assert call.args[0] == "memory.add"
    assert call.kwargs["dispatch_key"] == "proj-async-1:user-async-1"
    assert "wait" not in call.kwargs

    value = call.kwargs["value"]
    assert "context" in value
    assert "input" in value
    assert value["context"]["request_id"] == "req-async-1"
    assert value["context"]["account_id"] == "acc-async-1"
    assert value["input"]["messages"] == inp.model_dump(by_alias=True)["messages"]


@pytest.mark.asyncio
async def test_add_async_includes_add_record_id_when_supplied(monkeypatch) -> None:
    mock_producer = AsyncMock()
    monkeypatch.setattr("mindmemos.infra.kafka.get_producer", lambda: mock_producer)
    monkeypatch.setattr(
        vanilla_mod,
        "get_config",
        lambda: SimpleNamespace(kafka=SimpleNamespace(enabled=True)),
    )

    pipeline = make_pipeline()
    inp = AddPipelineInput(messages=[{"text": "Kafka async add"}])
    ctx = make_context()

    await pipeline.add_async(inp, ctx, add_record_id="add-rec-async-1")

    value = mock_producer.send.call_args.kwargs["value"]
    assert value["add_record_id"] == "add-rec-async-1"


@pytest.mark.asyncio
async def test_default_add_async_includes_add_record_id_when_supplied(monkeypatch) -> None:
    mock_producer = AsyncMock()
    monkeypatch.setattr("mindmemos.pipelines.add.default.get_producer", lambda: mock_producer)

    pipeline = make_default_pipeline()
    inp = AddPipelineInput(messages=[{"text": "Default async add"}])
    ctx = make_context()

    await pipeline.add_async(inp, ctx, add_record_id="add-rec-default-1")

    assert mock_producer.send.call_args.kwargs["dispatch_key"] == "proj-async-1:user-async-1"
    assert "wait" not in mock_producer.send.call_args.kwargs
    value = mock_producer.send.call_args.kwargs["value"]
    assert value["add_record_id"] == "add-rec-default-1"
    assert value["context"]["request_id"] == "req-async-1"
    assert value["input"]["messages"] == inp.model_dump(mode="json", by_alias=True)["messages"]


@pytest.mark.asyncio
async def test_add_async_raises_runtime_error_when_kafka_disabled(monkeypatch) -> None:
    """Verify add_async raises RuntimeError when kafka.enabled=False."""
    monkeypatch.setattr(
        vanilla_mod,
        "get_config",
        lambda: SimpleNamespace(kafka=SimpleNamespace(enabled=False)),
    )

    pipeline = make_pipeline()
    inp = AddPipelineInput(messages=[{"text": "Kafka disabled test"}])
    ctx = make_context()

    with pytest.raises(RuntimeError, match="kafka.enabled=true"):
        await pipeline.add_async(inp, ctx)


@pytest.mark.asyncio
async def test_add_async_message_round_trips_through_worker_deserialization(monkeypatch) -> None:
    """Verify the message published by add_async can be deserialized by the worker."""
    import json

    import mindmemos.workers.memory_add as memory_add_mod

    from mindmemos.infra.kafka import ConsumedMessage

    captured_value: dict = {}

    mock_producer = AsyncMock()

    async def capture_send(topic, value, **kwargs):
        captured_value.update(value)

    mock_producer.send = capture_send
    monkeypatch.setattr("mindmemos.infra.kafka.get_producer", lambda: mock_producer)
    monkeypatch.setattr(
        vanilla_mod,
        "get_config",
        lambda: SimpleNamespace(kafka=SimpleNamespace(enabled=True)),
    )

    pipeline = make_pipeline()
    inp = AddPipelineInput(messages=[{"text": "Round trip test"}])
    ctx = make_context()

    await pipeline.add_async(inp, ctx)

    msg = ConsumedMessage(
        topic="memory.add",
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(captured_value).encode("utf-8"),
    )

    worker_calls = []
    pipeline_names = []

    class FakeWorkerPipeline:
        async def add_sync(self, inp, context, *, add_record_id=None):
            worker_calls.append(SimpleNamespace(inp=inp, context=context, add_record_id=add_record_id))
            return SimpleNamespace(memories=[])

    monkeypatch.setattr(
        memory_add_mod,
        "create_pipeline",
        lambda *, type, name: pipeline_names.append(name) or FakeWorkerPipeline(),
    )

    await memory_add_mod.handle_memory_add(msg)

    assert len(worker_calls) == 1
    assert pipeline_names == ["vanilla_add"]
    assert worker_calls[0].context.request_id == "req-async-1"
    assert worker_calls[0].context.account_id == "acc-async-1"
    assert worker_calls[0].inp.mode == "sync"
    assert worker_calls[0].inp.messages[0].text == "Round trip test"
