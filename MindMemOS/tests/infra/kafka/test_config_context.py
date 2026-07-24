from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aiokafka import TopicPartition
from mindmemos.config import get_config, init_config, reset_config, update_config
from mindmemos.config.app import KafkaConfig, KafkaConsumerConfig
from mindmemos.infra.kafka.config_context import CONFIG_CONTEXT_HEADER
from mindmemos.infra.kafka.consumer import KafkaConsumer, _is_idle_status
from mindmemos.infra.kafka.producer import KafkaProducer


class FakeProducerClient:
    def __init__(self) -> None:
        self.calls = []

    async def send(self, topic, *, value, key=None, headers=None):
        self.calls.append(
            {
                "topic": topic,
                "value": value,
                "key": key,
                "headers": headers or [],
            }
        )
        # Mirror aiokafka: send() returns a future that resolves on broker ack.
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut


class FailingProducerClient:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, topic, *, value, key=None, headers=None):
        self.calls += 1
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        fut.set_exception(RuntimeError("broker rejected"))
        return fut


class PendingProducerClient:
    def __init__(self) -> None:
        self.future: asyncio.Future[None] | None = None

    async def send(self, topic, *, value, key=None, headers=None):
        self.future = asyncio.get_running_loop().create_future()
        return self.future


class FakeStatusConsumer:
    def __init__(self) -> None:
        self.tps = {TopicPartition("memory.add", 0), TopicPartition("memory.feedback", 0)}
        self.ends = {
            TopicPartition("memory.add", 0): 15,
            TopicPartition("memory.feedback", 0): 8,
        }
        self.commits = {
            TopicPartition("memory.add", 0): 10,
            TopicPartition("memory.feedback", 0): 8,
        }

    def assignment(self):
        return self.tps

    async def end_offsets(self, assignment):
        return {tp: self.ends[tp] for tp in assignment}

    async def committed(self, tp):
        return self.commits[tp]


@pytest.mark.asyncio
async def test_producer_injects_current_config_overrides_into_headers() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        update_config(project_config={"pipelines": {"get": "project_get"}})
        client = FakeProducerClient()
        producer = KafkaProducer(KafkaConfig())
        producer._producer = client

        # wait=True so broker-ack callbacks complete before we assert.
        await producer.send("memory.add", {"ok": True}, headers={"x-custom": "yes"}, wait=True)

        headers = {name: value.decode("utf-8") for name, value in client.calls[0]["headers"]}
        assert headers["x-custom"] == "yes"
        assert CONFIG_CONTEXT_HEADER in headers
        assert '"project_config":{"pipelines":{"get":"project_get"}}' in headers[CONFIG_CONTEXT_HEADER]
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_consumer_status_summary_groups_topics_with_pipe_separator() -> None:
    consumer = KafkaConsumer(
        KafkaConfig(),
        KafkaConsumerConfig(group_id="test", topics=["memory.add", "memory.feedback"], max_retries=0),
        lambda _msg: asyncio.sleep(0),
        dlq_producer=KafkaProducer(KafkaConfig()),
    )
    consumer._consumer = FakeStatusConsumer()
    consumer._running_by_topic["memory.add"] = 2

    assert await consumer._status_summary() == "memory.add:queued=3,running=2 | memory.feedback:queued=0,running=0"


def test_idle_kafka_status_is_detected_across_topics() -> None:
    assert _is_idle_status("memory.add:queued=0,running=0 | memory.feedback:queued=0,running=0")
    assert not _is_idle_status("memory.add:queued=1,running=0")
    assert not _is_idle_status("no_assignment")


@pytest.mark.asyncio
async def test_producer_does_not_retry_after_delivery_failure() -> None:
    client = FailingProducerClient()
    producer = KafkaProducer(KafkaConfig(producer_max_retries=3))
    producer._producer = client

    with pytest.raises(RuntimeError, match="broker rejected"):
        await producer.send("memory.add", {"ok": True}, wait=True)

    assert client.calls == 1


@pytest.mark.asyncio
async def test_producer_flush_waits_for_returned_ack_future() -> None:
    client = PendingProducerClient()
    producer = KafkaProducer(KafkaConfig())
    producer._producer = client

    future = await producer.send("memory.add", {"ok": True})
    assert future is client.future
    assert future in producer._pending

    future.set_result(None)
    await asyncio.sleep(0)
    await producer.flush()

    assert not producer._pending


@pytest.mark.asyncio
async def test_consumer_binds_config_context_for_handler_and_restores_afterward() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        update_config(project_config={"pipelines": {"get": "outer_get"}})
        seen: list[str] = []

        async def handler(_msg):
            seen.append(get_config().pipelines["get"])

        consumer = KafkaConsumer(
            KafkaConfig(),
            KafkaConsumerConfig(group_id="test", max_retries=0),
            handler,
            dlq_producer=KafkaProducer(KafkaConfig()),
        )
        record = SimpleNamespace(
            topic="memory.add",
            partition=0,
            offset=1,
            key=None,
            value=b"{}",
            headers=[
                (
                    CONFIG_CONTEXT_HEADER,
                    b'{"project_config":{"pipelines":{"get":"header_get"}}}',
                )
            ],
            timestamp=0,
        )

        await consumer._handle_record(record)

        assert seen == ["header_get"]
        assert get_config().pipelines["get"] == "outer_get"
    finally:
        reset_config()
