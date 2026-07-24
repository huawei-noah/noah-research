"""Asynchronous Kafka consumer wrapper."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Awaitable, Callable

from aiokafka import AIOKafkaConsumer, TopicPartition
from opentelemetry.trace import SpanKind, Status, StatusCode

from ...config import KafkaConfig, KafkaConsumerConfig
from ...logging import extract_trace_context, get_logger, get_tracer, headers_to_carrier
from ..retry import retry_delay
from .config_context import bind_config_context_from_headers
from .dispatcher import DISPATCH_KEY_HEADER, OrderedKeyedDispatcher
from .message import ConsumedMessage
from .producer import KafkaProducer

logger = get_logger(__name__)
tracer = get_tracer(__name__)
_STATUS_INTERVAL_SECONDS = 5.0

MessageHandler = Callable[[ConsumedMessage], Awaitable[None]]


class _OffsetTracker:
    """Track completed offsets per partition and compute safe commits."""

    def __init__(self) -> None:
        self._next: dict[TopicPartition, int] = {}
        self._done: dict[TopicPartition, set[int]] = {}

    def on_submit(self, tp: TopicPartition, offset: int) -> None:
        """Record the first submitted offset for a partition."""
        if tp not in self._next:
            self._next[tp] = offset
            self._done[tp] = set()

    def on_complete(self, tp: TopicPartition, offset: int) -> dict[TopicPartition, int] | None:
        """Mark one message complete and return committable offsets."""
        done = self._done[tp]
        done.add(offset)
        nxt = self._next[tp]
        advanced = False
        while nxt in done:
            done.discard(nxt)
            nxt += 1
            advanced = True
        if not advanced:
            return None
        self._next[tp] = nxt
        return {tp: nxt}


class KafkaConsumer:
    """Run one background Kafka consumer for a configured consumer group."""

    def __init__(
        self,
        kafka_config: KafkaConfig,
        consumer_config: KafkaConsumerConfig,
        handler: MessageHandler,
        dlq_producer: KafkaProducer,
        global_task_semaphore: asyncio.Semaphore | None = None,
    ):
        self._kafka = kafka_config
        self._cfg = consumer_config
        self._handler = handler
        self._dlq_producer = dlq_producer
        self._consumer: AIOKafkaConsumer | None = None
        self._task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None
        self._stopping = False
        self._tracker = _OffsetTracker()
        self._dispatcher: OrderedKeyedDispatcher | None = None
        self._running_by_topic: Counter[str] = Counter()
        self._global_task_semaphore = global_task_semaphore

    async def start(self) -> None:
        if self._consumer is not None:
            return
        self._consumer = AIOKafkaConsumer(
            *self._cfg.topics,
            bootstrap_servers=self._kafka.bootstrap_servers,
            client_id=self._kafka.client_id,
            group_id=self._cfg.group_id,
            enable_auto_commit=False,
            auto_offset_reset=self._cfg.auto_offset_reset,
            max_poll_records=self._cfg.max_poll_records,
            session_timeout_ms=self._cfg.session_timeout_ms,
            heartbeat_interval_ms=self._cfg.heartbeat_interval_ms,
            max_poll_interval_ms=self._cfg.max_poll_interval_ms,
        )
        await self._consumer.start()
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name=f"kafka-consumer-{self._cfg.group_id}")
        self._status_task = asyncio.create_task(self._log_status_loop(), name=f"kafka-status-{self._cfg.group_id}")
        logger.info("kafka consumer started", group_id=self._cfg.group_id, topics=self._cfg.topics)

    async def stop(self) -> None:
        self._stopping = True
        if self._status_task is not None:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            self._status_task = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
            logger.info("kafka consumer stopped", group_id=self._cfg.group_id)

    async def _run(self) -> None:
        assert self._consumer is not None
        dispatcher = OrderedKeyedDispatcher(
            global_max_concurrency=self._cfg.global_max_concurrency,
            per_key_max_concurrency=self._cfg.per_key_max_concurrency,
            max_buffered=self._cfg.max_buffered,
            process=self._handle_record,
            on_complete=self._commit_record,
            shared_global_semaphore=self._global_task_semaphore,
        )
        self._dispatcher = dispatcher
        try:
            async for record in self._consumer:
                if self._stopping:
                    break
                tp = TopicPartition(record.topic, record.partition)
                self._tracker.on_submit(tp, record.offset)
                # submit blocks when the buffer is full, pausing polling to create backpressure.
                await dispatcher.submit(self._dispatch_key(record), record)
            await dispatcher.drain()
        except asyncio.CancelledError:
            await dispatcher.aclose()
            raise
        except Exception:
            logger.exception("kafka consumer loop crashed", group_id=self._cfg.group_id)
            raise

    async def _log_status_loop(self) -> None:
        while True:
            await asyncio.sleep(_STATUS_INTERVAL_SECONDS)
            try:
                summary = await self._status_summary()
            except Exception as exc:
                logger.warning("kafka status failed", group_id=self._cfg.group_id, error=str(exc))
                continue
            log = logger.debug if _is_idle_status(summary) else logger.info
            log("kafka_status", group_id=self._cfg.group_id, summary=summary)

    async def _status_summary(self) -> str:
        assert self._consumer is not None
        assignment = self._consumer.assignment()
        if not assignment:
            return "no_assignment"

        end_offsets = await self._consumer.end_offsets(assignment)
        unfinished_by_topic: Counter[str] = Counter()
        for tp in assignment:
            committed = await self._consumer.committed(tp)
            if committed is None:
                committed = await self._consumer.position(tp)
            unfinished_by_topic[tp.topic] += max(0, end_offsets.get(tp, committed) - committed)

        parts = []
        for topic in sorted(set(unfinished_by_topic) | set(self._running_by_topic) | set(self._cfg.topics)):
            running = self._running_by_topic[topic]
            queued = max(0, unfinished_by_topic[topic] - running)
            parts.append(f"{topic}:queued={queued},running={running}")
        return " | ".join(parts)

    def _dispatch_key(self, record) -> str:
        """Resolve the dispatch key for a consumed Kafka message."""
        for name, value in record.headers or ():
            if name == DISPATCH_KEY_HEADER and value:
                return value.decode("utf-8")
        if record.key:
            return record.key.decode("utf-8")
        # Without an explicit key, fall back to Kafka partition ordering.
        return f"{record.topic}:{record.partition}"

    async def _commit_record(self, record) -> None:
        """Advance offset commits after a message finishes."""
        assert self._consumer is not None
        tp = TopicPartition(record.topic, record.partition)
        offsets = self._tracker.on_complete(tp, record.offset)
        if offsets is None:
            return
        try:
            await self._consumer.commit(offsets)
        except Exception:
            logger.exception("kafka offset commit failed", group_id=self._cfg.group_id, offsets=str(offsets))

    async def _handle_record(self, record) -> None:
        self._running_by_topic[record.topic] += 1
        carrier = headers_to_carrier(record.headers)
        parent_ctx = extract_trace_context(carrier)
        msg = ConsumedMessage(
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            key=record.key.decode("utf-8") if record.key else None,
            value=record.value,
            headers=carrier,
            timestamp_ms=record.timestamp,
        )

        with tracer.start_as_current_span(
            f"kafka.process {record.topic}",
            context=parent_ctx,
            kind=SpanKind.CONSUMER,
        ) as span:
            span.set_attribute("messaging.system", "kafka")
            span.set_attribute("messaging.source.name", record.topic)
            span.set_attribute("messaging.kafka.partition", record.partition)
            span.set_attribute("messaging.kafka.message.offset", record.offset)
            await asyncio.sleep(0)  # heartbeat
            try:
                with bind_config_context_from_headers(msg.headers):
                    await self._process_with_retry(msg)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                await self._to_dlq(msg, exc)
            finally:
                self._running_by_topic[record.topic] -= 1
                if self._running_by_topic[record.topic] <= 0:
                    del self._running_by_topic[record.topic]

    async def _process_with_retry(self, msg: ConsumedMessage) -> None:
        attempt = 0
        while True:
            try:
                await self._handler(msg)
                return
            except Exception:
                attempt += 1
                if attempt > self._cfg.max_retries:
                    logger.warning(
                        "message exhausted retries, routing to DLQ",
                        topic=msg.topic,
                        offset=msg.offset,
                        attempts=attempt,
                    )
                    raise
                logger.warning(
                    "message handler failed, retrying",
                    topic=msg.topic,
                    offset=msg.offset,
                    attempt=attempt,
                )
                await asyncio.sleep(retry_delay(self._cfg.retry_base_delay, attempt))

    async def _to_dlq(self, msg: ConsumedMessage, exc: Exception) -> None:
        dlq_topic = f"{msg.topic}{self._cfg.dlq_suffix}"
        await self._dlq_producer.send(
            dlq_topic,
            value=msg.value,
            key=msg.key,
            headers={**msg.headers, "x-dlq-error": str(exc)[:512]},
        )
        logger.error("message sent to DLQ", dlq_topic=dlq_topic, offset=msg.offset)


def _is_idle_status(summary: str) -> bool:
    return summary != "no_assignment" and all(part.endswith(":queued=0,running=0") for part in summary.split(" | "))
