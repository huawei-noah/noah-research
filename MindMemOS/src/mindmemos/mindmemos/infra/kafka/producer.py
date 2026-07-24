"""Asynchronous Kafka producer wrapper."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aiokafka import AIOKafkaProducer
from opentelemetry.trace import SpanKind, Status, StatusCode

from ...config import KafkaConfig
from ...logging import carrier_to_headers, get_logger, get_tracer, inject_trace_context
from .config_context import inject_config_context
from .dispatcher import DISPATCH_KEY_HEADER
from .message import serialize_value

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class KafkaProducer:
    """Shared asynchronous Kafka producer for the process.

    ``send()`` is fire-and-forget: it enqueues the message into the underlying
    aiokafka batch buffer and returns once the broker acknowledgement is no
    longer required for the caller to make progress. The actual broker round
    trip is driven by the aiokafka sender task and gated by ``linger_ms``.

    Call ``flush()`` (typically once per request handler or once at shutdown)
    to wait for all enqueued messages to be acknowledged by the broker. This
    keeps the previously synchronous semantics available at a single, explicit
    point instead of paying one round trip per ``send``.

    Callers that must report delivery success/failure back to a client should
    pass ``wait=True`` (or ``await`` the future returned by ``send``); the call
    then resolves only after the broker acknowledges *that* message and re-raises
    any delivery error, while other in-flight messages still batch in parallel.
    """

    def __init__(self, config: KafkaConfig):
        self._config = config
        self._producer: AIOKafkaProducer | None = None
        # Futures tracked per-message so flush() can await broker acks.
        self._pending: set[asyncio.Future[Any]] = set()
        # Bound the number of un-acked in-flight messages so a burst of
        # fire-and-forget producers cannot grow the buffer without limit.
        # 0/None disables the gate (previous unbounded behaviour).
        max_inflight = getattr(config, "producer_max_inflight", 0)
        self._inflight: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_inflight) if max_inflight and max_inflight > 0 else None
        )

    async def start(self) -> None:
        if self._producer is not None:
            return
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._config.bootstrap_servers,
            client_id=self._config.client_id,
            acks=self._config.acks,
            enable_idempotence=self._config.enable_idempotence,
            linger_ms=self._config.producer_linger_ms,
            max_batch_size=self._config.producer_max_batch_size,
            request_timeout_ms=self._config.request_timeout_ms,
        )
        await self._producer.start()
        logger.info("kafka producer started", servers=self._config.bootstrap_servers)

    async def stop(self) -> None:
        if self._producer is not None:
            # Drain anything still in flight before tearing the client down.
            await self.flush()
            await self._producer.stop()
            self._producer = None
            logger.info("kafka producer stopped")

    async def send(
            self,
            topic: str,
            value: Any,
            *,
            key: str | None = None,
            dispatch_key: str | None = None,
            headers: dict[str, str] | None = None,
            wait: bool = False,
    ) -> asyncio.Future[Any]:
        """Enqueue one Kafka message for asynchronous delivery.

        By default returns as soon as the message is buffered in the aiokafka
        producer (it does **not** wait for broker acknowledgement). Pair with
        ``flush()`` to await acknowledgement of the whole batch when needed.

        Args:
            wait: When ``True``, await this message's broker acknowledgement
                before returning and re-raise any delivery error. Use this on
                request paths that must report delivery success/failure to the
                caller. Concurrent ``wait=True`` sends still batch together, so
                throughput is bounded by broker delivery rate, not serialized.

        Returns:
            A future that resolves when the broker acknowledges this message
            (or carries the delivery exception). Callers may ignore it for
            fire-and-forget, or ``await`` it for confirmation.
        """
        if self._producer is None:
            raise RuntimeError("KafkaProducer not started; call start() first")

        headers = inject_config_context(headers)
        if dispatch_key is not None:
            headers = {**headers, DISPATCH_KEY_HEADER: dispatch_key}
            if key is None:
                key = dispatch_key

        with tracer.start_as_current_span(
                f"kafka.publish {topic}",
                kind=SpanKind.PRODUCER,
        ) as span:
            span.set_attribute("messaging.system", "kafka")
            span.set_attribute("messaging.destination.name", topic)
            if key is not None:
                span.set_attribute("messaging.kafka.message.key", key)

            # 1. 记录准备阶段开始
            prepare_start = time.perf_counter()

            # Inject while the publish span is active so downstream spans use it as parent.
            carrier = inject_trace_context()
            kafka_headers = carrier_to_headers(carrier)
            if headers:
                kafka_headers += [(k, v.encode("utf-8")) for k, v in headers.items()]

            payload = serialize_value(value)
            encoded_key = key.encode("utf-8") if key is not None else None

            # 1. 记录准备阶段结束
            prepare_duration = time.perf_counter() - prepare_start
            span.set_attribute("kafka.publish.prepare.duration_ms", prepare_duration * 1000)

            # 2. 记录信号量获取阶段开始
            semaphore_start = time.perf_counter()
            if self._inflight is not None:
                await self._inflight.acquire()
            semaphore_duration = time.perf_counter() - semaphore_start
            span.set_attribute("kafka.publish.semaphore.duration_ms", semaphore_duration * 1000)

            try:
                # 3. 记录发送阶段开始
                send_start = time.perf_counter()
                send_future = await self._producer.send(
                    topic,
                    value=payload,
                    key=encoded_key,
                    headers=kafka_headers,
                )
                send_duration = time.perf_counter() - send_start
                span.set_attribute("kafka.publish.send.duration_ms", send_duration * 1000)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                logger.error("kafka producer enqueue failed", topic=topic, error=str(exc))
                if self._inflight is not None:
                    self._inflight.release()
                raise

            self._pending.add(send_future)

            def _cleanup_done(done: asyncio.Future[Any]) -> None:
                try:
                    exc = done.exception()
                except asyncio.CancelledError as exc:
                    logger.error("kafka producer send cancelled", topic=topic, error=str(exc))
                else:
                    if exc is not None:
                        logger.error("kafka producer send failed", topic=topic, error=str(exc))
                finally:
                    self._pending.discard(done)
                    if self._inflight is not None:
                        self._inflight.release()

            send_future.add_done_callback(_cleanup_done)

            # 4. 记录回调设置完成时间（用于总耗时统计）
            span.set_attribute("kafka.publish.enqueue.complete", True)

        # Outside the publish span: optionally block on this message's broker
        # ack so the caller can report delivery success/failure. Awaiting the future
        # re-raises the delivery exception when the send ultimately failed.
        if wait:
            # 5. 记录等待确认阶段
            wait_start = time.perf_counter()
            await send_future
            wait_duration = time.perf_counter() - wait_start
            # 注意：这里在span外部，需要用当前的trace上下文或者创建一个新的span
            # 但为了不影响主span，我们可以用logger记录，或者创建一个子span
            logger.info(
                f"kafka message wait duration: {wait_duration * 1000:.2f}ms",
                topic=topic,
                wait_duration_ms=wait_duration * 1000
            )
        return send_future

    async def flush(self) -> None:
        """Wait for all enqueued messages to be acknowledged by the broker.

        Safe to call when nothing is pending: returns immediately. Callers
        that need the old ``send_and_wait`` semantics should pair ``send()``
        with a single ``flush()`` at the end of their batch.
        """
        pending = list(self._pending)
        if not pending:
            return
        # Gather ignores exceptions; the original _deliver task already
        # surfaced failures to the log / future, so we don't need to re-raise
        # them here (matches the previous per-attempt warning behaviour).
        await asyncio.gather(*pending, return_exceptions=True)
