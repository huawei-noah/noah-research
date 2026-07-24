"""Process-wide Kafka resource registry."""

from __future__ import annotations

import asyncio

from ...config import get_config
from ...logging import get_logger
from .admin import ensure_topics
from .consumer import KafkaConsumer, MessageHandler
from .producer import KafkaProducer

logger = get_logger(__name__)

_producer: KafkaProducer | None = None
_consumers: dict[str, KafkaConsumer] = {}
_handlers: dict[str, MessageHandler] = {}
_task_semaphore: asyncio.Semaphore | None = None


def get_producer() -> KafkaProducer:
    """Return the process-wide Kafka producer singleton."""
    global _producer
    if _producer is None:
        _producer = KafkaProducer(get_config().kafka)
    return _producer


def register_handler(group_id: str, handler: MessageHandler) -> None:
    """Register a message handler for a Kafka consumer group."""
    _handlers[group_id] = handler


async def start_kafka() -> None:
    """Start the Kafka producer and configured consumers."""
    global _task_semaphore
    cfg = get_config().kafka
    if not cfg.enabled:
        logger.info("kafka disabled, skip start")
        return

    await ensure_topics(cfg)
    await get_producer().start()
    _task_semaphore = (
        asyncio.Semaphore(cfg.global_max_concurrency) if cfg.global_max_concurrency > 0 else None
    )

    for consumer_cfg in cfg.consumers:
        handler = _handlers.get(consumer_cfg.group_id)
        if handler is None:
            logger.warning("no handler registered for consumer group", group_id=consumer_cfg.group_id)
            continue
        consumer = KafkaConsumer(cfg, consumer_cfg, handler, get_producer(), global_task_semaphore=_task_semaphore)
        await consumer.start()
        _consumers[consumer_cfg.group_id] = consumer


async def stop_kafka() -> None:
    """Stop Kafka consumers and close the producer."""
    global _task_semaphore
    for consumer in _consumers.values():
        await consumer.stop()
    _consumers.clear()
    _task_semaphore = None

    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None


def reset() -> None:
    """Clear in-process Kafka registry state for tests."""
    global _producer, _task_semaphore
    _producer = None
    _task_semaphore = None
    _consumers.clear()
    _handlers.clear()
