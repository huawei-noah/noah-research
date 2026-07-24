"""Kafka topic administration helpers."""

from __future__ import annotations

from aiokafka.admin import AIOKafkaAdminClient, NewPartitions, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, for_code

from ...config import KafkaConfig, KafkaTopicConfig
from ...logging import get_logger

logger = get_logger(__name__)


async def ensure_topics(config: KafkaConfig) -> None:
    """Create or expand configured Kafka topics before clients start."""

    if not config.topics:
        return

    admin = AIOKafkaAdminClient(
        bootstrap_servers=config.bootstrap_servers,
        client_id=f"{config.client_id}-admin",
        request_timeout_ms=config.request_timeout_ms,
    )
    await admin.start()
    try:
        existing = set(await admin.list_topics())
        expandable = set(existing)
        for topic in config.topics:
            if topic.name not in existing:
                if not await _create_topic(admin, topic, config.request_timeout_ms):
                    expandable.add(topic.name)
                existing.add(topic.name)

        if expandable:
            descriptions = await admin.describe_topics([topic.name for topic in config.topics if topic.name in expandable])
            expansions = _partition_expansions(config.topics, descriptions)
            if expansions:
                await admin.create_partitions(expansions, timeout_ms=config.request_timeout_ms)
                logger.info("kafka topics expanded", topics={name: spec.total_count for name, spec in expansions.items()})
    finally:
        await admin.close()


async def _create_topic(admin: AIOKafkaAdminClient, topic: KafkaTopicConfig, timeout_ms: int) -> bool:
    response = await admin.create_topics(
        [NewTopic(topic.name, topic.partitions, topic.replication_factor)],
        timeout_ms=timeout_ms,
    )
    for name, code, message in response.topic_errors:
        if not code:
            logger.info("kafka topic created", topic=name, partitions=topic.partitions)
            return True
        err_cls = for_code(code)
        if err_cls is TopicAlreadyExistsError:
            return False
        raise err_cls(f"Could not create topic {name}: {message}")
    return True


def _partition_expansions(
    configured: list[KafkaTopicConfig],
    descriptions: list[dict],
) -> dict[str, NewPartitions]:
    counts = {item["topic"]: len(item.get("partitions") or []) for item in descriptions}
    return {
        topic.name: NewPartitions(topic.partitions)
        for topic in configured
        if counts.get(topic.name, topic.partitions) < topic.partitions
    }
