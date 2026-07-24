from mindmemos.config import KafkaTopicConfig
from mindmemos.infra.kafka.admin import _partition_expansions


def test_partition_expansions_only_increase_existing_topics() -> None:
    expansions = _partition_expansions(
        [
            KafkaTopicConfig(name="memory.add", partitions=16),
            KafkaTopicConfig(name="memory.feedback", partitions=1),
        ],
        [
            {"topic": "memory.add", "partitions": [{"partition": 0}]},
            {"topic": "memory.feedback", "partitions": [{"partition": 0}, {"partition": 1}]},
        ],
    )

    assert set(expansions) == {"memory.add"}
    assert expansions["memory.add"].total_count == 16
