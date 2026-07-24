from mindmemos.infra.kafka import registry as kafka_registry
from mindmemos.workers import register_workers
from mindmemos.workers.memory_add import GROUP_ID as MEMORY_ADD_GROUP_ID
from mindmemos.workers.memory_dreaming import GROUP_ID as MEMORY_DREAMING_GROUP_ID
from mindmemos.workers.memory_feedback import GROUP_ID as MEMORY_FEEDBACK_GROUP_ID
from mindmemos.workers.schema_add_drain import GROUP_ID as SCHEMA_ADD_DRAIN_GROUP_ID
from mindmemos.workers.schema_add_episode import GROUP_ID as SCHEMA_ADD_EPISODE_GROUP_ID
from mindmemos.workers.skill_evolve import GROUP_ID as SKILL_EVOLVE_GROUP_ID


def test_register_workers_registers_business_and_schema_workers() -> None:
    try:
        kafka_registry.reset()
        register_workers()

        assert MEMORY_ADD_GROUP_ID in kafka_registry._handlers
        assert MEMORY_DREAMING_GROUP_ID in kafka_registry._handlers
        assert SCHEMA_ADD_DRAIN_GROUP_ID in kafka_registry._handlers
        assert MEMORY_FEEDBACK_GROUP_ID in kafka_registry._handlers
        assert SCHEMA_ADD_EPISODE_GROUP_ID in kafka_registry._handlers
        assert SKILL_EVOLVE_GROUP_ID in kafka_registry._handlers
    finally:
        kafka_registry.reset()
