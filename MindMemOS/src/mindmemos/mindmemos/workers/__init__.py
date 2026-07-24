"""Kafka worker registration entry point."""

from __future__ import annotations

from ..infra.kafka import register_handler
from ..logging import get_logger

logger = get_logger(__name__)


def register_workers() -> None:
    """Register all Kafka consumer handlers before Kafka startup."""
    from .memory_add import GROUP_ID as MEMORY_ADD_GROUP_ID
    from .memory_add import handle_memory_add
    from .memory_dreaming import GROUP_ID as MEMORY_DREAMING_GROUP_ID
    from .memory_dreaming import handle_memory_dreaming
    from .memory_feedback import GROUP_ID as MEMORY_FEEDBACK_GROUP_ID
    from .memory_feedback import handle_memory_feedback
    from .schema_add_drain import GROUP_ID as SCHEMA_ADD_DRAIN_GROUP_ID
    from .schema_add_drain import handle_schema_add_drain
    from .schema_add_episode import GROUP_ID as SCHEMA_ADD_EPISODE_GROUP_ID
    from .schema_add_episode import handle_schema_add_episode
    from .skill_evolve import GROUP_ID as SKILL_EVOLVE_GROUP_ID
    from .skill_evolve import handle_skill_evolve

    register_handler(MEMORY_ADD_GROUP_ID, handle_memory_add)
    register_handler(MEMORY_DREAMING_GROUP_ID, handle_memory_dreaming)
    register_handler(MEMORY_FEEDBACK_GROUP_ID, handle_memory_feedback)
    register_handler(SCHEMA_ADD_DRAIN_GROUP_ID, handle_schema_add_drain)
    register_handler(SCHEMA_ADD_EPISODE_GROUP_ID, handle_schema_add_episode)
    register_handler(SKILL_EVOLVE_GROUP_ID, handle_skill_evolve)
    logger.debug("business kafka workers registered")
