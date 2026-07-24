"""Kafka worker for asynchronous skill evolution tasks."""

from __future__ import annotations

from ..config import get_config
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..pipelines.skill import SKILL_EVOLVE_TOPIC, SkillEvolvePipeline

TOPIC = SKILL_EVOLVE_TOPIC
GROUP_ID = "skill-evolve-worker"

logger = get_logger(__name__)


async def handle_skill_evolve(msg: ConsumedMessage) -> None:
    """Consume a queued skill evolve task and execute the configured pipeline."""

    body = msg.json()
    project_id = body["project_id"]
    cloud_skill_id = body["cloud_skill_id"]
    pipeline: SkillEvolvePipeline = create_pipeline(
        type="skill_evolve",
        name=get_config().pipelines["skill_evolve"],
    )

    logger.info(
        "processing async skill evolve",
        request_id=body.get("request_id"),
        account_id=body.get("account_id"),
        project_id=project_id,
        cloud_skill_id=cloud_skill_id,
        topic=msg.topic,
        offset=msg.offset,
    )
    result = await pipeline.evolve(project_id=project_id, cloud_skill_id=cloud_skill_id)
    logger.info(
        "async skill evolve completed",
        request_id=body.get("request_id"),
        project_id=project_id,
        cloud_skill_id=cloud_skill_id,
        evolved=result.evolved,
        new_version_id=result.new_version_id,
    )
