"""Kafka worker for processing individual episode memory generation tasks."""

from __future__ import annotations

from ..api.algorithm import binding_for_memory_algorithm
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..pipelines.add import SCHEMA_ADD_EPISODE_TOPIC
from ..typing import MemoryRequestContext

TOPIC = SCHEMA_ADD_EPISODE_TOPIC
GROUP_ID = "schema-add-episode-worker"

logger = get_logger(__name__)


async def handle_schema_add_episode(msg: ConsumedMessage) -> None:
    """Consume one episode generation task and execute memory generation."""

    body = msg.json()
    context = MemoryRequestContext.model_validate(body["context"])
    add_record_ids: list[str] = body["add_record_ids"]
    episode_id: str = body["episode_id"]
    consistency: str | None = body.get("consistency")
    trigger_record_id: str | None = body.get("trigger_record_id")

    pipeline_name = binding_for_memory_algorithm("schema").add_pipeline
    pipeline = create_pipeline(type="add", name=pipeline_name)
    generate_episode = getattr(pipeline, "generate_episode", None)
    if generate_episode is None:
        raise RuntimeError(
            "schema add episode worker requires the schema add pipeline to support generate_episode; "
            f"configured schema add pipeline is {pipeline_name!r}; "
            f"request_id={context.request_id}; project_id={context.project_id}"
        )

    logger.info(
        "processing episode generation task",
        request_id=context.request_id,
        project_id=context.project_id,
        episode_id=episode_id,
        num_records=len(add_record_ids),
    )

    await generate_episode(
        context,
        add_record_ids,
        episode_id=episode_id,
        consistency=consistency,
        trigger_record_id=trigger_record_id,
    )
