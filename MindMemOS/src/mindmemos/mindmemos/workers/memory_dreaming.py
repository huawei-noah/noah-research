"""Kafka worker for offline memory consolidation tasks."""

from __future__ import annotations

from typing import Protocol

from ..config import get_config
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..typing import DreamingPipelineInput, MemoryRequestContext

TOPIC = "memory.dreaming"
GROUP_ID = "memory-dreaming-worker"

logger = get_logger(__name__)


class DreamingWorkerPipeline(Protocol):
    async def dream_sync(self, inp: DreamingPipelineInput, context: MemoryRequestContext): ...


async def handle_memory_dreaming(msg: ConsumedMessage) -> None:
    """Consume a queued dreaming task and execute consolidation synchronously."""

    body = msg.json()
    context = MemoryRequestContext.model_validate(body["context"])
    payload = DreamingPipelineInput.model_validate(body.get("input") or {})
    pipeline = create_pipeline(type="dreaming", name=get_config().pipelines.dreaming)
    if not hasattr(pipeline, "dream_sync"):
        raise TypeError("configured dreaming pipeline must expose dream_sync for Kafka worker execution")

    logger.info(
        "processing async memory dreaming",
        request_id=context.request_id,
        account_id=context.account_id,
        project_id=context.project_id,
        topic=msg.topic,
        offset=msg.offset,
    )
    await pipeline.dream_sync(payload, context)
