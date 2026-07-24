"""Kafka worker for asynchronous memory feedback tasks."""

from __future__ import annotations

from ..config import get_config
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..pipelines.feedback import MEMORY_FEEDBACK_TOPIC
from ..typing import FeedbackPipelineInput, MemoryRequestContext

TOPIC = MEMORY_FEEDBACK_TOPIC
GROUP_ID = "memory-feedback-worker"

logger = get_logger(__name__)


async def handle_memory_feedback(msg: ConsumedMessage) -> None:
    """Consume a queued feedback task and execute feedback synchronously."""

    body = msg.json()
    context = MemoryRequestContext.model_validate(body["context"])
    payload = FeedbackPipelineInput.model_validate({**(body.get("input") or {}), "mode": "sync"})
    pipeline = create_pipeline(type="feedback", name=get_config().pipelines.feedback)
    if not hasattr(pipeline, "feedback_sync"):
        raise TypeError("configured feedback pipeline must expose feedback_sync for Kafka worker execution")

    logger.info(
        "processing async memory feedback",
        request_id=context.request_id,
        account_id=context.account_id,
        project_id=context.project_id,
        topic=msg.topic,
        offset=msg.offset,
    )
    await pipeline.feedback_sync(payload, context)
