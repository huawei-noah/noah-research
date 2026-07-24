"""Kafka worker for asynchronous memory add tasks."""

from __future__ import annotations

from ..api.algorithm import binding_for_memory_algorithm
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..pipelines.add import AddPipeline
from ..pipelines.memory_db import MemoryOperationRecorder, suppress_recording_errors
from ..typing import AddPipelineInput, MemoryRequestContext

TOPIC = "memory.add"
GROUP_ID = "memory-add-worker"

logger = get_logger(__name__)


async def handle_memory_add(msg: ConsumedMessage) -> None:
    """Consume a queued add task and execute the configured add pipeline synchronously."""

    body = msg.json()
    context = MemoryRequestContext.model_validate(body["context"])
    payload = AddPipelineInput.model_validate({**body["input"], "mode": "sync"})
    add_record_id = body.get("add_record_id")
    if not context.memory_algorithm:
        raise RuntimeError("memory.add task missing memory_algorithm in context")
    pipeline_name = binding_for_memory_algorithm(context.memory_algorithm).add_pipeline
    pipeline: AddPipeline = create_pipeline(type="add", name=pipeline_name)
    recorder = MemoryOperationRecorder() if isinstance(add_record_id, str) and add_record_id else None

    logger.info(
        "processing async memory add",
        request_id=context.request_id,
        account_id=context.account_id,
        project_id=context.project_id,
        memory_algorithm=context.memory_algorithm,
        topic=msg.topic,
        offset=msg.offset,
    )
    try:
        if recorder is not None:
            await suppress_recording_errors(
                recorder.mark_add_processing(context, add_record_id),
                operation="add",
            )
        result = await pipeline.add_sync(payload, context, add_record_id=add_record_id)
        logger.info(
            "async memory add completed",
            request_id=context.request_id,
            memory_count=len(result.memories),
        )
    except Exception as exc:
        if recorder is not None:
            await suppress_recording_errors(
                recorder.mark_add_failed(context, add_record_id, str(exc)),
                operation="add",
            )
        logger.exception(
            "async memory add failed",
            request_id=context.request_id,
            account_id=context.account_id,
        )
        raise
