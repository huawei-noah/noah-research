"""Kafka worker for draining the schema add durable buffer."""

from __future__ import annotations

from ..api.algorithm import binding_for_memory_algorithm
from ..infra.kafka import ConsumedMessage
from ..logging import get_logger
from ..pipelines import create_pipeline
from ..pipelines.add import SCHEMA_ADD_DRAIN_TOPIC
from ..pipelines.memory_db import MemoryOperationRecorder, suppress_recording_errors
from ..typing import AddPipelineInput, MemoryRequestContext

TOPIC = SCHEMA_ADD_DRAIN_TOPIC
GROUP_ID = "schema-add-drain-worker"

logger = get_logger(__name__)


async def handle_schema_add_drain(msg: ConsumedMessage) -> None:
    """Consume one schema add drain task and process buffered add records."""

    body = msg.json()
    context = MemoryRequestContext.model_validate(body["context"])
    payload = AddPipelineInput.model_validate({**body["input"], "mode": "sync"}) if "input" in body else None
    trigger_record_id = body.get("trigger_record_id")
    pipeline_name = binding_for_memory_algorithm("schema").add_pipeline
    pipeline = create_pipeline(type="add", name=pipeline_name)
    drain_buffer = getattr(pipeline, "drain_buffer", None)
    if drain_buffer is None:
        raise RuntimeError(
            "schema add drain worker requires the schema add pipeline to support drain_buffer; "
            f"configured schema add pipeline is {pipeline_name!r}; "
            f"request_id={context.request_id}; project_id={context.project_id}"
        )

    recorder = MemoryOperationRecorder() if isinstance(trigger_record_id, str) and trigger_record_id else None
    if recorder is not None and payload is not None:
        await suppress_recording_errors(
            recorder.mark_add_processing(context, trigger_record_id),
            operation="add.schema_add.drain",
        )

    try:
        await drain_buffer(
            context,
            consistency=body.get("consistency"),
            force=bool(body.get("force", False)),
            trigger_record_id=trigger_record_id,
        )
    except Exception as exc:
        if recorder is not None:
            await suppress_recording_errors(
                recorder.mark_add_failed(context, trigger_record_id, str(exc)),
                operation="add.schema_add.drain",
            )
        raise
