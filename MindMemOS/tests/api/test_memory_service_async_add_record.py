from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from mindmemos.api.schemas import AddRequest, AuthContext
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.typing.memory import DialogueMessage
from mindmemos.typing.service import AddPipelineAsyncResult


def _auth() -> AuthContext:
    return AuthContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        memory_algorithm="vanilla",
    )


class _AsyncAddPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def add_async(self, inp, context, *, add_record_id=None, record_metadata=None):
        self.calls.append(
            {
                "inp": inp,
                "context": context,
                "add_record_id": add_record_id,
                "record_metadata": record_metadata,
            }
        )
        return AddPipelineAsyncResult(status="queued")


@dataclass
class _Recorder:
    add_inputs: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    async def record_add_input(
        self,
        inp,
        *,
        ctx,
        request_submitted_at,
        add_record_id,
        status,
        skill_bindings=None,
        score=None,
        task_id=None,
    ):
        self.add_inputs.append(
            {
                "inp": inp,
                "ctx": ctx,
                "request_submitted_at": request_submitted_at,
                "add_record_id": add_record_id,
                "status": status,
                "skill_bindings": skill_bindings,
                "score": score,
                "task_id": task_id,
            }
        )
        return add_record_id

    async def mark_add_failed(self, ctx, add_record_id, error):
        self.failures.append({"ctx": ctx, "add_record_id": add_record_id, "error": error})


@pytest.mark.asyncio
async def test_async_add_records_input_before_queue_ack() -> None:
    recorder = _Recorder()
    pipeline = _AsyncAddPipeline()
    service = MemoryService(
        add_pipeline=pipeline,
        get_pipeline=object(),
        delete_pipeline=object(),
        update_pipeline=object(),
        operation_recorder=recorder,
        skill_store=object(),
    )

    result = await service.add(
        _auth(),
        AddRequest(
            user_id="user-1",
            mode="async",
            messages=[DialogueMessage(role="user", content="remember async input", timestamp=1770000000000)],
            score=1.0,
            task_id="case-1",
        ),
    )

    assert result.status == "queued"
    assert recorder.failures == []
    assert len(recorder.add_inputs) == 1
    add_input = recorder.add_inputs[0]
    assert add_input["status"] == "queued"
    assert add_input["score"] == 1.0
    assert add_input["task_id"] == "case-1"
    assert pipeline.calls[0]["add_record_id"] == add_input["add_record_id"]
    assert pipeline.calls[0]["record_metadata"]["request_submitted_at"]
