from __future__ import annotations

import pytest
from mindmemos.pipelines.memory_db.operation_records import MemoryOperationRecorder
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import AddPipelineSyncResult, MemoryAddEventItem


class FakeAddRecordStore:
    def __init__(self) -> None:
        self.patches: list[tuple[str, str, dict]] = []

    async def patch(self, project_id: str, add_record_id: str, payload: dict) -> None:
        self.patches.append((project_id, add_record_id, payload))


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_operation_recorder_patches_async_add_lifecycle_payloads() -> None:
    store = FakeAddRecordStore()
    recorder = MemoryOperationRecorder(add_record_store=store)
    ctx = make_context()

    await recorder.mark_add_processing(ctx, "add-rec-1")
    await recorder.mark_add_completed(
        ctx,
        "add-rec-1",
        AddPipelineSyncResult(
            status="ok",
            memories=[MemoryAddEventItem(operation="add", content="remembered", memory_id="mem-1")],
        ),
    )
    await recorder.mark_add_failed(ctx, "add-rec-2", "pipeline db error")

    processing = store.patches[0]
    completed = store.patches[1]
    failed = store.patches[2]
    assert processing[0:2] == ("proj-1", "add-rec-1")
    assert processing[2]["status"] == "processing"
    assert "processing_at" in processing[2]
    assert completed[0:2] == ("proj-1", "add-rec-1")
    assert completed[2]["status"] == "ok"
    assert completed[2]["memories"][0]["memory_id"] == "mem-1"
    assert "task_completed_at" in completed[2]
    assert failed[0:2] == ("proj-1", "add-rec-2")
    assert failed[2]["status"] == "error"
    assert failed[2]["error"] == "pipeline db error"
