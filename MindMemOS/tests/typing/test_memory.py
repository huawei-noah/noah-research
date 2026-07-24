import inspect
from datetime import UTC, datetime

import mindmemos.typing.memory as memory
import pytest
from pydantic import BaseModel, ValidationError

from mindmemos.typing import (
    AddPipelineInput,
    DialogueMessage,
    MemoryDbSearchHit,
    MemoryDbWritePlan,
    MemoryDbWriteSummary,
    MemoryRequestContext,
    MemoryWrite,
)


def test_minimal_memory_dtos_are_exported_and_serializable() -> None:
    ctx = MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="demo_project",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )
    message = DialogueMessage(role="user", content="User likes FastAPI")
    add_input = AddPipelineInput(messages=[message], timestamp=1700000000000)
    write = MemoryWrite(
        memory_id="mem-1",
        account_id=ctx.account_id,
        project_id=ctx.project_id,
        api_key_uuid=ctx.api_key_uuid,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        content=message.content,
        mem_extract_version="test_v1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        root_id=["mem-1"],
    )
    plan = MemoryDbWritePlan(memories=[write])
    result = MemoryDbWriteSummary(status="ok", memory_ids=["mem-1"])

    dumped = result.model_dump()

    assert add_input.messages[0].content == "User likes FastAPI"
    assert add_input.timestamp == 1700000000000
    assert plan.memories[0].project_id == "demo_project"
    assert dumped["memory_ids"] == ["mem-1"]


def test_search_hit_allows_lightweight_memory_id_only() -> None:
    hit = MemoryDbSearchHit(memory_id="mem-1", score=0.9, source="semantic")

    assert hit.memory is None
    assert hit.memory_id == "mem-1"


def test_memory_class_docstrings_describe_role_and_usage() -> None:
    missing = []

    for name in (name for name in dir(memory) if not name.startswith("_")):
        obj = getattr(memory, name)
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj.__module__ == memory.__name__:
            doc = inspect.getdoc(obj) or ""
            if "Purpose:" not in doc or "Used in:" not in doc:
                missing.append(name)

    assert missing == []


def test_public_names_do_not_use_dto_suffix() -> None:
    assert not [name for name in dir(memory) if not name.startswith("_") and name.endswith("DTO")]


def test_add_pipeline_input_mode_accepts_only_sync_or_async() -> None:
    assert AddPipelineInput(messages=[]).mode == "sync"
    assert AddPipelineInput(messages=[], mode="async").mode == "async"

    with pytest.raises(ValidationError):
        AddPipelineInput(messages=[], mode="hybrid")


def test_add_input_defaults_request_level_event_timestamp_to_current_utc_millis() -> None:
    before = int(datetime.now(UTC).timestamp() * 1000)

    add_input = AddPipelineInput(
        messages=[
            {"role": "user", "content": "remember this"},
            {"text": "plain text"},
            {"url": "https://example.com"},
            {"file_name": "note.txt", "file_path": "/tmp/note.txt"},
        ]
    )
    after = int(datetime.now(UTC).timestamp() * 1000)

    assert before <= add_input.timestamp <= after
    assert add_input.timestamp_utc == datetime.fromtimestamp(add_input.timestamp / 1000, tz=UTC)
    assert add_input.event_timestamp == add_input.timestamp
    assert add_input.event_timestamp_utc == add_input.timestamp_utc


def test_add_input_request_level_event_timestamp_overrides_message_timestamp() -> None:
    add_input = AddPipelineInput(
        timestamp=1700000000000,
        messages=[{"role": "user", "content": "remember this", "timestamp": 1770000000000}],
    )

    assert add_input.event_timestamp == 1700000000000
    assert add_input.event_timestamp_utc == datetime.fromtimestamp(1700000000000 / 1000, tz=UTC)


def test_add_input_message_timestamp_overrides_default_request_level_timestamp() -> None:
    add_input = AddPipelineInput(
        messages=[{"role": "user", "content": "remember this", "timestamp": 1770000000000}],
    )

    assert add_input.event_timestamp == 1770000000000
    assert add_input.event_timestamp_utc == datetime.fromtimestamp(1770000000000 / 1000, tz=UTC)


def test_add_input_uses_last_valid_message_timestamp() -> None:
    add_input = AddPipelineInput(
        messages=[
            {"role": "user", "content": "older", "timestamp": 1700000000000},
            {"role": "assistant", "content": "missing"},
            {"role": "user", "content": "newer", "timestamp": 1770000000000},
        ],
    )

    assert add_input.event_timestamp == 1770000000000
