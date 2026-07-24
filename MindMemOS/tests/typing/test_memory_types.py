from mindmemos.typing.memory import MemoryWrite


def test_tool_trace_is_a_valid_memory_type() -> None:
    fields = MemoryWrite.model_fields

    write = MemoryWrite(
        memory_id="mem-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
        content="Tool call finished successfully.",
        mem_type="tool_trace",
        mem_extract_version="test_v1",
        created_at="2026-05-28T00:00:00+00:00",
    )

    assert "mem_type" in fields
    assert write.mem_type == "tool_trace"
