"""Tests for deterministic ID generation (components/id.py)."""

from mindmemos.components.id import generate_memory_id
from mindmemos.typing.memory import MemoryRequestContext


def _make_context(**overrides) -> MemoryRequestContext:
    defaults = dict(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
    )
    return MemoryRequestContext(**{**defaults, **overrides})


class TestGenerateMemoryId:
    """Deterministic memory ID via uuid5(project_id, namespace, content_hash)."""

    def test_same_input_produces_same_id(self) -> None:
        a = generate_memory_id("proj-1", "req-1", "hash-abc")
        b = generate_memory_id("proj-1", "req-1", "hash-abc")
        assert a == b

    def test_different_content_hash_produces_different_id(self) -> None:
        a = generate_memory_id("proj-1", "req-1", "hash-abc")
        b = generate_memory_id("proj-1", "req-1", "hash-xyz")
        assert a != b

    def test_different_namespace_produces_different_id(self) -> None:
        a = generate_memory_id("proj-1", "idem-1", "hash-abc")
        b = generate_memory_id("proj-1", "idem-2", "hash-abc")
        assert a != b

    def test_different_project_produces_different_id(self) -> None:
        a = generate_memory_id("proj-1", "req-1", "hash-abc")
        b = generate_memory_id("proj-2", "req-1", "hash-abc")
        assert a != b

    def test_output_is_valid_uuid_format(self) -> None:
        import uuid

        result = generate_memory_id("proj-1", "req-1", "hash-abc")
        parsed = uuid.UUID(result)
        assert parsed.version == 5
