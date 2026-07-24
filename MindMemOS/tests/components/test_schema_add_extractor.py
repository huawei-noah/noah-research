from datetime import UTC, datetime
from types import SimpleNamespace

import mindmemos.components.extractor._records as records
import pytest
from mindmemos.components.extractor.schema._schema_utils import (
    build_episode_entity,
    build_filtered_schema,
    dedupe_non_empty,
    entity_embedding_text,
    parse_json_object,
    schema_memory_type,
    strip_for_generation,
)
from mindmemos.components.extractor.schema.search_field import SchemaSearchFieldExtractor
from mindmemos.typing.memory import MemoryRequestContext


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="fallback-req",
        account_id="fallback-account",
        project_id="proj-1",
        api_key_uuid="fallback-key",
        user_id="fallback-user",
        session_id="fallback-session",
    )


def test_record_operators_restore_context_and_conversation_text() -> None:
    record = SimpleNamespace(
        add_record_id="record-1",
        payload={
            "request_id": "req-1",
            "account_id": "acc-1",
            "project_id": "proj-1",
            "api_key_uuid": "key-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "timestamp": 1770000000000,
            "messages": [{"role": "user", "content": "I like Qdrant."}],
            "metadata": {"buffer_message_index": 0},
            "force_generation": True,
        },
    )

    ctx = records.context([record], make_context())

    assert ctx.request_id == "req-1"
    assert ctx.user_id == "user-1"
    assert records.force_generation([record]) is True
    assert records.records_datetime([record]) == datetime(2026, 2, 2, 2, 40, tzinfo=UTC)
    assert records.metadata([record]) == {
        "add_record_ids": ["record-1"],
        "record_metadata": [{"buffer_message_index": 0}],
    }
    assert records.to_chunker_entries([record]) == [
        {
            "content": "I like Qdrant.",
            "speaker": "user",
            "timestamp": "2026-02-02 02:40:00",
            "add_record_id": "record-1",
        }
    ]
    assert records.to_conversation_text([record]) == "0. 2026-02-02 02:40:00 user: I like Qdrant."


def test_schema_add_conversation_text_preserves_named_speaker_identity() -> None:
    record = SimpleNamespace(
        add_record_id="record-rose",
        payload={
            "timestamp": 1770000000000,
            "messages": [{"role": "Rose", "content": "I moved to Boston."}],
        },
    )

    assert records.to_chunker_entries([record]) == [
        {
            "content": "I moved to Boston.",
            "speaker": "Rose",
            "timestamp": "2026-02-02 02:40:00",
            "add_record_id": "record-rose",
        }
    ]
    assert records.to_conversation_text([record]) == ("0. 2026-02-02 02:40:00 speaker=Rose: I moved to Boston.")


def test_schema_add_utils_filter_schema_and_build_episode_entity() -> None:
    schema = [
        {
            "entity_type": "user",
            "entity_description": "User profile",
            "dynamic_property": {
                "default_property": {"desc": "Fallback"},
                "preference": {"desc": "Preference"},
                "preference_summary": {"desc": "High order", "order": 2},
            },
        },
        {"entity_type": "episodes", "dynamic_property": {"input_messages": {"desc": "Episode"}}},
    ]

    generation_schema = strip_for_generation(schema)
    filtered_schema = build_filtered_schema(
        generation_schema,
        [{"entity_type": "user", "relevant_properties": ["preference"]}],
    )
    episode_entity = build_episode_entity(
        objectified_content="The user said they like Qdrant.",
        episode_description="Qdrant preference\nThe user likes Qdrant.",
        dialogue_date="2026-02-02",
        search_fields=["Qdrant preference"],
    )

    assert generation_schema == [
        {
            "entity_type": "user",
            "entity_description": "User profile",
            "dynamic_property": {
                "default_property": {"desc": "Fallback"},
                "preference": {"desc": "Preference"},
            },
        }
    ]
    assert filtered_schema[0]["dynamic_property"] == {
        "default_property": {"desc": "Fallback"},
        "preference": {"desc": "Preference"},
    }
    assert episode_entity["entity_type"] == "episodes"
    assert episode_entity["properties"][0]["property_name"] == "input_messages"
    assert "Qdrant" in entity_embedding_text(episode_entity)
    assert dedupe_non_empty([" a ", "", "a", "b"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_schema_search_field_extractor_uses_properties_then_dedupes() -> None:
    extractor = SchemaSearchFieldExtractor()

    fields = await extractor.extract_search_fields(
        entities=[
            {
                "entity_type": "user",
                "description": "fallback description",
                "properties": [
                    {"property_name": "preference", "value": "Likes Qdrant"},
                    {"property_name": "preference", "value": "Likes Qdrant"},
                    {"property_name": "input_messages", "value": "ignored"},
                ],
            },
            {"entity_type": "episodes", "description": "ignored episode"},
            {"entity_type": "project", "description": "Uses Neo4j"},
        ],
        context_text="conversation",
        max_fields=3,
    )

    assert fields == ["Likes Qdrant", "Uses Neo4j"]


@pytest.mark.asyncio
async def test_schema_search_field_extractor_can_augment_fields() -> None:
    class FakeLLM:
        async def chat(self, **kwargs):
            return SimpleNamespace(parsed=["Augmented Qdrant query"])

    prompt_set = SimpleNamespace(
        episode_search_field_augment="{episode_text}\n{existing_fields}\n{augment_count}",
    )
    extractor = SchemaSearchFieldExtractor(llm_client=FakeLLM(), prompt_set=prompt_set)

    fields = await extractor.extract_search_fields(
        entities=[{"entity_type": "project", "description": "Uses Qdrant"}],
        context_text="conversation",
        max_fields=3,
        augment=True,
        augment_count=1,
    )

    assert fields == ["Uses Qdrant", "Augmented Qdrant query"]


def test_parse_json_object_handles_fenced_json() -> None:
    assert parse_json_object('```json\n{"ok": true}\n```') == {"ok": True}


@pytest.mark.parametrize(
    ("entity_type", "property_name", "expected"),
    [
        ("episodes", "default_property", "episodic"),
        ("episode", "default_property", "episodic"),
        ("user", "preference", "profile"),
        ("person", "preference", "profile"),
        ("person", "task_experience", "experience"),
        ("task_experience", "default_property", "experience"),
        ("task", "task_experience", "experience"),
        ("organization", "service_info", "fact"),
        (None, None, "fact"),
    ],
)
def test_schema_memory_type_maps_schema_labels_to_display_types(entity_type, property_name, expected) -> None:
    assert schema_memory_type(entity_type, property_name) == expected


@pytest.mark.parametrize("content", ["not-json", "{bad"])
def test_parse_json_object_raises_for_invalid_json(content: str) -> None:
    with pytest.raises(ValueError):
        parse_json_object(content)
