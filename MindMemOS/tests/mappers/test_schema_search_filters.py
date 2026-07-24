from datetime import UTC, datetime

import pytest
from mindmemos.typing.memory import FieldCondition, MemoryRequestContext, SearchFilter

from mindmemos.errors import InvalidFilterError
from mindmemos.mappers import parse_schema_search_filters


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-from-context",
        api_key_uuid="key-1",
        user_id="user-from-context",
        session_id="session-1",
    )


def _field_conditions(sf: SearchFilter | None) -> dict[str, FieldCondition]:
    assert sf is not None
    conditions: dict[str, FieldCondition] = {}

    def visit(node: SearchFilter) -> None:
        for clause in [*node.must, *node.should, *node.must_not]:
            if isinstance(clause, SearchFilter):
                visit(clause)
            else:
                conditions[clause.field] = clause

    visit(sf)
    return conditions


def test_schema_search_filters_parse_request_dsl_to_internal_filters() -> None:
    parsed = parse_schema_search_filters(
        {
            "project_id": "proj-from-context",
            "user_id": "user-from-request",
            "mem_type": {"in": ["fact", "profile"]},
            "created_at": {"gte": "2026-01-01T00:00:00+00:00"},
            "entity_type": "person",
        },
        make_context(),
    )

    assert parsed.context.project_id == "proj-from-context"
    assert parsed.context.user_id == "user-from-request"
    assert parsed.has_time_filter is True

    memory_conditions = _field_conditions(parsed.memory_filter)
    assert "project_id" not in memory_conditions
    assert memory_conditions["user_id"].value == "user-from-request"
    assert memory_conditions["mem_type"].values == ["fact", "profile"]
    assert memory_conditions["created_at"].gte == datetime(2026, 1, 1, tzinfo=UTC)
    assert memory_conditions["entity_type"].value == "person"

    entity_conditions = _field_conditions(parsed.entity_filter)
    assert "mem_type" not in entity_conditions
    assert "created_at" not in entity_conditions
    assert entity_conditions["user_id"].value == "user-from-request"
    assert entity_conditions["entity_type"].value == "person"


@pytest.mark.parametrize(
    "filters",
    [
        {"project_id": {"in": ["proj-a", "proj-b"]}},
        {"OR": [{"project_id": "proj-a"}, {"project_id": "proj-b"}]},
    ],
)
def test_schema_search_project_filter_must_be_single_exact_scope(filters: dict) -> None:
    with pytest.raises(InvalidFilterError):
        parse_schema_search_filters(filters, make_context())


def test_schema_search_project_filter_must_match_request_context() -> None:
    with pytest.raises(InvalidFilterError):
        parse_schema_search_filters({"project_id": "other-project"}, make_context())
