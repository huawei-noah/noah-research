"""Guard tests for the public search-filter DSL.

Two invariants are protected here:

1. **Accuracy** — a representative DSL parses and translates end-to-end into the
   exact Qdrant primitives we expect, with ``project_id`` force-injected.
2. **Whitelist** — every allowlisted field is accepted, every removed/identity
   field is rejected, and the allowlist itself does not silently drift.
"""

from datetime import UTC, datetime

import pytest
from mindmemos.infra.db.filters import FILTERABLE_MEMORY_FIELDS
from mindmemos.typing.memory import (
    DSL_DATETIME_FIELDS,
    DSL_FILTERABLE_MEMORY_FIELDS,
    DSL_TEXT_FIELDS,
    MemoryRequestContext,
    SearchFilter,
)
from qdrant_client import models as qmodels

from mindmemos.errors import InvalidFilterError
from mindmemos.mappers import parse_search_dsl, search_filter_to_qdrant

# Fields the user explicitly removed from the public surface. They must never be
# accepted by the DSL even though some are indexed in the payload schema.
_FORBIDDEN_FIELDS = frozenset(
    {
        "project_id",
        "api_key_uuid",
        "request_id",
        "mem_extract_version",
        "status",
        "update_at",
        "status_changed_at",
        "reinforcement_count",
        "parent_ids",
        "root_id",
    }
)


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        agent_id="agent-1",
        session_id="session-1",
    )


# Guard 1: filter accuracy (DSL -> internal SearchFilter -> Qdrant)


def test_dsl_translates_every_operator_to_exact_qdrant_primitives() -> None:
    dsl = {
        "user_id": "user-1",
        "mem_type": {"in": ["fact", "profile"]},
        "entity_type": {"nin": ["temp"]},
        "created_at": {"gte": "2026-01-01T00:00:00+00:00", "lt": "2026-02-01T00:00:00+00:00"},
        "content": {"icontains": "qdrant"},
        "entity_id": "*",
    }

    qfilter = search_filter_to_qdrant(_ctx(), parse_search_dsl(dsl))

    assert qfilter.should is None
    assert qfilter.must_not is None
    assert qfilter.must is not None
    assert len(qfilter.must) == 7

    # project_id is always force-injected as the first must clause.
    project_clause = qfilter.must[0]
    assert project_clause.key == "project_id"
    assert isinstance(project_clause.match, qmodels.MatchValue)
    assert project_clause.match.value == "proj-1"

    assert qfilter.must[1].key == "user_id"
    assert isinstance(qfilter.must[1].match, qmodels.MatchValue)
    assert qfilter.must[1].match.value == "user-1"

    # in -> MatchAny
    assert qfilter.must[2].key == "mem_type"
    assert isinstance(qfilter.must[2].match, qmodels.MatchAny)
    assert qfilter.must[2].match.any == ["fact", "profile"]

    # nin -> MatchExcept
    assert qfilter.must[3].key == "entity_type"
    assert isinstance(qfilter.must[3].match, qmodels.MatchExcept)

    assert qfilter.must[4].key == "created_at"
    assert isinstance(qfilter.must[4].range, qmodels.DatetimeRange)
    assert qfilter.must[4].range.gte == datetime(2026, 1, 1, tzinfo=UTC)
    assert qfilter.must[4].range.lt == datetime(2026, 2, 1, tzinfo=UTC)

    # icontains -> MatchText
    assert qfilter.must[5].key == "content"
    assert isinstance(qfilter.must[5].match, qmodels.MatchText)
    assert qfilter.must[5].match.text == "qdrant"

    # wildcard -> nested filter with NOT is_empty
    wildcard = qfilter.must[6]
    assert isinstance(wildcard, qmodels.Filter)
    assert wildcard.must_not is not None
    assert isinstance(wildcard.must_not[0], qmodels.IsEmptyCondition)
    assert wildcard.must_not[0].is_empty.key == "entity_id"


def test_dsl_logical_combinators_map_to_qdrant_bool_clauses() -> None:
    dsl = {
        "AND": [{"session_id": "s-1"}],
        "OR": [{"agent_id": "a-1"}, {"app_id": "x-1"}],
        "NOT": [{"mem_extract_type": "summary"}],
    }

    qfilter = search_filter_to_qdrant(_ctx(), parse_search_dsl(dsl))

    # must = injected project_id + the AND clause
    assert qfilter.must is not None
    assert qfilter.must[0].key == "project_id"
    assert qfilter.must[1].must[0].key == "session_id"

    # Logical OR clauses must map to Qdrant should conditions.
    assert qfilter.should is not None
    assert {clause.must[0].key for clause in qfilter.should} == {"agent_id", "app_id"}

    assert qfilter.must_not is not None
    assert qfilter.must_not[0].must[0].key == "mem_extract_type"


def test_eq_and_ne_translate_to_match_and_negated_match() -> None:
    qfilter = search_filter_to_qdrant(
        _ctx(),
        parse_search_dsl({"mem_type": {"eq": "fact"}, "entity_type": {"ne": "temp"}}),
    )

    assert qfilter.must[1].key == "mem_type"
    assert qfilter.must[1].match.value == "fact"

    # ne -> nested filter whose must_not carries the match
    ne_clause = qfilter.must[2]
    assert isinstance(ne_clause, qmodels.Filter)
    assert ne_clause.must_not[0].key == "entity_type"
    assert ne_clause.must_not[0].match.value == "temp"


# Guard 2: whitelist compliance


@pytest.mark.parametrize("field", sorted(DSL_FILTERABLE_MEMORY_FIELDS))
def test_every_allowlisted_field_is_accepted(field: str) -> None:
    # The wildcard form is valid for every field category, so it cleanly proves
    # the field passes the allowlist regardless of its type.
    result = parse_search_dsl({field: "*"})
    clause = result.must[0]
    assert isinstance(clause, SearchFilter)
    assert clause.must_not[0].field == field
    assert clause.must_not[0].op == "is_empty"


@pytest.mark.parametrize("field", sorted(_FORBIDDEN_FIELDS))
def test_forbidden_field_is_rejected_at_top_level(field: str) -> None:
    with pytest.raises(InvalidFilterError):
        parse_search_dsl({field: "value"})


@pytest.mark.parametrize("field", sorted(_FORBIDDEN_FIELDS))
def test_forbidden_field_is_rejected_inside_logical_clause(field: str) -> None:
    with pytest.raises(InvalidFilterError):
        parse_search_dsl({"AND": [{field: "value"}]})


def test_allowlist_does_not_silently_drift() -> None:
    assert DSL_FILTERABLE_MEMORY_FIELDS == frozenset(
        {
            "memory_id",
            "account_id",
            "user_id",
            "app_id",
            "session_id",
            "agent_id",
            "mem_type",
            "mem_extract_type",
            "validate_from",
            "validate_to",
            "created_at",
            "property_name",
            "entity_id",
            "entity_type",
            "content",
        }
    )


def test_public_allowlist_is_subset_of_indexed_fields() -> None:
    # Every public field must be an indexed payload field, otherwise the db
    # layer would translate it into a slow unindexed scan.
    assert DSL_FILTERABLE_MEMORY_FIELDS <= FILTERABLE_MEMORY_FIELDS
    assert DSL_DATETIME_FIELDS <= DSL_FILTERABLE_MEMORY_FIELDS
    assert DSL_TEXT_FIELDS <= DSL_FILTERABLE_MEMORY_FIELDS


def test_removed_identity_fields_are_excluded_from_allowlist() -> None:
    assert DSL_FILTERABLE_MEMORY_FIELDS.isdisjoint(_FORBIDDEN_FIELDS)


def test_field_categories_partition_the_allowlist() -> None:
    keyword = DSL_FILTERABLE_MEMORY_FIELDS - DSL_DATETIME_FIELDS - DSL_TEXT_FIELDS
    assert DSL_DATETIME_FIELDS.isdisjoint(DSL_TEXT_FIELDS)
    assert keyword | DSL_DATETIME_FIELDS | DSL_TEXT_FIELDS == DSL_FILTERABLE_MEMORY_FIELDS


# Guard 3: per-field type validation (operator / value type must fit the field)


@pytest.mark.parametrize(
    "dsl",
    [
        pytest.param({"mem_type": {"gt": 1}}, id="range-on-keyword"),
        pytest.param({"entity_id": {"lte": "2026-01-01T00:00:00+00:00"}}, id="range-on-keyword-str"),
        pytest.param({"mem_type": {"contains": "fa"}}, id="contains-on-keyword"),
        pytest.param({"created_at": {"contains": "2026"}}, id="contains-on-datetime"),
        pytest.param({"created_at": {"eq": "2026-01-01T00:00:00+00:00"}}, id="eq-on-datetime"),
        pytest.param({"created_at": {"in": ["2026-01-01T00:00:00+00:00"]}}, id="in-on-datetime"),
        pytest.param({"content": {"eq": "hello"}}, id="eq-on-text"),
        pytest.param({"content": {"gt": 1}}, id="range-on-text"),
    ],
)
def test_operator_incompatible_with_field_type_is_rejected(dsl: dict) -> None:
    with pytest.raises(InvalidFilterError):
        parse_search_dsl(dsl)


@pytest.mark.parametrize(
    "dsl",
    [
        pytest.param({"user_id": 123}, id="int-for-keyword-match"),
        pytest.param({"user_id": True}, id="bool-for-keyword-match"),
        pytest.param({"mem_type": {"eq": 5}}, id="int-for-eq"),
        pytest.param({"mem_type": {"in": [1, 2]}}, id="int-list-for-in"),
        pytest.param({"mem_type": {"in": "fact"}}, id="scalar-for-in"),
        pytest.param({"created_at": {"gte": "not-a-date"}}, id="bad-datetime-string"),
        pytest.param({"created_at": {"gte": 1700000000}}, id="number-for-datetime"),
        pytest.param({"content": {"contains": 123}}, id="int-for-contains"),
        pytest.param({"created_at": "2026-01-01"}, id="bare-scalar-on-datetime"),
        pytest.param({"content": "hello"}, id="bare-scalar-on-text"),
    ],
)
def test_value_type_mismatch_is_rejected(dsl: dict) -> None:
    with pytest.raises(InvalidFilterError):
        parse_search_dsl(dsl)


def test_type_correct_clauses_still_parse() -> None:
    # Sanity counterpart: the type-correct version of each rejected shape works.
    assert parse_search_dsl({"user_id": "u-1"}).must[0].op == "match"
    assert parse_search_dsl({"mem_type": {"in": ["fact", "profile"]}}).must[0].op == "any"
    assert parse_search_dsl({"created_at": {"gte": "2026-01-01T00:00:00+00:00"}}).must[0].op == "datetime"
    assert parse_search_dsl({"content": {"icontains": "hello"}}).must[0].op == "text"
