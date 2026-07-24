"""Helpers for adapting public search filters to schema-search internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import InvalidFilterError
from ..infra.db import FILTERABLE_ENTITY_FIELDS, FILTERABLE_MEMORY_FIELDS
from ..typing import (
    DSL_DATETIME_FIELDS,
    DSL_FILTERABLE_MEMORY_FIELDS,
    FieldCondition,
    MemoryRequestContext,
    SearchFilter,
    search_filter_is_empty,
)
from .api import parse_search_dsl

_SCHEMA_EXTRA_DSL_FIELDS = frozenset({"project_id"})
_CONTEXT_OVERRIDE_FIELDS = frozenset({"account_id", "user_id", "app_id", "session_id", "agent_id"})
_SCHEMA_SEARCH_ENTITY_FILTER_FIELDS = (
    (FILTERABLE_ENTITY_FIELDS & DSL_FILTERABLE_MEMORY_FIELDS)
    - DSL_DATETIME_FIELDS
    - frozenset({"memory_id", "mem_type", "mem_extract_type", "property_name", "content"})
)


@dataclass(slots=True)
class ParsedRequestSearchFilters:
    """Parsed request filter bundle used by schema search."""

    context: MemoryRequestContext
    memory_filter: SearchFilter | None
    entity_filter: SearchFilter | None
    has_time_filter: bool = False


def parse_schema_search_filters(
    filters: dict[str, Any] | None,
    context: MemoryRequestContext,
) -> ParsedRequestSearchFilters:
    """Parse request DSL and prepare memory/entity filters for schema search."""

    search_filter = parse_search_dsl(filters, extra_filterable_fields=_SCHEMA_EXTRA_DSL_FIELDS)
    if search_filter_is_empty(search_filter):
        return ParsedRequestSearchFilters(context=context, memory_filter=None, entity_filter=None)

    scoped_context = _context_with_request_scope(context, search_filter)
    memory_filter = filter_search_filter_fields(search_filter, FILTERABLE_MEMORY_FIELDS)
    entity_filter = filter_search_filter_fields(search_filter, _SCHEMA_SEARCH_ENTITY_FILTER_FIELDS)
    return ParsedRequestSearchFilters(
        context=scoped_context,
        memory_filter=memory_filter,
        entity_filter=entity_filter,
        has_time_filter=search_filter_contains_fields(search_filter, DSL_DATETIME_FIELDS),
    )


def filter_search_filter_fields(sf: SearchFilter | None, allowed_fields: frozenset[str]) -> SearchFilter | None:
    """Return a copy of *sf* containing only conditions supported by one target."""

    if sf is None:
        return None

    def keep_clause(clause: FieldCondition | SearchFilter) -> FieldCondition | SearchFilter | None:
        if isinstance(clause, SearchFilter):
            nested = filter_search_filter_fields(clause, allowed_fields)
            return nested if nested is not None else None
        return clause.model_copy(deep=True) if clause.field in allowed_fields else None

    must = [item for clause in sf.must if (item := keep_clause(clause)) is not None]
    should = [item for clause in sf.should if (item := keep_clause(clause)) is not None]
    must_not = [item for clause in sf.must_not if (item := keep_clause(clause)) is not None]
    filtered = SearchFilter(must=must, should=should, must_not=must_not)
    return None if search_filter_is_empty(filtered) else filtered


def search_filter_contains_fields(sf: SearchFilter | None, fields: frozenset[str]) -> bool:
    """Return whether a filter tree references any of *fields*."""

    if sf is None:
        return False
    for clause in [*sf.must, *sf.should, *sf.must_not]:
        if isinstance(clause, SearchFilter):
            if search_filter_contains_fields(clause, fields):
                return True
        elif clause.field in fields:
            return True
    return False


def _context_with_request_scope(ctx: MemoryRequestContext, sf: SearchFilter) -> MemoryRequestContext:
    project_id = _single_positive_match(sf, "project_id")
    if search_filter_contains_fields(sf, frozenset({"project_id"})) and not project_id:
        raise InvalidFilterError("Field 'project_id' in schema_search filters must be a single exact value.")
    if project_id and project_id != ctx.project_id:
        raise InvalidFilterError("Field 'project_id' in schema_search filters must match the request context.")

    values = {field: _single_positive_match(sf, field) for field in _CONTEXT_OVERRIDE_FIELDS}

    updates: dict[str, str | None] = {}
    for field, value in values.items():
        if value is not None:
            updates[field] = value

    # If user_id is present in a non-exact form, clear the context user fallback
    # so graph neighbor hydration does not keep applying the caller header user.
    if "user_id" not in updates and search_filter_contains_fields(sf, frozenset({"user_id"})):
        updates["user_id"] = ""

    return ctx.model_copy(update=updates) if updates else ctx


def _single_positive_match(sf: SearchFilter, field: str) -> str | None:
    found: list[str] = []

    def visit_positive(node: SearchFilter) -> None:
        for clause in node.must:
            if isinstance(clause, SearchFilter):
                visit_positive(clause)
                continue
            if clause.field != field:
                continue
            if clause.op == "match" and isinstance(clause.value, str):
                found.append(clause.value)
            elif clause.op == "any" and clause.values and len(clause.values) == 1 and isinstance(clause.values[0], str):
                found.append(clause.values[0])

    visit_positive(sf)
    if search_filter_contains_fields(SearchFilter(should=sf.should, must_not=sf.must_not), frozenset({field})):
        return None
    unique = set(found)
    if len(unique) > 1:
        raise InvalidFilterError(f"Field '{field}' in schema_search filters has conflicting exact values.")
    return found[0] if found else None
