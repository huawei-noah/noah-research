"""Map user-facing filter DSLs to internal business memory DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from ..errors import InvalidFilterError
from ..typing import (
    DSL_DATETIME_FIELDS,
    DSL_FILTERABLE_MEMORY_FIELDS,
    DSL_TEXT_FIELDS,
    FieldCondition,
    SearchFilter,
)

_LOGICAL_KEYS = frozenset({"AND", "OR", "NOT"})
_RANGE_OPS = frozenset({"gt", "gte", "lt", "lte"})

FieldCategory = Literal["keyword", "datetime", "text"]


def parse_search_dsl(
    filters: dict[str, Any] | None = None,
    *,
    extra_filterable_fields: set[str] | frozenset[str] | None = None,
) -> SearchFilter:
    """Handle parse search dsl."""

    if not filters:
        return SearchFilter()
    allowed_fields = DSL_FILTERABLE_MEMORY_FIELDS | frozenset(extra_filterable_fields or ())
    return _parse_node(filters, allowed_fields)


def _category(field: str) -> FieldCategory:
    if field in DSL_DATETIME_FIELDS:
        return "datetime"
    if field in DSL_TEXT_FIELDS:
        return "text"
    return "keyword"


def _parse_node(node: Any, allowed_fields: frozenset[str]) -> SearchFilter:
    if not isinstance(node, dict):
        raise InvalidFilterError(f"Filter clause must be an object, got {type(node).__name__}.")
    if not node:
        return SearchFilter()

    must: list[FieldCondition | SearchFilter] = []
    should: list[FieldCondition | SearchFilter] = []
    must_not: list[FieldCondition | SearchFilter] = []

    for key, value in node.items():
        if key in _LOGICAL_KEYS:
            clauses = _parse_logical(key, value, allowed_fields)
            if key == "AND":
                must.extend(clauses)
            elif key == "OR":
                should.extend(clauses)
            else:
                must_not.extend(clauses)
        else:
            must.append(_parse_field(key, value, allowed_fields))

    return SearchFilter(must=must, should=should, must_not=must_not)


def _parse_logical(key: str, value: Any, allowed_fields: frozenset[str]) -> list[SearchFilter]:
    if not isinstance(value, list):
        raise InvalidFilterError(f"'{key}' expects a list of filter clauses.")
    return [_parse_node(item, allowed_fields) for item in value]


def _parse_field(field: str, value: Any, allowed_fields: frozenset[str]) -> FieldCondition | SearchFilter:
    if field not in allowed_fields:
        raise InvalidFilterError(f"Field '{field}' is not allowed in search filters.")
    category = _category(field)
    if isinstance(value, dict):
        return _parse_ops(field, category, value)
    if value == "*":
        return SearchFilter(must_not=[FieldCondition(field=field, op="is_empty")])
    # A bare scalar is an exact match, which is only meaningful for keyword fields.
    if category != "keyword":
        raise InvalidFilterError(
            f"Field '{field}' is a {category} field and requires an operator object "
            f"(e.g. {'gte/lt' if category == 'datetime' else 'contains/icontains'}), not a bare value."
        )
    return _match_condition(field, value)


def _parse_ops(field: str, category: FieldCategory, ops: dict[str, Any]) -> FieldCondition | SearchFilter:
    if not ops:
        raise InvalidFilterError(f"Field '{field}' has an empty operator object.")

    conditions: list[FieldCondition | SearchFilter] = []
    range_kwargs: dict[str, Any] = {}

    for op, operand in ops.items():
        if op in _RANGE_OPS:
            _require_category(field, category, "datetime", op)
            range_kwargs[op] = _coerce_datetime_operand(field, operand)
        elif op == "eq":
            _require_category(field, category, "keyword", op)
            conditions.append(_match_condition(field, operand))
        elif op == "ne":
            _require_category(field, category, "keyword", op)
            conditions.append(SearchFilter(must_not=[_match_condition(field, operand)]))
        elif op == "in":
            _require_category(field, category, "keyword", op)
            conditions.append(FieldCondition(field=field, op="any", values=_list_operand(field, op, operand)))
        elif op == "nin":
            _require_category(field, category, "keyword", op)
            conditions.append(FieldCondition(field=field, op="except", values=_list_operand(field, op, operand)))
        elif op in ("contains", "icontains"):
            _require_category(field, category, "text", op)
            conditions.append(_text_condition(field, operand))
        else:
            raise InvalidFilterError(f"Unsupported operator '{op}' for field '{field}'.")

    if range_kwargs:
        conditions.append(FieldCondition(field=field, op="datetime", **range_kwargs))

    if len(conditions) == 1:
        return conditions[0]
    return SearchFilter(must=conditions)


def _require_category(field: str, actual: FieldCategory, expected: FieldCategory, op: str) -> None:
    if actual != expected:
        raise InvalidFilterError(
            f"Operator '{op}' is not supported on '{field}' ({actual} field); it only applies to {expected} fields."
        )


def _match_condition(field: str, value: Any) -> FieldCondition:
    if not isinstance(value, str):
        raise InvalidFilterError(f"Field '{field}' match value must be a string, got {type(value).__name__}.")
    return FieldCondition(field=field, op="match", value=value)


def _list_operand(field: str, op: str, operand: Any) -> list[str]:
    if not isinstance(operand, list) or not operand:
        raise InvalidFilterError(f"Field '{field}' '{op}' expects a non-empty list.")
    if not all(isinstance(item, str) for item in operand):
        raise InvalidFilterError(f"Field '{field}' '{op}' values must all be strings.")
    return list(operand)


def _text_condition(field: str, operand: Any) -> FieldCondition:
    if not isinstance(operand, str):
        raise InvalidFilterError(f"Field '{field}' contains value must be a string, got {type(operand).__name__}.")
    return FieldCondition(field=field, op="text", value=operand)


def _coerce_datetime_operand(field: str, operand: Any) -> datetime:
    if isinstance(operand, datetime):
        return operand
    if isinstance(operand, str):
        try:
            return datetime.fromisoformat(operand.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidFilterError(f"Field '{field}' range value '{operand}' is not a valid datetime.") from exc
    raise InvalidFilterError(
        f"Field '{field}' range value must be a datetime or ISO string, got {type(operand).__name__}."
    )
