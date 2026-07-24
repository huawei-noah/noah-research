"""Schema add extraction normalization."""

from __future__ import annotations

from typing import Any

from ...memory_modeling.schema import EntitySchemaProvider
from .base import SchemaExtractionNormalizerProtocol


class SchemaExtractionNormalizer(SchemaExtractionNormalizerProtocol):
    """Normalize and validate raw schema extraction output without DB or LLM access."""

    def __init__(self, *, entity_manager: EntitySchemaProvider) -> None:
        self.entity_manager = entity_manager

    def normalize(self, raw_memory: dict[str, Any], dialogue_timestamp: str) -> dict[str, Any]:
        """Normalize extracted entities and edges for downstream planning."""

        raw_memory.setdefault("entities", [])
        raw_memory.setdefault("edges", [])
        record_time_default = dialogue_timestamp.split(" ", 1)[0]
        prepared_entities: list[dict[str, Any]] = []
        for entity in raw_memory.get("entities", []):
            if entity.get("entity_type") == "episodes":
                continue
            entity["record_time"] = entity.get("record_time") or record_time_default
            entity.setdefault("properties", [])
            for prop in entity["properties"]:
                prop.setdefault("operation", "set")
                prop.setdefault("time", entity["record_time"])
            prepared_entities.append(entity)
        raw_memory["entities"] = prepared_entities
        return raw_memory

    def validate(self, raw_memory: dict[str, Any]) -> str | None:
        """Validate raw schema extraction output and repair safe schema mismatches."""

        if not raw_memory or "entities" not in raw_memory:
            return None

        entity_names = {entity.get("name") for entity in raw_memory.get("entities", []) if entity.get("name")}
        edge_entities: set[str] = set()
        for edge in raw_memory.get("edges", []):
            if edge.get("link_entity1_name"):
                edge_entities.add(edge["link_entity1_name"])
            if edge.get("link_entity2_name"):
                edge_entities.add(edge["link_entity2_name"])
        missing = edge_entities - entity_names
        if missing:
            return f"Edge references entities not in entity list: {sorted(missing)}"

        valid_types = set(self.entity_manager.list_types())
        type_properties: dict[str, set[str]] = {}
        for schema_item in self.entity_manager.get_all_dicts():
            entity_type = schema_item.get("entity_type")
            if not entity_type:
                continue
            props: set[str] = set()
            static = schema_item.get("static_property", {})
            dynamic = schema_item.get("dynamic_property", {})
            if isinstance(static, dict):
                props.update(static)
            if isinstance(dynamic, dict):
                props.update(dynamic)
            type_properties[entity_type] = props

        fallback_types = [entity_type for entity_type in valid_types if entity_type != "episodes"]
        fallback_type = sorted(fallback_types)[0] if fallback_types else None
        for entity in raw_memory.get("entities", []):
            entity_type = entity.get("entity_type")
            if entity_type and entity_type not in valid_types and fallback_type:
                entity["entity_type"] = fallback_type

            valid_props = type_properties.get(entity.get("entity_type"), set())
            has_default = "default_property" in valid_props
            for prop in entity.get("properties", []):
                prop_name = prop.get("property_name")
                if prop_name and valid_props and prop_name not in valid_props and has_default:
                    prop["property_name"] = "default_property"
        return None
