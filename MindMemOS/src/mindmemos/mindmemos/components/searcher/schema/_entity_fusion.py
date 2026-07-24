"""Merge entity results returned by schema search retrieval paths."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from ....logging import get_logger
from ...memory_modeling.schema import TemporalEntity

logger = get_logger(__name__)


class SchemaSearchEntityFusionManager:
    """Merge entity results returned from multiple retrieval paths."""

    def __init__(self, entity_manager: Any | None = None) -> None:
        self._entity_manager = entity_manager

    def fuse_entities(
        self,
        entity_results: list[TemporalEntity],
        property_results: list[TemporalEntity],
    ) -> list[TemporalEntity]:
        """Handle fuse entities."""
        if not entity_results and not property_results:
            return []

        if not entity_results:
            logger.info("fusion_shortcut", reason="only_property_results")
            return property_results

        if not property_results:
            logger.info("fusion_shortcut", reason="only_entity_results")
            return entity_results

        entity_dict: dict[str, TemporalEntity] = {}
        merged_count = 0
        extended_count = 0

        for entity in entity_results:
            if entity and entity.entity_id:
                entity_dict[entity.entity_id] = entity
                entity._fusion_sources = ["entity_search"]

        for prop_entity in property_results:
            if not prop_entity or not prop_entity.entity_id:
                continue

            entity_id = prop_entity.entity_id

            if entity_id in entity_dict:
                existing_entity = entity_dict[entity_id]
                self._merge_entity_properties(existing_entity, prop_entity)

                if hasattr(existing_entity, "_fusion_sources"):
                    existing_entity._fusion_sources.append("property_search")
                else:
                    existing_entity._fusion_sources = ["entity_search", "property_search"]

                merged_count += 1
                logger.debug("merge_entity_properties", entity_name=existing_entity.name)
            else:
                prop_entity._fusion_sources = ["property_search"]
                entity_dict[entity_id] = prop_entity
                extended_count += 1
                logger.debug("extend_new_entity", entity_name=prop_entity.name)

        result_entities: list[TemporalEntity] = []

        for entity in entity_results:
            if entity and entity.entity_id and entity.entity_id in entity_dict:
                result_entities.append(entity_dict[entity.entity_id])
                del entity_dict[entity.entity_id]

        additional_entities = list(entity_dict.values())
        if additional_entities:
            additional_entities.sort(
                key=lambda e: getattr(e, "_property_search_score", 0.0),
                reverse=True,
            )
            result_entities.extend(additional_entities)

        logger.info(
            "fusion_complete",
            entity_count=len(entity_results),
            property_count=len(property_results),
            merged=merged_count,
            extended=extended_count,
            total=len(result_entities),
        )

        return result_entities

    def _merge_entity_properties(
        self,
        target_entity: TemporalEntity,
        source_entity: TemporalEntity,
    ) -> None:
        """Handle merge entity properties."""
        if not source_entity._properties:
            return

        for property_name, source_timeline in source_entity._properties.items():
            if property_name not in target_entity._properties:
                target_entity._properties[property_name] = list(source_timeline)
            else:
                target_timeline = target_entity._properties[property_name]

                existing_uids: set[str] = set()
                for entry in target_timeline:
                    existing_uids.add(entry.uid)

                for entry in source_timeline:
                    if entry.uid and entry.uid not in existing_uids:
                        target_timeline.append(entry)
                        existing_uids.add(entry.uid)
                    elif not entry.uid:
                        target_timeline.append(entry)

                target_entity._properties[property_name] = sorted(
                    target_timeline,
                    key=lambda x: x.timestamp if x else "",
                    reverse=True,
                )

    def assemble_entity_from_properties(
        self,
        property_results: list[dict[str, Any]],
        entity_template_source: TemporalEntity | None = None,
    ) -> list[TemporalEntity]:
        """Handle assemble entity from properties."""
        entity_properties: dict[str, list[dict[str, Any]]] = defaultdict(list)
        entity_metadatas: dict[str, dict[str, str]] = {}

        for result in property_results:
            metadata = result.get("metadata", {})
            entity_id = metadata.get("entity_id")

            if not entity_id:
                continue

            if entity_id not in entity_metadatas:
                entity_metadatas[entity_id] = {
                    "entity_name": metadata.get("entity_name", f"Entity_{entity_id}"),
                    "entity_type": metadata.get("entity_type", "unknown"),
                    "entity_id": entity_id,
                }

            property_info = {
                "property_name": metadata.get("property_name", "unknown"),
                "property_value": metadata.get("property_value", ""),
                "timestamp": metadata.get("timestamp", ""),
                "uid": metadata.get("uid", ""),
                "operation": metadata.get("operation", "set"),
                "score": result.get("score", 0.0),
            }
            entity_properties[entity_id].append(property_info)

        assembled_entities: list[TemporalEntity] = []
        for entity_id, properties in entity_properties.items():
            meta = entity_metadatas[entity_id]

            entity = TemporalEntity(
                entity_id=entity_id,
                name=meta["entity_name"],
                entity_type=meta["entity_type"],
                description=f"Entity assembled from property search: {meta['entity_name']}",
                entity_manager=self._entity_manager,
            )

            # Group properties by name, then insert via insert_property_value
            property_timelines: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            for prop in properties:
                ts = prop["timestamp"] if prop["timestamp"] else ""
                uid = prop.get("uid", "") or uuid.uuid4().hex[:12]
                property_timelines[prop["property_name"]].append((ts, prop["property_value"], uid))

            for prop_name, timeline in property_timelines.items():
                sorted_timeline = sorted(
                    timeline,
                    key=lambda x: x[0] or "",
                    reverse=True,
                )
                for ts, value, uid in sorted_timeline:
                    entity.insert_property_value(prop_name, ts, value, uid=uid)

            entity._property_search_score = sum(prop["score"] for prop in properties) / len(properties)
            assembled_entities.append(entity)

        logger.info("assemble_from_properties", assembled_count=len(assembled_entities))
        return assembled_entities
