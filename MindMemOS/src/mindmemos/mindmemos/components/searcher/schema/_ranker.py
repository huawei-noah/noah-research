"""Schema search result ranking and formatting helpers."""

from __future__ import annotations

import copy

from ...memory_modeling.schema import TemporalEntity


class SchemaSearchRanker:
    """Post-process and format schema search entities."""

    @staticmethod
    def merge_entity_properties_and_edges(existing: TemporalEntity, new_entity: TemporalEntity) -> TemporalEntity:
        """Merge properties and edges from a newer search result into an existing entity."""

        for prop_name, timeline in new_entity._properties.items():
            if prop_name not in existing._properties:
                existing._properties[prop_name] = copy.deepcopy(timeline)
                continue
            existing_timeline = existing._properties[prop_name]
            existing_uids = {entry.uid for entry in existing_timeline}
            added_count = 0
            for entry in timeline:
                uid = entry.uid
                if uid and uid not in existing_uids:
                    existing_timeline.append(entry)
                    existing_uids.add(uid)
                    added_count += 1
                elif not uid:
                    existing_timeline.append(entry)
                    added_count += 1
            if added_count > 0:
                existing._properties[prop_name] = sorted(existing_timeline, key=lambda x: x.timestamp)

        existing_edge_ids: set[tuple[str, str]] = set()
        for edge in existing.edges:
            target_id = edge.link_entity2_id if hasattr(edge, "link_entity2_id") else None
            desc = edge.link_description if hasattr(edge, "link_description") else None
            if target_id and desc:
                existing_edge_ids.add((target_id, desc))

        for new_edge in new_entity.edges:
            target_id = new_edge.link_entity2_id if hasattr(new_edge, "link_entity2_id") else None
            desc = new_edge.link_description if hasattr(new_edge, "link_description") else None
            if target_id and desc and (target_id, desc) not in existing_edge_ids:
                existing.edges.append(new_edge)

        return existing

    @staticmethod
    def format_entities_for_prompt(
        entities: list[TemporalEntity],
        *,
        max_edge_num: int | None = None,
        include_edges: bool,
    ) -> str:
        """Format schema search entities into prompt text."""

        if not entities:
            return "No entities found."

        lines: list[str] = []
        for i, entity in enumerate(entities, 1):
            block = f"{i}." + entity.format_entity_prompt(
                ignore_edge_num=max_edge_num,
                include_description=False,
                include_edges=include_edges,
            )
            lines.append(block)
        return "\n\n".join(lines)

    @staticmethod
    def filter_empty_properties(entities: list[TemporalEntity]) -> tuple[list[TemporalEntity], list[TemporalEntity]]:
        """Remove entities that have no disclosed properties."""

        kept = [entity for entity in entities if any(entity._properties.values())]
        removed = [entity for entity in entities if not any(entity._properties.values())]
        return kept, removed
