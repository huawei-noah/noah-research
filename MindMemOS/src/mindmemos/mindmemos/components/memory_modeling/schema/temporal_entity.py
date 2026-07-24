"""Temporal entity instance behavior migrated from the original modeling module."""

from __future__ import annotations

import bisect
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ....typing import EntityView, EntityWrite, GraphRelationship, MemoryView, MemoryWrite
from .edge import Edge
from .entity_manager import EntityManager


@dataclass(frozen=True, slots=True)
class PropertyEntry:
    """One timestamped property value in an entity timeline."""

    timestamp: str
    value: Any
    uid: str

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.timestamp, "value": self.value, "uid": self.uid}


class TemporalEntity:
    """Entity instance with independent temporal property timelines.

    This class intentionally has no database client, LLM client, or embedding
    dependency. Pipelines hydrate it from DTOs, use its entity behavior, and then
    convert results back to write DTOs at the pipeline boundary.
    """

    def __init__(
        self,
        *,
        entity_id: str | None = None,
        name: str = "",
        entity_type: str = "unknown",
        description: str | None = None,
        record_time: str | None = None,
        entity_manager: EntityManager | None = None,
        auto_create_properties: bool = True,
    ) -> None:
        self.entity_id = entity_id or str(uuid4())
        self.name = name
        self.entity_type = entity_type
        self.description = description or ""
        self.record_time = record_time or ""
        self._entity_manager = entity_manager
        self._auto_create_properties = auto_create_properties
        self._properties: dict[str, list[PropertyEntry]] = {}
        self._latest_cache: dict[str, PropertyEntry] = {}
        self.edges: list[Edge] = []
        self.search_fields: list[str] = []
        self._init_schema_properties()

    @property
    def properties(self) -> dict[str, list[PropertyEntry]]:
        return self._properties

    def modify_property(
        self,
        property_name: str,
        value: Any,
        timestamp: str | datetime,
        *,
        operation: str = "set",
        uid: str | None = None,
        auto_create: bool | None = None,
    ) -> bool:
        """Modify a property timeline.

        Args:
            property_name: Property name.
            value: New value. Delete operations ignore this value.
            timestamp: Property event time.
            operation: ``set`` and ``update`` append to the timeline; ``delete`` removes a matching entry.
            uid: Optional precise property value id for deletion.
            auto_create: Whether to create a missing property. Defaults to the instance policy.

        Returns:
            True when the timeline changed; False when the property is missing and cannot be created.

        Raises:
            ValueError: If ``operation`` is unsupported.
        """

        if not property_name:
            return False
        if property_name not in self._properties:
            should_create = self._auto_create_properties if auto_create is None else auto_create
            if not should_create:
                return False
            self._properties[property_name] = []

        timestamp_text = normalize_timestamp(timestamp)
        if operation == "delete":
            self.delete_property_value(property_name, timestamp_text, uid=uid)
            return True
        if operation in {"set", "update"}:
            self.insert_property_value(property_name, timestamp_text, value, uid=uid)
            return True
        raise ValueError(f"Unknown operation: {operation}")

    def insert_property_value(self, property_name: str, timestamp: str, value: Any, *, uid: str | None = None) -> str:
        """Insert one value into a property timeline in timestamp order.

        Args:
            property_name: Property name.
            timestamp: Normalized timestamp string.
            value: Property value.
            uid: Optional unique value id. Generated when omitted.

        Returns:
            UID of the inserted entry.
        """
        entry = PropertyEntry(timestamp=timestamp, value=value, uid=uid or uuid4().hex[:12])
        timeline = self._properties.setdefault(property_name, [])
        idx = self._find_insert_position(property_name, timestamp)
        timeline.insert(idx, entry)
        self._update_latest_cache(property_name)
        return entry.uid

    def delete_property_value(self, property_name: str, timestamp: str, *, uid: str | None = None) -> bool:
        """Delete a matching entry from a property timeline.

        Args:
            property_name: Property name.
            timestamp: Timestamp string used as a fallback match key.
            uid: Optional exact value id to match first.

        Returns:
            True when an entry was removed; False when no match was found.
        """
        timeline = self._properties.get(property_name)
        if not timeline:
            return False

        original_len = len(timeline)
        if uid:
            timeline[:] = [entry for entry in timeline if entry.uid != uid]
            if len(timeline) == original_len:
                timeline[:] = [entry for entry in timeline if entry.timestamp != timestamp]
        else:
            timeline[:] = [entry for entry in timeline if entry.timestamp != timestamp]

        if len(timeline) == original_len:
            return False
        self._update_latest_cache(property_name)
        return True

    def get_properties(
        self,
        *,
        timestamp: str | datetime | None = None,
        time_range: tuple[str | datetime, str | datetime] | None = None,
        property_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a property value snapshot.

        Without a time selector, this returns each property's latest value. With
        ``timestamp`` it returns values at that time. With ``time_range`` it returns
        the last value in range.

        Args:
            timestamp: Query timestamp.
            time_range: Query time range as ``(start, end)``.
            property_names: Optional property names to query. Defaults to all properties.

        Returns:
            Mapping of property name to value.
        """
        names = property_names or list(self._properties)
        if timestamp is None and time_range is None:
            return {name: self._latest_cache.get(name).value if name in self._latest_cache else None for name in names}

        if timestamp is not None:
            target = normalize_timestamp(timestamp)
            return {name: self.get_property_at_time(name, target) for name in names}

        if time_range is not None:
            start, end = time_range
            start_text = normalize_timestamp(start)
            end_text = normalize_timestamp(end)
            return {name: self.get_property_in_range(name, start_text, end_text) for name in names}

        return {}

    def get_property_at_time(self, property_name: str, timestamp: str) -> Any:
        timeline = self._properties.get(property_name, [])
        if not timeline:
            return None
        times = [entry.timestamp for entry in timeline]
        pos = bisect.bisect_right(times, timestamp) - 1
        return timeline[pos].value if pos >= 0 else None

    def get_property_in_range(self, property_name: str, start: str, end: str) -> Any:
        timeline = self._properties.get(property_name, [])
        if not timeline:
            return None
        times = [entry.timestamp for entry in timeline]
        pos_end = bisect.bisect_right(times, end) - 1
        if pos_end < 0:
            return None
        pos_start = bisect.bisect_left(times, start)
        if pos_start <= pos_end:
            return timeline[pos_end].value
        return timeline[pos_end].value

    def get_timeline(self, property_name: str | None = None) -> list[str]:
        if property_name is not None:
            return [entry.timestamp for entry in self._properties.get(property_name, [])]
        return sorted({entry.timestamp for timeline in self._properties.values() for entry in timeline})

    def get_property_history(self, property_name: str, *, include_uid: bool = False) -> list[tuple]:
        """Return the full history for one property.

        Args:
            property_name: Property name.
            include_uid: Whether to include the value UID in each tuple.

        Returns:
            Timestamp-sorted history tuples.
        """
        timeline = self._properties.get(property_name, [])
        if include_uid:
            return [(entry.timestamp, entry.value, entry.uid) for entry in timeline]
        return [(entry.timestamp, entry.value) for entry in timeline]

    def get_properties_in_range(
        self,
        property_range: list[str] | None = None,
        time_range: tuple[str | datetime | None, str | datetime | None] | None = None,
        *,
        include_uid: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return property values in a time range for search display.

        If the time range yields no values, the full timeline is returned as a fallback.

        Args:
            property_range: Optional property names. Defaults to all properties.
            time_range: Optional ``(start, end)`` range. Either side may be None.
            include_uid: Whether to include value UIDs.

        Returns:
            Mapping of property name to timestamped value dictionaries.
        """
        disclosed: dict[str, list[dict[str, Any]]] = {}
        names = property_range or list(self._properties)
        start, end = _normalize_time_range(time_range)

        for prop in names:
            timeline = self._properties.get(prop)
            if not timeline:
                continue

            filtered = timeline
            if time_range is not None:
                filtered = [
                    entry
                    for entry in timeline
                    if (start is None or entry.timestamp >= start) and (end is None or entry.timestamp <= end)
                ]
                if not filtered:
                    filtered = timeline

            values: list[dict[str, Any]] = []
            for entry in filtered:
                if entry.value is None:
                    continue
                item = {"timestamp": entry.timestamp, "value": entry.value}
                if include_uid:
                    item["uid"] = entry.uid
                values.append(item)
            if values:
                disclosed[prop] = values
        return disclosed

    def filter_by_time(
        self,
        time_window: tuple[str | datetime | None, str | datetime | None],
        *,
        new_entity_id: str | None = None,
    ) -> TemporalEntity:
        """Filter properties by time window and return a new entity copy.

        Episode entities are not filtered and keep all properties.

        Args:
            time_window: ``(start, end)`` window. Either side may be None for an unbounded range.
            new_entity_id: Optional id for the returned copy.

        Returns:
            Filtered TemporalEntity copy.
        """
        if self.entity_type == "episodes":
            return self._copy_with_properties(
                {prop: list(timeline) for prop, timeline in self._properties.items()}, new_entity_id=new_entity_id
            )

        start, end = _normalize_time_range(time_window)
        if start is not None and end is not None and start > end:
            raise ValueError("Start time must be before end time")

        return self._copy_with_properties(
            {
                prop: [
                    entry
                    for entry in timeline
                    if (start is None or entry.timestamp >= start) and (end is None or entry.timestamp <= end)
                ]
                for prop, timeline in self._properties.items()
            },
            new_entity_id=new_entity_id,
        )

    def filter_by_timepoints(
        self,
        timepoints: list[str | datetime],
        *,
        new_entity_id: str | None = None,
    ) -> TemporalEntity:
        if self.entity_type == "episodes":
            return self._copy_with_properties(
                {prop: list(timeline) for prop, timeline in self._properties.items()}, new_entity_id=new_entity_id
            )

        normalized = {normalize_timestamp(timepoint) for timepoint in timepoints}
        if not normalized:
            raise ValueError("timepoints list cannot be empty")

        return self._copy_with_properties(
            {
                prop: [entry for entry in timeline if entry.timestamp in normalized]
                for prop, timeline in self._properties.items()
            },
            new_entity_id=new_entity_id,
        )

    def get_time_range(self) -> tuple[str, str] | None:
        timeline = self.get_timeline()
        if not timeline:
            return None
        return timeline[0], timeline[-1]

    def format_entity_prompt(
        self,
        ignore_edge_num: int | None = None,
        *,
        include_description: bool = True,
        include_edges: bool = True,
    ) -> str:
        block = f"Entity: {self.name} (Type: {self.entity_type})"
        if include_description and self.description:
            description = self.description[:2000] + "..." if len(self.description) > 2000 else self.description
            block += f"\n   Description: {description}"

        for prop, timeline in self._properties.items():
            if not timeline:
                continue
            first = timeline[0]
            if len(timeline) > 1 or first.timestamp != self.record_time:
                block += f"\n   Property '{prop}':"
                for entry in timeline:
                    if entry.value is not None:
                        block += f"\n     - {entry.timestamp}: {entry.value}"
            elif first.value is not None:
                block += f"\n   Property '{prop}': {first.value}"

        if include_edges and self.edges:
            if ignore_edge_num is not None and len(self.edges) > ignore_edge_num:
                return block
            block += "\n   Edges:"
            for edge in self.edges:
                block += f"\n     - {edge.link_entity1_name} --[{edge.link_description}]--> {edge.link_entity2_name}"
        return block

    def format_short_description(self) -> str:
        return f"Name: {self.name}, Type: {self.entity_type}, Description: {self.description}"

    def set_search_fields(self, fields: list[str], *, max_fields: int | None = None) -> None:
        deduped: list[str] = []
        seen: set[str] = set()
        for field in fields:
            value = str(field).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
            if max_fields is not None and len(deduped) >= max_fields:
                break
        self.search_fields = deduped

    def apply_memory(self, memory: MemoryView | MemoryWrite, *, include_inactive: bool = False) -> bool:
        """Append one memory DTO to this entity's property timeline.

        Args:
            memory: Memory DTO.
            include_inactive: Whether to accept memories whose status is not active.

        Returns:
            True when a property value was appended; False when the memory does not apply.
        """
        if not include_inactive and memory.status != "active":
            return False
        if not memory.property_name:
            return False
        if memory.entity_id and memory.entity_id != self.entity_id:
            return False
        return self.modify_property(
            memory.property_name,
            memory.content,
            memory_timestamp(memory),
            operation="set",
            uid=memory.memory_id,
            auto_create=True,
        )

    def add_edge(
        self,
        *,
        target_entity_id: str,
        target_entity_name: str,
        description: str,
        reverse: bool = False,
        update_if_exists: bool = True,
    ) -> Edge | None:
        """Add an entity-to-entity relationship edge.

        When an edge to the same target already exists, ``update_if_exists=True``
        updates the description and returns None.

        Args:
            target_entity_id: Target entity id.
            target_entity_name: Target entity name.
            description: Relationship description.
            reverse: Whether the edge direction is target to self.
            update_if_exists: Whether to update an existing edge.

        Returns:
            New edge, or None when an existing edge was updated.
        """
        for edge in self.edges:
            existing_target_id = edge.link_entity1_id if reverse else edge.link_entity2_id
            if existing_target_id == target_entity_id:
                if update_if_exists:
                    edge.link_description = description
                return None

        if reverse:
            edge = Edge(
                link_entity1_id=target_entity_id,
                link_entity1_name=target_entity_name,
                link_entity2_id=self.entity_id,
                link_entity2_name=self.name,
                link_description=description,
            )
        else:
            edge = Edge(
                link_entity1_id=self.entity_id,
                link_entity1_name=self.name,
                link_entity2_id=target_entity_id,
                link_entity2_name=target_entity_name,
                link_description=description,
            )
        self.edges.append(edge)
        return edge

    def find_connected_entities(self) -> list[dict[str, Any]]:
        return [
            {
                "source_id": edge.get_entity1_id(),
                "source_name": edge.link_entity1_name,
                "target_id": edge.get_entity2_id(),
                "target_name": edge.link_entity2_name,
                "relation": edge.link_description,
            }
            for edge in self.edges
        ]

    def to_entity_view(self, *, project_id: str) -> EntityView:
        return EntityView(
            entity_id=self.entity_id,
            project_id=project_id,
            entity_name=self.name,
            entity_type=self.entity_type,
            description=self.description,
            metadata={"record_time": self.record_time, "search_fields": list(self.search_fields)},
        )

    def to_graph_relationships(self, *, project_id: str) -> list[GraphRelationship]:
        return [edge.to_graph_relationship(project_id=project_id) for edge in self.edges]

    def transform_to_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "record_time": self.record_time,
            "properties": {
                prop: [entry.to_dict() for entry in timeline] for prop, timeline in self._properties.items() if timeline
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "search_fields": list(self.search_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, entity_manager: EntityManager | None = None) -> TemporalEntity:
        entity = cls(
            entity_id=str(data.get("entity_id") or ""),
            name=str(data.get("name") or ""),
            entity_type=str(data.get("entity_type") or "unknown"),
            description=data.get("description"),
            record_time=str(data.get("record_time") or ""),
            entity_manager=entity_manager,
        )
        entity._properties.clear()
        for prop, timeline in (data.get("properties") or {}).items():
            entity._properties[prop] = [
                PropertyEntry(
                    timestamp=normalize_timestamp(item.get("time") or item.get("timestamp") or ""),
                    value=item.get("value"),
                    uid=str(item.get("uid") or uuid4().hex[:12]),
                )
                for item in timeline
                if isinstance(item, dict)
            ]
            entity._properties[prop].sort(key=lambda entry: entry.timestamp)
            entity._update_latest_cache(prop)
        entity.edges = [Edge.from_dict(item) for item in data.get("edges", []) if isinstance(item, dict)]
        entity.set_search_fields([str(field) for field in data.get("search_fields", [])])
        return entity

    @classmethod
    def from_entity_dto(
        cls,
        entity: EntityView | EntityWrite,
        *,
        entity_manager: EntityManager | None = None,
    ) -> TemporalEntity:
        """Build a TemporalEntity from an entity DTO without property values.

        Args:
            entity: Entity DTO.
            entity_manager: Optional schema manager used to initialize schema properties.

        Returns:
            TemporalEntity instance.
        """
        modeled = cls(
            entity_id=entity.entity_id,
            name=entity.entity_name,
            entity_type=entity.entity_type or "unknown",
            description=entity.description,
            record_time=str(entity.metadata.get("record_time", "")),
            entity_manager=entity_manager,
        )
        modeled.set_search_fields([str(field) for field in entity.metadata.get("search_fields", [])])
        return modeled

    @classmethod
    def from_views(
        cls,
        entity: EntityView | EntityWrite,
        memories: list[MemoryView | MemoryWrite],
        *,
        entity_manager: EntityManager | None = None,
    ) -> TemporalEntity:
        """Build a TemporalEntity from an entity DTO and its memory list."""
        modeled = cls.from_entity_dto(entity, entity_manager=entity_manager)
        for memory in memories:
            modeled.apply_memory(memory)
        return modeled

    def _init_schema_properties(self) -> None:
        if self._entity_manager is None:
            return
        schema = self._entity_manager.get(self.entity_type)
        if schema is None:
            return
        for prop in schema.all_property_names():
            self._properties.setdefault(prop, [])

    def _copy_with_properties(
        self, properties: dict[str, list[PropertyEntry]], *, new_entity_id: str | None = None
    ) -> TemporalEntity:
        clone = TemporalEntity(
            entity_id=new_entity_id or self.entity_id,
            name=self.name,
            entity_type=self.entity_type,
            description=self.description,
            record_time=self.record_time,
            entity_manager=self._entity_manager,
            auto_create_properties=self._auto_create_properties,
        )
        clone._properties = {prop: list(timeline) for prop, timeline in properties.items()}
        for prop in self._properties:
            clone._properties.setdefault(prop, [])
        clone.edges = deepcopy(self.edges)
        clone.search_fields = list(self.search_fields)
        clone._rebuild_cache()
        return clone

    def _rebuild_cache(self) -> None:
        self._latest_cache = {}
        for prop in self._properties:
            self._update_latest_cache(prop)

    def _find_insert_position(self, property_name: str, timestamp: str) -> int:
        timeline = self._properties.get(property_name, [])
        times = [entry.timestamp for entry in timeline]
        return bisect.bisect_right(times, timestamp)

    def _update_latest_cache(self, property_name: str) -> None:
        timeline = self._properties.get(property_name, [])
        if timeline:
            self._latest_cache[property_name] = timeline[-1]
        else:
            self._latest_cache.pop(property_name, None)

    def __repr__(self) -> str:
        return f"TemporalEntity(entity_id={self.entity_id!r}, name={self.name!r}, entity_type={self.entity_type!r})"


def normalize_timestamp(value: str | datetime) -> str:
    """Normalize a timestamp to an ISO-format string."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def memory_timestamp(memory: MemoryView | MemoryWrite) -> str:
    """Extract the property timestamp from a memory DTO."""
    value = memory.metadata.get("property_time") or memory.metadata.get("record_time")
    if value:
        return str(value)
    if memory.created_at:
        return normalize_timestamp(memory.created_at)
    return ""


def _normalize_time_range(
    time_range: tuple[str | datetime | None, str | datetime | None] | None,
) -> tuple[str | None, str | None]:
    if time_range is None:
        return None, None
    start, end = time_range
    return normalize_timestamp(start) if start is not None else None, normalize_timestamp(
        end
    ) if end is not None else None
