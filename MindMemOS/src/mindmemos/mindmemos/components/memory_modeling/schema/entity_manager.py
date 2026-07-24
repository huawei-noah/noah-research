"""File-backed entity schema manager used by memory modeling components."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from ....config import REPO_ROOT, get_config


@dataclass
class EntityProperty:
    """One structured entity property definition."""

    description: str = ""
    examples: list[str] = field(default_factory=list)
    constraints: str = ""
    value_type: str = "string"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityProperty:
        return cls(
            description=data.get("description") or data.get("desc", ""),
            examples=list(data.get("examples") or ([data["example"]] if data.get("example") else [])),
            constraints=data.get("constraints", ""),
            value_type=data.get("value_type") or data.get("type", "string"),
        )


@dataclass
class EntityType:
    """One entity type schema from entity_modeling.json."""

    entity_type: str
    entity_description: str = ""
    entity_instruction: str = ""
    search_weight: float = 1.0
    static_property: dict[str, Any] = field(default_factory=dict)
    dynamic_property: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_description": self.entity_description,
            "entity_instruction": self.entity_instruction,
            "search_weight": self.search_weight,
            "static_property": deepcopy(self.static_property),
            "dynamic_property": deepcopy(self.dynamic_property),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityType:
        return cls(
            entity_type=data["entity_type"],
            entity_description=data.get("entity_description", ""),
            entity_instruction=data.get("entity_instruction", ""),
            search_weight=float(data.get("search_weight", 1.0)),
            static_property=deepcopy(data.get("static_property", {})),
            dynamic_property=deepcopy(data.get("dynamic_property", {})),
        )

    def all_property_names(self) -> set[str]:
        return set(self.static_property) | set(self.dynamic_property)


class EntityManager:
    """Manage entity modeling schemas from a configured JSON file."""

    def __init__(self, file_path: str | Path | None = None) -> None:
        self._file_path = _resolve_schema_path(file_path)
        self._entities: dict[str, EntityType] = {}
        self._dirty = False
        self.load_from_file()

    @property
    def file_path(self) -> Path:
        return self._file_path

    def get_all(self) -> list[EntityType]:
        """Return all registered entity type schemas."""
        return list(self._entities.values())

    def get_all_dicts(self) -> list[dict[str, Any]]:
        """Return all entity type schemas as dictionaries for prompts."""
        return [entity.to_dict() for entity in self.get_all()]

    def get(self, entity_type: str) -> EntityType | None:
        return self._entities.get(entity_type)

    def get_dict(self, entity_type: str) -> dict[str, Any] | None:
        entity = self.get(entity_type)
        return entity.to_dict() if entity else None

    def list_types(self) -> list[str]:
        return list(self._entities.keys())

    def exists(self, entity_type: str) -> bool:
        return entity_type in self._entities

    def register(
        self,
        entity_type: str,
        *,
        entity_description: str = "",
        entity_instruction: str = "",
        search_weight: float | None = None,
        static_property: dict[str, Any] | None = None,
        dynamic_property: dict[str, Any] | None = None,
        merge: bool = False,
    ) -> EntityType:
        """Register or update an entity type schema.

        Args:
            entity_type: Entity type name.
            entity_description: Entity type description.
            entity_instruction: Extraction instruction for the LLM.
            search_weight: Retrieval weight.
            static_property: Static property definitions.
            dynamic_property: Dynamic property definitions.
            merge: Whether to merge into an existing schema instead of replacing it.

        Returns:
            Registered entity type schema.
        """
        static = deepcopy(static_property or {})
        dynamic = deepcopy(dynamic_property or {})

        if merge and entity_type in self._entities:
            existing = self._entities[entity_type]
            if entity_description:
                existing.entity_description = entity_description
            if entity_instruction:
                existing.entity_instruction = entity_instruction
            if search_weight is not None:
                existing.search_weight = search_weight
            existing.static_property.update(static)
            existing.dynamic_property.update(dynamic)
        else:
            self._entities[entity_type] = EntityType(
                entity_type=entity_type,
                entity_description=entity_description,
                entity_instruction=entity_instruction,
                search_weight=search_weight if search_weight is not None else 1.0,
                static_property=static,
                dynamic_property=dynamic,
            )

        self._dirty = True
        return self._entities[entity_type]

    def update_property(
        self, entity_type: str, property_name: str, property_value: Any, *, is_static: bool = True
    ) -> bool:
        entity = self.get(entity_type)
        if entity is None:
            return False
        target = entity.static_property if is_static else entity.dynamic_property
        target[property_name] = deepcopy(property_value)
        self._dirty = True
        return True

    def get_properties_by_order(self, entity_type: str, order: int = 1) -> dict[str, Any]:
        """Return dynamic properties with the requested order.

        Args:
            entity_type: Entity type name.
            order: Property order, where 1 is first-order and 2 or higher is higher-order.

        Returns:
            Mapping of property name to definition, or an empty mapping when the type is unknown.
        """
        entity = self.get(entity_type)
        if entity is None:
            return {}

        result: dict[str, Any] = {}
        for name, definition in entity.dynamic_property.items():
            if isinstance(definition, dict):
                if definition.get("order", 1) == order:
                    result[name] = deepcopy(definition)
            elif order == 1:
                result[name] = definition
        return result

    def has_higher_order_properties(self, entity_type: str) -> bool:
        """Return whether the entity type defines higher-order properties."""
        return bool(self.get_higher_order_property_names(entity_type))

    def get_higher_order_property_names(self, entity_type: str) -> set[str]:
        """Return all higher-order property names for an entity type.

        Args:
            entity_type: Entity type name.

        Returns:
            Dynamic property names with order greater than or equal to 2.
        """
        entity = self.get(entity_type)
        if entity is None:
            return set()
        return {
            name
            for name, definition in entity.dynamic_property.items()
            if isinstance(definition, dict) and definition.get("order", 1) >= 2
        }

    def load_from_file(self, file_path: str | Path | None = None) -> bool:
        """Load entity schemas from a JSON file.

        Args:
            file_path: Schema file path. Uses the configured path when omitted.

        Returns:
            True when schemas were loaded.

        Raises:
            FileNotFoundError: If the schema file does not exist.
            ValueError: If the JSON root type is unsupported.
        """
        path = _resolve_schema_path(file_path) if file_path is not None else self._file_path
        if not path.exists():
            raise FileNotFoundError(f"entity modeling file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict):
            entity_list = data.get("entity_types", [])
        elif isinstance(data, list):
            entity_list = data
        else:
            raise ValueError(f"unsupported entity modeling JSON root: {type(data).__name__}")

        self._entities.clear()
        for item in entity_list:
            entity = EntityType.from_dict(item)
            self._entities[entity.entity_type] = entity
        self._file_path = path
        self._dirty = False
        return True

    def save_to_file(self, file_path: str | Path | None = None, *, format: str = "list") -> Path:
        """Atomically write current entity schemas to a JSON file.

        Args:
            file_path: Destination path. Uses the configured path when omitted.
            format: ``list`` writes a plain array; ``object`` writes ``{version, entity_types}``.

        Returns:
            Written file path.
        """
        path = _resolve_schema_path(file_path) if file_path is not None else self._file_path
        path.parent.mkdir(parents=True, exist_ok=True)

        entity_list = [entity.to_dict() for entity in self._entities.values()]
        data: Any = {"version": "1.0", "entity_types": entity_list} if format == "object" else entity_list

        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        temp_path.replace(path)
        self._dirty = False
        return path

    def is_dirty(self) -> bool:
        return self._dirty


_managers: dict[str, EntityManager] = {}
_manager_lock = Lock()

_DEFAULT_PROJECT = "__default__"


def get_entity_manager(
    *, project_id: str | None = None, file_path: str | Path | None = None, reload: bool = False
) -> EntityManager:
    """Return the entity manager for a given project.

    Args:
        project_id: Project isolation key. None falls back to a shared default instance.
        file_path: Override schema file path.
        reload: Force reload from disk.
    """

    key = project_id or _DEFAULT_PROJECT
    target_path = _resolve_schema_path(file_path)
    with _manager_lock:
        existing = _managers.get(key)
        if reload or existing is None or existing.file_path != target_path:
            _managers[key] = EntityManager(target_path)
        return _managers[key]


def _resolve_schema_path(file_path: str | Path | None = None) -> Path:
    if file_path is None:
        file_path = get_config().algo_config.add.schema.entity_modeling_path
    path = Path(str(file_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()
