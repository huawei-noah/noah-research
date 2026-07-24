"""Higher-order memory generation for schema add."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Protocol

from ....llm import LLMClient
from ....logging import get_logger
from ....prompts import AddPromptSet
from ....typing import EntityWrite, MemoryRequestContext, MemoryView, MemoryWrite
from ...memory_modeling.schema import TemporalEntity, memory_timestamp
from ._schema_update_ops import SchemaMemoryUpdate
from ._schema_utils import (
    dedupe_non_empty,
    format_current_higher_order,
    format_first_order_memories,
    format_higher_order_schema,
    format_new_properties,
    memory_to_evidence,
    new_properties_for_higher_order,
    parse_json_object,
)

logger = get_logger(__name__)

EmbedTexts = Callable[[str, list[str]], Awaitable[list[list[float]]]]
MemoryFactory = Callable[..., MemoryWrite]


class ListEntityMemories(Protocol):
    """Callable boundary for listing stored memories belonging to an entity."""

    def __call__(
        self,
        entity_id: str,
        *,
        context: MemoryRequestContext,
        limit: int,
    ) -> Awaitable[list[MemoryView]]: ...


@dataclass(slots=True)
class SchemaHigherOrderGenerator:
    """Generate higher-order property memories from first-order evidence."""

    llm_client: LLMClient
    db_reader: Any
    entity_manager: Any
    prompt_set: AddPromptSet
    embed_texts: EmbedTexts
    list_entity_memories: ListEntityMemories
    memory_factory: MemoryFactory
    enabled: bool
    top_k: int
    min_evidence_count: int

    async def generate(
        self,
        *,
        entity_write: EntityWrite,
        raw_entity: dict[str, Any],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
        prompt_set: AddPromptSet | None = None,
    ) -> tuple[list[MemoryWrite], list[str], list[SchemaMemoryUpdate]]:
        """Generate higher-order memories and archive ids for an updated entity."""

        if (
            not self.enabled
            or entity_write.entity_type == "episodes"
            or entity_write.metadata.get("merge_action") != "update"
            or not entity_write.entity_type
            or not self.entity_manager.has_higher_order_properties(entity_write.entity_type)
        ):
            return [], [], []

        new_properties = new_properties_for_higher_order(raw_entity)
        if not new_properties:
            return [], [], []

        higher_order_names = self.entity_manager.get_higher_order_property_names(entity_write.entity_type)
        query_text = " | ".join(str(prop.get("value") or "")[:200] for prop in new_properties if prop.get("value"))
        if not query_text:
            return [], [], []

        first_order_memories = await self._retrieve_first_order_properties(
            entity_id=entity_write.entity_id,
            query_text=query_text,
            higher_order_names=higher_order_names,
            context=context,
        )
        current_higher_order = await self._current_higher_order(
            entity_id=entity_write.entity_id,
            higher_order_names=higher_order_names,
            context=context,
        )

        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.higher_order_generation.replace("{entity_name}", entity_write.entity_name)
            .replace("{entity_type}", entity_write.entity_type or "")
            .replace("{entity_description}", entity_write.description or "")
            .replace("{current_higher_order}", format_current_higher_order(current_higher_order))
            .replace("{first_order_memories}", format_first_order_memories(first_order_memories))
            .replace("{new_properties}", format_new_properties(new_properties))
            .replace(
                "{higher_order_schema}",
                format_higher_order_schema(self.entity_manager.get_properties_by_order(entity_write.entity_type, 2)),
            )
            .replace("{min_evidence_count}", str(self.min_evidence_count))
        )
        try:
            response = await self.llm_client.chat(
                task="memory.add.higher_order_generation",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            result = response.parsed if isinstance(response.parsed, dict) else {}
        except Exception:
            logger.warning("higher order generation failed", exc_info=True)
            return [], [], []

        return self._build_updates(
            entity_write=entity_write,
            result=result,
            higher_order_names=higher_order_names,
            current_higher_order=current_higher_order,
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
        )

    async def _retrieve_first_order_properties(
        self,
        *,
        entity_id: str,
        query_text: str,
        higher_order_names: set[str],
        context: MemoryRequestContext,
    ) -> list[dict[str, Any]]:
        try:
            vectors = await self.embed_texts("memory.add.higher_order", [query_text])
            result = await self.db_reader.search_entity_property_memories(
                context,
                query_vector=vectors[0] if vectors else [],
                entity_id=entity_id,
                limit=self.top_k,
            )
            memories = [
                hit.memory
                for hit in result.hits
                if hit.memory and hit.memory.status == "active" and hit.memory.property_name not in higher_order_names
            ]
            if memories:
                return [memory_to_evidence(memory) for memory in memories[: self.top_k]]
        except Exception:
            logger.warning("higher order evidence vector search failed; using filter fallback", exc_info=True)

        memories = await self.list_entity_memories(entity_id, context=context, limit=100)
        first_order = [
            memory
            for memory in memories
            if memory.status == "active" and memory.property_name not in higher_order_names
        ]
        first_order.sort(key=lambda memory: memory.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return [memory_to_evidence(memory) for memory in first_order[: self.top_k]]

    async def _current_higher_order(
        self,
        *,
        entity_id: str,
        higher_order_names: set[str],
        context: MemoryRequestContext,
    ) -> dict[str, list[dict[str, Any]]]:
        memories = await self.list_entity_memories(entity_id, context=context, limit=200)
        entity = TemporalEntity(entity_id=entity_id, entity_manager=self.entity_manager)
        for memory in memories:
            if memory.status == "active" and memory.property_name in higher_order_names:
                entity.modify_property(
                    memory.property_name or "",
                    memory.content,
                    memory_timestamp(memory),
                    uid=memory.memory_id,
                )

        current: dict[str, list[dict[str, Any]]] = {}
        for prop_name in higher_order_names:
            history = entity.get_property_history(prop_name, include_uid=True)[-5:]
            if history:
                current[prop_name] = [
                    {"timestamp": timestamp, "value": value, "uid": uid} for timestamp, value, uid in history
                ]
        return current

    def _build_updates(
        self,
        *,
        entity_write: EntityWrite,
        result: dict[str, Any],
        higher_order_names: set[str],
        current_higher_order: dict[str, list[dict[str, Any]]],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> tuple[list[MemoryWrite], list[str], list[SchemaMemoryUpdate]]:
        memories: list[MemoryWrite] = []
        archive_ids: list[str] = []
        updates: list[SchemaMemoryUpdate] = []
        for item in result.get("updates", []):
            if not isinstance(item, dict):
                continue
            prop_name = str(item.get("property_name") or "")
            action = item.get("action", "no_action")
            value = str(item.get("value") or "")
            if prop_name not in higher_order_names or action not in {"update", "add", "create"} or not value:
                continue

            latest = (current_higher_order.get(prop_name) or [])[-1] if current_higher_order.get(prop_name) else None
            if action == "update" and latest:
                if value.strip() == str(latest.get("value") or "").strip():
                    continue
            target_uid = str(latest["uid"]) if action == "update" and latest and latest.get("uid") else None

            memory = self.memory_factory(
                entity_write=entity_write,
                prop={
                    "property_name": prop_name,
                    "value": value,
                    "time": created_at.strftime("%Y-%m-%d"),
                    "operation": "set",
                },
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
                memory_id=target_uid,
                extra_metadata={
                    "higher_order": True,
                    "higher_order_action": "add" if action == "create" else action,
                    "higher_order_reasoning": item.get("reasoning"),
                },
            )
            if target_uid:
                updates.append(
                    SchemaMemoryUpdate(
                        target_memory_id=target_uid,
                        memory=memory,
                        reason="schema_add_higher_order_update",
                    )
                )
            else:
                memories.append(memory)
        return memories, dedupe_non_empty(archive_ids), updates
