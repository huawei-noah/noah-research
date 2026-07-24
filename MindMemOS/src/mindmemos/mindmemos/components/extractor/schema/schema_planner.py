"""Write-plan planner for schema add extraction results."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ....config import get_config
from ....llm import EmbedClient, LLMClient
from ....logging import get_logger
from ....prompts import AddPromptSet
from ....typing import (
    EntityView,
    EntityWrite,
    FieldCondition,
    GraphRelationship,
    MemoryAddEventItem,
    MemoryDbDeleteCommand,
    MemoryDbMutationPlan,
    MemoryDbMutationResult,
    MemoryDbUpdateCommand,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryView,
    MemoryWrite,
    SearchFilter,
)
from ...memory_modeling.schema import Edge, memory_timestamp
from ...text import SparseVectorEncoder, get_text_preprocessor
from ._schema_higher_order import SchemaHigherOrderGenerator
from ._schema_merge_policy import SchemaMergePolicy
from ._schema_property_similarity import SchemaPropertySimilaritySearcher
from ._schema_update_ops import SchemaMemoryUpdate
from ._schema_utils import (
    base_metadata,
    dedupe_entity_relationships,
    dedupe_non_empty,
    edge_relationships,
    entity_embedding_text,
    exact_candidate,
    format_candidate_episodes,
    format_property_delete_context,
    fuzzy_match_candidate,
    merge_description,
    parse_json_object,
    property_relationships,
    resolve_duplicate_name,
    schema_memory_type,
)
from ._schema_write_plan import SchemaWritePlanBuilder

logger = get_logger(__name__)


class SchemaAddPlanner:
    """Build write plans for schema add extracted entities."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embed_client: EmbedClient,
        db_reader: Any,
        db_writer: Any,
        entity_manager: Any,
        prompt_set: AddPromptSet,
        enable_entity_merge_decision: bool,
        entity_recall_top_k: int,
        max_merge_retries: int,
        use_property_merge: bool,
        secondary_search_limit: int,
        secondary_search_retries: int,
        higher_order_enabled: bool,
        higher_order_top_k: int,
        higher_order_min_evidence_count: int,
        episode_edge_top_k: int,
    ) -> None:
        self.llm_client = llm_client
        self.embed_client = embed_client
        self.db_reader = db_reader
        self.db_writer = db_writer
        self.entity_manager = entity_manager
        self.prompt_set = prompt_set
        self.enable_entity_merge_decision = enable_entity_merge_decision
        self.entity_recall_top_k = entity_recall_top_k
        self.max_merge_retries = max_merge_retries
        self.use_property_merge = use_property_merge
        self.secondary_search_limit = secondary_search_limit
        self.secondary_search_retries = secondary_search_retries
        self.higher_order_enabled = higher_order_enabled
        self.higher_order_top_k = higher_order_top_k
        self.higher_order_min_evidence_count = higher_order_min_evidence_count
        self.episode_edge_top_k = episode_edge_top_k
        text_cfg = get_config().algo_config.text_processing
        self._text_preprocessor = get_text_preprocessor()
        self._sparse_encoder = SparseVectorEncoder(text_cfg)
        self._merge_policy = SchemaMergePolicy()
        self._property_similarity = SchemaPropertySimilaritySearcher(
            db_reader=self.db_reader,
            embed_texts=self._embed_texts,
        )
        self._higher_order_generator = SchemaHigherOrderGenerator(
            llm_client=self.llm_client,
            db_reader=self.db_reader,
            entity_manager=self.entity_manager,
            prompt_set=self.prompt_set,
            embed_texts=self._embed_texts,
            list_entity_memories=self._list_entity_memories,
            memory_factory=self._memory_from_property,
            enabled=self.higher_order_enabled,
            top_k=self.higher_order_top_k,
            min_evidence_count=self.higher_order_min_evidence_count,
        )
        self._write_plan_builder = SchemaWritePlanBuilder(
            text_preprocessor=self._text_preprocessor,
            sparse_encoder=self._sparse_encoder,
            embed_texts=self._embed_texts,
        )

    async def build_write_plan(
        self,
        *,
        raw_entities: list[dict[str, Any]],
        raw_edges: list[dict[str, Any]],
        episode_entity: dict[str, Any],
        context: MemoryRequestContext,
        request_metadata: dict[str, Any],
        created_at: datetime,
        prompt_set: AddPromptSet | None = None,
    ) -> tuple[MemoryDbWritePlan, list[MemoryAddEventItem], list[str], list[SchemaMemoryUpdate]]:
        """Build a complete database write plan from extracted schema entities."""
        prompts = prompt_set or self.prompt_set
        merge_context = await self._merge_policy.prepare(
            raw_entities=raw_entities,
            raw_edges=raw_edges,
            episode_entity=episode_entity,
        )
        memories: list[MemoryWrite] = []
        entities: list[EntityWrite] = []
        relationships: list[GraphRelationship] = []
        events: list[MemoryAddEventItem] = []
        entity_by_name: dict[str, EntityWrite] = {}
        pending_archives: list[str] = []
        pending_updates: list[SchemaMemoryUpdate] = []

        extraction_entities = list(merge_context.raw_entities) + [merge_context.episode_entity]
        entity_embedding_texts = [entity_embedding_text(entity) for entity in extraction_entities]
        entity_vectors = await self._embed_texts("memory.add.entity", entity_embedding_texts)
        entity_vector_by_name = {
            entity.get("name", ""): vector for entity, vector in zip(extraction_entities, entity_vectors, strict=True)
        }

        resolve_tasks = [
            self._resolve_entity_write(
                entity,
                context=context,
                created_at=created_at,
                query_vector=entity_vector_by_name.get(entity.get("name", "")) or [],
                request_metadata=request_metadata,
                prompt_set=prompts,
            )
            for entity in merge_context.raw_entities
        ]
        entity_write_results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
        for entity, result in zip(merge_context.raw_entities, entity_write_results, strict=True):
            if isinstance(result, Exception):
                logger.error("entity_resolve_failed", entity_name=entity.get("name"), error=str(result))
                continue
            entities.append(result)
            entity_by_name[entity.get("name", "")] = result

        episode_write = self._new_entity_write(
            merge_context.episode_entity,
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
        )
        entities.append(episode_write)
        entity_by_name[merge_context.episode_entity["name"]] = episode_write
        merge_context.entity_by_name = dict(entity_by_name)

        episode_edge_task = asyncio.create_task(
            self._episode_edge_relationships(
                episode_write=episode_write,
                query_vector=entity_vector_by_name.get(merge_context.episode_entity.get("name", "")) or [],
                context=context,
                prompt_set=prompts,
            )
        )

        prop_tasks = []
        prop_entity_writes = []
        for entity in extraction_entities:
            entity_write = entity_by_name.get(entity.get("name", ""))
            if entity_write is None:
                continue
            prop_tasks.append(
                self._process_entity_properties(
                    entity=entity,
                    entity_write=entity_write,
                    context=context,
                    created_at=created_at,
                    request_metadata=request_metadata,
                    prompt_set=prompts,
                )
            )
            prop_entity_writes.append(entity_write)

        prop_results = await asyncio.gather(*prop_tasks, return_exceptions=True)
        for entity_write, result in zip(prop_entity_writes, prop_results, strict=True):
            if isinstance(result, Exception):
                logger.error("property_processing_failed", entity_name=entity_write.entity_name, error=str(result))
                continue
            entity_memories, archive_ids, higher_memories, higher_archives, update_ops = result
            pending_archives.extend(archive_ids)
            pending_archives.extend(higher_archives)
            pending_updates.extend(update_ops)
            all_prop_memories: list[MemoryWrite] = []
            for memory in entity_memories:
                memories.append(memory)
                all_prop_memories.append(memory)
                relationships.extend(property_relationships(context.project_id, entity_write.entity_id, memory))
            for memory in higher_memories:
                memories.append(memory)
                all_prop_memories.append(memory)
                relationships.extend(property_relationships(context.project_id, entity_write.entity_id, memory))
            if all_prop_memories:
                events.append(
                    MemoryAddEventItem(
                        operation="add",
                        content=_format_entity_add_content(entity_write, all_prop_memories),
                        memory_id=entity_write.entity_id,
                        mem_type=schema_memory_type(entity_write.entity_type),
                        related_memory_ids=[m.memory_id for m in all_prop_memories],
                    )
                )

        relationships.extend(edge_relationships(merge_context.raw_edges, entity_by_name, context.project_id))
        episode_edge_relationships = await episode_edge_task
        relationships.extend(episode_edge_relationships)
        relationships = dedupe_entity_relationships(relationships)
        merge_context.pending_archives = list(pending_archives)
        plan = await self._write_plan_builder.build(
            memories=memories,
            entities=entities,
            relationships=relationships,
            project_id=context.project_id,
            entity_context_memories=memories + [update.memory for update in pending_updates],
        )
        return (
            plan,
            events,
            pending_archives,
            pending_updates,
        )

    async def _process_entity_properties(
        self,
        *,
        entity: dict[str, Any],
        entity_write: EntityWrite,
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
        prompt_set: AddPromptSet | None = None,
    ) -> tuple[list[MemoryWrite], list[str], list[MemoryWrite], list[str], list[SchemaMemoryUpdate]]:
        entity_memories, archive_ids, entity_updates = await self._apply_property_operations(
            entity_write=entity_write,
            properties=entity.get("properties", []),
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
            prompt_set=prompt_set,
        )
        higher_memories, higher_archives, higher_updates = await self._higher_order_generator.generate(
            entity_write=entity_write,
            raw_entity=entity,
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
            prompt_set=prompt_set,
        )
        return entity_memories, archive_ids, higher_memories, higher_archives, entity_updates + higher_updates

    async def _resolve_entity_write(
        self,
        entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        created_at: datetime,
        query_vector: list[float],
        request_metadata: dict[str, Any],
        prompt_set: AddPromptSet | None = None,
    ) -> EntityWrite:
        """Resolve whether an extracted entity should create or update an entity record."""
        candidates = await self._recall_entity_candidates(entity, context=context, query_vector=query_vector)
        name2id: dict[str, EntityView] = {}
        for candidate in candidates:
            if candidate.entity_name not in name2id:
                name2id[candidate.entity_name] = candidate

        exact = exact_candidate(entity, candidates)
        if exact is not None:
            return await self._entity_write_from_view(
                exact, entity, context=context, created_at=created_at, prompt_set=prompt_set
            )

        decision = await self._llm_single_entity_decision(entity, candidates, name2id, prompt_set=prompt_set)
        action = decision.get("action", "create")

        if action == "update":
            target_name = decision.get("target_entity", "")
            return await self._execute_update(
                entity,
                target_name=target_name,
                name2id=name2id,
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
                query_vector=query_vector,
                prompt_set=prompt_set,
            )

        return await self._execute_create(
            entity,
            name2id=name2id,
            context=context,
            created_at=created_at,
            request_metadata=request_metadata,
            query_vector=query_vector,
            skip_duplicate_check=False,
            prompt_set=prompt_set,
        )

    async def _llm_single_entity_decision(
        self,
        entity: dict[str, Any],
        candidates: list[EntityView],
        name2id: dict[str, EntityView],
        *,
        prompt_set: AddPromptSet | None = None,
    ) -> dict[str, Any]:
        """Ask the LLM whether an extracted entity should create or update a candidate."""
        if not candidates or not self.enable_entity_merge_decision:
            return {"action": "create", "relation_candidates": []}

        existing_text = "\n".join(
            f"- name: {c.entity_name}, entity_type: {c.entity_type}, Description: {c.description or ''}"
            for c in candidates
            if c.entity_type != "episodes"
        )
        if not existing_text.strip():
            existing_text = "No existing entities."

        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.single_entity_merge.replace("{entity_name}", str(entity.get("name", "")))
            .replace("{entity_type}", str(entity.get("entity_type", "")))
            .replace("{entity_description}", str(entity.get("description", ""))[:200])
            .replace("{existing_entities}", existing_text)
        )

        for attempt in range(self.max_merge_retries):
            try:
                response = await self.llm_client.chat(
                    task="memory.add.entity_merge",
                    messages=[{"role": "user", "content": prompt}],
                    format_parser=parse_json_object,
                )
                decision = response.parsed if isinstance(response.parsed, dict) else {}
                if "action" not in decision:
                    prompt += f"\nPrevious answer: {decision}. ERROR: Must be a JSON object with 'action' field."
                    continue

                action = decision.get("action")
                if action == "update":
                    target = str(decision.get("target_entity", ""))
                    if target in name2id:
                        return decision
                    matched = fuzzy_match_candidate(target, name2id)
                    if matched:
                        logger.info("UPDATE target fuzzy matched", target=target, matched=matched)
                        decision["target_entity"] = matched
                        return decision
                    logger.warning("UPDATE target not in candidates, retrying", target=target)
                    prompt += (
                        f"\nPrevious answer: {json.dumps(decision, ensure_ascii=False)}. "
                        f"ERROR: target_entity '{target}' not in candidate list. "
                        f"Available: {list(name2id.keys())}. Please fix."
                    )
                    continue
                elif action == "create":
                    return decision
                else:
                    prompt += f"\nPrevious answer invalid. action must be 'create' or 'update'."
                    continue
            except Exception:
                logger.error("entity merge decision failed", attempt=attempt + 1, exc_info=True)

        entity_name = str(entity.get("name", ""))
        for c in candidates:
            if c.entity_name == entity_name and c.entity_name in name2id:
                logger.warning(
                    "all merge retries exhausted, defaulting to UPDATE for same-name candidate", entity_name=entity_name
                )
                return {"action": "update", "target_entity": entity_name, "relation_candidates": []}

        logger.error("all merge retries exhausted, defaulting to CREATE", entity_name=entity_name)
        return {"action": "create", "relation_candidates": []}

    async def _execute_create(
        self,
        entity: dict[str, Any],
        *,
        name2id: dict[str, EntityView],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
        query_vector: list[float],
        skip_duplicate_check: bool,
        prompt_set: AddPromptSet | None = None,
    ) -> EntityWrite:
        """Create an entity write, with duplicate-name handling."""
        entity_name = str(entity.get("name", ""))
        entity_type = entity.get("entity_type", "")

        if not skip_duplicate_check:
            if entity_name in name2id and name2id[entity_name] is not None:
                existing = name2id[entity_name]
                resolution = resolve_duplicate_name(entity, existing)
                if resolution["action"] == "merge":
                    logger.info(
                        "duplicate name in candidates, converting to UPDATE",
                        entity_name=entity_name,
                        target_id=existing.entity_id,
                    )
                    return await self._entity_write_from_view(
                        existing, entity, context=context, created_at=created_at, prompt_set=prompt_set
                    )
                else:
                    new_name = resolution["new_name"]
                    logger.info("duplicate name in candidates, renaming", old=entity_name, new=new_name)
                    entity["name"] = new_name

            existing_view = await self._find_entity_by_name(
                entity_name,
                context=context,
                query_vector=query_vector,
                entity_type=entity_type,
            )
            if existing_view is not None:
                resolution = resolve_duplicate_name(entity, existing_view)
                if resolution["action"] == "merge":
                    logger.info(
                        "duplicate name in DB, converting to UPDATE",
                        entity_name=entity_name,
                        target_id=existing_view.entity_id,
                    )
                    return await self._entity_write_from_view(
                        existing_view, entity, context=context, created_at=created_at, prompt_set=prompt_set
                    )
                else:
                    new_name = resolution["new_name"]
                    logger.info("duplicate name in DB, renaming", old=entity_name, new=new_name)
                    entity["name"] = new_name

        return self._new_entity_write(entity, context=context, created_at=created_at, request_metadata=request_metadata)

    async def _execute_update(
        self,
        entity: dict[str, Any],
        *,
        target_name: str,
        name2id: dict[str, EntityView],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
        query_vector: list[float],
        prompt_set: AddPromptSet | None = None,
    ) -> EntityWrite:
        """Update an existing entity, falling back to create when no target is found."""
        target_view: EntityView | None = None

        if target_name in name2id:
            target_view = name2id[target_name]
            logger.info(
                "UPDATE target matched from candidates", target_name=target_name, target_id=target_view.entity_id
            )

        if target_view is None:
            target_view = await self._find_entity_by_name(
                target_name,
                context=context,
                query_vector=query_vector,
            )

        if target_view is None:
            logger.warning("UPDATE target not found, converting to CREATE", target_name=target_name)
            return await self._execute_create(
                entity,
                name2id=name2id,
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
                query_vector=query_vector,
                skip_duplicate_check=True,
                prompt_set=prompt_set,
            )

        return await self._entity_write_with_description_update(
            target_view,
            entity,
            context=context,
            created_at=created_at,
            prompt_set=prompt_set,
        )

    async def _find_entity_by_name(
        self,
        name: str,
        *,
        context: MemoryRequestContext,
        query_vector: list[float],
        entity_type: str | None = None,
    ) -> EntityView | None:
        """Find an entity by exact name through dense recall plus name filtering."""
        if not query_vector:
            return None
        for attempt in range(self.secondary_search_retries):
            try:
                result = await self.db_reader.search_entities_dense(
                    context,
                    query=f"name: {name}",
                    query_vector=query_vector,
                    limit=self.secondary_search_limit,
                )
                for hit in result.hits:
                    if hit.entity and hit.entity.entity_name == name:
                        if entity_type is None or hit.entity.entity_type == entity_type:
                            return hit.entity
                if attempt < self.secondary_search_retries - 1:
                    await asyncio.sleep(0.2)
            except Exception:
                logger.warning("entity name search failed", attempt=attempt + 1, exc_info=True)
        return None

    async def _entity_write_with_description_update(
        self,
        target: EntityView,
        new_entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        created_at: datetime,
        prompt_set: AddPromptSet | None = None,
    ) -> EntityWrite:
        """Build an update EntityWrite and merge the description with the LLM."""
        description = await self._llm_description_update(
            entity_name=target.entity_name,
            entity_type=target.entity_type or "",
            current_description=target.description or "",
            properties=new_entity.get("properties", []),
            prompt_set=prompt_set,
        )
        metadata = dict(target.metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add_v1",
                "merge_action": "update",
                "latest_raw_entity_name": new_entity.get("name"),
                "record_time": new_entity.get("record_time"),
            }
        )
        return EntityWrite(
            entity_id=target.entity_id,
            account_id=target.account_id or context.account_id,
            project_id=context.project_id,
            api_key_uuid=target.api_key_uuid or context.api_key_uuid,
            user_id=target.user_id or context.user_id,
            app_id=target.app_id or context.app_id,
            session_id=target.session_id or context.session_id,
            agent_id=target.agent_id or context.agent_id,
            request_id=context.request_id,
            entity_name=target.entity_name,
            entity_type=target.entity_type or new_entity.get("entity_type"),
            description=description,
            schema_version=self.entity_manager.file_path.name,
            metadata=metadata,
            created_at=target.created_at or created_at,
            update_at=created_at,
        )

    async def _llm_description_update(
        self,
        *,
        entity_name: str,
        entity_type: str,
        current_description: str,
        properties: list[dict[str, Any]],
        prompt_set: AddPromptSet | None = None,
    ) -> str:
        """Ask the LLM to merge new properties into an existing entity description."""
        props_lines = []
        for prop in properties:
            prop_name = prop.get("property_name", "")
            value = prop.get("value", "")
            if value:
                props_lines.append(f"- {prop_name}: {value}")
        latest_props_text = "\n".join(props_lines) if props_lines else "No properties."

        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.des_update.replace("{entity_name}", entity_name)
            .replace("{entity_type}", entity_type)
            .replace("{current_description}", current_description)
            .replace("{latest_properties}", latest_props_text)
        )
        try:
            response = await self.llm_client.chat(
                task="memory.add.description_update",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content.strip()
            if "<description>" in content:
                content = content.split("<description>")[1]
            if "</description>" in content:
                content = content.split("</description>")[0]
            return content[:2000]
        except Exception:
            logger.warning("description update LLM failed; falling back to concatenation", exc_info=True)
            return (
                merge_description(current_description, " ".join(str(p.get("value") or "") for p in properties))
                or current_description
            )

    async def _apply_property_operations(
        self,
        *,
        entity_write: EntityWrite,
        properties: list[dict[str, Any]],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
        prompt_set: AddPromptSet | None = None,
    ) -> tuple[list[MemoryWrite], list[str], list[SchemaMemoryUpdate]]:
        """Convert extracted properties into memory writes and archive operations."""
        if not self.use_property_merge or not properties:
            return (
                [
                    self._memory_from_property(
                        entity_write=entity_write,
                        prop=prop,
                        context=context,
                        created_at=created_at,
                        request_metadata=request_metadata,
                    )
                    for prop in properties
                    if prop.get("operation") != "delete"
                ],
                [],
                [],
            )

        set_props = [prop for prop in properties if prop.get("operation") != "delete"]
        delete_archives = await self._property_delete_archives(
            entity_write=entity_write,
            properties=[prop for prop in properties if prop.get("operation") == "delete"],
            context=context,
            prompt_set=prompt_set,
        )
        if not set_props:
            return [], delete_archives, []

        similarity_results = await self._property_similarity.find_for_merge(
            context=context,
            entity_id=entity_write.entity_id,
            properties=set_props,
            limit=5,
        )

        existing_map: dict[str, dict[str, Any]] = {}
        new_with_similar: list[dict[str, Any]] = []
        direct_props: list[dict[str, Any]] = []
        p_counter = 0

        for result in similarity_results:
            prop = result.property
            similar = [match.memory for match in result.matches]
            if not similar:
                direct_props.append(prop)
            else:
                for memory in similar:
                    if memory.memory_id not in existing_map:
                        p_counter += 1
                        existing_map[memory.memory_id] = {
                            "p_id": f"p{p_counter}",
                            "memory_id": memory.memory_id,
                            "property_name": memory.property_name or "",
                            "timestamp": memory_timestamp(memory),
                            "value": memory.content,
                            "memory": memory,
                        }
                new_with_similar.append(prop)

        if not new_with_similar:
            return (
                [
                    self._memory_from_property(
                        entity_write=entity_write,
                        prop=prop,
                        context=context,
                        created_at=created_at,
                        request_metadata=request_metadata,
                    )
                    for prop in direct_props
                    if prop.get("operation") != "delete"
                ],
                delete_archives,
                [],
            )

        existing_list = sorted(existing_map.values(), key=lambda x: x["p_id"])
        existing_text = "".join(
            f'{e["p_id"]}: [{e["property_name"]}] time={e["timestamp"]}, value="{e["value"]}"\n' for e in existing_list
        )
        n_items: list[dict[str, Any]] = []
        new_text = ""
        for i, prop in enumerate(new_with_similar):
            n_id = f"n{i + 1}"
            new_text += f'{n_id}: [{prop.get("property_name", "")}] time={prop.get("time", "")}, value="{prop.get("value", "")}"\n'
            n_items.append({"n_id": n_id, "prop": prop})

        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.property_merge_decision.replace("{entity_name}", entity_write.entity_name)
            .replace("{entity_type}", entity_write.entity_type or "")
            .replace("{existing_properties}", existing_text)
            .replace("{new_properties}", new_text)
        )

        try:
            response = await self.llm_client.chat(
                task="memory.add.property_merge",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            decision = response.parsed if isinstance(response.parsed, dict) else {"existing": [], "new": []}
        except Exception:
            logger.warning("property merge LLM failed, appending all", exc_info=True)
            all_props = direct_props + [item["prop"] for item in n_items]
            return (
                [
                    self._memory_from_property(
                        entity_write=entity_write,
                        prop=prop,
                        context=context,
                        created_at=created_at,
                        request_metadata=request_metadata,
                    )
                    for prop in all_props
                    if prop.get("operation") != "delete"
                ],
                delete_archives,
                [],
            )

        final_memories = [
            self._memory_from_property(
                entity_write=entity_write,
                prop=prop,
                context=context,
                created_at=created_at,
                request_metadata=request_metadata,
            )
            for prop in direct_props
            if prop.get("operation") != "delete"
        ]
        archive_ids = list(delete_archives)
        update_ops: list[SchemaMemoryUpdate] = []
        used_update_targets: set[str] = set()
        p_id_to_info = {e["p_id"]: e for e in existing_list}
        n_id_to_item = {item["n_id"]: item for item in n_items}
        deleted_n_ids: set[str] = set()
        updated_n_ids: set[str] = set()

        def queue_update(target_info: dict[str, Any], memory: MemoryWrite, *, reason: str) -> bool:
            target_memory_id = str(target_info["memory_id"])
            if target_memory_id in used_update_targets:
                logger.warning(
                    "property_merge_update_target_already_used",
                    target_memory_id=target_memory_id,
                    entity_id=entity_write.entity_id,
                )
                return False
            used_update_targets.add(target_memory_id)
            update_ops.append(SchemaMemoryUpdate(target_memory_id=target_memory_id, memory=memory, reason=reason))
            return True

        for existing_decision in decision.get("existing", []):
            p_info = p_id_to_info.get(str(existing_decision.get("id", "")))
            if not p_info:
                continue
            op = existing_decision.get("op")
            if op == "delete":
                archive_ids.append(p_info["memory_id"])
            elif op == "update":
                merged_value = str(existing_decision.get("value") or p_info["value"])
                memory = self._memory_from_property(
                    entity_write=entity_write,
                    prop={
                        "property_name": p_info["property_name"],
                        "value": merged_value,
                        "time": p_info["timestamp"],
                        "operation": "update",
                    },
                    context=context,
                    created_at=created_at,
                    request_metadata=request_metadata,
                    memory_id=str(p_info["memory_id"]),
                    extra_metadata={
                        "property_merge_action": "update",
                        "merged_from_memory_ids": [p_info["memory_id"]],
                    },
                )
                queue_update(p_info, memory, reason="schema_add_property_merge")

        for new_decision in decision.get("new", []):
            n_id = str(new_decision.get("id", ""))
            item = n_id_to_item.get(n_id)
            if not item:
                continue
            op = new_decision.get("op")
            if op == "delete":
                deleted_n_ids.add(n_id)
            elif op == "update":
                target_p_id = str(new_decision.get("target", ""))
                p_info = p_id_to_info.get(target_p_id)
                merged_value = str(new_decision.get("value") or "")
                if not p_info or not merged_value:
                    continue
                updated_n_ids.add(n_id)
                memory = self._memory_from_property(
                    entity_write=entity_write,
                    prop={
                        "property_name": p_info["property_name"],
                        "value": merged_value,
                        "time": p_info["timestamp"],
                        "operation": "update",
                    },
                    context=context,
                    created_at=created_at,
                    request_metadata=request_metadata,
                    memory_id=str(p_info["memory_id"]),
                    extra_metadata={
                        "property_merge_action": "update",
                        "merged_from_memory_ids": [p_info["memory_id"]],
                    },
                )
                queue_update(p_info, memory, reason="schema_add_property_merge")

        for item in n_items:
            if item["n_id"] in deleted_n_ids or item["n_id"] in updated_n_ids:
                continue
            final_memories.append(
                self._memory_from_property(
                    entity_write=entity_write,
                    prop=item["prop"],
                    context=context,
                    created_at=created_at,
                    request_metadata=request_metadata,
                )
            )

        return final_memories, dedupe_non_empty(archive_ids), update_ops

    async def _property_delete_archives(
        self,
        *,
        entity_write: EntityWrite,
        properties: list[dict[str, Any]],
        context: MemoryRequestContext,
        prompt_set: AddPromptSet | None = None,
    ) -> list[str]:
        """Plan archives for extracted delete operations."""
        if not properties:
            return []

        delete_context: list[dict[str, Any]] = []
        memory_by_key: dict[tuple[str, str], str] = {}
        for prop in properties:
            value = str(prop.get("value") or prop.get("property_name") or "")
            if not value:
                continue
            similar_matches = await self._property_similarity.find_delete_candidates(
                context=context,
                entity_id=entity_write.entity_id,
                prop=prop,
                limit=5,
            )

            similar_history = []
            for match in similar_matches:
                memory = match.memory
                timestamp = memory_timestamp(memory)
                similar_history.append(
                    {
                        "property_name": memory.property_name or "",
                        "timestamp": timestamp,
                        "value": memory.content,
                        "similarity": match.score,
                    }
                )
                memory_by_key[(memory.property_name or "", timestamp)] = memory.memory_id
            if similar_history:
                delete_context.append(
                    {
                        "property_name": prop.get("property_name") or "",
                        "new_value": value,
                        "similar_history": similar_history,
                    }
                )

        if not delete_context:
            return []

        context_text = format_property_delete_context(delete_context)
        prompts = prompt_set or self.prompt_set
        prompt = prompts.property_delete_decision.replace("{entity_name}", entity_write.entity_name).replace(
            "{context_text}", context_text
        )
        try:
            response = await self.llm_client.chat(
                task="memory.add.property_delete",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            decisions = response.parsed if isinstance(response.parsed, list) else []
        except Exception:
            logger.warning("property delete LLM failed; keeping existing memories", exc_info=True)
            return []

        archive_ids: list[str] = []
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            memory_id = memory_by_key.get(
                (str(decision.get("property_name") or ""), str(decision.get("timestamp") or ""))
            )
            if memory_id:
                archive_ids.append(memory_id)
        return dedupe_non_empty(archive_ids)

    async def _recall_entity_candidates(
        self,
        entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        query_vector: list[float],
    ) -> list[EntityView]:
        """Recall existing entities that are semantically close to a new entity."""
        if not query_vector:
            return []
        filters = SearchFilter(must_not=[FieldCondition(field="entity_type", op="match", value="episodes")])
        result = await self.db_reader.search_entities_dense(
            context,
            query=entity_embedding_text(entity),
            query_vector=query_vector,
            filters=filters,
            limit=self.entity_recall_top_k,
        )
        return [hit.entity for hit in result.hits if hit.entity is not None]

    async def _list_entity_memories(
        self,
        entity_id: str,
        *,
        context: MemoryRequestContext,
        limit: int,
    ) -> list[MemoryView]:
        try:
            memories, _ = await self.db_reader.list_memories(
                context,
                filters=SearchFilter(must=[FieldCondition(field="entity_id", op="match", value=entity_id)]),
                limit=limit,
            )
            return memories
        except Exception:
            logger.warning("entity memory list fallback failed", exc_info=True)
            return []

    async def _episode_edge_relationships(
        self,
        *,
        episode_write: EntityWrite,
        query_vector: list[float],
        context: MemoryRequestContext,
        prompt_set: AddPromptSet | None = None,
    ) -> list[GraphRelationship]:
        """Build semantic relationships between a new episode and historical episodes."""
        if not query_vector or self.episode_edge_top_k <= 0:
            return []
        try:
            result = await self.db_reader.search_entities_dense(
                context,
                query=episode_write.description or episode_write.entity_name,
                query_vector=query_vector,
                filters=SearchFilter(must=[FieldCondition(field="entity_type", op="match", value="episodes")]),
                limit=self.episode_edge_top_k,
            )
            candidates = [
                hit.entity
                for hit in result.hits
                if hit.entity
                and hit.entity.entity_id != episode_write.entity_id
                and hit.entity.entity_type == "episodes"
            ]
        except Exception:
            logger.warning("episode edge candidate search failed", exc_info=True)
            return []

        if not candidates:
            return []

        prompts = prompt_set or self.prompt_set
        prompt = (
            prompts.episode_edge.replace("{new_episode_name}", episode_write.entity_name)
            .replace("{new_episode_description}", episode_write.description or "")
            .replace("{candidate_episodes}", format_candidate_episodes(candidates))
        )
        try:
            response = await self.llm_client.chat(
                task="memory.add.episode_edge",
                messages=[{"role": "user", "content": prompt}],
                format_parser=parse_json_object,
            )
            edges = response.parsed if isinstance(response.parsed, list) else []
        except Exception:
            logger.warning("episode edge LLM failed; no episode edges created", exc_info=True)
            return []

        candidate_by_id = {candidate.entity_id: candidate for candidate in candidates}
        relationships: list[GraphRelationship] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            target_id = str(edge.get("target_episode_id") or "")
            target = candidate_by_id.get(target_id)
            if target is None:
                continue
            relation = str(edge.get("relation") or "related_to")
            relationship = Edge.from_entity_dtos(
                episode_write,
                target,
                description=relation,
            ).to_graph_relationship(project_id=context.project_id)
            relationship.metadata["edge_source"] = "episode_edge_prompt"
            relationships.append(relationship)
        return relationships

    async def archive_memories(self, context: MemoryRequestContext, memory_ids: list[str], *, consistency: str) -> None:
        """Soft-delete the specified memory records."""
        commands = self.build_archive_memory_commands(memory_ids, consistency=consistency)
        if not commands:
            return

        try:
            result = await self.db_writer.apply_mutation_plan(
                context,
                MemoryDbMutationPlan(memory_deletes=commands),
                consistency=consistency,
            )
        except Exception as exc:
            logger.warning("archive_memory_failed", memory_ids=memory_ids, error=str(exc))
            return
        for command, mutation in zip(commands, result.mutations, strict=False):
            if not mutation.changed:
                logger.warning("archive_memory_noop", memory_id=command.memory_id)
        for error in result.errors:
            logger.warning("archive_memory_failed", error=error)

    def build_archive_memory_commands(self, memory_ids: list[str], *, consistency: str) -> list[MemoryDbDeleteCommand]:
        """Build archive commands for schema-add memory merge/delete operations."""

        return [
            MemoryDbDeleteCommand(
                memory_id=memory_id,
                hard=False,
                reason="schema_add_property_merge",
                consistency=consistency,
            )
            for memory_id in dedupe_non_empty(memory_ids)
        ]

    async def update_memories(
        self,
        context: MemoryRequestContext,
        updates: list[SchemaMemoryUpdate],
        *,
        consistency: str,
    ) -> list[MemoryAddEventItem]:
        """Apply schema-add one-to-one memory updates through MemoryDbWriter."""

        if not updates:
            return []

        commands = await self.build_memory_update_commands(context, updates, consistency=consistency)
        write_result = await self.db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan(memory_updates=commands),
            consistency=consistency,
        )
        return self.memory_update_events(updates, write_result.mutations)

    async def build_memory_update_commands(
        self,
        context: MemoryRequestContext,
        updates: list[SchemaMemoryUpdate],
        *,
        consistency: str,
    ) -> list[MemoryDbUpdateCommand]:
        """Build memory update commands for schema-add one-to-one memory updates."""

        if not updates:
            return []

        vectors = await self._write_plan_builder.build_memory_vectors(
            [update.memory for update in updates],
            project_id=context.project_id,
        )
        vector_by_memory_id = {vector.memory_id: vector for vector in vectors}
        commands: list[MemoryDbUpdateCommand] = []
        for update in updates:
            vector = vector_by_memory_id.get(update.memory.memory_id)
            sparse_vectors = (
                {
                    "bm25_indices": vector.bm25_indices,
                    "bm25_values": vector.bm25_values,
                }
                if vector
                else None
            )
            commands.append(
                MemoryDbUpdateCommand(
                    memory_id=update.target_memory_id,
                    content=update.memory.content,
                    metadata_patch=dict(update.memory.metadata or {}),
                    payload_patch=_memory_payload_patch(update.memory),
                    dense_vector=vector.semantic_vector if vector else None,
                    sparse_vectors=sparse_vectors,
                    reason=update.reason,
                    consistency=consistency,
                )
            )
        return commands

    def memory_update_events(
        self,
        updates: list[SchemaMemoryUpdate],
        results: list[MemoryDbMutationResult],
    ) -> list[MemoryAddEventItem]:
        """Build public add events from schema-add memory update results."""

        entity_groups: dict[str, list[tuple[SchemaMemoryUpdate, MemoryDbMutationResult]]] = {}
        for update, result in zip(updates, results, strict=False):
            if not result.changed:
                logger.warning(
                    "schema_memory_update_not_applied",
                    target_memory_id=update.target_memory_id,
                    entity_id=update.memory.entity_id,
                )
                continue
            eid = update.memory.entity_id or ""
            entity_groups.setdefault(eid, []).append((update, result))

        events: list[MemoryAddEventItem] = []
        for entity_id, items in entity_groups.items():
            first_memory = items[0][0].memory
            entity_name = (first_memory.metadata or {}).get("entity_name", "")
            entity_type = first_memory.entity_type or ""

            content = f"Entity: {entity_name} (Type: {entity_type})"
            related: list[str] = []
            for update, result in items:
                prop_name = update.memory.property_name or update.memory.mem_type or ""
                if update.memory.content:
                    content += f"\n   Property '{prop_name}': {update.memory.content}"
                related.append(result.memory_id)
                related.append(update.target_memory_id)

            events.append(
                MemoryAddEventItem(
                    operation="update",
                    content=content,
                    memory_id=entity_id,
                    mem_type=schema_memory_type(entity_type),
                    related_memory_ids=list(dict.fromkeys(related)),
                )
            )
        return events

    def _new_entity_write(
        self,
        entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
    ) -> EntityWrite:
        """Build an EntityWrite for a newly created entity."""
        entity_id = str(uuid4())
        metadata = base_metadata(request_metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add_v1",
                "record_time": entity.get("record_time"),
                "raw_entity_name": entity.get("name"),
            }
        )
        if entity.get("search_fields"):
            metadata["search_fields"] = list(entity["search_fields"])
        return EntityWrite(
            entity_id=entity_id,
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            user_id=context.user_id,
            app_id=context.app_id,
            session_id=context.session_id,
            agent_id=context.agent_id,
            request_id=context.request_id,
            entity_name=str(entity.get("name") or entity_id),
            entity_type=entity.get("entity_type"),
            description=entity.get("description"),
            schema_version=self.entity_manager.file_path.name,
            metadata=metadata,
            created_at=created_at,
        )

    async def _entity_write_from_view(
        self,
        target: EntityView,
        new_entity: dict[str, Any],
        *,
        context: MemoryRequestContext,
        created_at: datetime,
        prompt_set: AddPromptSet | None = None,
    ) -> EntityWrite:
        """Build an update EntityWrite from an existing entity view and a raw entity."""
        description = await self._llm_description_update(
            entity_name=target.entity_name,
            entity_type=target.entity_type or "",
            current_description=target.description or "",
            properties=new_entity.get("properties", []),
            prompt_set=prompt_set,
        )
        metadata = dict(target.metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add_v1",
                "merge_action": "update",
                "latest_raw_entity_name": new_entity.get("name"),
                "record_time": new_entity.get("record_time"),
            }
        )
        return EntityWrite(
            entity_id=target.entity_id,
            account_id=target.account_id or context.account_id,
            project_id=context.project_id,
            api_key_uuid=target.api_key_uuid or context.api_key_uuid,
            user_id=target.user_id or context.user_id,
            app_id=target.app_id or context.app_id,
            session_id=target.session_id or context.session_id,
            agent_id=target.agent_id or context.agent_id,
            request_id=context.request_id,
            entity_name=target.entity_name,
            entity_type=target.entity_type or new_entity.get("entity_type"),
            description=description,
            schema_version=self.entity_manager.file_path.name,
            metadata=metadata,
            created_at=target.created_at or created_at,
            update_at=created_at,
        )

    def _memory_from_property(
        self,
        *,
        entity_write: EntityWrite,
        prop: dict[str, Any],
        context: MemoryRequestContext,
        created_at: datetime,
        request_metadata: dict[str, Any],
        memory_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> MemoryWrite:
        """Build a MemoryWrite DTO from one extracted property dictionary."""
        memory_id = memory_id or str(uuid4())
        property_name = str(prop.get("property_name") or "default_property")
        metadata = base_metadata(request_metadata)
        metadata.update(
            {
                "add_algorithm": "schema_add_v1",
                "property_time": prop.get("time"),
                "property_operation": prop.get("operation", "set"),
                "entity_name": entity_write.entity_name,
            }
        )
        if extra_metadata:
            metadata.update(extra_metadata)
        mem_type = schema_memory_type(entity_write.entity_type, property_name)
        return MemoryWrite(
            memory_id=memory_id,
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            user_id=context.user_id,
            app_id=context.app_id,
            session_id=context.session_id,
            agent_id=context.agent_id,
            request_id=context.request_id,
            content=str(prop.get("value") or ""),
            mem_type=mem_type,
            mem_extract_type="schema",
            mem_extract_version="schema_add_v1",
            metadata=metadata,
            validate_from=_validate_from_property_time(prop.get("time")),
            created_at=created_at,
            parent_ids=[],
            root_id=[memory_id],
            property_name=property_name,
            entity_id=entity_write.entity_id,
            entity_type=entity_write.entity_type,
        )

    async def _embed_texts(self, task: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.embed_client.embed(task=task, text=texts)
        return response.embeddings


def _validate_from_property_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _format_entity_add_content(entity_write: EntityWrite, prop_memories: list[MemoryWrite]) -> str:
    """Format entity properties into a prompt matching schema search output."""
    block = f"Entity: {entity_write.entity_name} (Type: {entity_write.entity_type})"
    for memory in prop_memories:
        prop_name = memory.property_name or memory.mem_type or ""
        if memory.content:
            block += f"\n   Property '{prop_name}': {memory.content}"
    return block


def _memory_payload_patch(memory: MemoryWrite) -> dict[str, Any]:
    patch = {
        "account_id": memory.account_id,
        "project_id": memory.project_id,
        "api_key_uuid": memory.api_key_uuid,
        "user_id": memory.user_id,
        "app_id": memory.app_id,
        "session_id": memory.session_id,
        "agent_id": memory.agent_id,
        "request_id": memory.request_id,
        "mem_type": memory.mem_type,
        "mem_extract_type": memory.mem_extract_type,
        "mem_extract_version": memory.mem_extract_version,
        "validate_from": memory.validate_from,
        "validate_to": memory.validate_to,
        "property_name": memory.property_name,
        "entity_id": memory.entity_id,
        "entity_type": memory.entity_type,
    }
    return {key: value for key, value in patch.items() if value is not None}
