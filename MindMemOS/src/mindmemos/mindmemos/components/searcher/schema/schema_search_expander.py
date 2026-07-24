"""Schema-aware search expander for entity/property retrieval and graph expansion.

Ported from the original algorithm repository and adapted to the memos component architecture.
All DB / LLM / rerank interactions are injected via constructor dependencies.
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

from ....config.algo.search import SchemaSearchConfig
from ....llm import EmbedClient, RerankClient
from ....logging import get_logger
from ....typing import (
    EntityView,
    FieldCondition,
    MemoryRequestContext,
    SearchFilter,
    combine_search_filters,
)
from ...memory_modeling.schema import EntityManager, PropertyEntry, TemporalEntity
from ...text import SparseVectorEncoder, TextPreprocessor
from ..entity_recall import (
    EntityRecall,
    build_entity_type_filter,
    combine_entity_results_rrf,
)
from ..protocols import EntityHydrator, SearchStrategy
from ..rerank import rerank as _rerank_fn
from ..rerank import rerank_with_scores
from ._entity_fusion import SchemaSearchEntityFusionManager
from ._entity_shrink import schema_search_entity_local_shrink
from ._entity_weights import (
    schema_search_apply_weights_to_ranked,
    schema_search_load_entity_weights,
)
from .property_recall import PropertyRecall, combine_property_results_rrf

logger = get_logger(__name__)


def _episode_input_messages_exclusion_filter() -> SearchFilter:
    return SearchFilter(
        must_not=[
            SearchFilter(
                must=[
                    FieldCondition(field="entity_type", op="match", value="episodes"),
                    FieldCondition(field="property_name", op="match", value="input_messages"),
                ]
            )
        ]
    )


def _exclude_episode_input_messages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if not (
            (metadata := result.get("metadata", {})).get("entity_type") == "episodes"
            and metadata.get("property_name") == "input_messages"
        )
    ]


# Module-level helpers


class SchemaSearchExpander(SearchStrategy, EntityHydrator):
    """Orchestrates hybrid entity + property retrieval with fusion, reranking and graph expansion.

    Public entry point:

    * ``search()`` -- unified multi-hop schema search pipeline (entity store + property store
      -> fusion -> graph expansion -> property shrink -> temporal extension).
    """

    def __init__(
        self,
        *,
        db_reader: Any,
        embed_client: EmbedClient,
        rerank_client: RerankClient | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        entity_manager: EntityManager | None = None,
        config: SchemaSearchConfig,
    ) -> None:
        self.db_reader = db_reader
        self.embed_client = embed_client
        self.rerank_client = rerank_client
        self.text_preprocessor = text_preprocessor
        self.sparse_encoder = sparse_encoder
        self.entity_manager = entity_manager
        self.config = config

        self._entity_fusion = SchemaSearchEntityFusionManager(entity_manager=entity_manager)
        self._entity_recall = EntityRecall(
            db_reader=db_reader,
            embed_client=embed_client,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
            rrf_k=config.entity.rrf_k,
        )
        self._property_recall = PropertyRecall(
            db_reader=db_reader,
            embed_client=embed_client,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
            rrf_k=config.dual_path.property_rrf_k,
        )

        self._entity_weights: dict[str, float] = {}
        if entity_manager:
            self._entity_weights = schema_search_load_entity_weights(entity_manager)

    async def search(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_types: list[str] | None = None,
        property_filter: dict[str, list[str]] | None = None,
        time_window: tuple[str | None, str | None] | None = None,
        search_filter: SearchFilter | None = None,
        entity_search_filter: SearchFilter | None = None,
        num_hops: int | None = None,
        use_reranker: bool = False,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> list[TemporalEntity]:
        """Run the unified multi-hop schema search pipeline."""
        hops = num_hops or self.config.multi_hop
        prop_filter = property_filter or {}

        entity_results = await self._search_from_entity_store(
            ctx,
            query,
            entity_types=entity_types,
            search_filter=search_filter,
            entity_search_filter=entity_search_filter,
            use_reranker=use_reranker,
            top_k=top_k,
            top_n=top_n,
        )

        if self.config.use_entity_agent_search and entity_results:
            _shrink_a1_t0 = _time.monotonic()
            entity_results = await self._shrink_entities(
                ctx,
                query,
                entity_results,
                prop_filter,
            )
            _shrink_a1_elapsed = _time.monotonic() - _shrink_a1_t0
            logger.info("multi_hop_shrink_done", count=len(entity_results), wall_time_s=round(_shrink_a1_elapsed, 2))

        all_entities = list(entity_results)
        if hops > 1 and entity_results:
            unique_ids = {e.entity_id for e in entity_results}
            last_hop = entity_results

            for hop in range(1, hops):
                _hop_t0 = _time.monotonic()

                # Parallel: fetch all neighbor lists for entities in last_hop
                neighbor_tasks = [
                    self.db_reader.get_entity_neighbors(
                        ctx,
                        ent.entity_id,
                        limit=self.config.edge.neighbor_fetch_limit,
                    )
                    for ent in last_hop
                ]
                neighbor_results = await asyncio.gather(*neighbor_tasks, return_exceptions=True)

                # Per-entity: dedup + weight check + relevance filtering
                candidates_to_hydrate: list[EntityView] = []
                for ent, neighbors in zip(last_hop, neighbor_results):
                    if isinstance(neighbors, Exception):
                        logger.warning("multi_hop_neighbors_failed", entity_id=ent.entity_id, error=str(neighbors))
                        continue

                    unseen: list[EntityView] = []
                    for nv in neighbors:
                        if nv.entity_id in unique_ids:
                            continue
                        if self.config.entity_weights.force_balanced_split:
                            if self.config.entity_weights.episode_weight == 0 and nv.entity_type == "episodes":
                                continue
                            if self.config.entity_weights.non_episode_weight == 0 and nv.entity_type != "episodes":
                                continue
                        unseen.append(nv)

                    # Rerank-based neighbor filtering
                    if unseen:
                        filtered = await self._filter_neighbors_by_relevance(
                            query,
                            ent.name,
                            unseen,
                            top_k=self.config.edge.top_k,
                            min_score=self.config.edge.min_relevance_score,
                        )
                        for nv in filtered:
                            if nv.entity_id not in unique_ids:
                                unique_ids.add(nv.entity_id)
                                candidates_to_hydrate.append(nv)

                # Parallel: hydrate all filtered neighbors
                async def _hydrate_one(nv: EntityView) -> TemporalEntity | None:
                    try:
                        return await self.db_reader.get_entity_with_memories(ctx, nv.entity_id, filters=search_filter)
                    except Exception as e:
                        logger.warning("multi_hop_hydrate_failed", entity_id=nv.entity_id, error=str(e))
                        return None

                if candidates_to_hydrate:
                    hydrated_results = await asyncio.gather(*[_hydrate_one(nv) for nv in candidates_to_hydrate])
                    new_entities = [e for e in hydrated_results if e is not None]
                else:
                    new_entities = []

                _hop_elapsed = _time.monotonic() - _hop_t0

                if time_window is not None:
                    new_entities = [self._filter_by_time_with_fallback(e, time_window) for e in new_entities]
                    new_entities = [e for e in new_entities if e is not None]

                if self.config.use_entity_agent_search and new_entities:
                    _hop_shrink_t0 = _time.monotonic()
                    new_entities = await self._shrink_entities(
                        ctx,
                        query,
                        new_entities,
                        prop_filter,
                    )
                    logger.info(
                        "multi_hop_expand_shrink",
                        hop=hop,
                        count=len(new_entities),
                        wall_time_s=round(_time.monotonic() - _hop_shrink_t0, 2),
                    )

                all_entities.extend(new_entities)
                last_hop = new_entities
                logger.info(
                    "multi_hop_expand",
                    hop=hop,
                    candidates=len(candidates_to_hydrate),
                    new=len(new_entities),
                    wall_time_s=round(_hop_elapsed, 2),
                )

        property_results: list[TemporalEntity] = []
        if self.config.dual_path.enabled:
            try:
                property_results = await self._search_from_property_store(
                    ctx,
                    query,
                    entity_types=entity_types,
                    search_filter=search_filter,
                )
                logger.info("property_store_done", count=len(property_results))
            except Exception:
                logger.warning("property_store_failed", exc_info=True)

        if property_results:
            output_entities = self._entity_fusion.fuse_entities(all_entities, property_results)
            logger.info(
                "fusion_done", before=len(all_entities), property=len(property_results), after=len(output_entities)
            )
        else:
            output_entities = all_entities

        if self.config.property.use_property_extension and output_entities:
            _ext_t0 = _time.monotonic()
            output_entities = await self._apply_post_fusion_extension_batch(
                ctx,
                output_entities,
                self.config.property.extension_step,
                memory_filters=search_filter,
            )
            logger.info(
                "multi_hop_extension_done",
                count=len(output_entities),
                extension_step=self.config.property.extension_step,
                wall_time_s=round(_time.monotonic() - _ext_t0, 2),
            )

        if time_window is not None:
            output_entities = [self._filter_by_time_with_fallback(e, time_window) for e in output_entities]
            output_entities = [e for e in output_entities if e is not None]
            logger.info("time_filter_done", count=len(output_entities))

        logger.info("search_done", total=len(output_entities))
        return output_entities

    # Internal: per-entity property shrink with dynamic allocation

    async def _shrink_entities(
        self,
        ctx: MemoryRequestContext,
        query: str,
        entities: list[TemporalEntity],
        prop_filter: dict[str, list[str]],
    ) -> list[TemporalEntity]:
        """Shrink entity properties using a dynamic per-entity budget."""
        if not entities:
            return []

        _shrink_t0 = _time.monotonic()

        base_top_m = self.config.property.top_n
        min_factor = self.config.property.alloc_min_factor
        max_factor = self.config.property.alloc_max_factor
        min_budget = max(1, int(base_top_m * min_factor))
        max_budget = max(min_budget, int(base_top_m * max_factor))

        entity_property_counts: list[int] = []
        for entity in entities:
            allowed = prop_filter.get(entity.entity_type)
            disclosed = entity.get_properties_in_range(allowed, None)
            count = sum(len(tl) for tl in disclosed.values()) if disclosed else 0
            entity_property_counts.append(count)
        total_property_count = sum(entity_property_counts)

        budgets: list[int] = []
        for count in entity_property_counts:
            if total_property_count > 0:
                ratio = count / total_property_count
                allocated = max(1, int(base_top_m * len(entities) * ratio))
            else:
                allocated = base_top_m
            budgets.append(max(min_budget, min(max_budget, allocated)))

        shrink_count = 0
        passthrough_count = 0
        shrink_tasks = []
        for i, entity in enumerate(entities):
            budget = budgets[i]
            if entity_property_counts[i] <= budget:
                passthrough_count += 1

                async def _passthrough(e: TemporalEntity = entity) -> TemporalEntity:
                    return e

                shrink_tasks.append(_passthrough())
            else:
                shrink_count += 1
                shrink_tasks.append(
                    schema_search_entity_local_shrink(
                        entity,
                        query,
                        prop_filter,
                        db_reader=self.db_reader,
                        ctx=ctx,
                        embed_client=self.embed_client,
                        rerank_client=self.rerank_client,
                        text_preprocessor=self.text_preprocessor,
                        sparse_encoder=self.sparse_encoder,
                        entity_manager=self.entity_manager,
                        top_m=budget,
                        hybrid_config={
                            "recall_size": self.config.property.recall_size,
                            "rrf_k": self.config.property.rrf_k,
                            "top_k": self.config.property.top_k,
                        },
                        higher_order_ratio=self.config.property.higher_order_ratio,
                    )
                )

        logger.info(
            "shrink_entities_budget_debug",
            entity_count=len(entities),
            shrink_count=shrink_count,
            passthrough_count=passthrough_count,
            base_top_m=base_top_m,
            higher_order_ratio=self.config.property.higher_order_ratio,
            total_property_count=total_property_count,
            budgets=budgets[:20],
            entity_names=[e.name[:30] for e in entities[:20]],
            entity_prop_counts=entity_property_counts[:20],
        )

        results = await asyncio.gather(*shrink_tasks, return_exceptions=True)
        _shrink_elapsed = _time.monotonic() - _shrink_t0

        disclosed: list[TemporalEntity] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("entity_shrink_failed", entity=entities[i].name, error=str(result))
                disclosed.append(entities[i])
            elif isinstance(result, TemporalEntity):
                disclosed.append(result)
            else:
                disclosed.append(entities[i])

        _props_before = sum(entity_property_counts)
        _props_after = sum(sum(len(tl) for tl in e._properties.values()) for e in disclosed)
        logger.info(
            "shrink_entities_complete_debug",
            entity_count=len(entities),
            shrink_count=shrink_count,
            wall_time_s=round(_shrink_elapsed, 2),
            props_before=_props_before,
            props_after=_props_after,
        )

        return disclosed

    # Internal: entity store path

    async def _search_from_entity_store(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_types: list[str] | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        use_reranker: bool | None = None,
        search_filter: SearchFilter | None = None,
        entity_search_filter: SearchFilter | None = None,
    ) -> list[TemporalEntity]:
        """Hybrid entity-level retrieval: vector + BM25 -> RRF -> rerank -> hydrate.

        Supports balanced split (episode vs non-episode) when configured.
        """
        entity_top_k = top_k or self.config.entity.top_k
        entity_top_n = top_n or self.config.entity.top_n
        recall_size = self.config.entity.recall_size
        rrf_k = self.config.entity.rrf_k
        should_use_reranker = self.config.entity.use_reranker if use_reranker is None else use_reranker

        if self.config.entity_weights.force_balanced_split:
            return await self._balanced_entity_search(
                ctx,
                query,
                entity_types=entity_types,
                search_filter=search_filter,
                entity_search_filter=entity_search_filter,
                recall_size=recall_size,
                rrf_k=rrf_k,
                top_k=entity_top_k,
                top_n=entity_top_n,
                use_reranker=should_use_reranker,
            )

        type_filter = build_entity_type_filter(entity_types)
        final_entity_filter = combine_search_filters(type_filter, entity_search_filter)

        vector_results, bm25_results = await self._dual_entity_recall(
            ctx,
            query,
            filters=final_entity_filter,
            recall_size=recall_size,
        )

        rrf_results = combine_entity_results_rrf(
            vector_results,
            bm25_results,
            rrf_k=rrf_k,
            top_k=entity_top_k,
        )

        if self.config.entity.use_maxsim_rescore and rrf_results:
            rrf_results = await self._maxsim_rescore(query, rrf_results)

        if self.rerank_client and should_use_reranker and rrf_results:
            rrf_results = await self._rerank_entity_results(query, rrf_results, entity_top_n)

        entities = await self._hydrate_entities(ctx, rrf_results, memory_filters=search_filter)

        if self._entity_weights:
            entities = schema_search_apply_weights_to_ranked(entities, self._entity_weights)

        return entities

    async def _balanced_entity_search(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_types: list[str] | None = None,
        recall_size: int = 800,
        rrf_k: int = 60,
        top_k: int = 30,
        top_n: int = 15,
        use_reranker: bool | None = None,
        search_filter: SearchFilter | None = None,
        entity_search_filter: SearchFilter | None = None,
    ) -> list[TemporalEntity]:
        """Balanced split retrieval: episode entities and non-episode entities are searched separately."""
        ep_weight = self.config.entity_weights.episode_weight
        non_ep_weight = self.config.entity_weights.non_episode_weight
        should_use_reranker = self.config.entity.use_reranker if use_reranker is None else use_reranker

        ep_top_k = max(1, int(top_k * ep_weight))
        non_ep_top_k = max(1, int(top_k * non_ep_weight))
        ep_top_n = max(1, int(top_n * ep_weight))
        non_ep_top_n = max(1, int(top_n * non_ep_weight))

        ep_filter = combine_search_filters(build_entity_type_filter(["episodes"]), entity_search_filter)

        # Non-episode path filter: exclude "episodes"
        if entity_types:
            non_ep_types = [t for t in entity_types if t != "episodes"]
            non_ep_filter = build_entity_type_filter(non_ep_types) if non_ep_types else None
        else:
            non_ep_filter = SearchFilter(must_not=[FieldCondition(field="entity_type", op="match", value="episodes")])
        non_ep_filter = combine_search_filters(non_ep_filter, entity_search_filter)

        ep_vec_task = self._entity_recall.recall_dense(
            ctx,
            query=query,
            filters=ep_filter,
            limit=recall_size,
        )
        ep_bm25_task = self._entity_sparse_recall_safe(ctx, query, filters=ep_filter, limit=recall_size)
        non_ep_vec_task = self._entity_recall.recall_dense(
            ctx,
            query=query,
            filters=non_ep_filter,
            limit=recall_size,
        )
        non_ep_bm25_task = self._entity_sparse_recall_safe(ctx, query, filters=non_ep_filter, limit=recall_size)

        ep_vec, ep_bm25, non_ep_vec, non_ep_bm25 = await asyncio.gather(
            ep_vec_task,
            ep_bm25_task,
            non_ep_vec_task,
            non_ep_bm25_task,
        )

        ep_rrf = combine_entity_results_rrf(ep_vec, ep_bm25, rrf_k=rrf_k, top_k=ep_top_k)
        non_ep_rrf = combine_entity_results_rrf(non_ep_vec, non_ep_bm25, rrf_k=rrf_k, top_k=non_ep_top_k)

        if self.config.entity.use_maxsim_rescore:
            if ep_rrf:
                ep_rrf = await self._maxsim_rescore(query, ep_rrf)
            if non_ep_rrf:
                non_ep_rrf = await self._maxsim_rescore(query, non_ep_rrf)

        if self.rerank_client and should_use_reranker:
            if ep_rrf:
                ep_rrf = await self._rerank_entity_results(query, ep_rrf, ep_top_n)
            if non_ep_rrf:
                non_ep_rrf = await self._rerank_entity_results(query, non_ep_rrf, non_ep_top_n)
        else:
            ep_rrf = ep_rrf[:ep_top_n]
            non_ep_rrf = non_ep_rrf[:non_ep_top_n]

        ep_entities = await self._hydrate_entities(ctx, ep_rrf, memory_filters=search_filter)
        non_ep_entities = await self._hydrate_entities(ctx, non_ep_rrf, memory_filters=search_filter)

        # Apply entity type weights to non-episode path
        if self._entity_weights:
            non_ep_entities = schema_search_apply_weights_to_ranked(non_ep_entities, self._entity_weights)

        # Merge: non-episode first (higher precision), then episodes
        merged = non_ep_entities + ep_entities

        logger.info(
            "balanced_entity_search",
            ep_count=len(ep_entities),
            non_ep_count=len(non_ep_entities),
            total=len(merged),
        )
        return merged

    # Internal: property store path

    async def _search_from_property_store(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        entity_types: list[str] | None = None,
        search_filter: SearchFilter | None = None,
    ) -> list[TemporalEntity]:
        """Property-level hybrid retrieval: vector + BM25 -> RRF -> rerank -> assemble to entities."""
        if not self.config.dual_path.enabled:
            return []

        prop_recall_size = self.config.dual_path.property_recall_size
        prop_rrf_k = self.config.dual_path.property_rrf_k
        prop_top_k = self.config.dual_path.property_top_k
        prop_top_n = self.config.dual_path.property_top_n
        property_search_filter = combine_search_filters(
            build_entity_type_filter(entity_types),
            _episode_input_messages_exclusion_filter(),
            search_filter,
        )

        vector_results: list[dict[str, Any]] = []
        bm25_results: list[dict[str, Any]] = []

        async def _vector_recall() -> list[dict[str, Any]]:
            try:
                return await self._property_recall.recall_dense(
                    ctx,
                    query=query,
                    filters=property_search_filter,
                    limit=prop_recall_size,
                )
            except Exception as e:
                logger.warning("property_store_vector_recall_failed", error=str(e))
                return []

        async def _bm25_recall() -> list[dict[str, Any]]:
            if not (self.text_preprocessor and self.sparse_encoder):
                return []
            try:
                return await self._property_recall.recall_sparse(
                    ctx,
                    query=query,
                    filters=property_search_filter,
                    limit=prop_recall_size,
                )
            except Exception as e:
                logger.warning("property_store_bm25_recall_failed", error=str(e))
                return []

        vector_results, bm25_results = await asyncio.gather(_vector_recall(), _bm25_recall())
        vector_results = _exclude_episode_input_messages(vector_results)
        bm25_results = _exclude_episode_input_messages(bm25_results)

        if not vector_results and not bm25_results:
            return []

        rrf_results = combine_property_results_rrf(
            vector_results,
            bm25_results,
            rrf_k=prop_rrf_k,
            top_k=prop_top_k,
        )

        if self.rerank_client and rrf_results:
            docs = [r.get("content", "") for r in rrf_results]
            indices = await _rerank_fn(
                self.rerank_client,
                query,
                docs,
                min(prop_top_n, len(docs)),
            )
            rrf_results = [rrf_results[i] for i in indices if 0 <= i < len(rrf_results)]
        else:
            rrf_results = rrf_results[:prop_top_n]

        assembled = self._entity_fusion.assemble_entity_from_properties(rrf_results)

        logger.info(
            "property_store_search",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            rrf_count=len(rrf_results),
            assembled_count=len(assembled),
        )
        return assembled

    # Internal: dual recall + rerank helpers

    async def _dual_entity_recall(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        filters: SearchFilter | None = None,
        recall_size: int = 800,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parallel vector + BM25 entity recall."""
        vector_task = self._entity_recall.recall_dense(
            ctx,
            query=query,
            filters=filters,
            limit=recall_size,
        )
        bm25_task = self._entity_sparse_recall_safe(ctx, query, filters=filters, limit=recall_size)

        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        return vector_results, bm25_results

    async def _entity_sparse_recall_safe(
        self,
        ctx: MemoryRequestContext,
        query: str,
        *,
        filters: SearchFilter | None = None,
        limit: int = 800,
    ) -> list[dict[str, Any]]:
        """BM25 entity recall with graceful fallback when preprocessor/encoder unavailable."""
        if not self.text_preprocessor or not self.sparse_encoder:
            return []
        try:
            return await self._entity_recall.recall_sparse(
                ctx,
                query=query,
                filters=filters,
                limit=limit,
            )
        except Exception as e:
            logger.warning("entity_sparse_recall_failed", error=str(e))
            return []

    async def _rerank_entity_results(
        self,
        query: str,
        rrf_results: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Rerank RRF-fused entity results and return top-N."""
        if not self.rerank_client or not rrf_results:
            return rrf_results[:top_n]

        max_candidates = self.config.entity.max_rerank_candidates
        if len(rrf_results) > max_candidates:
            rrf_results = rrf_results[:max_candidates]

        docs: list[str] = []
        for r in rrf_results:
            best_sf = str(r.get("best_search_field") or "").strip()
            docs.append(best_sf or str(r.get("entity_id") or ""))

        indices = await _rerank_fn(
            self.rerank_client,
            query,
            docs,
            min(top_n, len(docs)),
        )
        return [rrf_results[i] for i in indices if 0 <= i < len(rrf_results)]

    async def _maxsim_rescore(
        self,
        query: str,
        rrf_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-score RRF results using MaxSim over per-entity search fields.

        For each entity, embeds its search_fields individually and computes
        cosine similarity against the query vector, taking the maximum.
        """
        if not rrf_results or not self.embed_client:
            return rrf_results

        _t0 = _time.monotonic()
        maxsim_weight = self.config.entity.maxsim_weight

        all_sf_texts: list[str] = []
        sf_ranges: list[tuple[str, int, int]] = []

        for r in rrf_results:
            eid = r.get("entity_id", "")
            entity_view = r.get("entity_view")
            sfs: list[str] = []
            if entity_view and hasattr(entity_view, "metadata") and entity_view.metadata:
                sfs = [sf for sf in entity_view.metadata.get("search_fields", []) if isinstance(sf, str) and sf.strip()]
            start = len(all_sf_texts)
            all_sf_texts.extend(sfs)
            sf_ranges.append((eid, start, len(all_sf_texts)))

        if not all_sf_texts:
            return rrf_results

        try:
            query_resp = await self.embed_client.embed(task="search.maxsim_query", text=query)
            sf_resp = await self.embed_client.embed(task="search.maxsim_fields", text=all_sf_texts)
        except Exception as e:
            logger.warning("maxsim_embed_failed", error=str(e))
            return rrf_results

        query_vec = query_resp.embeddings[0] if query_resp.embeddings else []
        if not query_vec or not sf_resp.embeddings:
            return rrf_results

        def _cosine(a: list[float], b: list[float]) -> float:
            if len(a) != len(b) or not a:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        entity_max_sim: dict[str, float] = {}
        entity_best_field: dict[str, tuple[str, int, float]] = {}
        for eid, start, end in sf_ranges:
            for i in range(start, end):
                if i < len(sf_resp.embeddings):
                    sim = _cosine(query_vec, sf_resp.embeddings[i])
                    if sim > entity_max_sim.get(eid, 0.0):
                        entity_max_sim[eid] = sim
                        entity_best_field[eid] = (all_sf_texts[i], i - start, sim)

        rrf_scores = [r.get("rrf_score", 0.0) for r in rrf_results]
        max_rrf = max(rrf_scores) if rrf_scores else 1.0
        if max_rrf <= 0:
            max_rrf = 1.0

        for r in rrf_results:
            eid = r.get("entity_id", "")
            norm_rrf = r.get("rrf_score", 0.0) / max_rrf
            msim = entity_max_sim.get(eid, 0.0)
            r["maxsim_score"] = msim
            if eid in entity_best_field:
                best_field_text, best_field_index, best_score = entity_best_field[eid]
                r["maxsim_search_field"] = best_field_text
                r["maxsim_search_field_index"] = best_field_index
                if best_field_text:
                    r["best_search_field"] = best_field_text
                    r["best_search_field_index"] = best_field_index
                    r["best_search_field_score"] = best_score
                    r["best_search_field_source"] = "maxsim"
            r["combined_score"] = (1 - maxsim_weight) * norm_rrf + maxsim_weight * msim

        rrf_results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)

        logger.info(
            "maxsim_rescore_done",
            entity_count=len(rrf_results),
            sf_texts=len(all_sf_texts),
            wall_time_s=round(_time.monotonic() - _t0, 2),
        )
        return rrf_results

    async def _filter_neighbors_by_relevance(
        self,
        query: str,
        source_name: str,
        neighbors: list[EntityView],
        *,
        top_k: int = 2,
        min_score: float = 0.1,
    ) -> list[EntityView]:
        """Rerank neighbor entities by edge relevance and filter by score threshold."""
        if not neighbors:
            return []

        if not self.rerank_client:
            return neighbors[:top_k]

        docs = [
            f"{source_name} --[{nv.metadata.get('_relation', '')}]--> {nv.entity_name} (Type: {nv.entity_type})"
            for nv in neighbors
        ]

        _t0 = _time.monotonic()
        try:
            scored = await rerank_with_scores(
                self.rerank_client,
                query,
                docs,
                min(top_k, len(docs)),
            )
        except Exception as e:
            logger.warning("neighbor_rerank_failed", source=source_name, error=str(e))
            return neighbors[:top_k]

        kept: list[EntityView] = []
        for idx, score in scored:
            if 0 <= idx < len(neighbors) and score >= min_score:
                kept.append(neighbors[idx])

        # Fallback: keep the best one if all below threshold
        if not kept and scored:
            best_idx, _ = scored[0]
            if 0 <= best_idx < len(neighbors):
                kept = [neighbors[best_idx]]

        logger.info(
            "neighbor_rerank_done",
            source=source_name,
            total=len(neighbors),
            kept=len(kept),
            threshold=min_score,
            wall_time_s=round(_time.monotonic() - _t0, 2),
        )
        return kept

    async def _hydrate_entities(
        self,
        ctx: MemoryRequestContext,
        rrf_results: list[dict[str, Any]],
        *,
        memory_filters: SearchFilter | None = None,
    ) -> list[TemporalEntity]:
        """Hydrate lightweight RRF result dicts to full TemporalEntity via db_reader."""
        _t0 = _time.monotonic()
        entity_ids = [r["entity_id"] for r in rrf_results if r.get("entity_id")]
        if not entity_ids:
            return []

        async def _hydrate(eid: str) -> TemporalEntity | None:
            try:
                return await self.db_reader.get_entity_with_memories(ctx, eid, filters=memory_filters)
            except Exception as e:
                logger.warning("hydrate_entity_failed", entity_id=eid, error=str(e))
                return None

        results = await asyncio.gather(*[_hydrate(eid) for eid in entity_ids])
        entities = [e for e in results if e is not None]
        logger.info(
            "hydrate_entities_done",
            requested=len(entity_ids),
            loaded=len(entities),
            wall_time_s=round(_time.monotonic() - _t0, 2),
        )
        return entities

    async def hydrate(self, ctx: MemoryRequestContext, entity_ids: list[str]) -> list[TemporalEntity]:
        """Hydrate entity IDs to full temporal entities through the searcher protocol."""

        return await self._hydrate_entities(ctx, [{"entity_id": entity_id} for entity_id in entity_ids])

    # Internal: post-fusion processing

    async def _apply_post_fusion_extension_batch(
        self,
        ctx: MemoryRequestContext,
        entities: list[TemporalEntity],
        extension_step: int = 3,
        *,
        memory_filters: SearchFilter | None = None,
    ) -> list[TemporalEntity]:
        """Hydrate originals and expand fused property timelines after fusion."""
        if not extension_step or extension_step <= 0:
            return entities

        async def _extend(entity: TemporalEntity) -> TemporalEntity:
            try:
                original = await self.db_reader.get_entity_with_memories(ctx, entity.entity_id, filters=memory_filters)
            except Exception as e:
                logger.warning("post_fusion_extension_hydrate_failed", entity_id=entity.entity_id, error=str(e))
                original = None
            if original is None:
                return entity
            return self._apply_post_fusion_extension(original, entity, extension_step)

        results = await asyncio.gather(*[_extend(e) for e in entities])
        return list(results)

    def _apply_post_fusion_extension(
        self,
        original: TemporalEntity,
        disclosed: TemporalEntity,
        extension_step: int = 3,
    ) -> TemporalEntity:
        """Expand disclosed property timelines by looking forward/backward in the original entity.

        For each disclosed property value, find its position in the original timeline
        and include ``extension_step`` entries before and after it.

        Args:
            original: The full entity before shrink.
            disclosed: The shrunk entity after property selection.
            extension_step: Number of entries to expand in each direction.

        Returns:
            A new TemporalEntity with extended property timelines.
        """
        if not extension_step or extension_step <= 0 or disclosed.entity_type == "episodes":
            return disclosed

        extended_properties: dict[str, list[PropertyEntry]] = {}

        for prop_name, disclosed_timeline in disclosed._properties.items():
            if prop_name == "default_property":
                extended_properties[prop_name] = list(disclosed_timeline)
                continue
            original_timeline = original._properties.get(prop_name, [])
            if not original_timeline or not disclosed_timeline:
                extended_properties[prop_name] = list(disclosed_timeline)
                continue

            disclosed_uids = {entry.uid for entry in disclosed_timeline}

            included_indices: set[int] = set()
            for i, entry in enumerate(original_timeline):
                if entry.uid in disclosed_uids:
                    start = max(0, i - extension_step)
                    end = min(len(original_timeline), i + extension_step + 1)
                    for j in range(start, end):
                        included_indices.add(j)

            # Build merged list from original timeline at included indices, sorted by timestamp
            merged: list[PropertyEntry] = [original_timeline[i] for i in sorted(included_indices)]
            extended_properties[prop_name] = merged

        result = TemporalEntity(
            entity_id=disclosed.entity_id,
            name=disclosed.name,
            entity_type=disclosed.entity_type,
            description=disclosed.description,
            record_time=disclosed.record_time,
            entity_manager=disclosed._entity_manager,
            auto_create_properties=False,
        )
        result._properties = extended_properties
        result.edges = list(disclosed.edges)
        result.search_fields = list(disclosed.search_fields)
        return result

    def _filter_by_time_with_fallback(
        self,
        entity: TemporalEntity,
        time_window: tuple[str | None, str | None],
    ) -> TemporalEntity:
        """Apply time-window filtering and fall back to the original entity when empty."""
        filtered = entity.filter_by_time(time_window)
        if entity.entity_type == "episodes":
            return filtered
        if any(timeline for timeline in filtered._properties.values()):
            return filtered
        return entity
