"""Entity local shrink: property-level recall, fusion, reranking for a single entity.

Extracted from ``TemporalEntity.entity_local_shrink()`` in the original algorithm repo
and adapted to the memos component architecture (standalone async function, no DB / LLM
dependency on the entity class itself).
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

from ....llm import EmbedClient, RerankClient
from ....logging import get_logger
from ....typing import MemoryRequestContext
from ...memory_modeling.schema import EntityManager, PropertyEntry, TemporalEntity
from ...text import SparseVectorEncoder, TextPreprocessor
from ..rerank import rerank
from .property_recall import PropertyRecall

logger = get_logger(__name__)


async def schema_search_entity_local_shrink(
    entity: TemporalEntity,
    user_query: str,
    global_property_filter: dict[str, list[str]],
    *,
    db_reader: Any,
    ctx: MemoryRequestContext,
    embed_client: EmbedClient | None = None,
    rerank_client: RerankClient | None = None,
    text_preprocessor: TextPreprocessor | None = None,
    sparse_encoder: SparseVectorEncoder | None = None,
    entity_manager: EntityManager | None = None,
    top_m: int = 10,
    hybrid_config: dict | None = None,
    higher_order_ratio: float = 0.0,
) -> TemporalEntity:
    """Entity self-retrieval: decide which properties to disclose.

    Pipeline:
      1. Determine allowed property range from *global_property_filter*.
      2. Fetch full property values in range.
      3. Dual-path recall (vector + BM25) -> RRF fusion -> Reranker precision (non-episodes only).

    Args:
        entity: The entity to shrink.
        user_query: User query string.
        global_property_filter: ``{entity_type: [property1, ...]}`` allowlist.
        db_reader: MemoryDbReader instance.
        ctx: Memory request context.
        embed_client: Optional embedding client for fallback reranking.
        rerank_client: Optional rerank client for precision reranking.
        text_preprocessor: Text preprocessor for BM25 sparse recall.
        sparse_encoder: Sparse vector encoder for BM25 sparse recall.
        entity_manager: Optional entity schema manager for higher-order split.
        top_m: Number of top properties to return.
        hybrid_config: Hybrid recall configuration (recall_size, rrf_k, top_k).
        higher_order_ratio: Ratio of budget allocated to higher-order properties.

    Returns:
        A new TemporalEntity with the same identity but shrunk properties.
    """

    if not isinstance(top_m, int):
        logger.warning("entity_local_shrink bad param: top_m should be int", actual_type=type(top_m).__name__)
        top_m = 10

    _entity_shrink_t0 = _time.monotonic()
    reasoning_parts: list[str] = []

    if isinstance(global_property_filter, dict):
        allowed_properties = global_property_filter.get(entity.entity_type, None)
    else:
        allowed_properties = None

    disclosed = entity.get_properties_in_range(allowed_properties)

    total_props_before = sum(len(v) for v in disclosed.values()) if disclosed else 0
    logger.info(
        "shrink_step2_properties",
        entity_name=entity.name,
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        field_count=len(disclosed),
        value_count=total_props_before,
    )

    reranked = False
    skip_shrink = entity.entity_type == "episodes"

    # Detect higher-order property split
    ho_prop_names: set[str] = set()
    if not skip_shrink and higher_order_ratio > 0 and entity_manager is not None:
        ho_prop_names = entity_manager.get_higher_order_property_names(entity.entity_type)

    has_ho_split = bool(ho_prop_names) and higher_order_ratio > 0

    if not skip_shrink and disclosed:
        recall_size = hybrid_config.get("recall_size", 45) if hybrid_config else 45
        rrf_k = hybrid_config.get("rrf_k", 60) if hybrid_config else 60
        rrf_top_k = hybrid_config.get("top_k", top_m * 2) if hybrid_config else top_m * 2
        property_recall = PropertyRecall(
            db_reader=db_reader,
            embed_client=embed_client,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
            rrf_k=rrf_k,
        )

        if has_ho_split:
            # Split budgets let first-order and higher-order properties shrink independently.
            ho_budget = max(1, int(top_m * higher_order_ratio))
            fo_budget = max(1, top_m - ho_budget)
            logger.info(
                "shrink_split_budget",
                entity_name=entity.name,
                fo_budget=fo_budget,
                ho_budget=ho_budget,
                ratio=higher_order_ratio,
            )

            # Unified dual recall for the whole entity, then split by property_name
            _dual_recall_t0 = _time.monotonic()
            pvm_results, bm25_results = await _dual_recall(
                entity,
                user_query,
                recall_size,
                db_reader,
                ctx,
                embed_client=embed_client,
                text_preprocessor=text_preprocessor,
                sparse_encoder=sparse_encoder,
            )
            _dual_recall_elapsed = _time.monotonic() - _dual_recall_t0
            logger.info(
                "shrink_dual_recall_timing",
                entity_name=entity.name,
                wall_time_s=round(_dual_recall_elapsed, 2),
                vector_count=len(pvm_results),
                bm25_count=len(bm25_results),
            )

            fo_pvm = [r for r in pvm_results if r.get("metadata", {}).get("property_name", "") not in ho_prop_names]
            fo_bm25 = [r for r in bm25_results if r.get("metadata", {}).get("property_name", "") not in ho_prop_names]
            ho_pvm = [r for r in pvm_results if r.get("metadata", {}).get("property_name", "") in ho_prop_names]
            ho_bm25 = [r for r in bm25_results if r.get("metadata", {}).get("property_name", "") in ho_prop_names]

            # First-order shrink
            _fo_t0 = _time.monotonic()
            fo_disclosed = await _shrink_single_group(
                entity,
                user_query,
                fo_pvm,
                fo_bm25,
                fo_budget,
                rrf_k,
                rrf_top_k,
                rerank_client,
                embed_client,
                "first-order",
            )
            _fo_elapsed = _time.monotonic() - _fo_t0
            # Higher-order shrink
            _ho_t0 = _time.monotonic()
            ho_disclosed = await _shrink_single_group(
                entity,
                user_query,
                ho_pvm,
                ho_bm25,
                ho_budget,
                rrf_k,
                rrf_top_k,
                rerank_client,
                embed_client,
                "higher-order",
            )
            _ho_elapsed = _time.monotonic() - _ho_t0
            logger.info(
                "shrink_split_rerank_timing",
                entity_name=entity.name,
                fo_time_s=round(_fo_elapsed, 2),
                ho_time_s=round(_ho_elapsed, 2),
                fo_docs=len(fo_pvm) + len(fo_bm25),
                ho_docs=len(ho_pvm) + len(ho_bm25),
            )

            disclosed = {}
            for d in [fo_disclosed, ho_disclosed]:
                for pn, vals in d.items():
                    disclosed.setdefault(pn, []).extend(vals)

            reranked = True
            merged_count = sum(len(v) for v in disclosed.values())
            reasoning_parts.append(
                f"split_budget: fo={fo_budget}->{sum(len(v) for v in fo_disclosed.values())}, "
                f"ho={ho_budget}->{sum(len(v) for v in ho_disclosed.values())}, total={merged_count}"
            )
        else:
            logger.info(
                "shrink_step3_start",
                entity_name=entity.name,
                recall_size=recall_size,
                rrf_k=rrf_k,
                rrf_top_k=rrf_top_k,
                top_m=top_m,
            )

            # Stage 1: Dual-path recall (vector + BM25, both filtered by entity_id) — parallel
            _unified_recall_t0 = _time.monotonic()

            async def _vector_recall() -> list[dict[str, Any]]:
                if not embed_client:
                    logger.warning("shrink_no_embed_client", entity_name=entity.name)
                    return []
                try:
                    results = await property_recall.recall_dense(
                        ctx,
                        query=user_query,
                        entity_id=entity.entity_id,
                        limit=recall_size,
                    )
                    logger.info("shrink_vector_recall", entity_name=entity.name, count=len(results))
                    return results
                except Exception as e:
                    logger.warning("shrink_vector_recall_failed", entity_name=entity.name, error=str(e))
                    return []

            async def _bm25_recall() -> list[dict[str, Any]]:
                if not (text_preprocessor and sparse_encoder):
                    logger.warning("shrink_no_sparse_components", entity_name=entity.name)
                    return []
                try:
                    results = await property_recall.recall_sparse(
                        ctx,
                        query=user_query,
                        entity_id=entity.entity_id,
                        limit=recall_size,
                    )
                    logger.info("shrink_bm25_recall", entity_name=entity.name, count=len(results))
                    return results
                except Exception as e:
                    logger.warning("shrink_bm25_recall_failed", entity_name=entity.name, error=str(e))
                    return []

            pvm_results, bm25_results = await asyncio.gather(_vector_recall(), _bm25_recall())

            # Stage 2: RRF fusion -> top_k
            _unified_recall_elapsed = _time.monotonic() - _unified_recall_t0
            logger.info(
                "shrink_unified_recall_timing",
                entity_name=entity.name,
                wall_time_s=round(_unified_recall_elapsed, 2),
                vector_count=len(pvm_results),
                bm25_count=len(bm25_results),
            )
            if pvm_results or bm25_results:
                fused = _rrf_fuse_properties(pvm_results, bm25_results, k=rrf_k, top_k=rrf_top_k)
                disclosed = _convert_fused_to_disclosed(fused)
                fused_count = sum(len(v) for v in disclosed.values())
                logger.info(
                    "shrink_rrf_fusion",
                    entity_name=entity.name,
                    vector_count=len(pvm_results),
                    bm25_count=len(bm25_results),
                    fused_count=len(fused),
                    disclosed_count=fused_count,
                )
                reasoning_parts.append(f"dual_recall(V:{len(pvm_results)}+B:{len(bm25_results)})->RRF->{fused_count}")

                # Stage 3: Reranker precision -> top_m
                if rerank_client and disclosed:
                    before_rerank = sum(len(v) for v in disclosed.values())
                    _rerank_t0 = _time.monotonic()
                    disclosed = await _rerank_properties_with_reranker(
                        entity,
                        user_query,
                        disclosed,
                        rerank_client,
                        top_m,
                    )
                    _rerank_elapsed = _time.monotonic() - _rerank_t0
                    after_rerank = sum(len(v) for v in disclosed.values())
                    logger.info(
                        "shrink_reranker",
                        entity_name=entity.name,
                        before=before_rerank,
                        after=after_rerank,
                        top_m=top_m,
                        rerank_time_s=round(_rerank_elapsed, 2),
                    )
                    reasoning_parts.append(f"reranker->top-{after_rerank}")
                    reranked = True
                elif not rerank_client:
                    logger.warning("shrink_no_reranker", entity_name=entity.name)
            else:
                logger.warning(
                    "shrink_no_recall_results",
                    entity_name=entity.name,
                    fallback_count=total_props_before,
                )
                reasoning_parts.append("dual_recall_empty, using_full_properties")

            # Fallback: embedding rerank if no reranker or no dual-recall results
            if not reranked and embed_client and disclosed:
                before_emb = sum(len(v) for v in disclosed.values())
                disclosed = await _rerank_with_embedding(
                    entity,
                    user_query,
                    disclosed,
                    embed_client,
                    top_m,
                )
                after_emb = sum(len(v) for v in disclosed.values())
                logger.info(
                    "shrink_embedding_rerank_fallback",
                    entity_name=entity.name,
                    before=before_emb,
                    after=after_emb,
                )
                reasoning_parts.append(f"embedding_rerank->top-{after_emb}")
                reranked = True

    elif not skip_shrink and not disclosed:
        logger.info("shrink_empty_properties", entity_name=entity.name)
        reasoning_parts.append("empty_properties, skip_shrink")
    else:
        logger.info(
            "shrink_episodes_skip",
            entity_name=entity.name,
            property_count=total_props_before,
        )
        reasoning_parts.append("episodes_entity, skip_property_shrink")

    final_count = sum(len(v) for v in disclosed.values()) if disclosed else 0
    logger.info(
        "shrink_step3_complete",
        entity_name=entity.name,
        before=total_props_before,
        after=final_count,
        reranked=reranked,
    )

    disclosed_entity = _build_disclosed_entity(entity, disclosed)
    _entity_shrink_elapsed = _time.monotonic() - _entity_shrink_t0
    logger.info(
        "entity_local_shrink_total_timing",
        entity_name=entity.name,
        entity_type=entity.entity_type,
        wall_time_s=round(_entity_shrink_elapsed, 2),
        props_before=total_props_before,
        props_after=final_count,
        has_ho_split=has_ho_split,
        top_m=top_m,
        higher_order_ratio=higher_order_ratio,
    )
    return disclosed_entity


def _get_property_id(result: dict[str, Any]) -> str:
    """Extract unique property identifier from a recall result dict."""
    meta = result.get("metadata", {})
    entity_id = meta.get("entity_id", "")
    prop_name = meta.get("property_name", "")
    uid = meta.get("uid", "")
    if uid:
        return f"{entity_id}#{prop_name}#{uid}"
    timestamp = meta.get("timestamp", "")
    return f"{entity_id}#{prop_name}#{timestamp}"


def _rrf_fuse_properties(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """RRF fusion of two property recall result lists, deduped by property_id."""
    scores: dict[str, float] = {}
    all_items: dict[str, dict[str, Any]] = {}

    for rank, r in enumerate(vector_results):
        pid = _get_property_id(r)
        scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
        if pid not in all_items:
            all_items[pid] = r

    for rank, r in enumerate(bm25_results):
        pid = _get_property_id(r)
        scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
        if pid not in all_items:
            all_items[pid] = r

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    result = [all_items[pid] for pid in sorted_ids if pid in all_items]
    logger.debug(
        "rrf_fuse_properties",
        vector_count=len(vector_results),
        bm25_count=len(bm25_results),
        candidates=len(all_items),
        output_count=len(result),
        k=k,
    )
    return result


def _convert_fused_to_disclosed(fused: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Convert fused recall result list to disclosed format ``{prop: [{value, timestamp}, ...]}``."""
    disclosed: dict[str, list[dict[str, Any]]] = {}
    for item in fused:
        meta = item.get("metadata", {})
        prop_name = meta.get("property_name", "")
        # property_value comes from the recall result; fall back to content
        prop_value = meta.get("property_value", "") or item.get("content", "")
        timestamp = meta.get("timestamp", "")
        if prop_name and prop_value:
            disclosed.setdefault(prop_name, []).append({"value": prop_value, "timestamp": timestamp})
    return disclosed


async def _dual_recall(
    entity: TemporalEntity,
    user_query: str,
    recall_size: int,
    db_reader: Any,
    ctx: MemoryRequestContext,
    *,
    embed_client: EmbedClient | None = None,
    text_preprocessor: TextPreprocessor | None = None,
    sparse_encoder: SparseVectorEncoder | None = None,
    property_recall: PropertyRecall | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Perform unified dual-path recall (vector + BM25) for a single entity."""
    recall_component = property_recall or PropertyRecall(
        db_reader=db_reader,
        embed_client=embed_client,
        text_preprocessor=text_preprocessor,
        sparse_encoder=sparse_encoder,
    )

    async def _vector() -> list[dict[str, Any]]:
        if not embed_client:
            return []
        try:
            return await recall_component.recall_dense(
                ctx,
                query=user_query,
                entity_id=entity.entity_id,
                limit=recall_size,
            )
        except Exception as e:
            logger.warning("dual_recall_vector_failed", entity_name=entity.name, error=str(e))
            return []

    async def _bm25() -> list[dict[str, Any]]:
        if not (text_preprocessor and sparse_encoder):
            return []
        try:
            return await recall_component.recall_sparse(
                ctx,
                query=user_query,
                entity_id=entity.entity_id,
                limit=recall_size,
            )
        except Exception as e:
            logger.warning("dual_recall_bm25_failed", entity_name=entity.name, error=str(e))
            return []

    return await asyncio.gather(_vector(), _bm25())


async def _shrink_single_group(
    entity: TemporalEntity,
    user_query: str,
    pvm_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    budget: int,
    rrf_k: int,
    rrf_top_k: int,
    rerank_client: RerankClient | None,
    embed_client: EmbedClient | None,
    group_label: str,
) -> dict[str, list[dict[str, Any]]]:
    """Execute RRF fusion + reranker precision for one property group, returning disclosed dict."""
    if not pvm_results and not bm25_results:
        logger.debug("shrink_group_empty", entity_name=entity.name, group=group_label)
        return {}

    fused = _rrf_fuse_properties(pvm_results, bm25_results, k=rrf_k, top_k=rrf_top_k)
    disclosed = _convert_fused_to_disclosed(fused)
    fused_count = sum(len(v) for v in disclosed.values())
    logger.info(
        "shrink_group_rrf",
        entity_name=entity.name,
        group=group_label,
        vector_count=len(pvm_results),
        bm25_count=len(bm25_results),
        fused_count=fused_count,
    )

    if rerank_client and disclosed:
        disclosed = await _rerank_properties_with_reranker(entity, user_query, disclosed, rerank_client, budget)
    elif embed_client and disclosed:
        disclosed = await _rerank_with_embedding(entity, user_query, disclosed, embed_client, budget)

    final_count = sum(len(v) for v in disclosed.values())
    logger.info(
        "shrink_group_complete",
        entity_name=entity.name,
        group=group_label,
        final_count=final_count,
        budget=budget,
    )
    return disclosed


async def _rerank_properties_with_reranker(
    entity: TemporalEntity,
    query: str,
    disclosed: dict[str, list[dict[str, Any]]],
    rerank_client: RerankClient,
    top_m: int,
) -> dict[str, list[dict[str, Any]]]:
    """Precision rerank disclosed properties using reranker."""
    items: list[dict[str, Any]] = []
    for prop, values in disclosed.items():
        for val_info in values:
            value = val_info.get("value", "") if isinstance(val_info, dict) else str(val_info)
            items.append({"prop": prop, "value_info": val_info, "doc": f"{prop}: {value}"})

    _MAX_RERANK_ITEMS = 100
    if len(items) > _MAX_RERANK_ITEMS:
        items = items[:_MAX_RERANK_ITEMS]

    if not items or len(items) <= top_m:
        logger.info(
            "reranker_skip",
            entity_name=entity.name,
            item_count=len(items),
            top_m=top_m,
        )
        return disclosed

    docs = [item["doc"] for item in items]
    try:
        output_ids = await rerank(rerank_client, query, docs, min(top_m, len(docs)))
        logger.info(
            "reranker_complete",
            entity_name=entity.name,
            input_count=len(docs),
            output_count=len(output_ids),
        )
    except Exception as e:
        logger.warning("reranker_failed", entity_name=entity.name, error=str(e))
        return disclosed

    result: dict[str, list[dict[str, Any]]] = {}
    for idx in output_ids:
        if idx < len(items):
            item = items[idx]
            result.setdefault(item["prop"], []).append(item["value_info"])
    return result


async def _rerank_with_embedding(
    entity: TemporalEntity,
    query: str,
    disclosed: dict[str, list[dict[str, Any]]],
    embed_client: EmbedClient,
    top_m: int,
) -> dict[str, list[dict[str, Any]]]:
    """Fallback rerank using embedding cosine similarity."""
    import numpy as np

    if not isinstance(top_m, int):
        logger.warning("rerank_embedding_bad_top_m", actual_type=type(top_m).__name__)
        top_m = 10

    prop_texts: list[str] = []
    prop_keys: list[str] = []
    timestamps: list[str] = []

    for prop, values in disclosed.items():
        for item in values:
            ts_str = item.get("timestamp", "")
            value = item.get("value", "")
            prop_texts.append(f"property: {prop}, value: {str(value)}")
            prop_keys.append(prop)
            timestamps.append(ts_str)

    if not prop_texts:
        limited: dict[str, list[dict[str, Any]]] = {}
        total = 0
        for prop, values in disclosed.items():
            if total >= top_m:
                break
            limited[prop] = values[: top_m - total]
            total += len(limited[prop])
        return limited

    try:
        query_resp = await embed_client.embed(task="search.property_rerank", text=query)
        all_resp = await embed_client.embed(task="search.property_rerank", text=prop_texts)

        if not query_resp.embeddings or not query_resp.embeddings[0]:
            raise ValueError("Empty query embedding response")

        query_emb = query_resp.embeddings[0]

        similarities: list[tuple[int, float]] = []
        for i, emb in enumerate(all_resp.embeddings):
            if emb:
                a = np.array(query_emb)
                b = np.array(emb)
                sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
                similarities.append((i, sim))
            else:
                similarities.append((i, -1.0))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_m]]

        reranked: dict[str, list[dict[str, Any]]] = {}
        for i in top_indices:
            prop = prop_keys[i]
            if prop not in reranked:
                reranked[prop] = []
            prop_values = disclosed[prop]
            target_ts = timestamps[i]
            for item in prop_values:
                if item.get("timestamp") == target_ts:
                    reranked[prop].append({"value": item.get("value"), "timestamp": timestamps[i]})
                    break

        return reranked

    except Exception as e:
        logger.warning("rerank_embedding_failed", entity_name=entity.name, error=str(e))
        limited = {}
        total = 0
        for prop, values in disclosed.items():
            if total >= top_m:
                break
            limited[prop] = values[: top_m - total]
            total += len(limited[prop])
        return limited


def _build_disclosed_entity(
    entity: TemporalEntity,
    disclosed: dict[str, list[dict[str, Any]]],
) -> TemporalEntity:
    """Build a shrunk TemporalEntity copy retaining only disclosed properties."""
    disclosed_entity = TemporalEntity(
        entity_id=entity.entity_id,
        name=entity.name,
        entity_type=entity.entity_type,
        description=entity.description,
        record_time=entity.record_time,
        entity_manager=entity._entity_manager,
        auto_create_properties=False,
    )
    disclosed_entity._properties = {}
    disclosed_entity.edges = list(entity.edges)

    for prop_name, values in disclosed.items():
        entries: list[PropertyEntry] = []
        for item in values:
            prop_value = item.get("value")
            if prop_value is None:
                continue
            ts = item.get("timestamp")
            if ts is None:
                ts = entity.record_time
            elif not isinstance(ts, str):
                ts = str(ts) if ts else entity.record_time
            uid = item.get("uid", "")
            if not uid:
                from uuid import uuid4

                uid = uuid4().hex[:12]
            entries.append(PropertyEntry(timestamp=ts, value=prop_value, uid=uid))
        if entries:
            disclosed_entity._properties[prop_name] = entries

    return disclosed_entity
