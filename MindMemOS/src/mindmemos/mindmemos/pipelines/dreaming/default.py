"""Default dreaming pipeline for offline memory consolidation."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...components.activity import RecentActivityCollector
from ...components.dreaming.action_planning import action_planning_parser
from ...components.dreaming.relation_detection import (
    DetectedMemoryIssueGroup,
    relation_detection_parser,
)
from ...components.extractor.schema import property_relationships
from ...components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...config import DreamingConfig, TextProcessingConfig, get_config
from ...infra.kafka import get_producer
from ...llm import EmbedClient, LLMClient, get_embed_client, get_llm_client
from ...logging import get_logger, traced
from ...prompts.EN.dreaming.action_planning import ACTION_PLANNING_PROMPT
from ...prompts.EN.dreaming.relation_detection import RELATION_DETECTION_PROMPT
from ...typing import (
    MemoryDbDeleteCommand,
    MemoryDbMutationPlan,
    MemoryDbUpdateCommand,
    MemoryDbWritePlan,
)
from ...typing.activity import ActivityScope
from ...typing.algo import ConsolidationAction
from ...typing.memory import (
    REL_NEXT_IN_PROPERTY_TIMELINE,
    REL_RELATED_TO,
    GraphNodeRef,
    GraphRelationship,
    MemoryEdgeFilter,
    MemoryRequestContext,
    MemoryView,
    MemoryWrite,
    VectorWrite,
)
from ...typing.service import DreamingPipelineInput, DreamingPipelineResult
from ..base import MemoryDbPipelineMixin
from ..registry import register

MEMORY_DREAMING_TOPIC = "memory.dreaming"
DREAMING_MUTATION_SOURCE = "dreaming"
logger = get_logger(__name__)


@dataclass(frozen=True)
class ConsolidationScope:
    """A hot memory scope selected for consolidation."""

    entity_id: str | None
    property_name: str | None
    root_id: str | None
    score: int
    seed_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    add_record_ids: tuple[str, ...] = field(default_factory=tuple)
    relation: str | None = None
    subject: str | None = None
    graph_entity_id: str | None = None
    graph_entity_name: str | None = None
    primary_memory_id: str | None = None
    graph_source: str | None = None


@register(type="dreaming", name="default_dreaming")
class DefaultDreamingPipeline(MemoryDbPipelineMixin):
    """Consolidate hot memory scopes with two LLM calls per scope."""

    def __init__(
        self,
        *,
        dreaming_config: DreamingConfig | None = None,
        text_config: TextProcessingConfig | None = None,
        llm_client: LLMClient | None = None,
        embed_client: EmbedClient | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        activity_collector: RecentActivityCollector | None = None,
        consistency: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        algo_cfg = None if dreaming_config and text_config else get_config().algo_config
        cfg = text_config or algo_cfg.text_processing
        self._cfg = dreaming_config or algo_cfg.dreaming
        self._llm_client = llm_client or get_llm_client()
        self._embed_client = embed_client or get_embed_client()
        self._text_preprocessor = text_preprocessor or get_text_preprocessor(cfg)
        self._sparse_encoder = sparse_encoder or SparseVectorEncoder(cfg)
        self._activity_collector = activity_collector
        self._consistency = consistency or _default_consistency()

    # -- public API ----------------------------------------------------------

    async def dream(self, inp: DreamingPipelineInput, context: MemoryRequestContext) -> DreamingPipelineResult:
        await get_producer().send(
            MEMORY_DREAMING_TOPIC,
            value={
                "context": context.model_dump(mode="json"),
                "input": inp.model_dump(mode="json"),
                "submitted_at": datetime.now(UTC).isoformat(),
            },
            dispatch_key=f"{context.project_id}:{context.user_id}",
        )
        return DreamingPipelineResult(status="queued", message="consolidation queued")

    async def dream_sync(self, _inp: DreamingPipelineInput, context: MemoryRequestContext) -> DreamingPipelineResult:
        summary = await self._consolidate_memory(context)
        return DreamingPipelineResult(status="ok", message=f"consolidation complete: {summary}")

    # -- main orchestration --------------------------------------------------

    @traced("dreaming.consolidate")
    async def _consolidate_memory(self, context: MemoryRequestContext) -> dict[str, int]:
        run_id = str(uuid4())

        scopes = await self._select_hot_scopes(context)
        cluster_scopes = await self._dedupe_cluster_scopes(context, scopes)
        summary = {"scopes": len(scopes), "clusters": 0, "actions": 0, "add_records_done": 0}
        archived_memory_ids: set[str] = set()
        marked_add_record_ids: set[str] = set()
        state_lock = asyncio.Lock()
        sem = asyncio.Semaphore(max(1, int(self._cfg.concurrency or 1)))

        async def _process_scope(scope: ConsolidationScope, memories: list[MemoryView]) -> None:
            if len(memories) < self._cfg.min_cluster_size:
                async with state_lock:
                    summary["add_records_done"] += await self._mark_scope_add_records_done(
                        context, scope, run_id, marked_add_record_ids
                    )
                return

            async with state_lock:
                deterministic = await self._apply_exact_duplicate_archives(context, memories, archived_memory_ids)
                memories = [m for m in memories if m.memory_id not in archived_memory_ids]
            if len(memories) < self._cfg.min_cluster_size:
                async with state_lock:
                    summary["clusters"] += 1
                    summary["actions"] += deterministic
                    summary["add_records_done"] += await self._mark_scope_add_records_done(
                        context, scope, run_id, marked_add_record_ids
                    )
                return

            # ── LLM #1: relation detection ──
            issue_groups = await self._call_relation_detection_llm(self._build_cluster_context(scope, memories))

            # ── LLM #2: focused action planning for each detected issue group ──
            action_count = 0
            if issue_groups is not None:
                issue_group_list = self._aggregate_detected_issue_groups(issue_groups, memories, scope)
                for issue_group in issue_group_list:
                    actions = await self._call_action_planning_llm(
                        self._build_group_context(scope, memories, issue_group)
                    )
                    if actions is None:
                        continue
                    group_action_count = self._count_actions(actions)
                    if not group_action_count:
                        continue
                    async with state_lock:
                        await self._apply_actions(context, actions, memories, archived_memory_ids, run_id, scope)
                        summary["actions"] += group_action_count
                    action_count += group_action_count

            await self._mark_cluster_consolidated(context, memories, run_id, scope, action_count)
            async with state_lock:
                summary["add_records_done"] += await self._mark_scope_add_records_done(
                    context, scope, run_id, marked_add_record_ids
                )
                summary["clusters"] += 1
                summary["actions"] += deterministic

        async def _wrapped(scope: ConsolidationScope, memories: list[MemoryView]) -> None:
            async with sem:
                await _process_scope(scope, memories)

        await asyncio.gather(*[_wrapped(scope, memories) for scope, memories in cluster_scopes])
        return summary

    async def _dedupe_cluster_scopes(
        self,
        context: MemoryRequestContext,
        scopes: list[ConsolidationScope],
    ) -> list[tuple[ConsolidationScope, list[MemoryView]]]:
        unique: dict[tuple[str, ...], tuple[ConsolidationScope, list[MemoryView]]] = {}
        for scope in scopes:
            memories = self._dedupe_memories(await self._hydrate_scope_memories(context, scope))
            cluster_key = tuple(sorted(memory.memory_id for memory in memories if memory.memory_id))
            if not cluster_key:
                cluster_key = tuple(sorted(mid for mid in scope.seed_memory_ids if mid))
            existing = unique.get(cluster_key)
            if existing is None:
                unique[cluster_key] = (scope, memories)
                continue
            merged_scope = _merge_scope_add_records(existing[0], scope)
            unique[cluster_key] = (merged_scope, existing[1])
        return list(unique.values())

    # -- scope selection -----------------------------------------------------

    async def _select_hot_scopes(self, context: MemoryRequestContext) -> list[ConsolidationScope]:
        seed_add_records_by_memory_id: dict[str, set[str]] = {}
        bundle = await self._get_activity_collector().collect(
            ActivityScope(
                project_id=context.project_id,
                user_id=context.user_id or None,
                session_id=context.session_id or None,
                agent_id=context.agent_id or None,
                app_id=context.app_id or None,
            ),
            lookback=timedelta(days=self._cfg.lookback_days),
            window_end=datetime.now(UTC),
            max_records=_optional_positive_int(self._cfg.max_seed_memories),
        )
        pending_add_record_ids = await self._pending_consolidation_add_record_ids(
            context,
            [rid for written in bundle.written_memories for rid in written.add_record_ids],
        )
        for written in bundle.written_memories:
            if not written.memory_id:
                continue
            add_record_ids = [rid for rid in written.add_record_ids if rid in pending_add_record_ids]
            if not add_record_ids:
                continue
            seed_add_records_by_memory_id.setdefault(written.memory_id, set()).update(add_record_ids)

        scopes = await self._select_graph_seed_scopes(
            context, tuple(seed_add_records_by_memory_id), seed_add_records_by_memory_id
        )
        scopes.sort(key=lambda s: s.score, reverse=True)
        max_scopes = _optional_positive_int(self._cfg.max_scopes_per_run)
        return scopes[:max_scopes] if max_scopes is not None else scopes

    async def _pending_consolidation_add_record_ids(
        self,
        context: MemoryRequestContext,
        add_record_ids: list[str],
    ) -> set[str]:
        unique_ids = list(dict.fromkeys(rid for rid in add_record_ids if rid))
        if not unique_ids or not hasattr(self.db_reader, "get_add_records_by_ids"):
            return set(unique_ids)
        records = await self.db_reader.get_add_records_by_ids(context, unique_ids)
        done_ids = {
            record.point_id
            for record in records
            if str((record.payload or {}).get("consolidation_status") or "").lower() == "done"
        }
        found_ids = {record.point_id for record in records}
        missing_ids = set(unique_ids) - found_ids
        return (set(unique_ids) - done_ids) | missing_ids

    def _get_activity_collector(self) -> RecentActivityCollector:
        if self._activity_collector is None:
            self._activity_collector = RecentActivityCollector(self.db_reader._clients.qdrant)
        return self._activity_collector

    async def _select_graph_seed_scopes(
        self,
        context: MemoryRequestContext,
        seed_memory_ids: tuple[str, ...],
        seed_add_records_by_memory_id: dict[str, set[str]],
    ) -> list[ConsolidationScope]:
        if not seed_memory_ids or not hasattr(self.db_reader, "list_memory_neighbor_scopes"):
            return []

        batch_size = self._cfg.scope_batch_size
        seed_list = list(seed_memory_ids)
        all_neighbor_scopes: list = []

        for batch_start in range(0, len(seed_list), batch_size):
            batch = seed_list[batch_start:batch_start + batch_size]
            try:
                batch_scopes = await self.db_reader.list_memory_neighbor_scopes(
                    context,
                    batch,
                    edge_filter=MemoryEdgeFilter(rel_types=("RELATED_TO",)),
                    limit_per_entity=self._cfg.max_memories_per_scope,
                    limit_direct_per_memory=self._cfg.max_memories_per_scope,
                    attach_direct_neighbors_to_entity_scopes=True,
                )
                all_neighbor_scopes.extend(batch_scopes)
            except Exception:
                logger.warning(
                    "dreaming graph scope recall batch failed",
                    project_id=context.project_id,
                    batch_size=len(batch),
                    batch_start=batch_start,
                    exc_info=True,
                )
                continue

        if not all_neighbor_scopes:
            return []

        seed_data: dict[str, dict[str, Any]] = {}
        for graph_scope in all_neighbor_scopes:
            sid = graph_scope.seed_memory_id
            if not sid:
                continue
            data = seed_data.setdefault(
                sid,
                {"memory_ids": set(), "entity_ids": [], "entity_types": [], "entity_names": [], "sources": []},
            )
            data["memory_ids"].update(graph_scope.memory_ids)
            if graph_scope.source and graph_scope.source not in data["sources"]:
                data["sources"].append(graph_scope.source)
            if graph_scope.entity_id and graph_scope.entity_id not in data["entity_ids"]:
                data["entity_ids"].append(graph_scope.entity_id)
                data["entity_types"].append(graph_scope.entity_type or "")
                data["entity_names"].append(graph_scope.entity_name or "")

        scopes: list[ConsolidationScope] = []
        for sid, data in seed_data.items():
            unique_ids = tuple(dict.fromkeys([sid, *data["memory_ids"]]))
            if len(unique_ids) < self._cfg.min_scope_updates:
                continue
            score = len(unique_ids) + 20
            if any(mid != sid for mid in unique_ids):
                score += 10
            entity_str = ",".join(data["entity_ids"][:5])
            scopes.append(
                ConsolidationScope(
                    entity_id=f"graph:{entity_str}" if entity_str else "graph:",
                    property_name=",".join(data["entity_types"][:5]) if data["entity_types"] else None,
                    root_id=None,
                    score=score,
                    seed_memory_ids=unique_ids,
                    add_record_ids=tuple(sorted(seed_add_records_by_memory_id.get(sid, set()))),
                    graph_entity_id=entity_str if entity_str else None,
                    graph_entity_name=",".join(data["entity_names"][:5]) if data["entity_names"] else None,
                    primary_memory_id=sid,
                    graph_source=",".join(data["sources"][:5]) if data["sources"] else None,
                )
            )
        return scopes

    # -- memory recall --------------------------------------------------------

    async def _hydrate_scope_memories(
        self, context: MemoryRequestContext, scope: ConsolidationScope
    ) -> list[MemoryView]:
        recalled: dict[str, MemoryView] = {}
        if scope.seed_memory_ids and hasattr(self.db_reader, "get_memories"):
            for memory in await self.db_reader.get_memories(context, list(scope.seed_memory_ids)):
                if memory.status == "active":
                    recalled[memory.memory_id] = memory
        primary_id = scope.primary_memory_id or (scope.seed_memory_ids[0] if scope.seed_memory_ids else None)
        return sorted(
            recalled.values(),
            key=lambda m: (m.memory_id == primary_id, m.memory_id in scope.seed_memory_ids, _memory_effective_time(m)),
            reverse=True,
        )[: self._cfg.max_memories_per_scope]

    def _dedupe_memories(self, memories: list[MemoryView]) -> list[MemoryView]:
        seen: set[str] = set()
        result: list[MemoryView] = []
        for m in memories:
            if m.memory_id not in seen:
                seen.add(m.memory_id)
                result.append(m)
        return result

    # -- exact-duplicate removal ----------------------------------------------

    async def _apply_exact_duplicate_archives(
        self,
        context: MemoryRequestContext,
        memories: list[MemoryView],
        archived_memory_ids: set[str],
    ) -> int:
        by_hash: dict[str, list[MemoryView]] = {}
        for m in memories:
            h = m.metadata.get("content_hash")
            if h:
                by_hash.setdefault(str(h), []).append(m)
        archived = 0
        for duplicates in by_hash.values():
            if len(duplicates) < 2:
                continue
            duplicates.sort(key=lambda m: m.update_at or m.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)
            for dup in duplicates[1:]:
                if dup.memory_id in archived_memory_ids:
                    continue
                await self._apply_memory_deletes(
                    context,
                    [
                        MemoryDbDeleteCommand(
                            memory_id=dup.memory_id,
                            reason=f"duplicate_of:{duplicates[0].memory_id}",
                            consistency=self._consistency,
                        )
                    ],
                )
                archived_memory_ids.add(dup.memory_id)
                archived += 1
        return archived

    async def _apply_memory_updates(
        self,
        context: MemoryRequestContext,
        commands: list[MemoryDbUpdateCommand],
    ) -> None:
        if not commands:
            return
        await self.db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan(memory_updates=commands),
            consistency=self._consistency,
        )

    async def _apply_memory_deletes(
        self,
        context: MemoryRequestContext,
        commands: list[MemoryDbDeleteCommand],
    ) -> None:
        if not commands:
            return
        await self.db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan(memory_deletes=commands),
            consistency=self._consistency,
        )

    async def _apply_write_plan(self, context: MemoryRequestContext, plan: MemoryDbWritePlan) -> None:
        mutation_plan = MemoryDbMutationPlan.from_write_plan(plan)
        if not mutation_plan.has_writes():
            return
        await self.db_writer.apply_mutation_plan(
            context,
            mutation_plan,
            consistency=self._consistency,
        )

    # -- LLM prompt builders -------------------------------------------------

    def _build_cluster_context(self, scope: ConsolidationScope, memories: list[MemoryView]) -> str:
        lines = [
            f"Scope entity_id: {scope.entity_id or 'none'}",
            f"Scope property_name: {scope.property_name or 'none'}",
            f"Scope graph_entity_id: {scope.graph_entity_id or 'none'}",
            f"Scope graph_entity_name: {scope.graph_entity_name or 'none'}",
            f"Primary recent memory_id: {scope.primary_memory_id or 'none'}",
            f"Scope hot score: {scope.score}",
            "",
        ]
        primary_id = scope.primary_memory_id or (scope.seed_memory_ids[0] if scope.seed_memory_ids else None)
        for i, mem in enumerate(memories, 1):
            role = "PRIMARY_RECENT_MEMORY" if mem.memory_id == primary_id else "RETRIEVED_NEIGHBOR_MEMORY"
            lines.extend(
                [
                    f"[{i}] role={role} memory_id={mem.memory_id}",
                    f"    content: {mem.content}",
                    f"    mem_type: {mem.mem_type}",
                    f"    entity_id: {mem.entity_id or ''}",
                    f"    entity_type: {mem.entity_type or ''}",
                    f"    property_name: {mem.property_name or ''}",
                    f"    root_id: {mem.root_id}",
                    f"    parent_ids: {mem.parent_ids}",
                    f"    effective_time: {_format_memory_time(_memory_effective_time(mem))}",
                    f"    validate_from: {mem.validate_from}",
                    f"    validate_to: {mem.validate_to}",
                    f"    source_timestamp_ms: {mem.metadata.get('source_timestamp_ms')}",
                    f"    created_at: {mem.created_at}",
                    f"    update_at: {mem.update_at}",
                    f"    metadata: {mem.metadata}",
                    "",
                ]
            )
        return "\n".join(lines)

    def _build_group_context(
        self,
        scope: ConsolidationScope,
        memories: list[MemoryView],
        issue_group: DetectedMemoryIssueGroup,
    ) -> str:
        mem_by_id = {m.memory_id: m for m in memories}
        issue_memory_ids = [mid for mid in issue_group.memory_ids if mid in mem_by_id]
        issue_memories = [mem_by_id[mid] for mid in issue_memory_ids]
        sorted_memories = sorted(issue_memories, key=_memory_effective_time, reverse=True)
        latest_id = sorted_memories[0].memory_id if sorted_memories else ""
        lines: list[str] = [
            "Focused memory issue group:",
            f"  scope_entity: {scope.graph_entity_name or scope.entity_id or 'none'}",
            f"  issue_type: {issue_group.issue_type}",
            f"  subject_hint: {issue_group.subject_hint or ''}",
            f"  predicate_hint: {issue_group.predicate_hint or ''}",
            f"  confidence: {issue_group.confidence}",
            f"  reason: {issue_group.reason}",
            f"  latest_memory_id_by_effective_time: {latest_id}",
            "  memories:",
        ]
        for mem in sorted_memories:
            lines.extend(
                [
                    f"    - memory_id: {mem.memory_id}",
                    f"      content: {mem.content}",
                    f"      effective_time: {_format_memory_time(_memory_effective_time(mem))}",
                    f"      value_hint: {issue_group.value_hints.get(mem.memory_id, '')}",
                    f"      related_memory_ids: {mem.metadata.get('related_memory_ids') or []}",
                    f"      has_update_intent: {_has_seed_update_intent(mem)}",
                    f"      extractor_confidence: {mem.metadata.get('extractor_confidence')}",
                ]
            )
        lines.append("")
        return "\n".join(lines)

    # -- LLM #1: relation detection ------------------------------------------

    async def _call_relation_detection_llm(self, context_str: str) -> list[DetectedMemoryIssueGroup] | None:
        prompt = RELATION_DETECTION_PROMPT.format(context=context_str)
        try:
            kwargs: dict[str, Any] = {}
            if self._cfg.consolidation_model:
                kwargs["model"] = self._cfg.consolidation_model
            result = await self._llm_client.chat(
                task="memory_relation_detection",
                messages=[{"role": "system", "content": prompt}],
                format_parser=relation_detection_parser,
                **kwargs,
            )
            parsed = result.parsed
            if hasattr(parsed, "issue_groups"):
                return parsed.issue_groups
            if hasattr(parsed, "candidates"):
                groups: list[DetectedMemoryIssueGroup] = []
                for candidate in parsed.candidates:
                    if getattr(candidate, "candidate_type", None) != "needs_consolidation":
                        continue
                    primary_id = getattr(candidate, "primary_memory_id", "")
                    neighbor_id = getattr(candidate, "neighbor_memory_id", "")
                    groups.append(
                        DetectedMemoryIssueGroup(
                            issue_type="ambiguous",
                            memory_ids=[primary_id, neighbor_id],
                            subject_hint=getattr(candidate, "subject_hint", None),
                            predicate_hint=getattr(candidate, "predicate_hint", None),
                            value_hints={
                                primary_id: getattr(candidate, "primary_value_hint", "") or "",
                                neighbor_id: getattr(candidate, "neighbor_value_hint", "") or "",
                            },
                            confidence=getattr(candidate, "confidence", "medium") or "medium",
                            reason=getattr(candidate, "reason", ""),
                        )
                    )
                return groups
            return None
        except Exception:
            logger.warning("dreaming relation detection llm failed", exc_info=True)
            return None

    def _aggregate_detected_issue_groups(
        self,
        issue_groups: list[DetectedMemoryIssueGroup],
        memories: list[MemoryView],
        scope: ConsolidationScope,
    ) -> list[DetectedMemoryIssueGroup]:
        mem_ids = {m.memory_id for m in memories}
        mem_by_id = {m.memory_id: m for m in memories}
        primary_id = scope.primary_memory_id
        dedup: dict[tuple[str, tuple[str, ...]], DetectedMemoryIssueGroup] = {}
        for group in issue_groups:
            if not isinstance(group, DetectedMemoryIssueGroup):
                group = DetectedMemoryIssueGroup(
                    issue_type=getattr(group, "issue_type", "ambiguous") or "ambiguous",
                    memory_ids=list(getattr(group, "memory_ids", []) or []),
                    subject_hint=getattr(group, "subject_hint", None),
                    predicate_hint=getattr(group, "predicate_hint", None),
                    value_hints=dict(getattr(group, "value_hints", {}) or {}),
                    confidence=getattr(group, "confidence", "medium") or "medium",
                    reason=getattr(group, "reason", ""),
                )
            valid_ids = [mid for mid in group.memory_ids if mid in mem_ids]
            if len(valid_ids) < 2:
                continue
            # Keep the first-stage detector seed-centric: the current scope's
            # primary memory must participate in the focused problem group.
            if primary_id and primary_id not in valid_ids:
                continue
            # Subject consistency guard: reject groups where memories have
            # different subject entities (entities[0]). This catches cross-subject
            # groupings (e.g. different people sharing the same country, or different
            # songs sharing the same performer) that the LLM misclassified.
            subject_entities: set[str] = set()
            for mid in valid_ids:
                mem = mem_by_id.get(mid)
                if mem is not None:
                    raw = mem.metadata.get("entities", [])
                    if raw:
                        subject_entities.add(raw[0])
            if len(subject_entities) > 1:
                continue
            key = (group.issue_type, tuple(sorted(valid_ids)))
            normalized = group.model_copy(update={"memory_ids": valid_ids})
            existing = dedup.get(key)
            if existing is None or _confidence_rank(normalized.confidence) > _confidence_rank(existing.confidence):
                dedup[key] = normalized
        return list(dedup.values())

    # -- LLM #2: action planning ---------------------------------------------

    async def _call_action_planning_llm(self, group_context: str) -> ConsolidationAction | None:
        prompt = ACTION_PLANNING_PROMPT.format(groups=group_context)
        _parse_attempts = 0
        _max_parse_attempts = 5

        def _bounded_parser(content: str) -> ConsolidationAction:
            nonlocal _parse_attempts
            _parse_attempts += 1
            if _parse_attempts > _max_parse_attempts:
                return ConsolidationAction()
            return action_planning_parser(content)

        try:
            kwargs: dict[str, Any] = {}
            if self._cfg.consolidation_model:
                kwargs["model"] = self._cfg.consolidation_model
            result = await self._llm_client.chat(
                task="memory_action_planning",
                messages=[{"role": "system", "content": prompt}],
                format_parser=_bounded_parser,
                **kwargs,
            )
            debug_dir = os.getenv("MINDMEMOS_DREAMING_ACTION_DEBUG_DIR")
            if debug_dir:
                try:
                    Path(debug_dir).mkdir(parents=True, exist_ok=True)
                    debug_path = Path(debug_dir) / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex}.json"
                    debug_payload = {
                        "task": "memory_action_planning",
                        "prompt": prompt,
                        "raw_response": result.content,
                        "parsed": result.parsed.model_dump(mode="json") if result.parsed is not None else None,
                    }
                    debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    logger.warning("dreaming action debug logging failed", exc_info=True)
            return result.parsed
        except Exception:
            logger.warning("dreaming action planning llm failed", exc_info=True)
            return None

    # -- apply actions --------------------------------------------------------

    async def _apply_actions(
        self,
        context,
        actions,
        cluster_memories,
        archived_memory_ids,
        run_id,
        scope,
    ) -> None:
        now = datetime.now(UTC)
        mem_by_id = {m.memory_id: m for m in cluster_memories}
        created_by_source_set: dict[tuple[str, ...], str] = {}

        creates = [self._memory_from_create(context, c, now, mem_by_id, run_id, scope) for c in actions.creates]
        merge_creates = [self._memory_from_merge(context, m, now, mem_by_id, run_id, scope) for m in actions.merges]
        all_creates = [m for m in [*creates, *merge_creates] if m is not None]

        for merge, memory in zip(actions.merges, merge_creates, strict=False):
            if memory is not None:
                created_by_source_set[tuple(sorted(merge.source_memory_ids))] = memory.memory_id

        if all_creates:
            await self._write_new_memories(context, all_creates, cluster_memories)

        for update in actions.updates:
            if update.memory_id not in mem_by_id:
                continue
            metadata_patch = dict(update.metadata_patch)
            if update.quality_signal:
                metadata_patch["quality_signal"] = update.quality_signal
            if update.reason:
                metadata_patch["quality_reason"] = update.reason
            metadata_patch.update(_dreaming_metadata_patch(run_id, scope, now, action_count=1))
            await self._apply_memory_updates(
                context,
                [
                    MemoryDbUpdateCommand(
                        memory_id=update.memory_id,
                        content=update.content,
                        reinforcement_count=update.reinforcement_count,
                        metadata_patch=metadata_patch,
                        consistency=self._consistency,
                    )
                ],
            )

        for merge in actions.merges:
            replacement_id = created_by_source_set.get(tuple(sorted(merge.source_memory_ids)))
            for source_id in merge.source_memory_ids:
                if source_id not in mem_by_id or source_id in archived_memory_ids:
                    continue
                await self._apply_memory_deletes(
                    context,
                    [
                        MemoryDbDeleteCommand(
                            memory_id=source_id,
                            reason=f"merged_into:{replacement_id}" if replacement_id else "merged",
                            consistency=self._consistency,
                        )
                    ],
                )
                archived_memory_ids.add(source_id)

        for archive in actions.archives:
            if archive.memory_id not in mem_by_id or archive.memory_id in archived_memory_ids:
                continue
            reason = archive.reason or "consolidated"
            if archive.replacement_memory_id:
                reason = f"{reason};replacement:{archive.replacement_memory_id}"
            await self._apply_memory_deletes(
                context,
                [MemoryDbDeleteCommand(memory_id=archive.memory_id, reason=reason, consistency=self._consistency)],
            )
            archived_memory_ids.add(archive.memory_id)

        link_relationships = [self._relationship_from_link(context, link, mem_by_id) for link in actions.links]
        link_relationships = [r for r in link_relationships if r is not None]
        if link_relationships:
            await self._apply_write_plan(context, MemoryDbWritePlan(relationships=link_relationships))

    # -- marking helpers ------------------------------------------------------

    async def _mark_cluster_consolidated(self, context, memories, run_id, scope, action_count) -> None:
        now = datetime.now(UTC)
        patch = _dreaming_metadata_patch(run_id, scope, now, action_count=action_count)
        for mem in memories:
            await self._apply_memory_updates(
                context,
                [
                    MemoryDbUpdateCommand(
                        memory_id=mem.memory_id,
                        metadata_patch=patch,
                        consistency=self._consistency,
                    )
                ],
            )

    async def _mark_scope_add_records_done(self, context, scope, run_id, marked_add_record_ids) -> int:
        if not scope.add_record_ids:
            return 0
        now = datetime.now(UTC)
        updated = 0
        for aid in scope.add_record_ids:
            if aid in marked_add_record_ids:
                continue
            await self.db_writer.patch_add_record(
                context,
                aid,
                {
                    "consolidation_status": "done",
                    "consolidated_at": now,
                    "consolidation_run_id": run_id,
                },
            )
            marked_add_record_ids.add(aid)
            updated += 1
        return updated

    # -- memory / entity / vector helpers -------------------------------------

    def _memory_from_create(self, context, create, now, mem_by_id, run_id, scope) -> MemoryWrite | None:
        if not create.content.strip():
            return None
        evidence = [mem_by_id[mid] for mid in create.evidence_memory_ids if mid in mem_by_id]
        entity_id = create.entity_id or _most_common([m.entity_id for m in evidence])
        property_name = create.property_name or _most_common([m.property_name for m in evidence])
        entity_type = create.entity_type or _most_common([m.entity_type for m in evidence])
        root_ids = create.root_id or _merged_root_ids(evidence)
        memory_id = str(uuid4())
        return MemoryWrite(
            memory_id=memory_id,
            account_id=_most_common([m.account_id for m in evidence]) or context.account_id,
            project_id=context.project_id,
            api_key_uuid=_most_common([m.api_key_uuid for m in evidence]) or context.api_key_uuid,
            user_id=_most_common([m.user_id for m in evidence]) or context.user_id,
            app_id=_most_common([m.app_id for m in evidence]) or context.app_id,
            session_id=_most_common([m.session_id for m in evidence]) or context.session_id,
            agent_id=_most_common([m.agent_id for m in evidence]) or context.agent_id,
            request_id=context.request_id,
            content=create.content.strip(),
            mem_type=create.mem_type,
            mem_extract_type="dreaming_create",
            mem_extract_version="dreaming_v2",
            metadata={
                **dict(create.metadata),
                "dreaming_reason": create.reason,
                "evidence_memory_ids": list(create.evidence_memory_ids),
                **_dreaming_metadata_patch(run_id, scope, now, action_count=1),
            },
            validate_from=_latest_validate_from(evidence),
            created_at=now,
            parent_ids=list(create.parent_ids or create.evidence_memory_ids),
            root_id=root_ids or [memory_id],
            property_name=property_name,
            entity_id=entity_id,
            entity_type=entity_type,
        )

    def _memory_from_merge(self, context, merge, now, mem_by_id, run_id, scope) -> MemoryWrite | None:
        if not merge.target_content.strip() or len(merge.source_memory_ids) < 2:
            return None
        sources = [mem_by_id[mid] for mid in merge.source_memory_ids if mid in mem_by_id]
        if len(sources) < 2:
            return None
        memory_id = str(uuid4())
        return MemoryWrite(
            memory_id=memory_id,
            account_id=_most_common([m.account_id for m in sources]) or context.account_id,
            project_id=context.project_id,
            api_key_uuid=_most_common([m.api_key_uuid for m in sources]) or context.api_key_uuid,
            user_id=_most_common([m.user_id for m in sources]) or context.user_id,
            app_id=_most_common([m.app_id for m in sources]) or context.app_id,
            session_id=_most_common([m.session_id for m in sources]) or context.session_id,
            agent_id=_most_common([m.agent_id for m in sources]) or context.agent_id,
            request_id=context.request_id,
            content=merge.target_content.strip(),
            mem_type="fact",
            mem_extract_type="dreaming_merge",
            mem_extract_version="dreaming_v2",
            metadata={
                "merge_reason": merge.merge_reason,
                "source_memory_ids": list(merge.source_memory_ids),
                **_dreaming_metadata_patch(run_id, scope, now, action_count=1),
            },
            validate_from=_latest_validate_from(sources),
            created_at=now,
            parent_ids=list(merge.source_memory_ids),
            root_id=merge.target_root_id or _merged_root_ids(sources) or [memory_id],
            property_name=merge.target_property_name or _most_common([m.property_name for m in sources]),
            entity_id=merge.target_entity_id or _most_common([m.entity_id for m in sources]),
            entity_type=merge.target_entity_type or _most_common([m.entity_type for m in sources]),
        )

    async def _write_new_memories(self, context, memories, cluster_memories) -> None:
        dense_vectors = await self._embed_texts("memory.dreaming.create", [m.content for m in memories])
        vectors = self._memory_vectors(memories, dense_vectors)
        relationships: list[GraphRelationship] = []
        existing_by_scope = _latest_by_scope(cluster_memories)
        for memory in memories:
            if memory.entity_id:
                relationships.extend(property_relationships(context.project_id, memory.entity_id, memory))
            previous = existing_by_scope.get((memory.entity_id, memory.property_name))
            if previous:
                relationships.append(
                    GraphRelationship(
                        source=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=previous.memory_id),
                        target=GraphNodeRef(kind="Memory", project_id=context.project_id, node_id=memory.memory_id),
                        rel_type=REL_NEXT_IN_PROPERTY_TIMELINE,
                        project_id=context.project_id,
                        entity_id=memory.entity_id,
                        property_name=memory.property_name,
                        metadata={"source": "dreaming"},
                    )
                )
            for source_id in memory.parent_ids:
                if source_id:
                    relationships.append(
                        GraphRelationship(
                            source=GraphNodeRef(
                                kind="Memory",
                                project_id=context.project_id,
                                node_id=memory.memory_id,
                            ),
                            target=GraphNodeRef(
                                kind="Memory",
                                project_id=context.project_id,
                                node_id=source_id,
                            ),
                            rel_type=REL_RELATED_TO,
                            project_id=context.project_id,
                            relation_type="dreaming_evidence",
                            metadata={"source": "dreaming"},
                        )
                    )
        await self._apply_write_plan(
            context,
            MemoryDbWritePlan(memories=memories, vectors=vectors, relationships=relationships),
        )

    async def _embed_texts(self, task: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._embed_client.embed(task=task, text=texts)
        return response.embeddings

    def _memory_vectors(self, memories: list[MemoryWrite], dense_vectors: list[list[float]]) -> list[VectorWrite]:
        vectors: list[VectorWrite] = []
        for i, memory in enumerate(memories):
            dense = dense_vectors[i] if i < len(dense_vectors) else None
            preprocessed = self._text_preprocessor.preprocess_text(memory.content, include_entities=False)
            sparse = self._sparse_encoder.encode_document(preprocessed.tokens)
            memory.metadata.setdefault("content_hash", preprocessed.content_hash)
            memory.metadata.setdefault("bm25_text", preprocessed.bm25_text)
            memory.metadata.setdefault("tokens", list(preprocessed.tokens))
            memory.metadata.setdefault("lang", preprocessed.lang)
            vectors.append(
                VectorWrite(
                    memory_id=memory.memory_id,
                    semantic_vector=dense,
                    bm25_indices=list(sparse.indices),
                    bm25_values=list(sparse.values),
                )
            )
        return vectors

    def _relationship_from_link(self, context, link, mem_by_id) -> GraphRelationship | None:
        # Link actions are limited to existing cluster memories. New dreaming
        # memories get evidence/timeline edges from _write_new_memories().
        if link.source_kind == "Memory" and link.source_id not in mem_by_id:
            return None
        if link.target_kind == "Memory" and link.target_id not in mem_by_id:
            return None
        return GraphRelationship(
            source=GraphNodeRef(kind=link.source_kind, project_id=context.project_id, node_id=link.source_id),
            target=GraphNodeRef(kind=link.target_kind, project_id=context.project_id, node_id=link.target_id),
            rel_type=REL_RELATED_TO,
            project_id=context.project_id,
            relation_type=link.relation_type,
            property_name=link.property_name,
            metadata={"reason": link.reason, **dict(link.metadata), "source": "dreaming"},
        )

    def _count_actions(self, actions: ConsolidationAction) -> int:
        return (
            len(actions.creates)
            + len(actions.updates)
            + len(actions.merges)
            + len(actions.archives)
            + len(actions.links)
        )


# =============================================================================
# module-level helpers
# =============================================================================


def _memory_effective_time(memory: MemoryView) -> datetime:
    if memory.validate_from is not None:
        return memory.validate_from
    millis = memory.metadata.get("source_timestamp_ms")
    if isinstance(millis, int | float):
        return datetime.fromtimestamp(millis / 1000, tz=UTC)
    return memory.created_at or memory.update_at or datetime.min.replace(tzinfo=UTC)


def _format_memory_time(value: datetime | None) -> str:
    return "" if value is None else value.isoformat()


def _confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(confidence, 1)


def _has_seed_update_intent(memory: MemoryView) -> bool:
    text = " ".join(
        str(memory.metadata.get(key) or "") for key in ("extractor_reason", "evidence_summary", "planner_reason")
    ).lower()
    return bool(re.search(r"\b(update|revise|correct|replace|supersede|consolidat|change|contradict)\b", text))


def _merged_root_ids(memories: list[MemoryView]) -> list[str]:
    roots: list[str] = []
    for m in memories:
        for rid in m.root_id:
            if rid and rid not in roots:
                roots.append(rid)
    if not roots:
        roots = [m.memory_id for m in memories if m.memory_id]
    return roots


def _latest_validate_from(memories: list[MemoryView]) -> datetime | None:
    values = [m.validate_from for m in memories if m.validate_from is not None]
    return max(values) if values else None


def _most_common(values: list[str | None]) -> str | None:
    non_empty = [v for v in values if v]
    return Counter(non_empty).most_common(1)[0][0] if non_empty else None


def _latest_by_scope(memories: list[MemoryView]) -> dict[tuple[str | None, str | None], MemoryView]:
    latest: dict[tuple[str | None, str | None], MemoryView] = {}
    for m in memories:
        key = (m.entity_id, m.property_name)
        existing = latest.get(key)
        if existing is None:
            latest[key] = m
            continue
        et = existing.update_at or existing.created_at or datetime.min.replace(tzinfo=UTC)
        mt = m.update_at or m.created_at or datetime.min.replace(tzinfo=UTC)
        if mt > et:
            latest[key] = m
    return latest


def _merge_scope_add_records(primary: ConsolidationScope, duplicate: ConsolidationScope) -> ConsolidationScope:
    add_record_ids = tuple(sorted(set(primary.add_record_ids) | set(duplicate.add_record_ids)))
    seed_memory_ids = tuple(dict.fromkeys([*primary.seed_memory_ids, *duplicate.seed_memory_ids]))
    return primary.__class__(
        entity_id=primary.entity_id,
        property_name=primary.property_name,
        root_id=primary.root_id,
        score=max(primary.score, duplicate.score),
        seed_memory_ids=seed_memory_ids,
        add_record_ids=add_record_ids,
        relation=primary.relation,
        subject=primary.subject,
        graph_entity_id=primary.graph_entity_id,
        graph_entity_name=primary.graph_entity_name,
        primary_memory_id=primary.primary_memory_id,
        graph_source=primary.graph_source,
    )


def _dreaming_metadata_patch(
    run_id: str, scope: ConsolidationScope, now: datetime, *, action_count: int
) -> dict[str, Any]:
    return {
        "last_updated_by": DREAMING_MUTATION_SOURCE,
        "dreaming": {
            "last_consolidated_at": now.isoformat(),
            "last_run_id": run_id,
            "last_action_count": action_count,
            "scope": {
                "entity_id": scope.entity_id,
                "property_name": scope.property_name,
                "root_id": scope.root_id,
                "relation": scope.relation,
                "subject": scope.subject,
                "graph_entity_id": scope.graph_entity_id,
                "graph_entity_name": scope.graph_entity_name,
                "primary_memory_id": scope.primary_memory_id,
                "graph_source": scope.graph_source,
            },
        },
    }


def _default_consistency() -> str:
    value = get_config().database.default_consistency
    return value if value in {"fast", "strong"} else "fast"


def _optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


__all__ = ["DefaultDreamingPipeline", "MEMORY_DREAMING_TOPIC"]
