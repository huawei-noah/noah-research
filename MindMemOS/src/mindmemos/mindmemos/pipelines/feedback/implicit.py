"""Implicit feedback material collection.

This module only prepares session-level context for later implicit feedback
analysis. It does not classify feedback signals or execute memory actions.
"""

from __future__ import annotations

import time
from datetime import timedelta
from itertools import groupby
from typing import Any

from ...logging import get_logger
from ...components.activity import RecentActivityCollector
from ...components.feedback import (
    FeedbackRoundCompactor,
    ImplicitFeedbackActionPlanner,
    ImplicitFeedbackQueryRewriter,
    ImplicitFeedbackSignalDetector,
)
from ...infra.db import DatabaseClients, match_value, resolve_database_clients
from ...typing import (
    ActivityMessage,
    ActivityScope,
    AddActivityEvent,
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    FieldCondition,
    ImplicitFeedbackRound,
    ImplicitFeedbackSessionMaterial,
    MemoryRequestContext,
    MemorySearchItem,
    MemoryView,
    SearchActivityEvent,
    SearchFilter,
    SearchPipelineInput,
)
from ..memory_db import MemoryDbReader, MemoryDbWriter
from ..registry import create_pipeline
from ..search import SearchPipeline
from .executor import FeedbackActionExecutor


class ImplicitFeedbackHandler:
    """Run implicit feedback collection and signal detection."""

    def __init__(
        self,
        *,
        collector: "ImplicitFeedbackRecordCollector | None" = None,
        signal_detector: ImplicitFeedbackSignalDetector | None = None,
        action_planner: ImplicitFeedbackActionPlanner | None = None,
        executor: FeedbackActionExecutor | None = None,
    ) -> None:
        self._collector = collector or ImplicitFeedbackRecordCollector()
        self._signal_detector = signal_detector or ImplicitFeedbackSignalDetector()
        self._action_planner = action_planner or ImplicitFeedbackActionPlanner()
        self._executor = executor or FeedbackActionExecutor()

    async def run(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Collect session materials and detect negative feedback rounds."""

        t0 = time.monotonic()
        logger = get_logger(__name__)

        t_collect = time.monotonic()
        sessions = await self._collector.collect(context)
        collect_elapsed = time.monotonic() - t_collect
        logger.info(
            "feedback.implicit.collect",
            sessions=len(sessions),
            elapsed=round(collect_elapsed, 2),
            user_id=context.user_id,
        )

        signal_count = 0
        actions = []
        for idx, material in enumerate(sessions):
            t_detect = time.monotonic()
            result = await self._signal_detector.detect(material)
            detect_elapsed = time.monotonic() - t_detect
            signal_count += len(result.signals)
            logger.info(
                "feedback.implicit.detect",
                session=f"{idx+1}/{len(sessions)}",
                rounds=len(material.rounds),
                memories=len(material.memories),
                signals=len(result.signals),
                elapsed=round(detect_elapsed, 2),
            )

            valid_signals = [signal for signal in result.signals if 0 <= signal.round_index < len(material.rounds)]
            valid_signals.sort(key=lambda item: item.round_index)
            for round_index, grouped in groupby(valid_signals, key=lambda item: item.round_index):
                signals = list(grouped)
                if not signals:
                    continue
                round_ = material.rounds[round_index]
                t_plan = time.monotonic()
                planned = await self._action_planner.plan(
                    round_=round_,
                    signals=signals,
                    memories=material.memories,
                )
                plan_elapsed = time.monotonic() - t_plan
                logger.info(
                    "feedback.implicit.plan",
                    round_index=round_index,
                    signals=len(signals),
                    actions=len(planned),
                    elapsed=round(plan_elapsed, 2),
                )
                actions.extend(await self._executor.execute(planned, context))
            await self._collector.mark_feedback_processed(context, material.source_add_record_ids)

        total_elapsed = time.monotonic() - t0
        logger.info(
            "feedback.implicit.done",
            signals=signal_count,
            actions=len(actions),
            sessions=len(sessions),
            elapsed=round(total_elapsed, 2),
        )
        return FeedbackPipelineResult(
            status="ok",
            message=f"processed {signal_count} implicit feedback signals in {len(sessions)} sessions",
            actions=actions,
        )


class ImplicitFeedbackRecordCollector:
    """Collect add/search records for implicit feedback by session."""

    def __init__(
        self,
        *,
        window_days: int = 3,
        page_size: int = 100,
        memory_reader: MemoryDbReader | None = None,
        memory_writer: MemoryDbWriter | None = None,
        activity_collector: RecentActivityCollector | None = None,
        clients: DatabaseClients | None = None,
        query_rewriter: "ImplicitFeedbackQueryRewriter | None" = None,
        search_pipeline: SearchPipeline | None = None,
        round_compactor: FeedbackRoundCompactor | None = None,
    ) -> None:
        self._window_days = window_days
        self._page_size = page_size
        inferred_clients = clients
        if inferred_clients is None and memory_reader is not None:
            inferred_clients = getattr(memory_reader, "clients", None)
        self._clients = resolve_database_clients(inferred_clients)
        self._memory_reader = memory_reader
        self._memory_writer = memory_writer
        self._activity_collector = activity_collector
        self._query_rewriter = query_rewriter or ImplicitFeedbackQueryRewriter()
        self._search = search_pipeline
        self._round_compactor = round_compactor or FeedbackRoundCompactor()

    async def collect(
        self,
        context: MemoryRequestContext,
        *,
        scope: str = "user",
    ) -> list[ImplicitFeedbackSessionMaterial]:
        """Collect messages from add records and memories from add/search records."""

        activity = await self._collector.collect(
            _activity_scope_from_context(context, scope=scope),
            lookback=timedelta(days=self._window_days),
            max_records=self._page_size,
            add_must_not=[match_value("feedback_processed", True)],
        )

        sessions: dict[str, ImplicitFeedbackSessionMaterial] = {}
        for conversation in activity.conversations:
            if not conversation.session_id or not conversation.add_events:
                continue
            session_id = conversation.session_id
            material = sessions.setdefault(session_id, ImplicitFeedbackSessionMaterial(session_id=session_id))
            add_events = sorted(conversation.add_events, key=_add_event_time)
            round_message_groups = self._round_messages_from_add_events(add_events)
            for event in add_events:
                material.source_add_record_ids.append(event.add_record_id)
                material.memories = _merge_memories(
                    material.memories,
                    await self._add_event_memories(context, event),
                )
            for group in round_message_groups:
                round_messages = self._round_compactor.compact(group)
                if round_messages:
                    material.rounds.append(ImplicitFeedbackRound(messages=round_messages))
                    material.messages.extend(round_messages)

            for event in sorted(conversation.search_events, key=_search_event_time):
                material.memories = _merge_memories(material.memories, _search_event_memories(event))
                material.memories = _merge_memories(
                    material.memories,
                    await self._supplemental_search_memories(context, event),
                )

        return sorted(
            (material for material in sessions.values() if material.source_add_record_ids),
            key=lambda item: item.session_id,
        )

    def _round_messages_from_add_events(self, events: list[AddActivityEvent]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for event in events:
            for message in _activity_messages_to_dicts(event.messages):
                if message.get("role") == "user" and current:
                    groups.append(current)
                    current = []
                current.append(message)
        if current:
            groups.append(current)
        return groups

    async def mark_feedback_processed(self, context: MemoryRequestContext, add_record_ids: list[str]) -> None:
        """Mark add records as processed by implicit feedback."""

        for add_record_id in add_record_ids:
            await self._writer.patch_add_record(context, add_record_id, {"feedback_processed": True})

    async def _add_event_memories(
        self,
        context: MemoryRequestContext,
        event: AddActivityEvent,
    ) -> list[MemorySearchItem]:
        if not event.memory_payloads:
            return []

        hydrated = await self._hydrate_add_event_memories(context, event)
        if not hydrated:
            return []

        by_content = {memory.content: memory for memory in hydrated}
        result: list[MemorySearchItem] = []
        for raw in event.memory_payloads:
            content = raw.get("content")
            if not isinstance(content, str):
                continue
            memory = by_content.get(content)
            if memory is not None:
                result.append(_memory_view_to_search_item(memory))
        return result

    async def _hydrate_add_event_memories(
        self,
        context: MemoryRequestContext,
        event: AddActivityEvent,
    ) -> list[MemoryView]:
        request_id = event.context.request_id
        if not isinstance(request_id, str) or not request_id:
            return []

        must: list[FieldCondition] = [FieldCondition(field="request_id", op="match", value=request_id)]
        session_id = event.context.session_id
        if isinstance(session_id, str) and session_id:
            must.append(FieldCondition(field="session_id", op="match", value=session_id))
        filters = SearchFilter(must=must)
        memories, _ = await self._reader.list_memories(context, filters=filters, limit=100)
        return memories

    @property
    def _reader(self) -> MemoryDbReader:
        if self._memory_reader is None:
            self._memory_reader = MemoryDbReader()
        return self._memory_reader

    @property
    def _writer(self) -> MemoryDbWriter:
        if self._memory_writer is None:
            self._memory_writer = MemoryDbWriter(clients=self._clients)
        return self._memory_writer

    @property
    def _collector(self) -> RecentActivityCollector:
        if self._activity_collector is None:
            self._activity_collector = RecentActivityCollector(self._clients.qdrant)
        return self._activity_collector

    @property
    def _search_pipeline(self) -> SearchPipeline:
        if self._search is None:
            self._search = create_pipeline(type="search", name="search_pipeline")
        return self._search

    async def _supplemental_search_memories(
        self,
        context: MemoryRequestContext,
        event: SearchActivityEvent,
    ) -> list[MemorySearchItem]:
        original_query = event.query
        if not isinstance(original_query, str) or not original_query.strip():
            return []
        rewritten = await self._query_rewriter.rewrite(original_query)
        if not rewritten.query.strip():
            return []
        result = await self._search_pipeline.search(
            SearchPipelineInput(
                query=rewritten.query,
                filters=_supplemental_search_filters(context),
                search_pipeline="vanilla",
            ),
            context,
        )
        return result.memories


def _supplemental_search_filters(context: MemoryRequestContext) -> dict[str, str]:
    filters: dict[str, str] = {}
    if context.user_id:
        filters["user_id"] = context.user_id
    if context.session_id:
        filters["session_id"] = context.session_id
    if context.app_id:
        filters["app_id"] = context.app_id
    if context.agent_id:
        filters["agent_id"] = context.agent_id
    return filters


def _activity_scope_from_context(context: MemoryRequestContext, *, scope: str = "user") -> ActivityScope:
    data: dict[str, str | None] = {
        "project_id": context.project_id,
        "account_id": context.account_id,
        "api_key_uuid": context.api_key_uuid,
        "user_id": context.user_id,
        "app_id": context.app_id,
        "agent_id": context.agent_id,
    }
    if scope == "session":
        data["session_id"] = context.session_id
    return ActivityScope(**data)


def _activity_messages_to_dicts(messages: list[ActivityMessage]) -> list[dict[str, Any]]:
    return [message.model_dump(exclude_none=True) for message in messages]


def _add_event_time(event: AddActivityEvent) -> str:
    return event.occurred_at.isoformat() if event.occurred_at is not None else ""


def _memory_view_to_search_item(memory: MemoryView) -> MemorySearchItem:
    updated_at = memory.update_at or memory.created_at
    return MemorySearchItem(
        id=memory.memory_id,
        memory=memory.content,
        last_update_at=updated_at.strftime("%Y-%m-%d %H:%M:%S") if updated_at else "",
    )


def _search_event_time(event: SearchActivityEvent) -> str:
    return event.occurred_at.isoformat() if event.occurred_at is not None else ""


def _search_event_memories(event: SearchActivityEvent) -> list[MemorySearchItem]:
    memories: list[MemorySearchItem] = []
    for ref in event.recalled_memories:
        payload = dict(ref.payload)
        payload.setdefault("id", ref.memory_id)
        if ref.content is not None:
            payload.setdefault("memory", ref.content)
        memories.append(MemorySearchItem.model_validate(payload))
    return memories


def _merge_memories(existing: list[MemorySearchItem], supplemental: list[MemorySearchItem]) -> list[MemorySearchItem]:
    by_id = {memory.id: memory for memory in existing}
    for memory in supplemental:
        by_id.setdefault(memory.id, memory)
    return list(by_id.values())
