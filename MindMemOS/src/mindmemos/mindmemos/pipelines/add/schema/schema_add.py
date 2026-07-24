"""Schema add-memory pipeline implementation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ....components.chunker import EpisodeBoundary, EpisodesChunker
from ....components.extractor import _records as add_record_ops
from ....components.extractor.schema import (
    SchemaAddExtractor,
    SchemaAddPlanner,
    SchemaSearchFieldExtractor,
    build_episode_entity,
)
from ....components.memory_modeling.schema import EntityManager, get_entity_manager
from ....components.text import detect_prompt_language
from ....config import get_config
from ....infra.kafka import get_producer
from ....llm import EmbedClient, LLMClient, get_embed_client, get_llm_client
from ....logging import get_logger, traced, traced_awaitable
from ....prompts import AddPromptSet, get_add_prompts
from ....typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    EntityVectorWrite,
    EntityWrite,
    MemoryAddEventItem,
    MemoryDbEntityUpdateCommand,
    MemoryDbMutationPlan,
    MemoryDbWritePlan,
    MemoryRequestContext,
)
from ...base import MemoryDbPipelineMixin
from ...memory_db import (
    AddRecordBuffer,
    BufferedAddRecord,
    MemoryOperationRecorder,
    buffer_key,
    context_from_record,
    suppress_recording_errors,
    utcnow,
)
from ...registry import register
from ..base import AddPipeline

logger = get_logger(__name__)

SCHEMA_ADD_DRAIN_TOPIC = "memory.add.drain"
SCHEMA_ADD_EPISODE_TOPIC = "memory.add.episode"


@dataclass(slots=True)
class _EpisodeTask:
    """A chunked episode ready for memory generation."""

    episode_id: str
    records: list[BufferedAddRecord]
    chunk_index: int = 0
    chunk_count: int = 1
    start_idx: int = 0
    end_idx: int = 0
    title: str = ""


@register(type="add", name="schema_add")
class SchemaAddPipeline(MemoryDbPipelineMixin, AddPipeline):
    """Schema-driven add pipeline migrated from the original algorithm."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        embed_client: EmbedClient | None = None,
        entity_manager: EntityManager | None = None,
        add_buffer: AddRecordBuffer | None = None,
        chunker: EpisodesChunker | None = None,
        recorder: MemoryOperationRecorder | None = None,
        enable_schema_selection: bool | None = None,
        enable_entity_merge_decision: bool | None = None,
        entity_recall_top_k: int | None = None,
        max_merge_retries: int | None = None,
        use_property_merge: bool | None = None,
        secondary_search_limit: int | None = None,
        secondary_search_retries: int | None = None,
        use_search_fields: bool | None = None,
        search_fields_max: int | None = None,
        episode_search_fields_augment: bool | None = None,
        episode_augment_count: int | None = None,
        higher_order_enabled: bool | None = None,
        higher_order_top_k: int | None = None,
        higher_order_min_evidence_count: int | None = None,
        episode_edge_top_k: int | None = None,
        prompt_language: str | None = None,
        prompt_set: AddPromptSet | None = None,
        search_field_extractor: SchemaSearchFieldExtractor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm_client = llm_client or get_llm_client()
        self.embed_client = embed_client or get_embed_client()
        self.entity_manager = entity_manager or get_entity_manager()
        algo_cfg = get_config().algo_config
        schema_cfg = algo_cfg.add.schema
        extraction_cfg = schema_cfg.extraction
        merge_cfg = schema_cfg.merge
        higher_order_cfg = schema_cfg.higher_order
        chunker_cfg = schema_cfg.chunker
        self.prompt_set = prompt_set or get_add_prompts(prompt_language or algo_cfg.common.prompt_language)
        self.search_field_extractor = search_field_extractor or SchemaSearchFieldExtractor(
            llm_client=self.llm_client,
            prompt_set=self.prompt_set,
        )
        self.add_buffer = add_buffer or AddRecordBuffer()
        self._recorder = recorder or MemoryOperationRecorder()
        self.recorder = self._recorder
        self.chunker = chunker or EpisodesChunker(
            mode=chunker_cfg.split_mode,
            llm_client=self.llm_client,
            max_messages=chunker_cfg.max_episode_length,
            max_minutes_from_first=chunker_cfg.max_minutes_from_first,
            split_on_user_speaker=chunker_cfg.split_on_user_speaker,
            boundary_prompt=self.prompt_set.conv_boundary_detection,
            resplit_prompt=self.prompt_set.conv_forced_resplit,
        )
        self._buffer_limit = chunker_cfg.max_buffer_size
        self._min_episode_length = chunker_cfg.min_episode_length
        self._consistency = _default_consistency()
        self._episode_max_retries = schema_cfg.drain.episode_generation_max_retries
        self._cleanup_processed_buffer = schema_cfg.drain.cleanup_processed_buffer
        self.enable_schema_selection = (
            extraction_cfg.enable_schema_selection if enable_schema_selection is None else enable_schema_selection
        )
        self.enable_entity_merge_decision = (
            merge_cfg.enable_entity_merge_decision
            if enable_entity_merge_decision is None
            else enable_entity_merge_decision
        )
        self.entity_recall_top_k = merge_cfg.entity_recall_top_k if entity_recall_top_k is None else entity_recall_top_k
        self.max_merge_retries = merge_cfg.max_merge_retries if max_merge_retries is None else max_merge_retries
        self.use_property_merge = merge_cfg.use_property_merge if use_property_merge is None else use_property_merge
        self.secondary_search_limit = (
            merge_cfg.secondary_search_limit if secondary_search_limit is None else secondary_search_limit
        )
        self.secondary_search_retries = (
            merge_cfg.secondary_search_retries if secondary_search_retries is None else secondary_search_retries
        )
        self.use_search_fields = extraction_cfg.use_search_fields if use_search_fields is None else use_search_fields
        self.search_fields_max = extraction_cfg.search_fields_max if search_fields_max is None else search_fields_max
        self.episode_search_fields_augment = (
            extraction_cfg.episode_search_fields_augment
            if episode_search_fields_augment is None
            else episode_search_fields_augment
        )
        self.episode_augment_count = (
            extraction_cfg.episode_augment_count if episode_augment_count is None else episode_augment_count
        )
        self.higher_order_enabled = higher_order_cfg.enabled if higher_order_enabled is None else higher_order_enabled
        self.higher_order_top_k = higher_order_cfg.top_k if higher_order_top_k is None else higher_order_top_k
        self.higher_order_min_evidence_count = (
            higher_order_cfg.min_evidence_count
            if higher_order_min_evidence_count is None
            else higher_order_min_evidence_count
        )
        self.episode_edge_top_k = schema_cfg.episode_edge.top_k if episode_edge_top_k is None else episode_edge_top_k
        self._processing_by_key: dict[str, bool] = defaultdict(bool)
        self._process_lock_by_key: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.extractor = SchemaAddExtractor(
            llm_client=self.llm_client,
            prompt_set=self.prompt_set,
            entity_manager=self.entity_manager,
            enable_schema_selection=self.enable_schema_selection,
        )
        self.planner = SchemaAddPlanner(
            llm_client=self.llm_client,
            embed_client=self.embed_client,
            db_reader=self.db_reader,
            db_writer=self.db_writer,
            entity_manager=self.entity_manager,
            prompt_set=self.prompt_set,
            enable_entity_merge_decision=self.enable_entity_merge_decision,
            entity_recall_top_k=self.entity_recall_top_k,
            max_merge_retries=self.max_merge_retries,
            use_property_merge=self.use_property_merge,
            secondary_search_limit=self.secondary_search_limit,
            secondary_search_retries=self.secondary_search_retries,
            higher_order_enabled=self.higher_order_enabled,
            higher_order_top_k=self.higher_order_top_k,
            higher_order_min_evidence_count=self.higher_order_min_evidence_count,
            episode_edge_top_k=self.episode_edge_top_k,
        )

    @traced("add_pipeline.sync", record_args=False)
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        """Append messages to the add buffer and drain them synchronously.

        Args:
            inp: Add request payload, including messages and force-generation options.
            context: Tenant and project context used for storage isolation.
            add_record_id: Optional add record id to write the output back onto.

        Returns:
            The generated memory events for this synchronous add request.
        """

        await self.add_buffer.append(
            context,
            inp,
            force_generation=inp.force_generation,
            source_add_record_id=add_record_id,
        )
        events = await self._ensure_drain_and_wait(
            context,
            consistency=self._consistency,
            force=True,
        )
        result = AddPipelineSyncResult(status="ok", memories=events)
        # Sync drains inline and produces the full output in one shot, so overwrite
        # the request-level record directly. The inline path does not thread the
        # trigger id into episodes, so there is no double write.
        await suppress_recording_errors(
            self.recorder.mark_add_completed(context, add_record_id, result),
            operation="add.schema_add.sync",
        )
        return result

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> AddPipelineAsyncResult:
        """Append messages to the add buffer and queue background draining.

        Args:
            inp: Add request payload, including messages and force-generation options.
            context: Tenant and project context used for storage isolation.
            add_record_id: Optional triggering add record id. Every episode produced
                by the drain this request kicks off is accumulated onto this record
                (trigger binding, not message provenance).

        Returns:
            A queued status result.

        Raises:
            RuntimeError: If Kafka is disabled for asynchronous add processing.
        """
        if not get_config().kafka.enabled:
            raise RuntimeError(
                "schema_add add_async requires Kafka to be enabled (kafka.enabled=true). "
                "Use mode='sync' or enable Kafka in config."
            )
        await self.add_buffer.append(
            context,
            inp,
            force_generation=inp.force_generation,
            source_add_record_id=add_record_id,
        )
        await self._ensure_drain_started(
            context,
            inp,
            force=inp.force_generation,
            trigger_record_id=add_record_id,
            record_metadata=record_metadata,
        )
        return AddPipelineAsyncResult(status="queued")

    async def has_pending(self, context: MemoryRequestContext) -> bool:
        """Check whether the project buffer still has unprocessed add records.

        Args:
            context: Tenant and project context used to select the buffer.

        Returns:
            True when buffered records are still pending.
        """
        return await self.add_buffer.has_pending(context)

    async def drain_buffer(
        self,
        context: MemoryRequestContext,
        *,
        consistency: str | None = None,
        force: bool = False,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Drain buffered add records from an external worker entry point."""
        contexts = [context]
        if not await self.add_buffer.list_buffered(context, limit=1):
            contexts = await self._contexts_for_project(context.project_id, limit=100)

        events: list[MemoryAddEventItem] = []
        dispatched = 0
        for drain_context in contexts:
            if not await self._try_start_loop(drain_context):
                continue
            loop_events, loop_dispatched = await self._process_loop(
                drain_context,
                consistency=consistency or self._consistency,
                force=force,
                trigger_record_id=trigger_record_id,
            )
            events.extend(loop_events)
            dispatched += loop_dispatched
        # Trigger produced no episode this drain: finalize it as ok/empty so the
        # request-level record does not linger at "queued". When episodes were
        # dispatched, the episode workers accumulate output onto the trigger.
        if trigger_record_id and dispatched == 0:
            await suppress_recording_errors(
                self.recorder.mark_add_completed(
                    context, trigger_record_id, AddPipelineSyncResult(status="ok", memories=[])
                ),
                operation="add.schema_add.drain_buffer",
            )
        return events

    async def generate_episode(
        self,
        context: MemoryRequestContext,
        add_record_ids: list[str],
        *,
        episode_id: str,
        consistency: str | None = None,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Generate one episode from an external worker entry point."""
        records = await self.add_buffer.get_by_ids(context, add_record_ids)
        if not records:
            logger.warning(
                "episode generation skipped: records not found in buffer",
                episode_id=episode_id,
                expected_count=len(add_record_ids),
            )
            return []
        await self.add_buffer.mark_processing(context, records)
        return await self._execute_episode_task(
            _EpisodeTask(episode_id=episode_id, records=records),
            context=context,
            consistency=consistency or self._consistency,
            trigger_record_id=trigger_record_id,
        )

    # Internal: drain orchestration

    async def _ensure_drain_started(
        self,
        context: MemoryRequestContext,
        inp: AddPipelineInput,
        *,
        force: bool,
        trigger_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish the drain trigger to Kafka; async mode is Kafka-only."""

        if not get_config().kafka.enabled:
            raise RuntimeError(
                "schema_add async drain requires Kafka to be enabled (kafka.enabled=true). "
                "Use mode='sync' or enable Kafka in config."
            )
        try:
            await self._publish_drain_task(
                context,
                inp,
                force=force,
                trigger_record_id=trigger_record_id,
                record_metadata=record_metadata,
            )
        except Exception:
            logger.error("schema add drain task publish failed", exc_info=True)
            raise

    async def _ensure_drain_and_wait(
        self,
        context: MemoryRequestContext,
        *,
        consistency: str,
        force: bool,
    ) -> list[MemoryAddEventItem]:
        """Start the drain loop and wait for generated events."""
        key = buffer_key(context)
        while True:
            async with self._process_lock_by_key[key]:
                if not self._processing_by_key[key]:
                    self._processing_by_key[key] = True
                    break
            await asyncio.sleep(0.05)
        events, _dispatched = await self._process_loop(context, consistency=consistency, force=force, inline=True)
        return events

    async def _publish_drain_task(
        self,
        context: MemoryRequestContext,
        inp: AddPipelineInput,
        *,
        force: bool,
        trigger_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> None:
        key = buffer_key(context)
        await get_producer().send(
            SCHEMA_ADD_DRAIN_TOPIC,
            value={
                "context": context.model_dump(mode="json"),
                "input": inp.model_dump(mode="json", by_alias=True),
                "force": force,
                "consistency": self._consistency,
                "trigger_record_id": trigger_record_id,
                "record_metadata": record_metadata,
            },
            dispatch_key=key,
        )

    async def _try_start_loop(self, context: MemoryRequestContext) -> bool:
        """Acquire processing ownership for the drain loop."""
        key = buffer_key(context)
        async with self._process_lock_by_key[key]:
            if self._processing_by_key[key]:
                return False
            self._processing_by_key[key] = True
            return True

    async def _finish_processing(self, context: MemoryRequestContext) -> None:
        key = buffer_key(context)
        async with self._process_lock_by_key[key]:
            self._processing_by_key[key] = False

    async def _context_for_buffer_key(self, project_id: str, key: str) -> MemoryRequestContext | None:
        records = await self.add_buffer.list_buffered_for_key(project_id, key, limit=1)
        if not records:
            return None
        return context_from_record(records[0])

    async def _contexts_for_project(self, project_id: str, *, limit: int) -> list[MemoryRequestContext]:
        contexts: list[MemoryRequestContext] = []
        for pending_key in await self.add_buffer.list_buffer_keys_with_new_records(limit=limit):
            if pending_key.project_id != project_id:
                continue
            context = await self._context_for_buffer_key(pending_key.project_id, pending_key.buffer_key)
            if context is not None:
                contexts.append(context)
        return contexts

    # Core: two-phase process loop (chunking → dispatch)

    async def _process_loop(
        self,
        context: MemoryRequestContext,
        *,
        consistency: str,
        force: bool,
        inline: bool = False,
        trigger_record_id: str | None = None,
    ) -> tuple[list[MemoryAddEventItem], int]:
        """Run the two-phase drain loop until no processable episodes remain.

        Returns the generated events plus the number of episodes dispatched in
        this drain, so the async caller can finalize a trigger record that
        produced nothing (``ok`` with empty output) instead of leaving it queued.
        """
        events: list[MemoryAddEventItem] = []
        dispatched = 0
        try:
            while True:
                # Phase 1: Chunking
                episode_tasks = await self._chunk_episodes(context, force=force)
                if not episode_tasks:
                    break

                # Phase 2: Dispatch
                dispatched += len(episode_tasks)
                round_events = await (
                    self._dispatch_episodes_inline(episode_tasks, context=context, consistency=consistency)
                    if inline
                    else self._dispatch_episodes_kafka(
                        episode_tasks,
                        context=context,
                        consistency=consistency,
                        trigger_record_id=trigger_record_id,
                    )
                )
                events.extend(round_events)

                if not inline or not round_events:
                    break
        finally:
            await self._finish_processing(context)
        return events, dispatched

    async def _chunk_episodes(self, context: MemoryRequestContext, *, force: bool) -> list[_EpisodeTask]:
        """Split buffered records into episode generation tasks."""
        records = await self.add_buffer.list_buffered(context, limit=self._buffer_limit)
        entries = add_record_ops.to_chunker_entries(records)
        if len(entries) < self._min_episode_length:
            if records:
                await self.add_buffer.mark_split_attempted(context, records)
            return []

        sample_text = " ".join(str(e.get("content", "")) for e in entries[:20])
        detected_lang = detect_prompt_language(
            sample_text,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = get_add_prompts(detected_lang)

        detect_force = force or add_record_ops.force_generation(records)
        boundaries = await traced_awaitable(
            "schema_add.chunk_episodes.detect_boundaries",
            self.chunker.detect_boundaries(
                entries,
                force=detect_force,
                boundary_prompt=request_prompts.conv_boundary_detection,
                resplit_prompt=request_prompts.conv_forced_resplit,
            ),
            attributes={
                "project_id": context.project_id,
                "entry_count": len(entries),
                "force": detect_force,
                "chunker.mode": self.chunker.mode,
                "chunker.max_messages": self.chunker.max_messages,
            },
            record_result=True,
            tracer_name=__name__,
        )
        if not boundaries and len(entries) >= self.chunker.max_messages:
            boundaries = [EpisodeBoundary(start_idx=0, end_idx=self.chunker.max_messages - 1)]
        if not boundaries:
            await self.add_buffer.mark_split_attempted(context, records)
            return []

        tasks: list[_EpisodeTask] = []
        chunk_count = len(boundaries)
        queued_record_ids: set[str] = set()
        for chunk_index, boundary in enumerate(boundaries):
            episode_records = records[boundary.start_idx : boundary.end_idx + 1]
            if not episode_records:
                continue
            episode_id = str(uuid4())
            await self.add_buffer.mark_episode_queued(context, episode_records, episode_id=episode_id)
            queued_record_ids.update(record.add_record_id for record in episode_records)
            tasks.append(
                _EpisodeTask(
                    episode_id=episode_id,
                    records=episode_records,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    start_idx=boundary.start_idx,
                    end_idx=boundary.end_idx,
                    title=boundary.title,
                )
            )
        remaining_records = [record for record in records if record.add_record_id not in queued_record_ids]
        if remaining_records:
            await self.add_buffer.mark_split_attempted(context, remaining_records)
        return tasks

    async def _dispatch_episodes_kafka(
        self,
        tasks: list[_EpisodeTask],
        *,
        context: MemoryRequestContext,
        consistency: str,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Publish episode generation tasks to Kafka."""
        producer = get_producer()
        key = buffer_key(context)
        for task in tasks:
            try:
                await producer.send(
                    SCHEMA_ADD_EPISODE_TOPIC,
                    value={
                        "context": context.model_dump(mode="json"),
                        "add_record_ids": [r.add_record_id for r in task.records],
                        "episode_id": task.episode_id,
                        "consistency": consistency,
                        "trigger_record_id": trigger_record_id,
                    },
                    dispatch_key=key,
                )
            except Exception:
                logger.error(
                    "failed to publish episode task to kafka; restoring records to buffered",
                    episode_id=task.episode_id,
                    exc_info=True,
                )
                await self.add_buffer.restore_buffered(context, task.records, error="kafka episode publish failed")
        return []

    async def _dispatch_episodes_inline(
        self,
        tasks: list[_EpisodeTask],
        *,
        context: MemoryRequestContext,
        consistency: str,
    ) -> list[MemoryAddEventItem]:
        """Execute episode generation tasks in the current process."""
        events: list[MemoryAddEventItem] = []
        for task in tasks:
            task_events = await self._execute_episode_task(task, context=context, consistency=consistency)
            events.extend(task_events)
        return events

    # Episode execution with retry, failure recording

    async def _execute_episode_task(
        self,
        task: _EpisodeTask,
        *,
        context: MemoryRequestContext,
        consistency: str,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Trace and execute one episode generation task."""
        return await traced_awaitable(
            "schema_add.episode_chunk",
            self._execute_episode_task_inner(
                task,
                context=context,
                consistency=consistency,
                trigger_record_id=trigger_record_id,
            ),
            attributes={
                "project_id": context.project_id,
                "episode_id": task.episode_id,
                "chunk.index": task.chunk_index,
                "chunk.count": task.chunk_count,
                "chunk.start_idx": task.start_idx,
                "chunk.end_idx": task.end_idx,
                "chunk.record_count": len(task.records),
                "chunk.title": task.title,
                "consistency": consistency,
            },
            record_result=False,
            tracer_name=__name__,
        )

    async def _execute_episode_task_inner(
        self,
        task: _EpisodeTask,
        *,
        context: MemoryRequestContext,
        consistency: str,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        for attempt in range(self._episode_max_retries):
            try:
                episode_events = await self._generate_episode_memory(
                    task.records,
                    context=context,
                    consistency=consistency,
                )
                await self.add_buffer.mark_processed(
                    context,
                    task.records,
                    episode_id=task.episode_id,
                    events=_events_to_payload(episode_events),
                )
                if self._cleanup_processed_buffer:
                    try:
                        await self.add_buffer.delete_processed(context, task.records)
                    except Exception:
                        logger.warning(
                            "failed to cleanup processed buffer records",
                            episode_id=task.episode_id,
                            exc_info=True,
                        )
                # Trigger binding: accumulate this episode's output onto the request
                # that kicked off the drain (async/Kafka path only; inline sync writes
                # the full output via add_sync, so trigger_record_id is None there).
                await suppress_recording_errors(
                    self.recorder.append_add_output(context, trigger_record_id, episode_events),
                    operation="add.schema_add.episode_chunk",
                )
                return episode_events
            except Exception as exc:
                if attempt < self._episode_max_retries - 1:
                    logger.warning(
                        "episode memory generation failed; retrying",
                        attempt=attempt + 1,
                        episode_id=task.episode_id,
                        exc_info=True,
                    )
                else:
                    error_msg = str(exc)
                    logger.error(
                        "episode memory generation failed permanently",
                        episode_id=task.episode_id,
                        exc_info=True,
                    )
                    try:
                        await self.add_buffer.mark_failed(context, task.records, error=error_msg)
                    except Exception:
                        logger.error("failed to mark episode records as failed", exc_info=True)
                    if trigger_record_id:
                        await suppress_recording_errors(
                            self._recorder.mark_add_failed(context, trigger_record_id, error_msg),
                            operation="add",
                        )
                    else:
                        await self._record_episode_failure(task.records, context=context)
        return []

    async def _record_episode_failure(self, records: list[BufferedAddRecord], *, context: MemoryRequestContext) -> None:
        """Record a failed episode generation attempt for audit history."""
        records_time = add_record_ops.records_added_datetime(records)
        reconstructed_input = _reconstruct_input_from_records(records)
        await suppress_recording_errors(
            self._recorder.record_add(
                reconstructed_input,
                None,
                ctx=context,
                request_submitted_at=records_time,
                task_completed_at=utcnow(),
            ),
            operation="add",
        )

    async def _generate_episode_memory(
        self,
        records: list[BufferedAddRecord],
        *,
        context: MemoryRequestContext,
        consistency: str,
    ) -> list[MemoryAddEventItem]:
        """Generate schema entities, vectors, and write events for one episode."""
        conversation_text = add_record_ops.to_conversation_text(records)
        if not conversation_text.strip():
            return []

        detected_lang = detect_prompt_language(
            conversation_text,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = get_add_prompts(detected_lang)

        episode_context = add_record_ops.context(records, context)
        event_at = add_record_ops.records_datetime(records)
        added_at = add_record_ops.records_added_datetime(records)
        dialogue_timestamp = add_record_ops.dialogue_timestamp(event_at)

        objectify_task = asyncio.create_task(
            self.extractor.objectify_conversation(conversation_text, dialogue_timestamp, prompt_set=request_prompts)
        )
        description_task = asyncio.create_task(
            self.extractor.generate_episode_description(
                conversation_text, dialogue_timestamp, prompt_set=request_prompts
            )
        )
        schema_selection_task = asyncio.create_task(
            self.extractor.select_schema(
                conversation_text, self.extractor.schema_for_generation(), prompt_set=request_prompts
            )
        )

        selected_schema = await schema_selection_task
        raw_memory = await self.extractor.extract_memory(
            entity_schema=selected_schema,
            dialogue_timestamp=dialogue_timestamp,
            conversation_text=conversation_text,
            prompt_set=request_prompts,
        )

        _raw_before_prepare = raw_memory.get("entities", [])
        logger.info(
            "schema_add drain: BEFORE prepare_raw_memory: %d entities, types=%s",
            len(_raw_before_prepare),
            [e.get("entity_type") for e in _raw_before_prepare],
        )

        raw_memory = self.extractor.prepare_raw_memory(raw_memory, dialogue_timestamp)

        _raw_entities = raw_memory.get("entities", [])
        _entity_types = [e.get("entity_type") for e in _raw_entities]
        logger.info(
            "schema_add drain: AFTER prepare_raw_memory: %d entities, types=%s, selected_schema_types=%s",
            len(_raw_entities),
            _entity_types,
            [s.get("entity_type") for s in selected_schema],
        )

        objectified_content = await objectify_task
        episode_description = await description_task
        episode_search_fields = (
            await self.search_field_extractor.extract_search_fields(
                entities=raw_memory.get("entities", []),
                context_text=conversation_text,
                max_fields=self.search_fields_max,
                augment=self.episode_search_fields_augment,
                augment_count=self.episode_augment_count,
                prompt_set=request_prompts,
            )
            if self.use_search_fields
            else []
        )
        episode_entity = build_episode_entity(
            objectified_content=objectified_content,
            episode_description=episode_description,
            dialogue_date=dialogue_timestamp.split(" ", 1)[0],
            search_fields=episode_search_fields,
        )

        plan, events, pending_archives, pending_updates = await self.planner.build_write_plan(
            raw_entities=raw_memory.get("entities", []),
            raw_edges=raw_memory.get("edges", []),
            episode_entity=episode_entity,
            context=episode_context,
            request_metadata=add_record_ops.metadata(records),
            created_at=added_at,
            prompt_set=request_prompts,
        )

        entity_updates = _split_entity_updates(plan)
        memory_update_commands = await self.planner.build_memory_update_commands(
            episode_context,
            pending_updates,
            consistency=consistency,
        )
        memory_delete_commands = self.planner.build_archive_memory_commands(pending_archives, consistency=consistency)
        mutation_plan = MemoryDbMutationPlan.from_write_plan(plan)
        mutation_plan.entity_updates.extend(_to_entity_update_commands(entity_updates, consistency=consistency))
        mutation_plan.memory_updates.extend(memory_update_commands)
        mutation_plan.memory_deletes.extend(memory_delete_commands)
        write_result = await self.db_writer.apply_mutation_plan(
            episode_context,
            mutation_plan,
            consistency=consistency,
        )
        update_results = write_result.mutations[: len(memory_update_commands)]
        update_events = self.planner.memory_update_events(pending_updates, update_results)
        return events + update_events


def _events_to_payload(events: list[MemoryAddEventItem]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="python") for event in events]


def _split_entity_updates(plan: MemoryDbWritePlan) -> list[tuple[EntityWrite, list[EntityVectorWrite]]]:
    update_ids = {
        entity.entity_id
        for entity in plan.entities
        if isinstance(entity.metadata, dict) and entity.metadata.get("merge_action") == "update"
    }
    if not update_ids:
        return []

    vectors_by_entity: dict[str, list[EntityVectorWrite]] = defaultdict(list)
    remaining_vectors: list[EntityVectorWrite] = []
    for vector in plan.entity_vectors:
        owner_id = vector.entity_id.split("#sf", 1)[0]
        if owner_id in update_ids:
            vectors_by_entity[owner_id].append(vector)
        else:
            remaining_vectors.append(vector)

    updates: list[tuple[EntityWrite, list[EntityVectorWrite]]] = []
    remaining_entities: list[EntityWrite] = []
    for entity in plan.entities:
        if entity.entity_id in update_ids:
            updates.append((entity, vectors_by_entity.get(entity.entity_id, [])))
        else:
            remaining_entities.append(entity)

    plan.entities = remaining_entities
    plan.entity_vectors = remaining_vectors
    return updates


def _to_entity_update_commands(
    updates: list[tuple[EntityWrite, list[EntityVectorWrite]]],
    *,
    consistency: str,
) -> list[MemoryDbEntityUpdateCommand]:
    commands: list[MemoryDbEntityUpdateCommand] = []
    for entity, vectors in updates:
        commands.append(
            MemoryDbEntityUpdateCommand(
                entity_id=entity.entity_id,
                entity=entity,
                core_vector=next((vector for vector in vectors if vector.entity_id == entity.entity_id), None),
                search_field_vectors=[vector for vector in vectors if vector.entity_id != entity.entity_id],
                consistency=consistency,
            )
        )
    return commands


def _reconstruct_input_from_records(records: list[BufferedAddRecord]) -> AddPipelineInput:
    """Rebuild a minimal add pipeline input from buffer records."""
    messages = []
    metadata: dict[str, Any] = {}
    for record in records:
        payload = record.payload
        record_messages = payload.get("messages", [])
        messages.extend(record_messages)
        if not metadata and payload.get("metadata"):
            metadata = payload["metadata"]
    try:
        return AddPipelineInput(messages=messages, metadata=metadata)
    except Exception:
        return AddPipelineInput(metadata=metadata)


def _default_consistency() -> str:
    value = get_config().database.default_consistency
    return value if value in {"fast", "strong"} else "fast"
