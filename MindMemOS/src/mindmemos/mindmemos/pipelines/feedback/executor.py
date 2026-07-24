"""Execute planned feedback actions through database mutation primitives."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ...components.extractor.schema import memory_embedding_text
from ...components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ...config import TextProcessingConfig, get_config
from ...llm import EmbedClient, get_embed_client
from ...typing import (
    REL_DERIVED_FROM,
    FeedbackActionResult,
    FeedbackAddAction,
    FeedbackDeleteAction,
    FeedbackUpdateAction,
    GraphNodeRef,
    GraphRelationship,
    MemoryDbDeleteCommand,
    MemoryDbMutationPlan,
    MemoryDbUpdateCommand,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    VectorWrite,
)
from ..memory_db import MemoryDbReader, MemoryDbWriter


class FeedbackActionExecutor:
    """Execute feedback action plans without re-running memory pipelines."""

    def __init__(
        self,
        *,
        db_reader: MemoryDbReader | None = None,
        db_writer: MemoryDbWriter | None = None,
        embed_client: EmbedClient | None = None,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
    ) -> None:
        self._db_reader = db_reader
        self._db_writer = db_writer
        self._embed_client = embed_client
        self._text_config = text_config
        self._text_preprocessor = text_preprocessor
        self._sparse_encoder = sparse_encoder

    async def execute(
        self,
        actions: list[FeedbackActionResult],
        context: MemoryRequestContext,
    ) -> list[FeedbackActionResult]:
        """Execute planned actions and return per-action results."""

        results: list[FeedbackActionResult] = []
        for action in actions:
            if action.action == "add":
                results.append(await self._execute_add(action, context))
            elif action.action == "update":
                results.append(await self._execute_update(action, context))
            elif action.action == "delete":
                results.append(await self._execute_delete(action, context))
            else:
                results.append(action)
        return results

    async def _execute_add(
        self,
        action: FeedbackAddAction,
        context: MemoryRequestContext,
    ) -> FeedbackActionResult:
        now = datetime.now(UTC)
        memory_id = str(uuid4())
        vector = await self._vectorize(action.after_content)
        result = await self._db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan.from_write_plan(
                MemoryDbWritePlan(
                    memories=[
                        MemoryWrite(
                            memory_id=memory_id,
                            account_id=context.account_id,
                            project_id=context.project_id,
                            api_key_uuid=context.api_key_uuid,
                            user_id=context.user_id,
                            app_id=context.app_id,
                            session_id=context.session_id,
                            agent_id=context.agent_id,
                            request_id=context.request_id,
                            content=action.after_content,
                            mem_type="fact",
                            mem_extract_type="feedback",
                            mem_extract_version="feedback_v1",
                            metadata=_feedback_metadata(action.reason, now),
                            created_at=now,
                            root_id=[memory_id],
                        )
                    ],
                    vectors=[vector.model_copy(update={"memory_id": memory_id})],
                )
            ),
            consistency="strong",
        )
        status = "error" if result.errors else "ok"
        return action.model_copy(update={"result_memory_id": memory_id, "status": status})

    async def _execute_update(
        self,
        action: FeedbackUpdateAction,
        context: MemoryRequestContext,
    ) -> FeedbackActionResult:
        """通过创建新记忆版本 + 归档旧记忆来实现更新，版本关系通过 Neo4j DERIVED_FROM 维系。"""
        now = datetime.now(UTC)
        memory = await self._db_reader.get_memory(context, action.target_memory_id)
        if memory is None:
            return action.model_copy(update={"result_memory_id": action.target_memory_id, "status": "error"})

        # 创建新版本记忆
        new_memory_id = str(uuid4())
        vector_text = memory_embedding_text(memory.model_copy(update={"content": action.after_content}))
        vector = await self._vectorize(vector_text)

        # 合并旧 metadata 与 feedback 元信息
        merged_metadata = (memory.metadata or {}).copy()
        merged_metadata.update(_feedback_metadata(action.reason, now))

        write_plan = MemoryDbWritePlan(
            memories=[
                MemoryWrite(
                    memory_id=new_memory_id,
                    account_id=context.account_id,
                    project_id=context.project_id,
                    api_key_uuid=context.api_key_uuid,
                    user_id=context.user_id,
                    app_id=context.app_id,
                    session_id=context.session_id,
                    agent_id=context.agent_id,
                    request_id=context.request_id,
                    content=action.after_content,
                    mem_type=memory.mem_type or "fact",
                    mem_extract_type="feedback",
                    mem_extract_version="feedback_v1",
                    metadata=merged_metadata,
                    parent_ids=[action.target_memory_id],
                    root_id=memory.root_id or [new_memory_id],
                    created_at=now,
                )
            ],
            vectors=[vector.model_copy(update={"memory_id": new_memory_id})],
            relationships=[
                GraphRelationship(
                    source=GraphNodeRef(
                        kind="Memory", project_id=context.project_id, node_id=new_memory_id
                    ),
                    target=GraphNodeRef(
                        kind="Memory", project_id=context.project_id, node_id=action.target_memory_id
                    ),
                    rel_type=REL_DERIVED_FROM,
                    project_id=context.project_id,
                    metadata={"reason": action.reason or "", "created_at": now.isoformat()},
                )
            ],
        )

        # 归档旧记忆
        archive_meta = _feedback_metadata(action.reason, now)
        archive_meta["derived_to"] = new_memory_id
        archive_command = MemoryDbUpdateCommand(
            memory_id=action.target_memory_id,
            status="archived",
            reason="feedback_update",
            metadata_patch=archive_meta,
        )

        plan = MemoryDbMutationPlan.from_write_plan(write_plan)
        plan.memory_updates = [archive_command]

        result = await self._db_writer.apply_mutation_plan(context, plan, consistency="strong")

        write_success = new_memory_id in result.memory_ids
        mutation = result.mutations[0] if result.mutations else None
        archive_success = bool(mutation and mutation.changed)
        changed = write_success and archive_success

        status = "ok" if changed else "error"
        return action.model_copy(
            update={"result_memory_id": new_memory_id if changed else action.target_memory_id, "status": status}
        )

    async def _execute_delete(
        self,
        action: FeedbackDeleteAction,
        context: MemoryRequestContext,
    ) -> FeedbackActionResult:
        """归档记忆并记录原因。旧记忆在 Qdrant 中保留（status="archived"），不再写 FeedbackPatch。"""
        memory = await self._db_reader.get_memory(context, action.target_memory_id)
        command = MemoryDbDeleteCommand(memory_id=action.target_memory_id, reason="feedback_delete")
        result = await self._db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan(memory_deletes=[command]),
            consistency=command.consistency,
        )
        mutation = result.mutations[0] if result.mutations else None
        changed = bool(mutation and mutation.changed)
        status = "ok" if changed else "error"
        return action.model_copy(
            update={"result_memory_id": action.target_memory_id, "status": status}
        )

    async def _vectorize(self, text: str) -> VectorWrite:
        embed_resp = await self._embed_client.embed(task="memory.feedback.mutation", text=text)
        dense_vector = embed_resp.embeddings[0] if embed_resp.embeddings else None
        preprocessed = self._text_preprocessor.preprocess_text(text, include_entities=False)
        sparse = self._sparse_encoder.encode_document(preprocessed.tokens)
        return VectorWrite(
            memory_id="",
            semantic_vector=dense_vector,
            bm25_indices=list(sparse.indices),
            bm25_values=list(sparse.values),
        )

    @property
    def _db_reader(self) -> MemoryDbReader:
        if self.__db_reader is None:
            self.__db_reader = MemoryDbReader()
        return self.__db_reader

    @_db_reader.setter
    def _db_reader(self, value: MemoryDbReader | None) -> None:
        self.__db_reader = value

    @property
    def _db_writer(self) -> MemoryDbWriter:
        if self.__db_writer is None:
            self.__db_writer = MemoryDbWriter()
        return self.__db_writer

    @_db_writer.setter
    def _db_writer(self, value: MemoryDbWriter | None) -> None:
        self.__db_writer = value

    @property
    def _embed_client(self) -> EmbedClient:
        if self.__embed_client is None:
            self.__embed_client = get_embed_client()
        return self.__embed_client

    @_embed_client.setter
    def _embed_client(self, value: EmbedClient | None) -> None:
        self.__embed_client = value

    @property
    def _text_preprocessor(self) -> TextPreprocessor:
        if self.__text_preprocessor is None:
            cfg = self._text_config or get_config().algo_config.text_processing
            self.__text_preprocessor = get_text_preprocessor(cfg)
        return self.__text_preprocessor

    @_text_preprocessor.setter
    def _text_preprocessor(self, value: TextPreprocessor | None) -> None:
        self.__text_preprocessor = value

    @property
    def _sparse_encoder(self) -> SparseVectorEncoder:
        if self.__sparse_encoder is None:
            cfg = self._text_config or get_config().algo_config.text_processing
            self.__sparse_encoder = SparseVectorEncoder(cfg)
        return self.__sparse_encoder

    @_sparse_encoder.setter
    def _sparse_encoder(self, value: SparseVectorEncoder | None) -> None:
        self.__sparse_encoder = value




def _feedback_metadata(reason: str | None, now: datetime) -> dict[str, str]:
    metadata = {
        "mutation_source": "feedback",
        "feedback_mutation_at": now.isoformat(),
    }
    if reason:
        metadata["feedback_reason"] = reason
    return metadata
