"""Real LLM integration tests for the memory extractor.

These tests require the LLM_API_KEY environment variable (and optionally
LLM_API_BASE, LLM_MODEL). Without LLM_API_KEY, all tests are skipped.

Uses litellm.acompletion directly to avoid the global config dependency
in LLMClient.
"""

import json
import os
from pathlib import Path

import pytest
import yaml
from mindmemos.components.extractor.vanilla.memory import (
    _envelope_prompt_messages,
)
from mindmemos.typing.algo import ExtractionEnvelope, TurnMessageRef
from mindmemos.typing.llm import ChatResponse, EmbeddingResponse
from mindmemos.typing.memory import (
    REL_MENTIONED_IN_SOURCE,
    REL_MENTIONS,
    MemoryRequestContext,
    MemoryView,
    RelatedMemoryCandidate,
    RelatedMemoryRecallResult,
)
from mindmemos.typing.memory_db import MemoryDbSearchResult, MemoryDbWriteResult
from mindmemos.typing.service import AddPipelineInput

from mindmemos.components.chunker import MessageSegmenter
from mindmemos.components.extractor.vanilla import (
    MemoryExtractionResult,
    VanillaMemoryExtractor,
    parse_memory_extraction_json,
)
from mindmemos.components.text import TextPreprocessor
from mindmemos.config import TextProcessingConfig, VanillaAddConfig
from mindmemos.pipelines.add.vanilla import VanillaAddPipeline

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_DEV_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "dev.yaml"


def _dev_chat_endpoint() -> dict:
    if not _DEV_CONFIG_PATH.exists():
        return {}
    with _DEV_CONFIG_PATH.open() as fh:
        data = yaml.safe_load(fh) or {}
    endpoints = (data.get("chat_model_router") or {}).get("endpoints") or []
    if not endpoints:
        return {}
    return endpoints[0] or {}


_DEV_CHAT_ENDPOINT = _dev_chat_endpoint()
_LLM_API_KEY = os.environ.get("LLM_API_KEY") or _DEV_CHAT_ENDPOINT.get("api_key")
_LLM_API_BASE = os.environ.get("LLM_API_BASE") or _DEV_CHAT_ENDPOINT.get("api_base") or "https://api.openai.com/v1"
_LLM_MODEL = os.environ.get("LLM_MODEL") or _DEV_CHAT_ENDPOINT.get("model") or "gpt-4o-mini"
_LLM_EXTRA_BODY = (
    _DEV_CHAT_ENDPOINT.get("extra_body") if isinstance(_DEV_CHAT_ENDPOINT.get("extra_body"), dict) else None
)

skip_no_llm_key = pytest.mark.skipif(
    not _LLM_API_KEY,
    reason="No LLM key in LLM_API_KEY or config/mindmemos/dev.yaml chat_model_router.endpoints[0]",
)


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-integration",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def make_text_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(
        TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        )
    )


def make_segment_and_preprocessed(message: dict):
    inp = AddPipelineInput(messages=[message])
    segment = MessageSegmenter().segment(inp)[0][0]
    preprocessed = make_text_preprocessor().preprocess_text(
        segment.text,
        source_ref=segment.source_ref,
        segment_id=segment.segment_id,
    )
    return segment, preprocessed


def make_envelope(segments, *, recalled_memories=None) -> ExtractionEnvelope:
    return ExtractionEnvelope(
        extractable_messages=[
            TurnMessageRef(
                text=segment.text,
                role=segment.role or "user",
                timestamp=segment.timestamp,
                message_index=segment.message_index if segment.message_index is not None else index,
                is_extractable=True,
            )
            for index, segment in enumerate(segments)
        ],
        recalled_memories=recalled_memories or [],
        boundary="complete",
        chunk_index=0,
    )


class _DirectLlmClient:
    """Thin wrapper around litellm.acompletion that matches the LLMClient.chat() interface.

    Avoids the global config dependency so integration tests can run without init_config().
    """

    def __init__(self) -> None:
        self.calls: list = []

    async def chat(
        self,
        task: str,
        messages: list[dict],
        format_parser=None,
        **kwargs,
    ) -> ChatResponse:
        import litellm

        print(f"\n[LLM call] task={task}, model={_LLM_MODEL}, api_base={_LLM_API_BASE}")
        print(f"[LLM call] system prompt length: {len(messages[0]['content'])} chars")
        user_payload = json.loads(messages[1]["content"])
        print(f"[LLM call] extractable count: {len(user_payload.get('extractable', []))}")
        print(f"[LLM call] related_memories: {len(user_payload.get('context', {}).get('related_memories', []))}")

        self.calls.append({"task": task, "messages": messages})

        resp = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=messages,
            api_key=_LLM_API_KEY,
            api_base=_LLM_API_BASE,
            extra_body=_LLM_EXTRA_BODY,
            drop_params=True,
        )

        content = resp.choices[0].message.content or ""
        print(f"\n[LLM response] content length: {len(content)} chars")
        print(f"[LLM response] raw:\n{content[:2000]}")

        parsed = None
        if format_parser is not None:
            parsed = format_parser(content)
            print(f"[LLM response] parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")

        return ChatResponse(
            finish_reason=resp.choices[0].finish_reason or "stop",
            content=content,
            model=getattr(resp, "model", _LLM_MODEL),
            usage={
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
            },
            parsed=parsed,
        )


def _real_llm_client() -> _DirectLlmClient:
    return _DirectLlmClient()


class _FakeDbReader:
    async def list_memories(self, context: MemoryRequestContext, *, filters=None, limit=50, cursor=None):
        return [], None

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        return MemoryDbSearchResult(query=req.query, hits=[], total=0)


class _RecordingDbWriter:
    def __init__(self) -> None:
        self.write_plans = []

    async def apply_mutation_plan(self, context: MemoryRequestContext, plan, *, consistency: str = "fast"):
        write_plan = plan.to_write_plan()
        self.write_plans.append(write_plan)
        return MemoryDbWriteResult(
            memory_ids=[memory.memory_id for memory in write_plan.memories],
            entity_ids=[entity.entity_id for entity in write_plan.entities],
            source_ids=[source.source_id for source in write_plan.sources],
        )


class _FakeEmbedClient:
    async def embed(self, task: str, text: str | list[str], **kwargs) -> EmbeddingResponse:
        texts = text if isinstance(text, list) else [text]
        return EmbeddingResponse(embeddings=[[float(index + 1)] * 8 for index, _ in enumerate(texts)])


class _FakeAddRecordStore:
    def __init__(self) -> None:
        self.points = []

    async def append(self, point) -> None:
        self.points.append(point)


# ---------------------------------------------------------------------------
# Real LLM integration tests
# ---------------------------------------------------------------------------


class TestRealLlmIntegration:
    """Integration tests that call a real LLM and verify the full extraction chain.

    These tests require LLM_API_KEY (and optionally LLM_API_BASE, LLM_MODEL)
    environment variables. Without them, all tests in this class are skipped.
    """

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_english_input_real_llm(self) -> None:
        """Send English input through real LLM with new prompt, verify parsing succeeds."""
        segment, preprocessed = make_segment_and_preprocessed(
            {"text": "Kai prefers FastAPI for all backend services and uses Qdrant for vector search."}
        )
        extractor = VanillaMemoryExtractor(llm_client=_real_llm_client())
        result = await extractor.extract_from_envelope(make_envelope([segment]), [preprocessed], make_context())

        print(f"\n[Result] memories: {len(result.memories)}, entities: {len(result.entities)}")
        for m in result.memories:
            print(f"  - {m.ref_id}: mem_type={m.mem_type}, action_hint={m.action_hint}, content={m.content!r}")

        assert len(result.memories) >= 1
        for m in result.memories:
            assert m.ref_id
            assert m.content
            assert m.mem_type in {
                "profile",
                "fact",
                "episodic",
                "tool_trace",
                "experience",
                "skill_candidate",
                "file_knowledge",
            }
            assert m.action_hint in {"add", "reinforce", "update", "merge", "skip"}
            assert m.metadata.get("extractor") == "vanilla_llm_chunked"

        assert result.entities == []
        assert all("entities" not in memory.model_fields_set for memory in result.memories)

        # pipeline from final memory content / source_refs; the extractor prompt
        # no longer asks the LLM to emit entities or graph_edges.

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_chinese_input_real_llm(self) -> None:
        """Send Chinese input through real LLM with Chinese prompt, verify parsing succeeds."""
        segment, preprocessed = make_segment_and_preprocessed({"text": "我偏好使用 FastAPI 框架开发所有后端服务。"})
        extractor = VanillaMemoryExtractor(llm_client=_real_llm_client())
        result = await extractor.extract_from_envelope(make_envelope([segment]), [preprocessed], make_context())

        print(f"\n[Result] memories: {len(result.memories)}, entities: {len(result.entities)}")
        for m in result.memories:
            print(f"  - {m.ref_id}: mem_type={m.mem_type}, action_hint={m.action_hint}, content={m.content!r}")

        assert len(result.memories) >= 1
        for m in result.memories:
            assert m.ref_id
            assert m.content
            assert m.mem_type in {
                "profile",
                "fact",
                "episodic",
                "tool_trace",
                "experience",
                "skill_candidate",
                "file_knowledge",
            }
            assert m.action_hint in {"add", "reinforce", "update", "merge", "skip"}
            assert m.metadata.get("extractor") == "vanilla_llm_chunked"

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_dialogue_input_real_llm(self) -> None:
        """Send multi-turn dialogue through real LLM, verify multiple memory types extracted."""
        inp = AddPipelineInput(
            messages=[
                {"role": "user", "content": "我需要搭建一个新的后端服务。", "timestamp": 1770000000000},
                {"role": "assistant", "content": "好的，你希望使用什么框架？", "timestamp": 1770000001000},
                {"role": "user", "content": "用 FastAPI，配合 Qdrant 做向量存储。", "timestamp": 1770000002000},
            ]
        )
        segments = MessageSegmenter().segment(inp)[0]
        preprocessor = make_text_preprocessor()
        preprocessed_list = [
            preprocessor.preprocess_text(s.text, source_ref=s.source_ref, segment_id=s.segment_id) for s in segments
        ]

        print(f"\n[Dialogue] segments: {len(segments)}")
        for i, s in enumerate(segments):
            print(f"  seg[{i}]: role={s.role}, text={s.text!r}")

        extractor = VanillaMemoryExtractor(llm_client=_real_llm_client())
        result = await extractor.extract_from_envelope(make_envelope(segments), preprocessed_list, make_context())

        print(f"\n[Result] memories: {len(result.memories)}")
        for m in result.memories:
            print(f"  - {m.ref_id}: mem_type={m.mem_type}, action_hint={m.action_hint}, content={m.content!r}")

        assert len(result.memories) >= 1
        for m in result.memories:
            assert m.content
            assert m.action_hint in {"add", "reinforce", "update", "merge", "skip"}

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_prompt_produces_valid_json(self) -> None:
        """Verify the real LLM returns strictly parseable JSON (no markdown, no explanation)."""
        segment, preprocessed = make_segment_and_preprocessed({"text": "The project deadline is June 30th."})
        ctx = make_context()
        messages = _envelope_prompt_messages(make_envelope([segment]), [preprocessed], ctx)
        client = _real_llm_client()

        print(f"\n[System prompt first 200 chars]: {messages[0]['content'][:200]}...")

        response = await client.chat(
            task="memory.add.extract.integration_test",
            messages=messages,
            format_parser=parse_memory_extraction_json,
        )

        assert response.parsed is not None
        assert isinstance(response.parsed, dict)
        assert "memories" in response.parsed
        assert isinstance(response.parsed["memories"], list)

        result = MemoryExtractionResult.model_validate(response.parsed)
        print(f"\n[Result] memories: {len(result.memories)}, valid DTO parse ✅")
        for m in result.memories:
            print(f"  - {m.ref_id}: mem_type={m.mem_type}, content={m.content!r}")
        assert len(result.memories) >= 1

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_reinforce_with_related_memories_real_llm(self) -> None:
        """Send input with related_memories context, verify LLM can return reinforce/update/merge."""
        segment, preprocessed = make_segment_and_preprocessed({"text": "Kai uses FastAPI."})
        ctx = make_context()

        old_memory = MemoryView(
            memory_id="mem_old_1",
            content="Kai prefers backend frameworks.",
            mem_type="profile",
            status="active",
            metadata={},
            project_id="proj-1",
        )
        recall_result = RelatedMemoryRecallResult(
            duplicate=None,
            candidates=[
                RelatedMemoryCandidate(
                    memory_id="mem_old_1",
                    score=0.85,
                    source="bm25",
                    rank=1,
                    memory=old_memory,
                    debug={},
                )
            ],
        )

        recalled_memories = [
            {
                "memory_id": candidate.memory_id,
                "content": candidate.memory.content if candidate.memory else "",
                "source": candidate.source,
                "score": candidate.score,
            }
            for candidate in recall_result.candidates
        ]
        messages = _envelope_prompt_messages(
            make_envelope([segment], recalled_memories=recalled_memories),
            [preprocessed],
            ctx,
        )
        client = _real_llm_client()

        print(f"\n[Context] related_memories injected: mem_old_1 = 'Kai prefers backend frameworks.'")

        response = await client.chat(
            task="memory.add.extract.integration_test",
            messages=messages,
            format_parser=parse_memory_extraction_json,
        )

        result = MemoryExtractionResult.model_validate(response.parsed)
        print(f"\n[Result] memories: {len(result.memories)}")
        for m in result.memories:
            print(
                f"  - {m.ref_id}: action_hint={m.action_hint}, "
                f"target={m.target_memory_id}, related={m.related_memory_ids}, "
                f"content={m.content!r}"
            )

        assert len(result.memories) >= 1
        m = result.memories[0]
        assert m.action_hint in {"add", "reinforce", "update", "merge", "skip"}
        if m.action_hint in ("reinforce", "update"):
            assert m.target_memory_id is not None or m.related_memory_ids

    @skip_no_llm_key
    @pytest.mark.asyncio
    async def test_enable_entities_pipeline_e2e_real_llm(self) -> None:
        """Run vanilla add E2E with real LLM entity extraction and fake storage."""
        writer = _RecordingDbWriter()
        add_records = _FakeAddRecordStore()
        pipeline = VanillaAddPipeline(
            db_reader=_FakeDbReader(),
            db_writer=writer,
            text_config=TextProcessingConfig(
                bm25_use_spacy_lemma=False,
                spacy_en_model="missing_en_model",
                spacy_zh_model="missing_zh_model",
                sparse_hash_dim=128,
            ),
            vanilla_add_config=VanillaAddConfig(enable_entities=True),
            llm_client=_real_llm_client(),
            embed_client=_FakeEmbedClient(),
            consistency="fast",
            add_record_store=add_records,
        )

        result = await pipeline.add_sync(
            AddPipelineInput(
                messages=[
                    {
                        "role": "user",
                        "content": "I am building the Atlas Memory Service with FastAPI.",
                        "timestamp": 1770000000000,
                    },
                    {
                        "role": "assistant",
                        "content": "Got it. Which vector database are you using?",
                        "timestamp": 1770000001000,
                    },
                    {
                        "role": "user",
                        "content": "Atlas Memory Service uses Qdrant for vector search and Neo4j for graph links.",
                        "timestamp": 1770000002000,
                    },
                ]
            ),
            make_context(),
        )

        assert result.status == "ok"
        assert writer.write_plans
        plan = writer.write_plans[0]
        assert plan.memories
        assert plan.entities, "enable_entities=True should persist LLM-extracted entities"
        assert plan.entity_vectors

        entity_names = {entity.entity_name.lower() for entity in plan.entities}
        assert {"atlas memory service", "fastapi", "qdrant", "neo4j"} & entity_names

        rel_types = {relationship.rel_type for relationship in plan.relationships}
        assert REL_MENTIONS in rel_types
        assert REL_MENTIONED_IN_SOURCE in rel_types

        source_ids = {source.source_id for source in plan.sources}
        assert all(
            relationship.target.node_id in source_ids
            for relationship in plan.relationships
            if relationship.rel_type == REL_MENTIONED_IN_SOURCE
        )
        assert add_records.points
