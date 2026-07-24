"""Trace add_sync full pipeline — print data at each phase for manual inspection."""

import json
from datetime import UTC, datetime

from mindmemos.components.extractor.vanilla import AddSafetyGate, CandidateDeduplicator
from mindmemos.components.extractor.vanilla.add_builder import AddCoreBuilder
from mindmemos.components.extractor.vanilla.add_recall import RelatedMemoryRecall
from mindmemos.components.extractor.vanilla.memory import VanillaMemoryExtractor
from mindmemos.components.text import SparseVectorEncoder, TextPreprocessor
from mindmemos.components.text.vectorizer import MemoryVectorizer
from mindmemos.config import TextProcessingConfig
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import AddPipelineInput

# ── helpers ──────────────────────────────────────────────────────────────────

TEXT_CONFIG = TextProcessingConfig(
    bm25_use_spacy_lemma=False,
    spacy_en_model="missing_en_model",
    spacy_zh_model="missing_zh_model",
    sparse_hash_dim=128,
)


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-trace",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def sep(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}\n")


def print_json(label: str, obj) -> None:
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    elif isinstance(obj, (list, dict)):
        data = obj
    else:
        data = str(obj)
    print(f"[{label}]")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str)[:3000])
    print()


# ── main trace ───────────────────────────────────────────────────────────────


async def trace_add_sync() -> None:
    # Input — same as test_add_sync_writes_memory_entities_vectors_and_mentions_relationships
    inp = AddPipelineInput(
        messages=[{"text": 'Kai uses QDRANT in "Memory Service".'}],
    )
    ctx = make_context()
    now = datetime.now(UTC)

    print("=" * 60)
    print("INPUT")
    print("=" * 60)
    print_json("AddPipelineInput", inp)
    print_json("MemoryRequestContext", ctx)

    # Build components (same as VanillaAddPipeline.__init__)
    text_preprocessor = TextPreprocessor(TEXT_CONFIG)
    sparse_encoder = SparseVectorEncoder(TEXT_CONFIG)
    extractor = VanillaMemoryExtractor()  # no LLM → fallback path
    recall = RelatedMemoryRecall(
        db_reader=_FakeReader(),
        sparse_encoder=sparse_encoder,
    )
    vectorizer = MemoryVectorizer(
        sparse_encoder=sparse_encoder,
        embed_client=None,
    )

    builder = AddCoreBuilder(
        text_preprocessor=text_preprocessor,
        memory_extractor=extractor,
        candidate_deduplicator=CandidateDeduplicator(),
        related_memory_recall=recall,
        safety_gate=AddSafetyGate(),
        vectorizer=vectorizer,
    )

    # ── Full build ────────────────────────────────────────────────────────
    sep("Full Build")
    plan, events, update_commands = await builder.build(inp, ctx, consistency="fast")
    print_json("MemoryDbWritePlan", plan)
    print(f"  memories:    {len(plan.memories)}")
    print(f"  entities:    {len(plan.entities)}")
    print(f"  sources:     {len(plan.sources)}")
    print(f"  vectors:     {len(plan.vectors)}")
    print(f"  relationships: {len(plan.relationships)}")
    print(f"  events:      {len(events)}")
    print(f"  update_cmds: {len(update_commands)}")

    for i, mem in enumerate(plan.memories):
        print(f"\n  memory[{i}]:")
        print(f"    memory_id:  {mem.memory_id}")
        print(f"    content:    {mem.content!r}")
        print(f"    mem_type:   {mem.mem_type}")
        print(
            f"    metadata:   { {k: v for k, v in mem.metadata.items() if k in ('content_hash', 'bm25_text', 'lang', 'source_timestamp_ms', 'source_role', 'extractor')} }"
        )

    for i, vec in enumerate(plan.vectors):
        print(f"\n  vector[{i}]:")
        print(f"    memory_id:      {vec.memory_id}")
        print(f"    bm25_indices:   {list(vec.bm25_indices)[:20]}")
        print(f"    semantic_vector: {vec.semantic_vector}")

    for i, event in enumerate(events):
        print(f"\n  event[{i}]:")
        print(f"    operation: {event.operation}")
        print(f"    content:   {event.content!r}")
        print(f"    memory_id: {event.memory_id}")

    # ── Summary ──────────────────────────────────────────────────────────
    sep("SUMMARY")
    print(f"  memories:      {len(plan.memories)}")
    print(f"  entities:      {len(plan.entities)}")
    print(f"  sources:       {len(plan.sources)}")
    print(f"  vectors:       {len(plan.vectors)}")
    print(f"  relationships: {len(plan.relationships)}")
    print(f"  events:        {len(events)}")
    print(f"  update_cmds:   {len(update_commands)}")


# ── Fake DB reader (same as test) ────────────────────────────────────────────


class _FakeReader:
    def __init__(self) -> None:
        self.listed_memories = []
        self.sparse_hits = []

    async def list_memories(self, ctx, *, filters=None, limit=50, cursor=None):
        return self.listed_memories, None

    async def search_sparse(self, ctx, req, *, indices, values):
        from mindmemos.typing.memory_db import MemoryDbSearchResult

        return MemoryDbSearchResult(query=req.query, hits=self.sparse_hits, total=len(self.sparse_hits))


if __name__ == "__main__":
    import asyncio

    asyncio.run(trace_add_sync())
