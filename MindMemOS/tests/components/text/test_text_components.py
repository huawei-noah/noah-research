from datetime import UTC, datetime

import pytest
from mindmemos.components.text import (
    Bm25TextAnalyzer,
    ContentHasher,
    InMemoryCorpusStatsProvider,
    LanguageAwareEntityExtractor,
    LanguageDetector,
    SparseVectorEncoder,
    TextNormalizer,
    TextPreprocessor,
)
from mindmemos.components.text.vectorizer import MemoryVectorizer
from mindmemos.config import TextProcessingConfig
from mindmemos.errors import EmbeddingDimensionError
from mindmemos.typing.llm import EmbeddingResponse
from mindmemos.typing.memory import EntityWrite, SourceRef


def make_config(**overrides) -> TextProcessingConfig:
    base = {
        "bm25_use_spacy_lemma": False,
        "spacy_en_model": "missing_en_model",
        "spacy_zh_model": "missing_zh_model",
    }
    base.update(overrides)
    return TextProcessingConfig(**base)


def test_normalize_hash_and_language_detection() -> None:
    cfg = make_config()
    normalizer = TextNormalizer(cfg)
    normalized = normalizer.normalize("  用户\u200b 喜欢   FastAPI\n")

    assert normalized == "用户 喜欢 FastAPI"
    assert ContentHasher(cfg).hash_text(normalized) == ContentHasher(cfg).hash_text(normalized)

    lang = LanguageDetector(cfg).detect(normalized)
    assert lang.lang == "mixed"
    assert lang.zh_ratio > 0
    assert lang.latin_ratio > 0


def test_bm25_text_analyzer_uses_terms_not_llm_tokens() -> None:
    analyzer = Bm25TextAnalyzer(make_config())

    result = analyzer.analyze("Building FastAPI services with Qdrant", "en")

    assert result.term_count == len(result.terms)
    assert "fastapi" in result.terms
    assert "qdrant" in result.terms
    assert result.bm25_text


def test_sparse_encoder_supports_fallback_and_bm25_stats() -> None:
    cfg = make_config(sparse_hash_dim=128)
    encoder = SparseVectorEncoder(cfg)
    stats_provider = InMemoryCorpusStatsProvider()
    stats_provider.observe_document("project", "mem-1", ["fastapi", "qdrant", "qdrant"])
    stats_provider.observe_document("project", "mem-2", ["neo4j", "qdrant"])

    fallback = encoder.encode_query(["qdrant"])
    stats = stats_provider.get_stats("project", ["qdrant"])
    bm25 = encoder.encode_query(["qdrant"], stats)

    assert fallback.model == cfg.sparse_fallback_model_name
    assert bm25.model == cfg.sparse_bm25_model_name
    assert bm25.indices == sorted(bm25.indices)
    assert len(bm25.indices) == len(bm25.values)
    assert stats.document_frequency["qdrant"] == 2


def test_entity_extractor_falls_back_to_rules_when_spacy_is_unavailable() -> None:
    extractor = LanguageAwareEntityExtractor(make_config(max_entity_count=8))

    entities = extractor.extract('Project uses FastAPI in "Memory Service"', "en")
    names = {entity.name for entity in entities}

    assert "FastAPI" in names or "Memory Service" in names


def test_text_preprocessor_outputs_shared_preprocessed_text_dto() -> None:
    cfg = make_config(max_entity_count=8)
    preprocessor = TextPreprocessor(cfg)

    result = preprocessor.preprocess_text("用户使用 FastAPI 和 Qdrant。")

    assert result.normalized_text == "用户使用 FastAPI 和 Qdrant。"
    assert result.content_hash
    assert result.bm25_text
    assert result.tokens
    assert result.metadata["term_count"] == len(result.tokens)


def test_text_preprocessor_many_matches_single_outputs() -> None:
    cfg = make_config(max_entity_count=8)
    preprocessor = TextPreprocessor(cfg)
    texts = [
        "Kai uses FastAPI with Qdrant.",
        "用户使用 FastAPI 和 Qdrant。",
        'Project stores notes in "Memory Service".',
    ]
    source_refs = [
        SourceRef(source_type="message", message_id=f"message-{index}", is_parsed=True) for index in range(len(texts))
    ]
    segment_ids = [f"segment-{index}" for index in range(len(texts))]

    singles = [
        preprocessor.preprocess_text(text, source_ref=source_ref, segment_id=segment_id)
        for text, source_ref, segment_id in zip(texts, source_refs, segment_ids, strict=True)
    ]
    batched = preprocessor.preprocess_many(texts, source_refs=source_refs, segment_ids=segment_ids)

    assert [item.model_dump() for item in batched] == [item.model_dump() for item in singles]

    single_without_entities = [preprocessor.preprocess_text(text, include_entities=False) for text in texts]
    batched_without_entities = preprocessor.preprocess_many(texts, include_entities=False)

    assert [item.model_dump() for item in batched_without_entities] == [
        item.model_dump() for item in single_without_entities
    ]


def test_bm25_analyze_many_uses_spacy_pipe_without_changing_output() -> None:
    class FakeToken:
        def __init__(self, text: str) -> None:
            self.lemma_ = text.lower()
            self.is_alpha = text.isalpha()
            self.is_stop = False

    class FakeNlp:
        def __init__(self) -> None:
            self.pipe_calls: list[list[str]] = []

        def __call__(self, text: str):
            return [FakeToken(part) for part in text.split()]

        def pipe(self, texts: list[str]):
            self.pipe_calls.append(list(texts))
            return [self(text) for text in texts]

    texts = ["Kai builds FastAPI", "Qdrant stores memory"]
    cfg = make_config(bm25_use_spacy_lemma=True)
    batch_analyzer = Bm25TextAnalyzer(cfg)
    single_analyzer = Bm25TextAnalyzer(cfg)
    fake_batch_nlp = FakeNlp()
    batch_analyzer._spacy_en = fake_batch_nlp
    single_analyzer._spacy_en = FakeNlp()

    batched = batch_analyzer.analyze_many(texts, ["en", "en"])
    singles = [single_analyzer.analyze(text, "en") for text in texts]

    assert [item.model_dump() for item in batched] == [item.model_dump() for item in singles]
    assert fake_batch_nlp.pipe_calls == [texts]


@pytest.mark.asyncio
async def test_vectorize_many_batches_memory_embeddings_in_chunks_of_10() -> None:
    class RecordingEmbedClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | list[str]]] = []

        async def embed(self, task: str, text: str | list[str], **kwargs):
            self.calls.append((task, text))
            texts = text if isinstance(text, list) else [text]
            embeddings = [[float(item.removeprefix("memory-"))] for item in texts]
            return EmbeddingResponse(embeddings=embeddings)

    cfg = make_config(sparse_hash_dim=128)
    preprocessor = TextPreprocessor(cfg)
    embed_client = RecordingEmbedClient()
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=embed_client,
        text_preprocessor=preprocessor,
    )
    items = [
        (f"mem-{index}", preprocessor.preprocess_text(f"memory-{index}"), f"memory-{index}") for index in range(17)
    ]

    vectors, pending = await vectorizer.vectorize_many(items)

    assert pending == [False] * 17
    assert embed_client.calls == [
        ("memory.add.embed", [f"memory-{index}" for index in range(10)]),
        ("memory.add.embed", [f"memory-{index}" for index in range(10, 17)]),
    ]
    assert [vector.memory_id for vector in vectors] == [f"mem-{index}" for index in range(17)]
    assert [vector.semantic_vector for vector in vectors] == [[float(index)] for index in range(17)]
    assert all(vector.bm25_indices for vector in vectors)


@pytest.mark.asyncio
async def test_vectorize_many_falls_back_to_single_embeddings_after_batch_failure() -> None:
    class BatchFailingEmbedClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | list[str]]] = []

        async def embed(self, task: str, text: str | list[str], **kwargs):
            self.calls.append((task, text))
            if isinstance(text, list):
                raise RuntimeError("batch unavailable")
            if text == "memory-1":
                raise RuntimeError("single unavailable")
            return EmbeddingResponse(embeddings=[[float(text.removeprefix("memory-"))]])

    cfg = make_config(sparse_hash_dim=128)
    preprocessor = TextPreprocessor(cfg)
    embed_client = BatchFailingEmbedClient()
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=embed_client,
        text_preprocessor=preprocessor,
    )
    items = [(f"mem-{index}", preprocessor.preprocess_text(f"memory-{index}"), f"memory-{index}") for index in range(3)]

    vectors, pending = await vectorizer.vectorize_many(items, consistency="fast")

    assert embed_client.calls == [
        ("memory.add.embed", ["memory-0", "memory-1", "memory-2"]),
        ("memory.add.embed", "memory-0"),
        ("memory.add.embed", "memory-1"),
        ("memory.add.embed", "memory-2"),
    ]
    assert pending == [False, True, False]
    assert [vector.semantic_vector for vector in vectors] == [[0.0], None, [2.0]]


@pytest.mark.asyncio
async def test_vectorize_many_falls_back_for_short_batch_response() -> None:
    class ShortBatchEmbedClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | list[str]]] = []

        async def embed(self, task: str, text: str | list[str], **kwargs):
            self.calls.append((task, text))
            if isinstance(text, list):
                return EmbeddingResponse(embeddings=[[0.0]])
            return EmbeddingResponse(embeddings=[[float(text.removeprefix("memory-"))]])

    cfg = make_config(sparse_hash_dim=128)
    preprocessor = TextPreprocessor(cfg)
    embed_client = ShortBatchEmbedClient()
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=embed_client,
        text_preprocessor=preprocessor,
    )
    items = [(f"mem-{index}", preprocessor.preprocess_text(f"memory-{index}"), f"memory-{index}") for index in range(3)]

    vectors, pending = await vectorizer.vectorize_many(items)

    assert embed_client.calls == [
        ("memory.add.embed", ["memory-0", "memory-1", "memory-2"]),
        ("memory.add.embed", "memory-1"),
        ("memory.add.embed", "memory-2"),
    ]
    assert pending == [False, False, False]
    assert [vector.semantic_vector for vector in vectors] == [[0.0], [1.0], [2.0]]


@pytest.mark.asyncio
async def test_vectorize_entities_batches_single_entity_core_and_search_field_embeddings() -> None:
    class RecordingEmbedClient:
        def __init__(self) -> None:
            self.calls = []

        async def embed(self, task: str, text: str | list[str], **kwargs):
            self.calls.append((task, text))
            texts = text if isinstance(text, list) else [text]
            return EmbeddingResponse(embeddings=[[float(index)] for index, _ in enumerate(texts, start=1)])

    cfg = make_config(sparse_hash_dim=128)
    embed_client = RecordingEmbedClient()
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=embed_client,
        text_preprocessor=TextPreprocessor(cfg),
    )
    entity = EntityWrite(
        entity_id="ent-1",
        account_id="acct",
        project_id="project",
        api_key_uuid="key",
        user_id="user",
        session_id="session",
        entity_name="Ada Lovelace",
        entity_type="person",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"search_fields": ["first programmer", "analytical engine"]},
    )

    vectors, pending = await vectorizer.vectorize_entities([entity])

    assert pending is False
    assert embed_client.calls == [
        (
            "memory_vectorizer.add.entity",
            [
                "Ada Lovelace person first programmer analytical engine",
                "first programmer",
                "analytical engine",
            ],
        )
    ]
    assert [vector.entity_id for vector in vectors] == ["ent-1", "ent-1#sf0", "ent-1#sf1"]
    assert [vector.semantic_vector for vector in vectors] == [[1.0], [2.0], [3.0]]


@pytest.mark.asyncio
async def test_vectorize_entities_propagates_embedding_dimension_error_in_fast_mode() -> None:
    # An embedding dimension mismatch is a fundamental config/environment error,
    # not a transient failure; it must surface even in fast consistency mode
    # instead of being swallowed into vector_pending (which would cause silent
    # data loss and a false-success add).
    class DimensionErrorEmbedClient:
        async def embed(self, task: str, text: str | list[str], **kwargs):
            raise EmbeddingDimensionError(expected=1024, actual=2, model="embedding", task=task)

    cfg = make_config(sparse_hash_dim=128)
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=DimensionErrorEmbedClient(),
        text_preprocessor=TextPreprocessor(cfg),
    )
    entity = EntityWrite(
        entity_id="ent-1",
        account_id="acct",
        project_id="project",
        api_key_uuid="key",
        user_id="user",
        session_id="session",
        entity_name="Ada Lovelace",
        entity_type="person",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"search_fields": ["first programmer"]},
    )

    with pytest.raises(EmbeddingDimensionError):
        await vectorizer.vectorize_entities([entity], consistency="fast")


@pytest.mark.asyncio
async def test_vectorize_entities_batches_all_entities_and_search_fields() -> None:
    class RecordingEmbedClient:
        def __init__(self) -> None:
            self.calls = []

        async def embed(self, task: str, text: str | list[str], **kwargs):
            self.calls.append((task, text))
            texts = text if isinstance(text, list) else [text]
            return EmbeddingResponse(embeddings=[[float(index)] for index, _ in enumerate(texts, start=1)])

    cfg = make_config(sparse_hash_dim=128)
    embed_client = RecordingEmbedClient()
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=embed_client,
        text_preprocessor=TextPreprocessor(cfg),
    )
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    entities = [
        EntityWrite(
            entity_id="ent-1",
            account_id="acct",
            project_id="project",
            api_key_uuid="key",
            user_id="user",
            session_id="session",
            entity_name="Ada Lovelace",
            entity_type="person",
            created_at=created_at,
            metadata={"search_fields": ["first programmer", "analytical engine"]},
        ),
        EntityWrite(
            entity_id="ent-2",
            account_id="acct",
            project_id="project",
            api_key_uuid="key",
            user_id="user",
            session_id="session",
            entity_name="Qdrant",
            entity_type="product",
            created_at=created_at,
            metadata={"search_fields": ["vector database"]},
        ),
    ]

    vectors, pending = await vectorizer.vectorize_entities(entities)

    assert pending is False
    assert embed_client.calls == [
        (
            "memory_vectorizer.add.entity",
            [
                "Ada Lovelace person first programmer analytical engine",
                "first programmer",
                "analytical engine",
                "Qdrant product vector database",
                "vector database",
            ],
        )
    ]
    assert [vector.entity_id for vector in vectors] == ["ent-1", "ent-1#sf0", "ent-1#sf1", "ent-2", "ent-2#sf0"]
    assert [vector.semantic_vector for vector in vectors] == [[1.0], [2.0], [3.0], [4.0], [5.0]]


@pytest.mark.asyncio
async def test_vectorize_entities_marks_pending_when_batch_response_is_short() -> None:
    class ShortEmbedClient:
        async def embed(self, task: str, text: str | list[str], **kwargs):
            return EmbeddingResponse(embeddings=[[1.0]])

    cfg = make_config(sparse_hash_dim=128)
    vectorizer = MemoryVectorizer(
        sparse_encoder=SparseVectorEncoder(cfg),
        embed_client=ShortEmbedClient(),
        text_preprocessor=TextPreprocessor(cfg),
    )
    entity = EntityWrite(
        entity_id="ent-1",
        account_id="acct",
        project_id="project",
        api_key_uuid="key",
        user_id="user",
        session_id="session",
        entity_name="Ada Lovelace",
        entity_type="person",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"search_fields": ["first programmer", "analytical engine"]},
    )

    vectors, pending = await vectorizer.vectorize_entities([entity])

    assert pending is True
    assert [vector.semantic_vector for vector in vectors] == [[1.0], None, None]
