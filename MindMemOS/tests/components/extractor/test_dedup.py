"""Tests for CandidateDeduplicator component."""

from mindmemos.components.extractor.vanilla import CandidateDeduplicator, ExtractedMemoryCandidate


def _candidate(
    content: str = "test content",
    mem_type: str = "fact",
    confidence: float = 0.8,
    entities: list[str] | None = None,
    source_refs: list[str] | None = None,
    content_hash: str = "",
    related_memory_ids: list[str] | None = None,
) -> ExtractedMemoryCandidate:
    return ExtractedMemoryCandidate(
        ref_id=f"m_{id(content)}",
        content=content,
        mem_type=mem_type,
        confidence=confidence,
        entities=entities or [],
        source_refs=source_refs or [],
        related_memory_ids=related_memory_ids or [],
        metadata={"content_hash": content_hash or content},
    )


class TestCandidateDeduplicator:
    def test_empty_batch(self):
        dedup = CandidateDeduplicator()
        assert dedup.dedup([]) == []

    def test_single_candidate_unchanged(self):
        dedup = CandidateDeduplicator()
        c = _candidate(content="hello")
        result = dedup.dedup([c])
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_identical_hash_same_type_merged(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="same", mem_type="fact", confidence=0.7, content_hash="h1")
        c2 = _candidate(content="same", mem_type="fact", confidence=0.9, content_hash="h1")
        result = dedup.dedup([c1, c2])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_different_types_not_merged(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="same", mem_type="fact", content_hash="h1")
        c2 = _candidate(content="same", mem_type="profile", content_hash="h1")
        result = dedup.dedup([c1, c2])
        assert len(result) == 2

    def test_different_hash_not_merged(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="alpha", content_hash="h1")
        c2 = _candidate(content="beta", content_hash="h2")
        result = dedup.dedup([c1, c2])
        assert len(result) == 2

    def test_entity_union_merged(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="same", content_hash="h1", entities=["FastAPI", "Python"])
        c2 = _candidate(content="same", content_hash="h1", entities=["fastapi", "Docker"])
        result = dedup.dedup([c1, c2])
        assert len(result) == 1
        # "FastAPI" and "fastapi" normalize to same, so 3 unique entities
        merged_entities = result[0].entities
        assert len(merged_entities) == 3
        assert "FastAPI" in merged_entities
        assert "Python" in merged_entities
        assert "Docker" in merged_entities

    def test_source_refs_union_merged(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="same", content_hash="h1", source_refs=["s1", "s2"])
        c2 = _candidate(content="same", content_hash="h1", source_refs=["s2", "s3"])
        result = dedup.dedup([c1, c2])
        assert set(result[0].source_refs) == {"s1", "s2", "s3"}

    def test_no_duplicates_returns_same_order(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="a", content_hash="h1")
        c2 = _candidate(content="b", content_hash="h2")
        c3 = _candidate(content="c", content_hash="h3")
        result = dedup.dedup([c1, c2, c3])
        assert len(result) == 3
        assert [r.content for r in result] == ["a", "b", "c"]

    def test_related_memory_ids_union(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="same", content_hash="h1", related_memory_ids=["mem1", "mem2"])
        c2 = _candidate(content="same", content_hash="h1", related_memory_ids=["mem2", "mem3"])
        result = dedup.dedup([c1, c2])
        assert set(result[0].related_memory_ids) == {"mem1", "mem2", "mem3"}

    def test_winner_metadata_preserved(self):
        dedup = CandidateDeduplicator()
        c1 = _candidate(content="same", content_hash="h1", confidence=0.5)
        c1.metadata["key_a"] = "from_c1"
        c2 = _candidate(content="same", content_hash="h1", confidence=0.9)
        c2.metadata["key_b"] = "from_c2"
        result = dedup.dedup([c1, c2])
        # c2 wins (higher confidence), its metadata takes precedence
        assert result[0].metadata.get("key_b") == "from_c2"
        # c1's unique keys are also merged in
        assert result[0].metadata.get("key_a") == "from_c1"
