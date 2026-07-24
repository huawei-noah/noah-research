from mindmemos.components.extractor.vanilla.add_recall import weighted_related_memory_rrf
from mindmemos.typing.memory import MemoryView, RelatedMemoryCandidate


def candidate(memory_id: str, source: str, rank: int) -> RelatedMemoryCandidate:
    return RelatedMemoryCandidate(
        memory_id=memory_id,
        score=1.0,
        source=source,
        rank=rank,
        memory=MemoryView(
            memory_id=memory_id,
            project_id="proj-1",
            content=f"memory {memory_id}",
            mem_type="fact",
            status="active",
        ),
    )


def test_weighted_rrf_merges_duplicate_candidates_with_channel_weights() -> None:
    fused = weighted_related_memory_rrf(
        [
            candidate("mem-1", "bm25", 1),
            candidate("mem-2", "bm25", 2),
            candidate("mem-1", "entity", 1),
        ],
        top_k=2,
        weights={"semantic": 1.5, "bm25": 1.0, "entity": 1.2},
        k=60,
    )

    assert [hit.memory_id for hit in fused] == ["mem-1", "mem-2"]
    assert fused[0].source == "rrf"
    assert fused[0].debug["channels"] == ["bm25", "entity"]
    assert fused[0].score > fused[1].score
