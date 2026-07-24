from mindmemos.components.searcher.schema.property_recall import combine_property_results_rrf


def property_result(memory_id: str, value: str, *, uid: str) -> dict:
    return {
        "memory_id": memory_id,
        "content": value,
        "score": 1.0,
        "metadata": {
            "entity_id": "person-alice",
            "property_name": "preference",
            "property_value": value,
            "timestamp": "2026-01-01 10:00:00",
            "uid": uid,
        },
    }


def test_property_rrf_keeps_distinct_memories_with_same_timestamp() -> None:
    results = combine_property_results_rrf(
        [
            property_result("mem-1", "likes tea", uid="mem-1"),
            property_result("mem-2", "likes jazz", uid="mem-2"),
        ],
        [],
        top_k=10,
    )

    assert [item["memory_id"] for item in results] == ["mem-1", "mem-2"]


def test_property_rrf_merges_same_property_memory_across_channels() -> None:
    results = combine_property_results_rrf(
        [property_result("mem-1", "likes tea", uid="mem-1")],
        [property_result("mem-1", "likes tea", uid="mem-1")],
        top_k=10,
    )

    assert [item["memory_id"] for item in results] == ["mem-1"]
    assert "score_list_0" in results[0]
    assert "score_list_1" in results[0]
