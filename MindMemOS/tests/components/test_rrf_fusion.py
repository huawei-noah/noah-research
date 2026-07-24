from mindmemos.components.searcher.entity_recall import combine_entity_results_rrf


def test_entity_rrf_keeps_best_search_field_after_fusion() -> None:
    results = combine_entity_results_rrf(
        vector_entities=[
            {
                "entity_id": "ent-1",
                "score": 0.2,
                "best_search_field": "weak dense field",
                "entity_view": object(),
            },
            {
                "entity_id": "ent-2",
                "score": 0.9,
                "best_search_field": "other field",
                "entity_view": object(),
            },
        ],
        bm25_entities=[
            {
                "entity_id": "ent-1",
                "score": 0.95,
                "best_search_field": "best sparse field",
                "entity_view": object(),
            }
        ],
        rrf_k=60,
        top_k=10,
    )

    ent_one = next(result for result in results if result["entity_id"] == "ent-1")
    assert ent_one["best_search_field"] == "best sparse field"
    assert ent_one["best_search_field_source"] == "bm25"
    assert [result["entity_id"] for result in results].count("ent-1") == 1
