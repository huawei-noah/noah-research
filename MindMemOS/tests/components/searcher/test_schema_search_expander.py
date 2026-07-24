from __future__ import annotations

from typing import Any

import pytest
from mindmemos.typing.llm import EmbeddingResponse, RerankHit, RerankResponse
from mindmemos.typing.memory import (
    EntityView,
    FieldCondition,
    MemoryRequestContext,
    MemoryView,
    SearchFilter,
)
from mindmemos.typing.memory_db import MemoryDbSearchHit, MemoryDbSearchQuery, MemoryDbSearchResult

from mindmemos.components.memory_modeling.schema import TemporalEntity
from mindmemos.components.searcher.schema import SchemaSearchExpander
from mindmemos.config.algo.search.schema import (
    DualPathConfig,
    EdgeSearchConfig,
    PropertySearchConfig,
    SchemaSearchConfig,
)


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeEmbedClient:
    async def embed(self, *, task: str, text: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[[0.1, 0.2]], model="fake")


class FakeDbReader:
    def __init__(self) -> None:
        self.dense_queries: list[MemoryDbSearchQuery] = []
        self.neighbor_limits: list[int | None] = []

    async def search_dense(
        self,
        ctx: MemoryRequestContext,
        query: MemoryDbSearchQuery,
        *,
        query_vector: list[float],
    ) -> MemoryDbSearchResult:
        self.dense_queries.append(query)
        hits = [
            _hit("mem-episode-input", entity_type="episodes", property_name="input_messages", score=0.99),
            _hit("mem-episode-summary", entity_type="episodes", property_name="summary", score=0.8),
            _hit("mem-user-pref", entity_type="person", property_name="preference", score=0.7),
        ]
        return MemoryDbSearchResult(query=query.query, hits=hits, total=len(hits))

    async def get_entity_neighbors(
        self,
        ctx: MemoryRequestContext,
        entity_id: str,
        *,
        direction: str = "both",
        rel_type: str | None = None,
        limit: int | None = None,
    ) -> list[EntityView]:
        self.neighbor_limits.append(limit)
        return []


class FakeRerankClient:
    available = True
    has_external_model = True

    async def rerank(self, query: str, documents: list[str], top_n: int) -> RerankResponse:
        return RerankResponse(
            results=[
                RerankHit(index=-1, relevance_score=1.0),
                RerankHit(index=0, relevance_score=0.9),
                RerankHit(index=2, relevance_score=0.8),
            ],
            model="fake",
        )


@pytest.mark.asyncio
async def test_property_store_excludes_episode_input_messages_from_dual_path() -> None:
    db_reader = FakeDbReader()
    expander = SchemaSearchExpander(
        db_reader=db_reader,
        embed_client=FakeEmbedClient(),
        config=SchemaSearchConfig(),
    )

    results = await expander._search_from_property_store(make_context(), "what happened")

    assert [(entity.entity_type, sorted(entity._properties)) for entity in results] == [
        ("episodes", ["summary"]),
        ("person", ["preference"]),
    ]
    assert db_reader.dense_queries[0].filters is not None
    assert _has_episode_input_messages_exclusion(db_reader.dense_queries[0].filters)


@pytest.mark.asyncio
async def test_property_store_ignores_negative_rerank_indices() -> None:
    db_reader = FakeDbReader()
    expander = SchemaSearchExpander(
        db_reader=db_reader,
        embed_client=FakeEmbedClient(),
        rerank_client=FakeRerankClient(),
        config=SchemaSearchConfig(),
    )

    results = await expander._search_from_property_store(make_context(), "what happened")

    assert [(entity.entity_type, sorted(entity._properties)) for entity in results] == [
        ("episodes", ["summary"]),
    ]


@pytest.mark.asyncio
async def test_multi_hop_passes_neighbor_fetch_limit_to_db_reader(monkeypatch) -> None:
    db_reader = FakeDbReader()
    config = SchemaSearchConfig(
        dual_path=DualPathConfig(enabled=False),
        edge=EdgeSearchConfig(neighbor_fetch_limit=7),
        property=PropertySearchConfig(use_property_extension=False),
        use_entity_agent_search=False,
    )
    expander = SchemaSearchExpander(
        db_reader=db_reader,
        embed_client=FakeEmbedClient(),
        config=config,
    )

    async def fake_entity_store(*args: Any, **kwargs: Any) -> list[TemporalEntity]:
        return [TemporalEntity(entity_id="ent-1", name="Kai", entity_type="person")]

    monkeypatch.setattr(expander, "_search_from_entity_store", fake_entity_store)

    await expander.search(make_context(), "who knows Kai?", num_hops=2)

    assert db_reader.neighbor_limits == [7]


def _hit(memory_id: str, *, entity_type: str, property_name: str, score: float) -> MemoryDbSearchHit:
    return MemoryDbSearchHit(
        memory_id=memory_id,
        score=score,
        memory=MemoryView(
            memory_id=memory_id,
            project_id="proj-1",
            content=f"{entity_type}:{property_name}",
            mem_type="episodic" if entity_type == "episodes" else "fact",
            status="active",
            property_name=property_name,
            entity_id=f"entity-{entity_type}",
            entity_type=entity_type,
            metadata={
                "entity_name": "Episode" if entity_type == "episodes" else "User",
                "property_time": "2026-01-01 00:00:00",
            },
        ),
    )


def _has_episode_input_messages_exclusion(search_filter: SearchFilter) -> bool:
    for clause in search_filter.must_not:
        if isinstance(clause, SearchFilter) and _has_conditions(
            clause.must,
            [
                ("entity_type", "episodes"),
                ("property_name", "input_messages"),
            ],
        ):
            return True
        if isinstance(clause, SearchFilter) and _has_episode_input_messages_exclusion(clause):
            return True
    for clause in [*search_filter.must, *search_filter.should]:
        if isinstance(clause, SearchFilter) and _has_episode_input_messages_exclusion(clause):
            return True
    return False


def _has_conditions(clauses: list[FieldCondition | SearchFilter], expected: list[tuple[str, str]]) -> bool:
    found = {
        (clause.field, clause.value)
        for clause in clauses
        if isinstance(clause, FieldCondition) and clause.op == "match"
    }
    return all(item in found for item in expected)
