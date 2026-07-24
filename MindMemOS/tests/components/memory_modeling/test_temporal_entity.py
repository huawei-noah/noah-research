from datetime import UTC, datetime

from mindmemos.typing.memory import REL_RELATED_TO, EntityView, MemoryView

from mindmemos.components.memory_modeling.schema import Edge, EntityManager, TemporalEntity


def test_temporal_entity_keeps_independent_property_timelines(tmp_path) -> None:
    manager = _manager(tmp_path)
    entity = TemporalEntity(
        entity_id="entity-user",
        name="User",
        entity_type="user",
        description="Primary user",
        entity_manager=manager,
    )

    entity.modify_property("preference", "likes Qdrant", "2026-05-01")
    entity.modify_property("preference", "likes Qdrant and Neo4j", "2026-05-03")
    entity.modify_property("plan", "ship schema add", "2026-05-02")

    assert entity.get_properties()["preference"] == "likes Qdrant and Neo4j"
    assert entity.get_property_at_time("preference", "2026-05-02") == "likes Qdrant"
    assert entity.get_timeline() == ["2026-05-01", "2026-05-02", "2026-05-03"]
    assert entity.get_property_history("preference", include_uid=True)[0][2]


def test_temporal_entity_delete_search_fields_edges_and_serialization(tmp_path) -> None:
    manager = _manager(tmp_path)
    entity = TemporalEntity(name="User", entity_type="user", entity_manager=manager)
    uid = entity.insert_property_value("preference", "2026-05-01", "likes Qdrant")

    assert entity.delete_property_value("preference", "2026-05-01", uid=uid) is True
    assert entity.get_properties(property_names=["preference"]) == {"preference": None}

    entity.set_search_fields(["Qdrant", "Qdrant", "Neo4j"], max_fields=2)
    edge = entity.add_edge(target_entity_id="episode-1", target_entity_name="Episode", description="mentioned in")

    assert edge == Edge(
        link_entity1_id=entity.entity_id,
        link_entity1_name="User",
        link_entity2_id="episode-1",
        link_entity2_name="Episode",
        link_description="mentioned in",
    )
    assert entity.search_fields == ["Qdrant", "Neo4j"]
    assert (
        TemporalEntity.from_dict(entity.to_dict(), entity_manager=manager).find_connected_entities()[0]["relation"]
        == "mentioned in"
    )


def test_temporal_entity_hydrates_from_entity_and_memory_views(tmp_path) -> None:
    manager = _manager(tmp_path)
    entity = EntityView(
        entity_id="entity-user",
        project_id="proj-1",
        entity_name="User",
        entity_type="user",
        description="Primary user",
        metadata={"search_fields": ["database preference"]},
    )
    memories = [
        MemoryView(
            memory_id="mem-1",
            project_id="proj-1",
            content="likes Qdrant",
            mem_type="fact",
            status="active",
            property_name="preference",
            entity_id="entity-user",
            entity_type="user",
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            metadata={"property_time": "2026-05-01"},
        ),
        MemoryView(
            memory_id="mem-2",
            project_id="proj-1",
            content="likes databases",
            mem_type="fact",
            status="archived",
            property_name="preference",
            entity_id="entity-user",
            entity_type="user",
        ),
    ]

    modeled = TemporalEntity.from_views(entity, memories, entity_manager=manager)

    assert modeled.search_fields == ["database preference"]
    assert modeled.get_property_history("preference", include_uid=True) == [("2026-05-01", "likes Qdrant", "mem-1")]


def test_temporal_entity_dto_exports_and_edge_relationships(tmp_path) -> None:
    manager = _manager(tmp_path)
    source = EntityView(
        entity_id="entity-user",
        project_id="proj-1",
        entity_name="User",
        entity_type="user",
        description="Primary user",
        metadata={"record_time": "2026-05-01", "search_fields": ["database preference"]},
    )
    target = EntityView(
        entity_id="entity-episode",
        project_id="proj-1",
        entity_name="Episode",
        entity_type="episodes",
    )
    memory = MemoryView(
        memory_id="mem-1",
        project_id="proj-1",
        content="likes Qdrant",
        mem_type="fact",
        status="active",
        property_name="preference",
        entity_id="entity-user",
        entity_type="user",
        metadata={"property_time": "2026-05-01"},
    )

    modeled = TemporalEntity.from_entity_dto(source, entity_manager=manager)
    assert modeled.apply_memory(memory) is True
    modeled.edges.append(Edge.from_entity_dtos(source, target, description="mentioned in"))

    exported = modeled.to_entity_view(project_id="proj-1")
    assert exported.entity_id == source.entity_id
    assert exported.entity_name == source.entity_name
    assert exported.metadata["search_fields"] == ["database preference"]

    [relationship] = modeled.to_graph_relationships(project_id="proj-1")
    assert relationship.rel_type == REL_RELATED_TO
    assert relationship.source.node_id == "entity-user"
    assert relationship.target.node_id == "entity-episode"
    assert relationship.relation_type == "mentioned in"


def test_temporal_entity_old_modeling_filter_and_format_helpers(tmp_path) -> None:
    manager = _manager(tmp_path)
    entity = TemporalEntity(
        entity_id="entity-user",
        name="User",
        entity_type="user",
        description="Primary user",
        record_time="2026-05-01",
        entity_manager=manager,
    )
    entity.modify_property("preference", "likes Qdrant", "2026-05-01", uid="p1")
    entity.modify_property("preference", "likes Neo4j", "2026-05-03", uid="p2")
    entity.modify_property("plan", "ship schema add", "2026-05-02", uid="p3")
    entity.add_edge(target_entity_id="episode-1", target_entity_name="Episode", description="mentioned in")

    assert entity.get_time_range() == ("2026-05-01", "2026-05-03")
    assert entity.get_properties_in_range(["preference"], ("2026-05-02", "2026-05-04"), include_uid=True) == {
        "preference": [{"timestamp": "2026-05-03", "value": "likes Neo4j", "uid": "p2"}]
    }

    filtered = entity.filter_by_time(("2026-05-02", "2026-05-04"), new_entity_id="entity-user-filtered")
    assert filtered.entity_id == "entity-user-filtered"
    assert filtered.get_timeline() == ["2026-05-02", "2026-05-03"]
    assert filtered.find_connected_entities()[0]["target_id"] == "episode-1"

    point_filtered = entity.filter_by_timepoints(["2026-05-01"])
    assert point_filtered.get_properties()["preference"] == "likes Qdrant"
    assert point_filtered.get_properties()["plan"] is None

    prompt = entity.format_entity_prompt()
    assert "Entity: User (Type: user)" in prompt
    assert "Property 'preference':" in prompt
    assert "User --[mentioned in]--> Episode" in prompt
    assert entity.transform_to_dict() == entity.to_dict()


def _manager(tmp_path) -> EntityManager:
    path = tmp_path / "entity_modeling.json"
    path.write_text(
        """
        [
          {
            "entity_type": "user",
            "entity_description": "Primary user",
            "static_property": {"name": "User name"},
            "dynamic_property": {
              "preference": {"type": "string", "order": 1},
              "plan": {"type": "string", "order": 1},
              "preference_summary": {"type": "string", "order": 2}
            }
          }
        ]
        """,
        encoding="utf-8",
    )
    return EntityManager(path)
