import json

from mindmemos.components.memory_modeling.schema import EntityManager


def test_entity_manager_loads_list_schema_and_queries_properties(tmp_path) -> None:
    schema_path = tmp_path / "entity_modeling.json"
    schema_path.write_text(
        json.dumps(
            [
                {
                    "entity_type": "user",
                    "entity_description": "The primary user.",
                    "entity_instruction": "Use complete sentences.",
                    "search_weight": 1.0,
                    "static_property": {"name": "User name"},
                    "dynamic_property": {
                        "preference": {
                            "type": "string",
                            "order": 1,
                            "desc": "User preference.",
                            "example": "As of 2024, User likes concise docs.",
                        },
                        "preference_summary": {
                            "type": "string",
                            "order": 2,
                            "desc": "Higher-order preference summary.",
                            "example": "As of 2024, User prefers clear technical writing.",
                        },
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manager = EntityManager(schema_path)

    assert manager.list_types() == ["user"]
    assert manager.get_dict("user")["entity_description"] == "The primary user."
    assert set(manager.get_properties_by_order("user", order=1)) == {"preference"}
    assert manager.get_higher_order_property_names("user") == {"preference_summary"}


def test_entity_manager_register_merge_and_save(tmp_path) -> None:
    schema_path = tmp_path / "entity_modeling.json"
    schema_path.write_text("[]", encoding="utf-8")
    manager = EntityManager(schema_path)

    manager.register(
        "organization",
        entity_description="Organizations.",
        static_property={"name": "Organization name"},
        dynamic_property={"membership_event": {"type": "string", "desc": "Membership changes."}},
    )
    manager.register(
        "organization",
        dynamic_property={"location_info": {"type": "string", "desc": "Location information."}},
        merge=True,
    )
    saved_path = manager.save_to_file()

    reloaded = EntityManager(saved_path)
    organization = reloaded.get_dict("organization")

    assert organization is not None
    assert set(organization["dynamic_property"]) == {"membership_event", "location_info"}
    assert not reloaded.is_dirty()
