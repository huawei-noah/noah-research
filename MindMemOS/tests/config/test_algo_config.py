from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from mindmemos.config import (
    EpisodesChunkerConfig,
    MemoryAlgoConfig,
    SchemaAddConfig,
    SearchConfig,
    TextProcessingConfig,
    VanillaAddConfig,
    VanillaSearchConfig,
    build_config,
    default_config_path,
)
from mindmemos.config.app import ModelEndpointConfig
from mindmemos.llm.router import build_litellm_params


def test_dev_example_declares_all_algorithm_config_fields() -> None:
    raw_algo_config = _load_dev_example_algo_config()

    _assert_declares_fields(raw_algo_config, MemoryAlgoConfig)
    _assert_declares_fields(raw_algo_config["add"]["schema"], SchemaAddConfig)
    _assert_declares_fields(raw_algo_config["add"]["vanilla"], VanillaAddConfig)
    _assert_declares_fields(raw_algo_config["add"]["schema"]["chunker"], EpisodesChunkerConfig)
    _assert_declares_fields(raw_algo_config["text_processing"], TextProcessingConfig)
    _assert_declares_fields(raw_algo_config["search"], SearchConfig)
    _assert_declares_fields(raw_algo_config["search"]["vanilla"], VanillaSearchConfig)


def test_dev_configs_keep_vanilla_settings_in_canonical_sections() -> None:
    legacy_flat_alias = "non" + "_schema"
    for path in ("config/mindmemos/dev.yaml", "config/mindmemos/dev.example.yaml"):
        raw_algo_config = _load_algo_config(path)

        assert "vanilla_add" not in raw_algo_config
        assert "vanilla_search" not in raw_algo_config
        assert "vanilla" in raw_algo_config["add"]
        assert "vanilla" in raw_algo_config["search"]
        assert legacy_flat_alias not in raw_algo_config.get("add", {})
        assert legacy_flat_alias not in raw_algo_config.get("search", {})

        cfg = build_config(config_path=path).algo_config
        assert cfg.add.vanilla.chunk_hard_token_budget == 32000
        assert cfg.search.request_top_k_max == 100
        assert cfg.search.vanilla.recall_size == 20


def test_dev_config_name_resolves_to_mindmemos_dir() -> None:
    assert default_config_path("dev") == Path("config/mindmemos/dev.yaml").resolve()


def test_memory_algo_config_no_longer_exposes_top_level_vanilla_sections() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml").algo_config

    assert not hasattr(cfg, "vanilla_add")
    assert not hasattr(cfg, "vanilla_search")


def test_search_config_no_longer_exposes_legacy_flat_alias() -> None:
    legacy_flat_alias = "non" + "_schema"
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml").algo_config.search

    assert not hasattr(cfg, legacy_flat_alias)


def test_schema_add_defaults_align_with_original_generation_config() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml").algo_config

    schema = cfg.add.schema
    assert schema.merge.entity_recall_top_k == 15
    assert schema.merge.max_merge_retries == 8
    assert schema.merge.use_property_merge is False
    assert schema.extraction.use_search_fields is True
    assert schema.extraction.search_fields_max == 10
    assert schema.extraction.episode_search_fields_augment is True
    assert schema.extraction.episode_augment_count == 4
    assert schema.higher_order.enabled is True
    assert schema.higher_order.top_k == 10
    assert schema.higher_order.min_evidence_count == 2
    assert schema.episode_edge.top_k == 10


def test_dreaming_config_declares_concurrency() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml").algo_config.dreaming

    assert cfg.concurrency == 8


def test_config_docs_document_algorithm_config_fields() -> None:
    doc = Path("docs/config/README.md").read_text(encoding="utf-8")

    assert _missing_doc_fields(doc, MemoryAlgoConfig) == []
    assert _missing_doc_fields(doc, EpisodesChunkerConfig) == []
    assert _missing_doc_fields(doc, TextProcessingConfig) == []
    assert _missing_doc_fields(doc, VanillaAddConfig) == []
    assert _missing_doc_fields(doc, VanillaSearchConfig) == []
    # Connection endpoints/credentials remain the only env-configurable fields.
    assert "MINDMEMOS_QDRANT_URL" in doc
    assert "MINDMEMOS_NEO4J_URI" in doc


def test_litellm_params_omit_unconfigured_temperature() -> None:
    endpoint = ModelEndpointConfig(model="gpt-test", api_key="sk-test", api_base="https://example.test/v1")

    params = build_litellm_params(endpoint)

    assert "temperature" not in params


def test_litellm_params_preserve_explicit_temperature() -> None:
    endpoint = ModelEndpointConfig(
        model="gpt-test",
        api_key="sk-test",
        api_base="https://example.test/v1",
        temperature=0.7,
    )

    params = build_litellm_params(endpoint)

    assert params["temperature"] == 0.7


def test_locomo_schema_sets_router_failure_cooldown_policy() -> None:
    cfg = build_config(config_path="config/locomo_schema.yaml")

    for router in (cfg.chat_model_router, cfg.embed_model_router, cfg.rerank_model_router):
        assert router.allowed_fails == 100
        assert router.cool_down == 0
        assert len(router.endpoints) == 3


def _load_dev_example_algo_config() -> dict[str, Any]:
    return _load_algo_config("config/mindmemos/dev.example.yaml")


def _load_algo_config(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["algo_config"]


def _assert_declares_fields(section: dict[str, Any], schema: type) -> None:
    missing = [field.name for field in fields(schema) if field.name not in section]
    assert missing == []


def _missing_doc_fields(doc: str, schema: type) -> list[str]:
    return [field.name for field in fields(schema) if field.name not in doc]
