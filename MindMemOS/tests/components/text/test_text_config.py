import pytest
from mindmemos.components.text import get_text_preprocessor
from mindmemos.config import TextProcessingConfig, bind_config_overrides, build_config, init_config, reset_config
from omegaconf.errors import ReadonlyConfigError


def test_text_processing_config_defaults_are_available() -> None:
    cfg = TextProcessingConfig()

    assert cfg.spacy_en_model == "en_core_web_sm"
    assert cfg.sparse_hash_dim > 0
    assert cfg.entity_fallback_on_empty is True


def test_get_text_preprocessor_reuses_explicit_config() -> None:
    cfg = TextProcessingConfig(spacy_en_model="missing_en_model", spacy_zh_model="missing_zh_model")

    assert get_text_preprocessor(cfg) is get_text_preprocessor(cfg)


def test_build_config_includes_text_processing_section() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml")

    assert cfg.algo_config.common.prompt_language == "EN"
    assert cfg.algo_config.text_processing.spacy_en_model == "en_core_web_sm"
    assert cfg.algo_config.text_processing.sparse_bm25_model_name == "hash_bm25_v1"


def test_text_processing_config_rejects_request_scoped_override() -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")

        with pytest.raises(ReadonlyConfigError):
            with bind_config_overrides(
                project_config={"algo_config": {"text_processing": {"bm25_use_spacy_lemma": False}}}
            ):
                pass

        assert get_text_preprocessor() is get_text_preprocessor()
    finally:
        reset_config()


def test_build_config_includes_pipeline_selection_section() -> None:
    cfg = build_config(config_path="config/mindmemos/dev.example.yaml")

    assert cfg.pipelines["get"] == "default_get"
    assert cfg.pipelines.delete == "default_delete"
    assert cfg.pipelines["update"] == "default_update"
