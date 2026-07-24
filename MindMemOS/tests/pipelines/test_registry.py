import pytest
from mindmemos.pipelines.registry import create_pipeline, register

from mindmemos.config import init_config, reset_config


def test_register_decorator_registers_pipeline_class_for_factory_creation() -> None:
    @register(type="add", name="unit_test_add")
    class UnitTestAddPipeline:
        def __init__(self, *, marker: str) -> None:
            self.marker = marker

    pipeline = create_pipeline(type="add", name="unit_test_add", marker="created")

    assert isinstance(pipeline, UnitTestAddPipeline)
    assert pipeline.marker == "created"


def test_register_rejects_duplicate_pipeline_name_for_same_type() -> None:
    @register(type="search", name="unit_test_duplicate_search")
    class FirstSearchPipeline:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register(type="search", name="unit_test_duplicate_search")
        class SecondSearchPipeline:
            pass


def test_create_pipeline_rejects_unknown_pipeline_name() -> None:
    with pytest.raises(ValueError, match="Unknown search pipeline"):
        create_pipeline(type="search", name="missing_search")


def test_builtin_get_delete_update_pipelines_are_registered() -> None:
    from mindmemos.components.extractor.vanilla._safety_gate import AddSafetyGate as NestedAddSafetyGate
    from mindmemos.pipelines.add.schema import SchemaAddPipeline
    from mindmemos.pipelines.add.vanilla.vanilla_add import VanillaAddPipeline as NestedVanillaAddPipeline
    from mindmemos.pipelines.delete.default import DefaultDeletePipeline
    from mindmemos.pipelines.get.default import DefaultGetPipeline
    from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
    from mindmemos.pipelines.update.default import DefaultUpdatePipeline

    from mindmemos.components.extractor.vanilla import AddSafetyGate
    from mindmemos.pipelines.add.vanilla import VanillaAddPipeline

    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")

        assert AddSafetyGate is NestedAddSafetyGate
        assert VanillaAddPipeline is NestedVanillaAddPipeline
        assert isinstance(create_pipeline(type="add", name="vanilla_add"), VanillaAddPipeline)
        assert isinstance(create_pipeline(type="search", name="search_pipeline"), SearchPipelineImpl)
        assert isinstance(create_pipeline(type="add", name="schema_add"), SchemaAddPipeline)
        assert isinstance(create_pipeline(type="get", name="default_get"), DefaultGetPipeline)
        assert isinstance(create_pipeline(type="delete", name="default_delete"), DefaultDeletePipeline)
        assert isinstance(create_pipeline(type="update", name="default_update"), DefaultUpdatePipeline)
    finally:
        reset_config()
