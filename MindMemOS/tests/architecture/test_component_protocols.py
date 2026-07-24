from mindmemos.components.extractor.protocols import AddRecallStrategy
from mindmemos.components.extractor.schema import SchemaAddExtractor, SchemaExtractionNormalizer
from mindmemos.components.extractor.schema._schema_merge_policy import SchemaMergePolicy
from mindmemos.components.extractor.schema._schema_write_plan import SchemaWritePlanBuilder
from mindmemos.components.extractor.schema.base import (
    SchemaEpisodeExtractor,
    SchemaExtractionNormalizerProtocol,
    SchemaMergePolicyProtocol,
    SchemaSearchFieldExtractorProtocol,
    SchemaWritePlanBuilderProtocol,
)
from mindmemos.components.extractor.schema.search_field import SchemaSearchFieldExtractor
from mindmemos.components.extractor.vanilla.add_recall import RelatedMemoryRecall
from mindmemos.components.memory_modeling.schema import EntitySchemaProvider
from mindmemos.components.searcher.protocols import EntityHydrator, SearchStrategy
from mindmemos.components.searcher.schema import SchemaSearchExpander


def test_schema_add_components_declare_extractor_protocols() -> None:
    assert SchemaEpisodeExtractor in SchemaAddExtractor.__mro__
    assert SchemaExtractionNormalizerProtocol in SchemaExtractionNormalizer.__mro__
    assert SchemaMergePolicyProtocol in SchemaMergePolicy.__mro__
    assert SchemaWritePlanBuilderProtocol in SchemaWritePlanBuilder.__mro__


def test_schema_search_components_declare_searcher_protocols() -> None:
    assert AddRecallStrategy in RelatedMemoryRecall.__mro__
    assert SchemaSearchFieldExtractorProtocol in SchemaSearchFieldExtractor.__mro__
    assert SearchStrategy in SchemaSearchExpander.__mro__
    assert EntityHydrator in SchemaSearchExpander.__mro__


def test_entity_schema_provider_belongs_to_modeling_layer() -> None:
    assert EntitySchemaProvider.__module__ == "mindmemos.components.memory_modeling.schema.base"
