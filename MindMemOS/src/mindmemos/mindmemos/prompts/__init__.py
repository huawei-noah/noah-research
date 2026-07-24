"""Shared prompt catalog for components and pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from .EN.add.conv_split import CONV_BOUNDARY_DETECTION_PROMPT, CONV_FORCED_RESPLIT_PROMPT
from .EN.add.entity_generation import ENTITY_GENERATION_PROMPT
from .EN.add.entity_merge import (
    DES_UPDATE_PROMPT,
    DUPLICATE_NAME_RESOLUTION_PROMPT,
    MERGE_DECISION_PROMPT,
    SINGLE_ENTITY_MERGE_PROMPT,
)
from .EN.add.episode_description import EPISODE_DESCRIPTION_PROMPT
from .EN.add.episode_edge import EPISODE_EDGE_PROMPT
from .EN.add.episode_inference import EPISODE_INFERENCE_PROMPT
from .EN.add.episode_objectify import EPISODE_OBJECTIFY_PROMPT
from .EN.add.higher_order_generation import HIGHER_ORDER_GENERATION_PROMPT
from .EN.add.property_delete import PROPERTY_DELETE_DECISION_PROMPT
from .EN.add.property_merge import PROPERTY_MERGE_DECISION_PROMPT
from .EN.add.schema_selection import SCHEMA_SELECTION_FOR_GENERATION_PROMPT
from .EN.add.search_field_generation import (
    EPISODE_SEARCH_FIELD_AUGMENT_PROMPT,
    SEARCH_FIELD_GENERATION_PROMPT,
    SEARCH_FIELD_UPDATE_PROMPT,
)
from .EN.add.vanilla import EXTRACTION_SYSTEM_PROMPT
from .EN.add.vanilla_entity import EXTRACTION_SYSTEM_PROMPT_ENTITY
from .EN.dreaming.action_planning import ACTION_PLANNING_PROMPT
from .EN.dreaming.relation_detection import RELATION_DETECTION_PROMPT
from .EN.retrieve.agentic_retrieve import (
    GLOBAL_PROPERTY_RERANK_PROMPT,
    MULTI_QUERY_GENERATION_PROMPT,
    PROPERTY_FILTER_SELECTION_PROMPT,
    SUFFICIENCY_CHECK_PROMPT,
    TIME_EXTRACTION_PROMPT,
)
from .EN.retrieve.entity_relevance_filter import (
    BATCH_ENTITY_RELEVANCE_PROMPT,
    ENTITY_RELEVANCE_FILTER_PROMPT,
)
from .ZH.add.conv_split import CONV_BOUNDARY_DETECTION_PROMPT as CONV_BOUNDARY_DETECTION_PROMPT_ZH
from .ZH.add.conv_split import CONV_FORCED_RESPLIT_PROMPT as CONV_FORCED_RESPLIT_PROMPT_ZH
from .ZH.add.entity_generation import ENTITY_GENERATION_PROMPT as ENTITY_GENERATION_PROMPT_ZH
from .ZH.add.entity_merge import DES_UPDATE_PROMPT as DES_UPDATE_PROMPT_ZH
from .ZH.add.entity_merge import DUPLICATE_NAME_RESOLUTION_PROMPT as DUPLICATE_NAME_RESOLUTION_PROMPT_ZH
from .ZH.add.entity_merge import MERGE_DECISION_PROMPT as MERGE_DECISION_PROMPT_ZH
from .ZH.add.entity_merge import SINGLE_ENTITY_MERGE_PROMPT as SINGLE_ENTITY_MERGE_PROMPT_ZH
from .ZH.add.episode_description import EPISODE_DESCRIPTION_PROMPT as EPISODE_DESCRIPTION_PROMPT_ZH
from .ZH.add.episode_edge import EPISODE_EDGE_PROMPT as EPISODE_EDGE_PROMPT_ZH
from .ZH.add.episode_inference import EPISODE_INFERENCE_PROMPT as EPISODE_INFERENCE_PROMPT_ZH
from .ZH.add.episode_objectify import EPISODE_OBJECTIFY_PROMPT as EPISODE_OBJECTIFY_PROMPT_ZH
from .ZH.add.higher_order_generation import HIGHER_ORDER_GENERATION_PROMPT as HIGHER_ORDER_GENERATION_PROMPT_ZH
from .ZH.add.property_delete import PROPERTY_DELETE_DECISION_PROMPT as PROPERTY_DELETE_DECISION_PROMPT_ZH
from .ZH.add.property_merge import PROPERTY_MERGE_DECISION_PROMPT as PROPERTY_MERGE_DECISION_PROMPT_ZH
from .ZH.add.schema_selection import (
    SCHEMA_SELECTION_FOR_GENERATION_PROMPT as SCHEMA_SELECTION_FOR_GENERATION_PROMPT_ZH,
)
from .ZH.add.search_field_generation import (
    EPISODE_SEARCH_FIELD_AUGMENT_PROMPT as EPISODE_SEARCH_FIELD_AUGMENT_PROMPT_ZH,
)
from .ZH.add.search_field_generation import SEARCH_FIELD_GENERATION_PROMPT as SEARCH_FIELD_GENERATION_PROMPT_ZH
from .ZH.add.search_field_generation import SEARCH_FIELD_UPDATE_PROMPT as SEARCH_FIELD_UPDATE_PROMPT_ZH
from .ZH.add.vanilla import EXTRACTION_SYSTEM_PROMPT_ZH
from .ZH.add.vanilla_entity import EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH
from .ZH.retrieve.agentic_retrieve import (
    GLOBAL_PROPERTY_RERANK_PROMPT as GLOBAL_PROPERTY_RERANK_PROMPT_ZH,
)
from .ZH.retrieve.agentic_retrieve import (
    MULTI_QUERY_GENERATION_PROMPT as MULTI_QUERY_GENERATION_PROMPT_ZH,
)
from .ZH.retrieve.agentic_retrieve import (
    PROPERTY_FILTER_SELECTION_PROMPT as PROPERTY_FILTER_SELECTION_PROMPT_ZH,
)
from .ZH.retrieve.agentic_retrieve import SUFFICIENCY_CHECK_PROMPT as SUFFICIENCY_CHECK_PROMPT_ZH
from .ZH.retrieve.agentic_retrieve import TIME_EXTRACTION_PROMPT as TIME_EXTRACTION_PROMPT_ZH
from .ZH.retrieve.entity_relevance_filter import (
    BATCH_ENTITY_RELEVANCE_PROMPT as BATCH_ENTITY_RELEVANCE_PROMPT_ZH,
)
from .ZH.retrieve.entity_relevance_filter import (
    ENTITY_RELEVANCE_FILTER_PROMPT as ENTITY_RELEVANCE_FILTER_PROMPT_ZH,
)


@dataclass(frozen=True, slots=True)
class AddPromptSet:
    conv_boundary_detection: str
    conv_forced_resplit: str
    entity_generation: str
    episode_edge: str
    episode_inference: str
    episode_description: str
    episode_objectify: str
    higher_order_generation: str
    property_delete_decision: str
    property_merge_decision: str
    schema_selection_for_generation: str
    search_field_generation: str
    search_field_update: str
    episode_search_field_augment: str
    merge_decision: str
    duplicate_name_resolution: str
    des_update: str
    single_entity_merge: str


def get_add_prompts(language: str | None = None) -> AddPromptSet:
    normalized = (language or "EN").upper()
    if normalized == "ZH":
        return AddPromptSet(
            conv_boundary_detection=CONV_BOUNDARY_DETECTION_PROMPT_ZH,
            conv_forced_resplit=CONV_FORCED_RESPLIT_PROMPT_ZH,
            entity_generation=ENTITY_GENERATION_PROMPT_ZH,
            episode_edge=EPISODE_EDGE_PROMPT_ZH,
            episode_inference=EPISODE_INFERENCE_PROMPT_ZH,
            episode_description=EPISODE_DESCRIPTION_PROMPT_ZH,
            episode_objectify=EPISODE_OBJECTIFY_PROMPT_ZH,
            higher_order_generation=HIGHER_ORDER_GENERATION_PROMPT_ZH,
            property_delete_decision=PROPERTY_DELETE_DECISION_PROMPT_ZH,
            property_merge_decision=PROPERTY_MERGE_DECISION_PROMPT_ZH,
            schema_selection_for_generation=SCHEMA_SELECTION_FOR_GENERATION_PROMPT_ZH,
            search_field_generation=SEARCH_FIELD_GENERATION_PROMPT_ZH,
            search_field_update=SEARCH_FIELD_UPDATE_PROMPT_ZH,
            episode_search_field_augment=EPISODE_SEARCH_FIELD_AUGMENT_PROMPT_ZH,
            merge_decision=MERGE_DECISION_PROMPT_ZH,
            duplicate_name_resolution=DUPLICATE_NAME_RESOLUTION_PROMPT_ZH,
            des_update=DES_UPDATE_PROMPT_ZH,
            single_entity_merge=SINGLE_ENTITY_MERGE_PROMPT_ZH,
        )
    return AddPromptSet(
        conv_boundary_detection=CONV_BOUNDARY_DETECTION_PROMPT,
        conv_forced_resplit=CONV_FORCED_RESPLIT_PROMPT,
        entity_generation=ENTITY_GENERATION_PROMPT,
        episode_edge=EPISODE_EDGE_PROMPT,
        episode_inference=EPISODE_INFERENCE_PROMPT,
        episode_description=EPISODE_DESCRIPTION_PROMPT,
        episode_objectify=EPISODE_OBJECTIFY_PROMPT,
        higher_order_generation=HIGHER_ORDER_GENERATION_PROMPT,
        property_delete_decision=PROPERTY_DELETE_DECISION_PROMPT,
        property_merge_decision=PROPERTY_MERGE_DECISION_PROMPT,
        schema_selection_for_generation=SCHEMA_SELECTION_FOR_GENERATION_PROMPT,
        search_field_generation=SEARCH_FIELD_GENERATION_PROMPT,
        search_field_update=SEARCH_FIELD_UPDATE_PROMPT,
        episode_search_field_augment=EPISODE_SEARCH_FIELD_AUGMENT_PROMPT,
        merge_decision=MERGE_DECISION_PROMPT,
        duplicate_name_resolution=DUPLICATE_NAME_RESOLUTION_PROMPT,
        des_update=DES_UPDATE_PROMPT,
        single_entity_merge=SINGLE_ENTITY_MERGE_PROMPT,
    )


@dataclass(frozen=True, slots=True)
class SearchPromptSet:
    sufficiency_check: str
    multi_query_generation: str
    property_filter_selection: str
    time_extraction: str
    global_property_rerank: str
    entity_relevance_filter: str
    batch_entity_relevance: str


def get_extraction_system_prompt(lang: str, *, enable_entities: bool = False) -> str:
    """Return extraction system prompt for the given language.

    When ``enable_entities`` is True, return the variant that emits the
    top-level ``entities`` array referenced from each memory via ref_id.
    """
    if enable_entities:
        return EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH if lang == "zh" else EXTRACTION_SYSTEM_PROMPT_ENTITY
    if lang == "zh":
        return EXTRACTION_SYSTEM_PROMPT_ZH
    return EXTRACTION_SYSTEM_PROMPT


def get_search_prompts(language: str | None = None) -> SearchPromptSet:
    normalized = (language or "EN").upper()
    if normalized == "ZH":
        return SearchPromptSet(
            sufficiency_check=SUFFICIENCY_CHECK_PROMPT_ZH,
            multi_query_generation=MULTI_QUERY_GENERATION_PROMPT_ZH,
            property_filter_selection=PROPERTY_FILTER_SELECTION_PROMPT_ZH,
            time_extraction=TIME_EXTRACTION_PROMPT_ZH,
            global_property_rerank=GLOBAL_PROPERTY_RERANK_PROMPT_ZH,
            entity_relevance_filter=ENTITY_RELEVANCE_FILTER_PROMPT_ZH,
            batch_entity_relevance=BATCH_ENTITY_RELEVANCE_PROMPT_ZH,
        )
    return SearchPromptSet(
        sufficiency_check=SUFFICIENCY_CHECK_PROMPT,
        multi_query_generation=MULTI_QUERY_GENERATION_PROMPT,
        property_filter_selection=PROPERTY_FILTER_SELECTION_PROMPT,
        time_extraction=TIME_EXTRACTION_PROMPT,
        global_property_rerank=GLOBAL_PROPERTY_RERANK_PROMPT,
        entity_relevance_filter=ENTITY_RELEVANCE_FILTER_PROMPT,
        batch_entity_relevance=BATCH_ENTITY_RELEVANCE_PROMPT,
    )


__all__ = [
    "AddPromptSet",
    "ACTION_PLANNING_PROMPT",
    "BATCH_ENTITY_RELEVANCE_PROMPT",
    "BATCH_ENTITY_RELEVANCE_PROMPT_ZH",
    "CONV_BOUNDARY_DETECTION_PROMPT",
    "CONV_BOUNDARY_DETECTION_PROMPT_ZH",
    "CONV_FORCED_RESPLIT_PROMPT",
    "CONV_FORCED_RESPLIT_PROMPT_ZH",
    "DES_UPDATE_PROMPT",
    "DES_UPDATE_PROMPT_ZH",
    "DUPLICATE_NAME_RESOLUTION_PROMPT",
    "DUPLICATE_NAME_RESOLUTION_PROMPT_ZH",
    "ENTITY_GENERATION_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT_ENTITY",
    "EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH",
    "EXTRACTION_SYSTEM_PROMPT_ZH",
    "ENTITY_GENERATION_PROMPT_ZH",
    "ENTITY_RELEVANCE_FILTER_PROMPT",
    "ENTITY_RELEVANCE_FILTER_PROMPT_ZH",
    "EPISODE_DESCRIPTION_PROMPT",
    "EPISODE_DESCRIPTION_PROMPT_ZH",
    "EPISODE_EDGE_PROMPT",
    "EPISODE_EDGE_PROMPT_ZH",
    "EPISODE_INFERENCE_PROMPT",
    "EPISODE_INFERENCE_PROMPT_ZH",
    "EPISODE_OBJECTIFY_PROMPT",
    "EPISODE_OBJECTIFY_PROMPT_ZH",
    "EPISODE_SEARCH_FIELD_AUGMENT_PROMPT",
    "EPISODE_SEARCH_FIELD_AUGMENT_PROMPT_ZH",
    "GLOBAL_PROPERTY_RERANK_PROMPT",
    "GLOBAL_PROPERTY_RERANK_PROMPT_ZH",
    "HIGHER_ORDER_GENERATION_PROMPT",
    "HIGHER_ORDER_GENERATION_PROMPT_ZH",
    "MERGE_DECISION_PROMPT",
    "MERGE_DECISION_PROMPT_ZH",
    "MULTI_QUERY_GENERATION_PROMPT",
    "MULTI_QUERY_GENERATION_PROMPT_ZH",
    "PROPERTY_DELETE_DECISION_PROMPT",
    "PROPERTY_DELETE_DECISION_PROMPT_ZH",
    "PROPERTY_FILTER_SELECTION_PROMPT",
    "PROPERTY_FILTER_SELECTION_PROMPT_ZH",
    "PROPERTY_MERGE_DECISION_PROMPT",
    "PROPERTY_MERGE_DECISION_PROMPT_ZH",
    "RELATION_DETECTION_PROMPT",
    "SCHEMA_SELECTION_FOR_GENERATION_PROMPT",
    "SCHEMA_SELECTION_FOR_GENERATION_PROMPT_ZH",
    "SEARCH_FIELD_GENERATION_PROMPT",
    "SEARCH_FIELD_GENERATION_PROMPT_ZH",
    "SEARCH_FIELD_UPDATE_PROMPT",
    "SEARCH_FIELD_UPDATE_PROMPT_ZH",
    "SINGLE_ENTITY_MERGE_PROMPT",
    "SINGLE_ENTITY_MERGE_PROMPT_ZH",
    "SUFFICIENCY_CHECK_PROMPT",
    "SUFFICIENCY_CHECK_PROMPT_ZH",
    "SearchPromptSet",
    "TIME_EXTRACTION_PROMPT",
    "TIME_EXTRACTION_PROMPT_ZH",
    "get_add_prompts",
    "get_extraction_system_prompt",
    "get_search_prompts",
]
