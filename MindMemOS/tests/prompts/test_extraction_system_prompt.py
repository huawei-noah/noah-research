"""Tests for extraction system prompt dispatch by language and entity mode."""

from mindmemos.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT_ENTITY,
    EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH,
    EXTRACTION_SYSTEM_PROMPT_ZH,
    get_extraction_system_prompt,
)


def test_default_returns_base_prompt() -> None:
    assert get_extraction_system_prompt("en") == EXTRACTION_SYSTEM_PROMPT
    assert get_extraction_system_prompt("zh") == EXTRACTION_SYSTEM_PROMPT_ZH
    assert get_extraction_system_prompt("en", enable_entities=False) == EXTRACTION_SYSTEM_PROMPT


def test_enable_entities_returns_entity_prompt() -> None:
    assert get_extraction_system_prompt("en", enable_entities=True) == EXTRACTION_SYSTEM_PROMPT_ENTITY
    assert get_extraction_system_prompt("zh", enable_entities=True) == EXTRACTION_SYSTEM_PROMPT_ENTITY_ZH


def test_entity_prompt_emits_top_level_entities() -> None:
    en = get_extraction_system_prompt("en", enable_entities=True)
    zh = get_extraction_system_prompt("zh", enable_entities=True)
    # memory references entity by ref_id, entity lives in the top-level entities array
    assert '"entities": ["e1"]' in en
    assert '"ref_id": "e1"' in en
    assert "entity_name" in en
    assert "不输出顶层" in zh
