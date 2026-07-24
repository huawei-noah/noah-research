from mindmemos.prompts import get_add_prompts


def test_add_prompt_selector_keeps_english_and_chinese_prompts() -> None:
    en_prompts = get_add_prompts("EN")
    zh_prompts = get_add_prompts("ZH")

    assert "conversation analysis expert" in en_prompts.conv_boundary_detection
    assert "professional entity and relationship extraction expert" in en_prompts.entity_generation
    assert "higher-order personal traits" in en_prompts.higher_order_generation
    assert "memory property merge expert" in en_prompts.property_merge_decision
    assert "search optimization expert" in en_prompts.search_field_generation
    assert zh_prompts.conv_boundary_detection
    assert zh_prompts.entity_generation
    assert zh_prompts.higher_order_generation
    assert zh_prompts.property_merge_decision
    assert zh_prompts.search_field_generation
    assert zh_prompts.conv_boundary_detection != en_prompts.conv_boundary_detection
