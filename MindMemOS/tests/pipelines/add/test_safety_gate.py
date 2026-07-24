from mindmemos.components.extractor.vanilla import AddSafetyGate
from mindmemos.components.text import TextPreprocessor
from mindmemos.config import TextProcessingConfig


def make_preprocessed(text: str):
    return TextPreprocessor(
        TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        )
    ).preprocess_text(text, segment_id="segment-1")


def test_safety_gate_returns_skip_when_normalized_content_is_empty_or_too_short() -> None:
    action = AddSafetyGate().gate_segment(make_preprocessed("   "))
    assert action.action == "SKIP"
    assert action.reason == "empty_or_too_short"


def test_safety_gate_returns_skip_when_memory_type_is_not_allowed() -> None:
    action = AddSafetyGate(allowed_memory_types={"fact"}).gate_segment(
        make_preprocessed("tool call finished"),
        mem_type="tool_trace",
    )
    assert action.action == "SKIP"
    assert action.reason == "memory_type_not_allowed"


def test_safety_gate_returns_add_and_preserves_tool_trace_memory_type() -> None:
    action = AddSafetyGate().gate_segment(
        make_preprocessed("tool call finished"),
        mem_type="tool_trace",
    )
    assert action.action == "ADD"
    assert action.mem_type == "tool_trace"


def test_safety_gate_skip_hint() -> None:
    action = AddSafetyGate().gate_segment(make_preprocessed("skip me"), action_hint="skip")
    assert action.action == "SKIP"
    assert action.reason == "extractor_skip_hint"


def test_update_hint_with_target_and_high_confidence() -> None:
    """UPDATE: target_memory_id provided + confidence >= threshold → UPDATE."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("update this"),
        action_hint="update",
        confidence=0.85,
        target_memory_id="mem-1",
    )
    assert action.action == "UPDATE"
    assert action.target_memory_id == "mem-1"
    assert action.reason == "extractor_update_hint"


def test_update_hint_without_target_downgrades_to_add() -> None:
    """UPDATE: no target_memory_id → downgrade to ADD."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("update this"),
        action_hint="update",
        confidence=0.9,
    )
    assert action.action == "ADD"
    assert action.reason == "update_no_target"


def test_update_hint_low_confidence_downgrades_to_add() -> None:
    """UPDATE: confidence < threshold → downgrade to ADD."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("update this"),
        action_hint="update",
        confidence=0.5,
        target_memory_id="mem-1",
    )
    assert action.action == "ADD"
    assert action.reason == "update_low_confidence"


def test_merge_hint_with_enough_targets_and_high_confidence() -> None:
    """MERGE: >= 2 related_memory_ids + confidence >= threshold → MERGE."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("merge these"),
        action_hint="merge",
        confidence=0.9,
        related_memory_ids=["mem-1", "mem-2"],
    )
    assert action.action == "MERGE"
    assert action.reason == "extractor_merge_hint"
    assert len(action.related_memory_ids) == 2


def test_merge_hint_insufficient_targets_downgrades_to_add() -> None:
    """MERGE: < 2 related_memory_ids → downgrade to ADD."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("merge these"),
        action_hint="merge",
        confidence=0.9,
        related_memory_ids=["mem-1"],
    )
    assert action.action == "ADD"
    assert action.reason == "merge_insufficient_targets"


def test_merge_hint_low_confidence_downgrades_to_add() -> None:
    """MERGE: confidence < threshold → downgrade to ADD."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("merge these"),
        action_hint="merge",
        confidence=0.5,
        related_memory_ids=["mem-1", "mem-2"],
    )
    assert action.action == "ADD"
    assert action.reason == "merge_low_confidence"


def test_reinforce_hint_with_target() -> None:
    """REINFORCE: target_memory_id provided → REINFORCE."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("reinforce this"),
        action_hint="reinforce",
        target_memory_id="mem-1",
    )
    assert action.action == "REINFORCE"
    assert action.target_memory_id == "mem-1"
    assert action.reason == "extractor_reinforce_hint"


def test_reinforce_hint_without_target_downgrades_to_add() -> None:
    """REINFORCE: no target_memory_id → downgrade to ADD."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("reinforce this"),
        action_hint="reinforce",
    )
    assert action.action == "ADD"
    assert action.reason == "reinforce_no_target"


def test_default_add_action() -> None:
    """No action_hint → ADD."""
    action = AddSafetyGate().gate_segment(make_preprocessed("new fact"))
    assert action.action == "ADD"
    assert action.reason == "extractor_add_hint"


def test_add_hint_with_related_memory_ids() -> None:
    """ADD with related_memory_ids preserves them."""
    action = AddSafetyGate().gate_segment(
        make_preprocessed("new fact"),
        action_hint="add",
        related_memory_ids=["mem-1", "mem-2"],
    )
    assert action.action == "ADD"
    assert action.related_memory_ids == ["mem-1", "mem-2"]
