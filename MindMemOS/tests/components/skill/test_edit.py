"""Unit tests for the line-addressed SKILL.md edit-op applier."""

from __future__ import annotations

import json

import pytest
from mindmemos.components.skill import apply_edit_ops, apply_patch_ops, format_numbered, parse_edit_ops

from mindmemos.errors import SkillEditError

# Lines (1-based): 1 "# Title", 2 "", 3 "Alpha line.", 4 "Beta line.", 5 "Gamma line."
SKILL = "# Title\n\nAlpha line.\nBeta line.\nGamma line.\n"


def test_format_numbered_gutter():
    out = format_numbered(SKILL)
    assert "1| # Title" in out
    assert "4| Beta line." in out
    # gutter is right-aligned to the widest number; here single digit
    assert "5| Gamma line." in out


def test_format_numbered_empty():
    assert format_numbered("") == "(empty document)"


def test_replace_single_line():
    out = apply_edit_ops(SKILL, [{"op": "replace", "start": 4, "end": 4, "new": "Beta CHANGED."}])
    assert "Beta CHANGED." in out
    assert "Beta line." not in out
    assert "Alpha line.\n" in out and "Gamma line." in out


def test_replace_multi_line():
    out = apply_edit_ops(SKILL, [{"op": "replace", "start": 3, "end": 5, "new": "Only line."}])
    assert out == "# Title\n\nOnly line.\n"


def test_delete_removes_lines():
    out = apply_edit_ops(SKILL, [{"op": "delete", "start": 4, "end": 4}])
    assert "Beta line." not in out
    assert "Alpha line." in out and "Gamma line." in out


def test_insert_after_line():
    out = apply_edit_ops(SKILL, [{"op": "insert", "after": 3, "new": "Delta line."}])
    assert "Alpha line.\nDelta line.\nBeta line." in out


def test_insert_at_top_and_end():
    assert apply_edit_ops(SKILL, [{"op": "insert", "after": 0, "new": "HEAD"}]).startswith("HEAD\n")
    assert apply_edit_ops(SKILL, [{"op": "insert", "after": 5, "new": "TAIL"}]).endswith("TAIL\n")


def test_new_without_trailing_newline_does_not_fuse():
    # 'new' lacks a trailing newline; the applier owns separators so it must not
    # fuse onto the following line.
    out = apply_edit_ops(SKILL, [{"op": "insert", "after": 3, "new": "Mid"}])
    assert "Alpha line.\nMid\nBeta line." in out


def test_insert_after_final_line_without_trailing_newline_does_not_fuse():
    out = apply_edit_ops("Last line without newline", [{"op": "insert", "after": 1, "new": "Appended"}])
    assert out == "Last line without newline\nAppended\n"


def test_empty_edits_is_noop():
    assert apply_edit_ops(SKILL, []) == SKILL


def test_multi_edit_applied_against_original_numbers():
    # Both ops reference ORIGINAL line numbers; bottom-up apply keeps them valid.
    edits = [
        {"op": "replace", "start": 3, "end": 3, "new": "First."},
        {"op": "insert", "after": 5, "new": "Last."},
    ]
    out = apply_edit_ops(SKILL, edits)
    assert "First." in out
    assert out.endswith("Last.\n")


def test_old_first_line_match_ok():
    out = apply_edit_ops(SKILL, [{"op": "replace", "start": 4, "end": 4, "new": "B.", "old_first_line": "Beta line."}])
    assert "B." in out


def test_old_first_line_tolerates_whitespace():
    out = apply_edit_ops(
        SKILL, [{"op": "replace", "start": 4, "end": 4, "new": "B.", "old_first_line": "  Beta   line.  "}]
    )
    assert "B." in out


def test_old_first_line_mismatch_raises():
    with pytest.raises(SkillEditError, match="does not match"):
        apply_edit_ops(SKILL, [{"op": "delete", "start": 4, "end": 4, "old_first_line": "Gamma line."}])


def test_old_string_prefix_match_ok_without_full_line_copy():
    skill = "First line.\n" + "Very long line " + "x" * 200 + "\nLast line.\n"
    out = apply_edit_ops(
        skill,
        [{"op": "replace", "start": 2, "end": 2, "new": "Short.", "old_string_prefix": "Very long line"}],
    )
    assert out == "First line.\nShort.\nLast line.\n"


def test_old_string_prefix_tolerates_whitespace():
    out = apply_edit_ops(
        SKILL,
        [{"op": "replace", "start": 4, "end": 4, "new": "B.", "old_string_prefix": "  Beta   "}],
    )
    assert "B." in out


def test_old_string_prefix_mismatch_raises():
    with pytest.raises(SkillEditError, match="old_string_prefix"):
        apply_edit_ops(SKILL, [{"op": "delete", "start": 4, "end": 4, "old_string_prefix": "Gamma"}])


def test_line_out_of_range_raises():
    with pytest.raises(SkillEditError, match="out of range"):
        apply_edit_ops(SKILL, [{"op": "replace", "start": 99, "end": 99, "new": "x"}])


def test_inverted_range_raises():
    with pytest.raises(SkillEditError, match="out of range"):
        apply_edit_ops(SKILL, [{"op": "replace", "start": 4, "end": 2, "new": "x"}])


def test_overlapping_ranges_raise():
    edits = [
        {"op": "replace", "start": 3, "end": 4, "new": "x"},
        {"op": "delete", "start": 4, "end": 5},
    ]
    with pytest.raises(SkillEditError, match="overlapping"):
        apply_edit_ops(SKILL, edits)


def test_unknown_op_raises():
    with pytest.raises(SkillEditError, match="unknown op"):
        apply_edit_ops(SKILL, [{"op": "rewrite", "start": 3, "end": 3, "new": "x"}])


def test_missing_line_field_raises():
    with pytest.raises(SkillEditError, match="missing required integer"):
        apply_edit_ops(SKILL, [{"op": "replace", "new": "x"}])


def test_bool_is_not_a_valid_line_number():
    with pytest.raises(SkillEditError, match="missing required integer"):
        apply_edit_ops(SKILL, [{"op": "insert", "after": True, "new": "x"}])


def test_empty_insert_new_raises():
    with pytest.raises(SkillEditError, match="insert 'new'"):
        apply_edit_ops(SKILL, [{"op": "insert", "after": 3, "new": ""}])


def test_parse_accepts_edits_object_bare_list_and_fence():
    assert parse_edit_ops('{"edits": [{"op": "delete", "start": 1, "end": 1}]}') == [
        {"op": "delete", "start": 1, "end": 1}
    ]
    assert parse_edit_ops('[{"op": "delete", "start": 1, "end": 1}]') == [{"op": "delete", "start": 1, "end": 1}]
    fenced = "```json\n" + json.dumps({"edits": []}) + "\n```"
    assert parse_edit_ops(fenced) == []


def test_parse_rejects_non_json_and_wrong_shape():
    with pytest.raises(SkillEditError):
        parse_edit_ops("not json")
    with pytest.raises(SkillEditError, match="list"):
        parse_edit_ops('{"foo": 1}')


def test_apply_patch_ops_end_to_end():
    raw = json.dumps({"edits": [{"op": "replace", "start": 4, "end": 4, "new": "B."}]})
    assert "B." in apply_patch_ops(SKILL, raw)
