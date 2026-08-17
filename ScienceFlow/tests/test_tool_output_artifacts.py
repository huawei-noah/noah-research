# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

from pathlib import Path

from scienceflow.core.agent.tools.tool_output_artifacts import (
    ToolOutputArtifactStore,
    attach_tool_output_reference,
    reduce_tool_feedback_for_memory,
)
from scienceflow.core.agent.memory.resource_feedback_memory import (
    RESOURCE_STATE_SUMMARY_MARKER,
    ResourceFeedbackMemoryDeduper,
)


def test_tool_output_artifact_store_writes_txt_and_index(tmp_path: Path) -> None:
    store = ToolOutputArtifactStore(tmp_path)
    raw = "hello\nworld\n"
    ref = store.reserve("bash", raw)
    assert ref is not None
    assert ref.raw_id == "tool_000001_bash.txt"
    assert store.write_raw(ref, raw)

    store.append_index(ref, reducer_name="none", compressed_chars=12)

    out_dir = tmp_path / ".logs" / "tool_outputs"
    assert (out_dir / "tool_000001_bash.txt").read_text() == raw
    idx = (out_dir / "index.txt").read_text()
    assert "seq=1 raw_id=tool_000001_bash.txt tool=bash" in idx
    assert "raw_chars=12 compressed_chars=12" in idx
    assert "reducer=none" in idx


def test_tool_output_artifact_store_mirrors_split_traj_outputs(tmp_path: Path) -> None:
    store = ToolOutputArtifactStore(
        tmp_path,
        output_parts=("interaction", "tool_outputs"),
        mirror_parts=("traj_interaction", "tool_outputs"),
        mirror_raw_id_prefix="S02",
    )
    raw = "full raw output"
    ref = store.reserve("bash", raw)
    assert ref is not None
    assert store.write_raw(ref, raw)
    store.append_index(ref, reducer_name="none", compressed_chars=5)

    local = tmp_path / ".logs" / "interaction" / "tool_outputs"
    traj = tmp_path / ".logs" / "traj_interaction" / "tool_outputs"
    assert (local / "tool_000001_bash.txt").read_text(encoding="utf-8") == raw
    assert (traj / "S02_tool_000001_bash.txt").read_text(encoding="utf-8") == raw
    assert "source_raw_id=tool_000001_bash.txt" in (traj / "index.txt").read_text(encoding="utf-8")


def test_attach_tool_output_reference_keeps_write_first_line_and_snapshot_suffix() -> None:
    feedback = (
        "File `solution.py` written successfully (2 lines, sha256~abc).\n\n"
        "[auto-snapshot after successful write: solution.py]\n"
        "[CANONICAL `solution.py`]\n"
        "     1|print('ok')"
    )

    out = attach_tool_output_reference(
        feedback,
        raw_id="tool_000001_write.txt",
        raw_chars=123,
        reducer="write_snapshot_existing",
    )

    assert out.splitlines()[0] == "File `solution.py` written successfully (2 lines, sha256~abc)."
    assert "[tool-output raw_id=tool_000001_write.txt raw_chars=123 reducer=write_snapshot_existing]" in out
    assert out.index("[tool-output raw_id=") < out.index("[auto-snapshot after successful write:")


def test_bash_reducer_preserves_training_signals_and_tail() -> None:
    lines = ["[exit=0, 99.0s]"]
    lines += [f"progress {i}" for i in range(80)]
    lines += [
        "fold=0 validation rmse=0.1234",
        "best_score=0.1201 best_iteration=42",
        "saved submission.csv rows=240",
    ]
    lines += [f"tail {i}" for i in range(80, 140)]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="bash",
        args={"command": "python3 solution.py"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )

    assert reducer == "bash_training_signal_v2"
    assert len(reduced) < len(feedback)
    assert "validation rmse=0.1234" in reduced
    assert "best_score=0.1201" in reduced
    assert "saved submission.csv rows=240" in reduced
    assert "tail 139" in reduced
    assert "progress 0" not in reduced


def test_bash_reducer_training_failure_preserves_traceback() -> None:
    lines = ["Error: non-zero exit code 1", "[exit=1, 3.0s]"]
    lines += [f"[DATA] Fold {i % 5} metric={0.08 + i / 10000:.5f}" for i in range(30)]
    lines += [
        "[DATASET] Generating submission...",
        "Traceback (most recent call last):",
        '  File "solution.py", line 256, in <module>',
        '    sub_df[target] = test_preds[target].values.astype("float64")',
        "AttributeError: 'numpy.ndarray' object has no attribute 'values'",
    ]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="bash",
        args={"command": "python3 solution.py"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=True,
    )

    assert reducer == "bash_training_signal_v2"
    assert "preserved error/traceback lines" in reduced
    assert "AttributeError" in reduced
    assert "test_preds[target].values" in reduced


def test_bash_reducer_pytest_success_drops_passed_spam() -> None:
    lines = ["[exit=0, 9.0s]"]
    lines += [f"tests/test_{i}.py::test_ok PASSED" for i in range(120)]
    lines += ["======================= 120 passed in 4.20s ======================="]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="bash",
        args={"command": "CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests -vv"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )

    assert reducer == "bash_pytest_summary_v2"
    assert "120 passed" in reduced
    assert "test_0.py::test_ok PASSED" not in reduced


def test_bash_reducer_pytest_failure_keeps_assert_and_summary() -> None:
    lines = ["[exit=1, 5.0s]"]
    lines += [f"tests/test_{i}.py::test_ok PASSED" for i in range(90)]
    lines += [
        "=================================== FAILURES ===================================",
        "FAILED tests/test_model.py::test_score - assert 0.1 < 0.05",
        "E       assert 0.1 < 0.05",
        "==================== 1 failed, 90 passed in 5.00s ====================",
    ]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="bash",
        args={"command": "pytest tests -vv"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=True,
    )

    assert reducer == "bash_pytest_summary_v2"
    assert "FAILED tests/test_model.py::test_score" in reduced
    assert "assert 0.1 < 0.05" in reduced
    assert "1 failed, 90 passed" in reduced
    assert "test_0.py::test_ok PASSED" not in reduced


def test_bash_reducer_install_success_keeps_five_lines_or_less() -> None:
    lines = ["[exit=0, 12.0s]"]
    lines += [f"Downloading package-{i}" for i in range(60)]
    lines += ["Successfully installed a-1.0 b-2.0", "done"]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="bash",
        args={"command": "pip install a b"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )

    assert reducer == "bash_install_summary_v2"
    assert "Successfully installed a-1.0 b-2.0" in reduced
    assert "Downloading package-0" not in reduced
    body = [ln for ln in reduced.splitlines() if not ln.startswith("[")]
    assert len(body) <= 5


def test_read_reducer_keeps_eda_signals() -> None:
    lines = ["[dataset/train.csv: 400 lines total]"]
    lines += [f"{i}| filler" for i in range(90)]
    lines += [
        "shape: (388, 14)",
        "columns: id,target,feature_a,feature_b",
        "missing values: feature_b=3",
    ]
    lines += [f"{i}| tail" for i in range(90, 170)]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="read",
        args={"path": "dataset/train.csv"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )

    assert reducer == "read_eda_v1"
    assert "shape: (388, 14)" in reduced
    assert "columns: id,target,feature_a,feature_b" in reduced
    assert "missing values: feature_b=3" in reduced


def test_grep_reducer_groups_and_folds_by_file() -> None:
    lines = [f"solution.py:{i}:def helper_{i}(): pass" for i in range(1, 9)]
    lines += ["other.py:3:target = 1", "other.py:7:target = 2", "[10 matches]"]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="grep",
        args={"pattern": "target"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )

    assert reducer == "grep_grouped_v2"
    assert "solution.py: 8 matches (lines 1,2,3,4,5,6,7,8)" in reduced
    assert "other.py:" in reduced
    assert "  3| target = 1" in reduced
    assert reduced.count("solution.py:") == 1


def test_glob_ls_reducer_uses_type_groups_without_eda_signals() -> None:
    lines = ["[glob: '**/*' under .]"]
    lines += [f"pkg/path_{i:03d}.py" for i in range(50)]
    lines += [f"cfg/config_{i}.yaml" for i in range(20)]
    lines += [f"data/train_{i}.csv" for i in range(20)]
    lines += [f"pkg/dir_{i}/" for i in range(20)]
    lines += [f"notes/readme_{i}.md" for i in range(50)]
    feedback = "\n".join(lines)

    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="glob",
        args={"pattern": "**/*"},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )

    assert reducer == "glob_ls_typed_v2"
    assert "pkg/path_000.py" in reduced
    assert "cfg/config_0.yaml" in reduced
    assert "data/train_0.csv" in reduced
    assert "pkg/dir_0/" in reduced
    assert "other files omitted" in reduced
    assert "preserved EDA signal lines" not in reduced

    reduced_ls, reducer_ls = reduce_tool_feedback_for_memory(
        tool_name="ls",
        args={"path": ".", "recursive": True},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )
    assert reducer_ls == "glob_ls_typed_v2"
    assert "--- Python files" in reduced_ls


def test_glob_ls_reducer_keeps_small_output_unchanged() -> None:
    feedback = "[ls: .]\nFILE a.py (10 bytes)\n[1 entries]"
    reduced, reducer = reduce_tool_feedback_for_memory(
        tool_name="ls",
        args={"path": "."},
        feedback=feedback,
        raw_text=feedback,
        tool_error=False,
    )
    assert reducer == "none"
    assert reduced == feedback

def test_resource_feedback_dedup_repeats_materially_unchanged_state() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )

    first, first_deduped = deduper.reduce(feedback)
    second, second_deduped = deduper.reduce(feedback)
    third, third_deduped = deduper.reduce(feedback)

    assert first == ""
    assert first_deduped is True
    assert second == ""
    assert second_deduped is True
    assert third == ""
    assert third_deduped is True

    summary = deduper.summary_text()
    assert summary.startswith(RESOURCE_STATE_SUMMARY_MARKER)
    assert summary.count("active_resource_plan_guard") == 1
    assert "status=DENIED_REPLAN" in summary
    assert "blocked=heavy_gpu_train" in summary
    assert "repeat_count=3" in summary


def test_resource_feedback_dedup_counts_multiple_states_in_one_tool_result() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    feedback = (
        "RESOURCE_FEEDBACK: recommend_stop_command because no useful output heartbeat arrived within the configured stalled-output threshold; "
        "requires_llm_decision=true.\n"
        "RESOURCE_FEEDBACK: recommend_stop_command because no useful output heartbeat arrived within the configured stalled-output threshold; "
        "requires_llm_decision=true.\n"
        "RESOURCE_FEEDBACK: recommend_stop_command because no useful output heartbeat arrived within the configured stalled-output threshold; "
        "requires_llm_decision=true.\n"
    )

    reduced, deduped = deduper.reduce(feedback)

    assert reduced == ""
    assert deduped is True
    summary = deduper.summary_text()
    assert "status=RECOMMEND_STOP_COMMAND" in summary
    assert summary.count("stalled-output threshold") == 1
    assert "repeat_count=3" in summary


def test_resource_feedback_dedup_breaks_on_research_state_change() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    first = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; valid_best_score=0.894680.\n"
    )
    changed = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; valid_best_score=0.912300.\n"
    )

    assert deduper.reduce(first) == ("", True)
    assert "valid_best_score=0.894680" in deduper.summary_text()
    assert deduper.reduce(changed) == ("", True)
    assert "valid_best_score=0.912300" in deduper.summary_text()
    repeated, deduped = deduper.reduce(changed)

    assert repeated == ""
    assert deduped is True
    assert "repeat_count=2" in deduper.summary_text()


def test_resource_feedback_dedup_keeps_different_blocked_class_separate() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    train = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train.\n"
    )
    tt = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=gpu_tt_light.\n"
    )

    assert deduper.reduce(train) == ("", True)
    assert deduper.reduce(tt) == ("", True)
    summary = deduper.summary_text()
    assert "blocked=heavy_gpu_train" in summary
    assert "blocked=gpu_tt_light" in summary


def test_resource_feedback_dedup_collapses_worker_deliverable_schema_gate() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    heavy = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because invalid_deliverable_schema_preflight; "
        "mode=RED; scope=worker_deliverable; blocked=heavy_gpu_candidate; "
        "unlock_condition=submission_schema_valid; schema_state=invalid.\n"
    )
    light = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because invalid_deliverable_schema_preflight; "
        "mode=RED; scope=worker_deliverable; blocked=light_gpu_probe; "
        "unlock_condition=submission_schema_valid; schema_state=invalid.\n"
    )

    assert deduper.reduce(heavy) == ("", True)
    reduced, deduped = deduper.reduce(light)

    assert reduced == ""
    assert deduped is True
    summary = deduper.summary_text()
    assert summary.count("invalid_deliverable_schema_preflight") == 1
    assert "scope=worker_deliverable" in summary
    assert "unlock_condition=submission_schema_valid" in summary
    assert "schema_state=invalid" in summary
    assert "repeat_count=2" in summary

