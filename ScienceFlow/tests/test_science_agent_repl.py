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

"""ScienceAgent loop + REPL-oriented behaviour (mock LLM, no API keys)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from click import ClickException
from deepcraft_core import Memory, Message

from scienceflow.core.agent.run_policy import AutoContinuePolicy, DefaultPolicy
from scienceflow.core.agent import ScienceAgent
from scienceflow.core.agent.tools.bash_command_classifier import classify_bash_command
from scienceflow.core.agent.io.interaction_log import (
    format_tool_call_lines_for_interaction_log,
)
from scienceflow.core.agent.runtime.lnr_hooks import _adapt_write_tool_hint_for_bash_only
from scienceflow.core.agent.prompts.system_prompt import _code_agent_core_prompt
from scienceflow.config.settings import Config, apply_profile_overrides, apply_repl_manifest_defaults
from scienceflow.cli import _repl_resolve_runtime_options
from scienceflow.core.orchestrator import (
    _plain_repl_auto_continue_session,
    _repl_bash_file_write_prompt,
    _repl_code_agent_append_prompt,
    _repl_code_organization_prompt,
    _repl_environment_context_prompt,
    _repl_workspace_git_prompt,
)
from scienceflow.core.tools import create_tool_collection
from scienceflow.utils.node_paths import find_node_log_path
from scienceflow.utils.workspace_git import (
    archive_workspace_candidate_artifact,
    auto_checkpoint_workspace_source,
    ensure_workspace_source_git,
    render_workspace_gitignore,
    workspace_source_changed,
)


def test_candidate_artifact_archive_is_metric_and_git_independent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    snapshots = tmp_path / "logs" / "submission_snapshots"
    ledger = tmp_path / "logs" / "checkpoints" / "artifact_archive.jsonl"
    workspace.mkdir()
    submission = workspace / "submission.csv"
    submission.write_text("id,y\n1,0.5\n", encoding="utf-8")

    first = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="submission.csv",
        artifact_kind="submission_csv",
        snapshot_dir=snapshots,
        ledger_path=ledger,
        trigger="bash",
    )
    duplicate = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="submission.csv",
        artifact_kind="submission_csv",
        snapshot_dir=snapshots,
        ledger_path=ledger,
        trigger="read",
    )
    submission.write_text("id,y\n1,0.7\n", encoding="utf-8")
    second = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="submission.csv",
        artifact_kind="submission_csv",
        snapshot_dir=snapshots,
        ledger_path=ledger,
        trigger="bash",
        tool_error=True,
    )

    assert first.archived is True
    assert first.snapshot_path.endswith(".csv")
    assert duplicate.archived is False
    assert duplicate.snapshot_path == first.snapshot_path
    assert second.archived is True
    assert second.snapshot_path != first.snapshot_path
    assert (workspace / first.snapshot_path).read_text(encoding="utf-8") == "id,y\n1,0.5\n"
    assert (workspace / second.snapshot_path).read_text(encoding="utf-8") == "id,y\n1,0.7\n"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["metric_value"] is None
    assert rows[0]["selection_eligible"] is None
    assert rows[1]["tool_error"] is True


def test_candidate_artifact_archive_supports_nested_optimization_json(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifact = workspace / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"score": 2.5}\n', encoding="utf-8")

    result = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="artifacts/best_solution.json",
        artifact_kind="json_solution",
        snapshot_dir=tmp_path / "logs" / "artifact_snapshots",
        ledger_path=tmp_path / "logs" / "checkpoints" / "artifact_archive.jsonl",
        trigger="write",
    )

    assert result.archived is True
    assert result.artifact_kind == "json_solution"
    assert result.snapshot_path.endswith(".json")
    assert (workspace / result.snapshot_path).read_text(encoding="utf-8") == '{"score": 2.5}\n'


def test_candidate_artifact_archive_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.json").write_text("{}\n", encoding="utf-8")

    result = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="../outside.json",
        snapshot_dir=tmp_path / "snapshots",
        ledger_path=tmp_path / "ledger.jsonl",
    )

    assert result.archived is False
    assert "workspace-relative" in result.message


def test_repl_manifest_defaults_merge_repl_config_from_defaults_and_task() -> None:
    cfg = Config()

    apply_repl_manifest_defaults(
        cfg,
        {
            "config": "scienceflow/config/default.yaml",
            "workspace_base": "/tmp/ignored",
            "qa_max_steps": 33,
            "repl_max_steps": 120,
            "repl_tool_preset": "write_edit",
            "repl_bash_max_output_chars": 7000,
        },
        {
            "run_id": "one",
            "cpu_list": "0-3",
            "repl_max_steps": 240,
            "repl_tool_preset": "bash_write",
            "repl_pin_environment_context": False,
            "repl_code_organization_hint": "mle_multifile",
            "repl_workspace_git_enabled": False,
            "repl_workspace_git_track_globs": ["*.py", "*.md", "*.ipynb"],
            "repl_workspace_git_auto_checkpoint": False,
            "repl_workspace_git_auto_review": True,
            "repl_bash_max_stream_line_chars": 900,
            "repl_bash_observation_summary": False,
        },
    )

    assert cfg.qa_max_steps == 33
    assert cfg.repl_max_steps == 240
    assert cfg.repl_tool_preset == "bash_write"
    assert cfg.repl_pin_environment_context is False
    assert cfg.repl_code_organization_hint == "mle_multifile"
    assert cfg.repl_workspace_git_enabled is False
    assert cfg.repl_workspace_git_track_globs == ["*.py", "*.md", "*.ipynb"]
    assert cfg.repl_workspace_git_auto_checkpoint is False
    assert cfg.repl_workspace_git_auto_review is True
    assert cfg.repl_bash_max_output_chars == 7000
    assert cfg.repl_bash_max_stream_line_chars == 900
    assert cfg.repl_bash_observation_summary is False
    assert not hasattr(cfg, "workspace_base")
    assert not hasattr(cfg, "cpu_list")


def test_profile_overrides_keep_unregistered_task_defaults_neutral() -> None:
    cfg = Config()

    apply_profile_overrides(cfg)

    assert cfg.evaluator.task_profile == "auto"
    assert cfg.evaluator.backend == "auto"
    assert cfg.lnr.code_organization_hint == "beyond_mfiles"
    assert cfg.repl_code_organization_hint == ""
    assert cfg.evaluator.candidate.artifact == ""


def test_profile_overrides_isolate_opt_solver_prompt_defaults() -> None:
    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"

    apply_profile_overrides(cfg)

    assert cfg.lnr.code_organization_hint == "opt_solver"
    assert cfg.repl_code_organization_hint == "opt_solver"
    assert cfg.evaluator.candidate.artifact == ""


def test_profile_overrides_allow_configured_opt_solver_parameters() -> None:
    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"
    cfg.profile_overrides = {
        "opt_solver": {
            "lnr": {
                "code_organization_hint": "custom_opt",
                "resource_gpu_queue_enabled": False,
            },
            "repl": {"repl_code_organization_hint": "custom_repl_opt"},
            "evaluator": {
                "backend": "artifact_command",
                "candidate": {
                    "artifact": "artifacts/best_solution.json",
                    "artifact_kind": "json_solution",
                },
            },
        }
    }

    apply_profile_overrides(cfg)

    assert cfg.lnr.code_organization_hint == "custom_opt"
    assert cfg.lnr.resource_gpu_queue_enabled is False
    assert cfg.repl_code_organization_hint == "custom_repl_opt"
    assert cfg.evaluator.backend == "artifact_command"
    assert cfg.evaluator.candidate.artifact == "artifacts/best_solution.json"
    assert cfg.evaluator.candidate.artifact_kind == "json_solution"


def test_partial_profile_override_preserves_builtin_opt_solver_defaults() -> None:
    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"
    cfg.profile_overrides = {
        "opt_solver": {
            "lnr": {"resource_gpu_queue_enabled": False},
        }
    }

    apply_profile_overrides(cfg)

    assert cfg.lnr.code_organization_hint == "opt_solver"
    assert cfg.repl_code_organization_hint == "opt_solver"
    assert cfg.lnr.resource_gpu_queue_enabled is False


def test_explicit_lnr_override_can_win_after_profile_overrides() -> None:
    from scienceflow.config.settings import _apply_parallel_manifest_lnr_payload

    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"
    apply_profile_overrides(cfg)

    _apply_parallel_manifest_lnr_payload(
        cfg.lnr,
        {"code_organization_hint": "mle_multifile"},
        label="lnr",
    )

    assert cfg.lnr.code_organization_hint == "mle_multifile"


def test_repl_runtime_options_default_to_lite_profile() -> None:
    cfg = Config()

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=False,
    )

    assert opts["profile"] == "lite"
    assert opts["tool_preset"] == "bash_write"
    assert opts["max_steps"] == 200
    assert opts["stable_system_prompt"] is True
    assert opts["pin_environment_context"] is True
    assert opts["code_organization_hint"] == ""
    assert opts["workspace_git_enabled"] is True
    assert opts["workspace_git_track_globs"] == ["*.py", "*.md"]
    assert opts["workspace_git_auto_review"] is False
    assert opts["workspace_git_auto_checkpoint"] is True
    assert opts["pin_task_description"] is True
    assert opts["repl_bash_write_mode"] is True
    assert opts["bash_max_output_chars"] == 6000
    assert opts["bash_max_stream_line_chars"] == 1200
    assert opts["bash_observation_summary"] is True




@pytest.mark.parametrize("profile", ["codex_like", "codex", "native", "full_native"])
def test_repl_runtime_options_rejects_removed_profile_aliases(profile: str) -> None:
    cfg = Config()
    cfg.repl_profile = profile

    with pytest.raises(ClickException, match="Invalid repl_profile"):
        _repl_resolve_runtime_options(
            cfg,
            auto_first_user_enabled=False,
        )


def test_repl_runtime_options_auto_first_user_avoids_duplicate_task_pin() -> None:
    cfg = Config()

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["pin_task_description"] is False
    cfg.repl_pin_task_description_when_auto_first_user = True
    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )
    assert opts["pin_task_description"] is True


def test_repl_runtime_options_legacy_and_fallback_steps() -> None:
    cfg = Config()
    cfg.qa_max_steps = 17
    cfg.repl_max_steps = 0
    cfg.repl_profile = "legacy"
    cfg.repl_tool_preset = "write_edit"
    cfg.repl_stable_system_prompt = False
    cfg.repl_pin_environment_context = False

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=False,
    )

    assert opts["profile"] == "legacy"
    assert opts["tool_preset"] == "write_edit"
    assert opts["max_steps"] == 17
    assert opts["stable_system_prompt"] is False
    assert opts["pin_environment_context"] is False
    assert opts["repl_bash_write_mode"] is False
    assert opts["workspace_git_enabled"] is False
    assert opts["workspace_git_auto_checkpoint"] is False


def test_repl_runtime_options_expose_code_organization_hint() -> None:
    cfg = Config()
    cfg.repl_code_organization_hint = "mle_multifile"

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["code_organization_hint"] == "mle_multifile"


def test_repl_runtime_options_expose_workspace_git_auto_review() -> None:
    cfg = Config()
    cfg.repl_workspace_git_auto_review = True

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["workspace_git_auto_review"] is True


def test_repl_runtime_options_expose_workspace_git_auto_checkpoint() -> None:
    cfg = Config()
    cfg.repl_workspace_git_auto_checkpoint = False

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["workspace_git_auto_checkpoint"] is False


def test_repl_code_organization_hint_can_request_mle_multifile_layout() -> None:
    text = _repl_code_organization_prompt("mle_multifile")

    assert "Code organization preference" in text
    assert "`train.py`" in text
    assert "`predict.py`" in text
    assert "`util.py`" in text
    assert "build_features" in text
    assert "one identical feature pipeline" in text
    assert "Do not duplicate feature-engineering code" in text
    assert "submission.csv" in text
    assert "Only add an extra submit/wrapper file" in text
    assert "thin compatible wrapper" in text
    assert "single runnable entrypoint" in text
    assert "solution.py" not in text
    assert _repl_code_organization_prompt("") == ""
    assert _repl_code_organization_prompt("off") == ""


def test_repl_code_organization_hint_can_request_reusable_predict_boundary() -> None:
    text = _repl_code_organization_prompt("beyond_mfiles")

    assert "Code organization preference" in text
    assert "reusable train/predict boundary" in text
    assert "single `solution.py` is acceptable" in text
    assert "`train.py`" in text
    assert "`predict.py`" in text
    assert "`util.py`" in text
    assert "without retraining" in text
    assert "tokenizer/vectorizer" in text
    assert "thresholds" in text
    assert "predict-only iterations" in text
    assert "test-time augmentation" in text
    assert "Do not duplicate train/test feature logic" in text


def test_repl_code_organization_hint_can_request_opt_solver_layout() -> None:
    text = _repl_code_organization_prompt("opt_solver")

    assert "Code organization preference" in text
    assert "solver-oriented workspace files" in text
    assert "`solution.py`" in text
    assert "configured candidate artifact path" in text
    assert "do not invent ML submission files" in text
    assert "resume state" in text
    assert "submission.csv" not in text
    assert "`train.py`" not in text
    assert "`predict.py`" not in text


def test_score_contract_feedback_prefers_finalize_existing_artifacts() -> None:
    from scienceflow.core.agent.run_control.embedded_fullrun import _score_contract_user_message

    text = _score_contract_user_message(
        "missing required `Final Validation Score: <finite_float>` line",
        script_label="train.py",
    )

    assert "[SCORE-CONTRACT-INVALID]" in text
    assert "do not retrain from scratch" in text
    assert "`predict.py` / `score_existing.py`" in text
    assert "loads the existing artifacts" in text
    assert "Final Validation Score" in text
    assert "single-fold progress metric" in text
    assert "cheapest finalization command" in text


def test_repl_workspace_git_prompt_is_short_source_control_hint() -> None:
    text = _repl_workspace_git_prompt(True, ["*.py", "*.md"])

    assert "Workspace source checkpoints" in text
    assert "`git status --short`" in text
    assert "`git diff -- '*.py' '*.md'`" in text
    assert "`git log --oneline" in text
    assert "`git restore --source=<commit>" in text
    assert "Do not create commits manually" in text
    assert "datasets, model weights, logs, and submissions" in text
    assert "`*.py`" in text
    assert "Auto-review is enabled" not in text
    assert _repl_workspace_git_prompt(False, ["*.py"]) == ""

    strong = _repl_workspace_git_prompt(
        True,
        ["*.py", "*.md"],
        auto_review=True,
    )
    assert "Auto-review is enabled" in strong
    assert "restore the best tracked source" in strong
    assert "Do not commit manually" in strong


def test_workspace_source_git_initializes_source_doc_repo(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "data.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "description.md").write_text("# dataset\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.bin").write_bytes(b"model")

    result = ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    assert result.ready is True
    assert result.initialized is True
    assert result.committed is True
    assert (tmp_path / ".git").is_dir()
    ignore_text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert render_workspace_gitignore(["*.py", "*.md"]).strip() in ignore_text

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    assert set(tracked) == {".gitignore", "notes.md", "solution.py"}


def test_workspace_source_git_ignores_runtime_control_markdown(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    (tmp_path / ".run_results.md").write_text("Final Validation Score: 0.9\n", encoding="utf-8")
    assert workspace_source_changed(tmp_path, track_globs=["*.py", "*.md"]) is False
    result = auto_checkpoint_workspace_source(tmp_path, track_globs=["*.py", "*.md"], tool_name="bash")
    assert result.source_changed is False
    assert result.committed is False

    (tmp_path / "solution.py").write_text("print('changed')\n", encoding="utf-8")
    assert workspace_source_changed(tmp_path, track_globs=["*.py", "*.md"]) is True


def test_workspace_source_git_ignores_tmp_runtime_python_files(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    deps = tmp_path / "tmp" / "deps" / "example_pkg"
    deps.mkdir(parents=True)
    (deps / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tmp" / "notes.md").write_text("# runtime note\n", encoding="utf-8")

    assert workspace_source_changed(tmp_path, track_globs=["*.py", "*.md"]) is False
    result = auto_checkpoint_workspace_source(tmp_path, track_globs=["*.py", "*.md"], tool_name="bash")
    assert result.source_changed is False
    assert result.committed is False

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    assert "tmp/deps/example_pkg/__init__.py" not in tracked
    assert "tmp/notes.md" not in tracked


def test_workspace_source_git_auto_checkpoint_commits_and_snapshots(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "util.py").write_text("SEED = 1\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    (tmp_path / "util.py").write_text("SEED = 2\n", encoding="utf-8")
    (tmp_path / "submission.csv").write_text("id,y\n1,0.5\n", encoding="utf-8")
    (tmp_path / "ml_run_results.md").write_text(
        "| run_index | timestamp_utc | command | exit_status | validation_metric | metric_direction | submission_path | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | now | python train.py | 0 | 0.1234 | lower_is_better | submission.csv | ok |\n",
        encoding="utf-8",
    )

    result = auto_checkpoint_workspace_source(
        tmp_path,
        track_globs=["*.py", "*.md"],
        tool_name="bash",
    )

    assert result.ready is True
    assert result.committed is True
    assert result.metric_value == pytest.approx(0.1234)
    assert result.submission_snapshot.endswith(".csv")
    assert (tmp_path / result.submission_snapshot).read_text(encoding="utf-8") == "id,y\n1,0.5\n"
    assert (tmp_path / ".scienceflow_checkpoints" / "ledger.jsonl").is_file()
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    assert "ckpt: metric_0p1234 after bash" in log


def test_workspace_source_git_auto_checkpoint_can_bind_current_stage(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "train.py").write_text("SEED = 1\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    (tmp_path / "submission.csv").write_text("id,y\n1,0.7\n", encoding="utf-8")
    generic_result = auto_checkpoint_workspace_source(
        tmp_path,
        track_globs=["*.py", "*.md"],
        tool_name="bash",
    )

    (tmp_path / "train.py").write_text("SEED = 2\n", encoding="utf-8")
    result = auto_checkpoint_workspace_source(
        tmp_path,
        track_globs=["*.py", "*.md"],
        tool_name="lhr_stage_s01",
        metric_value_override=0.0612,
        stage_id="S01",
    )

    assert result.ready is True
    assert result.committed is True
    assert result.stage_id == "S01"
    assert generic_result.submission_snapshot
    assert result.submission_snapshot != generic_result.submission_snapshot
    assert "_s01_" in result.submission_snapshot
    ledger = tmp_path / ".scienceflow_checkpoints" / "ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert row["stage_id"] == "S01"
    assert row["submission_snapshot"] == result.submission_snapshot
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    assert "ckpt: S01 metric_0p0612 after lhr_stage_s01" in log


class _FakeLLM:
    """Yields pre-baked assistant messages from ``ask_tool_stream``."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        if not self._replies:
            raise RuntimeError("no fake replies left")
        msg = self._replies.pop(0)
        content = getattr(msg, "content", None) or ""
        if content:
            await handle.put(content)
        handle.finish()
        return msg


class _RecordingFakeLLM(_FakeLLM):
    """Like ``_FakeLLM`` but records ``messages=`` passed to ``ask_tool_stream``."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.message_snapshots: list[list[Any]] = []
        self.kwarg_snapshots: list[dict[str, Any]] = []

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        msgs = kwargs.get("messages")
        self.message_snapshots.append(list(msgs) if msgs is not None else [])
        self.kwarg_snapshots.append({k: v for k, v in kwargs.items() if k != "handle"})
        return await super().ask_tool_stream(handle=handle, **kwargs)


@pytest.mark.asyncio
async def test_long_horizon_compact_inband_drops_recent_tail(tmp_path: Path) -> None:
    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="compact summary")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    agent.memory.add_message(Message.assistant_message("assistant-tail-should-drop"))
    agent.memory.add_message(Message.tool_message("tool-tail-should-drop", "bash", "tc1"))

    out = await agent.compact_inband(mid_run=True)

    assert "kept 1 leading + 0 recent" in out
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "FIRST USER QUERY: keep me stable" in text
    assert "compact summary" in text
    assert "assistant-tail-should-drop" not in text
    assert "tool-tail-should-drop" not in text


@pytest.mark.asyncio
async def test_long_horizon_compact_preserves_protected_eda_prefix(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="compact summary")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    agent.memory.add_message(Message.assistant_message("EDA command: inspect train.csv"))
    agent.memory.add_message(Message.tool_message("EDA output: train shape is (1944, 14)", "bash", "eda1"))
    agent.memory.add_message(Message.assistant_message("later exploration should compact away"))
    agent.memory.add_message(Message.tool_message("later output should compact away", "bash", "late1"))

    with caplog.at_level(logging.WARNING):
        info = agent._memory_ctx.set_protected_raw_prefix(3, warn_chars=1, label="test EDA prefix")
    assert info["message_count"] == 2
    assert "exceeding warning threshold 1" in caplog.text

    out = await agent.compact_inband(mid_run=True)

    assert "kept 1 leading + 2 protected raw + 0 recent" in out
    assert agent._memory_ctx._protected_raw_prefix_end_index == 3
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "FIRST USER QUERY: keep me stable" in text
    assert "EDA command: inspect train.csv" in text
    assert "EDA output: train shape is (1944, 14)" in text
    assert "compact summary" in text
    assert "later exploration should compact away" not in text
    assert "later output should compact away" not in text


@pytest.mark.asyncio
async def test_long_horizon_compact_can_replace_protected_eda_with_facts_card(tmp_path: Path) -> None:
    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="compact summary")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
    )
    agent._lnr_compact_on_context_threshold = True
    agent._run_tokens_in = 0
    agent._run_tokens_out = 0
    agent._run_tokens_cached = 0
    agent._run_llm_calls = 0
    agent._run_compact_tokens_in = 0
    agent._run_compact_tokens_out = 0
    agent._run_compact_tokens_cached = 0
    agent._run_compact_llm_calls = 0
    large_scratch = "python3 -c " + repr("print(\'scratch eda\')\n" * 200)
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    agent.memory.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "tc_eda",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": large_scratch}),
                    },
                }
            ],
        )
    )
    agent.memory.add_message(
        Message.tool_message(
            "train shape is (1944, 14)\nmissing values: 0\nFinal Validation Score: 0.77",
            "bash",
            "tc_eda",
        )
    )
    agent.memory.add_message(Message.assistant_message("later exploration should compact away"))

    info = agent._memory_ctx.replace_protected_raw_prefix_with_summary(
        3,
        "EDA facts:\n- train shape is (1944, 14)\n- missing values: 0",
        warn_chars=50_000,
        label="LHR protected EDA facts",
    )

    assert info["mode"] == "summary"
    assert info["message_count"] == 1
    assert info["original_chars"] > info["chars"]

    out = await agent.compact_inband(mid_run=True)

    assert "kept 1 leading + 1 protected raw + 0 recent" in out
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "FIRST USER QUERY: keep me stable" in text
    assert "EDA facts:" in text
    assert "train shape is (1944, 14)" in text
    assert "scratch eda" not in text
    assert "Final Validation Score: 0.77" not in text
    assert "later exploration should compact away" not in text


@pytest.mark.asyncio
async def test_agentic_text_only_route_prompt_is_ephemeral(tmp_path: Path) -> None:
    route_prompt = (
        "You returned pure text without a tool call.\n"
        "<run_results.md>\n"
        "### S01\nmetric: 0.061\nlower_is_better: true\n"
        "solution design: compact baseline\n"
        "</run_results.md>"
    )
    route_json = (
        '{"action":"rewind_to_step","target_step":"S01",'
        '"reason":"best base","next_plan":"try a distinct regularized run"}'
    )
    llm = _RecordingFakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="No further concrete tool step."),
            SimpleNamespace(tool_calls=[], content=route_json),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    agent._lnr_agentic_text_only_route_prompt = route_prompt

    out = await agent.run("work on the branch")

    assert "No further concrete tool step" in out
    assert "rewind_to_step" in out
    assert agent._lnr_agentic_text_only_route_prompt_injected is True
    assert agent._lnr_agentic_text_only_route_without_result_md is True
    assert llm.kwarg_snapshots[1].get("tool_choice") == "none"
    assert route_prompt in (llm.message_snapshots[1][-1].content or "")

    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    memory_text = "\n".join(
        str(r.memory_record.message.content or "")
        for r in stored
    )
    assert "work on the branch" in memory_text
    assert "No further concrete tool step" not in memory_text
    assert "You returned pure text without a tool call" not in memory_text
    assert "rewind_to_step" not in memory_text
    assert "run_results.md" not in memory_text

    response_path = tmp_path / ".logs" / "agentic_route_response.md"
    assert response_path.read_text(encoding="utf-8").strip() == route_json
    rows = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agentic_route_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[-1]["trigger"] == "text_only"
    assert rows[-1]["input_compact"] == route_prompt
    assert rows[-1]["raw_output"] == route_json


def test_repl_code_agent_suffix_is_generic() -> None:
    text = _repl_code_agent_append_prompt()
    assert "continuous workspace session" in text
    assert "Inspect files, modify files, run commands, validate outcomes" in text
    assert "fixed user task" in text
    assert "targeted shell-level changes" in text
    assert "edit files" not in text
    assert "targeted edit" not in text
    assert "targeted reads with offsets or limits" in text
    assert "stage machines" in text
    assert "solution.py" not in text
    assert "submission.csv" not in text
    assert "ml_run_results.md" not in text


def test_stable_system_prompt_omits_round_budget(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=10,
        stable_system_prompt=True,
    )
    text = agent._build_system_prompt()
    assert "## Round Budget" not in text
    assert "coding and workspace assistant" in text


def test_repl_system_messages_are_layered_for_stable_code_agent(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=10,
        stable_system_prompt=True,
    )
    msgs = agent._build_system_messages()
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "system"
    assert "continuous REPL session" in (msgs[0].content or "")
    assert "workspace root" in (msgs[1].content or "")
    assert "Nomad2018" not in (msgs[0].content or "")
    assert "Nomad2018" not in (msgs[1].content or "")


def test_code_agent_core_has_cache_aware_observation_without_task_details() -> None:
    text = _code_agent_core_prompt()
    assert "structured, compact observations" in text
    assert "programmatic summaries" in text
    assert "whole-file rewrites" in text
    assert "Nomad2018" not in text
    assert "submission.csv" not in text


def test_bash_write_core_guides_large_observation_summaries() -> None:
    text = _code_agent_core_prompt(bash_file_write_mode=True)
    assert "do not use raw `cat`, recursive `ls`, or broad `find` output" in text
    assert "summary that scans the full target" in text
    assert "shape, columns, dtypes, missing values" in text
    assert "extension/type counts" in text
    assert "summarized or truncated bash result is not enough" in text
    assert "write the full log to a workspace file" in text
    assert "Track the current best observed metric" in text
    assert "Do not run verbose training or validation as `command | tail`" in text
    assert "After every substantive training/validation command returns" in text
    assert "immediately update a workspace ledger" in text
    assert "promote the best known artifact" in text
    assert "run the matching prediction/export command" in text
    assert "Nomad2018" not in text
    assert "submission.csv" not in text


def test_bash_write_core_and_prefix_are_generic() -> None:
    core = _code_agent_core_prompt(bash_file_write_mode=True)
    prefix = _repl_bash_file_write_prompt()
    assert "modify files through bash" in core
    assert "exact rewrite operation" in core
    assert "Available tools in this REPL session are `bash`, `read`, `grep`, `glob`, and `ls`" in prefix
    assert "Do not call tools named" not in prefix
    assert "`edit`" not in prefix
    assert "exact rewrite operation" in prefix
    assert "A bash observation may be summarized, deduplicated, or truncated" in prefix
    assert "recover exact facts" in prefix
    assert "record the current best metric" in prefix
    assert "command | tail" in prefix
    assert "workspace ledger" in prefix
    assert "update that ledger before" in prefix
    assert "preserve the matching artifact" in prefix
    assert "align any metric file" in prefix
    assert "cat >" not in core
    assert "cat >>" not in prefix
    assert "python3 - <<" not in core
    assert "python3 - <<" not in prefix
    assert "Nomad2018" not in core
    assert "Nomad2018" not in prefix
    assert "solution.py" not in core
    assert "solution.py" not in prefix
    assert "submission.csv" not in core
    assert "submission.csv" not in prefix


def test_repl_bash_write_mode_removes_write_edit_tools(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=10,
        stable_system_prompt=True,
        include_write_edit_tools=False,
    )
    names = set(agent.availableTools.tool_map)
    assert {"bash", "read", "grep", "glob", "ls"}.issubset(names)
    assert "write" not in names
    assert "edit" not in names

    runtime = agent._build_system_prompt()
    assert "Available tools in this REPL mode are `bash`, `read`, `grep`, `glob`, and `ls`" in runtime
    assert "Tool names must come from that available-tools list" in runtime
    assert "Do not call tools named" not in runtime
    assert "`edit`" not in runtime
    assert "exact rewrite operation" in runtime
    assert "cat >" not in runtime
    assert "**read** before **edit**" not in runtime

    messages = agent._build_system_messages()
    assert len(messages) == 2
    assert "modify files through bash" in (messages[0].content or "")


@pytest.mark.asyncio
async def test_repl_bash_write_mode_normalizes_file_tool_to_bash(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("print('old')\n")
    llm = _FakeLLM(
        [
            _tool_msg(
                "edit",
                {
                    "path": "solution.py",
                    "old_str": "print('old')",
                    "new_str": "print('new')",
                },
            ),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        include_write_edit_tools=False,
    )

    out = await agent.run("modify a file")

    assert "done" in out
    assert (tmp_path / "solution.py").read_text() == "print('new')\n"
    stored = [
        r.memory_record.message
        for r in agent.memory.chat_history_memory.retrieve(window_size=None)
    ]
    assert not any(
        m.role == "user" and "Unavailable tool call blocked" in str(m.content)
        for m in stored
    )
    assistant_tool_names = [
        (tc["function"]["name"] if isinstance(tc, dict) else tc.function.name)
        for m in stored
        if m.role == "assistant"
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    assert "edit" not in assistant_tool_names
    assert "bash" in assistant_tool_names
    tool_names = [getattr(m, "name", "") for m in stored if m.role == "tool"]
    assert "edit" not in tool_names
    assert "bash" in tool_names


def test_repl_bash_write_mode_bash_schema_mentions_file_changes(tmp_path: Path) -> None:
    tools = create_tool_collection(tmp_path, include_write_edit_tools=False).to_params()
    bash_tool = next(t for t in tools if t["function"]["name"] == "bash")
    desc = bash_tool["function"]["description"]
    command_desc = bash_tool["function"]["parameters"]["properties"]["command"][
        "description"
    ]
    assert "file creation or file modification" in desc
    assert "complete file content" in desc
    assert "exact rewrite operation" in desc
    assert "do not dump raw content" in desc
    assert "read`, `grep`, `glob`, and `ls`" in desc
    assert "A compact or truncated bash result is not complete ground truth" in desc
    assert "write verbose logs to workspace files" in desc
    assert "save the full log first" in desc
    assert "update a workspace ledger immediately" in desc
    assert "preserve improved best-known artifacts" in desc
    assert "promote the best known artifact" in desc
    assert "full-scan summaries" in command_desc
    assert "recover exact details" in command_desc
    assert "do not use `command | tail`" in command_desc
    assert "For file changes" in command_desc
    assert "edit" not in desc.lower()


def test_repl_bash_output_caps_are_configurable_on_tool_collection(tmp_path: Path) -> None:
    tools = create_tool_collection(
        tmp_path,
        include_write_edit_tools=False,
        max_bash_output_chars=1234,
        max_bash_stream_line_chars=321,
        bash_observation_summary_enabled=True,
    )
    bash_tool = tools.tool_map["bash"]
    assert bash_tool.max_output_chars == 1234
    assert bash_tool.max_stream_line_chars == 321
    assert bash_tool.observation_summary_enabled is True


@pytest.mark.asyncio
async def test_repl_bash_observation_summary_preserves_raw_artifact(tmp_path: Path) -> None:
    lines = [f"row-{i:03d},value-{i:03d}" for i in range(300)]
    (tmp_path / "large.csv").write_text("\n".join(lines) + "\n")
    llm = _FakeLLM(
        [
            _tool_msg("bash", {"command": "cat large.csv"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        bash_max_output_chars=500,
        bash_observation_summary_enabled=True,
    )

    await agent.run("inspect large file")

    stored = [
        r.memory_record.message
        for r in agent.memory.chat_history_memory.retrieve(window_size=None)
    ]
    tool_texts = [
        m.content or ""
        for m in stored
        if m.role == "tool" and getattr(m, "name", "") == "bash"
    ]
    assert tool_texts
    assert "bash_observation_head_tail_v1" in tool_texts[-1]
    assert "raw_id=tool_000001_bash.txt" in tool_texts[-1]
    assert "path=tool_outputs/tool_000001_bash.txt" not in tool_texts[-1]
    assert "row-000,value-000" in tool_texts[-1]
    assert "row-299,value-299" in tool_texts[-1]
    assert "row-150,value-150" not in tool_texts[-1]

    raw = find_node_log_path(tmp_path, "tool_outputs") / "tool_000001_bash.txt"
    raw_text = raw.read_text()
    assert "row-000,value-000" in raw_text
    assert "row-150,value-150" in raw_text
    assert "row-299,value-299" in raw_text
    assert not (tmp_path / "tool_outputs" / "tool_000001_bash.txt").exists()


def test_repl_bash_write_mode_rewrites_write_tool_nudges() -> None:
    text = (
        "Plan directly from `description.md` + `dataset/`, "
        "then **`write` `solution.py`**.\n"
        "You must **`write` `solution.py`** now (minimal pipeline is OK)."
    )
    out = _adapt_write_tool_hint_for_bash_only(text)
    assert "`write` `solution.py`" not in out
    assert "single bash command" in out
    assert "heredoc" not in out


def test_repl_environment_context_is_small_and_static_shape(tmp_path: Path) -> None:
    text = _repl_environment_context_prompt(tmp_path)
    assert text.startswith("<environment_context>")
    assert f"<cwd>{tmp_path.resolve()}</cwd>" in text
    assert "<shell>" in text
    assert "<current_date>" in text
    assert "validation" not in text.lower()
    assert "round" not in text.lower()


def test_bash_command_classifier_covers_file_actions() -> None:
    assert classify_bash_command("cat > solution.py <<'PYEOF'\nprint(1)\nPYEOF") == "write"
    assert classify_bash_command("cat >> ml_run_results.md <<'EOF'\nok\nEOF") == "append"
    assert (
        classify_bash_command(
            "python3 - <<'PY'\nfrom pathlib import Path\n"
            "p=Path('solution.py')\ns=p.read_text()\n"
            "p.write_text(s.replace('a','b'))\nPY",
        )
        == "edit"
    )
    assert classify_bash_command("cp solution.py solution_v2.py") == "copy"
    assert classify_bash_command("python3 solution.py") == "run_solution"


def test_interaction_log_bash_tool_call_includes_kind() -> None:
    lines = format_tool_call_lines_for_interaction_log(
        "bash",
        {"command": "cat > solution.py <<'PYEOF'\nprint(1)\nPYEOF"},
        full=False,
    )
    assert len(lines) == 1
    assert '"bash_kind": "write"' in lines[0]


def _tool_msg(name: str, args: dict[str, Any], tc_id: str = "tc1") -> Any:
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id=tc_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args),
                ),
            ),
        ],
        content="",
    )


@pytest.mark.asyncio
async def test_science_agent_bash_then_text(tmp_path: Path) -> None:
    llm = _FakeLLM(
        [
            _tool_msg("bash", {"command": "echo roundtrip"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    out = await agent.run("run echo")
    assert "done" in out
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    roles = [r.memory_record.message.role for r in stored]
    assert "user" in roles
    assert "tool" in roles


def test_plain_repl_disables_auto_snapshot_policy() -> None:
    assert _plain_repl_auto_continue_session(
        AutoContinuePolicy(max_text_only_retries=2),
        "off",
    )
    assert not _plain_repl_auto_continue_session(
        AutoContinuePolicy(max_text_only_retries=2),
        "full",
    )
    assert not _plain_repl_auto_continue_session(DefaultPolicy(), "off")


@pytest.mark.asyncio
async def test_repl_write_result_omits_auto_snapshot_but_keeps_tool_call_code(
    tmp_path: Path,
) -> None:
    llm = _RecordingFakeLLM(
        [
            _tool_msg("write", {"path": "solution.py", "content": "print('ok')\n"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        write_auto_snapshot_enabled=False,
    )

    await agent.run("write solution")

    stored = [
        r.memory_record.message
        for r in agent.memory.chat_history_memory.retrieve(window_size=None)
    ]
    write_tool_results = [
        m.content or ""
        for m in stored
        if m.role == "tool" and getattr(m, "name", "") == "write"
    ]
    assert write_tool_results
    assert "File `solution.py` written successfully" in write_tool_results[-1]
    assert "[auto-snapshot after successful write:" not in write_tool_results[-1]

    assistant_tool_calls = [
        tc
        for m in stored
        for tc in (getattr(m, "tool_calls", None) or [])
        if m.role == "assistant"
    ]
    assert assistant_tool_calls
    tc = assistant_tool_calls[-1]
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    assert "print('ok')" in args


@pytest.mark.asyncio
async def test_science_agent_saves_tool_output_txt_artifact(tmp_path: Path) -> None:
    llm = _FakeLLM(
        [
            _tool_msg("bash", {"command": "python3 -c \"print('raw artifact check')\""}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )

    await agent.run("run echo")

    out_dir = find_node_log_path(tmp_path, "tool_outputs")
    raw = out_dir / "tool_000001_bash.txt"
    assert raw.is_file()
    assert "raw artifact check" in raw.read_text()
    idx = (out_dir / "index.txt").read_text()
    assert "raw_id=tool_000001_bash.txt" in idx
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    tool_messages = [
        r.memory_record.message.content
        for r in stored
        if r.memory_record.message.role == "tool"
    ]
    assert any("raw_id=tool_000001_bash.txt" in str(m) for m in tool_messages)


@pytest.mark.asyncio
async def test_science_agent_multiturn_memory(tmp_path: Path) -> None:
    llm = _FakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="one"),
            SimpleNamespace(tool_calls=[], content="two"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    await agent.run("first")
    await agent.run("second")
    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    texts = [
        r.memory_record.message.content
        for r in stored
        if r.memory_record.message.role == "user"
    ]
    joined = " ".join(str(t) for t in texts)
    assert "first" in joined and "second" in joined


def test_science_agent_mid_run_compact_enabled_flag(tmp_path: Path) -> None:
    """Config ``mid_run_compact_enabled`` is stored for run_loop mid-session compact."""
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        mid_run_compact_enabled=False,
    )
    assert agent._mid_run_compact_enabled is False


@pytest.mark.asyncio
async def test_run_mid_run_compact_triggers_when_window_omits_messages(tmp_path: Path) -> None:
    """When the REPL window overflows, run_loop compacts before the next LLM call."""
    calls: list[bool] = []

    async def fake_compact_inband(*, mid_run: bool = False) -> str:
        calls.append(mid_run)
        return "Compacted 3 messages into in-band summary (10 chars)."

    llm = _FakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="after compact"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=3,
        sliding_window_budget_chars=2000,
        mid_run_compact_enabled=False,
    )
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    for i in range(16):
        agent.memory.add_message(
            Message.assistant_message(f"assistant-{i}: " + "x" * 1000),
        )
        agent.memory.add_message(
            Message.tool_message(f"tool-{i}: " + "y" * 1000, "bash", str(i)),
        )
    agent.compact_inband = fake_compact_inband  # type: ignore[method-assign]

    out = await agent.run("go")
    assert "after compact" in out
    assert calls == [True]


@pytest.mark.asyncio
async def test_run_context_limit_compact_retries_durable_when_inband_still_omits(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    omitted_values = [1, 1, 0]

    async def fake_compact_inband(*, mid_run: bool = False) -> str:
        calls.append(f"inband:{mid_run}")
        return "Compacted 3 messages into in-band summary (10 chars)."

    async def fake_compact(*, mid_run: bool = False) -> str:
        calls.append(f"durable:{mid_run}")
        return "Compacted 2 messages into summary (8 chars)."

    llm = _FakeLLM([SimpleNamespace(tool_calls=[], content="after fallback")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=1,
        sliding_window_budget_chars=2000,
        mid_run_compact_enabled=False,
    )
    agent._lnr_compact_on_context_threshold = True
    agent.memory.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))

    def fake_build_messages_for_llm_with_stats() -> tuple[list[Message], int]:
        omitted = omitted_values.pop(0) if omitted_values else 0
        return [Message.user_message("visible context")], omitted

    agent._memory_ctx.build_messages_for_llm_with_stats = (
        fake_build_messages_for_llm_with_stats
    )
    agent.compact_inband = fake_compact_inband  # type: ignore[method-assign]
    agent.compact = fake_compact  # type: ignore[method-assign]

    out = await agent.run("go")

    assert "after fallback" in out
    assert calls == ["inband:True", "durable:True"]


@pytest.mark.asyncio
async def test_run_window_compact_preserves_first_user_prefix(tmp_path: Path) -> None:
    """Overflow compaction must not erase the original task / first user query."""

    llm = _RecordingFakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="summary of old work"),
            SimpleNamespace(tool_calls=[], content="after compact"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=1,
        sliding_window_budget_chars=2000,
        mid_run_compact_enabled=False,
    )
    first = "FIRST USER QUERY: solve nomad2018 and write submission.csv"
    agent.memory.add_message(Message.user_message(first))
    for _ in range(6):
        agent.memory.add_message(Message.assistant_message("x" * 5000))

    out = await agent.run("continue")
    assert "after compact" in out

    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    contents = [r.memory_record.message.content or "" for r in stored]
    assert first in contents[0]
    assert any("summary of old work" in c for c in contents)
    assert any(c == "continue" for c in contents)
    assert llm.kwarg_snapshots[0].get("tool_choice") == "none"
    assert "Internal context maintenance request" in (
        llm.message_snapshots[0][-1].content or ""
    )


@pytest.mark.asyncio
async def test_compact_clears_history(tmp_path: Path) -> None:
    class _SummarizerLLM:
        async def ask(self, **kwargs: Any) -> str:
            return "summary text"

    mem = Memory(max_messages=50)
    mem.add_message(Message.user_message("old stuff"))
    from scienceflow.core.mem.memory_context import MemoryContextManager

    ctx = MemoryContextManager(mem, tmp_path, budget_chars=10_000)
    out = await ctx.compact(_SummarizerLLM())
    assert "Compacted" in out
    after = mem.chat_history_memory.retrieve(window_size=None)
    assert len(after) >= 1
    # New contract: leading user pin (msg[0]) is preserved verbatim across compact;
    # the LLM summary is appended *after* the pin instead of replacing it.
    contents = [r.memory_record.message.content or "" for r in after]
    assert "old stuff" in contents[0], "msg[0] head must survive compact verbatim"
    assert any("summary" in c.lower() for c in contents), "summary must be appended after the pin"


def test_replace_history_with_compacted_summary_drops_old_leading_summaries(
    tmp_path: Path,
) -> None:
    from scienceflow.core.mem.memory_context import (
        COMPACTED_CONVERSATION_SUMMARY_MARKER,
        MemoryContextManager,
    )

    mem = Memory(max_messages=50)
    mem.add_message(Message.user_message("FIRST USER QUERY: keep me stable"))
    mem.add_message(
        Message.user_message(f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\nold one"),
    )
    mem.add_message(
        Message.user_message(f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\nold two"),
    )
    mem.add_message(Message.assistant_message("old assistant turn"))

    ctx = MemoryContextManager(mem, tmp_path, budget_chars=10_000)
    out = ctx.replace_history_with_compacted_summary("new summary", recent_messages=0)

    assert "kept 1 leading + 0 recent" in out
    after = mem.chat_history_memory.retrieve(window_size=None)
    contents = [r.memory_record.message.content or "" for r in after]
    assert contents == [
        "FIRST USER QUERY: keep me stable",
        f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\nnew summary",
    ]


def test_round_budget_prompt_cap_never_exceeds_soft_total(tmp_path: Path) -> None:
    """Soft cap: LLM must not see current/total like 20/15 when actual round > cap."""
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=50,
        round_budget_prompt_cap=15,
    )
    agent._effective_max_steps = 50
    agent._current_round = 19  # 20th round (1-based current would be 20)
    text = agent._format_round_budget()
    assert "Round 15/15" in text
    assert "remaining: 0" in text
    assert "/15" in text
    assert "20/15" not in text
    assert "16/15" not in text


@pytest.mark.asyncio
async def test_no_progress_hardstop_warning_seen_before_terminate(tmp_path: Path) -> None:
    """First-stage inject must appear in LLM input before run ends on second stall."""
    llm = _RecordingFakeLLM(
        [
            _tool_msg("bash", {"command": "echo a"}),
            _tool_msg("bash", {"command": "echo b"}),
            _tool_msg("bash", {"command": "echo c"}),
            _tool_msg("bash", {"command": "echo d"}),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=12,
        no_progress_hard_stop_after=2,
        lnr_explore_streak_inject_after=0,
        lnr_single_read_streak_inject_after=0,
        no_success_run_soft_threshold=100,
        no_success_run_hard_threshold=200,
    )
    out = await agent.run("go")
    assert "no-progress hardstop" in (out or "").lower()

    assert len(llm.message_snapshots) >= 3
    joined_round_3 = "\n".join(
        str(getattr(m, "content", "") or "")
        for m in llm.message_snapshots[2]
    )
    joined_round_3_lower = joined_round_3.lower()
    assert "next round" in joined_round_3_lower
    assert (
        "edit" in joined_round_3_lower
        or "write" in joined_round_3_lower
        or "file-change mechanism" in joined_round_3_lower
    )

def test_workspace_source_git_auto_checkpoint_can_write_submission_snapshots_outside_workspace(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    workspace = tmp_path / "workspace"
    logs = tmp_path / ".logs"
    workspace.mkdir()
    (workspace / "train.py").write_text("SEED = 1\n", encoding="utf-8")
    ensure_workspace_source_git(workspace, track_globs=["*.py", "*.md"])

    (workspace / "submission.csv").write_text("id,y\n1,0.7\n", encoding="utf-8")
    result = auto_checkpoint_workspace_source(
        workspace,
        track_globs=["*.py", "*.md"],
        tool_name="lhr_stage_s01",
        metric_value_override=0.0612,
        stage_id="S01",
        submission_snapshot_dir=logs / "submission_snapshots",
    )

    assert result.ready is True
    assert result.submission_snapshot.startswith("../.logs/submission_snapshots/")
    assert not (workspace / "submission_snapshots").exists()
    assert (workspace / result.submission_snapshot).read_text(encoding="utf-8") == "id,y\n1,0.7\n"
    ledger = workspace / ".scienceflow_checkpoints" / "ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert row["submission_snapshot"] == result.submission_snapshot
