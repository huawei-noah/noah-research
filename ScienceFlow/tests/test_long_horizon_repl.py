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

import asyncio
import csv
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from deepcraft_core import Message
from deepcraft_core.tool import ToolResult
from scienceflow.gates import GateService
from scienceflow.core.agent.agent import ScienceAgent
from scienceflow.core.agent.memory.resource_feedback_memory import (
    RESOURCE_STATE_SUMMARY_MARKER,
    ResourceFeedbackMemoryDeduper,
)
from scienceflow.core.agent.runtime.run_loop import RunLoopMixin
from scienceflow.core.agent.runtime.lnr_hooks import LNRHooksMixin
from scienceflow.core.bash_solution_cmd import (
    looks_like_bare_solution_run,
    python_script_run_rel_path,
)
from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
)
from scienceflow.safety.execution_policy import EnsureFullRunResult, write_fullrun_tail_snapshot
from scienceflow.solver.lnr.resource_observer import LHRResourceObserver
from scienceflow.solver.lnr.prompt_template_store import load_prompt_template
from scienceflow.solver.lnr.prompts import (
    KEEP_CURRENT_COMPACT_TEMPLATE,
    ML_FIRST_USER_TEMPLATE,
    ESTRA_DECISION_TEMPLATE,
    ESTRA_RESUME_TEMPLATE,
    build_first_user_prompt,
    build_keep_current_compact_prompt,
    build_estra_prompt,
    build_estra_resume_prompt,
    build_stage_commit_prompt,
)
from scienceflow.solver.lnr.snapshot_store import SnapshotStore, StageSnapshot
from scienceflow.solver.lnr import solver as lnr_solver_module
from scienceflow.solver.lnr.state_machine import LHRStateMachineStore
from scienceflow.solver.lnr.solver import (
    LHR_STAGE_PERFORMANCE_COLUMNS,
    LnrSolver,
    _append_lnr_main_agent_protocols,
    _effective_lnr_bash_timeout_sec,
    _validation_leakage_reason,
)
from scienceflow.solver.lnr.stage.stage_ledger import (
    append_archived_trajectory_summary,
    append_estra_summary,
    append_stage_event_summary,
    next_stage_id,
    parse_stage_cards,
    render_stage_cards,
    salvage_append_only_stage_commit,
    tail_summary_after,
    tail_summary_from_stage,
    validate_append_only_stage_commit,
    validate_stage_card,
)


def test_lhr_script_run_parser_accepts_relative_variant_entrypoints() -> None:
    assert python_script_run_rel_path("python3 solution_v7.py 2>&1") == "solution_v7.py"
    assert python_script_run_rel_path("python -u train.py") == "train.py"
    assert python_script_run_rel_path("python3 ./predict.py") == "predict.py"
    assert python_script_run_rel_path("python3 ../train.py") is None
    assert python_script_run_rel_path("python3 /tmp/train.py") is None
    assert python_script_run_rel_path("QUICK_TEST_ROWS=10 python3 train.py") is None
    assert looks_like_bare_solution_run("python3 solution.py 2>&1")
    assert not looks_like_bare_solution_run("python3 solution_v7.py 2>&1")


def test_lhr_first_user_prompt_allows_reusable_train_predict_layout() -> None:
    text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)

    assert "continuous REPL workspace" in text
    assert "root-level `submission.csv`" in text
    assert "`train.py` / `predict.py` / `util.py`" in text
    assert "simple or short-run tasks" in text
    assert "same validation split or folds" in text
    assert "train/test feature logic does not drift" in text
    assert "Use allocated compute deliberately" in text
    assert "sustained idle assigned compute" in text
    assert "build or improve `solution.py`" not in text


def test_lhr_first_user_prompt_includes_resource_context_snapshot() -> None:
    text = build_first_user_prompt(
        "Task body",
        wall_clock_budget_sec=300,
        resource_context="allocated_compute:\n  - worker_cpu_list: 32-39\n  - visible_gpu_list: 6",
    )

    assert "ResourceContext snapshot:" in text
    assert "worker_cpu_list: 32-39" in text
    assert "visible_gpu_list: 6" in text


def test_lhr_bash_timeout_respects_configured_cap_and_remaining_fuse() -> None:
    assert _effective_lnr_bash_timeout_sec(300, 6900) == 300
    assert _effective_lnr_bash_timeout_sec(14_400, 6900) == 6900
    assert _effective_lnr_bash_timeout_sec(300, 120) == 120


def test_lnr_runtime_context_appends_to_latest_existing_tool_message() -> None:
    class _History:
        def __init__(self, messages: list[Message]) -> None:
            self.messages = messages

        def retrieve(self, window_size=None):
            return [
                SimpleNamespace(memory_record=SimpleNamespace(message=message))
                for message in self.messages
            ]

    class _MemoryContext:
        def __init__(self, history: _History) -> None:
            self.history = history

        def rewrite_messages(self, messages: list[Message]) -> None:
            self.history.messages = list(messages)

    class _Harness(LNRHooksMixin):
        pass

    history = _History(
        [
            Message.user_message("work"),
            Message.tool_message("tool output", "bash", "call-1"),
        ],
    )
    harness = _Harness()
    harness.memory = SimpleNamespace(chat_history_memory=history)
    harness._memory_ctx = _MemoryContext(history)
    remaining = iter((100, 90))
    harness._lnr_runtime_context_provider = lambda: (
        "Runtime context (current worker limits for planning the next action):\n"
        f"wall_clock_remaining_sec: {next(remaining)}\n"
        "effective_bash_timeout_sec: 30"
    )

    harness._lnr_maybe_periodic_inject_at_round_start(1)
    harness._lnr_maybe_periodic_inject_at_round_start(1)

    assert len(history.messages) == 2
    assert str(history.messages[-1].content).count("Runtime context (") == 1
    assert "wall_clock_remaining_sec: 90" in str(history.messages[-1].content)
    assert "wall_clock_remaining_sec: 100" not in str(history.messages[-1].content)


def test_lnr_runtime_context_uses_existing_user_message_when_no_tool_result() -> None:
    class _Harness(LNRHooksMixin):
        pass

    messages = [Message.user_message("work")]
    history = SimpleNamespace(
        retrieve=lambda window_size=None: [
            SimpleNamespace(memory_record=SimpleNamespace(message=message))
            for message in messages
        ],
    )
    harness = _Harness()
    harness.memory = SimpleNamespace(chat_history_memory=history)
    harness._memory_ctx = SimpleNamespace(
        rewrite_messages=lambda updated: messages.__setitem__(slice(None), updated),
    )
    harness._lnr_runtime_context_provider = lambda: (
        "Runtime context (current worker limits for planning the next action):\n"
        "wall_clock_remaining_sec: 90"
    )

    harness._lnr_maybe_periodic_inject_at_round_start(0)

    assert len(messages) == 1
    assert "wall_clock_remaining_sec: 90" in str(messages[0].content)


def test_lnr_runtime_context_preserves_ordinary_text_containing_marker() -> None:
    class _Harness(LNRHooksMixin):
        pass

    marker = "Runtime context (current worker limits for planning the next action):"
    messages = [Message.user_message(f"quote: {marker}\nkeep this explanation")]
    history = SimpleNamespace(
        retrieve=lambda window_size=None: [
            SimpleNamespace(memory_record=SimpleNamespace(message=message))
            for message in messages
        ],
    )
    harness = _Harness()
    harness.memory = SimpleNamespace(chat_history_memory=history)
    harness._memory_ctx = SimpleNamespace(
        rewrite_messages=lambda updated: messages.__setitem__(slice(None), updated),
    )
    harness._lnr_runtime_context_provider = lambda: (
        f"{marker}\nwall_clock_remaining_sec: 90\n"
        "effective_bash_timeout_sec: 30"
    )

    harness._lnr_maybe_periodic_inject_at_round_start(0)

    content = str(messages[0].content)
    assert "keep this explanation" in content
    assert content.count(marker) == 2


def test_lhr_runtime_context_reports_effective_bash_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.deadline = 1100.0
    monkeypatch.setattr(lnr_solver_module.time, "monotonic", lambda: 1000.0)

    context = solver._runtime_context_for_agent(SimpleNamespace(_bash_timeout_sec=600))

    assert "wall_clock_remaining_sec: 100" in context
    assert "effective_bash_timeout_sec: 100" in context


def test_lhr_make_agent_keeps_normal_and_slow_bash_caps() -> None:
    solver = object.__new__(LnrSolver)
    solver.deadline = lnr_solver_module.time.monotonic() + 6900
    solver.worker_id = ""
    solver.worker_index = 0
    solver.worker_count = 1
    solver.worker_extra_env = {}
    solver.memory_dir = Path("memory")
    solver.log_dir = Path("logs")
    solver.task_root_dir = Path("task")
    solver.ledger_filename = ".run_results.md"
    solver.resource_observer = None
    solver.evaluation_service = object()
    solver.skill_registry = None
    solver.skill_task_category = ""
    solver.skill_allow_names = ()
    solver.skill_tool_mode = "category_only"
    solver.skill_allow_generic_wildcard = False
    solver.skill_visible_max = 1
    solver.lhr = SimpleNamespace(
        max_steps=10,
        workspace_git_enabled=False,
        workspace_git_track_globs=None,
        workspace_git_auto_review=False,
        resource_bash_hard_fuse_finalization_reserve_sec=900,
        clean_repl_mode=False,
        compact_on_context_limit=True,
    )
    solver.cfg = SimpleNamespace(
        exp_id="test",
        repl_bash_max_output_chars=8000,
        repl_bash_max_stream_line_chars=2400,
        mlebench_data_root_dir="",
    )
    bash_tool = SimpleNamespace(
        bash_timeout_sec=300,
        bash_timeout_slow_sec=600,
    )
    agent = SimpleNamespace(
        availableTools=SimpleNamespace(tool_map={"bash": bash_tool}),
        _system_prompt_core="",
        systemPrompt="",
    )
    solver.orchestrator = SimpleNamespace(
        make_llm_call_tracer=lambda **_kwargs: None,
        create_science_agent=lambda **_kwargs: agent,
    )
    solver._worker_llm_stage_override = lambda: None
    solver._task_runtime_extra_env = lambda: {}
    solver._code_organization_hint = lambda: ""
    solver._attach_lnr_interaction_logger = lambda _agent: None
    solver._sanitize_agent_prompt_surfaces = lambda _agent: None
    solver._restore_protected_eda_prefix_marker = lambda _agent: None
    solver._evaluator_stage_source_mode = lambda: "primary"
    solver._evaluator_task_profile = lambda: ""
    solver._evaluator_backend_name = lambda: ""
    solver._evaluator_candidate_artifact = lambda: "submission.csv"
    solver._worker_uid_prefix = lambda: "W00"

    built = solver._make_agent(load_existing_memory=False)

    assert bash_tool.bash_timeout_sec == 300
    assert bash_tool.bash_timeout_slow_sec == 600
    assert built._bash_timeout_sec == 300
    assert built._bash_timeout_slow_sec == 600


def test_lhr_first_user_prompt_category_skill_hint_is_optional() -> None:
    default_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)
    hinted_text = build_first_user_prompt(
        "Task body",
        wall_clock_budget_sec=300,
        skill_hint="A category-specific skill is available. Use `skill list`.",
    )

    assert "Use `skill list`" not in default_text
    assert "A category-specific skill is available. Use `skill list`." in hinted_text


def test_lhr_first_user_prompt_seed_is_optional() -> None:
    default_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)
    seeded_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300, seed=2222)

    assert "Base experiment seed" not in default_text
    assert "Base experiment seed: 2222" in seeded_text
    assert "random_state" in seeded_text


def test_lhr_first_user_prompt_worker_id_is_optional() -> None:
    default_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)
    worker_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300, worker_id="w01")

    assert "Worker identity" not in default_text
    assert "Worker identity: W01" in worker_text
    assert "worker-local notes" in worker_text


def test_lhr_opt_solver_prompt_uses_profile_template_without_submission_contract() -> None:
    text = build_first_user_prompt(
        "Produce artifacts/best_solution.json.",
        wall_clock_budget_sec=300,
        task_profile="opt_solver",
        task_runtime_contract=(
            "- Candidate artifact path: `artifacts/best_solution.json`.\n"
            "- Task Python executable: `/opt/env/bin/python`."
        ),
    )

    assert "optimization task" in text
    assert "artifacts/best_solution.json" in text
    assert "Task Python executable" in text
    assert "Progress protocol for optimization search" in text
    assert "tmp/progress.json" in text
    assert "SCIENCEFLOW_HB v=1" in text
    assert "root-level `submission.csv`" not in text
    assert "Final Validation Score" not in text


def test_lhr_default_prompt_does_not_include_opt_solver_progress_protocol() -> None:
    text = build_first_user_prompt(
        "Train a model and write submission.csv.",
        wall_clock_budget_sec=300,
        task_profile="mlebench",
    )

    assert "Progress protocol for optimization search" not in text
    assert "tmp/progress.json" not in text


def test_lhr_prompt_template_loader_accepts_profile_subdir() -> None:
    text = load_prompt_template("task/opt_solver/first_user.md")

    assert "optimization task" in text
    assert "Progress protocol for optimization search" in text


def _make_metric_snapshot_agent(ws: Path):
    agent = ScienceAgent.__new__(ScienceAgent)
    mem = MagicMock()
    mem.add_message = MagicMock()
    object.__setattr__(agent, "_workspace_dir", ws)
    object.__setattr__(agent, "_embedded_full_run_enabled", False)
    object.__setattr__(agent, "_lnr_allow_any_stage_script", False)
    object.__setattr__(agent, "_lnr_mlebench_validate_enabled", True)
    object.__setattr__(agent, "_mlebench_data_dir", None)
    object.__setattr__(agent, "_mlebench_exp_id", None)
    object.__setattr__(agent, "_scienceflow_task_profile", "mlebench")
    object.__setattr__(agent, "_scienceflow_evaluator_backend", "task_package")
    object.__setattr__(agent, "_lnr_candidate_artifact_rel", "submission.csv")
    object.__setattr__(agent, "_lnr_run_control_max_fix_rounds", 5)
    object.__setattr__(agent, "_run_control_user_injections", 0)
    object.__setattr__(agent, "_inject_run_control_user_message", MagicMock())
    object.__setattr__(agent, "_fullrun_output_tail_stdout_lines", 50)
    object.__setattr__(agent, "_fullrun_output_tail_stderr_lines", 0)
    object.__setattr__(agent, "_fullrun_output_tail_max_chars", 4000)
    object.__setattr__(agent, "_submission_history_archive_enabled", False)
    object.__setattr__(agent, "_log_info", MagicMock())
    object.__setattr__(agent, "_log_warning", MagicMock())
    object.__setattr__(agent, "memory", mem)
    return agent


def test_lhr_agent_metric_only_snapshot_without_submission(tmp_path: Path) -> None:
    ws = tmp_path / "metric_only_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)
    interpretation_cb = MagicMock()
    object.__setattr__(agent, "_lnr_metric_output_interpretation_callback", interpretation_cb)

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="[exit=0, 1.0s]\nok\nFinal Validation Score: 0.42\n", error=""),
        )
    )

    assert getattr(agent, "_lnr_snapshot_ok", False) is True
    assert getattr(agent, "_lnr_snapshot_reason", "") == "metric_only_tail_snapshot"
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.42)
    assert data["metric_name"] == "Final Validation Score"
    assert data["bash_cmd"] == "python3 solution.py"
    assert data["submission_status"] == "missing_submission"
    assert "submission_sha" not in data
    assert data["solution_path"] == "solution.py"
    assert data["solution_sha"]
    interpretation_cb.assert_not_called()


def test_lhr_agent_metric_only_snapshot_accepts_inline_python_without_solution(tmp_path: Path) -> None:
    ws = tmp_path / "inline_metric_ws"
    ws.mkdir()
    agent = _make_metric_snapshot_agent(ws)
    cmd = "python3 <<'PY'\nprint('Final Validation Score: 0.33')\nPY"

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": cmd},
            ToolResult(output="[exit=0, 0.5s]\nFinal Validation Score: 0.33\n", error=""),
        )
    )

    assert getattr(agent, "_lnr_snapshot_ok", False) is True
    assert getattr(agent, "_lnr_snapshot_reason", "") == "metric_only_tail_snapshot"
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.33)
    assert data["submission_status"] == "missing_submission"
    assert data["execution_mode"] == "inline_bash"
    assert data["solution_path"] == ""
    assert "solution_sha" not in data


def test_lhr_agent_submission_snapshot_waits_for_unified_evaluator(tmp_path: Path) -> None:
    ws = tmp_path / "invalid_submission_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "submission.csv").write_text("bad\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="[exit=0, 2.0s]\nFinal Validation Score: 0.51\n", error=""),
        )
    )

    assert getattr(agent, "_lnr_snapshot_ok", False) is True
    assert getattr(agent, "_lnr_snapshot_reason", "") == "candidate_tail_snapshot"
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.51)
    assert "submission_validation_ok" not in data
    assert data["submission_status"] == "pending_evaluator"
    assert data["submission_sha"]
    agent._inject_run_control_user_message.assert_not_called()


def test_lhr_agent_uses_grounded_llm_metric_fallback(tmp_path: Path) -> None:
    ws = tmp_path / "interpreted_metric_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)

    async def interpret(**_: object) -> dict[str, object]:
        return {
            "metric_found": True,
            "metric_name": "weighted_auc",
            "metric_value": 0.641136,
            "split": "validation",
            "is_final": True,
            "evidence_line": "Best Validation wAUC: 0.641136",
            "confidence": "high",
        }

    object.__setattr__(agent, "_lnr_metric_output_interpretation_callback", interpret)
    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="Best Validation wAUC: 0.641136\n", error=""),
        )
    )

    assert agent._lnr_snapshot_ok is True
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.641136)
    assert data["metric_name"] == "weighted_auc"


def test_lhr_agent_does_not_fallback_from_explicit_invalid_contract(tmp_path: Path) -> None:
    ws = tmp_path / "invalid_contract_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)
    interpretation_cb = MagicMock()
    object.__setattr__(agent, "_lnr_metric_output_interpretation_callback", interpretation_cb)

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="Final Validation Score: nan\nBest Validation AUC: 0.9\n", error=""),
        )
    )

    assert agent._lnr_snapshot_ok is False
    assert agent._lnr_snapshot_reason == "score_contract_invalid"
    interpretation_cb.assert_not_called()
    agent._inject_run_control_user_message.assert_called_once()


def test_lhr_estra_prompt_can_include_checkpoint_context() -> None:
    text = build_estra_prompt(
        ledger_filename=".run_results.md",
        ledger_text="### S01\nmetric: 0.06\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
        latest_stage="S02",
        switch_candidate_stages=["S01"],
        stage_checkpoint_context="- S01: metric=0.06; source_commit=abc123",
    )

    assert "Choose the next research direction for one ML search trajectory" in text
    assert "Decide two axes:" in text
    assert "`startpoint`: `current_workspace` or `previous_stage`" in text
    assert "`intent`: `continue` or `redirect`" in text
    assert "Metric is evidence, not the only selection rule" in text
    assert "route potential" in text
    assert "Stage checkpoint context" in text
    assert "source_commit=abc123" in text
    assert "Return exactly one JSON object" in text
    assert "\"startpoint\":\"current_workspace|previous_stage\"" in text


def test_lhr_stage_commit_prompt_includes_metric_semantics_fields() -> None:
    text = build_stage_commit_prompt(
        stage_id="S02",
        ledger_filename=".run_results.md",
        metric_event={
            "metric_value": 0.061,
            "lower_is_better": True,
            "run_time_sec": 12.3,
            "val_score_type": "holdout",
            "selection_note": "",
        },
        existing_ledger="### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
    )

    assert "Append exactly one new stage entry" in text
    assert "Preserve all existing entries unchanged" in text
    assert "metric_type:" in text
    assert "metric_note:" in text
    assert "run_time_sec:" in text
    assert "copy `run_time_sec`" in text
    assert "copy `val_score_type`" in text
    assert "FILES:" in text
    assert "exclude outputs" in text


def test_lhr_prompt_templates_are_markdown_backed() -> None:
    first_user_template = load_prompt_template(ML_FIRST_USER_TEMPLATE)
    estra_template = load_prompt_template(ESTRA_DECISION_TEMPLATE)

    assert "continuous REPL workspace" in first_user_template
    assert "{task}" in first_user_template
    estra_resume_template = load_prompt_template(ESTRA_RESUME_TEMPLATE)
    keep_current_template = load_prompt_template(KEEP_CURRENT_COMPACT_TEMPLATE)

    assert "Return exactly one JSON object only" in estra_template
    assert "{ledger}" in estra_template
    assert "{base_stage_memory_policy}" in estra_resume_template
    assert "{base_stage_memory_policy}" in keep_current_template
    assert "Exploration state:" in estra_resume_template
    assert "Exploration state:" in keep_current_template


def test_lhr_resume_prompts_use_dynamic_base_stage() -> None:
    switched = build_estra_resume_prompt(
        target_stage="S02",
        next_stage="S04",
        ledger_filename=".run_results.md",
        base_stage="S02",
        compact_summary="S03 was abandoned",
    )
    kept = build_keep_current_compact_prompt(
        terminal_stage="S03",
        next_stage="S04",
        base_stage="S02",
        compact_summary="recent branch compacted",
    )

    assert "original S02 base stage" in switched
    assert "Historical exploration after the restored stage" in switched
    assert "abandoned-branch evidence" in switched
    assert "original S01 base stage" not in switched
    assert "original S02 base stage" in kept
    for prompt in (switched, kept):
        assert "Exploration state:" in prompt
        assert "completed evidence, not a stop signal" in prompt
        assert "not something to repeat" in prompt
        assert "Do not overwrite the best submission without a validated candidate" in prompt


def test_lhr_keep_current_compact_frames_rsna_final_summary_as_evidence() -> None:
    summary = (
        "## Final deliverable - S05 restored, all exploration complete\n"
        "The workspace is in its best confirmed state.\n"
        "Best deliverable remains submission.csv at 0.2324 holdout WLL."
    )
    prompt = build_keep_current_compact_prompt(
        terminal_stage="S14",
        next_stage="S15",
        base_stage="S05",
        compact_summary=summary,
        state_packet="ESTRA action: keep_current\nLatest stage: S14",
    )

    assert prompt.index("Exploration state:") < prompt.index("Current compact state note:")
    assert "Prior summaries are completed evidence, not a stop signal" in prompt
    assert "not something to repeat" in prompt
    assert "Final deliverable - S05 restored" in prompt


def test_lnr_system_core_adds_lnr_ledger_boundary_once() -> None:
    core = "After every substantive training/validation command returns, immediately update a workspace ledger."
    text = _append_lnr_main_agent_protocols(core)
    text2 = _append_lnr_main_agent_protocols(text)

    assert "workspace ledger" in text
    assert "LNR ledger boundary" in text
    assert "Do not create extra planning systems" in text
    assert "stage-like ledgers" in text
    assert "tmp/experiment_notes.md" in text
    assert "hidden stage ledger" in text
    assert "Rules:" in text
    assert "PENDING/REPLAN/policy blocked" in text
    assert "STOP_BOUNDARY_VIOLATION" in text
    assert "Heartbeat format: SCIENCEFLOW_HB v=1" in text
    assert text2.count("LNR ledger boundary") == 1
    assert text2.count("Resource feedback is runtime context") == 1
    assert text2.count("Optional reduction payload boundary") == 1
    assert "targets three distinct defensible artifacts" in text
    assert "Do not create duplicate or cosmetically perturbed artifacts" in text
    assert "merge_payload/" in text


def test_lhr_config_defaults_enable_repl_style_git_and_code_hint() -> None:
    from scienceflow.config.settings import LnrConfig

    cfg = LnrConfig()
    assert cfg.code_organization_hint == "beyond_mfiles"
    assert cfg.workspace_git_enabled is True
    assert cfg.workspace_git_auto_checkpoint is True
    assert cfg.workspace_git_track_globs == ["*.py", "*.md"]
    assert cfg.force_estra_capture_duplicate_submissions is False
    assert cfg.state_packet_max_chars == 12000
    assert cfg.resource_bash_monitor_all_enabled is True
    assert cfg.resource_bash_hard_fuse_finalization_reserve_sec == 900.0
    assert cfg.resource_research_cadence_enabled is True
    assert cfg.resource_research_cadence_observe_sec == 300.0
    assert cfg.resource_first_comparable_metric_budget_sec == 900.0
    assert cfg.resource_proven_route_metric_budget_sec == 1800.0
    assert cfg.resource_arbiter_min_progress_windows == 2
    assert cfg.resource_arbiter_kill_requires_high_confidence is True
    assert cfg.resource_gpu_util_observer_enabled is True
    assert cfg.resource_gpu_util_sample_interval_sec == 30.0
    assert cfg.resource_gpu_capacity_slots == 1.0
    assert cfg.resource_gpu_tt_max_per_gpu == 3
    assert cfg.resource_gpu_feature_max_per_gpu == 2
    assert cfg.resource_gpu_share_tt_with_train is False
    assert cfg.resource_queue_timeout_hard_gate_enabled is True
    assert cfg.resource_queue_timeout_block_train_after == 1
    assert cfg.resource_queue_timeout_tt_only_after == 2


def test_lhr_flat_dataset_adapts_legacy_split_without_exposing_shallow(tmp_path: Path) -> None:
    from scienceflow.solver.lnr.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    split_root = tmp_path / "dataset_split"
    deep = split_root / "Deep"
    shallow = split_root / "Shallow"
    deep.mkdir(parents=True)
    shallow.mkdir(parents=True)
    (deep / "train.csv").write_text("x\n1\n2\n", encoding="utf-8")
    (shallow / "train.csv").write_text("x\n1\n", encoding="utf-8")

    source, layout = resolve_workspace_dataset_source(split_root)
    ws_dataset = tmp_path / "workspace" / "dataset"
    prepare_workspace_dataset_flat(source, ws_dataset)

    assert layout == "legacy_split_deep"
    assert (ws_dataset / "train.csv").resolve() == (deep / "train.csv").resolve()
    assert not (ws_dataset / "Shallow").exists()
    assert not (ws_dataset / "Deep").exists()


def test_lhr_flat_dataset_falls_back_from_self_referential_deep_symlink(tmp_path: Path) -> None:
    from scienceflow.solver.lnr.prep_fs import resolve_workspace_dataset_source

    prepared = tmp_path / "prepared"
    split_root = prepared / "dataset_split"
    deep = split_root / "Deep"
    shallow = split_root / "Shallow"
    public = prepared / "public"
    deep.mkdir(parents=True)
    shallow.mkdir(parents=True)
    public.mkdir(parents=True)
    (public / "train.csv").write_text("x\n1\n", encoding="utf-8")
    (deep / "train").symlink_to(deep / "train")

    source, layout = resolve_workspace_dataset_source(deep)
    assert source == public
    assert layout == "legacy_split_public_fallback"

    source, layout = resolve_workspace_dataset_source(split_root)
    assert source == public
    assert layout == "legacy_split_public_fallback"


def test_lhr_flat_dataset_repairs_broken_deep_media_symlink(tmp_path: Path) -> None:
    from scienceflow.solver.lnr.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    prepared = tmp_path / "prepared"
    split_root = prepared / "dataset_split"
    deep = split_root / "Deep"
    shallow = split_root / "Shallow"
    public = prepared / "public"
    deep.mkdir(parents=True)
    shallow.mkdir(parents=True)
    public.mkdir(parents=True)
    (public / "train_images").mkdir()
    (public / "train_images" / "000.jpg").write_text("image", encoding="utf-8")
    (deep / "train_metadata.json").write_text("{}", encoding="utf-8")
    (deep / "train_images").symlink_to("/missing/legacy/train_images")

    source, layout = resolve_workspace_dataset_source(deep)
    assert source == deep
    assert layout == "flat"

    source, layout = resolve_workspace_dataset_source(split_root)
    assert source == deep
    assert layout == "legacy_split_deep"

    ws_dataset = tmp_path / "workspace" / "dataset"
    prepare_workspace_dataset_flat(source, ws_dataset)
    assert (ws_dataset / "train_images").resolve() == (public / "train_images").resolve()
    assert (ws_dataset / "train_metadata.json").resolve() == (deep / "train_metadata.json").resolve()


def test_config_load_ignores_removed_deep_shallow_keys(tmp_path: Path) -> None:
    from scienceflow.config.settings import load_cfg

    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "enable_deep_shallow: true",
                "prep_shallow_fraction: 0.1",
                "deep_submission_dir: legacy_deep_submissions",
                "deep_solution_dir: legacy_deep_solutions",
                "agent:",
                "  deep_feedback_to_shallow: true",
                "lnr:",
                "  deep_enabled: true",
                "  ensemble_enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning):
        cfg = load_cfg(cfg_path, cli_args=False)

    assert not hasattr(cfg, "enable_deep_shallow")
    assert not hasattr(cfg, "prep_shallow_fraction")
    assert not hasattr(cfg, "deep_submission_dir")
    assert not hasattr(cfg.agent, "deep_feedback_to_shallow")
    assert not hasattr(cfg.lnr, "deep_enabled")


def test_lhr_state_packet_includes_stage_checkpoint_metadata(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        state_packet_max_chars=12000,
        state_packet_stage_card_max_chars=420,
        state_packet_archived_branch_max_chars=1500,
    )
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="snap-s01",
            snapshot_path=tmp_path / "snap-s01",
            metric_value=0.07,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=1,
            source_event={
                "source_commit_sha": "abcdef1234567890",
                "submission_snapshot": "submission_snapshots/s01.csv",
                "validation_ok": True,
                "lineage_id": "L01",
            },
            node_uid="W00:L01:S01",
            lineage_id="L01",
        ),
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="snap-s02",
            snapshot_path=tmp_path / "snap-s02",
            metric_value=0.08,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=2,
            source_event={
                "source_commit_sha": "123456abcdef9999",
                "submission_snapshot": "submission_snapshots/s02.csv",
                "validation_ok": True,
                "lineage_id": "L01",
            },
            node_uid="W00:L01:S02",
            lineage_id="L01",
        ),
    }

    packet = solver._build_lhr_state_packet(
        action="switch_stage",
        target_stage="S01",
        terminal_stage="S02",
        reason="S01 is better",
        tail_summary="S02 regressed",
    )

    assert "LHR State Packet v2" in packet
    assert "ESTRA action: switch_stage" in packet
    assert "ESTRA target stage: S01" in packet
    assert "Terminal stage before estra: S02" in packet
    assert "Best known restorable stage: S01" in packet
    assert "source_commit=abcdef123456" in packet
    assert "submission_snapshot=" not in packet
    assert "S02 regressed" in packet
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    built = [row for row in rows if row["event"] == "state_packet_built"][-1]
    assert built["payload"]["target_stage"] == "S01"
    assert built["payload"]["best_stage"] == "S01"


def test_lhr_state_packet_uses_adjudicated_metric_validity(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        state_packet_max_chars=12000,
        state_packet_stage_card_max_chars=420,
        state_packet_archived_branch_max_chars=1500,
    )
    solver.ledger_path.write_text(
        "### S01\n"
        "metric: 0.91\n"
        "lower_is_better: false\n"
        "metric_validity: high\n"
        "selection_eligible: true\n"
        "BRIEF: raw ledger claim\n"
        "WHY: raw\n\n"
        "### S02\n"
        "metric: 0.90\n"
        "lower_is_better: false\n"
        "metric_validity: high\n"
        "selection_eligible: true\n"
        "BRIEF: second\n"
        "WHY: second\n",
        encoding="utf-8",
    )
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="snap-s01",
            snapshot_path=tmp_path / "snap-s01",
            metric_value=0.91,
            metric_name="Final Validation Score",
            lower_is_better=False,
            memory_cut=1,
            source_event={
                "metric_value": 0.91,
                "lower_is_better": False,
                "metric_validity": "medium",
                "selection_eligible": False,
                "metric_validity_reason_code": "unknown_protocol",
                "source_commit_sha": "abcdef1234567890",
            },
            node_uid="W00:L01:S01",
            lineage_id="L01",
        ),
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="snap-s02",
            snapshot_path=tmp_path / "snap-s02",
            metric_value=0.90,
            metric_name="Final Validation Score",
            lower_is_better=False,
            memory_cut=2,
            source_event={"source_commit_sha": "123456abcdef9999"},
            node_uid="W00:L01:S02",
            lineage_id="L01",
        ),
    }

    packet = solver._build_lhr_state_packet(
        action="keep_current",
        target_stage="S02",
        terminal_stage="S02",
        reason="compact",
    )

    assert "metric_validity=medium" in packet
    assert "selection_eligible=false" in packet
    assert "metric_validity_reason=unknown_protocol" in packet
    s01_line = packet.split("- S01", 1)[1].split("- S02", 1)[0]
    assert "metric_validity=high" not in s01_line


def test_fullrun_tail_snapshot_records_actual_entrypoint_sha(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    source = ws / "train.py"
    source.write_text("print('train')\n")
    (ws / "submission.csv").write_text("id,target\n1,0\n")
    result = EnsureFullRunResult(
        executed=True,
        skipped=False,
        reason="ok",
        exit_code=0,
        wall_sec=1.0,
        metric_value=0.123,
        metric_name="Final Validation Score",
        lower_is_better=True,
        stdout="Final Validation Score: 0.123\n",
        stderr="",
    )
    write_fullrun_tail_snapshot(
        ws,
        result,
        bash_cmd="python3 train.py 2>&1",
        validation_ok=True,
        solution_path="train.py",
    )
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text())
    assert data["solution_path"] == "train.py"
    assert data["solution_sha"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_stage_ledger_accepts_compact_cards_and_bold_fields() -> None:
    text = """
### S01
**metric:** 0.062
**lower_is_better:** true
**run_time_sec:** 12.3
**metric_type:** holdout
**metric_note:** valid held-out validation evidence
**metric_validity:** high
**BRIEF:** catboost baseline
**WHY:** first valid checkpoint
**FILES:** code=train.py,model.py weights=models/fold0.ckpt
route_evidence: verdict=continue; reason=valid cheap baseline; next=blend logits
"""
    cards = parse_stage_cards(text)
    assert [c.stage_id for c in cards] == ["S01"]
    assert cards[0].metric == "0.062"
    assert cards[0].run_time_sec == "12.3"
    assert cards[0].metric_type == "holdout"
    assert cards[0].metric_note == "valid held-out validation evidence"
    assert cards[0].metric_validity == "high"
    assert cards[0].files == "code=train.py,model.py weights=models/fold0.ckpt"
    assert cards[0].route_evidence == "verdict=continue; reason=valid cheap baseline; next=blend logits"
    ok, reason, card = validate_stage_card(text, "S01")
    assert ok, reason
    assert card is not None and card.brief == "catboost baseline"
    assert card.route_evidence.startswith("verdict=continue")
    rendered = render_stage_cards(cards)
    assert "FILES: code=train.py,model.py weights=models/fold0.ckpt" in rendered
    assert next_stage_id(cards) == "S02"


def test_stage_ledger_requires_files_for_stage_commit() -> None:
    text = """
### S01
metric: 0.062
lower_is_better: true
BRIEF: catboost baseline
WHY: first valid checkpoint
"""
    ok, reason, card = validate_stage_card(text, "S01")

    assert not ok
    assert card is not None
    assert "FILES" in reason




def test_stage_event_summary_does_not_create_new_stage(tmp_path: Path) -> None:
    path = tmp_path / "run_results.md"
    path.write_text(
        "### S01\nmetric: 0.060612\nlower_is_better: true\nBRIEF: Optuna LGBM route\nWHY: validation route found\nFILES: code=train.py\n",
        encoding="utf-8",
    )
    append_stage_event_summary(
        path,
        target_stage="S01",
        summary="Ready submission generated from the same route; no new research stage.",
    )

    cards = parse_stage_cards(path.read_text(encoding="utf-8"))
    assert [card.stage_id for card in cards] == ["S01"]
    assert next_stage_id(cards) == "S02"
    assert cards[0].stage_events == ("Ready submission generated from the same route; no new research stage.",)
    rendered = render_stage_cards(cards)
    assert "EVENT: Ready submission generated" in rendered

def test_append_only_stage_commit_allows_estra_summary_then_next_stage(tmp_path: Path) -> None:
    before = """# Run Results

### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py
"""
    new_file = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py
"""
    ok, reason = validate_append_only_stage_commit("", new_file, "S01")
    assert ok, reason

    path = tmp_path / "run_results.md"
    path.write_text(before)
    append_archived_trajectory_summary(path, target_stage="S01", summary="S02 regressed and was abandoned")
    with_summary = path.read_text()
    assert with_summary.startswith(before)
    assert "Archived Trajectory before estra to S01" in with_summary
    assert "HISTORICAL_EXPLORATION" in with_summary
    assert "not the active route" in with_summary

    after = with_summary + "\n### S02\nmetric: 0.06\nlower_is_better: true\nBRIEF: new branch after estra\nWHY: improves from restored stage\nFILES: code=train.py\n"
    ok, reason = validate_append_only_stage_commit(with_summary, after, "S02")
    assert ok, reason


def test_append_only_stage_commit_rejects_rewriting_existing_stage() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py

### S02
metric: 0.06
lower_is_better: true
BRIEF: original branch
WHY: first improvement
FILES: code=train.py
"""
    after = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py

### S02
metric: 0.05
lower_is_better: true
BRIEF: rewritten branch
WHY: silently changed old stage
FILES: code=train.py

### S03
metric: 0.049
lower_is_better: true
BRIEF: next branch
WHY: appended after rewrite
FILES: code=train.py
"""
    ok, reason = validate_append_only_stage_commit(before, after, "S03")
    assert not ok
    assert "append-only" in reason


def test_salvage_append_only_stage_commit_extracts_new_stage_from_rewrite() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: preserved baseline
WHY: original branch
FILES: code=train.py
"""
    attempted = """### S01
metric: 0.01
lower_is_better: true
BRIEF: incorrectly rewritten baseline
WHY: should be ignored
FILES: code=train.py

### S02
metric: 0.06
lower_is_better: true
BRIEF: new branch after estra
WHY: valid new stage
FILES: code=train.py
"""
    ok, salvaged, reason = salvage_append_only_stage_commit(before, attempted, "S02")
    assert ok, reason
    assert salvaged.startswith(before)
    assert "incorrectly rewritten" not in salvaged
    assert "new branch after estra" in salvaged
    ok, reason = validate_append_only_stage_commit(before, salvaged, "S02")
    assert ok, reason


def test_append_only_stage_commit_rejects_stage_gap() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py
"""
    after = before + "\n### S03\nmetric: 0.06\nlower_is_better: true\nBRIEF: skipped S02 after estra\nWHY: invalid gap\nFILES: code=train.py\n"
    ok, reason = validate_append_only_stage_commit(before, after, "S03")
    assert not ok
    assert "next active stage" in reason


def test_append_only_stage_commit_rejects_stage_regression() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py

### S03
metric: 0.06
lower_is_better: true
BRIEF: later stage
WHY: global monotonic id
FILES: code=train.py
"""
    after = before + "\n### S02\nmetric: 0.065\nlower_is_better: true\nBRIEF: duplicate old id\nWHY: invalid numbering\nFILES: code=train.py\n"
    ok, reason = validate_append_only_stage_commit(before, after, "S02")
    assert not ok
    assert "next active stage" in reason


def test_estra_tail_summary_is_non_selectable(tmp_path: Path) -> None:
    ledger = """
### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start

### S02
metric: 0.06
lower_is_better: true
BRIEF: tuned model
WHY: better validation
"""
    summary = tail_summary_after(ledger, "S01", max_chars=500)
    assert "S02" in summary
    path = tmp_path / "run_results.md"
    path.write_text("### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: baseline\nWHY: start\n")
    append_estra_summary(path, target_stage="S01", summary=summary)
    text = path.read_text()
    assert "## ESTRA Summary after S01" in text
    assert [c.stage_id for c in parse_stage_cards(text)] == ["S01"]




def test_estra_compact_summary_always_starts_from_s02() -> None:
    ledger = """
### S01
metric: 0.070
lower_is_better: true
BRIEF: base EDA and first valid solution
WHY: preserve root understanding

### S02
metric: 0.066
lower_is_better: true
BRIEF: first branch
WHY: tries more features

### S03
metric: 0.064
lower_is_better: true
BRIEF: restored target candidate
WHY: useful code state

### S04
metric: 0.068
lower_is_better: true
BRIEF: abandoned tail
WHY: no gain
"""
    summary = tail_summary_from_stage(ledger, start_stage="S02", target_stage="S03", max_chars=1000)
    assert "S01" not in summary
    assert "S02" in summary
    assert "S03" in summary
    assert "S04" in summary
    assert "estra to S03" in summary


def test_snapshot_store_preserves_logs_and_memory(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / ".logs").mkdir()
    (ws / ".logs" / "fullrun_tail_snapshot.json").write_text("{}\n")
    (ws / ".memory" / "ScienceAgent").mkdir(parents=True)
    (ws / ".memory" / "ScienceAgent" / "short_term.json").write_text("{}\n")
    (ws / "solution.py").write_text("print('ok')\n")
    store = SnapshotStore(
        root_dir=tmp_path,
        workspace_dir=ws,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    snap = store.capture(
        stage_id="S01",
        metric_value=0.1,
        metric_name="score",
        lower_is_better=True,
        memory_cut=1,
        source_event={"metric_value": 0.1},
    )
    assert (snap.snapshot_path / ".logs" / "fullrun_tail_snapshot.json").is_file()
    assert (snap.snapshot_path / ".memory" / "ScienceAgent" / "short_term.json").is_file()
    (ws / "solution.py").write_text("print('changed')\n")
    archive = store.restore(snap)
    assert archive.is_dir()
    archive_meta = json.loads((archive / "terminal_archive_meta.json").read_text())
    archive_manifest = json.loads(Path(archive_meta["workspace_snapshot_manifest_path"]).read_text())
    assert archive_manifest["entries"]["solution.py"]["size"] == len("print('changed')\n")
    assert not (archive / "solution.py").exists()
    assert (ws / "solution.py").read_text() == "print('ok')\n"
    assert (ws / ".logs").is_dir()
    assert (ws / ".memory").is_dir()


def test_snapshot_store_falls_back_to_workspace_root_when_stage_root_blocked(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    ws = worker_root / "workspace"
    ws.mkdir(parents=True)
    (ws / "solution.py").write_text("print('ok')\n")
    blocked_root = worker_root / "snaps"
    original_mkdir = Path.mkdir

    def guarded_mkdir(self: Path, *args, **kwargs):
        if self == blocked_root:
            raise PermissionError("blocked snapshot root")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=ws,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.1,
        metric_name="score",
        lower_is_better=True,
        memory_cut=0,
        source_event={"metric_value": 0.1},
    )

    assert snap.snapshot_path.parent == ws / "snaps"
    assert (snap.snapshot_path / "solution.py").read_text() == "print('ok')\n"
    assert not (snap.snapshot_path / "snaps").exists()


def test_submission_sha_duplicate_stage_lookup() -> None:
    snap = StageSnapshot(
        stage_id="S01",
        snapshot_id="S01-abc",
        snapshot_path=Path("/tmp/snap"),
        metric_value=0.06,
        metric_name="Final Validation Score",
        lower_is_better=True,
        memory_cut=3,
        source_event={"submission_sha": "same-submission"},
    )
    assert LnrSolver._stage_with_submission_sha({"S01": snap}, "same-submission") is snap
    assert LnrSolver._stage_with_submission_sha({"S01": snap}, "different") is None
    assert LnrSolver._stage_with_submission_sha({"S01": snap}, "") is None


def test_artifact_sha_duplicate_stage_lookup() -> None:
    snap = StageSnapshot(
        stage_id="S01",
        snapshot_id="S01-abc",
        snapshot_path=Path("/tmp/snap"),
        metric_value=2.5,
        metric_name="radii_sum",
        lower_is_better=False,
        memory_cut=3,
        source_event={"artifact_sha": "same-artifact"},
    )
    assert LnrSolver._stage_with_artifact_sha({"S01": snap}, "same-artifact") is snap
    assert LnrSolver._stage_with_artifact_sha({"S01": snap}, "different") is None
    assert LnrSolver._stage_with_artifact_sha({"S01": snap}, "") is None


def test_estra_parser_recovers_deepseek_switch_text_and_pseudo_tool() -> None:
    raw = """<think>S02 is worse than S01. Let me estra to S01 and improve from there.</think>
<｜｜DSML｜｜tool_calls>cat > run_results.md ...</｜｜DSML｜｜tool_calls>"""
    parsed = LnrSolver._parse_estra_decision(
        raw,
        switch_candidates=["S01"],
    )
    assert parsed["action"] == "switch_stage"
    assert parsed["target_stage"] == "S01"


def test_estra_parser_uses_first_json_object_not_greedy_block() -> None:
    raw = 'prefix {"action":"switch_stage","target_stage":"S02","reason":"better root"} suffix {"command":"ignored"}'
    parsed = LnrSolver._parse_estra_decision(
        raw,
        switch_candidates=["S01", "S02"],
    )
    assert parsed["target_stage"] == "S02"

def test_estra_parser_accepts_keep_but_redirect_diagnostics() -> None:
    raw = json.dumps(
        {
            "action": "keep_but_redirect",
            "exploration_summary": "S04-S06 repeated local tweaks.",
            "bottleneck": "The route has not diagnosed validation weakness.",
            "evidence": "Metric stayed flat across recent stages.",
            "decision_reason": "Keep files but redirect the next stage.",
            "redirect_focus": "Audit validation and missing signal.",
        }
    )
    parsed = LnrSolver._parse_estra_decision(raw, switch_candidates=["S01"])
    assert parsed["action"] == "keep_but_redirect"
    assert "validation weakness" in parsed["bottleneck"]


def test_estra_parser_accepts_two_axis_previous_stage_redirect(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    raw = json.dumps(
        {
            "startpoint": "previous_stage",
            "intent": "redirect",
            "target_stage": "S01",
            "exploration_summary": "Current path repeated shallow changes.",
            "bottleneck": "The restored route needs a different validation audit.",
            "evidence": "Recent stages plateaued after parameter tweaks.",
            "decision_reason": "Restore S01 but redirect around the bottleneck.",
            "redirect_focus": "Audit CV split and missing signal.",
        }
    )
    parsed = LnrSolver._parse_estra_decision(raw, switch_candidates=["S01"])
    normalized = solver._normalize_estra_decision(
        parsed,
        latest="S03",
        switch_candidates=["S01"],
        trigger_source="force_stage_capture",
    )
    assert normalized["action"] == "switch_stage"
    assert normalized["startpoint"] == "previous_stage"
    assert normalized["intent"] == "redirect"
    assert normalized["target_stage"] == "S01"
    assert normalized["compact"] is True


def test_multi_worker_cpu_slice_is_contiguous() -> None:
    ids = list(range(128, 160))
    assert LnrSolver._slice_cpu_ids(ids, worker_index=0, worker_count=2) == list(range(128, 144))
    assert LnrSolver._slice_cpu_ids(ids, worker_index=1, worker_count=2) == list(range(144, 160))
    assert LnrSolver._format_cpu_ids(list(range(128, 144))) == "128-143"


def test_multi_worker_extra_env_exposes_worker_cpu_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCIENCEFLOW_TASK_CPU_LIST", "96-127")
    solver = object.__new__(LnrSolver)
    solver.cfg = SimpleNamespace(exec=SimpleNamespace(cpu_list=""))
    solver.lhr = SimpleNamespace(omp_threads_cap=8, seed=0)

    env = solver._worker_extra_env(worker_index=2, worker_count=4)

    assert env["SCIENCEFLOW_TASK_CPU_LIST"] == "96-127"
    assert env["SCIENCEFLOW_WORKER_CPU_LIST"] == "112-119"
    assert env["SCIENCEFLOW_CPU_LIST"] == "112-119"
    assert env["_SCIENCEFLOW_CPU_SET"] == "112-119"
    assert env["OMP_NUM_THREADS"] == "8"


def test_multi_worker_failure_kind_uses_llm_quota_error() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[
            {"status": "failed", "error": "APIStatusError: Error code: 402 - Insufficient Balance"},
            {"status": "failed", "error": "APIStatusError: Error code: 402 - Insufficient Balance"},
        ],
    )

    assert failure_kind == "llm_quota_error"
    assert kinds == ["llm_quota_error"]


def test_multi_worker_failure_kind_reports_mixed_worker_failures() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[
            {"status": "failed", "error": "APIStatusError: Error code: 402 - Insufficient Balance"},
            {"status": "failed", "error": "RuntimeError: lnr compact did not fit context"},
        ],
    )

    assert failure_kind == "worker_failed_mixed"
    assert kinds == ["context_compact_failed", "llm_quota_error"]


def test_multi_worker_failure_kind_handles_failed_workers_without_error() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[{"status": "failed"}, {"status": "failed", "error": ""}],
    )

    assert failure_kind == "all_workers_failed"
    assert kinds == []


def test_multi_worker_stop_reason_preserves_success_budget_semantics() -> None:
    reason, kinds = LnrSolver._multi_worker_stop_reason(
        run_succeeded=True,
        worker_results=[{"status": "success", "error": ""}],
    )

    assert reason == "budget_expired"
    assert kinds == []


def test_worker_error_kind_classifies_tool_output_and_transport_errors() -> None:
    assert (
        LnrSolver._worker_error_kind("ValueError: Separator is found, but chunk is longer than limit")
        == "tool_output_limit"
    )
    assert LnrSolver._worker_error_kind("ReadError: stream disconnected") == "llm_transport_error"


def test_multi_worker_partial_merge_does_not_mask_all_worker_failure(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = object.__new__(LnrSolver)
        solver.lhr = SimpleNamespace(num_workers=2, merge_enabled=True)
        solver.root_dir = tmp_path / "root"
        solver.log_dir = tmp_path / "logs"
        solver.solver_name = "lnr"
        solver.ledger_filename = "run_results.md"
        solver.cfg = SimpleNamespace(submission_dir=None)
        statuses: list[tuple[str, dict[str, object]]] = []
        events: list[dict[str, object]] = []
        worker_results = [
            {
                "worker_id": "W00",
                "worker_index": 0,
                "status": "failed",
                "error": "ValueError: Separator is found, but chunk is longer than limit",
            },
            {
                "worker_id": "W01",
                "worker_index": 1,
                "status": "failed",
                "error": "ReadError: stream disconnected",
            },
        ]
        candidate = {"candidate_id": "W01:S01", "metric_value": 0.5}

        solver.state_machine = SimpleNamespace(
            mark_run_status=lambda status, payload=None: statuses.append((status, dict(payload or {})))
        )
        solver._jsonl = lambda _name, record: events.append(dict(record))

        async def run_one_worker(worker_index: int, _worker_count: int) -> dict[str, object]:
            return dict(worker_results[worker_index])

        solver._run_one_worker = run_one_worker
        solver._aggregate_worker_state = lambda **_kwargs: None
        solver._load_worker_candidates = lambda result: [candidate] if result.get("worker_id") == "W01" else []
        solver._write_stage_collection_outputs = lambda **_kwargs: tmp_path / "merge"
        solver._write_global_time_trace = lambda _worker_results: None
        solver._cleanup_coordinator_workspace_shell = lambda: None
        solver._merge_dir = lambda: tmp_path / "merge"

        async def write_merge_outputs(**_kwargs: object) -> dict[str, object]:
            return dict(candidate)

        solver._write_merge_outputs = write_merge_outputs

        result = await LnrSolver._run_multi_worker(solver)

        assert result["status"] == "failed"
        assert result["stop_reason"] == ""
        assert result["failure_kind"] == "worker_failed_mixed"
        assert result["worker_error_kinds"] == ["llm_transport_error", "tool_output_limit"]
        assert result["partial_merge_available"] is True
        assert "selected_candidate_id" not in result
        assert result["merge_enabled"] is True
        assert statuses[-1][0] == "failed"
        assert statuses[-1][1]["partial_merge_available"] is True
        done = [event for event in events if event.get("event") == "multi_worker_done"]
        assert done and done[-1]["status"] == "failed"

    asyncio.run(_run())


def test_multi_worker_does_not_require_merge_enabled() -> None:
    async def _run() -> None:
        solver = object.__new__(LnrSolver)
        solver.lhr = SimpleNamespace(num_workers=3, merge_enabled=False)
        events: list[tuple[str, dict[str, object]]] = []
        called: list[str] = []

        solver._jsonl = lambda name, record: events.append((name, record))

        async def run_multi_worker() -> dict[str, object]:
            called.append("multi")
            return {"mode": "multi", "merge_enabled": solver.lhr.merge_enabled}

        async def run_single() -> dict[str, object]:
            called.append("single")
            return {"mode": "single"}

        solver._run_multi_worker = run_multi_worker
        solver._run_single = run_single

        result = await LnrSolver.run(solver)

        assert result == {"mode": "multi", "merge_enabled": False}
        assert called == ["multi"]
        assert solver.lhr.merge_enabled is False
        assert events == []

    asyncio.run(_run())


def test_lhr_run_uses_coordinator_layout_for_one_worker() -> None:
    async def _run() -> None:
        solver = object.__new__(LnrSolver)
        solver.lhr = SimpleNamespace(num_workers=1, merge_enabled=True)
        called: list[str] = []

        async def run_multi_worker() -> dict[str, object]:
            called.append("multi")
            return {"mode": "multi", "num_workers": solver.lhr.num_workers}

        async def run_single() -> dict[str, object]:  # pragma: no cover - must not run
            called.append("single")
            return {"mode": "single"}

        solver._run_multi_worker = run_multi_worker
        solver._run_single = run_single

        result = await LnrSolver.run(solver)

        assert result == {"mode": "multi", "num_workers": 1}
        assert called == ["multi"]

    asyncio.run(_run())


def test_multi_worker_budget_reserves_time_for_global_merge() -> None:
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        merge_enabled=True,
        wall_clock_budget_sec=3600,
        global_merge_wall_clock_sec=900,
    )

    assert LnrSolver._global_merge_reserve_sec(solver) == 540
    assert LnrSolver._worker_wall_clock_budget_sec(solver) == 3060


def test_multi_worker_stage_collection_writes_candidates_without_selection(tmp_path: Path) -> None:
    solver = object.__new__(LnrSolver)
    solver._merge_dir = lambda: tmp_path / "merge"
    worker_results = [
        {"worker_id": "W00", "status": "success", "best_stage": "S01", "best_metric": 0.1},
        {"worker_id": "W01", "status": "success", "best_stage": "S02", "best_metric": 0.2},
    ]
    candidates = [
        {"candidate_id": "W00:S01", "metric_value": 0.1, "snapshot_path": "/tmp/w00/s01", "submission_sha": "aaa"},
        {"candidate_id": "W01:S02", "metric_value": 0.2, "snapshot_path": "/tmp/w01/s02", "submission_sha": "bbb"},
    ]

    out_dir = LnrSolver._write_stage_collection_outputs(
        solver,
        worker_results=worker_results,
        candidates=candidates,
    )

    assert out_dir == tmp_path / "merge"
    assert (out_dir / "global_candidates.jsonl").read_text(encoding="utf-8").count("candidate_id") == 2
    stage_map = json.loads((out_dir / "global_stage_map.json").read_text(encoding="utf-8"))
    assert "selected" not in stage_map
    assert stage_map["merge"]["merge_mode"] == "not_run"
    report = (out_dir / "stage_collection_report.md").read_text(encoding="utf-8")
    assert "candidate_count: 2" in report
    assert "W00:S01" in report


def test_validation_leakage_guard_flags_train_val_metric_reuse() -> None:
    source = """
import pandas as pd
train = pd.read_csv('dataset/train.csv')
val = pd.read_csv('dataset/validation.csv')
full_train = pd.concat([train, val], axis=0)
print(f"Final Validation Score: {0.029353:.6f}")
"""
    stdout = """
=== Training final models ===
=== Optimizing ensemble weights on validation ===
=== Final Validation Score: 0.029353 ===
"""
    assert _validation_leakage_reason(source, stdout)


def test_validation_leakage_guard_allows_post_metric_final_retrain() -> None:
    source = """
print(f"Final Validation Score: {0.061:.6f}")
full_train = pd.concat([train, val], axis=0)
"""
    assert _validation_leakage_reason(source, "Final Validation Score: 0.061") == ""

def test_metric_semantics_detects_multifile_fulltrain_replay(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "train.py").write_text(
        """
import pandas as pd
train = pd.read_csv('dataset/train.csv')
val = pd.read_csv('dataset/validation.csv')
X_full = pd.concat([train, val], axis=0)
""",
        encoding="utf-8",
    )
    (solver.workspace_dir / "predict.py").write_text(
        "print('predict using saved full-train artifacts')\n",
        encoding="utf-8",
    )
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.0004834944942880695,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": (
                    "Model trained on train+val (2160 samples)\n"
                    "Loading validation features for score computation...\n"
                    "Final Validation Score: 0.0004834944942880695\n"
                ),
                "stderr_tail": "",
                "bash_cmd": "python3 predict.py 2>&1",
                "solution_path": "predict.py",
                "validation_ok": True,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["reported_val_score"] == 0.0004834944942880695
    assert event["val_score_type"] == "post_fulltrain_replay"
    assert event["selection_eligible"] is False
    assert event["selection_score"] == ""
    assert event["selection_note"] == "fulltrain_replay_metric_kept_but_not_used_for_selection"
    assert event["metric_validity"] == "low"
    assert event["execution_mode"] == "predict_only_reuse_artifacts"
    assert event["validation_ok"] is True


def test_metric_event_uses_task_direction_over_declared_snapshot_value(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.task_desc = """
    ## Target metric (evaluation)
    Kendall tau over all notebooks (higher better).
    """
    solver._task_metric_lower_is_better = False
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "score_validation.py").write_text("print('Final Validation Score: 0.742407')\n", encoding="utf-8")
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.742407,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": "Final Validation Score: 0.742407\n",
                "stderr_tail": "",
                "bash_cmd": "python3 score_validation.py",
                "solution_path": "score_validation.py",
                "validation_ok": True,
                "val_score_type": "holdout",
                "selection_eligible": True,
                "candidate_ready": True,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["lower_is_better"] is False
    assert event["declared_lower_is_better"] is True
    assert event["metric_direction_source"] == "task_description"
    assert event["metric_direction_conflict"] is True


def test_authoritative_evaluator_direction_overrides_task_text_hint(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver._task_metric_lower_is_better = True
    event = {
        "metric_name": "global_ndcg",
        "lower_is_better": False,
        "metric_authoritative": True,
    }

    decision = solver._metric_lower_is_better_decision(event)

    assert decision["lower_is_better"] is False
    assert decision["metric_direction_source"] == "authoritative_evaluator"
    assert decision["metric_direction_conflict"] is False


def test_stage_result_audit_rewrites_formal_entry_before_commit(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Kendall tau over notebooks, higher better."
        solver._task_metric_lower_is_better = False
        solver.lhr = SimpleNamespace(metric_validity_adjudicator_enabled=False)
        metric_event = {
            "metric_value": 0.99,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "declared_lower_is_better": True,
            "metric_validity": "high",
            "selection_eligible": True,
            "candidate_ready": True,
            "validation_ok": True,
            "val_score_type": "holdout",
            "submission_status": "ready",
            "run_time_sec": 12.0,
        }
        judgment = {
            "brief": "Stacking route reports a strong validation score.",
            "why": "Meta model overfits the same validation set, so this should not drive selection.",
            "metric_validity": "high",
            "lower_is_better": "true",
        }

        await solver._audit_stage_result_before_commit(
            agent=SimpleNamespace(),
            stage_id="S02",
            metric_event=metric_event,
            judgment=judgment,
            block_text="STAGE_COMMIT_BEGIN\nlower_is_better: true\nmetric_validity: high\nSTAGE_COMMIT_END",
            source="main_agent_text_block",
        )
        entry = solver._stage_commit_entry_from_metric_event(stage_id="S02", metric_event=metric_event, judgment=judgment)

        assert metric_event["lower_is_better"] is False
        assert metric_event["metric_direction_source"] == "task_description"
        assert metric_event["metric_direction_conflict"] is True
        assert metric_event["metric_validity"] == "low"
        assert metric_event["selection_eligible"] is False
        assert "lower_is_better: false" in entry
        assert "metric_validity: low" in entry
        assert "BRIEF:" in entry
        assert "WHY:" in entry

    asyncio.run(_run())


def test_stage_commit_files_filter_keeps_code_and_ml_weights_only(tmp_path: Path) -> None:
    (tmp_path / "solve.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "repair.py").write_text("print(2)\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "helper.py").write_text("print(3)\n", encoding="utf-8")
    (tmp_path / "best_solution.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fold0.ckpt").write_text("weights\n", encoding="utf-8")

    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S03",
        metric_event={"metric_value": 1.2, "lower_is_better": False, "solution_path": "solve.py"},
        judgment={
            "brief": "repair route",
            "why": "valid method code should be preserved",
            "files": "code=solve.py,repair.py,outputs/helper.py weights=best_solution.json,models/fold0.ckpt",
        },
        workspace_dir=tmp_path,
    )

    assert "FILES: code=solve.py,repair.py weights=models/fold0.ckpt" in entry
    assert "best_solution.json" not in entry
    assert "outputs/helper.py" not in entry
    cards = parse_stage_cards(entry)
    assert cards[0].files == "code=solve.py,repair.py weights=models/fold0.ckpt"


def test_stage_commit_preserves_full_brief_and_why_in_ledger() -> None:
    long_brief = " ".join(["geometry feature branch"] * 20)
    long_why = " ".join(["preserve this route lesson because the validation behavior differs"] * 30)

    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S04",
        metric_event={
            "metric_value": 0.061,
            "lower_is_better": True,
            "run_time_sec": 12.0,
            "val_score_type": "holdout",
            "metric_validity": "high",
        },
        judgment={
            "brief": long_brief,
            "why": long_why,
            "files": "code=train.py",
        },
    )

    assert f"BRIEF: {long_brief}" in entry
    assert f"WHY: {long_why}" in entry
    assert "..." not in entry
    card = parse_stage_cards(entry)[0]
    assert card.brief == long_brief
    assert card.why == long_why


def test_stage_commit_rejects_invalid_files_instead_of_guessing_workspace_code(tmp_path: Path) -> None:
    for name in ("util.py", "train.py", "predict.py"):
        (tmp_path / name).write_text("print('stage')\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "helper.py").write_text("print('output')\n", encoding="utf-8")

    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S04",
        metric_event={"metric_value": 0.0609, "lower_is_better": True},
        judgment={
            "brief": "bowing features improved CV.",
            "why": "valid candidate should remain traceable.",
            "files": 'code name, df in [=("train", train), ("val", val)]',
        },
        workspace_dir=tmp_path,
    )

    assert "FILES:" not in entry
    ok, reason, _card = validate_stage_card(entry, "S04")
    assert not ok
    assert "FILES" in reason


def test_stage_commit_accepts_explicit_no_files_without_retry(tmp_path: Path) -> None:
    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S04",
        metric_event={"metric_value": 0.0609, "lower_is_better": True},
        judgment={
            "brief": "artifact-only evaluator candidate.",
            "why": "valid evaluator metric with no reusable code or weights.",
            "files": "none",
        },
        workspace_dir=tmp_path,
    )

    assert "FILES: none" in entry
    ok, reason, card = validate_stage_card(entry, "S04")
    assert ok, reason
    assert card is not None
    assert card.files == "none"


def test_metric_semantics_system_overrides_declared_holdout_replay(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "predict.py").write_text("print('predict')\n", encoding="utf-8")
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.0123,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": (
                    "Metric Protocol: holdout\n"
                    "Model trained on train+val (2160 samples)\n"
                    "Loading validation features for score computation...\n"
                    "Final Validation Score: 0.0123\n"
                ),
                "stderr_tail": "",
                "bash_cmd": "python3 predict.py",
                "solution_path": "predict.py",
                "validation_ok": True,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["metric_protocol"] == "holdout"
    assert event["val_score_type"] == "post_fulltrain_replay"
    assert event["selection_eligible"] is False
    assert event["selection_note"] == "agent_declared_holdout_but_system_detected_fulltrain_replay"
    assert event["metric_validity"] == "low"


def test_metric_semantics_allows_holdout_then_full_retrain(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "train.py").write_text(
        """
print('Final Validation Score: 0.061')
full_train = pd.concat([train, val], axis=0)
print('Retraining on full train+val for submission')
""",
        encoding="utf-8",
    )
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.061,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": (
                    "Validation RMSLE: 0.061\n"
                    "Retraining on full train+val for submission\n"
                    "Final Validation Score: 0.061\n"
                ),
                "stderr_tail": "",
                "bash_cmd": "python3 train.py",
                "solution_path": "train.py",
                "validation_ok": True,
                "wall_sec": 8.75,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["val_score_type"] == "holdout"
    assert event["selection_eligible"] is True
    assert event["selection_score"] == 0.061
    assert event["metric_validity"] == "high"
    assert event["run_time_sec"] == 8.75


def test_snapshot_store_does_not_create_roots_until_used(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = SnapshotStore(
        root_dir=tmp_path,
        workspace_dir=ws,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    assert not (tmp_path / "snaps").exists()
    assert not (tmp_path / "archives").exists()
    snap = store.capture(
        stage_id="S01",
        metric_value=0.1,
        metric_name="score",
        lower_is_better=True,
        memory_cut=0,
        source_event={"metric_value": 0.1},
    )
    assert snap.snapshot_path.is_dir()
    assert (tmp_path / "snaps").is_dir()
    assert not (tmp_path / "archives").exists()

def test_lhr_state_machine_store_writes_unified_events_and_state(tmp_path: Path) -> None:
    store = LHRStateMachineStore(
        log_dir=tmp_path,
        worker_id="W00",
        worker_index=0,
        worker_count=1,
        ledger_filename="run_results.md",
    )
    store.mark_run_status("running")
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-abc",
            "metric_event": {
                "metric_value": 0.061,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    store.append_event(
        "estra_invalid",
        task_type="estra_decision",
        task_id="estra_decision:W00:S01",
        status="failed",
        payload={"raw": "S10"},
    )
    events = (tmp_path / "lhr_events.jsonl").read_text().splitlines()
    assert len(events) == 3
    state = __import__("json").loads((tmp_path / "lhr_state.json").read_text())
    assert state["run_status"] == "running"
    assert state["stage_count"] == 1
    assert state["estra_invalid_decisions"] == 1
    assert state["global_best"]["candidate_id"] == "W00:S01"
    assert state["workers"]["W00"]["stage_count"] == 1

def test_lhr_control_payload_relativizes_workspace_paths(tmp_path: Path) -> None:
    solver = object.__new__(LnrSolver)
    solver.task_root_dir = tmp_path / "task"
    solver.root_dir = tmp_path / "task"
    solver.workspace_dir = tmp_path / "task" / "workspace"
    payload = {
        "snapshot_path": str(tmp_path / "task" / ".snapshots" / "S01-abc"),
        "metric_event": {"snapshot_path": str(tmp_path / "task" / "workspace" / ".logs" / "fullrun_tail_snapshot.json")},
    }
    clean = solver._relativize_control_payload(payload)
    assert clean["snapshot_path"] == "./.snapshots/S01-abc"
    assert clean["metric_event"]["snapshot_path"] == "workspace/.logs/fullrun_tail_snapshot.json"

def test_lhr_next_stage_id_follows_active_ledger_after_estra(tmp_path: Path) -> None:
    solver = object.__new__(LnrSolver)
    solver.ledger_path = tmp_path / "run_results.md"
    solver.stage_snapshots = {"S01": object(), "S02": object()}
    solver.ledger_path.write_text("### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: restored root\nWHY: estra target\n")
    assert solver._next_stage_id_for_logging() == "S02"

def test_lhr_node_uid_uses_worker_lineage_and_visible_stage(tmp_path: Path) -> None:
    solver = object.__new__(LnrSolver)
    solver.worker_id = "W01"
    solver.worker_index = 1
    solver.current_lineage_id = "L03"
    solver.current_lineage_no = 3

    assert solver._stage_node_uid("S02") == "W01:L03:S02"


def test_lhr_parallel_worker_snapshot_uses_global_best_per_worker(tmp_path: Path) -> None:
    solver = object.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(worker_peer_summary_enabled=True, worker_peer_summary_max_chars=2200)
    solver.worker_id = "W01"
    solver.worker_index = 1
    solver.worker_count = 3
    solver.global_log_dir = tmp_path / "logs"
    solver.global_log_dir.mkdir()
    perf = solver.global_log_dir / "lhr_stage_performance.csv"
    with perf.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LHR_STAGE_PERFORMANCE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "row_order": "1",
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "worker_index": "0",
                "worker_stage_order": "1",
                "stage_id": "S01",
                "lineage_id": "L01",
                "node_uid": "W00:L01:S01",
                "metric_value": "0.090000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "brief": "baseline features",
                "why": "first try",
                "elapsed_min": "2.0",
            }
        )
        writer.writerow(
            {
                "row_order": "2",
                "candidate_id": "W00:L01:S02",
                "worker_id": "W00",
                "worker_index": "0",
                "worker_stage_order": "2",
                "stage_id": "S02",
                "lineage_id": "L01",
                "node_uid": "W00:L01:S02",
                "metric_value": "0.070000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "brief": "lightgbm tuned",
                "why": "better",
                "elapsed_min": "12.3",
            }
        )
        writer.writerow(
            {
                "row_order": "3",
                "candidate_id": "W02:L01:S01",
                "worker_id": "W02",
                "worker_index": "2",
                "worker_stage_order": "1",
                "stage_id": "S01",
                "lineage_id": "L01",
                "node_uid": "W02:L01:S01",
                "metric_value": "0.010000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "0",
                "validation_issue": "train_validation_concat_before_reported_metric",
                "brief": "invalid leak",
                "why": "skip",
                "elapsed_min": "5.5",
            }
        )
        writer.writerow(
            {
                "row_order": "4",
                "candidate_id": "W02:L01:S02",
                "worker_id": "W02",
                "worker_index": "2",
                "worker_stage_order": "2",
                "stage_id": "S02",
                "lineage_id": "L01",
                "node_uid": "W02:L01:S02",
                "metric_value": "0.080000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "brief": "safe ridge baseline",
                "why": "valid fallback",
                "elapsed_min": "8.0",
            }
        )

    snapshot = solver._parallel_worker_snapshot_for_prompt()

    assert "There are 3 workers" in snapshot
    assert "- W00: best_metric=0.070000 (ok) at L01:S02; elapsed=12.3m; method=lightgbm tuned" in snapshot
    assert "- W01 (this worker): no metric-backed stage recorded yet." in snapshot
    assert "- W02: best_metric=0.010000 at L01:S01; elapsed=5.5m; status=suspicious" in snapshot
    assert "ok_best_metric=0.080000 at L01:S02; ok_elapsed=8.0m; ok_method=safe ridge baseline" in snapshot
    assert "train_validation_concat_before_reported_metric" in snapshot


def test_lhr_estra_prunes_active_snapshots_to_restored_ledger(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.ledger_path.write_text(
        "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: restored root\nWHY: estra target\n",
        encoding="utf-8",
    )
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="S01-a",
            snapshot_path=tmp_path / "snapshots" / "S01-a",
            metric_value=0.07,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=2,
            source_event={"submission_sha": "keep"},
        ),
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="S02-b",
            snapshot_path=tmp_path / "snapshots" / "S02-b",
            metric_value=0.08,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=4,
            source_event={"submission_sha": "abandoned"},
        ),
    }

    solver._prune_active_stage_snapshots_to_ledger()

    assert sorted(solver.stage_snapshots) == ["S01"]
    assert solver._next_stage_id_for_logging() == "S02"
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    pruned = [row for row in rows if row["event"] == "estra_active_stage_pruned"][-1]
    assert pruned["payload"]["active_stages"] == ["S01"]
    assert pruned["payload"]["removed_stages"] == ["S02"]


def test_lhr_jsonl_writes_unified_events_without_legacy_files(tmp_path: Path) -> None:
    solver = object.__new__(LnrSolver)
    solver.log_dir = tmp_path / "logs"
    solver.root_dir = tmp_path
    solver.task_root_dir = tmp_path
    solver.workspace_dir = tmp_path / "workspace"
    solver.worker_id = ""
    solver.worker_index = 0
    solver.state_machine = LHRStateMachineStore(
        log_dir=solver.log_dir,
        worker_id="W00",
        worker_index=0,
        worker_count=1,
        ledger_filename="run_results.md",
    )
    solver._jsonl("lhr_stage_events.jsonl", {"event": "stage_capture_failed", "stage_id": "S01"})
    assert not (solver.log_dir / "lhr_stage_events.jsonl").exists()
    assert (solver.log_dir / "lhr_events.jsonl").is_file()
    assert solver._count_jsonl_events("lhr_stage_events.jsonl", "stage_capture_failed") == 1


def test_lhr_state_machine_aggregates_worker_event_logs(tmp_path: Path) -> None:
    out = tmp_path / "logs"
    w0 = tmp_path / "w00" / "logs"
    w1 = tmp_path / "w01" / "logs"
    s0 = LHRStateMachineStore(log_dir=w0, worker_id="W00", worker_index=0, worker_count=2)
    s1 = LHRStateMachineStore(log_dir=w1, worker_id="W01", worker_index=1, worker_count=2)
    s0.mark_run_status("finished")
    s0.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-a",
            "metric_event": {
                "metric_value": 0.07,
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    s1.mark_run_status("finished")
    s1.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W01:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-b",
            "metric_event": {
                "metric_value": 0.06,
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    state = LHRStateMachineStore.aggregate_logs(
        output_log_dir=out,
        input_log_dirs=[w0, w1],
        worker_count=2,
        run_status="finished",
        ledger_filename="run_results.md",
    )
    assert state["stage_count"] == 2
    assert state["global_best"]["candidate_id"] == "W01:S01"
    assert state["workers"]["W00"]["stage_count"] == 1
    assert state["workers"]["W01"]["stage_count"] == 1
    assert (out / "lhr_events.jsonl").is_file()
    assert (out / "lhr_state.json").is_file()


def test_lhr_state_machine_aggregate_global_best_ignores_unverified_stage_event(tmp_path: Path) -> None:
    out = tmp_path / "logs"
    w0 = tmp_path / "w00" / "logs"
    store = LHRStateMachineStore(log_dir=w0, worker_id="W00", worker_index=0, worker_count=1)
    store.mark_run_status("finished")
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-raw",
            "metric_event": {
                "metric_value": 0.01,
                "metric_name": "solver_score",
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": False,
                "selection_eligible": False,
                "metric_validity": "high",
            },
        },
    )
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S02",
        status="succeeded",
        payload={
            "stage_id": "S02",
            "snapshot_id": "S02-valid",
            "metric_event": {
                "metric_value": 0.08,
                "metric_name": "solver_score",
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "medium",
                "evaluator_backend": "artifact_command",
                "evaluator_status": "ok",
                "submission_status": "ready",
            },
        },
    )

    state = LHRStateMachineStore.aggregate_logs(
        output_log_dir=out,
        input_log_dirs=[w0],
        worker_count=1,
        run_status="finished",
        ledger_filename="run_results.md",
    )

    assert state["global_best"]["stage_id"] == "S02"
    assert state["global_best"]["metric_value"] == 0.08
    assert state["global_best"]["validity"] == "valid_comparable"

def test_lhr_state_machine_aggregate_backfills_stage_performance_csv(tmp_path: Path) -> None:
    out = tmp_path / "logs"
    out.mkdir(parents=True, exist_ok=True)
    w0 = tmp_path / "w00" / "logs"
    store = LHRStateMachineStore(log_dir=w0, worker_id="W00", worker_index=0, worker_count=1)
    store.mark_run_status("finished")
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S99",
        status="succeeded",
        payload={
            "stage_id": "S99",
            "snapshot_id": "S99-raw",
            "metric_event": {
                "metric_value": 1.0,
                "metric_name": "Final Validation Score",
                "lower_is_better": False,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    with (out / "lhr_stage_performance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "worker_id",
                "stage_id",
                "metric_value",
                "metric_name",
                "lower_is_better",
                "validation_ok",
                "candidate_ready",
                "selection_eligible",
                "metric_validity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.80",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        )

    state = LHRStateMachineStore.aggregate_logs(
        output_log_dir=out,
        input_log_dirs=[w0],
        worker_count=1,
        run_status="finished",
        ledger_filename="run_results.md",
    )

    assert state["stage_count"] == 1
    assert state["workers"]["W00"]["stage_count"] == 1
    assert state["global_best"]["candidate_id"] == "W00:L01:S01"
    assert state["global_best"]["metric_value"] == 0.80
    assert state["global_best"]["score_source"] == "lhr_stage_performance.csv"


@pytest.mark.asyncio
async def test_lhr_multi_worker_live_aggregation_refreshes_until_cancelled() -> None:
    solver = object.__new__(LnrSolver)
    calls: list[tuple[int, str]] = []

    def fake_aggregate_worker_state(*, n_workers: int, run_status: str) -> dict[str, object]:
        calls.append((n_workers, run_status))
        return {"stage_count": len(calls)}

    solver._aggregate_worker_state = fake_aggregate_worker_state  # type: ignore[method-assign]
    task = asyncio.create_task(
        solver._aggregate_worker_state_periodically(
            n_workers=2,
            run_status="running",
            interval_sec=60.0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == [(2, "running")]


def test_lhr_resource_observer_ignores_short_light_jobs(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(state_machine=store, worker_id="W00", min_register_sec=600)
    job = observer.job_created(command="ls -la", inferred_class="readonly_cpu", gpu_ids=[], timeout_sec=30)
    assert job is None
    assert not (tmp_path / "lhr_events.jsonl").exists()


def test_lhr_resource_observer_promotes_long_job_with_filtered_signal(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        min_register_sec=5,
        check_interval_sec=2,
        stalled_stdout_sec=20,
        kill_enabled=True,
    )
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_cpu_candidate", gpu_ids=[], timeout_sec=1800)
    assert job
    assert not (tmp_path / "lhr_events.jsonl").exists()
    decision = observer.active_intervention_decision(
        job,
        elapsed_sec=6,
        stdout_age_sec=1,
        stdout_lines=10,
        stdout_bytes=200,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )
    assert decision["enabled"] is True
    assert decision["terminate"] is False
    observer.job_finished(job, status="success", elapsed_sec=7, returncode=0, reason=None)
    events = (tmp_path / "lhr_events.jsonl").read_text().splitlines()
    assert len(events) == 2
    state = __import__("json").loads((tmp_path / "lhr_state.json").read_text())
    assert state["resource_jobs"] == 1
    assert state["workers"]["W00"]["resource_jobs"] == 1
    assert "python3 solution.py" not in (tmp_path / "lhr_events.jsonl").read_text()


def test_lhr_resource_observer_can_request_stalled_termination(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        min_register_sec=0,
        stalled_stdout_sec=3,
        kill_enabled=True,
        kill_mode="auto",
        review_state_enabled=False,
    )
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_cpu_candidate", gpu_ids=[], timeout_sec=1800)
    decision = observer.active_intervention_decision(
        job,
        elapsed_sec=10,
        stdout_age_sec=4,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="",
    )
    assert decision["terminate"] is True

def test_lhr_resource_observer_does_not_promote_static_gpu_class_without_gpu_id(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(state_machine=store, worker_id="W00", min_register_sec=600)
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_gpu_candidate", gpu_ids=[], timeout_sec=1800)
    assert job
    assert not (tmp_path / "lhr_events.jsonl").exists()

def test_lhr_resource_observer_does_not_promote_gpu_env_allocation_alone(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(state_machine=store, worker_id="W00", min_register_sec=600)
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_gpu_candidate", gpu_ids=["0"], timeout_sec=1800)
    assert job
    assert not (tmp_path / "lhr_events.jsonl").exists()

def test_lhr_resource_observer_monitor_agent_shadow_event_is_filtered(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        min_register_sec=0,
        stalled_stdout_sec=99,
        monitor_agent_mode="shadow",
        monitor_agent_min_interval_sec=1,
    )
    job = observer.job_created(command="python3 solution.py --secret /home/user/path", inferred_class="heavy_cpu_candidate", gpu_ids=[], timeout_sec=1800)
    observer.active_intervention_decision(
        job,
        elapsed_sec=2,
        stdout_age_sec=0,
        stdout_lines=3,
        stdout_bytes=120,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )
    text = (tmp_path / "lhr_events.jsonl").read_text()
    assert "resource_monitor_agent_shadow" in text
    assert "python3 solution.py" not in text
    assert "/home/user/path" not in text
    state = __import__("json").loads((tmp_path / "lhr_state.json").read_text())
    assert state["resource_monitor_agent_events"] == 1


def _lhr_resource_observer_with_gpu_queue(tmp_path: Path, worker_id: str) -> LHRResourceObserver:
    store = LHRStateMachineStore(log_dir=tmp_path / worker_id / "logs", worker_id=worker_id)
    return LHRResourceObserver(
        state_machine=store,
        worker_id=worker_id,
        min_register_sec=600,
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_queue_max_wait_sec=2,
        gpu_queue_heartbeat_sec=1,
        gpu_max_heavy_per_gpu=1,
        gpu_lease_ttl_sec=600,
    )


def test_lhr_gpu_queue_blocks_same_gpu_heavy_job(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job1
    acquired = first.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["enabled"] is True
    assert acquired["acquired"] is True

    job2 = second.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job2
    blocked = second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert blocked["enabled"] is True
    assert blocked["acquired"] is False
    assert blocked["reason"] == "gpu_slot_unavailable"

    first.job_finished(job1, status="success", elapsed_sec=5, returncode=0)
    acquired_after_release = second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired_after_release["enabled"] is True
    assert acquired_after_release["acquired"] is True

    first_events = (tmp_path / "W00" / "logs" / "lhr_events.jsonl").read_text()
    second_events = (tmp_path / "W01" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_gpu_lease_acquired" in first_events
    assert "resource_gpu_lease_released" in first_events
    assert "resource_gpu_queue_wait_started" in second_events
    first_state = json.loads((tmp_path / "W00" / "logs" / "lhr_state.json").read_text())
    second_state = json.loads((tmp_path / "W01" / "logs" / "lhr_state.json").read_text())
    assert first_state["resource_requests"] == 1
    assert first_state["resource_gpu_lease_acquired"] == 1
    assert first_state["resource_gpu_lease_released"] == 1
    assert first_state["resource_active_leases"] == []
    assert second_state["resource_requests"] == 1
    assert second_state["resource_gpu_queue_waits"] == 1
    assert len(second_state["resource_active_leases"]) == 1
    assert second_state["resource_active_leases"][0]["job_id"] == job2
    assert second_state["resource_pending_gpu_jobs"] == []
    aggregate = LHRStateMachineStore.aggregate_logs(
        output_log_dir=tmp_path / "task_logs",
        input_log_dirs=[tmp_path / "W00" / "logs", tmp_path / "W01" / "logs"],
        worker_count=2,
        run_status="running",
        ledger_filename=".run_results.md",
    )
    assert aggregate["resource_gpu_lease_acquired"] == 2
    assert aggregate["resource_gpu_lease_released"] == 1
    assert aggregate["resource_requests"] == 2
    assert aggregate["resource_gpu_queue_waits"] == 1
    assert len(aggregate["resource_active_leases"]) == 1
    assert aggregate["resource_active_leases"][0]["worker_id"] == "W01"
    assert aggregate["resource_pending_gpu_jobs"] == []
    assert [event["event"] for event in aggregate["resource_last_events"]][-1] == "resource_gpu_lease_acquired"



def test_lhr_gpu_tt_light_jobs_share_same_gpu_slot(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=900,
    )
    assert job1
    acquired = first.queue_try_acquire(job1, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert acquired["enabled"] is True
    assert acquired["acquired"] is True
    assert acquired["slot_weight"] == 0.25

    job2 = second.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=900,
    )
    assert job2
    acquired2 = second.queue_try_acquire(job2, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert acquired2["enabled"] is True
    assert acquired2["acquired"] is True
    state = json.loads((tmp_path / "W01" / "logs" / "lhr_state.json").read_text())
    assert state["resource_active_leases"][0]["resource_class"] == RESOURCE_GPU_TT_LIGHT


def test_lhr_gpu_tt_light_does_not_share_with_train_by_default(tmp_path: Path) -> None:
    train = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    tt = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    train_job = train.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert train_job
    assert train.queue_try_acquire(train_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True

    tt_job = tt.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=900,
    )
    assert tt_job
    blocked = tt.queue_try_acquire(tt_job, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert blocked["enabled"] is True
    assert blocked["acquired"] is False
    assert blocked["reason"] == "gpu_slot_unavailable"
    details = blocked["details"]["blocker_details"]["0"]
    assert "incompatible_active_class" in details["reasons"]


def test_lhr_gpu_queue_timeout_hard_gate_blocks_later_train(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job1
    assert first.queue_try_acquire(job1, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    job2 = second.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job2
    assert second.queue_try_acquire(job2, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is False
    second.queue_timeout(job2, elapsed_sec=2.1, reason="gpu_train_queue_wait_exceeded_2s")

    later = second.job_created(
        command="torchrun --nproc_per_node 1 train_again.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert later
    decision = second.resource_preflight_decision(
        later,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )
    assert decision["allowed"] is False
    assert "RESOURCE_FEEDBACK" in decision["feedback"]
    assert decision["reason"] == "block_train_after_gpu_pressure"
    assert "scope=per_gpu" in decision["feedback"]
    text = (tmp_path / "W01" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_policy_gate" in text


def test_lhr_gpu_queue_noops_for_non_gpu_heavy_job(tmp_path: Path) -> None:
    observer = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    (tmp_path / "train.py").write_text("print(42)\n", encoding="utf-8")
    job = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
        workspace_dir=tmp_path,
        classifier_reason="train_entrypoint",
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert decision == {"enabled": False, "acquired": True}


def test_lhr_gpu_lease_mode_assigns_one_visible_gpu(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=[],
        gpu_default_request=1,
        gpu_max_request=1,
        gpu_assignment="lease",
    )
    job = observer.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0", "1"],
        timeout_sec=1800,
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0", "1"])
    assert decision["enabled"] is True
    assert decision["acquired"] is True
    assert decision["gpu_ids"] == ["0"]
    env = observer.lease_env_updates(job)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL"] == "0"


def test_lhr_gpu_lease_mode_uses_configured_pool_without_visible_env(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["2", "3"],
        gpu_default_request=1,
        gpu_max_request=1,
        gpu_assignment="lease",
    )
    job = observer.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=1800,
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert decision["acquired"] is True
    assert decision["gpu_ids"] in (["2"], ["3"])
    env = observer.lease_env_updates(job)
    assert env["CUDA_VISIBLE_DEVICES"] == decision["gpu_ids"][0]
    assert env["SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL"] == decision["gpu_ids"][0]
    assert env["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] == "2,3"


def test_lhr_gpu_lease_mode_honors_torchrun_multi_gpu_request(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["0", "1", "2"],
        gpu_default_request=1,
        gpu_max_request=4,
        gpu_assignment="lease",
    )
    job = observer.job_created(
        command="torchrun --nproc_per_node 2 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=1800,
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert decision["acquired"] is True
    assert decision["gpu_ids"] == ["0", "1"]
    env = observer.lease_env_updates(job)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL"] == "0,1"


def test_lhr_source_hint_detects_cuda_entrypoint_without_leaking_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        "import torch\nmodel.to('cuda')\n",
        encoding="utf-8",
    )
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["0"],
        gpu_assignment="lease",
        gpu_source_hint_enabled=True,
        gpu_source_hint_mode="observe",
    )
    job = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=1800,
        workspace_dir=workspace,
        classifier_reason="heavy_keyword",
    )
    assert job
    text = (tmp_path / "W00" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_source_hint_detected" in text
    assert "train.py" in text
    assert str(workspace) not in text
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert decision["acquired"] is True


def test_lhr_bash_tool_applies_leased_cuda_env_from_source_hint(tmp_path: Path) -> None:
    from scienceflow.core.tools.bash_tool import BashTool

    (tmp_path / "train.py").write_text(
        "import os\n# torch.cuda source hint\nprint(os.environ.get('CUDA_VISIBLE_DEVICES', ''))\n",
        encoding="utf-8",
    )
    store = LHRStateMachineStore(log_dir=tmp_path / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["2"],
        gpu_assignment="lease",
        gpu_source_hint_enabled=True,
        gpu_source_hint_mode="observe",
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5,
        bash_timeout_slow_sec=5,
        resource_observer=observer,
    )

    async def _run() -> None:
        result = await tool.execute("python3 train.py")
        assert result.error is None
        assert "\n2" in (result.output or "")

    asyncio.run(_run())
    text = (tmp_path / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_source_hint_detected" in text
    assert "resource_gpu_lease_acquired" in text
    assert "resource_gpu_lease_released" in text


def test_lhr_gpu_queue_timeout_records_event(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job1
    assert first.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    job2 = second.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job2
    assert second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is False
    second.queue_timeout(job2, elapsed_sec=2.1, reason="gpu_queue_wait_exceeded_2s")
    text = (tmp_path / "W01" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_gpu_queue_timeout" in text
    assert "python3 train.py" not in text
    state = json.loads((tmp_path / "W01" / "logs" / "lhr_state.json").read_text())
    assert state["resource_pending_gpu_jobs"] == []


def test_lhr_gpu_util_sample_is_recorded_without_command_text(tmp_path: Path) -> None:
    observer = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    assert observer.resource_runtime is not None

    def fake_sample(*, gpu_ids: list[str]) -> dict[str, object]:
        return {
            "available": True,
            "reason": "",
            "gpus": [
                {
                    "gpu_id": gpu_ids[0],
                    "utilization_gpu_pct": 42.0,
                    "memory_used_mb": 1024.0,
                    "memory_total_mb": 8192.0,
                }
            ],
        }

    observer.resource_runtime.sample_gpu_util = fake_sample  # type: ignore[method-assign]
    job = observer.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job
    assert observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    observer.active_intervention_decision(
        job,
        elapsed_sec=12,
        stdout_age_sec=1,
        stdout_lines=5,
        stdout_bytes=128,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )
    state = json.loads((tmp_path / "W00" / "logs" / "lhr_state.json").read_text())
    assert state["resource_gpu_util_samples"] == 1
    assert state["resource_last_events"][-1]["event"] == "resource_gpu_util_sampled"
    assert state["resource_last_events"][-1]["gpu_util"][0]["utilization_gpu_pct"] == 42.0
    assert "torchrun --nproc_per_node" not in (tmp_path / "W00" / "logs" / "lhr_events.jsonl").read_text()



def _lhr_test_ledger() -> str:
    return "\n".join([
        "### S01",
        "metric: 0.07",
        "lower_is_better: true",
        "BRIEF: baseline",
        "WHY: start",
        "",
        "### S02",
        "metric: 0.08",
        "lower_is_better: true",
        "BRIEF: worse branch",
        "WHY: test route",
        "",
    ])

def _minimal_lhr_solver(tmp_path: Path) -> LnrSolver:
    solver = object.__new__(LnrSolver)
    solver.ledger_filename = "run_results.md"
    solver.ledger_path = tmp_path / "run_results.md"
    solver.log_dir = tmp_path / "logs"
    solver.root_dir = tmp_path
    solver.task_root_dir = tmp_path
    solver.workspace_dir = tmp_path / "workspace"
    solver.worker_id = ""
    solver.worker_index = 0
    solver.current_lineage_no = 1
    solver.current_lineage_id = "L01"
    solver.pending_stage_commit_transaction = None
    solver.state_machine = LHRStateMachineStore(log_dir=solver.log_dir, worker_id="W00")
    solver.ledger_path.write_text(_lhr_test_ledger(), encoding="utf-8")
    return solver


def test_lhr_protected_eda_prefix_stops_before_s01_stage_commit(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        preserve_prefix_and_eda=True,
        protected_eda_mode="facts",
        protected_eda_facts_max_chars=6000,
        protected_eda_warn_chars=50_000,
    )
    messages = [
        Message.user_message("FIRST USER QUERY"),
        Message.assistant_message("inspect the dataset"),
        Message.tool_message("train shape is (100, 8)", "bash", "eda-1"),
    ]

    class _MemoryContext:
        end_index = None

        def replace_protected_raw_prefix_with_summary(
            self,
            end_index: int,
            summary: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.end_index = end_index
            messages[:] = [Message.user_message(summary)] + messages[end_index:]
            return {
                "end_index": 1,
                "message_count": 1,
                "chars": len(summary),
                "original_chars": 100,
                "original_message_count": end_index,
            }

    ctx = _MemoryContext()
    agent = SimpleNamespace(_memory_ctx=ctx)
    solver._agent_memory_messages = lambda _agent: list(messages)
    solver._count_memory_records = lambda: len(messages)
    solver._capture_s01_eda_prefix_end(stage_id="S01")
    messages.append(
        Message.assistant_message(
            "STAGE_COMMIT_BEGIN\nstage_id: S01\nmetric: 0.2\nSTAGE_COMMIT_END"
        )
    )

    info = solver._mark_protected_eda_prefix(agent, stage_id="S01")

    assert ctx.end_index == 3
    assert info["boundary_source"] == "pre_stage_commit"
    assert info["protected_end_index"] == 1
    assert solver._s01_eda_prefix_end_index == 1
    assert len(messages) == 2
    assert "Fixed EDA facts" in str(messages[0].content)
    assert "STAGE_COMMIT_BEGIN" in str(messages[1].content)


def test_lhr_protected_eda_prefix_resume_uses_pre_commit_boundary(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        preserve_prefix_and_eda=True,
        protected_eda_warn_chars=50_000,
    )
    solver._s01_eda_prefix_end_index = None
    solver.stage_snapshots = {
        "S01": SimpleNamespace(
            memory_cut=2,
            source_event={"protected_eda_end_index": 1},
        )
    }

    class _MemoryContext:
        end_index = None

        def set_protected_raw_prefix(
            self,
            end_index: int,
            **_kwargs: object,
        ) -> None:
            self.end_index = end_index

    ctx = _MemoryContext()
    solver._restore_protected_eda_prefix_marker(SimpleNamespace(_memory_ctx=ctx))

    assert ctx.end_index == 1
    assert solver._s01_eda_prefix_end_index == 1


def test_lhr_agent_eda_summary_is_opt_in_and_replaces_only_eda_prefix(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Rank scientific candidates."
        solver.lhr = SimpleNamespace(
            preserve_prefix_and_eda=True,
            protected_eda_mode="agent",
            protected_eda_facts_max_chars=6000,
            protected_eda_summary_max_chars=8000,
            protected_eda_warn_chars=50_000,
            stage_commit_llm_timeout_sec=30,
        )
        solver.stage_tokens_in = 0
        solver.stage_tokens_out = 0
        solver.stage_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver._s01_agent_eda_summary = ""
        messages = [
            Message.user_message("Rank scientific candidates."),
            Message.assistant_message("I will inspect source and candidate distributions."),
            Message.tool_message(
                "source shape is (100, 8); candidate shape is (20, 8)",
                "bash",
                "eda-1",
            ),
        ]

        class _LLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 50
            _last_call_input_cached_tokens = 0

            async def ask(self, **kwargs):
                assert kwargs["stream"] is False
                assert "AUTHORITATIVE S01 STAGE CARD" in kwargs["messages"][0].content
                return "## Data contract\nRank 20 candidates.\n\n## Observed data facts\nSource has 100 rows."

        class _MemoryContext:
            summary = ""
            end_index = None

            def replace_protected_raw_prefix_with_summary(self, end_index, summary, **_kwargs):
                self.end_index = end_index
                self.summary = summary
                return {
                    "end_index": 1,
                    "message_count": 1,
                    "chars": len(summary),
                    "original_chars": 100,
                    "original_message_count": 2,
                }

        ctx = _MemoryContext()
        agent = SimpleNamespace(
            llm=_LLM(),
            _memory_ctx=ctx,
            _record_llm_call=lambda *_args, **_kwargs: None,
        )
        solver._agent_memory_messages = lambda _agent: list(messages)
        solver._count_memory_records = lambda: len(messages)
        solver._capture_s01_eda_prefix_end(stage_id="S01")
        messages.append(
            Message.assistant_message(
                "STAGE_COMMIT_BEGIN\nstage_id: S01\nSTAGE_COMMIT_END",
            ),
        )

        await solver._prepare_protected_eda_agent_summary(agent, stage_id="S01")
        info = solver._mark_protected_eda_prefix(agent, stage_id="S01")

        assert ctx.end_index == 3
        assert "## Data contract" in ctx.summary
        assert info["mode"] == "agent"
        assert solver.stage_llm_calls == 1

    asyncio.run(_run())


def test_lhr_agent_eda_summary_falls_back_to_facts(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        preserve_prefix_and_eda=True,
        protected_eda_mode="agent",
        protected_eda_facts_max_chars=6000,
        protected_eda_summary_max_chars=8000,
        protected_eda_warn_chars=50_000,
    )
    solver._s01_agent_eda_summary = ""
    messages = [
        Message.user_message("FIRST USER QUERY"),
        Message.tool_message("train shape is (100, 8)", "bash", "eda-1"),
    ]

    class _MemoryContext:
        summary = ""

        def replace_protected_raw_prefix_with_summary(self, end_index, summary, **_kwargs):
            self.summary = summary
            return {
                "end_index": 1,
                "message_count": 1,
                "chars": len(summary),
                "original_chars": 100,
                "original_message_count": 1,
            }

    ctx = _MemoryContext()
    solver._agent_memory_messages = lambda _agent: list(messages)
    solver._count_memory_records = lambda: len(messages)
    solver._s01_eda_prefix_end_index = len(messages)

    info = solver._mark_protected_eda_prefix(
        SimpleNamespace(_memory_ctx=ctx),
        stage_id="S01",
    )

    assert info["mode"] == "facts_fallback"
    assert "train shape is (100, 8)" in ctx.summary


def test_lhr_agent_eda_summary_provider_error_uses_facts_fallback(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Rank candidates."
        solver.lhr = SimpleNamespace(
            protected_eda_mode="agent",
            protected_eda_summary_max_chars=8000,
            stage_commit_llm_timeout_sec=30,
        )
        solver._s01_eda_prefix_end_index = 2

        class _LLM:
            async def ask(self, **_kwargs):
                raise RuntimeError("provider unavailable")

        agent = SimpleNamespace(
            llm=_LLM(),
            _record_llm_call=lambda *_args, **_kwargs: None,
        )
        events = []
        solver._jsonl = lambda _name, record: events.append(record)
        solver._agent_memory_messages = lambda _agent: [
            Message.user_message("task"),
            Message.tool_message("shape=(100, 8)", "bash", "eda"),
        ]

        await solver._prepare_protected_eda_agent_summary(agent, stage_id="S01")

        assert solver._s01_agent_eda_summary == ""
        assert events[-1]["fallback"] == "facts"

    asyncio.run(_run())


def test_lhr_agent_eda_summary_propagates_cancellation(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Rank candidates."
        solver.lhr = SimpleNamespace(
            protected_eda_mode="agent",
            protected_eda_summary_max_chars=8000,
            stage_commit_llm_timeout_sec=30,
        )
        solver._s01_eda_prefix_end_index = 1

        class _LLM:
            async def ask(self, **_kwargs):
                raise asyncio.CancelledError()

        agent = SimpleNamespace(llm=_LLM())
        solver._agent_memory_messages = lambda _agent: [Message.user_message("task")]

        with pytest.raises(asyncio.CancelledError):
            await solver._prepare_protected_eda_agent_summary(agent, stage_id="S01")

    asyncio.run(_run())


def test_lhr_process_resume_rehydrates_stage_snapshots(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.ledger_path.write_text(
        "### S01\nmetric: 0.5\nlower_is_better: false\nBRIEF: accepted\nWHY: evaluator accepted\n",
        encoding="utf-8",
    )
    store = SnapshotStore(
        root_dir=tmp_path,
        workspace_dir=solver.workspace_dir,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        control_log_dir=solver.log_dir,
        metadata_dirname="logs",
        strict_layout=True,
        workspace_snapshot_enabled=False,
    )
    (solver.workspace_dir / "solution.py").write_text("print(1)\n", encoding="utf-8")
    snapshot = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="score",
        lower_is_better=False,
        memory_cut=2,
        source_event={
            "solution_sha": "solution-one",
            "artifact_sha": "artifact-one",
            "gate_accepted": True,
            "_gate_evaluated": True,
            "lineage_id": "L03",
        },
        node_uid="W00:L03:S01",
        lineage_id="L03",
    )
    solver.snapshot_store = store
    solver.stage_snapshots = {"S01": snapshot}
    solver._write_stage_map()

    solver.stage_snapshots = {}
    solver.last_captured_solution_sha = ""
    solver.last_captured_run_signature = ""
    solver.current_lineage_no = 1
    solver.current_lineage_id = "L01"
    solver._load_existing_stage_snapshots()

    assert list(solver.stage_snapshots) == ["S01"]
    assert solver.stage_snapshots["S01"].snapshot_id == snapshot.snapshot_id
    assert solver.archived_stage_snapshots["W00:L03:S01"].snapshot_id == snapshot.snapshot_id
    assert solver.last_captured_solution_sha == "solution-one"
    assert solver.current_lineage_id == "L03"
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    event = [row for row in rows if row["event"] == "stage_snapshots_rehydrated"][-1]
    assert event["payload"]["loaded_stages"] == ["S01"]


def test_lhr_pending_estra_restores_exact_archived_node_uid(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = solver.workspace_dir / ".agent_memory"
        solver.memory_dir.mkdir(parents=True)
        old = StageSnapshot(
            "S01",
            "old-snapshot",
            tmp_path / "snapshots" / "old",
            0.5,
            "score",
            False,
            1,
            {},
            node_uid="W00:L01:S01",
            lineage_id="L01",
        )
        current = StageSnapshot(
            "S01",
            "current-snapshot",
            tmp_path / "snapshots" / "current",
            0.6,
            "score",
            False,
            2,
            {},
            node_uid="W00:L02:S01",
            lineage_id="L02",
        )
        restored: list[StageSnapshot] = []
        archive_dir = tmp_path / "terminal-archive"
        archive_dir.mkdir()
        solver.snapshot_store = SimpleNamespace(restore=lambda snap: restored.append(snap) or archive_dir)
        solver.stage_snapshots = {"S01": current}
        solver.archived_stage_snapshots = {
            "W00:L01:S01": old,
            "W00:L02:S01": current,
        }
        solver.pending_estra = {
            "action": "switch_stage",
            "target_stage": "S01",
            "target_node_uid": "W00:L01:S01",
            "tail_summary": "discarded tail",
            "state_packet": "state",
        }
        solver.current_lineage_no = 2
        solver.current_lineage_id = "L02"
        solver._close_lnr_interaction_loggers = lambda: None
        solver._prepare_dataset_symlink = lambda: None
        solver._prune_workspace_control_artifacts = lambda: None
        solver._prune_active_stage_snapshots_to_ledger = lambda: None
        solver._rebuild_memory_after_estra = lambda **_kwargs: (True, 3)
        solver._append_traj_summary = lambda **_kwargs: None
        solver._reset_interaction_stage_files = lambda: None
        monkeypatch.setattr(lnr_solver_module, "append_archived_trajectory_summary", lambda *_args, **_kwargs: None)

        await solver._restore_pending_estra()

        assert restored == [old]
        assert solver.stage_snapshots["S01"] is old
        assert solver.current_restored_from_node_uid == "W00:L01:S01"
        stage_map = json.loads((solver.log_dir / "lhr_stage_map.json").read_text(encoding="utf-8"))
        assert stage_map["stages"]["S01"]["node_uid"] == "W00:L01:S01"
        assert set(stage_map["archive"]) == {"W00:L01:S01", "W00:L02:S01"}

    asyncio.run(_run())


def test_lhr_failed_estra_unfold_keeps_active_lineage_and_records_failure(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        snapshot = StageSnapshot(
            "S01",
            "snapshot-one",
            tmp_path / "snapshots" / "S01",
            0.5,
            "score",
            False,
            1,
            {},
            node_uid="W00:L01:S01",
            lineage_id="L01",
        )
        solver.stage_snapshots = {"S01": snapshot}
        solver.archived_stage_snapshots = {"W00:L01:S01": snapshot}
        solver.pending_estra = {
            "action": "switch_stage",
            "target_stage": "S01",
            "target_node_uid": "W00:L01:S01",
            "tail_summary": "discarded tail",
            "state_packet": "state",
        }
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.snapshot_store = SimpleNamespace(
            restore=lambda _snap: (_ for _ in ()).throw(OSError("injected unfold failure")),
        )
        solver._close_lnr_interaction_loggers = lambda: None

        await solver._restore_pending_estra()

        assert solver.current_lineage_no == 1
        assert solver.current_lineage_id == "L01"
        assert solver.stage_snapshots == {"S01": snapshot}
        assert solver.pending_estra is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        failed = [row for row in rows if row["event"] == "estra_failed"][-1]
        assert failed["payload"]["reason"] == "unfold_transaction_failed"
        assert failed["payload"]["error_type"] == "OSError"

    asyncio.run(_run())


def test_lhr_failed_post_restore_step_rolls_back_terminal_archive(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        snapshot = StageSnapshot(
            "S01",
            "snapshot-one",
            tmp_path / "snapshots" / "S01",
            0.5,
            "score",
            False,
            1,
            {},
            node_uid="W00:L01:S01",
            lineage_id="L01",
        )
        solver.stage_snapshots = {"S01": snapshot}
        solver.archived_stage_snapshots = {"W00:L01:S01": snapshot}
        solver.pending_estra = {
            "action": "switch_stage",
            "target_stage": "S01",
            "target_node_uid": "W00:L01:S01",
        }
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        archive = tmp_path / "terminal-archive"
        restored_archives: list[Path] = []
        solver.snapshot_store = SimpleNamespace(
            restore=lambda _snap: archive,
            restore_terminal_archive=lambda path: restored_archives.append(Path(path)),
        )
        solver._close_lnr_interaction_loggers = lambda: None
        solver._prepare_dataset_symlink = lambda: (_ for _ in ()).throw(
            OSError("injected post-restore failure")
        )

        await solver._restore_pending_estra()

        assert restored_archives == [archive]
        assert solver.current_lineage_no == 1
        assert solver.current_lineage_id == "L01"
        assert solver.stage_snapshots == {"S01": snapshot}
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        failed = [row for row in rows if row["event"] == "estra_failed"][-1]
        assert failed["payload"]["rollback_attempted"] is True
        assert failed["payload"]["rollback_succeeded"] is True

    asyncio.run(_run())


def test_lhr_resume_rolls_back_ledger_when_stage_snapshot_is_missing(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    ledger_before = solver.ledger_path.read_text(encoding="utf-8")
    ledger_after = ledger_before + "### S03\nmetric: 0.09\nlower_is_better: true\nBRIEF: pending\nWHY: pending\n"
    metric_event = {"artifact_sha": "new-artifact", "gate_accepted": True}
    solver.snapshot_store = SimpleNamespace(discover=lambda: {})

    solver._prepare_stage_commit_transaction(
        stage_id="S03",
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    solver.ledger_path.write_text(ledger_after, encoding="utf-8")
    solver.pending_stage_commit_transaction = None  # simulate process restart
    solver._recover_stage_commit_transactions()

    assert solver.ledger_path.read_text(encoding="utf-8") == ledger_before
    manifest = json.loads((solver.log_dir / "stage_transactions" / "L01-S03.json").read_text())
    assert manifest["status"] == "rolled_back"
    assert manifest["rollback_reason"] == "resume_missing_snapshot"


def test_lhr_resume_completes_only_the_matching_stage_transaction_snapshot(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    ledger_before = solver.ledger_path.read_text(encoding="utf-8")
    ledger_after = ledger_before + "### S03\nmetric: 0.09\nlower_is_better: true\nBRIEF: pending\nWHY: pending\n"
    metric_event = {"artifact_sha": "new-artifact", "gate_accepted": True}
    solver._prepare_stage_commit_transaction(
        stage_id="S03",
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    snapshot = StageSnapshot(
        stage_id="S03",
        snapshot_id="snapshot-three",
        snapshot_path=tmp_path / "snapshots" / "S03",
        metric_value=0.09,
        metric_name="score",
        lower_is_better=True,
        memory_cut=3,
        source_event=dict(metric_event),
        node_uid="W00:L01:S03",
        lineage_id="L01",
    )
    solver.snapshot_store = SimpleNamespace(discover=lambda: {"S03": snapshot})
    solver.ledger_path.write_text(ledger_after, encoding="utf-8")
    solver.pending_stage_commit_transaction = None  # simulate process restart
    solver._recover_stage_commit_transactions()

    assert solver.ledger_path.read_text(encoding="utf-8") == ledger_after
    manifest = json.loads((solver.log_dir / "stage_transactions" / "L01-S03.json").read_text())
    assert manifest["status"] == "committed"
    assert manifest["recovered_after_restart"] is True


def test_lhr_resume_does_not_match_an_older_snapshot_with_same_stage_id(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    ledger_before = solver.ledger_path.read_text(encoding="utf-8")
    ledger_after = ledger_before + "### S03\nmetric: 0.09\nlower_is_better: true\nBRIEF: pending\nWHY: pending\n"
    metric_event = {"artifact_sha": "same-artifact", "gate_accepted": True}
    solver._prepare_stage_commit_transaction(
        stage_id="S03",
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    old_snapshot = StageSnapshot(
        stage_id="S03",
        snapshot_id="old-snapshot",
        snapshot_path=tmp_path / "snapshots" / "old-S03",
        metric_value=0.09,
        metric_name="score",
        lower_is_better=True,
        memory_cut=3,
        source_event={"artifact_sha": "same-artifact"},
        node_uid="W00:L01:S03",
        lineage_id="L01",
    )
    solver.snapshot_store = SimpleNamespace(discover=lambda: {"S03": old_snapshot})
    solver.ledger_path.write_text(ledger_after, encoding="utf-8")
    solver.pending_stage_commit_transaction = None
    solver._recover_stage_commit_transactions()

    assert solver.ledger_path.read_text(encoding="utf-8") == ledger_before
    manifest = json.loads((solver.log_dir / "stage_transactions" / "L01-S03.json").read_text())
    assert manifest["status"] == "rolled_back"


def test_lhr_primary_gate_fails_closed_when_evaluator_emits_no_outcome(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.cfg = SimpleNamespace(
        exp_id="test-task",
        evaluator=SimpleNamespace(
            task_profile="mlebench",
            backend="task_package",
            stage_source_mode="primary",
            event_log="evaluator_events.jsonl",
        ),
    )
    solver.deadline = float("inf")
    solver.gate_service = SimpleNamespace(evaluate=lambda _request: [])

    facts = solver._record_evaluator_stage_events(
        stage_id="S03",
        metric_event={
            "metric_value": 0.8,
            "metric_name": "score",
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
        },
    )

    assert facts["metric_value"] == 0.8
    assert facts["validation_ok"] is False
    assert facts["candidate_ready"] is False
    assert facts["gate_accepted"] is False
    assert facts["gate_reason_code"] == "evaluator_no_outcome"
    assert facts["_gate_evaluated"] is True


def test_lhr_primary_gate_accepts_complete_result_without_submission(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.cfg = SimpleNamespace(
        exp_id="nomad2018-predict-transparent-conductors",
        evaluator=SimpleNamespace(
            enabled=True,
            task_profile="mlebench",
            backend="task_package",
            stage_source_mode="primary",
            event_log="evaluator_events.jsonl",
            metric=SimpleNamespace(selection_requires_direction=True),
        ),
    )
    solver.deadline = float("inf")
    solver.gate_service = GateService.default()

    facts = solver._record_evaluator_stage_events(
        stage_id="S03",
        metric_event={
            "metric_value": 0.8,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "val_score_type": "holdout",
            "execution_scale": "direct_full",
            "bash_cmd": "python3 solution.py",
            "solution_sha": "complete-source",
            "submission_status": "missing_submission",
        },
    )

    assert facts["gate_accepted"] is True
    assert facts["gate_reason_code"] == "eligible"
    assert facts["evaluator_backend"] == "result_signal"
    assert facts["candidate_ready"] is True
    assert facts["submission_status"] == "missing_submission"
    assert facts["artifact_sha"] == ""


def test_lhr_estra_memory_compact_records_context_preview(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_compact_enabled=True)
    solver.cfg = SimpleNamespace(max_messages=100)
    solver.memory_dir = tmp_path / "memory"
    snap_dir = tmp_path / "snapshots" / "S01-test"
    agent_dir = snap_dir / ".memory" / "ScienceAgent"
    agent_dir.mkdir(parents=True)
    records = [
        solver._memory_user_record("FIRST USER QUERY"),
        solver._memory_user_record("[LHR protected EDA facts] train shape is stable"),
    ]
    for name in ("short_term.json", "long_term.jsonl"):
        (agent_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="S01-test",
            snapshot_path=snap_dir,
            metric_value=0.1,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=2,
            source_event={},
        )
    }

    ok, n_records = solver._rebuild_memory_after_estra(target_stage="S01", summary="S02 failed; avoid noisy features")

    assert ok
    assert n_records == 3
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    ctx = [row for row in rows if row["event"] == "estra_context_prepared"][-1]
    assert ctx["payload"]["target_stage"] == "S01"
    assert ctx["payload"]["base_stage"] == "S01"
    assert ctx["payload"]["memory_records"] == 3
    assert "run_results.md" not in ctx["payload"]["resume_prompt_excerpt"]
    assert ".run_results.md" not in ctx["payload"]["resume_prompt_excerpt"]
    assert "S02 failed" in ctx["payload"]["tail_summary_excerpt"]
    assert "estra resume" in " ".join(ctx["payload"]["context_shape"])


def test_lhr_estra_memory_compact_uses_first_completed_stage_as_base(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.ledger_path.write_text(
        "### S02\nmetric: 0.10\nlower_is_better: true\nBRIEF: first completed\nWHY: no S01 metric\n",
        encoding="utf-8",
    )
    solver.lhr = SimpleNamespace(estra_compact_enabled=True)
    solver.cfg = SimpleNamespace(max_messages=100)
    solver.memory_dir = tmp_path / "memory"
    snap_dir = tmp_path / "snapshots" / "S02-test"
    agent_dir = snap_dir / ".memory" / "ScienceAgent"
    agent_dir.mkdir(parents=True)
    records = [solver._memory_user_record("FIRST COMPLETED STAGE MEMORY")]
    for name in ("short_term.json", "long_term.jsonl"):
        (agent_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    solver.stage_snapshots = {
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="S02-test",
            snapshot_path=snap_dir,
            metric_value=0.10,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=1,
            source_event={},
        )
    }

    ok, n_records = solver._rebuild_memory_after_estra(target_stage="S02", summary="S03 failed")

    assert ok
    assert n_records == 2
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    ctx = [row for row in rows if row["event"] == "estra_context_prepared"][-1]
    assert ctx["payload"]["base_stage"] == "S02"
    assert "S02 prefix-safe memory records" in ctx["payload"]["context_shape"]
    assert "original S02 base stage" in ctx["payload"]["resume_prompt_excerpt"]
    assert "original S01 base stage" not in ctx["payload"]["resume_prompt_excerpt"]


def test_lhr_keep_current_strict_context_limit_uses_minimal_prefix(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_compact_enabled=True)
    solver.cfg = SimpleNamespace(max_messages=240)
    solver.memory_dir = tmp_path / "memory"
    snap_dir = tmp_path / "snapshots" / "S01-test"
    agent_dir = snap_dir / ".memory" / "ScienceAgent"
    agent_dir.mkdir(parents=True)
    records = [
        {"message": {"role": "system", "content": "SYSTEM PREFIX"}, "role": "system"},
        solver._memory_user_record("FIRST TASK PROMPT"),
    ]
    for idx in range(80):
        records.append({"message": {"role": "assistant", "content": f"assistant noise {idx}"}, "role": "assistant"})
        records.append({"message": {"role": "tool", "content": f"tool noise {idx}"}, "role": "tool"})
    for name in ("short_term.json", "long_term.jsonl"):
        (agent_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="S01-test",
            snapshot_path=snap_dir,
            metric_value=0.1,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=len(records),
            source_event={},
        )
    }

    ok, n_records = solver._rebuild_memory_after_keep_current(
        terminal_stage="S02",
        summary="S02 compact summary",
        state_packet="LHR State Packet v2",
        strict_context_limit=True,
    )

    assert ok
    assert n_records == 3
    written = (solver.memory_dir / "ScienceAgent" / "short_term.json").read_text(encoding="utf-8")
    assert "SYSTEM PREFIX" in written
    assert "FIRST TASK PROMPT" in written
    assert "assistant noise" not in written
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    ctx = [row for row in rows if row["event"] == "estra_keep_current_context_prepared"][-1]
    assert ctx["payload"]["compact_strength"] == "strict_context_limit"
    assert "minimal task prefix records" in ctx["payload"]["context_shape"][0]


def test_lhr_context_compact_events_are_unified_and_stage_scoped(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.stage_snapshots = {"S01": object(), "S03": object()}

    solver._record_context_compact_event(
        phase="started",
        mode="inband",
        reason="context_limit",
        round=4,
        omitted_before=9,
        recent_keep=0,
    )
    solver._record_context_compact_event(
        phase="finished",
        mode="inband",
        status="ok",
        round=4,
        omitted_after=0,
        summary_chars=1234,
        tokens_in=100,
        tokens_out=10,
        tokens_cached=95,
    )

    assert not (solver.log_dir / "lhr_context_events.jsonl").exists()
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    events = [row for row in rows if row["event"].startswith("context_compact_")]
    assert [row["event"] for row in events] == [
        "context_compact_started",
        "context_compact_finished",
    ]
    assert events[0]["task_id"] == "repl_search:W00:S02"
    assert events[0]["payload"]["latest_stage"] == "S02"
    assert events[0]["payload"]["next_stage"] == "S03"
    assert events[0]["status"] == "running"
    assert events[1]["status"] == "succeeded"
    assert events[1]["payload"]["summary_chars"] == 1234


def test_lhr_ask_estra_records_text_only_trigger_source(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {"S01": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                return '{"action":"switch_stage","target_stage":"S01","reason":"best"}'

        class FakeAgent:
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="text_only")
        assert decision["action"] == "switch_stage"
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["latest_stage"] == "S02"
        assert route["payload"]["target_stage"] == "S01"
        assert route["payload"]["trigger_source"] == "text_only"
        assert route["payload"]["reason"] == "best"
        assert route["payload"]["action"] == "switch_stage"
        assert route["payload"]["decision_kind"] == "switch"
        assert route["payload"]["response_chars"] > 0
        assert route["task_id"] == "estra_decision:W00:S02"

    asyncio.run(_run())


def test_lhr_ask_estra_context_limit_keep_current_means_compact_continue(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {
            "S01": StageSnapshot("S01", "snap1", tmp_path / "s1", 0.07, "Final Validation Score", True, 1, {}, node_uid="W00:L01:S01", lineage_id="L01"),
            "S02": StageSnapshot("S02", "snap2", tmp_path / "s2", 0.08, "Final Validation Score", True, 2, {}, node_uid="W00:L01:S02", lineage_id="L01"),
        }
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                return '{"action":"keep_current","reason":"continue from terminal"}'

        class FakeAgent:
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="context_limit")
        assert decision["action"] == "keep_current"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["target_node_uid"] == "W00:L01:S02"
        assert route["payload"]["decision_kind"] == "continue"
        assert route["payload"]["candidate_node_uids"]["S01"] == "W00:L01:S01"
        assert route["payload"]["candidate_node_uids"]["S02"] == "W00:L01:S02"

    asyncio.run(_run())


def test_lhr_ask_estra_keep_but_redirect_compacts_with_diagnostics(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {
            "S01": StageSnapshot("S01", "snap1", tmp_path / "s1", 0.07, "Final Validation Score", True, 1, {}, node_uid="W00:L01:S01", lineage_id="L01"),
            "S02": StageSnapshot("S02", "snap2", tmp_path / "s2", 0.08, "Final Validation Score", True, 2, {}, node_uid="W00:L01:S02", lineage_id="L01"),
        }
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                return json.dumps(
                    {
                        "startpoint": "current_workspace",
                        "intent": "redirect",
                        "exploration_summary": "S01-S02 repeated local parameter changes.",
                        "bottleneck": "The route has not diagnosed why validation is flat.",
                        "evidence": "Recent stage cards show no material metric gain.",
                        "missing_evidence": None,
                        "is_route_flaw": False,
                        "is_execution_flaw": True,
                        "target_stage": None,
                        "decision_reason": "Keep workspace but redirect the next stage.",
                        "redirect_focus": "Diagnose validation and missing signal.",
                    }
                )

        class FakeAgent:
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="force_stage_capture")
        assert decision["action"] == "keep_but_redirect"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        assert "validation is flat" in decision["bottleneck"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["action"] == "keep_but_redirect"
        assert route["payload"]["decision_kind"] == "redirect"
        assert "validation is flat" in route["payload"]["bottleneck"]
        assert "Recent stage cards" in route["payload"]["evidence"]

    asyncio.run(_run())


def test_lhr_ask_estra_skips_llm_without_historical_switch_candidate(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {"S02": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            async def ask(self, **kwargs):
                raise AssertionError("estra model should not be called without a historical switch candidate")

            async def ask_tool_stream(self, **kwargs):
                raise AssertionError("estra model should not be called without a historical switch candidate")

        class FakeAgent:
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="text_only")
        assert decision["action"] == "keep_current"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        assert decision["reason"] == "no valid historical switch candidates"
        assert solver.estra_llm_calls == 0
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_no_valid_switch_candidate"][-1]
        payload = route["payload"]
        assert route["task_type"] == "estra_decision"
        assert payload["estra_action"] == "keep_current"
        assert payload["switch_candidates"] == []
        assert payload["candidates"] == ["S02"]

    asyncio.run(_run())


def test_lhr_estra_trigger_requires_historical_switch_candidate(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_enabled=True, estra_max_decisions=2, estra_trigger_stage_count=2)
    solver.estra_decisions = 0
    solver.last_estra_observation_key = ""
    solver.last_estra_stage_count = 0
    cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))

    assert not LnrSolver._estra_trigger_allowed(solver, cards=cards, candidates=["S02"])
    assert LnrSolver._estra_trigger_allowed(solver, cards=cards, candidates=["S01", "S02"])


def test_lhr_estra_max_decisions_is_deprecated_noop(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_enabled=True, estra_max_decisions=1, estra_trigger_stage_count=2)
    solver.estra_decisions = 99
    solver.last_estra_observation_key = ""
    solver.last_estra_stage_count = 0
    cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))

    assert LnrSolver._estra_trigger_allowed(solver, cards=cards, candidates=["S01", "S02"])


def test_lhr_ask_estra_uses_main_agent_context_when_enabled(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True, estra_use_main_agent_context=True)
        solver.stage_snapshots = {"S01": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 5
            _last_call_input_cached_tokens = 95

            def __init__(self) -> None:
                self.seen_messages = []
                self.seen_tool_choice = None

            async def ask_tool_stream(self, **kwargs):
                self.seen_messages = list(kwargs["messages"])
                self.seen_tool_choice = kwargs.get("tool_choice")
                return Message.assistant_message('{"action":"switch_stage","target_stage":"S01","reason":"main context best"}')

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                return [Message.user_message("MAIN_MEMORY_MARKER")]

        class FakeAgent:
            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self._llm_stream_timeout_sec = 5
                self._tools_with_thought = []

            @staticmethod
            def _build_system_messages():
                return [Message.system_message("sys")]

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        agent = FakeAgent()
        decision = await LnrSolver._ask_estra(solver, agent, trigger_source="force_stage_capture")
        assert decision["action"] == "switch_stage"
        assert agent.llm.seen_tool_choice == "none"
        assert agent.llm.seen_messages[0].content == "MAIN_MEMORY_MARKER"
        assert "Stage checkpoint context" in agent.llm.seen_messages[-1].content
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["decision_mode"] == "main_agent_context"
        assert route["payload"]["trigger_source"] == "force_stage_capture"

    asyncio.run(_run())


def test_lhr_ask_estra_falls_back_when_main_context_emits_tool_markup(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True, estra_use_main_agent_context=True)
        solver.stage_snapshots = {"S01": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 5
            _last_call_input_cached_tokens = 90

            async def ask_tool_stream(self, **kwargs):
                return Message.assistant_message(
                    '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="bash"></｜｜DSML｜｜invoke>'
                )

            async def ask(self, **kwargs):
                return '{"action":"switch_stage","target_stage":"S01","reason":"fallback best"}'

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                return [Message.user_message("MAIN_MEMORY_MARKER")]

        class FakeAgent:
            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self._llm_stream_timeout_sec = 5
                self._tools_with_thought = []

            @staticmethod
            def _build_system_messages():
                return [Message.system_message("sys")]

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(
            solver,
            FakeAgent(),
            trigger_source="force_stage_capture",
        )
        assert decision["action"] == "switch_stage"
        assert decision["startpoint"] == "previous_stage"
        assert decision["intent"] == "continue"
        assert decision["target_stage"] == "S01"
        assert decision["compact"] is True
        assert decision["reason"] == "fallback best"
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows][-2:] == [
            "estra_main_context_invalid_fallback",
            "estra_decision",
        ]
        route = rows[-1]
        assert route["payload"]["decision_mode"] == "isolated_fallback"
        assert route["payload"]["trigger_source"] == "force_stage_capture"

    asyncio.run(_run())


def test_lhr_force_estra_after_s02_sets_pending_estra(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            force_estra_after_stage_count=2,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.ledger_path.write_text(
            "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n\n"
            "### S02\nmetric: 0.08\nlower_is_better: true\nBRIEF: worse\nWHY: test route\n",
            encoding="utf-8",
        )

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "force_stage_capture"
            return {"action": "switch_stage", "target_stage": "S01", "reason": "forced validation"}

        solver._ask_estra = fake_ask_estra
        cards = [SimpleNamespace(stage_id="S01"), SimpleNamespace(stage_id="S02")]
        out = await LnrSolver._force_estra_after_stage_capture(
            solver,
            agent=object(),
            cards_after=cards,
        )

        assert out == "[lnr] estra switch_stage chosen from forced stage estra: S01"
        assert solver.estra_decisions == 1
        assert solver.pending_estra["target_stage"] == "S01"
        assert solver.pending_estra["reason"] == "forced validation"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        check = [row for row in rows if row["event"] == "force_stage_estra_check"][-1]
        assert check["payload"]["latest_stage"] == "S02"
        assert check["payload"]["candidate_count"] == 2
        assert set(check["payload"]["candidate_node_uids"]) == {"S01", "S02"}
        assert rows[-1]["event"] == "state_packet_built"

    asyncio.run(_run())


def test_lhr_context_hygiene_compact_uses_real_estra_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            context_hygiene_compact_enabled=True,
            context_hygiene_large_tool_output_chars=12000,
            context_hygiene_max_stages_without_compact=25,
            context_hygiene_code_churn_stage_threshold=10,
            context_hygiene_large_file_repeat_threshold=10,
            context_hygiene_low_incremental_cache_rate=0.80,
            context_hygiene_low_cache_window=5,
            context_hygiene_min_tokens_since_compact=1,
            context_hygiene_tool_output_min_interval_sec=1800.0,
            estra_enabled=True,
            estra_max_decisions=2,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.cfg = SimpleNamespace(max_messages=100)
        solver.stage_snapshots = {
            "S01": StageSnapshot("S01", "snap1", tmp_path / "s1", 0.07, "Final Validation Score", True, 1, {"source_commit_sha": "aaa111"}, node_uid="W00:L01:S01", lineage_id="L01"),
            "S02": StageSnapshot("S02", "snap2", tmp_path / "s2", 0.08, "Final Validation Score", True, 2, {"source_commit_sha": "bbb222"}, node_uid="W00:L01:S02", lineage_id="L01"),
        }
        solver.estra_decisions = 0
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0
        solver.pending_estra = None
        solver.main_tokens_in = 10
        solver.main_tokens_cached = 8
        solver.context_hygiene_last_stage_tokens_in = 0
        solver.context_hygiene_last_stage_tokens_cached = 0
        solver.context_hygiene_last_compact_tokens_in = 0
        solver.context_hygiene_last_compact_stage_count = 0
        solver.context_hygiene_last_compact_ts = 0.0
        solver.context_hygiene_cache_rates = []
        solver.started_at = 0.0

        monkeypatch.setattr(
            lnr_solver_module,
            "evaluate_context_hygiene_compact",
            lambda **kwargs: SimpleNamespace(should_compact=True, reason="unit hygiene", facts={"unit": True}),
        )

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                assert "ESTRA trigger" not in kwargs["messages"][0].content
                return '{"action":"switch_stage","target_stage":"S01","reason":"better historical route"}'

        class FakeAgent:
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5
            _run_tokens_in = 0
            _run_tokens_cached = 0

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))
        out = await LnrSolver._context_hygiene_compact_after_stage_capture(
            solver,
            agent=FakeAgent(),
            cards_after=cards,
        )

        assert out == "[lnr] estra switch_stage chosen from context hygiene: S01"
        assert solver.estra_decisions == 1
        assert solver.pending_estra["action"] == "switch_stage"
        assert solver.pending_estra["target_stage"] == "S01"
        assert solver.pending_estra["trigger_source"] == "context_hygiene"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        estra = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert estra["payload"]["trigger_source"] == "context_hygiene"
        assert estra["payload"]["action"] == "switch_stage"
        triggered = [row for row in rows if row["event"] == "cache_hygiene_compact_triggered"][-1]
        assert triggered["payload"]["action"] == "switch_stage"
        assert triggered["payload"]["target_stage"] == "S01"

    asyncio.run(_run())


def test_lhr_force_estra_cannot_capture_duplicate_submission(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            estra_enabled=True,
            estra_max_decisions=1,
            force_estra_after_stage_count=2,
            force_estra_target_stage="S01",
            force_estra_capture_duplicate_submissions=True,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                0.07,
                "Final Validation Score",
                True,
                1,
                {"submission_sha": "same", "solution_sha": "old"},
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_estra_stage_count = 0
        solver.last_force_estra_observation_count = 0
        solver.duplicate_submission_skip_keys = set()
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver.ledger_path.write_text(
            "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.071,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "new",
            "submission_sha": "same",
            "solution_path": "predict.py",
            "validation_ok": True,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):  # pragma: no cover - should not run
            raise AssertionError("force estra must not capture duplicate candidate artifacts")

        class FakeSnapshotStore:
            def capture(self, **kwargs):
                return StageSnapshot(
                    kwargs["stage_id"],
                    "snap2",
                    tmp_path / "snap2",
                    kwargs.get("metric_value"),
                    kwargs.get("metric_name", ""),
                    kwargs.get("lower_is_better"),
                    kwargs.get("memory_cut", 0),
                    kwargs.get("source_event", {}),
                    node_uid=kwargs.get("node_uid", ""),
                    lineage_id=kwargs.get("lineage_id", ""),
                )

        async def fake_ask_estra(agent, *, trigger_source: str):  # pragma: no cover - should not run
            raise AssertionError("duplicate skip should happen before forced stage estra")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver.snapshot_store = FakeSnapshotStore()
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None
        solver._ask_estra = fake_ask_estra

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert solver.estra_decisions == 0
        assert solver.pending_estra is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert "stage_duplicate_submission_skipped" in events
        assert "stage_captured" not in events
        assert "force_stage_estra_check" not in events
        skipped = [row for row in rows if row["event"] == "stage_duplicate_submission_skipped"][-1]
        assert skipped["payload"]["force_capture_requested"] is True
        assert skipped["payload"]["stage_policy"] == "duplicate_candidate_same_artifact_different_source"

    asyncio.run(_run())


def test_lhr_duplicate_submission_without_semantic_source_change_is_skipped(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                0.07,
                "Final Validation Score",
                True,
                1,
                {"submission_sha": "same", "solution_sha": "old", "candidate_ready": True},
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.duplicate_submission_skip_keys = set()
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver.ledger_path.write_text(
            """### S01
metric: 0.07
lower_is_better: true
BRIEF: base
WHY: ok
""",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.061,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "same-source",
            "submission_sha": "same",
            "solution_path": "train.py",
            "validation_ok": True,
            "selection_eligible": True,
            "selection_score": 0.061,
            "source_changed": False,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):  # pragma: no cover - should not run
            raise AssertionError("duplicate capture without semantic source change should be skipped")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S02" not in solver.stage_snapshots
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert "stage_duplicate_submission_skipped" in events
        assert "stage_duplicate_submission_metric_stage_kept" not in events
        assert "stage_captured" not in events
        skipped = [row for row in rows if row["event"] == "stage_duplicate_submission_skipped"][-1]
        assert skipped["payload"]["stage_policy"] == "duplicate_candidate_no_semantic_workspace_change"

    asyncio.run(_run())


def test_lhr_duplicate_submission_same_design_is_skipped_even_with_source_change(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=True,
            force_estra_after_stage_count=2,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                0.07,
                "Final Validation Score",
                True,
                1,
                {
                    "submission_sha": "same",
                    "solution_sha": "same-source",
                    "artifact_sha": "same",
                    "candidate_ready": True,
                },
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.duplicate_submission_skip_keys = set()
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver.ledger_path.write_text(
            "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.07,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "same-source",
            "submission_sha": "same",
            "artifact_sha": "same",
            "solution_path": "predict.py",
            "validation_ok": True,
            "selection_eligible": True,
            "selection_score": 0.07,
            "source_changed": True,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):  # pragma: no cover - should not run
            raise AssertionError("same design duplicate should not be captured")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S02" not in solver.stage_snapshots
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        skipped = [row for row in rows if row["event"] == "stage_duplicate_submission_skipped"][-1]
        assert skipped["payload"]["semantic_source_changed"] is True
        assert skipped["payload"]["same_design_duplicate"] is True
        assert skipped["payload"]["stage_policy"] == "duplicate_candidate_same_design"
        assert "stage_captured" not in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_duplicate_artifact_without_submission_is_skipped(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=False,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md", "artifacts/*.json"],
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                2.617322,
                "radii_sum",
                False,
                1,
                {
                    "artifact_sha": "same-artifact",
                    "artifact_path": "artifacts/best_solution.json",
                    "candidate_ready": True,
                },
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver.ledger_path.write_text(
            "### S01\nmetric: 2.617322\nlower_is_better: false\nBRIEF: base\nWHY: ok\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 2.617322,
            "metric_name": "radii_sum",
            "lower_is_better": False,
            "artifact_sha": "same-artifact",
            "artifact_path": "artifacts/best_solution.json",
            "validation_ok": True,
            "selection_eligible": True,
            "source_changed": True,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):  # pragma: no cover - should not run
            raise AssertionError("duplicate artifact without submission should not be captured")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        skipped = [row for row in rows if row["event"] == "stage_duplicate_submission_skipped"][-1]
        assert skipped["payload"]["artifact_sha"] == "same-artifact"
        assert skipped["payload"]["stage_policy"] == "duplicate_candidate_same_artifact_different_source"
        assert "stage_captured" not in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_primary_evaluator_error_is_returned_to_agent(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._evaluator_candidate_artifact = lambda: "artifacts/best_solution.json"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            "metric_value": None,
            "evaluator_backend": "artifact_command",
            "evaluator_status": "evaluator_error",
            "artifact_path": "artifacts/best_solution.json",
            "metric_source_note": "evaluator command exited with 2",
            "evaluator_stderr_tail": "kttsp_eval error: leg 6 tof 0.0 is below min_tof 0.001",
        }

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is not None
        assert out.startswith("EVALUATOR_INVALID_ARTIFACT")
        assert "artifacts/best_solution.json" in out
        assert "evaluator_error" in out
        assert "leg 6 tof 0.0" in out
        assert "satisfies the task evaluator" in out

    asyncio.run(_run())


def test_lhr_primary_evaluator_ready_candidate_requires_text_commit(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.deadline = 10.0
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            stage_commit_llm_timeout_sec=60,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            "metric_value": 0.42,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "artifact_path": "submission.csv",
            "artifact_sha": "ready-sha",
            "submission_sha": "ready-sha",
            "submission_status": "ok",
            "evaluator_backend": "task_package",
            "evaluator_status": "ok",
            "gate_action": "accept",
            "gate_accepted": True,
            "gate_reason_code": "eligible",
            "_gate_evaluated": True,
            "_evaluator_stage_facts_applied": True,
        }

        async def fake_stage_commit(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("primary evaluator ready candidate must wait for text commit")

        solver._ephemeral_stage_commit = fake_stage_commit
        agent = SimpleNamespace(_run_policy=SimpleNamespace(deadline_monotonic=10.0))

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=agent,
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert solver.pending_text_stage_commit["stage_id"] == "S03"
        assert solver.pending_text_stage_commit["metric_event"]["artifact_path"] == "submission.csv"
        assert "### S03" not in solver.ledger_path.read_text(encoding="utf-8")
        assert "STAGE_COMMIT_BEGIN" in agent._lnr_transient_user_prompt
        assert agent._lnr_stage_commit_text_pending is True
        assert agent._run_policy.deadline_monotonic > 10.0

    asyncio.run(_run())


def test_lhr_primary_duplicate_artifact_is_skipped_before_gate(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._candidate_artifact_sha_from_workspace = lambda _event: "same-artifact"
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snapshot-one",
                tmp_path / "snapshot-one",
                0.42,
                "score",
                False,
                1,
                {"artifact_sha": "same-artifact", "gate_accepted": True},
            )
        }

        def should_not_evaluate(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("an unchanged committed artifact must not re-enter Gate")

        solver._record_evaluator_stage_events = should_not_evaluate
        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        skipped = [row for row in rows if row["event"] == "duplicate_candidate_pre_gate_skipped"]
        assert skipped[-1]["payload"]["duplicate_of_stage"] == "S01"

    asyncio.run(_run())


def test_lhr_metric_missing_artifact_is_not_re_evaluated_while_metric_absent(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._candidate_artifact_sha_from_workspace = lambda _event: "pending-artifact"
        solver._evaluator_feedback_for_agent = lambda _event: "metric missing"
        evaluations = 0

        def reject_missing_metric(**_kwargs):
            nonlocal evaluations
            evaluations += 1
            return {
                "_gate_evaluated": True,
                "gate_accepted": False,
                "gate_reason_code": "metric_missing",
            }

        solver._record_evaluator_stage_events = reject_missing_metric
        first = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="artifact written"),
        )
        second = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="unrelated tool"),
        )

        assert first == "metric missing"
        assert second is None
        assert evaluations == 1
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        skipped = [
            row
            for row in rows
            if row["event"] == "duplicate_pending_metric_pre_gate_skipped"
        ]
        assert skipped[-1]["payload"]["artifact_sha"] == "pending-artifact"

    asyncio.run(_run())


def test_lhr_gate_retry_with_valid_metric_does_not_create_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.52,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "solution_sha": "src",
            "submission_sha": "candidate-sha",
        }
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            **metric_event,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": False,
            "metric_validity": "medium",
            "artifact_path": "submission.csv",
            "artifact_sha": "candidate-sha",
            "evaluator_backend": "task_package",
            "evaluator_status": "ok",
            "gate_action": "retry",
            "gate_accepted": False,
            "gate_reason_code": "metric_validity_below_gate",
            "gate_message": "metric validity is below required high",
            "_gate_evaluated": True,
        }
        solver._evaluator_feedback_for_agent = lambda _event: "gate feedback"

        async def fake_stage_commit(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("a rejected gate outcome must not request a stage commit")

        solver._ephemeral_stage_commit = fake_stage_commit

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out == "gate feedback"
        assert not hasattr(solver, "pending_text_stage_commit") or solver.pending_text_stage_commit is None
        assert "### S03" not in solver.ledger_path.read_text(encoding="utf-8")
        assert "S03" not in solver.stage_snapshots
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        rejected = [row for row in rows if row["event"] == "stage_gate_rejected"]
        assert rejected[-1]["payload"]["gate_reason_code"] == "metric_validity_below_gate"

    asyncio.run(_run())


def test_lhr_primary_evaluator_invalid_candidate_is_skipped(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.52,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "solution_sha": "src",
            "submission_sha": "bad-sha",
        }
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            **metric_event,
            "validation_ok": False,
            "candidate_ready": False,
            "selection_eligible": False,
            "artifact_path": "submission.csv",
            "artifact_sha": "bad-sha",
            "evaluator_backend": "task_package",
            "evaluator_status": "invalid_submission",
        }
        solver._evaluator_feedback_for_agent = lambda _event: "feedback"

        async def fake_stage_commit(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("invalid evaluator candidate must not commit")

        solver._ephemeral_stage_commit = fake_stage_commit

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out == "feedback"
        assert "S03" not in solver.stage_snapshots
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "stage_invalid_evaluator_candidate_skipped" in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_ready_submission_materializes_existing_stage_without_new_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                0.060612,
                "Final Validation Score",
                True,
                1,
                {"submission_sha": "stale-old", "solution_sha": "same-source", "candidate_ready": False},
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver.ledger_path.write_text(
            "### S01\nmetric: 0.060612\nlower_is_better: true\nBRIEF: Optuna LGBM route\nWHY: validation route found, submission not ready\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.060612,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "same-source",
            "submission_sha": "ready-new",
            "solution_path": "train.py",
            "validation_ok": True,
            "selection_eligible": True,
            "selection_score": 0.060612,
            "source_changed": False,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):  # pragma: no cover - must not run
            raise AssertionError("materializing the same route must not request a new stage commit")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("materialization event must not append a new stage row")
        )

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert sorted(solver.stage_snapshots) == ["S01"]
        assert solver.stage_snapshots["S01"].source_event["candidate_ready"] is True
        assert solver.stage_snapshots["S01"].source_event["materialized_ready_submission"] is True
        cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))
        assert [card.stage_id for card in cards] == ["S01"]
        assert cards[0].stage_events == ("Ready submission generated from the same route; no new research stage.",)
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert "stage_materialized_ready_submission" in events
        assert "stage_captured" not in events
        materialized = [row for row in rows if row["event"] == "stage_materialized_ready_submission"][-1]
        assert materialized["payload"]["target_stage"] == "S01"
        assert materialized["payload"]["stage_policy"] == "materialized_existing_stage_no_new_research_stage"

    asyncio.run(_run())

def test_lhr_metric_stage_without_submission_marks_missing_submission(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {}
        solver.last_captured_solution_sha = ""
        solver.last_captured_run_signature = ""
        solver.last_stage_commit_ts = 0.0
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.061,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "train-only-source",
            "solution_path": "train.py",
            "submission_sha": "",
            "validation_ok": True,
            "submission_status": "missing_submission",
            "selection_eligible": True,
            "selection_score": 0.061,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):
            assert stage_id == "S03"
            assert metric_event["candidate_ready"] is False
            assert metric_event["submission_status"] == "missing_submission"
            with solver.ledger_path.open("a", encoding="utf-8") as f:
                f.write("\n### S03\nmetric: 0.061\nlower_is_better: true\nBRIEF: validation-only train stage\nWHY: no submission required for stage\n")
            return True, ""

        class FakeSnapshotStore:
            def capture(self, **kwargs):
                return StageSnapshot(
                    kwargs["stage_id"],
                    "snap1",
                    tmp_path / "snap1",
                    kwargs.get("metric_value"),
                    kwargs.get("metric_name", ""),
                    kwargs.get("lower_is_better"),
                    kwargs.get("memory_cut", 0),
                    kwargs.get("source_event", {}),
                    node_uid=kwargs.get("node_uid", ""),
                    lineage_id=kwargs.get("lineage_id", ""),
                )

        solver._ephemeral_stage_commit = fake_stage_commit
        solver.snapshot_store = FakeSnapshotStore()
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        async def fake_force_estra_after_stage_capture(**kwargs):
            return None

        solver._force_estra_after_stage_capture = fake_force_estra_after_stage_capture

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S03" in solver.stage_snapshots
        assert solver.stage_snapshots["S03"].source_event["candidate_ready"] is False
        assert solver.stage_snapshots["S03"].source_event["submission_status"] == "missing_submission"
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "stage_captured" in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_invalid_submission_is_not_captured_as_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {}
        solver.last_captured_solution_sha = ""
        solver.last_captured_run_signature = ""
        solver.last_stage_commit_ts = 0.0
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.052,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "new-source",
            "solution_path": "solution.py",
            "submission_sha": "bad-submission",
            "validation_ok": True,
            "submission_validation_ok": False,
            "submission_status": "invalid_submission",
            "selection_eligible": True,
            "selection_score": 0.052,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):
            raise AssertionError("invalid submission must not request a stage commit")

        class FakeSnapshotStore:
            def capture(self, **kwargs):
                raise AssertionError("invalid submission must not create a stage snapshot")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver.snapshot_store = FakeSnapshotStore()
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        async def fake_force_estra_after_stage_capture(**kwargs):
            raise AssertionError("invalid submission must not trigger estra")

        solver._force_estra_after_stage_capture = fake_force_estra_after_stage_capture

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S03" not in solver.stage_snapshots
        assert solver.last_captured_solution_sha == "new-source"
        assert solver.last_captured_run_signature
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert "stage_invalid_submission_skipped" in events
        assert "stage_invalid_submission_metric_stage_kept" not in events
        assert "stage_captured" not in events

    asyncio.run(_run())


def test_lhr_run_signature_includes_cli_hyperparameters() -> None:
    base = {
        "metric_value": 0.061,
        "metric_name": "Final Validation Score",
        "solution_sha": "same-source",
        "submission_sha": "same-submission",
    }
    sig_lr_007 = LnrSolver._stage_run_signature(
        {**base, "bash_cmd": "python3 train.py --lr 0.07 --depth 6"}
    )
    sig_lr_003 = LnrSolver._stage_run_signature(
        {**base, "bash_cmd": "python3 train.py --lr 0.03 --depth 6"}
    )
    sig_env_007 = LnrSolver._stage_run_signature(
        {**base, "bash_cmd": "LR=0.07 python3 train.py --depth 6"}
    )
    sig_env_003 = LnrSolver._stage_run_signature(
        {**base, "bash_cmd": "LR=0.03 python3 train.py --depth 6"}
    )
    assert sig_lr_007 != sig_lr_003
    assert sig_env_007 != sig_env_003


def test_lhr_stage_commit_persists_agent_write_without_controller_prompt(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            stage_commit_max_turns=1,
            stage_commit_llm_timeout_sec=17,
            stage_commit_persist_to_memory=False,
            stage_commit_persist_agent_write_to_memory=True,
            stage_commit_persist_prompt_to_memory=False,
        )
        solver.stage_tokens_in = 0
        solver.stage_tokens_out = 0
        solver.stage_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        (solver.workspace_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        class FakeLLM:
            _last_call_input_tokens = 50
            _last_call_output_tokens = 4
            _last_call_input_cached_tokens = 45

            def __init__(self) -> None:
                self.timeout_seen = None
                self.tool_choice_seen = None
                self.tools_seen = None

            async def ask_tool_stream(self, **kwargs):
                self.timeout_seen = kwargs.get("timeout")
                self.tool_choice_seen = kwargs.get("tool_choice")
                self.tools_seen = kwargs.get("tools")
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "brief": "main-context LightGBM metadata experiment",
                            "why": "preserves a comparable validation route judgment",
                            "route_evidence": "verdict=continue; reason=useful cheap baseline; next=blend",
                            "files": "code=train.py",
                        }
                    ),
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc-hallucinated",
                            function=SimpleNamespace(name="read", arguments=json.dumps({"path": "train.py"})),
                        )
                    ],
                )

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                return [Message.user_message("FIRST USER")]

        class FakeMemory:
            def __init__(self) -> None:
                self.messages = []

            def add_message(self, msg):
                self.messages.append(msg)

        class FakeTools:
            async def execute(self, *, name, tool_input):  # pragma: no cover - stage commit must not execute tools
                raise AssertionError("stage commit fork must not execute tools")

        class FakeAgent:
            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self.memory = FakeMemory()
                self.availableTools = FakeTools()
                self._llm_stream_timeout_sec = 5
                self._tools_with_thought = []

            @staticmethod
            def _build_system_messages():
                return [Message.system_message("sys")]

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        agent = FakeAgent()
        ok, reason = await LnrSolver._ephemeral_stage_commit(
            solver,
            agent=agent,
            stage_id="S03",
            metric_event={
                "metric_value": 0.06,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "run_time_sec": 12.0,
                "val_score_type": "cv",
                "metric_validity": "high",
            },
        )
        assert ok, reason
        assert agent.llm.timeout_seen == 17
        assert agent.llm.tool_choice_seen == "none"
        assert agent.llm.tools_seen == []
        ledger = solver.ledger_path.read_text(encoding="utf-8")
        assert "### S03" in ledger
        assert "metric: 0.06" in ledger
        assert "BRIEF: main-context LightGBM metadata experiment" in ledger
        assert "WHY: preserves a comparable validation route judgment" in ledger
        assert len(agent.memory.messages) == 1
        assert "Forked stage bookkeeping judgment" not in str(agent.memory.messages[0].content)
        assert not getattr(agent.memory.messages[0], "tool_calls", None)
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        persisted = [row for row in rows if row["event"] == "stage_commit_persisted_to_memory"][-1]
        assert persisted["payload"]["agent_write_persisted"] is True
        assert persisted["payload"]["prompt_persisted"] is False
        assert persisted["payload"]["deterministic_append"] is True

    asyncio.run(_run())


def test_lhr_text_only_callback_records_noop_without_stage_candidates(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.ledger_path.write_text("", encoding="utf-8")
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        async def fake_ask_estra(agent, *, trigger_source: str):
            raise AssertionError("no-op text-only estra must not call the estra LLM")

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="The task is complete.",
            round_idx=1,
            max_steps=10,
        )
        assert out is None
        assert solver.estra_decisions == 0
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["text_only_estra_noop"]
        payload = rows[0]["payload"]
        assert payload["stage_count"] == 0
        assert payload["candidate_count"] == 0
        assert payload["noop_reason"] == "no_stage_candidates"

    asyncio.run(_run())


def test_lhr_text_only_callback_suppresses_repeated_completion_context(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.ledger_path.write_text("", encoding="utf-8")
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        class _Memory:
            def __init__(self) -> None:
                self.messages = []

            def add_message(self, msg) -> None:
                self.messages.append(msg)

        agent = SimpleNamespace(memory=_Memory())
        text = "The task is complete. Final Validation Score: 0.8320054902957775"

        first = await LnrSolver._text_only_estra_callback(
            solver,
            agent=agent,
            assistant_text=text,
            round_idx=1,
            max_steps=10,
        )
        assert first is None
        assert not getattr(agent, "_lnr_suppress_current_text_only_memory", "")

        second = await LnrSolver._text_only_estra_callback(
            solver,
            agent=agent,
            assistant_text=text,
            round_idx=2,
            max_steps=10,
        )
        assert second == "[lnr] repeated text-only terminal response suppressed; continue-search prompt injected"
        assert getattr(agent, "_lnr_suppress_current_text_only_memory", "").startswith(
            "repeated_text_only_completion:2",
        )
        assert len(agent.memory.messages) == 1
        continue_prompt = str(agent.memory.messages[0].content)
        assert "[LNR_CONTINUE_SEARCH]" in continue_prompt
        assert "workers stop only when the time budget expires" in continue_prompt
        assert "Exploration state:" in continue_prompt
        assert "completed evidence, not a stop signal" in continue_prompt
        assert "not something to repeat" in continue_prompt
        assert "Do not overwrite the best submission without a validated candidate" in continue_prompt
        assert solver.estra_decisions == 0
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert events == ["text_only_estra_noop", "text_only_completion_duplicate_suppressed"]
        suppressed = rows[-1]["payload"]
        assert suppressed["repeat_count"] == 2
        assert suppressed["signature"] == "task_complete|final_validation_score:0.8320054902957775"
        assert suppressed["continue_prompt_injected"] is True
        assert "LNR_CONTINUE_SEARCH" in suppressed["continue_prompt"]
        assert "completed evidence, not a stop signal" in suppressed["continue_prompt"]

    asyncio.run(_run())


def test_lhr_text_only_completion_signature_covers_terminal_status_text() -> None:
    assert LnrSolver._text_only_completion_signature("Goodbye.") == "terminal_text"
    assert LnrSolver._text_only_completion_signature(
        "I understand. The conversation has definitively concluded. I will not respond further. Goodbye."
    ) == "terminal_text"
    assert LnrSolver._text_only_completion_signature(
        "Final status confirmed. All experiments concluded. No further actions from this worker."
    ) == "terminal_text"


def test_lhr_text_only_callback_sets_pending_estra_without_stage_capture(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "text_only"
            return {"action": "switch_stage", "target_stage": "S01", "reason": "best"}

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="I am done.",
            round_idx=3,
            max_steps=10,
        )
        assert out == "[lnr] estra switch_stage chosen from text-only: S01"
        assert solver.estra_decisions == 1
        assert solver.pending_estra["target_stage"] == "S01"
        assert "S02" in solver.pending_estra["tail_summary"]
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["text_only_estra_check", "state_packet_built"]
        check = [row for row in rows if row["event"] == "text_only_estra_check"][0]["payload"]
        assert check["assistant_text"] == "I am done."
        assert check["assistant_text_chars"] == len("I am done.")
        assert check["round"] == 3
        assert check["max_steps"] == 10

    asyncio.run(_run())



def test_lhr_text_only_callback_keep_current_compacts_without_switch_candidate(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "text_only"
            return {"action": "keep_current", "target_stage": "S02", "reason": "continue current route"}

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="Final answer: 0.89468 AUC",
            round_idx=5,
            max_steps=10,
        )

        assert out == "[lnr] estra keep_current chosen from text-only; compact current stage: S02"
        assert solver.estra_decisions == 1
        assert solver.pending_estra["action"] == "keep_current"
        assert solver.pending_estra["target_stage"] == "S02"
        assert solver.pending_estra["trigger_source"] == "text_only"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["text_only_estra_check", "state_packet_built"]
        check = rows[0]["payload"]
        assert check["assistant_text"] == "Final answer: 0.89468 AUC"

    asyncio.run(_run())

def test_lhr_switch_estra_synthesizes_target_tail_archive_summary(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.ledger_path.write_text(
            """### S01
metric: 0.90
lower_is_better: false
BRIEF: stable target route
WHY: best reusable checkpoint

### S02
metric: 0.88
lower_is_better: false
BRIEF: noisy feature branch
WHY: regressed after adding noisy aggregate

### S03
metric: 0.87
lower_is_better: false
BRIEF: deeper variant
WHY: repeated shallow tuning without recovery
""",
            encoding="utf-8",
        )
        solver.lhr = SimpleNamespace(
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
            estra_archive_summary_llm_enabled=True,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object(), "S03": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 50
            _last_call_output_tokens = 8
            _last_call_input_cached_tokens = 45

            async def ask(self, **kwargs):
                prompt = kwargs["messages"][0].content
                assert "historical exploration" in prompt.lower()
                assert "Abandoned trajectory after S01 before estra" in prompt
                assert "S02" in prompt and "S03" in prompt
                assert "stable target route" not in prompt
                return "Avoid noisy aggregate feature branch; deeper variant repeated shallow tuning without recovery.\nRestart from S01 and test a materially different signal."

        class FakeAgent:
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        await LnrSolver._set_pending_estra_from_decision(
            solver,
            agent=FakeAgent(),
            target="S01",
            decision={"action": "switch_stage", "target_stage": "S01", "compact": True, "reason": "return to stable target"},
        )

        summary = solver.pending_estra["tail_summary"]
        assert "Avoid noisy aggregate" in summary
        assert "Restart from S01" in summary
        assert "Abandoned trajectory after" not in summary
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        assert "Historical Exploration Summary" in solver.pending_estra["state_packet"]
        assert "abandoned branch evidence" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "estra_archive_summary_synthesized" in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_context_limit_callback_sets_pending_estra_before_compact(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "context_limit"
            return {"action": "switch_stage", "target_stage": "S01", "reason": "context full"}

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=9,
            max_steps=100,
            omitted=7,
        )
        assert out == "[lnr] estra switch_stage chosen from context-limit: S01"
        assert solver.estra_decisions == 1
        assert solver.pending_estra["target_stage"] == "S01"
        assert solver.pending_estra["reason"] == "context full"
        assert "S02" in solver.pending_estra["tail_summary"]
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["context_limit_estra_check", "state_packet_built"]
        check = [row for row in rows if row["event"] == "context_limit_estra_check"][0]
        assert check["payload"]["trigger_source"] == "context_limit"
        assert check["payload"]["omitted_before"] == 7

    asyncio.run(_run())


def test_lhr_context_limit_uses_deterministic_estra_when_estra_not_allowed(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=0,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 2
        solver.estra_decisions = 0
        solver.pending_estra = None

        async def fail_ask_estra(agent, *, trigger_source: str):
            raise AssertionError("context-limit fallback should not call estra model")

        solver._ask_estra = fail_ask_estra
        out = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=10,
            max_steps=100,
            omitted=3,
        )

        assert out == "[lnr] estra switch_stage chosen from context-limit: S01"
        assert solver.pending_estra["target_stage"] == "S01"
        assert "deterministic estra fallback" in solver.pending_estra["reason"]
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "estra_deterministic_fallback" in [row["event"] for row in rows]
        assert "state_packet_built" in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_context_limit_suppresses_repeated_restore_key(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=0,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 2
        solver.estra_decisions = 0
        solver.pending_estra = None

        async def fail_ask_estra(agent, *, trigger_source: str):
            raise AssertionError("context-limit fallback should not call estra model")

        solver._ask_estra = fail_ask_estra
        first = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=10,
            max_steps=100,
            omitted=3,
        )
        assert first == "[lnr] estra switch_stage chosen from context-limit: S01"
        assert solver.pending_estra["compact_strength"] == "strict_context_limit"
        first_key = solver.pending_estra["restore_key"]
        solver.pending_estra = None

        second = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=10,
            max_steps=100,
            omitted=3,
        )

        assert second is None
        assert solver.pending_estra is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        suppressed = [row for row in rows if row["event"] == "context_limit_estra_restore_suppressed"]
        assert suppressed
        assert suppressed[-1]["payload"]["restore_key"] == first_key

    asyncio.run(_run())


def test_lhr_hidden_ledger_is_omitted_from_main_tool_feedback() -> None:
    dummy = SimpleNamespace(
        _agent_hidden_workspace_filenames=(".run_results.md",),
        _agent_hidden_workspace_path_prefixes=(
            ".run_results.md",
            "logs",
            "submission_snapshots",
            ".logs",
            ".memory",
            ".scienceflow_checkpoints",
            ".git",
        ),
        _tool_output_artifacts=None,
        _exec_feedback_max_chars=4000,
    )
    for name in (
        "_get_agent_hidden_workspace_filenames",
        "_normalize_hidden_workspace_path",
        "_get_agent_hidden_workspace_path_prefixes",
        "_text_mentions_hidden_workspace_prefix",
        "_hide_agent_hidden_workspace_filename_mentions",
        "_path_mentions_hidden_workspace_file",
        "_tool_request_mentions_hidden_workspace_file",
        "_hide_agent_hidden_workspace_file_lines",
        "_prepare_tool_feedback_for_memory",
        "_dedup_resource_feedback_for_memory",
    ):
        setattr(dummy, name, MethodType(getattr(ScienceAgent, name), dummy))

    class FakeMemoryCtx:
        def record_tool_result(self, *args, **kwargs):
            raise AssertionError("hidden ledger reads must not be recorded verbatim")

    dummy._memory_ctx = FakeMemoryCtx()
    feedback = dummy._prepare_tool_feedback_for_memory(
        "read",
        {"path": ".run_results.md"},
        ToolResult(output="[.run_results.md]\nsecret stage ledger"),
    )
    assert ".run_results.md" not in feedback
    assert "secret stage ledger" not in feedback
    assert "hidden control file" in feedback

    filtered = dummy._hide_agent_hidden_workspace_file_lines(
        "FILE solution.py\nDIR logs/\nDIR submission_snapshots/\nDIR .memory/\nFILE .logs/train.log\nFILE .logs/agentic_route_response.md\nFILE .run_results.md (100 bytes)\nFILE submission.csv"
    )
    assert ".run_results.md" not in filtered
    assert "submission_snapshots" not in filtered
    assert "DIR logs" not in filtered
    assert ".memory" not in filtered
    assert "agentic_route_response" not in filtered
    assert ".logs/train.log" not in filtered
    assert "solution.py" in filtered
    assert "submission.csv" in filtered

    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "ls -la *.py artifacts/ submission_snapshots/ | head"}
    ) is True
    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "tail -30 .logs/train_classifier.log"}
    ) is True
    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "tail -30 tmp/train_classifier.log"}
    ) is False
    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "ls -la logs/ artifacts/ submission.csv"}
    ) is True


def test_lhr_resource_state_summary_slot_is_removed_from_memory() -> None:
    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )

    class FakeStorage:
        def __init__(self, owner):
            self.owner = owner

        def clear(self):
            self.owner.messages.clear()

    class FakeChatHistory:
        def __init__(self, owner):
            self.owner = owner
            self.storage = FakeStorage(owner)

        def retrieve(self, window_size=None):
            del window_size
            return [SimpleNamespace(memory_record=SimpleNamespace(message=m)) for m in self.owner.messages]

    class FakeMemory:
        def __init__(self):
            self.messages = [
                Message.user_message("task context"),
                Message.user_message(RESOURCE_STATE_SUMMARY_MARKER + "\nold"),
            ]
            self.chat_history_memory = FakeChatHistory(self)

        def add_message(self, message):
            self.messages.append(message)

    dummy = SimpleNamespace(
        memory=FakeMemory(),
        _resource_feedback_memory_deduper=ResourceFeedbackMemoryDeduper(),
        _resource_state_summary_last_text="",
    )
    setattr(
        dummy,
        "_sync_resource_state_summary_slot",
        MethodType(getattr(ScienceAgent, "_sync_resource_state_summary_slot"), dummy),
    )

    dummy._resource_feedback_memory_deduper.reduce(feedback)
    dummy._sync_resource_state_summary_slot()
    joined = "\n".join(str(m.content or "") for m in dummy.memory.messages)
    assert RESOURCE_STATE_SUMMARY_MARKER not in joined
    assert "repeat_count=1" not in joined
    assert "old" not in joined

    dummy._resource_feedback_memory_deduper.reduce(feedback)
    dummy._sync_resource_state_summary_slot()
    joined = "\n".join(str(m.content or "") for m in dummy.memory.messages)
    assert RESOURCE_STATE_SUMMARY_MARKER not in joined
    assert "repeat_count=2" not in joined
    assert "task context" in joined


def test_lhr_resource_state_summary_slot_cleanup_rewrites_long_term(tmp_path) -> None:
    from scienceflow.core.agent_runtime import create_agent_memory
    from scienceflow.core.mem.memory_context import MemoryContextManager

    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )
    mem = create_agent_memory(tmp_path / "summary_mem", "ScienceAgent", 200)
    mem.add_message(Message.user_message("You are solving one ML task in a continuous REPL workspace. Task body."))
    mem.add_message(Message.user_message(RESOURCE_STATE_SUMMARY_MARKER + "\nold"))
    ctx = MemoryContextManager(mem, tmp_path / "workspace", budget_chars=60000)
    dummy = SimpleNamespace(
        memory=mem,
        _memory_ctx=ctx,
        _resource_feedback_memory_deduper=ResourceFeedbackMemoryDeduper(),
        _resource_state_summary_last_text="",
    )
    setattr(
        dummy,
        "_sync_resource_state_summary_slot",
        MethodType(getattr(ScienceAgent, "_sync_resource_state_summary_slot"), dummy),
    )

    for _ in range(2):
        dummy._resource_feedback_memory_deduper.reduce(feedback)
        dummy._sync_resource_state_summary_slot()

    messages = [r.memory_record.message for r in mem.chat_history_memory.retrieve(window_size=None)]
    joined = "\n".join(str(m.content or "") for m in messages)
    long_term = tmp_path / "summary_mem" / "ScienceAgent" / "long_term.jsonl"
    long_term_lines = [line for line in long_term.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(long_term_lines) == len(messages)
    assert joined.count("You are solving one ML task in a continuous REPL workspace.") == 1
    assert RESOURCE_STATE_SUMMARY_MARKER not in joined
    assert "repeat_count=2" not in joined


def test_lhr_resource_feedback_repeats_are_deduped_before_memory() -> None:
    dummy = SimpleNamespace(
        _agent_hidden_workspace_filenames=(),
        _agent_hidden_workspace_path_prefixes=(),
        _tool_output_artifacts=None,
        _exec_feedback_max_chars=4000,
    )
    for name in (
        "_get_agent_hidden_workspace_filenames",
        "_normalize_hidden_workspace_path",
        "_get_agent_hidden_workspace_path_prefixes",
        "_text_mentions_hidden_workspace_prefix",
        "_hide_agent_hidden_workspace_filename_mentions",
        "_path_mentions_hidden_workspace_file",
        "_tool_request_mentions_hidden_workspace_file",
        "_hide_agent_hidden_workspace_file_lines",
        "_prepare_tool_feedback_for_memory",
        "_dedup_resource_feedback_for_memory",
    ):
        setattr(dummy, name, MethodType(getattr(ScienceAgent, name), dummy))

    class FakeMemoryCtx:
        def record_tool_result(self, tool_name, args, tool_result, **kwargs):
            return tool_result.output

    dummy._memory_ctx = FakeMemoryCtx()
    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )

    first = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "python train.py"},
        ToolResult(output=feedback),
    )
    second = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "python train.py"},
        ToolResult(output=feedback),
    )
    third = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "python train.py"},
        ToolResult(output=feedback),
    )

    assert first == ""
    assert second == ""
    assert third == ""
    summary = dummy._resource_feedback_memory_deduper.summary_text()
    assert "RESOURCE_STATE_SUMMARY" in summary
    assert "active_resource_plan_guard" in summary
    assert "repeat_count=3" in summary



def test_lhr_stage_commit_text_block_parser_accepts_negative_metric() -> None:
    text = """
    STAGE_COMMIT_BEGIN
    stage_id: S04
    metric: -4.835054
    metric_validity: medium
    brief: valid negative metric stage.
    why: lower-is-better metric improved relative to prior run.
    route_evidence: verdict=continue; reason=metric is comparable; next=refine calibration
    files: code=train.py weights=models/best.pt
    STAGE_COMMIT_END
    """

    parsed, block_text, reason = LnrSolver._parse_stage_commit_text_block(text)

    assert reason == ""
    assert parsed["stage_id"] == "S04"
    assert float(parsed["metric"]) == pytest.approx(-4.835054)
    assert parsed["metric_validity"] == "medium"
    assert parsed["files"] == "code=train.py weights=models/best.pt"
    assert block_text.startswith("STAGE_COMMIT_BEGIN")


def test_lhr_stage_commit_text_block_parser_requires_files() -> None:
    text = """
    STAGE_COMMIT_BEGIN
    stage_id: S04
    metric: 0.12
    metric_validity: medium
    brief: valid metric but missing files.
    why: this must not be accepted because lineage would be ambiguous.
    STAGE_COMMIT_END
    """

    _parsed, _block_text, reason = LnrSolver._parse_stage_commit_text_block(text)

    assert reason == "missing=files"


def test_lhr_stage_commit_json_block_parser_requires_one_fenced_object() -> None:
    text = """```json
{
  "stage_id": "S04",
  "metric": -4.835054,
  "metric_validity": "high",
  "metric_source": "official evaluator",
  "lower_is_better": true,
  "run_time_sec": 12.3,
  "brief": "validated model and feature route",
  "why": "improved the held-out result",
  "files": "code=train.py weights=models/best.pt"
}
```"""

    parsed, block_text, reason = LnrSolver._parse_stage_commit_json_block(text)

    assert reason == ""
    assert parsed["stage_id"] == "S04"
    assert parsed["lower_is_better"] is True
    assert parsed["files"] == "code=train.py weights=models/best.pt"
    assert block_text == text

    _parsed, _block_text, reason = LnrSolver._parse_stage_commit_json_block(
        "extra text\n" + text,
    )
    assert reason == "missing_json_block"


def test_lhr_stage_commit_files_retry_exhaustion_uses_deterministic_fallback(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.log_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_commit_persist_agent_write_to_memory=False,
            stage_commit_persist_to_memory=False,
        )
        solver.pending_text_stage_commit = {
            "stage_id": "S03",
            "attempts": 2,
            "now": 1.0,
            "solution_sha": "solution-sha",
            "run_signature": "run-signature",
            "metric_event": {
                "metric_value": 0.42,
                "metric_name": "score",
                "metric_validity": "high",
                "lower_is_better": False,
                "submission_status": "ok",
                "artifact_path": "artifacts/best_solution.json",
                "artifact_sha": "artifact-sha",
                "gate_accepted": True,
            },
        }

        async def no_audit(**_kwargs):
            return None

        async def finalize(**_kwargs):
            return "finalized"

        solver._audit_stage_result_before_commit = no_audit
        solver._finalize_stage_capture_after_commit = finalize
        agent = SimpleNamespace()
        assistant_text = """
        STAGE_COMMIT_BEGIN
        stage_id: S03
        metric: 0.42
        metric_validity: high
        brief: accepted metric with a stale file reference.
        why: preserve the accepted evaluator evidence.
        files: code=missing.py
        STAGE_COMMIT_END
        """

        out = await solver._handle_pending_stage_commit_text(
            agent=agent,
            assistant_text=assistant_text,
        )

        assert out == "finalized"
        assert solver.pending_text_stage_commit is None
        ledger = solver.ledger_path.read_text(encoding="utf-8")
        assert "### S03" in ledger
        assert "FILES: none" in ledger
        events = (solver.log_dir / "lhr_events.jsonl").read_text(encoding="utf-8")
        assert "stage_commit_text_fallback_applied" in events
        assert "fallback_after_files_retry_exhausted" in events

    asyncio.run(_run())


def test_lhr_stage_commit_compact_context_skips_prior_stage_records() -> None:
    class Dummy(RunLoopMixin):
        pass

    dummy = Dummy()
    base_messages = [
        Message.user_message("You are solving one optimization task in a continuous REPL workspace.\nTask body"),
        Message.assistant_message(
            "STAGE_COMMIT_BEGIN\nstage_id: S01\nmetric: 1.0\nSTAGE_COMMIT_END\n\n"
            "bash/edit output:\n[stage append-only write]\nstatus: ok",
        ),
        Message.tool_message("recent solver output " + ("x" * 6000), "bash", "tool-call-1"),
        Message.assistant_message("recent design note: local repair improved the candidate"),
    ]

    compact = dummy._build_lnr_stage_commit_compact_messages(
        base_messages=base_messages,
        transient_user_prompt="[LNR_STAGE_COMMIT_REQUEST]\nFACTS:\nstage_id: S02",
    )
    text = "\n".join(str(msg.content or "") for msg in compact)

    assert "[LNR_STAGE_COMMIT_TASK_CONTEXT]" in text
    assert "[LNR_STAGE_COMMIT_RECENT_CONTEXT]" in text
    assert "recent design note" in text
    assert "[stage append-only write]" not in text
    assert "STAGE_COMMIT_BEGIN\nstage_id: S01" not in text
    assert len(text) < 12000


def test_lhr_stage_commit_transient_prompt_disables_tools(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    agent = SimpleNamespace()

    solver._set_stage_commit_transient_prompt(
        agent,
        stage_id="S01",
        metric_event={
            "metric_value": 0.123,
            "metric_name": "Final Validation Score",
            "metric_validity": "medium",
            "lower_is_better": False,
        },
    )

    assert "STAGE_COMMIT_BEGIN" in agent._lnr_transient_user_prompt
    assert agent._lnr_transient_tool_choice_none is True
    assert agent._lnr_transient_context_mode == "stage_commit_compact"
    assert agent._lnr_stage_commit_text_pending is True
    assert agent._lnr_stage_commit_text_handled is False

    solver._clear_stage_commit_transient_prompt(agent)

    assert agent._lnr_transient_user_prompt == ""
    assert agent._lnr_transient_tool_choice_none is False
    assert agent._lnr_transient_context_mode == ""
    assert agent._lnr_stage_commit_text_pending is False


def test_lhr_stage_commit_transient_prompt_can_inherit_context_and_keep_tool_prefix(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        stage_commit_context_mode="inherit",
        stage_commit_tool_choice="auto",
    )
    agent = SimpleNamespace()

    solver._set_stage_commit_transient_prompt(
        agent,
        stage_id="S01",
        metric_event={"metric_value": 0.123, "metric_validity": "medium"},
    )

    assert agent._lnr_transient_context_mode == ""
    assert agent._lnr_transient_tool_choice_none is False
    assert agent._lnr_stage_commit_text_pending is True


def test_lhr_stage_commit_transient_prompt_rejects_unknown_modes(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_tool_choice="required")
    agent = SimpleNamespace()

    with pytest.raises(ValueError, match="stage_commit_tool_choice"):
        solver._set_stage_commit_transient_prompt(
            agent,
            stage_id="S01",
            metric_event={"metric_value": 0.123},
        )

    assert not hasattr(agent, "_lnr_transient_user_prompt")
    assert not hasattr(agent, "_lnr_transient_tool_choice_none")


def test_lhr_abandon_pending_stage_commit_clears_live_agent_state(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.pending_text_stage_commit = {"stage_id": "S03"}
    agent = SimpleNamespace(
        _lnr_transient_user_prompt="stale stage prompt",
        _lnr_transient_tool_choice_none=True,
        _lnr_transient_context_mode="stage_commit_compact",
        _lnr_transient_user_prompt_active=True,
        _lnr_stage_commit_text_pending=True,
        _lnr_stage_commit_text_handled=True,
        _lnr_suppress_current_text_only_memory="stale",
    )

    solver._abandon_pending_stage_commit_text(agent)

    assert solver.pending_text_stage_commit is None
    assert agent._lnr_transient_user_prompt == ""
    assert agent._lnr_transient_tool_choice_none is False
    assert agent._lnr_transient_context_mode == ""
    assert agent._lnr_transient_user_prompt_active is False
    assert agent._lnr_stage_commit_text_pending is False
    assert agent._lnr_stage_commit_text_handled is False
    assert agent._lnr_suppress_current_text_only_memory == ""


def test_lhr_stage_commit_json_prompt_is_opt_in(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        stage_commit_context_mode="inherit",
        stage_commit_tool_choice="auto",
        stage_commit_output_format="json",
    )

    prompt = solver._build_stage_commit_text_prompt(
        stage_id="S01",
        metric_event={
            "metric_value": 0.123,
            "metric_validity": "high",
            "metric_source_note": "official evaluator",
            "lower_is_better": False,
            "run_time_sec": 12.3,
        },
    )

    assert "Return exactly one fenced JSON object and no additional text" in prompt
    assert "```json" in prompt
    assert '"stage_id": "S01"' in prompt
    assert '"lower_is_better": false' in prompt
    assert "STAGE_COMMIT_BEGIN" not in prompt


def test_lhr_stage_commit_experiment_state_uses_only_structured_query_fields(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver._task_metric_lower_is_better = True
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)
    metric_event = {
        "metric_value": 0.06,
        "metric_validity": "high",
        "lower_is_better": True,
        "selection_eligible": True,
        "metric_source_note": "untrusted text says queries_remaining=99/99",
        "extra": {
            "queries_remaining": 7,
            "queries_used": 3,
            "query_limit": 10,
        },
    }

    state = solver._build_stage_commit_experiment_state(
        stage_id="S03",
        metric_event=metric_event,
    )

    assert "global_best_stage: W00:L01:S03" in state
    assert "global_best_metric: 0.06" in state
    assert "queries_remaining: 7" in state
    assert "queries_used: 3" in state
    assert "query_limit: 10" in state
    assert "99/99" not in state


def test_lhr_stage_commit_experiment_state_keeps_abandoned_lineage_global_best(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.global_log_dir = tmp_path / "task_logs"
    solver.global_log_dir.mkdir()
    solver._task_metric_lower_is_better = False
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)
    history_path = solver.global_log_dir / lnr_solver_module.LHR_STAGE_PERFORMANCE_CSV
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "worker_id",
                "candidate_id",
                "stage_id",
                "lineage_id",
                "metric_value",
                "validation_ok",
                "metric_validity",
                "selection_eligible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "worker_id": "W00",
                "candidate_id": "W00:L01:S04",
                "stage_id": "S04",
                "lineage_id": "L01",
                "metric_value": 0.91,
                "validation_ok": "true",
                "metric_validity": "high",
                "selection_eligible": "true",
            }
        )

    state = solver._build_stage_commit_experiment_state(
        stage_id="S03",
        metric_event={
            "metric_value": 0.80,
            "metric_validity": "high",
            "lower_is_better": False,
            "selection_eligible": True,
        },
    )

    assert "global_best_stage: W00:L01:S04" in state
    assert "global_best_metric: 0.91" in state


def test_lhr_stage_commit_experiment_state_does_not_readd_rejected_active_stage(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.global_log_dir = tmp_path / "task_logs"
    solver.global_log_dir.mkdir()
    solver._task_metric_lower_is_better = False
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)
    solver.stage_snapshots = {
        "S01": SimpleNamespace(
            node_uid="W00:L01:S01",
            source_event={
                "validation_ok": False,
                "selection_eligible": False,
                "metric_validity": "low",
            },
        )
    }
    history_path = solver.global_log_dir / lnr_solver_module.LHR_STAGE_PERFORMANCE_CSV
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "worker_id",
                "candidate_id",
                "stage_id",
                "metric_value",
                "validation_ok",
                "metric_validity",
                "selection_eligible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "worker_id": "W00",
                "candidate_id": "W00:L01:S01",
                "stage_id": "S01",
                "metric_value": 0.99,
                "validation_ok": "false",
                "metric_validity": "low",
                "selection_eligible": "false",
            }
        )

    state = solver._build_stage_commit_experiment_state(
        stage_id="S03",
        metric_event={
            "metric_value": 0.06,
            "metric_validity": "high",
            "lower_is_better": False,
            "selection_eligible": True,
        },
    )

    assert "global_best_stage: S02" in state
    assert "global_best_metric: 0.08" in state
    assert "global_best_metric: 0.99" not in state


def test_lhr_stage_commit_experiment_state_does_not_repeat_committed_stage_as_pending(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver._task_metric_lower_is_better = True
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)

    state = solver._build_stage_commit_experiment_state(
        stage_id="S02",
        metric_event={
            "metric_value": 0.08,
            "metric_validity": "high",
            "lower_is_better": True,
            "selection_eligible": True,
        },
    )

    assert "current stage pending summary" not in state
    assert state.count("- S02:") == 1


def test_lhr_stage_commit_experiment_state_default_prompt_stays_compact(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=False)

    prompt = solver._build_stage_commit_text_prompt(
        stage_id="S03",
        metric_event={"metric_value": 0.06, "lower_is_better": True},
    )

    assert "EXPERIMENT_STATE" not in prompt
    assert "brief: <one compact judgment sentence>" in prompt
    assert "only your STAGE_COMMIT block and the append confirmation" in prompt
    assert "files: <code=core.py,helper.py weights=model.ckpt" in prompt
    assert "FILES guidance" not in prompt



def test_lhr_resource_control_agents_use_feedback_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = object.__new__(LnrSolver)
        feedback_stage = SimpleNamespace(model="feedback-model")
        code_stage = SimpleNamespace(model="code-model")
        solver.cfg = SimpleNamespace(agent=SimpleNamespace(code=code_stage, feedback=feedback_stage), exp_id="unit")
        solver.lhr = SimpleNamespace(
            resource_arbiter_enabled=True,
            resource_arbiter_mode="llm",
            resource_admission_llm_enabled=True,
        )
        solver.deadline = 999999999.0
        solver.global_log_dir = tmp_path / "global_logs"
        solver._resource_arbiter_agent = None
        solver._resource_admission_agent = None
        created_overrides = []

        class FakeAgent:
            async def run_ephemeral_agentic_route_prompt(self, prompt, *, trigger, base_messages=None):
                if trigger == "resource_arbiter":
                    return '{"action":"DENY_KILL","confidence":"high","reason":"unit"}'
                return '{"action":"RUN_NOW","confidence":"high","reason":"unit"}'

        class FakeOrchestrator:
            @staticmethod
            def make_llm_call_tracer(**kwargs):
                assert "llm_role=feedback" in kwargs.get("detail_prefix", "")
                return lambda payload: None

            @staticmethod
            def create_science_agent(**kwargs):
                created_overrides.append(kwargs.get("llm_stage_override"))
                return FakeAgent()

        solver.orchestrator = FakeOrchestrator()
        arbiter_decider = solver._make_resource_arbiter_decider()
        admission_decider = solver._make_resource_admission_decider()

        arbiter_result = await arbiter_decider({"proposal_id": "p1", "proposal_type": "kill_proposal"})
        admission_result = await admission_decider(
            {"task_id": "t1", "resource_class": "gpu_tt_light"},
            {"action": "RUN_NOW", "lease_grantable_by_llm": True},
        )

        assert arbiter_result["action"] == "DENY_KILL"
        assert "RUN_NOW" in admission_result
        assert created_overrides == [feedback_stage, feedback_stage]

    asyncio.run(_run())
