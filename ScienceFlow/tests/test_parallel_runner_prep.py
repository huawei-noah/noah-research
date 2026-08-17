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

"""ParallelRunner manifest validation and prep vs run command shape (no subprocess)."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

import scienceflow.core.parallel_runner as parallel_runner_mod
from scienceflow.utils.llm_config_summary import (
    summarize_llm_config_from_agent_patch,
    summarize_llm_config_from_env,
)

from scienceflow.core.parallel_runner import (
    ParallelRunner,
    TaskResult,
    TaskSpec,
    _env_sticky_primary_index,
    _manifest_merge_lnr,
    _manifest_run_type,
    _manifest_input_data_dir,
    _manifest_scienceflow_interaction_log_full,
    _manifest_scienceflow_interaction_log_level,
    _manifest_scienceflow_interaction_log_llm_stream,
    _manifest_task_exp_id,
    _manifest_task_run_id,
    _parallel_failure_kind_from_text,
    _resolve_parallel_task_text,
    _resolve_task_workspace,
    _safe_filename,
    _validate_parallel_manifest,
    _workspace_lnr_failure_kind,
    _workspace_lnr_result_status,
    resolve_manifest_task_workspace,
)


def test_parallel_resume_budget_uses_charged_elapsed_sec(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"failed","elapsed_sec":1200,"charged_elapsed_sec":0,"failure_kind":"llm_quota_error"}',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), time_limit=3600)
    runner = object.__new__(ParallelRunner)

    assert ParallelRunner._check_resume(runner, spec) == 3600


def test_parallel_resume_budget_uses_lnr_wall_clock_and_patches_child_budget(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"failed","elapsed_sec":1200,"charged_elapsed_sec":1200}',
        encoding="utf-8",
    )
    spec = TaskSpec(
        exp_id="e",
        task="t",
        workspace=str(workspace),
        time_limit=3700,
        lnr_patch={"wall_clock_budget_sec": 3600, "max_steps": 20},
    )
    runner = object.__new__(ParallelRunner)

    remaining = ParallelRunner._check_resume(runner, spec)
    resumed = ParallelRunner._with_resume_remaining_budget(spec, remaining or 0)

    assert remaining == 2400
    assert resumed.time_limit == 2400
    assert resumed.lnr_patch == {"wall_clock_budget_sec": 2400, "max_steps": 20}


def test_parallel_resume_state_preserves_cumulative_charged_elapsed(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"failed","elapsed_sec":1200,"charged_elapsed_sec":1200}',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=2400)
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_running_state(runner, spec)
    running = __import__("json").loads((logs / "state.json").read_text(encoding="utf-8"))
    assert running["charged_elapsed_sec"] == 1200.0
    assert running["resume_prior_charged_elapsed_sec"] == 1200.0
    assert running["resume_total_budget_sec"] == 3600.0

    result = TaskResult(exp_id="e", run_id="r", status="success", exit_code=0, elapsed_sec=10.0)
    ParallelRunner._write_state(runner, spec, result)
    finished = __import__("json").loads((logs / "state.json").read_text(encoding="utf-8"))
    assert finished["charged_elapsed_sec"] == 1210.0
    assert finished["resume_prior_charged_elapsed_sec"] == 1200.0


def test_parallel_interrupted_state_marks_stopped_by_user(tmp_path, monkeypatch):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        (
            '{"status":"running","run_started_at":1000,'
            '"charged_elapsed_sec":120,"resume_prior_charged_elapsed_sec":120,'
            '"llm_config":{"code":{"model":"glm-5.2"}}}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(parallel_runner_mod.time, "time", lambda: 1015.5)
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_interrupted_state(runner, spec, error="interrupted by user")

    state = __import__("json").loads((logs / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "stopped_by_user"
    assert state["elapsed_sec"] == 15.5
    assert state["segment_elapsed_sec"] == 15.5
    assert state["charged_elapsed_sec"] == 135.5
    assert state["resume_prior_charged_elapsed_sec"] == 120.0
    assert state["resume_total_budget_sec"] == 3720.0
    assert state["error"] == "interrupted by user"
    assert state["llm_config"]["code"]["model"] == "glm-5.2"


def test_parallel_resume_zero_remaining_writes_skipped_state(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"failed","elapsed_sec":3600,"charged_elapsed_sec":3600,"resume_prior_charged_elapsed_sec":3600}',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    runner = object.__new__(ParallelRunner)
    runner._manifest = {"resume": True}
    runner.log_dir = None

    result = asyncio.run(ParallelRunner._run_task(runner, spec, asyncio.Semaphore(1)))

    assert result.status == "skipped"
    assert result.log_file == str(logs / "parallel_subprocess.log")
    state = __import__("json").loads((logs / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "skipped"
    assert state["charged_elapsed_sec"] == 3600.0
    assert state["time_limit_sec"] == 0


def test_parallel_write_running_state_records_sanitized_llm_config(tmp_path):
    workspace = tmp_path / "task"
    log_file = tmp_path / "run.log"
    llm_config = summarize_llm_config_from_env(
        {
            "CODE_MODEL": "glm-5.2",
            "API_KEY": "sk-secret",
            "BASE_URL": "https://user:pass@llm.example/v1?token=secret",
            "SCIENCEFLOW_LLM_ROUTING_MODE": "sticky_failover",
            "SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX": "2",
        },
        source="test",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_running_state(runner, spec, llm_config=llm_config, log_file=str(log_file))

    raw = (workspace / "task_logs" / "state.json").read_text(encoding="utf-8")
    assert "sk-secret" not in raw
    assert "pass" not in raw
    assert "token=secret" not in raw
    state = __import__("json").loads(raw)
    assert state["status"] == "running"
    assert state["log_file"] == str(log_file)
    assert state["llm_config"]["code"]["model"] == "glm-5.2"
    assert state["llm_config"]["code"]["key_count"] == 1
    assert state["llm_config"]["code"]["base_url"] == "https://llm.example/v1"
    assert state["llm_config"]["code"]["routing_mode"] == "sticky_failover"
    assert state["llm_config"]["code"]["sticky_primary_index"] == "2"


def test_parallel_agent_patch_llm_config_summary_expands_env(monkeypatch):
    monkeypatch.setenv("NF_TEST_CODE_KEY", "sk-secret")
    monkeypatch.setenv("NF_TEST_CODE_URL", "https://user:pass@llm.example/v1?token=secret")

    summary = summarize_llm_config_from_agent_patch(
        {
            "code": {
                "models": ["model-a", "model-b"],
                "api_keys": ["${NF_TEST_CODE_KEY}", "key-b"],
                "base_urls": ["${NF_TEST_CODE_URL}", "https://b.example/v1"],
                "api_routing_mode": "sticky_failover",
            },
            "feedback": {"model": "judge-a"},
        },
        source="test",
    )

    assert summary["code"]["models"] == ["model-a", "model-b"]
    assert summary["code"]["key_count"] == 2
    assert summary["code"]["base_url"] == "https://llm.example/v1"
    assert summary["code"]["base_url_count"] == 2
    assert summary["code"]["routing_mode"] == "sticky_failover"
    assert summary["feedback"]["model"] == "judge-a"
    assert "sk-secret" not in str(summary)
    assert "pass" not in str(summary)
    assert "token=secret" not in str(summary)


def test_parallel_write_running_state_clears_terminal_failure_fields(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"failed","failure_kind":"worker_timeout","resume_retriable":false,"resume_budget_policy":"external_llm_failure_not_charged"}',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_running_state(runner, spec)

    state = __import__("json").loads((logs / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running"
    assert "failure_kind" not in state
    assert "resume_retriable" not in state
    assert "resume_budget_policy" not in state


def test_parallel_write_state_marks_llm_api_failure_resumeable(tmp_path):
    workspace = tmp_path / "task"
    log_file = tmp_path / "run.log"
    log_file.write_text("Result: {'status': 'failed', 'stop_reason': 'llm_quota_error'}\n", encoding="utf-8")
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    result = TaskResult(
        exp_id="e",
        run_id="r",
        status="failed",
        exit_code=1,
        elapsed_sec=1200.0,
        log_file=str(log_file),
    )
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_state(runner, spec, result)

    state = __import__("json").loads((workspace / "task_logs" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["failure_kind"] == "llm_quota_error"
    assert state["resume_retriable"] is True
    assert state["charged_elapsed_sec"] == 0.0
    assert state["resume_budget_policy"] == "external_llm_failure_not_charged"


def test_parallel_write_state_charges_non_external_failures(tmp_path):
    workspace = tmp_path / "task"
    log_file = tmp_path / "run.log"
    log_file.write_text("ordinary process failure\n", encoding="utf-8")
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    result = TaskResult(
        exp_id="e",
        run_id="r",
        status="failed",
        exit_code=1,
        elapsed_sec=123.0,
        log_file=str(log_file),
    )
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_state(runner, spec, result)

    state = __import__("json").loads((workspace / "task_logs" / "state.json").read_text(encoding="utf-8"))
    assert "failure_kind" not in state
    assert state["charged_elapsed_sec"] == 123.0


def test_parallel_write_state_does_not_infer_failure_kind_for_success(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "lhr_events.jsonl").write_text(
        '{"event":"multi_worker_done","payload":{"status":"success","stop_reason":"budget_expired"}}\n',
        encoding="utf-8",
    )
    log_file = tmp_path / "run.log"
    log_file.write_text('{"timeout_sec": 3500, "stop_reason": "budget_expired"}\n', encoding="utf-8")
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=3600)
    result = TaskResult(
        exp_id="e",
        run_id="r",
        status="success",
        exit_code=0,
        elapsed_sec=3600.0,
        log_file=str(log_file),
    )
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_state(runner, spec, result)

    state = __import__("json").loads((workspace / "task_logs" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["stop_reason"] == "budget_expired"
    assert "failure_kind" not in state
    assert "resume_retriable" not in state


def test_lnr_parallel_state_and_gpu_assignment_use_task_logs(tmp_path):
    workspace = tmp_path / "task"
    spec = TaskSpec(
        exp_id="e",
        task="t",
        workspace=str(workspace),
        run_id="r",
        time_limit=3600,
        gpu_auto=True,
        gpu_list_raw="auto",
    )
    result = TaskResult(exp_id="e", run_id="r", status="success", exit_code=0, elapsed_sec=12.0)
    runner = object.__new__(ParallelRunner)

    parallel_runner_mod._write_gpu_assignment_json(spec, "0")
    ParallelRunner._write_state(runner, spec, result, resolved_gpu="0")

    assert (workspace / "task_logs" / "gpu_assignment.json").is_file()
    assert (workspace / "task_logs" / "state.json").is_file()
    assert not (workspace / "logs").exists()

def test_parallel_failure_kind_detects_402_text() -> None:
    assert _parallel_failure_kind_from_text("APIStatusError: Error code: 402 - Insufficient Balance") == "llm_quota_error"


def test_parallel_failure_kind_detects_budget_exhaustion_text() -> None:
    assert _parallel_failure_kind_from_text("time budget exhausted after 43158s") == "time_budget_expired"


def test_parallel_write_state_records_budget_done_as_stop_reason(tmp_path):
    workspace = tmp_path / "task"
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), run_id="r", time_limit=10)
    result = TaskResult(
        exp_id="e",
        run_id="r",
        status="budget_done",
        error="time budget exhausted after 10s",
        elapsed_sec=10.0,
    )
    runner = object.__new__(ParallelRunner)

    ParallelRunner._write_state(runner, spec, result)

    state = __import__("json").loads((workspace / "task_logs" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "budget_done"
    assert state["stop_reason"] == "time_budget_expired"
    assert "failure_kind" not in state
    assert "resume_retriable" not in state


def test_parallel_failure_kind_detects_real_llm_resume_errors() -> None:
    assert _parallel_failure_kind_from_text("AuthenticationError: Error code: 401 - invalid token") == "llm_api_error"
    assert _parallel_failure_kind_from_text("BadRequestError: reasoning_content must be passed back") == "llm_api_error"
    assert _parallel_failure_kind_from_text("PermissionDeniedError: token quota is not enough") == "llm_quota_error"


def test_parallel_resume_remaining_policy_charges_prior_elapsed(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"failed","elapsed_sec":1200,"charged_elapsed_sec":1200}',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), time_limit=3600)
    runner = object.__new__(ParallelRunner)

    assert ParallelRunner._check_resume(runner, spec) == 2400


def test_parallel_resume_fresh_policy_ignores_prior_elapsed(tmp_path):
    workspace = tmp_path / "task"
    logs = workspace / "task_logs"
    logs.mkdir(parents=True)
    (logs / "state.json").write_text(
        '{"status":"timeout","elapsed_sec":1200,"charged_elapsed_sec":1200}',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), time_limit=3600)
    runner = object.__new__(ParallelRunner)
    runner._resume_budget_policy = "fresh"

    assert ParallelRunner._check_resume(runner, spec) == 3600


def test_parallel_resume_does_not_skip_old_completed_state_when_child_lnr_failed(tmp_path):
    workspace = tmp_path / "task"
    task_logs = workspace / "task_logs"
    task_logs.mkdir(parents=True)
    (task_logs / "state.json").write_text(
        '{"status":"completed","elapsed_sec":900}',
        encoding="utf-8",
    )
    (task_logs / "lhr_events.jsonl").write_text(
        '{"event":"multi_worker_done","payload":{"status":"failed","stop_reason":"llm_quota_error"}}\n',
        encoding="utf-8",
    )
    spec = TaskSpec(exp_id="e", task="t", workspace=str(workspace), time_limit=3600)
    runner = object.__new__(ParallelRunner)

    assert _workspace_lnr_result_status(str(workspace)) == "failed"
    assert _workspace_lnr_failure_kind(str(workspace)) == "llm_quota_error"
    assert ParallelRunner._check_resume(runner, spec) == 3600


def test_manifest_input_data_dir_prefers_canonical_key() -> None:
    d: dict = {"input_data_dir": "/a", "data_dir": "/b"}
    t: dict = {"input_data_dir": "/x", "data_dir": "/y"}
    assert _manifest_input_data_dir(d, t) == "/x"


def test_manifest_input_data_dir_legacy_task_data_dir() -> None:
    d: dict = {}
    t: dict = {"data_dir": "/legacy"}
    assert _manifest_input_data_dir(d, t) == "/legacy"


def test_manifest_input_data_dir_defaults() -> None:
    d: dict = {"input_data_dir": "/def"}
    t: dict = {}
    assert _manifest_input_data_dir(d, t) == "/def"


def test_manifest_scienceflow_interaction_log_full_task_overrides_defaults() -> None:
    assert _manifest_scienceflow_interaction_log_full({}, {}) is None
    assert _manifest_scienceflow_interaction_log_full({"scienceflow_interaction_log_full": True}, {}) is True
    assert _manifest_scienceflow_interaction_log_full({"scienceflow_interaction_log_full": False}, {}) is False
    assert _manifest_scienceflow_interaction_log_full(
        {"scienceflow_interaction_log_full": True},
        {"scienceflow_interaction_log_full": False},
    ) is False


def test_manifest_scienceflow_interaction_log_level_task_overrides_defaults() -> None:
    assert _manifest_scienceflow_interaction_log_level({}, {}) is None
    assert _manifest_scienceflow_interaction_log_level({}, {"scienceflow_interaction_log_level": "verbose"}) == "verbose"
    assert _manifest_scienceflow_interaction_log_level(
        {"scienceflow_interaction_log_level": "normal"},
        {"scienceflow_interaction_log_level": "minimal"},
    ) == "minimal"


def test_manifest_scienceflow_interaction_log_llm_stream_task_overrides_defaults() -> None:
    assert _manifest_scienceflow_interaction_log_llm_stream({}, {}) is None
    assert _manifest_scienceflow_interaction_log_llm_stream(
        {"scienceflow_interaction_log_llm_stream": True},
        {},
    ) is True
    assert _manifest_scienceflow_interaction_log_llm_stream(
        {"scienceflow_interaction_log_llm_stream": False},
        {},
    ) is False
    assert _manifest_scienceflow_interaction_log_llm_stream(
        {"scienceflow_interaction_log_llm_stream": True},
        {"scienceflow_interaction_log_llm_stream": False},
    ) is False


def test_validate_prep_requires_input_data_dir() -> None:
    spec = TaskSpec(
        exp_id="t1",
        task="desc",
        workspace="/tmp/w",
        phase="prep",
        input_data_dir="",
    )
    with pytest.raises(ValueError, match="input_data_dir"):
        _validate_parallel_manifest([spec])


def test_validate_prep_input_data_dir_must_exist(tmp_path) -> None:
    missing = tmp_path / "nope"
    spec = TaskSpec(
        exp_id="t1",
        task="desc",
        workspace=str(tmp_path / "w"),
        phase="prep",
        input_data_dir=str(missing),
    )
    with pytest.raises(ValueError, match="not an existing directory"):
        _validate_parallel_manifest([spec])


def test_validate_prep_ok(tmp_path) -> None:
    data = tmp_path / "in"
    data.mkdir()
    spec = TaskSpec(
        exp_id="t1",
        task="desc",
        workspace=str(tmp_path / "w"),
        phase="prep",
        input_data_dir=str(data),
    )
    _validate_parallel_manifest([spec])


def test_validate_run_ignores_missing_input_data_dir(tmp_path) -> None:
    spec = TaskSpec(
        exp_id="t1",
        task="desc",
        workspace=str(tmp_path / "w"),
        phase="run",
        input_data_dir="",
    )
    _validate_parallel_manifest([spec])


def test_validate_rejects_removed_draft_phase(tmp_path) -> None:
    spec = TaskSpec(
        exp_id="t1",
        task="desc",
        workspace=str(tmp_path / "w"),
        phase="draft",
        input_data_dir="",
    )
    with pytest.raises(ValueError, match="invalid phase"):
        _validate_parallel_manifest([spec])


def test_validate_invalid_phase() -> None:
    spec = TaskSpec(
        exp_id="t1",
        task="desc",
        workspace="/tmp/w",
        phase="train",
    )
    with pytest.raises(ValueError, match="invalid phase"):
        _validate_parallel_manifest([spec])


def test_resolve_parallel_task_inline_overrides_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scienceflow.core.parallel_runner._scienceflow_repo_root",
        lambda: tmp_path,
    )
    desc = tmp_path / "tasks" / "ml" / "mlebench" / "mytask" / "description_lite.md"
    desc.parent.mkdir(parents=True)
    desc.write_text("from_file", encoding="utf-8")
    (desc.parent / "task.yaml").write_text("id: mytask\ndescription: description_lite.md\n", encoding="utf-8")
    assert _resolve_parallel_task_text("mytask", "inline body") == "inline body"
    assert _resolve_parallel_task_text("mytask", "  spaced  ") == "spaced"


def test_resolve_parallel_task_default_description_lite(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scienceflow.core.parallel_runner._scienceflow_repo_root",
        lambda: tmp_path,
    )
    desc = tmp_path / "tasks" / "ml" / "mlebench" / "mytask" / "description_lite.md"
    desc.parent.mkdir(parents=True)
    desc.write_text("from_file\n", encoding="utf-8")
    (desc.parent / "task.yaml").write_text("id: mytask\ndescription: description_lite.md\n", encoding="utf-8")
    assert _resolve_parallel_task_text("mytask", None) == "from_file\n"
    assert _resolve_parallel_task_text("mytask", "") == "from_file\n"


def test_resolve_parallel_task_missing_default_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scienceflow.core.parallel_runner._scienceflow_repo_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError, match="default description file"):
        _resolve_parallel_task_text("missing", None)


def test_resolve_parallel_task_rejects_ambiguous_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scienceflow.core.parallel_runner._scienceflow_repo_root",
        lambda: tmp_path,
    )
    for category in ("ml/mlebench", "opt_solver"):
        desc = tmp_path / "tasks" / category / "dupe" / "description_lite.md"
        desc.parent.mkdir(parents=True)
        desc.write_text(category, encoding="utf-8")
        (desc.parent / "task.yaml").write_text("id: dupe\ndescription: description_lite.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous task package"):
        _resolve_parallel_task_text("dupe", None)


def test_manifest_task_exp_id_accepts_legacy_name_key() -> None:
    with pytest.warns(DeprecationWarning, match="name"):
        assert _manifest_task_exp_id({"name": "legacy-slug"}, 0) == "legacy-slug"


def test_parallel_runner_accepts_legacy_manifest_data_dir_key(tmp_path) -> None:
    indir = tmp_path / "prepared_public"
    indir.mkdir()
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            tasks:
              - exp_id: leg
                task: "t"
                workspace: "{tmp_path / "ws"}"
                phase: prep
                data_dir: "{indir}"
            """
        ).strip(),
        encoding="utf-8",
    )
    r = ParallelRunner(manifest)
    assert len(r._tasks) == 1
    assert r._tasks[0].input_data_dir == str(indir)


def test_parallel_manifest_ignores_removed_prep_config(tmp_path) -> None:
    indir = tmp_path / "in"
    indir.mkdir()
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            defaults:
              config: scienceflow/config/default.yaml
              phase: prep
              prep_val_fraction: 0.1
              data_split:
                mode: enabled
            tasks:
              - exp_id: x
                task: "t"
                workspace: "{tmp_path / "ws"}"
                input_data_dir: "{indir}"
                prep_val_fraction: 0.2
                data_split:
                  mode: required
            """
        ).strip(),
        encoding="utf-8",
    )
    r = ParallelRunner(manifest)
    spec = r._tasks[0]
    assert "prep_val_fraction" not in (spec.manifest_cfg_patch or {})
    assert "data_split" not in (spec.manifest_cfg_patch or {})


def test_parallel_child_cli_argv_lnr_run_type(tmp_path) -> None:
    from scienceflow.core.parallel_runner import ParallelRunner, TaskSpec

    ws = str(tmp_path / "ws")
    spec = TaskSpec(
        exp_id="e",
        task="hello",
        workspace=ws,
        type="lnr",
        phase="run",
    )
    argv = ParallelRunner.child_cli_argv(spec)
    assert argv[argv.index("--type") + 1] == "lnr"
    assert "--resume" not in argv


def test_parallel_child_cli_uses_resolved_project_python(monkeypatch, tmp_path) -> None:
    child_python = tmp_path / ".venv" / "bin" / "python"
    child_bin = child_python.parent
    monkeypatch.setattr(
        parallel_runner_mod,
        "resolve_project_python",
        lambda: (str(child_python), str(child_bin)),
    )
    spec = TaskSpec(
        exp_id="e",
        task="hello",
        workspace=str(tmp_path / "ws"),
        type="lnr",
        phase="run",
    )
    argv = ParallelRunner.child_cli_argv(spec)
    assert argv[:3] == [str(child_python), "-m", "scienceflow.cli"]


def test_parallel_manifest_loads_lnr_patch(tmp_path) -> None:
    indir = tmp_path / "in"
    indir.mkdir()
    ws = tmp_path / "ws"
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            lnr:
              max_steps: 21
            defaults:
              lnr:
                num_workers: 2
            tasks:
              - exp_id: ex
                run_id: r1
                workspace: "{ws}"
                input_data_dir: "{indir}"
                task: "x"
                lnr:
                  max_steps: 7
            """
        ).strip(),
        encoding="utf-8",
    )
    r = ParallelRunner(manifest)
    spec = r._tasks[0]
    assert spec.type == "lnr"
    assert spec.lnr_patch == {"max_steps": 7, "num_workers": 2}
    argv = ParallelRunner.child_cli_argv(spec)
    assert argv[argv.index("--type") + 1] == "lnr"


def test_manifest_merge_lnr_root_defaults_task() -> None:
    manifest = {"lnr": {"max_steps": 100, "num_workers": 1}}
    defaults = {"lnr": {"num_workers": 2}}
    task = {"lnr": {"max_steps": 10}}
    merged = _manifest_merge_lnr(manifest, defaults, task, 0, "e")
    assert merged == {"max_steps": 10, "num_workers": 2}


def test_manifest_merge_lnr_expands_resource_mode_per_layer() -> None:
    manifest = {"lnr": {"resource_control_mode": "resource_smart_llm"}}
    defaults = {"lnr": {"resource_control_mode": "resource_smart_policy"}}
    task = {"lnr": {"resource_control_mode": "off"}}

    merged = _manifest_merge_lnr(manifest, defaults, task, 0, "e")

    assert merged["resource_control_mode"] == "off"
    assert merged["resource_monitor_enabled"] is False
    assert merged["resource_runtime_enabled"] is False
    assert merged["resource_arbiter_enabled"] is False
    assert merged["resource_arbiter_mode"] == "policy"
    assert merged["resource_gpu_share_enabled"] is False


def test_manifest_merge_lnr_keeps_task_resource_expert_override() -> None:
    manifest = {"lnr": {"resource_control_mode": "resource_smart_llm"}}
    defaults = {"lnr": {}}
    task = {
        "lnr": {
            "resource_control_mode": "resource_smart_policy",
            "resource_admission_llm_enabled": True,
        }
    }

    merged = _manifest_merge_lnr(manifest, defaults, task, 0, "e")

    assert merged["resource_control_mode"] == "resource_smart_policy"
    assert merged["resource_arbiter_mode"] == "policy"
    assert merged["resource_gpu_share_enabled"] is False
    assert merged["resource_admission_llm_enabled"] is True


def test_manifest_run_type_accepts_lnr_only() -> None:
    assert _manifest_run_type({"type": "lnr"}, {}, 0, "e") == "lnr"
    with pytest.raises(ValueError, match="use 'lnr'"):
        _manifest_run_type({"type": "legacy_solver"}, {}, 0, "e")


def test_safe_filename_sanitizes_and_caps() -> None:
    assert _safe_filename("a/b") == "a_b"
    assert _safe_filename("nomad-run-1") == "nomad-run-1"
    assert _safe_filename("") == "_"


def test_manifest_task_run_id_defaults_and_explicit() -> None:
    assert _manifest_task_run_id({}, 0, "exp-slug") == "exp-slug"
    assert _manifest_task_run_id({"run_id": "r1"}, 0, "exp-slug") == "r1"
    assert _manifest_task_run_id({"run_id": "  "}, 0, "exp-slug") == "exp-slug"


def test_resolve_task_workspace_explicit_overrides_base(tmp_path) -> None:
    explicit = tmp_path / "explicit_ws"
    defaults = {"workspace_base": str(tmp_path / "ignored_base")}
    task = {"workspace": str(explicit), "workspace_base": str(tmp_path / "ignored_base")}
    assert _resolve_task_workspace(defaults, task, 0, "run-a", "exp-a") == str(explicit)


def test_resolve_task_workspace_from_base(tmp_path) -> None:
    base = tmp_path / "batch"
    defaults = {"workspace_base": str(base)}
    task: dict = {}
    ws = _resolve_task_workspace(defaults, task, 0, "run-a", "exp-a")
    assert ws == str(base / "run-a" / "exp-a")


def test_resolve_manifest_task_workspace_matches_helpers(tmp_path) -> None:
    base = tmp_path / "batch"
    defaults = {"workspace_base": str(base)}
    task = {"run_id": "r1", "exp_id": "e1"}
    exp_id = _manifest_task_exp_id(task, 0)
    assert resolve_manifest_task_workspace(defaults, task, 0, exp_id) == str(base / "r1" / "e1")


def test_validate_duplicate_run_id_raises() -> None:
    s1 = TaskSpec(exp_id="a", task="t", workspace="/w1", run_id="dup")
    s2 = TaskSpec(exp_id="b", task="t", workspace="/w2", run_id="dup")
    with pytest.raises(ValueError, match="Duplicate run_id"):
        _validate_parallel_manifest([s1, s2])


def test_validate_same_exp_id_distinct_run_id_ok() -> None:
    s1 = TaskSpec(exp_id="x", task="t", workspace="/w1", run_id="r1")
    s2 = TaskSpec(exp_id="x", task="t", workspace="/w2", run_id="r2")
    _validate_parallel_manifest([s1, s2])


def test_parallel_runner_duplicate_default_run_id_raises(tmp_path) -> None:
    """Two tasks with same exp_id and no explicit run_id → both run_id=exp_id → duplicate."""
    manifest = tmp_path / "m.yaml"
    w1 = tmp_path / "w1"
    w2 = tmp_path / "w2"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            tasks:
              - exp_id: same-exp
                task: "t"
                workspace: "{w1}"
              - exp_id: same-exp
                task: "t"
                workspace: "{w2}"
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate run_id"):
        ParallelRunner(manifest)


def test_parallel_runner_workspace_base_derives_path(tmp_path) -> None:
    base = tmp_path / "batch_root"
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            defaults:
              workspace_base: "{base}"
            tasks:
              - exp_id: myexp
                run_id: run-z
                task: "hello"
            """
        ).strip(),
        encoding="utf-8",
    )
    r = ParallelRunner(manifest)
    assert len(r._tasks) == 1
    assert r._tasks[0].workspace == str(base / "run-z" / "myexp")
    assert r._tasks[0].run_id == "run-z"


def test_parallel_runner_default_log_file_is_task_log(tmp_path) -> None:
    manifest = tmp_path / "m.yaml"
    workspace = tmp_path / "ws"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            tasks:
              - exp_id: exp-a
                run_id: run-a
                task: "hello"
                workspace: "{workspace}"
            """
        ).strip(),
        encoding="utf-8",
    )

    r = ParallelRunner(manifest)
    spec = r._tasks[0]

    assert r.log_dir is None
    assert r._task_subprocess_log_file(spec) == workspace / "task_logs" / "parallel_subprocess.log"


def test_parallel_runner_rejects_repo_parallel_logs_override(tmp_path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 1
            tasks:
              - exp_id: exp-a
                run_id: run-a
                task: "hello"
                workspace: "{tmp_path / "ws"}"
            """
        ).strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="parallel_logs"):
        ParallelRunner(manifest, log_dir=Path("parallel_logs") / "run-a")
    with pytest.raises(ValueError, match="parallel_logs"):
        ParallelRunner(manifest, log_dir=Path(".") / "parallel_logs" / "run-a")


def test_parallel_runner_top_level_api_keys_become_task_sticky_failover_pool(tmp_path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            max_concurrent: 2
            api_keys:
              - key-a
              - key-b
              - key-c
            base_url: https://llm.example/v1
            tasks:
              - exp_id: exp-a
                run_id: run-a
                task: "hello"
                workspace: "{tmp_path / "ws-a"}"
              - exp_id: exp-b
                run_id: run-b
                task: "hello"
                workspace: "{tmp_path / "ws-b"}"
            """
        ).strip(),
        encoding="utf-8",
    )
    r = ParallelRunner(manifest)
    first, second = r._tasks

    assert first.api_key == "key-a"
    assert first.api_keys == "key-a,key-b,key-c"
    assert first.base_url == "https://llm.example/v1"
    assert first.base_urls == ""
    assert first.llm_routing_mode == "sticky_failover"
    assert first.llm_sticky_id == "run-a"
    assert first.llm_sticky_primary_index == 0

    assert second.api_key == "key-b"
    assert second.api_keys == "key-b,key-c,key-a"
    assert second.base_url == "https://llm.example/v1"
    assert second.base_urls == ""
    assert second.llm_routing_mode == "sticky_failover"
    assert second.llm_sticky_id == "run-b"
    assert second.llm_sticky_primary_index == 0


def test_parallel_runner_inherited_env_key_pool_uses_task_index_primary() -> None:
    env = {"CODE_API_KEYS": "key-a,key-b,key-c"}

    assert _env_sticky_primary_index(env, 0) == 0
    assert _env_sticky_primary_index(env, 1) == 1
    assert _env_sticky_primary_index(env, 2) == 2
    assert _env_sticky_primary_index(env, 3) == 0


def test_parallel_runner_inherited_env_key_pool_prefers_code_pool() -> None:
    env = {
        "CODE_API_KEYS": "code-a,code-b",
        "API_KEYS": "generic-a,generic-b,generic-c",
    }

    assert _env_sticky_primary_index(env, 2) == 0


def test_parallel_runner_format_summary_uses_run_id() -> None:
    rows = [
        TaskResult(exp_id="e", run_id="run-a", status="success", elapsed_sec=1.0, exit_code=0),
        TaskResult(exp_id="e", run_id="", status="failed", elapsed_sec=0.0, exit_code=1),
    ]
    text = ParallelRunner.format_summary(rows)
    assert "run-a" in text
    assert "run_id" in text.splitlines()[0]
