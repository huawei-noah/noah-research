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

import json

import pytest

from scienceflow.core.tools.bash_tool import BashTool, _executed_resource_termination_feedback
from scienceflow.core.tools.resource_classifier import RESOURCE_PURE_TT_CPU
from scienceflow.solver.lnr.resource_runtime.source_hints import ResourceSourceHint
from tests.lnr_resource_test_utils import event_types, make_observer, payloads


def _heavy_job(observer, tmp_path) -> str:
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    job_id = observer.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def _resource_events(tmp_path) -> list[dict]:
    events_path = tmp_path / "resource" / "resource_events.jsonl"
    if not events_path.exists():
        return []
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_executed_kill_feedback_is_unambiguous() -> None:
    feedback = _executed_resource_termination_feedback(
        "Resource arbiter approved kill because the remaining budget is insufficient.\n"
        "RESOURCE_RESEARCH_SIGNAL: outcome=KILL; confidence=high.",
        action="KILL_AND_REPLAN",
    )

    assert feedback.startswith("The command has already been terminated")
    assert "Do not wait for it" in feedback
    assert "has already terminated this command" in feedback
    assert "approved kill because" not in feedback


def test_stalled_guard_decision_promotes_after_threshold(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=1.0,
        stalled_stdout_sec=0.5,
        review_state_enabled=False,
    )
    job_id = _heavy_job(observer, tmp_path)

    early = observer.stalled_guard_decision(job_id, elapsed_sec=0.2, stdout_age_sec=1.0)
    assert early["enabled"] is False
    assert early["terminate"] is False

    decision = observer.stalled_guard_decision(job_id, elapsed_sec=2.0, stdout_age_sec=1.0)
    assert decision["enabled"] is True
    assert decision["terminate"] is False
    assert decision["would_terminate"] is True
    assert decision["requires_llm_decision"] is True
    assert decision["reason"] == "stalled_stdout"
    assert decision["feedback"].startswith("RESOURCE_FEEDBACK: recommend_stop_command because")


def test_stalled_guard_delegates_to_state_machine_when_enabled(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=1.0,
        stalled_stdout_sec=0.5,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.stalled_guard_decision(job_id, elapsed_sec=2.0, stdout_age_sec=1.0)

    assert decision["enabled"] is False
    assert decision["terminate"] is False
    assert decision["would_terminate"] is False
    assert decision["reason"] == "state_machine_handles_stalled_stdout"


@pytest.mark.asyncio
async def test_bash_tool_terminates_stalled_heavy_command(tmp_path) -> None:
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.1,
        kill_mode="auto",
        review_state_enabled=False,
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=10.0,
        bash_timeout_slow_sec=10.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("python3 -c 'import time; time.sleep(5)' --epochs 1")

    assert result.error == "Resource guard terminated stalled heavy"
    assert "RESOURCE_FEEDBACK: terminated_stalled_command because" in (result.output or "")
    assert "resource_guard_action" in event_types(sm)


@pytest.mark.asyncio
async def test_bash_tool_arbiter_kill_without_advisory_observes_more(tmp_path) -> None:
    async def approve_kill(_proposal):
        return {"action": "KILL_AND_REPLAN", "reason": "repeated low progress with no useful output", "confidence": "high"}

    (tmp_path / "train.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=approve_kill,
        arbiter_min_progress_windows=1,
        review_state_enabled=False,
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=10.0,
        bash_timeout_slow_sec=10.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("python3 train.py --epochs 10")

    assert result.error is None
    assert "Resource arbiter approved kill" not in (result.output or "")
    actions = payloads(sm, "resource_guard_action")
    assert not any(row.get("action") == "terminate_by_arbiter" and row.get("executed") is True for row in actions)
    runtime_events = _resource_events(tmp_path)
    assert not any(event["event_type"] == "resource_kill_executed" for event in runtime_events)
    executions = [event for event in runtime_events if event["event_type"] == "execution"]
    assert not any((event.get("payload") or {}).get("execution_outcome") == "KILL" for event in executions)
    strict_gate_events = [
        event for event in executions
        if (event.get("payload") or {}).get("strict_gate_reason") == "owning_agent_advisory_missing"
    ]
    if strict_gate_events:
        assert all((event.get("payload") or {}).get("execution_outcome") == "NO_ACTION" for event in strict_gate_events)


@pytest.mark.asyncio
async def test_bash_tool_recommends_stalled_stop_without_auto_kill(tmp_path) -> None:
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.1,
        review_state_enabled=False,
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=2.0,
        bash_timeout_slow_sec=2.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("python3 -c 'import time; time.sleep(0.5)' --epochs 1")

    assert result.error is None
    actions = payloads(sm, "resource_guard_action")
    assert actions
    assert actions[0]["action"] == "NO_ACTION"
    assert actions[0]["raw_action"] == "recommend_stop"


def test_low_progress_guard_uses_progress_heartbeat_age(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1.0,
        stdout_age_sec=0.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )

    assert decision["terminate"] is False
    assert decision["would_terminate"] is True
    assert decision["requires_llm_decision"] is True
    assert decision["reason"] == "active_intervention:low_progress_heartbeat_stalled"
    assert decision["feedback"].startswith("RESOURCE_FEEDBACK: recommend_stop_command because")


def test_low_progress_guard_ignores_pure_tt_cpu_with_assigned_gpu_pool(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
    )
    job_id = observer.job_created(
        command="python3 predict.py",
        inferred_class=RESOURCE_PURE_TT_CPU,
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1.0,
        stdout_age_sec=0.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="validation_or_inference",
    )

    assert decision["terminate"] is False


def test_low_progress_guard_respects_final_score(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1.0,
        stdout_age_sec=0.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=True,
        current_phase="final_scoring",
    )

    assert decision["terminate"] is False


def test_idle_gpu_lease_guard_can_terminate_without_queue_pressure(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": str(gpu_ids[0]),
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 0.0,
                    "memory_total_mb": 81920.0,
                },
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        gpu_idle_lease_guard_enabled=True,
        kill_mode="auto",
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_min_samples=2,
        gpu_idle_lease_require_pressure=False,
        gpu_idle_lease_action_mode="terminate",
    )
    job_id = _heavy_job(observer, tmp_path)
    assert observer.queue_try_acquire(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    first = observer.active_intervention_decision(job_id, elapsed_sec=1.0, stdout_age_sec=0.0)
    observer._last_gpu_util_emit[job_id] = 0.0
    second = observer.active_intervention_decision(job_id, elapsed_sec=2.0, stdout_age_sec=0.0)

    assert first["terminate"] is False
    assert second["terminate"] is True
    assert second["reason"].startswith("active_intervention:idle_gpu_lease_under_pressure")
    assert "terminated_idle_gpu_lease because" in second["feedback"]



def test_dataloader_bottleneck_guard_can_terminate_low_gpu_high_cpu(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": str(gpu_ids[0]),
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 4096.0,
                    "memory_total_mb": 81920.0,
                },
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        gpu_dataloader_bottleneck_guard_enabled=True,
        gpu_dataloader_bottleneck_warmup_sec=0.0,
        gpu_dataloader_bottleneck_min_samples=2,
        gpu_dataloader_bottleneck_util_pct=15.0,
        gpu_dataloader_bottleneck_min_mem_gb=2.0,
        gpu_dataloader_bottleneck_child_cpu_pct=200.0,
        gpu_dataloader_bottleneck_busy_children=2,
        kill_mode="auto",
    )
    job_id = _heavy_job(observer, tmp_path)
    assert observer.queue_try_acquire(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    cpu = {"available": True, "child_cpu_pct": 390.0, "busy_child_count": 4}

    first = observer.active_intervention_decision(job_id, elapsed_sec=1.0, stdout_age_sec=0.0, process_tree_cpu=cpu)
    observer._last_gpu_util_emit[job_id] = 0.0
    second = observer.active_intervention_decision(job_id, elapsed_sec=2.0, stdout_age_sec=0.0, process_tree_cpu=cpu)

    assert first["terminate"] is False
    assert second["terminate"] is True
    assert second["reason"] == "active_intervention:dataloader_bottleneck_low_gpu_high_cpu"
    assert "terminated_dataloader_bottleneck because" in second["feedback"]


def test_kill_recommendation_writes_unified_resource_files(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1.0,
        stdout_age_sec=1.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )

    assert decision["would_terminate"] is True
    resource_dir = tmp_path / "resource"
    state_path = resource_dir / "resource_state.json"
    events_path = resource_dir / "resource_events.jsonl"
    assert state_path.is_file()
    assert events_path.is_file()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert state["active_proposals"]

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_types = [event["event_type"] for event in events]
    assert "snapshot" in event_types
    assert "kill_proposal" in event_types
    assert "arbiter_input" in event_types
    proposal = next(event for event in events if event["event_type"] == "kill_proposal")
    payload = proposal["payload"]
    assert payload["requires_llm_decision"] is True
    assert payload["suppressed"] is False
    assert payload["progress_snapshot"]["progress_signal"] == "stalled"
    assert payload["progress_snapshot"]["progress_confidence"] == "medium"
    assert payload["progress_snapshot"]["multi_window_low_progress"] is False
    assert payload["resource_snapshot"]["lease"]["resource_type"] == "gpu"


def test_deliverable_completion_guard_terminates_completed_submission_resource_hold(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n", encoding="utf-8")
    (tmp_path / "submission.csv").write_text("id,target\n1,0.1\n2,0.2\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="auto",
        deliverable_completion_guard_enabled=True,
        deliverable_completion_warmup_sec=0.0,
        deliverable_completion_settle_sec=0.0,
        deliverable_completion_scan_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=10.0,
        stdout_age_sec=0.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )

    assert decision["terminate"] is False
    state = observer._jobs[job_id].last_deliverable_completion_check["state"]
    assert state["complete"] is True
    assert state["mode"] == "submission"


def test_metric_health_guard_terminates_nan_or_inf_training_signal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="auto",
        metric_health_guard_enabled=True,
        metric_health_warmup_sec=0.0,
        metric_health_invalid_min_events=2,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=10.0,
        stdout_age_sec=0.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        invalid_metric_events=2,
        last_invalid_metric_text="val predictions contain NaN",
    )

    assert decision["terminate"] is True
    assert decision["reason"] == "active_intervention:invalid_training_metrics_nan_inf"
    assert "terminated_invalid_training_metrics" in decision["feedback"]
    assert decision["invalid_metric_events"] == 2


def test_metric_health_guard_terminates_zero_validation_score_after_progress(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="auto",
        metric_health_guard_enabled=True,
        metric_health_warmup_sec=0.0,
        metric_health_zero_score_min_events=2,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=10.0,
        stdout_age_sec=0.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        zero_score_events=2,
        last_zero_score_text="val_auc=0.0000",
    )

    assert decision["terminate"] is True
    assert decision["reason"] == "active_intervention:invalid_training_metrics_zero_score"
    assert "terminated_invalid_training_metrics" in decision["feedback"]
    assert decision["zero_score_events"] == 2




def test_idle_gpu_lease_release_keeps_cpu_only_process_alive_path(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 0.0,
                    "memory_total_mb": 81920.0,
                },
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    (tmp_path / "train.py").write_text("print('cpu xgboost style train')\n", encoding="utf-8")
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        gpu_idle_lease_guard_enabled=True,
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_min_samples=2,
        gpu_idle_lease_require_pressure=False,
        gpu_idle_lease_action_mode="release",
    )
    job_id = observer.job_created(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_train python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    assert observer.resource_runtime is not None
    acquired = observer.resource_runtime.queue_try_acquire(
        job_id=job_id,
        resource_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        metadata={"resource_class": "heavy_gpu_candidate", "command_digest": "test-cpu-only"},
    )
    assert acquired["acquired"] is True
    observer._jobs[job_id].gpu_queue_relevant = True
    observer._jobs[job_id].resource_class = "heavy_gpu_candidate"
    observer._jobs[job_id].source_hint = ResourceSourceHint(command_cpu_only=True, source_files_inspected=1)

    first = observer.active_intervention_decision(job_id, elapsed_sec=1.0, stdout_age_sec=0.0)
    observer._last_gpu_util_emit[job_id] = 0.0
    second = observer.active_intervention_decision(job_id, elapsed_sec=2.0, stdout_age_sec=0.0)

    assert first["terminate"] is False
    assert second["terminate"] is False
    assert second["release_idle_lease"] is True
    assert second["idle_gpu_release_action"] == "RELEASE_IDLE_LEASE"
    release = observer.release_idle_gpu_lease(job_id, elapsed_sec=2.0, reason=second["reason"], feedback=second["feedback"])
    assert release["released"] is True
    assert observer.resource_runtime is not None
    assert observer.resource_runtime.has_active_lease(job_id=job_id) is False
    released_idle = observer.resource_runtime.gpu_store.snapshot_active()["released_idle"]
    assert job_id in released_idle
    assert released_idle[job_id]["metadata"]["lease_mode"] == "released_idle"
    assert "resource_idle_lease_released" in event_types(sm)


def test_idle_gpu_lease_release_blocks_when_gpu_mem_not_small(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 4096.0,
                    "memory_total_mb": 81920.0,
                },
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        gpu_idle_lease_guard_enabled=True,
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_min_samples=2,
        gpu_idle_lease_require_pressure=False,
        gpu_idle_lease_action_mode="release",
    )
    job_id = observer.job_created(
        command="python3 support.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    assert observer.resource_runtime is not None
    acquired = observer.resource_runtime.queue_try_acquire(
        job_id=job_id,
        resource_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        metadata={"resource_class": "heavy_gpu_candidate", "command_digest": "test-mem-active"},
    )
    assert acquired["acquired"] is True
    observer._jobs[job_id].gpu_queue_relevant = True
    observer._jobs[job_id].resource_class = "heavy_gpu_candidate"
    observer._jobs[job_id].source_hint = ResourceSourceHint(command_cpu_only=True, source_files_inspected=1)

    observer.active_intervention_decision(job_id, elapsed_sec=1.0, stdout_age_sec=0.0)
    observer._last_gpu_util_emit[job_id] = 0.0
    decision = observer.active_intervention_decision(job_id, elapsed_sec=2.0, stdout_age_sec=0.0)

    assert decision["terminate"] is False
    assert decision.get("release_idle_lease") is not True
    observer._jobs[job_id].idle_gpu_lease_samples = 2
    safety = observer._idle_lease_release_safety(observer._jobs[job_id], observer._jobs[job_id].last_signal)
    assert safety["safe"] is False
    assert safety["reason"] == "gpu_mem_above_release_threshold"
    assert observer.resource_runtime.has_active_lease(job_id=job_id) is True


def test_idle_gpu_lease_release_allows_cuda_source_when_idle_small_mem(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 0.0,
                    "memory_total_mb": 81920.0,
                },
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    (tmp_path / "train.py").write_text("import torch\nmodel = torch.nn.Linear(1, 1).to('cuda')\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        gpu_idle_lease_guard_enabled=True,
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_min_samples=2,
        gpu_idle_lease_require_pressure=False,
        gpu_idle_lease_action_mode="release",
    )
    job_id = observer.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    assert observer.resource_runtime is not None
    acquired = observer.resource_runtime.queue_try_acquire(
        job_id=job_id,
        resource_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        metadata={"resource_class": "heavy_gpu_candidate", "command_digest": "test-cuda-source"},
    )
    assert acquired["acquired"] is True
    observer._jobs[job_id].gpu_queue_relevant = True

    observer.active_intervention_decision(job_id, elapsed_sec=1.0, stdout_age_sec=0.0)
    observer._last_gpu_util_emit[job_id] = 0.0
    decision = observer.active_intervention_decision(job_id, elapsed_sec=2.0, stdout_age_sec=0.0)

    assert decision["terminate"] is False
    assert decision.get("release_idle_lease") is True
    assert decision["idle_gpu_release_mode"] == "release"
    assert decision["idle_gpu_release_safety"]["safe"] is True
    assert decision["idle_gpu_release_safety"]["reason"] == "idle_small_mem_release_with_late_gpu_risk"
    assert decision["idle_gpu_release_safety"]["risk"] == "possible_late_gpu_use"
    assert decision["idle_gpu_release_safety"]["old_job_effect"] == "continues_with_existing_cuda_visible_devices"
    assert observer.resource_runtime is not None
    release = observer.release_idle_gpu_lease(job_id, elapsed_sec=2.0, reason=decision["reason"], feedback=decision["feedback"])
    assert release["released"] is True
    assert observer.resource_runtime.has_active_lease(job_id=job_id) is False
