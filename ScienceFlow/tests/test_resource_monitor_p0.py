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
import os
import re

import pytest

from scienceflow.core.tools.bash_tool import BashTool, _parse_progress_signals
from scienceflow.core.tools.resource_classifier import (
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    classify_bash_command,
)
from scienceflow.solver.lnr.resource_runtime.unified_store import (
    UnifiedResourceStore,
    format_resource_run_summary,
    summarize_resource_run,
)
from tests.lnr_resource_test_utils import event_types, make_observer, payloads


def test_parse_scienceflow_hb_progress_signal() -> None:
    signals = _parse_progress_signals(
        "SCIENCEFLOW_HB v=1 phase=train tick=31 elapsed_s=930 "
        "progress=3/5 unit=epochs metric=val_auc:0.847 loss=0.412 artifact=models/fold0_best.pt"
    )

    assert signals["phase"] == "train"
    assert signals["epochs"] == {"current": 3.0, "total": 5.0}
    assert signals["metrics"]["val_auc"] == pytest.approx(0.847)
    assert signals["metrics"]["loss"] == pytest.approx(0.412)
    assert signals["heartbeat"]["artifact_path"] == "models/fold0_best.pt"


def test_parse_tqdm_item_progress_signal() -> None:
    signals = _parse_progress_signals(
        "  8%|8         | 998/11925 [1:37:00<17:42:00,  5.83s/it]"
    )

    assert signals["items"] == {"current": 998.0, "total": 11925.0, "source": "tqdm"}


def test_parse_counted_feature_extraction_progress_signal() -> None:
    signals = _parse_progress_signals("[156160/252000] extracted")

    assert signals["items"] == {"current": 156160.0, "total": 252000.0, "source": "extracted"}


def test_scienceflow_hb_reference_format_regex_contract() -> None:
    heartbeat_re = re.compile(
        r"^SCIENCEFLOW_HB v=1 phase=[a-z0-9_-]+ tick=\d+ elapsed_s=\d+(?:\.\d+)? "
        r"progress=(?:\d+|\?)/(?:\d+|\?) unit=[a-z0-9_-]+ "
        r"metric=(?:[a-z0-9_.-]+:[^\s]+|na) loss=(?:[-+0-9.eE]+|na) "
        r"artifact=(?:\S+|none)$"
    )
    good = (
        "SCIENCEFLOW_HB v=1 phase=train tick=31 elapsed_s=930 "
        "progress=3/5 unit=epochs metric=val_auc:0.847 loss=0.412 artifact=models/fold0_best.pt"
    )

    assert heartbeat_re.match(good)
    assert not heartbeat_re.match(good.replace("v=1 ", ""))
    assert not heartbeat_re.match(good.rsplit(" artifact=", 1)[0])


def test_lnr_resource_observer_records_gpu_lease_events(tmp_path) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])

    job_id = observer.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    acquired = observer.queue_try_acquire(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["acquired"] is True

    observer.lease_registered(job_id, pid=os.getpid(), pgid=os.getpgrp())
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=1.0,
        phase="training",
        signals={"epoch": 1, "metric": 0.5},
        stdout_lines=3,
        stdout_bytes=64,
    )
    observer.job_finished(job_id, status="success", returncode=0, elapsed_sec=0.25)

    assert "resource_gpu_lease_acquired" in event_types(sm)
    assert "progress_heartbeat" in event_types(sm)
    assert "resource_gpu_lease_released" in event_types(sm)
    assert observer.resource_runtime is not None
    assert observer.resource_runtime.gpu_store.snapshot_active()["leases"] == {}


def test_progress_snapshot_marks_tqdm_phase_unaffordable_under_deadline(tmp_path) -> None:
    observer, _ = make_observer(tmp_path, worker_id="W00", gpu_pool=[])
    job_id = observer.job_created(
        command="python3 validate.py",
        inferred_class=RESOURCE_HEAVY_CPU_CANDIDATE,
        gpu_ids=[],
        timeout_sec=7200.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=5820.0,
        phase="validation_or_inference",
        signals={"items": {"current": 998.0, "total": 11925.0, "source": "tqdm"}},
        stdout_lines=100,
        stdout_bytes=4096,
    )

    snapshot = observer._progress_snapshot_for_job(
        observer._jobs[job_id],
        {
            "current_phase": "validation_or_inference",
            "deadline_event": True,
            "deadline_remaining_sec": 800.0,
            "finalization_reserve_sec": 900.0,
            "stdout_age_sec": 0.0,
            "stdout_lines": 100,
            "stdout_bytes": 4096,
            "saw_training_progress": True,
            "saw_final_score": False,
        },
        elapsed_sec=5820.0,
    )

    assert snapshot["progress_unit"] == "items"
    assert snapshot["progress_units_done"] == pytest.approx(998.0)
    assert snapshot["progress_units_total"] == pytest.approx(11925.0)
    assert snapshot["eta_to_current_phase_end_sec"] > 60000.0
    assert snapshot["finish_feasible"] is False
    assert snapshot["finish_feasibility_reason"] == "eta_exceeds_remaining_useful_budget"


def test_progress_snapshot_protects_near_complete_feature_extraction(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        worker_id="W01",
        gpu_pool=[],
        review_heartbeat_sec=60.0,
    )
    job_id = observer.job_created(
        command="python3 extract.py",
        inferred_class=RESOURCE_HEAVY_CPU_CANDIDATE,
        gpu_ids=[],
        timeout_sec=7200.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=2200.0,
        phase="feature_extraction",
        signals={"items": {"current": 230000.0, "total": 252000.0, "source": "extracted"}},
        stdout_lines=100,
        stdout_bytes=4096,
    )
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=2260.0,
        phase="feature_extraction",
        signals={"items": {"current": 245000.0, "total": 252000.0, "source": "extracted"}},
        stdout_lines=110,
        stdout_bytes=4352,
    )

    snapshot = observer._progress_snapshot_for_job(
        observer._jobs[job_id],
        {
            "current_phase": "feature_extraction",
            "deadline_event": False,
            "stdout_age_sec": 0.0,
            "stdout_lines": 110,
            "stdout_bytes": 4352,
        },
        elapsed_sec=2260.0,
    )

    assert snapshot["eta_source"] == "progress_interval:items"
    assert snapshot["eta_to_current_phase_end_sec"] == pytest.approx(28.0)
    assert snapshot["progress_fraction"] == pytest.approx(245000.0 / 252000.0)
    assert snapshot["phase_completion_protected"] is True


@pytest.mark.asyncio
async def test_bash_tool_emits_scienceflow_hb_as_progress_heartbeat(tmp_path) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"], min_register_sec=0.0)
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
        resource_progress_heartbeat_min_interval_sec=0.0,
    )
    cmd = (
        'python3 -u -c "print(\'SCIENCEFLOW_HB v=1 phase=train tick=1 elapsed_s=1 '
        'progress=1/5 unit=epochs metric=val_auc:0.847 loss=0.412 artifact=model.pt\', flush=True)" --epochs 1'
    )

    result = await tool.execute(cmd)

    assert result.error is None
    progress_events = payloads(sm, "progress_heartbeat")
    assert progress_events
    signals = progress_events[-1]["signals"]
    assert signals["heartbeat"]["format"] == "SCIENCEFLOW_HB"
    assert signals["epochs"] == {"current": 1.0, "total": 5.0}
    assert signals["metrics"]["val_auc"] == pytest.approx(0.847)
    assert "val_auc" in progress_events[-1]["metric_history_text"]


@pytest.mark.asyncio
async def test_bash_tool_emits_monitor_heartbeat_for_silent_cpu_pipeline(tmp_path) -> None:
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=[],
        min_register_sec=0.0,
        stalled_stdout_sec=999.0,
        bash_monitor_all_enabled=True,
        kill_mode="recommend",
        review_state_enabled=True,
        review_heartbeat_sec=0.1,
        review_warmup_windows=0,
        review_inactive_windows=5,
        review_value_windows=2,
        check_interval_sec=0.1,
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=3.0,
        bash_timeout_slow_sec=3.0,
        resource_observer=observer,
    )
    cmd = 'python3 -u -c "import time; end=time.time()+1.8;\nwhile time.time()<end: pass" 2>&1 | tail -20'

    result = await tool.execute(cmd)

    assert result.error is None
    heartbeat_payloads = payloads(sm, "resource_monitor_heartbeat")
    assert len(heartbeat_payloads) >= 2
    assert heartbeat_payloads[-1]["process_alive"] is True
    assert heartbeat_payloads[-1]["stdout_age_sec"] > 0
    assert {payload["cpu_bucket"] for payload in heartbeat_payloads} & {"active", "heavy"}


@pytest.mark.asyncio
async def test_bash_tool_observer_is_observe_only_on_light_success(tmp_path) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("python3 -c 'print(1)'")

    assert not result.error
    assert "1" in (result.output or "")
    assert payloads(sm, "resource_gpu_lease_acquired") == []


@pytest.mark.asyncio
async def test_bash_tool_observer_releases_on_timeout(tmp_path) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=0.1,
        bash_timeout_slow_sec=0.1,
        resource_observer=observer,
    )

    result = await tool.execute("python3 -c 'import time; time.sleep(2)'")

    assert result.error
    assert "timed out" in result.error.lower()
    assert "RESOURCE_FEEDBACK: command timed out because" in (result.output or "")
    assert "resource_gpu_lease_acquired" not in event_types(sm)


def _heavy_job(observer, tmp_path):
    job_id = observer.job_created(
        command="python3 train.py --epochs 2",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def test_resource_monitor_gap_records_event(tmp_path) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    job_id = _heavy_job(observer, tmp_path)

    result = observer.resource_monitor_gap(
        job_id,
        elapsed_sec=120.0,
        gap_sec=45.0,
        stdout_age_sec=45.0,
        stdout_lines=10,
        stdout_bytes=1024,
        reason="resource_guard_stopped:RuntimeError",
        process_tree_cpu={"available": True, "total_cpu_pct": 0.0},
    )

    assert result["recorded"] is True
    gap_payloads = payloads(sm, "resource_monitor_gap")
    assert gap_payloads[-1]["gap_sec"] == pytest.approx(45.0)
    assert gap_payloads[-1]["reason"] == "resource_guard_stopped:RuntimeError"
    assert observer.resource_runtime is not None
    event_lines = observer.resource_runtime.unified_store.events_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in event_lines if line.strip()]
    assert any(event.get("event_type") == "resource_monitor_gap" for event in events)


def test_idle_gpu_lease_guard_releases_by_default_when_safe(tmp_path, monkeypatch) -> None:
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
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="auto",
        gpu_idle_lease_guard_enabled=True,
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_min_samples=2,
        gpu_idle_lease_require_pressure=False,
    )
    job_id = _heavy_job(observer, tmp_path)
    assert observer.queue_try_acquire(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    first = observer.active_intervention_decision(job_id, elapsed_sec=1.0, stdout_age_sec=0.0)
    observer._last_gpu_util_emit[job_id] = 0.0
    second = observer.active_intervention_decision(job_id, elapsed_sec=2.0, stdout_age_sec=0.0)

    assert first["terminate"] is False
    assert second["terminate"] is False
    assert second["release_idle_lease"] is True
    assert second["idle_gpu_release_candidate"] is True
    assert second["idle_gpu_release_mode"] == "release"
    release = observer.release_idle_gpu_lease(job_id, elapsed_sec=2.0, reason=second["reason"], feedback=second["feedback"])
    assert release["released"] is True
    assert "resource_idle_lease_released" in event_types(sm)
    assert observer.resource_runtime is not None
    assert observer.resource_runtime.has_active_lease(job_id=job_id) is False



def test_gpu_active_bucket_requires_job_process_gpu_usage(tmp_path, monkeypatch) -> None:
    def busy_card_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": str(gpu_ids[0]),
                    "utilization_gpu_pct": 92.0,
                    "memory_used_mb": 8192.0,
                    "memory_total_mb": 81920.0,
                },
            ],
        }

    def no_process_gpu_usage(root_pid, allowed_gpu_ids):
        return {
            "available": True,
            "reason": "",
            "root_pid": int(root_pid or 0),
            "allowed_gpu_ids": list(allowed_gpu_ids or []),
            "process_count": 2,
            "used_gpu_processes": [],
            "violations": [],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", busy_card_sample)
    monkeypatch.setattr(
        "scienceflow.solver.lnr.resource_runtime.observer.controller.process_tree_gpu_placement_snapshot",
        no_process_gpu_usage,
    )
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        review_state_enabled=False,
        kill_mode="auto",
        gpu_idle_lease_guard_enabled=True,
        gpu_idle_lease_warmup_sec=0.0,
        gpu_idle_lease_min_samples=1,
        gpu_idle_lease_require_pressure=False,
    )
    job_id = _heavy_job(observer, tmp_path)
    assert observer.queue_try_acquire(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    observer.lease_registered(job_id, pid=12345, pgid=12345)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=2.0,
        stdout_age_sec=0.0,
        process_tree_cpu={"available": True, "total_cpu_pct": 400.0, "busy_child_count": 1},
    )
    heartbeat = observer.monitor_heartbeat(
        job_id,
        elapsed_sec=3.5,
        stdout_age_sec=0.0,
        process_tree_cpu={"available": True, "total_cpu_pct": 400.0, "busy_child_count": 1},
    )

    job = observer._jobs[job_id]
    assert decision["release_idle_lease"] is True
    assert job.idle_gpu_lease_samples == 1
    assert observer._last_gpu_sample_active(job) is False
    assert heartbeat["recorded"] is True
    assert payloads(sm, "resource_monitor_heartbeat")[-1]["gpu_bucket"] == "idle_or_none"
    assert payloads(sm, "resource_gpu_util_sampled")[-1]["process_gpu_placement"]["used_gpu_processes"] == []

def test_resource_run_summary_counts_resource_events(tmp_path) -> None:
    resource_dir = tmp_path / "task-a" / "logs" / "resource"
    store = UnifiedResourceStore(resource_dir)
    store.append_event(
        "admission_granted",
        worker_id="W00",
        command_id="W00:bash:00001",
        payload={"result": {"admission_action": "RUN_NOW"}},
    )
    store.append_event(
        "decision",
        worker_id="W00",
        command_id="W00:bash:00001",
        payload={"decision": {"action": "DENY_KILL"}},
    )
    store.append_event(
        "resource_idle_lease_release_candidate",
        worker_id="W00",
        command_id="W00:bash:00001",
        payload={"action": "RELEASE_IDLE_LEASE", "executed": False},
    )

    summary = summarize_resource_run(tmp_path)
    assert summary["resource_event_files"] == 1
    task = summary["tasks"][0]
    assert task["label"] == "task-a"
    assert task["admission_actions"]["RUN_NOW"] == 1
    assert task["decision_actions"]["DENY_KILL"] == 1
    assert task["idle_release_candidates"] == 1
    rendered = format_resource_run_summary(summary)
    assert "task-a" in rendered
    assert "idle_release_candidates=1" in rendered


def test_runtime_red_pressure_allows_cpu_support_but_blocks_gpu_train(tmp_path, monkeypatch) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    assert observer.resource_runtime is not None

    def busy_sample(gpu_ids):
        return {
            "available": True,
            "reason": "",
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 95.0,
                    "memory_used_mb": 8192.0,
                    "memory_total_mb": 143771.0,
                }
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", busy_sample)
    observer.resource_runtime.record_queue_timeout_pressure(
        job_id="gpu-train-1",
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        elapsed_sec=30.0,
        reason="queue_timeout",
        command_digest="a",
    )
    observer.resource_runtime.record_queue_timeout_pressure(
        job_id="gpu-train-2",
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        elapsed_sec=30.0,
        reason="queue_timeout",
        command_digest="b",
    )

    cpu_decision = observer.resource_runtime.pressure_gate_decision(
        resource_class=RESOURCE_HEAVY_CPU_CANDIDATE,
        gpu_ids=["0"],
        command_digest="cpu-support",
    )
    gpu_decision = observer.resource_runtime.pressure_gate_decision(
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        command_digest="gpu-train-3",
    )

    assert cpu_decision["blocked"] is False
    assert RESOURCE_HEAVY_CPU_CANDIDATE in cpu_decision["allowed_classes"]
    assert gpu_decision["blocked"] is True
    assert gpu_decision["reason"] == "tt_only_after_gpu_pressure"


def test_red_pressure_reconciles_when_gpu_is_observed_free(tmp_path, monkeypatch) -> None:
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    assert observer.resource_runtime is not None
    observer.resource_runtime.record_runtime_pressure(
        job_id="old-train",
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        elapsed_sec=600.0,
        reason="stale_resource_context",
        command_digest="old",
    )

    def fake_sample(gpu_ids):
        return {
            "available": True,
            "reason": "",
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 0.0,
                    "memory_total_mb": 143771.0,
                }
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)

    decision = observer.resource_runtime.pressure_gate_decision(
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        command_digest="new-train",
    )

    assert decision["blocked"] is False
    pressure = observer.resource_runtime.pressure_snapshot(gpu_ids=["0"])
    assert pressure["gpus"]["0"]["mode"] == "GREEN"
    assert pressure["gpus"]["0"]["cooldown_remaining_sec"] == 0.0
    assert pressure["gpus"]["0"].get("last_reconciled_reason") == "observed_free_before_pressure_gate"
    events_path = tmp_path / "resource" / "resource_events.jsonl"
    assert "gpu_pressure_reconciled_free" in events_path.read_text()


def test_resource_preflight_allows_cpu_support_after_queue_timeouts(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        timeout_block_train_after=1,
        timeout_tt_only_after=2,
    )
    observer._gpu_queue_timeout_count = 2
    cmd = "SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 train.py --fold 0"
    classified = classify_bash_command(cmd)
    assert classified.resource_class == RESOURCE_HEAVY_CPU_CANDIDATE
    job_id = observer.job_created(
        command=cmd,
        inferred_class=classified.resource_class,
        gpu_ids=["0"],
        timeout_sec=30.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None

    decision = observer.resource_preflight_decision(
        job_id,
        inferred_class=classified.resource_class,
        gpu_ids=["0"],
    )

    assert decision["allowed"] is True
    assert decision["resource_class"] == RESOURCE_HEAVY_CPU_CANDIDATE
    assert decision["gate"] == "tt_only_after_queue_timeouts"


def test_cpu_resource_intent_is_overridden_by_source_gpu_evidence(tmp_path) -> None:
    (tmp_path / "train.py").write_text(
        (
            "import torch\n"
            "model = torch.nn.Linear(1, 1).cuda()\n"
            "model.train()\n"
        ),
        encoding="utf-8",
    )
    observer, sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    cmd = "SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 train.py --epochs 1"
    classified = classify_bash_command(cmd)
    assert classified.resource_class == RESOURCE_HEAVY_CPU_CANDIDATE

    job_id = observer.job_created(
        command=cmd,
        inferred_class=classified.resource_class,
        gpu_ids=["0"],
        timeout_sec=30.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    job = observer._jobs[job_id]

    assert job.resource_class == RESOURCE_HEAVY_GPU_CANDIDATE
    assert job.gpu_queue_relevant is True


def test_lnr_observer_can_track_all_bash_when_enabled(tmp_path) -> None:
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=[],
        min_register_sec=0.0,
        bash_monitor_all_enabled=True,
    )

    job_id = observer.job_created(
        command="echo hello",
        inferred_class="readonly_cpu",
        gpu_ids=[],
        timeout_sec=30.0,
        workspace_dir=tmp_path,
    )

    assert job_id is not None
    observer.active_intervention_decision(
        job_id,
        elapsed_sec=1.0,
        stdout_age_sec=0.0,
        stdout_lines=1,
        stdout_bytes=6,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="inspection",
    )
    observer.job_finished(job_id, status="success", elapsed_sec=1.0, returncode=0)

    assert "resource_job_started" in event_types(sm)
    assert "resource_job_finished" in event_types(sm)


def test_lnr_observer_deadline_event_creates_arbiter_candidate(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1200.0,
        stdout_age_sec=10.0,
        stdout_lines=5,
        stdout_bytes=128,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        deadline_event=True,
        deadline_remaining_sec=800.0,
        finalization_reserve_sec=900.0,
    )

    assert decision["would_terminate"] is True
    assert decision["terminate"] is False
    assert decision["arbiter_enabled"] is True
    assert decision["proposal"]["progress_snapshot"]["deadline_event"] is True


def test_parse_scienceflow_hb_scope_and_checkpoint_contract() -> None:
    signals = _parse_progress_signals(
        "SCIENCEFLOW_HB v=1 phase=valid route=cnn_v2 fold=2 protocol=group_cv "
        "checkpoint_kind=fold_best safe_to_resume=yes mergeable=yes "
        "tick=4 elapsed_s=240 progress=4/5 unit=epochs metric=val_auc:0.85 "
        "loss=0.3 artifact=models/fold2.pt"
    )

    heartbeat = signals["heartbeat"]
    assert heartbeat["route_id"] == "cnn_v2"
    assert heartbeat["fold"] == "2"
    assert heartbeat["validation_protocol"] == "group_cv"
    assert heartbeat["checkpoint_kind"] == "fold_best"
    assert heartbeat["safe_to_resume"] == "yes"
    assert heartbeat["mergeable"] == "yes"


def test_progress_snapshot_prefers_route_progress_over_near_complete_batch(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        review_heartbeat_sec=60.0,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=7200.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    signals = {
        "batches": {"current": 95.0, "total": 100.0, "source": "tqdm"},
        "epochs": {"current": 2.0, "total": 10.0, "source": "trainer"},
    }
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=1200.0,
        phase="train",
        signals=signals,
        stdout_lines=100,
        stdout_bytes=4096,
    )

    snapshot = observer._progress_snapshot_for_job(
        observer._jobs[job_id],
        {
            "current_phase": "train",
            "deadline_event": False,
            "stdout_age_sec": 0.0,
            "stdout_lines": 100,
            "stdout_bytes": 4096,
        },
        elapsed_sec=1200.0,
    )

    assert snapshot["progress_unit"] == "epochs"
    assert snapshot["progress_scope"] == "route"
    assert snapshot["progress_fraction"] == pytest.approx(0.2)
    assert snapshot["phase_completion_protected"] is False


def test_near_complete_training_batch_does_not_protect_whole_route(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        review_heartbeat_sec=60.0,
    )
    job_id = observer.job_created(
        command="python3 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=7200.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=1140.0,
        phase="train",
        signals={"batches": {"current": 90.0, "total": 100.0, "source": "tqdm"}},
        stdout_lines=90,
        stdout_bytes=3600,
    )
    observer.progress_heartbeat(
        job_id,
        elapsed_sec=1200.0,
        phase="train",
        signals={"batches": {"current": 99.0, "total": 100.0, "source": "tqdm"}},
        stdout_lines=100,
        stdout_bytes=4096,
    )

    snapshot = observer._progress_snapshot_for_job(
        observer._jobs[job_id],
        {
            "current_phase": "train",
            "deadline_event": False,
            "stdout_age_sec": 0.0,
            "stdout_lines": 100,
            "stdout_bytes": 4096,
        },
        elapsed_sec=1200.0,
    )

    assert snapshot["progress_scope"] == "subphase"
    assert snapshot["progress_fraction"] == pytest.approx(0.99)
    assert snapshot["phase_completion_protected"] is False
    assert snapshot["eta_to_next_comparable_metric_sec"] is None
