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
from pathlib import Path

from scienceflow.core.tools.resource_classifier import RESOURCE_GPU_TT_LIGHT, RESOURCE_UNKNOWN_GPU_EXEC
from scienceflow.solver.lnr.resource_runtime.review.arbiter import enforce_proposal_action_allowlist
from scienceflow.solver.lnr.resource_runtime.gpu_sharing import (
    build_gpu_share_config,
    evaluate_share_phase_a,
    secondary_estimated_peak_gb,
)
from tests.lnr_resource_test_utils import make_observer


def _events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "resource" / "resource_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _job(observer, tmp_path: Path, command: str, resource_class: str, gpu_ids: list[str]) -> str:
    job_id = observer.job_created(
        command=command,
        inferred_class=resource_class,
        gpu_ids=gpu_ids,
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def test_gpu_share_proposal_action_allowlist_downgrades_primary_kill() -> None:
    proposal = {"proposal_id": "rp_share", "proposal_type": "task_gpu_share_review"}

    decision = enforce_proposal_action_allowlist(
        {"action": "KILL_AND_REPLAN", "reason": "wrong channel", "confidence": "high"},
        proposal,
    )

    assert decision["action"] == "CONTINUE_SHARED_OBSERVE"
    assert decision["gate"]["blocked_reason"] == "proposal_action_not_allowed"
    assert decision["gate"]["fallback_action"] == "CONTINUE_SHARED_OBSERVE"



def test_gpu_share_unknown_secondary_is_denied_and_default_peak_is_conservative() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="observe",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )
    observation = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [{"gpu_id": "0", "utilization_gpu_pct": 5.0, "memory_used_mb": 2048.0, "memory_total_mb": 49152.0}],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
        snapshot={
            "now": 100.0,
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_UNKNOWN_GPU_EXEC,
                    "slot_weight": 0.25,
                    "submitted_at": 10.0,
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert secondary_estimated_peak_gb(RESOURCE_UNKNOWN_GPU_EXEC, cfg) >= cfg.secondary_estimated_peak_gb_light_train
    assert observation["secondary_allowed"]["allowed"] is False
    assert observation["secondary_allowed"]["reason"] == "unknown_gpu_secondary_requires_revocable_trial"
    assert observation["share_eligible"] is False


def test_gpu_share_phase_a_emits_candidate_and_eligible_events(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": str(gpu_ids[0]),
                    "utilization_gpu_pct": 8.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="env_only",
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        gpu_share_enabled=True,
        gpu_share_phase="observe",
        gpu_share_tt_with_train=False,
        gpu_util_sample_interval_sec=1.0,
    )
    (tmp_path / "predict.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(observer, tmp_path, "python3 train.py --device cuda --epochs 10", "heavy_gpu_candidate", ["0"])
    assert observer.queue_try_acquire(primary, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    waiter = _job(observer, tmp_path, "python3 predict.py --device cuda --tta", RESOURCE_GPU_TT_LIGHT, ["0"])
    assert observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is False

    decision = observer.active_intervention_decision(
        primary,
        elapsed_sec=600.0,
        stdout_age_sec=1.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
    )

    assert decision["terminate"] is False
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "gpu_share_candidate" in event_types
    assert "gpu_share_eligible" in event_types


def test_gpu_share_handoff_opens_v5_periodic_review_for_weak_primary(tmp_path, monkeypatch) -> None:
    def fake_sample(gpu_ids):
        return {
            "available": True,
            "gpus": [
                {
                    "gpu_id": str(gpu_ids[0]),
                    "utilization_gpu_pct": 3.0,
                    "memory_used_mb": 4096.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        }

    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", fake_sample)
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="env_only",
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_periodic_review_enabled=False,
        arbiter_periodic_min_interval_sec=1.0,
        arbiter_proposal_coalesce_window_sec=0.0,
        gpu_share_enabled=True,
        gpu_share_phase="observe",
        gpu_share_tt_with_train=False,
        gpu_dataloader_bottleneck_min_samples=1,
        gpu_util_sample_interval_sec=1.0,
    )
    (tmp_path / "predict.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(observer, tmp_path, "python3 train.py --device cuda --epochs 10", "heavy_gpu_candidate", ["0"])
    assert observer.queue_try_acquire(primary, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    waiter = _job(observer, tmp_path, "python3 predict.py --device cuda --tta", RESOURCE_GPU_TT_LIGHT, ["0"])
    assert observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is False

    decision = observer.active_intervention_decision(
        primary,
        elapsed_sec=600.0,
        stdout_age_sec=1.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
    )

    assert decision["arbiter_review"] is True
    assert decision["gpu_share_handoff"] is True
    assert decision["proposal"]["proposal_type"] == "periodic_efficiency_review"
    assert decision["proposal"]["reason_code"] == "gpu_share_handoff:primary_kill_replan_candidate"
    events = _events(tmp_path)
    event_types = [event["event_type"] for event in events]
    assert "primary_kill_replan_candidate" in event_types
    assert "resource_review_proposal" in event_types
