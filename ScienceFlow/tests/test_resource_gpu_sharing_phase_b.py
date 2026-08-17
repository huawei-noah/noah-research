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

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
)
from scienceflow.solver.lnr.resource_runtime.gpu_sharing import share_waiters_for_primary
from tests.lnr_resource_test_utils import make_observer


def _events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "resource" / "resource_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _job(observer, tmp_path: Path, command: str, resource_class: str, gpu_ids: list[str], *, cpu_set: str | None = None) -> str:
    job_id = observer.job_created(
        command=command,
        inferred_class=resource_class,
        gpu_ids=gpu_ids,
        cpu_set=cpu_set,
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def _fake_sample(gpu_ids):
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


def test_share_waiter_selection_prefers_current_override_and_skips_leased_waiter() -> None:
    snapshot = {
        "now": 1000.0,
        "leases": {
            "primary": {"job_id": "primary", "gpu_ids": ["0"], "metadata": {}},
            "already_shared": {"job_id": "already_shared", "gpu_ids": ["0"], "metadata": {"lease_mode": "shared_secondary"}},
        },
        "waiters": {
            "old": {
                "job_id": "old",
                "worker_id": "W01",
                "gpu_ids": ["0"],
                "resource_class": RESOURCE_GPU_TT_LIGHT,
                "slot_weight": 0.25,
                "priority_score": 0.9,
                "submitted_at": 100.0,
                "metadata": {},
            },
            "current": {
                "job_id": "current",
                "worker_id": "W01",
                "gpu_ids": ["0"],
                "resource_class": RESOURCE_GPU_TT_LIGHT,
                "slot_weight": 0.25,
                "priority_score": 0.8,
                "submitted_at": 900.0,
                "metadata": {
                    "share_override": {"candidate": True, "primary_job_ids": ["primary"], "requested_at": 995.0},
                    "share_override_requested_at": 995.0,
                },
            },
            "already_shared": {
                "job_id": "already_shared",
                "worker_id": "W01",
                "gpu_ids": ["0"],
                "resource_class": RESOURCE_GPU_TT_LIGHT,
                "slot_weight": 0.25,
                "priority_score": 1.0,
                "submitted_at": 950.0,
                "metadata": {
                    "share_override": {"candidate": True, "primary_job_ids": ["primary"], "requested_at": 999.0},
                    "share_override_requested_at": 999.0,
                },
            },
        },
    }

    waiters = share_waiters_for_primary(snapshot, primary_job_id="primary", primary_gpu_ids=["0"])

    assert [row["job_id"] for row in waiters] == ["current", "old"]
    assert waiters[0]["share_override_primary_match"] is True


def _make_pair(tmp_path: Path):
    common = {
        "gpu_pool": ["0"],
        "assignment": "lease",
        "min_register_sec": 0.0,
        "stalled_stdout_sec": 0.0,
        "gpu_share_enabled": True,
        "gpu_share_phase": "tt_share",
        "gpu_share_tt_with_train": False,
        "gpu_util_sample_interval_sec": 1.0,
        "gpu_dataloader_bottleneck_min_samples": 99,
        "metric_health_invalid_min_events": 1,
        "arbiter_proposal_coalesce_window_sec": 0.0,
    }
    primary_observer, _ = make_observer(tmp_path, worker_id="W00", **common)
    waiter_observer, _ = make_observer(tmp_path, worker_id="W01", **common)
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    (tmp_path / "predict.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(primary_observer, tmp_path, "python3 train.py --device=cuda --epochs 10", "heavy_gpu_candidate", ["0"])
    waiter = _job(waiter_observer, tmp_path, "python3 predict.py --device=cuda --tta", RESOURCE_GPU_TT_LIGHT, ["0"])
    assert primary_observer.queue_try_acquire(primary, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    blocked = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert blocked["acquired"] is False
    return primary_observer, waiter_observer, primary, waiter


def _drive_primary_share_review(observer, job_id: str):
    return observer.active_intervention_decision(
        job_id,
        elapsed_sec=600.0,
        stdout_age_sec=1.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
    )


def test_phase_b_grants_tt_shared_lease_and_waiter_acquires(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    primary_observer, waiter_observer, primary, waiter = _make_pair(tmp_path)

    decision = _drive_primary_share_review(primary_observer, primary)
    assert decision["terminate"] is False

    result = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert result["acquired"] is True
    lease = result["details"]["lease"]
    assert lease["metadata"]["lease_mode"] == "shared_secondary"
    assert lease["metadata"]["shared_primary_job_id"] == primary
    assert result["env_updates"]["CUDA_VISIBLE_DEVICES"] == "0"

    snapshot = primary_observer.resource_runtime.gpu_store.snapshot_active()
    primary_meta = snapshot["leases"][primary]["metadata"]
    assert primary_meta["lease_mode"] == "shared_primary"
    assert primary_meta["shared_secondary_job_ids"] == [waiter]
    assert "shared_gpu_lease_granted" in [event["event_type"] for event in _events(tmp_path)]


def test_active_share_review_returns_direct_shared_grant(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    common = {
        "gpu_pool": ["0"],
        "assignment": "lease",
        "min_register_sec": 0.0,
        "stalled_stdout_sec": 30.0,
        "kill_mode": "arbiter",
        "arbiter_enabled": True,
        "arbiter_mode": "policy",
        "gpu_share_enabled": True,
        "gpu_share_phase": "tt_share",
        "gpu_util_sample_interval_sec": 1.0,
        "gpu_dataloader_bottleneck_min_samples": 99,
        "metric_health_invalid_min_events": 1,
        "arbiter_proposal_coalesce_window_sec": 0.0,
    }
    primary_observer, _ = make_observer(tmp_path, worker_id="W00", **common)
    waiter_observer, _ = make_observer(tmp_path, worker_id="W01", **common)
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    (tmp_path / "predict.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(primary_observer, tmp_path, "python3 train.py --device=cuda --epochs 10", "heavy_gpu_candidate", ["0"])
    waiter = _job(waiter_observer, tmp_path, "python3 predict.py --device=cuda --tta", RESOURCE_GPU_TT_LIGHT, ["0"])
    assert primary_observer.queue_try_acquire(primary, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is False

    preview = primary_observer.active_intervention_decision(
        primary,
        elapsed_sec=600.0,
        stdout_age_sec=60.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
    )

    assert preview["terminate"] is False
    assert preview["would_terminate"] is False
    assert preview["shared_lease_grant"]["granted"] is True
    assert "GRANT_SHARED_GPU_LEASE" in preview["feedback"]
    snapshot = primary_observer.resource_runtime.gpu_store.snapshot_active()
    assert snapshot["leases"][waiter]["metadata"]["lease_mode"] == "shared_secondary"
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "gpu_share_candidate" in event_types
    assert "shared_gpu_lease_granted" in event_types


def test_post_arbiter_observe_more_records_share_without_grant(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    primary_observer, waiter_observer, primary, waiter = _make_pair(tmp_path)
    primary_observer._promote(primary_observer._jobs[primary], reason="test_visible", elapsed_sec=600.0)

    post = primary_observer.post_arbiter_continue_review(
        primary,
        arbiter_action="OBSERVE_MORE",
        elapsed_sec=600.0,
        stdout_age_sec=60.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
    )

    assert post["enabled"] is True
    assert post["observe_only_after_arbiter"] is True
    assert post["shared_lease_grant"]["reason"] == "post_arbiter_observe_only"
    result = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert result["acquired"] is False
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "gpu_share_candidate" in event_types
    assert "gpu_share_eligible" in event_types
    assert "shared_gpu_lease_granted" not in event_types


def test_shared_secondary_release_returns_primary_to_exclusive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    primary_observer, waiter_observer, primary, waiter = _make_pair(tmp_path)
    _drive_primary_share_review(primary_observer, primary)
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is True

    release = waiter_observer.resource_runtime.release(job_id=waiter, elapsed_sec=12.0, status="finished")

    assert release["released"] is True
    snapshot = primary_observer.resource_runtime.gpu_store.snapshot_active()
    assert waiter not in snapshot["leases"]
    assert snapshot["leases"][primary]["metadata"]["lease_mode"] == "exclusive"
    assert "shared_secondary_job_ids" not in snapshot["leases"][primary]["metadata"]


def test_shared_runtime_degradation_revokes_secondary_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    primary_observer, waiter_observer, primary, waiter = _make_pair(tmp_path)
    _drive_primary_share_review(primary_observer, primary)
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is True

    decision = primary_observer.active_intervention_decision(
        primary,
        elapsed_sec=600.0,
        stdout_age_sec=600.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
        invalid_metric_events=1,
    )

    assert decision["terminate"] is False
    assert decision["action"] == "STOP_SECONDARY_SHARED_JOB"
    snapshot = primary_observer.resource_runtime.gpu_store.snapshot_active()
    assert primary in snapshot["leases"]
    assert waiter not in snapshot["leases"]
    assert snapshot["leases"][primary]["metadata"]["lease_mode"] == "exclusive"
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "shared_gpu_lease_revoked" in event_types


def test_shared_runtime_primary_cpu_busy_unknown_progress_keeps_secondary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    primary_observer, waiter_observer, primary, waiter = _make_pair(tmp_path)
    _drive_primary_share_review(primary_observer, primary)
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is True

    decision = primary_observer.active_intervention_decision(
        primary,
        elapsed_sec=600.0,
        stdout_age_sec=1.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
        process_tree_cpu={"available": True, "child_cpu_pct": 250.0, "busy_child_count": 1},
        invalid_metric_events=0,
    )

    assert decision["terminate"] is False
    assert decision.get("action") != "STOP_SECONDARY_SHARED_JOB"
    snapshot = primary_observer.resource_runtime.gpu_store.snapshot_active()
    assert primary in snapshot["leases"]
    assert waiter in snapshot["leases"]
    assert snapshot["leases"][waiter]["metadata"]["lease_mode"] == "shared_secondary"
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "shared_gpu_lease_revoked" not in event_types


def test_shared_secondary_self_terminates_after_revocation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    primary_observer, waiter_observer, primary, waiter = _make_pair(tmp_path)
    _drive_primary_share_review(primary_observer, primary)
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is True
    primary_observer.resource_runtime.revoke_shared_gpu_lease(
        primary_job_id=primary,
        secondary_job_id=waiter,
        reason="test_revocation",
    )

    decision = waiter_observer.active_intervention_decision(
        waiter,
        elapsed_sec=240.0,
        stdout_age_sec=1.0,
        stdout_lines=4,
        stdout_bytes=80,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="inference",
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "busy_child_count": 0},
    )

    assert decision["terminate"] is True
    assert decision["reason"] == "active_intervention:shared_secondary_lease_revoked"




def test_policy_deferred_light_train_requires_llm_before_grant(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    common = {
        "gpu_pool": ["0"],
        "assignment": "lease",
        "min_register_sec": 0.0,
        "stalled_stdout_sec": 0.0,
        "gpu_share_enabled": True,
        "gpu_share_phase": "feature_share",
        "gpu_share_tt_with_train": False,
        "gpu_util_sample_interval_sec": 1.0,
        "gpu_dataloader_bottleneck_min_samples": 99,
        "metric_health_invalid_min_events": 1,
        "arbiter_enabled": False,
        "arbiter_proposal_coalesce_window_sec": 0.0,
    }
    primary_observer, _ = make_observer(tmp_path, worker_id="W00", **common)
    waiter_observer, _ = make_observer(tmp_path, worker_id="W01", **common)
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(primary_observer, tmp_path, "python3 train.py --device=cuda --epochs 10", "heavy_gpu_candidate", ["0"])
    waiter = _job(waiter_observer, tmp_path, "SCIENCEFLOW_RESOURCE_INTENT=light_train python3 train.py --device=cuda --epochs 2", RESOURCE_GPU_LIGHT_TRAIN, ["0"])
    assert primary_observer.queue_try_acquire(primary, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_LIGHT_TRAIN, gpu_ids=["0"])["acquired"] is False

    decision = _drive_primary_share_review(primary_observer, primary)

    assert decision["terminate"] is False
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "gpu_share_eligible" in event_types
    assert "shared_gpu_lease_granted" not in event_types
    result = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_LIGHT_TRAIN, gpu_ids=["0"])
    assert result["acquired"] is False

def test_llm_mode_share_review_requires_arbiter_grant_before_shared_lease(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    common = {
        "gpu_pool": ["0"],
        "assignment": "lease",
        "min_register_sec": 0.0,
        "stalled_stdout_sec": 0.0,
        "gpu_share_enabled": True,
        "gpu_share_phase": "tt_share",
        "gpu_share_tt_with_train": False,
        "gpu_util_sample_interval_sec": 1.0,
        "gpu_dataloader_bottleneck_min_samples": 99,
        "metric_health_invalid_min_events": 1,
        "arbiter_enabled": True,
        "arbiter_mode": "llm",
        "arbiter_proposal_coalesce_window_sec": 0.0,
    }
    primary_observer, _ = make_observer(tmp_path, worker_id="W00", **common)
    waiter_observer, _ = make_observer(tmp_path, worker_id="W01", **common)
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    (tmp_path / "predict.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(primary_observer, tmp_path, "python3 train.py --device=cuda --epochs 10", "heavy_gpu_candidate", ["0"])
    waiter = _job(waiter_observer, tmp_path, "python3 predict.py --device=cuda --tta", RESOURCE_GPU_TT_LIGHT, ["0"])
    assert primary_observer.queue_try_acquire(primary, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    assert waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is False

    decision = _drive_primary_share_review(primary_observer, primary)

    assert decision["arbiter_review"] is True
    assert decision["proposal"]["proposal_type"] == "task_gpu_share_review"
    facts = decision["proposal"]["share_decision_facts"]
    assert facts["decision_source"] == "primary_runtime"
    assert facts["share_eligible"] is True
    assert facts["primary_gpu_util_p90_pct"] <= 35.0
    assert "shared_gpu_lease_granted" not in [event["event_type"] for event in _events(tmp_path)]

    applied = primary_observer.apply_task_gpu_share_decision(
        primary,
        arbiter_action="GRANT_SHARED_GPU_LEASE",
        proposal=decision["proposal"],
        arbiter_decision={"action": "GRANT_SHARED_GPU_LEASE", "confidence": "high"},
    )

    assert applied["granted"] is True
    result = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert result["acquired"] is True
    assert result["details"]["lease"]["metadata"]["lease_mode"] == "shared_secondary"
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert "gpu_share_decision" in event_types
    assert "shared_gpu_lease_granted" in event_types
    assert "shared_gpu_lease_consumed" in event_types


def test_revocable_heavy_trial_requires_arbiter_grant_before_shared_lease(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    common = {
        "gpu_pool": ["0"],
        "assignment": "lease",
        "min_register_sec": 0.0,
        "stalled_stdout_sec": 0.0,
        "gpu_share_enabled": True,
        "gpu_share_phase": "feature_share",
        "gpu_share_tt_with_train": False,
        "gpu_util_sample_interval_sec": 1.0,
        "gpu_dataloader_bottleneck_min_samples": 99,
        "metric_health_invalid_min_events": 1,
        "arbiter_enabled": True,
        "arbiter_mode": "llm",
        "arbiter_proposal_coalesce_window_sec": 0.0,
    }
    primary_observer, _ = make_observer(tmp_path, worker_id="W00", **common)
    waiter_observer, _ = make_observer(tmp_path, worker_id="W01", **common)
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    (tmp_path / "train_alt.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    primary = _job(
        primary_observer,
        tmp_path,
        "python3 train.py --device=cuda --epochs 10",
        RESOURCE_HEAVY_GPU_CANDIDATE,
        ["0"],
        cpu_set="0-3",
    )
    waiter = _job(
        waiter_observer,
        tmp_path,
        "python3 train_alt.py --device=cuda --epochs 2",
        RESOURCE_HEAVY_GPU_CANDIDATE,
        ["0"],
        cpu_set="4-7",
    )
    assert primary_observer.queue_try_acquire(primary, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    blocked = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])
    assert blocked["acquired"] is False
    assert blocked["share_override_candidate"] is True
    assert blocked["share_override"]["trial_share"] is True

    decision = _drive_primary_share_review(primary_observer, primary)

    assert decision["arbiter_review"] is True
    assert decision["reason"] == "task_gpu_share_review:revocable_trial_waiter"
    proposal = decision["proposal"]
    assert proposal["resource_snapshot"]["trial_share"]["enabled"] is True
    assert proposal["resource_snapshot"]["secondary_allowed"]["policy_gate"] == "revocable_heavy_train_trial"
    assert proposal["resource_snapshot"]["secondary_allowed"]["requires_llm_policy_decision"] is True
    assert "shared_gpu_lease_granted" not in [event["event_type"] for event in _events(tmp_path)]

    applied = primary_observer.apply_task_gpu_share_decision(
        primary,
        arbiter_action="GRANT_SHARED_GPU_LEASE",
        proposal=proposal,
        arbiter_decision={"action": "GRANT_SHARED_GPU_LEASE", "confidence": "high"},
    )

    assert applied["granted"] is True
    result = waiter_observer.queue_try_acquire(waiter, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])
    assert result["acquired"] is True
    lease_meta = result["details"]["lease"]["metadata"]
    assert lease_meta["lease_mode"] == "shared_secondary"
    assert lease_meta["trial_share"] is True
    assert lease_meta["trial_mode"] == "revocable_secondary_trial"
    assert lease_meta["trial_primary_protected"] is True
    assert lease_meta["shared_effective_slot_weight"] == 0.5

def test_feature_share_light_train_soft_red_feedback_reaches_arbiter_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_contention_review_enabled=True,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    source = _job(observer, tmp_path, "python3 train.py --device=cuda --epochs 10", RESOURCE_GPU_LIGHT_TRAIN, ["0"])
    observer._remember_resource_feedback(
        observer._jobs[source],
        status="DENIED_REPLAN",
        reason="post_feedback_blocked_class",
        resource_mode="RED",
        blocked_class=RESOURCE_GPU_LIGHT_TRAIN,
        allowed_classes=["pure_tt_cpu", "gpu_tt_light", "readonly_cpu", "light_cpu"],
        cooldown_sec=300.0,
    )

    waiter = _job(
        observer,
        tmp_path,
        "SCIENCEFLOW_RESOURCE_INTENT=light_train python3 train.py --device=cuda --epochs 2",
        RESOURCE_GPU_LIGHT_TRAIN,
        ["0"],
    )
    decision = observer.resource_preflight_decision(waiter, inferred_class=RESOURCE_GPU_LIGHT_TRAIN, gpu_ids=["0"])

    assert observer._jobs[waiter].post_feedback_gate == {}
    assert decision["allowed"] is True
    assert decision.get("reason") not in {"post_feedback_blocked_class", "active_resource_plan_guard"}


def test_feature_share_light_train_hard_schema_feedback_still_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_contention_review_enabled=True,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    source = _job(observer, tmp_path, "python3 train.py --device=cuda --epochs 10", RESOURCE_HEAVY_GPU_CANDIDATE, ["0"])
    observer._pending_resource_feedback = {
        "feedback_id": "rf-test-schema",
        "status": "DENIED_REPLAN",
        "reason": "invalid_deliverable_schema_preflight",
        "resource_mode": "RED",
        "blocked_class": RESOURCE_GPU_LIGHT_TRAIN,
        "allowed_classes": ["pure_tt_cpu", "readonly_cpu", "light_cpu"],
        "job_id": source,
        "command_digest": observer._jobs[source].command_digest,
        "resource_class": RESOURCE_HEAVY_GPU_CANDIDATE,
        "gpu_ids": ["0"],
        "created_at": observer._jobs[source].created_at,
        "pressure_generation": observer._current_pressure_generation(),
        "unlock_condition": "submission_schema_valid",
        "blocked_until_unlock": True,
    }

    waiter = _job(
        observer,
        tmp_path,
        "SCIENCEFLOW_RESOURCE_INTENT=light_train python3 train.py --device=cuda --epochs 2",
        RESOURCE_GPU_LIGHT_TRAIN,
        ["0"],
    )
    decision = observer.resource_preflight_decision(waiter, inferred_class=RESOURCE_GPU_LIGHT_TRAIN, gpu_ids=["0"])

    assert decision["allowed"] is False
    assert decision["reason"] == "post_feedback_blocked_class"

def test_soft_red_heavy_feedback_expires_when_gpu_slot_is_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="lease",
        arbiter_enabled=True,
        arbiter_mode="llm",
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    source = _job(observer, tmp_path, "python3 train.py --device=cuda --epochs 10", RESOURCE_HEAVY_GPU_CANDIDATE, ["0"])
    observer._remember_resource_feedback(
        observer._jobs[source],
        status="DENIED_REPLAN",
        reason="post_feedback_blocked_class",
        resource_mode="RED",
        blocked_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        allowed_classes=["pure_tt_cpu", "gpu_tt_light", "readonly_cpu", "light_cpu"],
        cooldown_sec=300.0,
    )

    retry = _job(observer, tmp_path, "python3 train.py --device=cuda --epochs 2", RESOURCE_HEAVY_GPU_CANDIDATE, ["0"])
    decision = observer.resource_preflight_decision(retry, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])

    assert observer._jobs[retry].post_feedback_gate == {}
    assert decision["allowed"] is True


def test_soft_red_heavy_feedback_still_blocks_when_gpu_slot_has_holder(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_sample)
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="lease",
        arbiter_enabled=True,
        arbiter_mode="llm",
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    holder = _job(observer, tmp_path, "python3 train.py --device=cuda --epochs 10", RESOURCE_HEAVY_GPU_CANDIDATE, ["0"])
    assert observer.queue_try_acquire(holder, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    observer._remember_resource_feedback(
        observer._jobs[holder],
        status="DENIED_REPLAN",
        reason="post_feedback_blocked_class",
        resource_mode="RED",
        blocked_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        allowed_classes=["pure_tt_cpu", "gpu_tt_light", "readonly_cpu", "light_cpu"],
        cooldown_sec=300.0,
    )

    retry = _job(observer, tmp_path, "python3 train.py --device=cuda --epochs 2", RESOURCE_HEAVY_GPU_CANDIDATE, ["0"])
    decision = observer.resource_preflight_decision(retry, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])

    assert decision["allowed"] is False
    assert decision["reason"] == "post_feedback_blocked_class"
