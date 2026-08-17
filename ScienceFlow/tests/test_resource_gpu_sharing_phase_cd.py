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

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_TRAIN,
    classify_bash_command,
)
from scienceflow.solver.lnr.resource_runtime.gpu_sharing import build_gpu_share_config, evaluate_share_phase_a


def _observation(*, phase: str, resource_class: str, child_cpu_pct: float = 0.0, busy_child_count: int = 0):
    cfg = build_gpu_share_config(
        enabled=True,
        phase=phase,
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )
    return evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 6.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={
            "available": True,
            "child_cpu_pct": child_cpu_pct,
            "total_cpu_pct": child_cpu_pct,
            "busy_child_count": busy_child_count,
        },
        snapshot={
            "now": 100.0,
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": resource_class,
                    "slot_weight": 0.5,
                    "submitted_at": 10.0,
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )


def test_phase_c_feature_secondary_requires_feature_phase_and_low_cpu() -> None:
    allowed = _observation(phase="feature_share", resource_class=RESOURCE_GPU_FEATURE_EXTRACT, child_cpu_pct=0.0)
    medium_cpu = _observation(phase="feature_share", resource_class=RESOURCE_GPU_FEATURE_EXTRACT, child_cpu_pct=120.0)
    wrong_phase = _observation(phase="tt_share", resource_class=RESOURCE_GPU_FEATURE_EXTRACT, child_cpu_pct=0.0)

    assert allowed["share_eligible"] is True
    assert medium_cpu["share_eligible"] is False
    assert medium_cpu["secondary_allowed"]["reason"] == "cpu_pressure_medium_feature_denied"
    assert wrong_phase["secondary_allowed"]["reason"] == "feature_secondary_phase_disabled"


def test_phase_d_light_train_intent_is_explicit_and_policy_deferred_in_feature_phase() -> None:
    classified = classify_bash_command("SCIENCEFLOW_RESOURCE_INTENT=light_train python3 train_small.py --device=cuda --epochs 1")
    allowed = _observation(phase="light_train", resource_class=RESOURCE_GPU_LIGHT_TRAIN, child_cpu_pct=0.0)
    feature_phase = _observation(phase="feature_share", resource_class=RESOURCE_GPU_LIGHT_TRAIN, child_cpu_pct=0.0)
    tt_phase = _observation(phase="tt_share", resource_class=RESOURCE_GPU_LIGHT_TRAIN, child_cpu_pct=0.0)

    assert classified.resource_class == RESOURCE_GPU_LIGHT_TRAIN
    assert allowed["share_eligible"] is True
    assert allowed["memory"]["secondary_estimated_peak_gb"] >= 8.0
    assert feature_phase["share_eligible"] is True
    assert feature_phase["secondary_allowed"]["reason"] == "light_train_policy_deferred_to_arbiter"
    assert feature_phase["secondary_allowed"]["requires_llm_policy_decision"] is True
    assert tt_phase["share_eligible"] is False
    assert tt_phase["secondary_allowed"]["reason"] == "light_train_phase_disabled"



def test_phase_c_low_footprint_train_waiter_can_share_when_static_gpu_evidence_absent() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )
    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "total_cpu_pct": 0.0, "busy_child_count": 0},
        snapshot={
            "now": 1000.0,
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": "heavy_gpu_train",
                    "slot_weight": 1.0,
                    "submitted_at": 100.0,
                    "metadata": {
                        "source_hint": {
                            "command_train_evidence": True,
                            "source_train_evidence": True,
                            "command_gpu_compute_evidence": False,
                            "command_gpu_evidence": False,
                            "command_gpu_request": None,
                            "source_gpu_request": None,
                        }
                    },
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert observed["share_eligible"] is True
    assert observed["secondary_allowed"]["reason"] == "low_footprint_train_secondary_allowed"
    assert observed["secondary_allowed"]["effective_slot_weight"] <= 0.5


def test_phase_c_train_waiter_with_explicit_gpu_compute_evidence_is_not_shared() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )
    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "total_cpu_pct": 0.0, "busy_child_count": 0},
        snapshot={
            "now": 1000.0,
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": "heavy_gpu_train",
                    "slot_weight": 1.0,
                    "submitted_at": 100.0,
                    "metadata": {
                        "source_hint": {
                            "command_train_evidence": True,
                            "command_gpu_compute_evidence": True,
                        }
                    },
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert observed["share_eligible"] is False
    assert observed["secondary_allowed"]["reason"] == "secondary_slot_weight_too_high"


def test_phase_c_high_primary_cpu_can_share_feature_when_worker_cpus_are_isolated() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 400.0, "total_cpu_pct": 400.0, "busy_child_count": 4},
        snapshot={
            "now": 1000.0,
            "leases": {
                "W00:bash:00001": {
                    "job_id": "W00:bash:00001",
                    "gpu_ids": ["0"],
                    "metadata": {"cpu_set": "0-15"},
                }
            },
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_GPU_FEATURE_EXTRACT,
                    "slot_weight": 0.5,
                    "submitted_at": 100.0,
                    "metadata": {"cpu_set": "16-31"},
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert observed["cpu_pressure"] == "high"
    assert observed["cpu_isolation"]["mode"] == "isolated"
    assert observed["hard_gates"]["cpu_pressure_share_safe"] is True
    assert observed["secondary_allowed"]["reason"] == "allowed_cpu_isolated"
    assert observed["share_eligible"] is True

def test_phase_c_isolated_heavy_train_waiter_can_share_with_revocable_weight() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 160.0, "total_cpu_pct": 160.0, "busy_child_count": 2},
        snapshot={
            "now": 1000.0,
            "leases": {
                "W00:bash:00001": {
                    "job_id": "W00:bash:00001",
                    "gpu_ids": ["0"],
                    "metadata": {"cpu_set": "0-15"},
                }
            },
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_HEAVY_GPU_TRAIN,
                    "slot_weight": 1.0,
                    "submitted_at": 100.0,
                    "metadata": {
                        "cpu_set": "16-31",
                        "source_hint": {
                            "command_train_evidence": True,
                            "command_gpu_compute_evidence": True,
                        },
                    },
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert observed["cpu_isolation"]["mode"] == "isolated"
    assert observed["secondary_allowed"]["reason"] == "isolated_train_secondary_allowed"
    assert observed["secondary_allowed"]["effective_slot_weight"] <= 0.5
    assert observed["secondary_allowed"]["requires_llm_policy_decision"] is True
    assert observed["secondary_allowed"]["policy_gate"] == "bounded_heavy_train_secondary"
    assert observed["share_eligible"] is True


def test_phase_c_isolated_active_work_counts_as_productive_primary_signal() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "unknown", "stdout_lines": 0},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 130.0, "total_cpu_pct": 130.0, "busy_child_count": 1},
        snapshot={
            "now": 1000.0,
            "leases": {
                "W00:bash:00001": {
                    "job_id": "W00:bash:00001",
                    "gpu_ids": ["0"],
                    "metadata": {"cpu_set": "48-63"},
                }
            },
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_GPU_FEATURE_EXTRACT,
                    "slot_weight": 0.5,
                    "submitted_at": 100.0,
                    "metadata": {"cpu_set": "0-15"},
                }
            },
        },
        primary_kill_replan_candidate={"candidate": True, "share_blocking_candidate": False, "reasons": ["dataloader_bottleneck_samples_present"]},
    )

    assert observed["primary_signal"]["recent_artifact_or_metric_or_stdout"] is False
    assert observed["primary_signal"]["isolated_active_work_signal"] is True
    assert observed["hard_gates"]["productive_primary_signal"] is True
    assert observed["share_eligible"] is True

def test_phase_c_dataloader_bottleneck_review_only_does_not_block_isolated_share() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 400.0, "total_cpu_pct": 400.0, "busy_child_count": 4},
        snapshot={
            "now": 1000.0,
            "leases": {
                "W00:bash:00001": {
                    "job_id": "W00:bash:00001",
                    "gpu_ids": ["0"],
                    "metadata": {"cpu_set": "0-15"},
                }
            },
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_GPU_FEATURE_EXTRACT,
                    "slot_weight": 0.5,
                    "submitted_at": 100.0,
                    "metadata": {"cpu_set": "16-31"},
                }
            },
        },
        primary_kill_replan_candidate={
            "candidate": True,
            "share_blocking_candidate": False,
            "reasons": ["dataloader_bottleneck_samples_present"],
            "share_blocking_reasons": [],
        },
    )

    assert observed["hard_gates"]["primary_kill_replan_candidate_false"] is True
    assert observed["share_eligible"] is True


def test_phase_c_true_kill_candidate_still_blocks_share() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class="heavy_gpu_train",
        primary_gpu_ids=["0"],
        elapsed_sec=600.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "total_cpu_pct": 0.0, "busy_child_count": 0},
        snapshot={
            "now": 1000.0,
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_GPU_FEATURE_EXTRACT,
                    "slot_weight": 0.5,
                    "submitted_at": 100.0,
                }
            },
        },
        primary_kill_replan_candidate={
            "candidate": True,
            "share_blocking_candidate": True,
            "reasons": ["idle_gpu_lease_samples_present"],
            "share_blocking_reasons": ["idle_gpu_lease_samples_present"],
        },
    )

    assert observed["hard_gates"]["primary_kill_replan_candidate_false"] is False
    assert observed["share_eligible"] is False


def test_revocable_heavy_trial_uses_short_warmup_without_relaxing_memory_or_cpu_gates() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class=RESOURCE_HEAVY_GPU_TRAIN,
        primary_gpu_ids=["0"],
        elapsed_sec=90.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 4096.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=6.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "total_cpu_pct": 0.0, "busy_child_count": 0},
        snapshot={
            "now": 1000.0,
            "leases": {
                "W00:bash:00001": {
                    "job_id": "W00:bash:00001",
                    "gpu_ids": ["0"],
                    "metadata": {"cpu_set": "0-7"},
                }
            },
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_HEAVY_GPU_TRAIN,
                    "slot_weight": 1.0,
                    "submitted_at": 100.0,
                    "metadata": {"cpu_set": "8-15"},
                    "share_override": {
                        "trial_share": True,
                        "trial_mode": "revocable_secondary_trial",
                        "trial_initial_observe_sec": 180.0,
                        "primary_job_ids": ["W00:bash:00001"],
                    },
                    "share_override_requested_at": 990.0,
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert observed["standard_warmup_sec"] == 300.0
    assert observed["share_warmup_sec"] == 60.0
    assert observed["share_candidate"] is True
    assert observed["share_eligible"] is True
    assert observed["hard_gates"]["memory_headroom_for_secondary"] is True
    assert observed["cpu_isolation"]["mode"] == "isolated"
    assert observed["secondary_allowed"]["policy_gate"] == "revocable_heavy_train_trial"
    assert observed["secondary_allowed"]["requires_llm_policy_decision"] is True


def test_non_trial_share_still_requires_standard_warmup() -> None:
    cfg = build_gpu_share_config(
        enabled=True,
        phase="feature_share",
        policy_profile="conservative",
        memory_profile="conservative",
        cpu_policy="conservative",
    )

    observed = evaluate_share_phase_a(
        cfg=cfg,
        primary_job_id="W00:bash:00001",
        primary_resource_class=RESOURCE_HEAVY_GPU_TRAIN,
        primary_gpu_ids=["0"],
        elapsed_sec=90.0,
        progress_snapshot={"progress_signal": "active", "stdout_lines": 10},
        gpu_sample={
            "available": True,
            "gpus": [
                {
                    "gpu_id": "0",
                    "utilization_gpu_pct": 0.0,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 49152.0,
                }
            ],
        },
        previous_mem_peak_gb=2.0,
        process_tree_cpu={"available": True, "child_cpu_pct": 0.0, "total_cpu_pct": 0.0, "busy_child_count": 0},
        snapshot={
            "now": 1000.0,
            "leases": {
                "W00:bash:00001": {
                    "job_id": "W00:bash:00001",
                    "gpu_ids": ["0"],
                    "metadata": {"cpu_set": "0-7"},
                }
            },
            "waiters": {
                "W01:bash:00002": {
                    "job_id": "W01:bash:00002",
                    "worker_id": "W01",
                    "gpu_ids": ["0"],
                    "resource_class": RESOURCE_GPU_FEATURE_EXTRACT,
                    "slot_weight": 0.5,
                    "submitted_at": 100.0,
                    "metadata": {"cpu_set": "8-15"},
                }
            },
        },
        primary_kill_replan_candidate={"candidate": False, "reasons": []},
    )

    assert observed["standard_warmup_sec"] == 300.0
    assert observed["share_warmup_sec"] == 300.0
    assert observed["share_candidate"] is False
    assert observed["share_eligible"] is False
    assert observed["hard_gates"]["primary_runtime_sec_gte_warmup"] is False

