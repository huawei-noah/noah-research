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

from scienceflow.solver.lnr.resource_runtime.control_profile import (
    apply_control_profile_to_decision_delay,
    build_resource_control_profile,
)
from tests.lnr_resource_test_utils import make_observer, payloads


def test_high_resource_control_profile_tightens_review_cadence(tmp_path: Path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        resource_control_profile="high",
        stalled_stdout_sec=900.0,
        arbiter_min_progress_windows=4,
        arbiter_kill_requires_high_confidence=True,
        stale_pressure_observe_window_sec=180.0,
        stale_pressure_observe_max_sec=300.0,
        arbiter_proposal_coalesce_window_sec=60.0,
        main_agent_advisory_min_interval_sec=600.0,
        min_register_sec=600.0,
        gpu_idle_lease_warmup_sec=180.0,
        gpu_idle_lease_min_samples=3,
        gpu_dataloader_bottleneck_warmup_sec=600.0,
        gpu_dataloader_bottleneck_min_samples=3,
        quick_probe_expected_runtime_sec=300.0,
        quick_probe_hard_review_sec=600.0,
        arbiter_periodic_min_runtime_sec=480.0,
        arbiter_periodic_min_interval_sec=120.0,
    )

    assert observer.resource_control_profile == "high"
    assert observer.stalled_stdout_sec == 300.0
    assert observer.kill_proposal_cooldown_sec == 180.0
    assert observer.arbiter_min_progress_windows == 1
    assert observer.arbiter_kill_requires_high_confidence is False
    assert observer.stale_pressure_observe_window_sec == 60.0
    assert observer.stale_pressure_observe_max_sec == 180.0
    assert observer.arbiter_proposal_coalesce_window_sec == 30.0
    assert observer.main_agent_advisory_min_interval_sec == 120.0
    assert observer.min_register_sec == 120.0
    assert observer.gpu_idle_lease_warmup_sec == 120.0
    assert observer.gpu_idle_lease_min_samples == 2
    assert observer.gpu_dataloader_bottleneck_warmup_sec == 360.0
    assert observer.gpu_dataloader_bottleneck_min_samples == 2
    assert observer.quick_probe_expected_runtime_sec == 180.0
    assert observer.quick_probe_hard_review_sec == 420.0
    assert observer.arbiter_periodic_min_runtime_sec == 300.0
    assert observer.arbiter_periodic_min_interval_sec == 60.0
    assert observer.review_config.warmup_windows == 1
    assert observer.review_config.value_windows == 2

    profile = build_resource_control_profile("high")
    observe = apply_control_profile_to_decision_delay(
        {"action": "OBSERVE_MORE", "observe_more_sec": 600.0},
        profile=profile,
    )
    deny = apply_control_profile_to_decision_delay(
        {"action": "DENY_KILL", "ttl_sec": 900.0},
        profile=profile,
    )

    assert observe["observe_more_sec"] == 120.0
    assert deny["ttl_sec"] == 180.0
    assert observe["resource_control_profile"] == "high"
    assert deny["resource_control_profile"] == "high"


def test_high_profile_clears_stale_post_feedback_gate_on_observed_free_gpu(tmp_path: Path) -> None:
    observer, sm = make_observer(
        tmp_path,
        gpu_pool=["0"],
        assignment="lease",
        resource_control_profile="high",
        timeout_hard_gate_enabled=False,
        admission_llm_enabled=False,
    )
    job_id = observer.job_created(
        command="python3 train.py --device cuda --epochs 2",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    job = observer._jobs[job_id]
    job.post_feedback_gate = {
        "reason": "post_feedback_blocked_class",
        "source_gpu_ids": ["0"],
        "source_resource_mode": "YELLOW",
        "allowed_classes": ["pure_tt_cpu", "readonly_cpu", "light_cpu"],
        "unlock_condition": "resource_slot_available",
        "blocked_until_unlock": True,
    }

    def _free_gpu(**_: object) -> dict[str, object]:
        return {"cleared": False, "reason": "no_active_global_pressure", "gpu_ids": ["0"]}

    assert observer.resource_runtime is not None
    observer.resource_runtime.reconcile_free_gpu_pressure = _free_gpu  # type: ignore[method-assign]

    decision = observer.resource_preflight_decision(job_id, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert decision["allowed"] is True
    assert decision["reason"] == "observed_free_gpu_cleared_post_feedback_gate"
    assert job.post_feedback_gate == {}
    events = payloads(sm, "resource_soft_gate_cleared")
    assert events and events[-1]["resource_control_profile"] == "high"
