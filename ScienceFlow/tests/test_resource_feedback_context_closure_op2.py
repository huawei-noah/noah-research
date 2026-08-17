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

from scienceflow.core.agent.memory.resource_feedback_memory import ResourceFeedbackMemoryDeduper
from tests.lnr_resource_test_utils import FakeStateMachine, event_types, make_observer


def test_resource_feedback_summary_preserves_wait_option_current_state() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    first = (
        "RESOURCE_FEEDBACK: LOCAL_GPU_BUSY_USE_CPU_SUPPORT because shared GPU trial was not approved; "
        "mode=YELLOW; scope=per_gpu; blocked=heavy_gpu_train; allowed=readonly_cpu,light_cpu; "
        "gpu=6; holder=W00:bash:00012; eta_bucket=medium; eta_confidence=low; "
        "unlock_condition=holder_released_or_shared_trial_granted; blocked_until_unlock=true; "
        "action_options=cpu_support,resource_wait; wait_tool=resource_wait; "
        "wait_token=rw-W01-1234567890-0003; wait_max_sec=420; wait_reason=resource_busy.\n"
    )
    second = first.replace("rw-W01-1234567890-0003", "rw-W01-1234567890-0004")

    assert deduper.reduce(first) == ("", True)
    assert deduper.reduce(second) == ("", True)

    summary = deduper.summary_text()
    assert summary.count("shared GPU trial was not approved") == 1
    assert "repeat_count=2" in summary
    assert "allowed=light_cpu,readonly_cpu" in summary
    assert "gpu=6" in summary
    assert "wait=available" in summary
    assert "wait_tool=resource_wait" in summary
    assert "wait_token=rw-W01-1234567890-0004" in summary
    assert "wait_max_sec=420" in summary
    assert "wait_reason=resource_busy" in summary


def test_resource_feedback_summary_marks_wait_timeout_unavailable() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    feedback = (
        "RESOURCE_FEEDBACK: RESOURCE_WAIT_TIMEOUT because no material resource state change; "
        "mode=YELLOW; scope=per_gpu; blocked=heavy_gpu_train; gpu=6; "
        "unlock_condition=resource_state_change_required; resource_wait_allowed=false; "
        "retry_allowed=false; same_command_retry_allowed=false.\n"
    )

    assert deduper.reduce(feedback) == ("", True)

    summary = deduper.summary_text()
    assert "wait=unavailable" in summary
    assert "resource_wait_allowed=false" in summary
    assert "retry_allowed=false" in summary
    assert "same_command_retry_allowed=false" in summary
    assert "wait_token=" not in summary


def test_resource_feedback_summary_folds_repeated_stale_context_state() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    first = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because stale_resource_context; "
        "mode=RED; scope=per_gpu; blocked=heavy_gpu_train; allowed=readonly_cpu,light_cpu; "
        "gpu=5; eta_bucket=long; eta_confidence=low; "
        "unlock_condition=resource_pressure_cleared; blocked_until_unlock=true; pressure_generation=10.\n"
    )
    repeated = first.replace("pressure_generation=10", "pressure_generation=11")

    assert deduper.reduce(first) == ("", True)
    assert deduper.reduce(repeated) == ("", True)

    summary = deduper.summary_text()
    assert summary.count("stale_resource_context") == 1
    assert "repeat_count=2" in summary
    assert "allowed=light_cpu,readonly_cpu" in summary
    assert "blocked_until_unlock=true" in summary


def test_policy_preflight_bypasses_guard_when_current_class_is_allowed(tmp_path) -> None:
    sm = FakeStateMachine()
    obs, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])
    job = obs.job_created(
        command="ls -l",
        inferred_class="readonly_cpu",
        gpu_ids=["0"],
        timeout_sec=5.0,
        workspace_dir=tmp_path,
    )
    assert job is not None
    obs._jobs[job].post_feedback_gate = {
        "allowed_classes": ["readonly_cpu", "light_cpu"],
        "source_resource_mode": "YELLOW",
        "source_gpu_ids": ["0"],
    }

    result = obs.resource_preflight_decision(job, inferred_class="readonly_cpu", gpu_ids=["0"])

    assert result["allowed"] is True
    assert result["reason"] == "allowed_class_bypass"
    assert result["bypassed_reason"] == "post_feedback_blocked_class"
    assert "resource_allowed_class_bypass" in event_types(sm)


def test_resource_research_signal_folds_into_current_state_summary() -> None:
    deduper = ResourceFeedbackMemoryDeduper()
    first = (
        "RESOURCE_RESEARCH_SIGNAL: outcome=KILL_AND_REPLAN; confidence=high; "
        "review_facts=prior_observe_count=2,unchanged_windows=2,last_action=OBSERVE_MORE; "
        "resource_facts=intent_device_mismatch,device=gpu_to_cpu,assigned_resource_idle=true; "
        "progress_facts=progress_signal=stalled,progress_confidence=high,recent_metric_or_submission=false; "
        "value_facts=record_count=3,valid_record_count=1,valid_best=0.44; "
        "execution_focus=change_method_search_space_schedule_validation_target_or_stopping_condition.\n"
    )
    repeated = first.replace("confidence=high", "confidence=high")

    assert deduper.reduce(first) == ("", True)
    assert deduper.reduce(repeated) == ("", True)

    summary = deduper.summary_text()
    assert "status=RESEARCH_SIGNAL" in summary
    assert "KILL_AND_REPLAN" in summary
    assert "repeat_count=2" in summary
    assert "review_facts=prior_observe_count=2,unchanged_windows=2,last_action=OBSERVE_MORE" in summary
    assert "resource_facts=intent_device_mismatch,device=gpu_to_cpu,assigned_resource_idle=true" in summary
    assert "progress_facts=progress_signal=stalled,progress_confidence=high,recent_metric_or_submission=false" in summary
    assert "value_facts=record_count=3,valid_record_count=1,valid_best=0.44" in summary
    assert "execution_focus=change_method_search_space_schedule_validation_target_or_stopping_condition" in summary
    assert "category=" not in summary
    assert "decision_hint=" not in summary
    assert "route_action=" not in summary
