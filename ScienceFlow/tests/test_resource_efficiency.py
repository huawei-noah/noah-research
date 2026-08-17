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

import pytest

from scienceflow.core.tools.bash_tool import _parse_progress_signals
from scienceflow.solver.lnr.global_merge.fallback import infer_lower_is_better
from scienceflow.solver.lnr.resource_runtime.metric_history import is_metric_history_line
from scienceflow.solver.lnr.resource_runtime.review.efficiency import (
    assess_resource_efficiency,
    cpu_set_count,
)
from scienceflow.solver.lnr.resource_runtime.review.execution_facts import (
    build_execution_facts,
)
from scienceflow.solver.lnr.resource_runtime.review.state_generation import (
    build_resource_state_generation,
)
from tests.lnr_resource_test_utils import make_observer


def _assessment(
    *,
    previous_state: dict | None = None,
    eta_sec: float = 3600.0,
    phase: str = "validation",
    llm_call_count: int = 0,
    completion_protected: bool = False,
) -> dict:
    return assess_resource_efficiency(
        previous_state=previous_state,
        cpu_set="168-183",
        process_tree_cpu={"available": True, "total_cpu_pct": 100.0},
        assigned_gpu_count=0,
        gpu_expected=False,
        gpu_active=False,
        gpu_sample_available=False,
        runtime_sec=3600.0,
        phase=phase,
        eta_to_deliverable_sec=eta_sec,
        eta_confidence="high",
        remaining_useful_budget_sec=7200.0,
        phase_completion_protected=completion_protected,
        metric_useful=False,
        deliverable_validity="none",
        progress_source="stdout_pattern",
        progress_evidence_trust="agent_reported",
        progress_interval_samples=2,
        llm_call_count=llm_call_count,
    )


def test_parser_accepts_prefix_counted_work_with_provenance() -> None:
    signals = _parse_progress_signals("Processed 50000/302644 rows")

    assert signals["items"] == {
        "current": 50000.0,
        "total": 302644.0,
        "source": "processed",
    }
    assert signals["progress_evidence"] == {
        "source": "stdout_pattern",
        "trust": "agent_reported",
    }


def test_parser_accepts_levenshtein_metric_with_provenance() -> None:
    text = "Validation Results: avg_levenshtein=9.3403"
    signals = _parse_progress_signals(text)

    assert signals["metrics"]["avg_levenshtein"] == pytest.approx(9.3403)
    assert signals["metric_evidence"]["trust"] == "agent_reported"
    assert is_metric_history_line(text) is True
    assert infer_lower_is_better([{"metric_name": "avg_levenshtein"}]) is True


def test_cpu_set_count_handles_ranges_and_duplicates() -> None:
    assert cpu_set_count("168-183") == 16
    assert cpu_set_count("0-3,2,6,8-9") == 7
    assert cpu_set_count("bad,5-3") == 0


def test_sustained_long_resource_mismatch_requires_bounded_review() -> None:
    first = _assessment()
    second = _assessment(previous_state=first)

    assert first["cpu_capacity_utilization"] == pytest.approx(0.0625)
    assert first["mismatch_windows"] == 1
    assert first["efficiency_review_started"] is False
    assert first["review_required"] is False
    assert second["mismatch_windows"] == 2
    assert second["efficiency_review_started"] is True
    assert second["review_required"] is True
    assert second["kill_authority"] == "review_only"


def test_short_eta_or_cpu_only_gpu_assignment_does_not_trigger_review() -> None:
    short = _assessment(eta_sec=300.0)
    gpu_irrelevant = assess_resource_efficiency(
        previous_state=None,
        cpu_set="0",
        process_tree_cpu={"available": True, "total_cpu_pct": 100.0},
        assigned_gpu_count=1,
        gpu_expected=False,
        gpu_active=False,
        gpu_sample_available=True,
        runtime_sec=3600.0,
        phase="inference",
        eta_to_deliverable_sec=3600.0,
        eta_confidence="high",
        remaining_useful_budget_sec=7200.0,
        phase_completion_protected=False,
        metric_useful=False,
        deliverable_validity="none",
        progress_source="stdout_pattern",
        progress_evidence_trust="agent_reported",
        progress_interval_samples=2,
        llm_call_count=0,
    )

    assert short["long_eta"] is False
    assert short["review_required"] is False
    assert gpu_irrelevant["gpu_idle_mismatch"] is False
    assert gpu_irrelevant["resource_mismatch"] is False


def test_active_gpu_suppresses_cpu_underutilization_mismatch() -> None:
    assessment = assess_resource_efficiency(
        previous_state=None,
        cpu_set="0-15",
        process_tree_cpu={"available": True, "total_cpu_pct": 100.0},
        assigned_gpu_count=1,
        gpu_expected=True,
        gpu_active=True,
        gpu_sample_available=True,
        runtime_sec=3600.0,
        phase="training",
        eta_to_deliverable_sec=3600.0,
        eta_confidence="high",
        remaining_useful_budget_sec=7200.0,
        phase_completion_protected=False,
        metric_useful=False,
        deliverable_validity="none",
        progress_source="stdout_pattern",
        progress_evidence_trust="agent_reported",
        progress_interval_samples=2,
        llm_call_count=0,
    )

    assert assessment["cpu_underutilized"] is True
    assert assessment["cpu_underutilized_mismatch"] is False
    assert assessment["resource_mismatch"] is False
    assert assessment["review_required"] is False


def test_missing_gpu_sample_fails_closed_for_cpu_underutilization() -> None:
    assessment = assess_resource_efficiency(
        previous_state=None,
        cpu_set="0-15",
        process_tree_cpu={"available": True, "total_cpu_pct": 100.0},
        assigned_gpu_count=1,
        gpu_expected=True,
        gpu_active=False,
        gpu_sample_available=False,
        runtime_sec=3600.0,
        phase="training",
        eta_to_deliverable_sec=3600.0,
        eta_confidence="high",
        remaining_useful_budget_sec=7200.0,
        phase_completion_protected=False,
        metric_useful=False,
        deliverable_validity="none",
        progress_source="stdout_pattern",
        progress_evidence_trust="agent_reported",
        progress_interval_samples=2,
        llm_call_count=0,
    )

    assert assessment["cpu_underutilized"] is True
    assert assessment["cpu_underutilized_mismatch"] is False
    assert assessment["gpu_idle_mismatch"] is False
    assert assessment["resource_mismatch"] is False


def test_efficiency_review_has_two_call_limit_and_resets_by_phase() -> None:
    first = _assessment()
    second = _assessment(previous_state=first)
    exhausted = _assessment(previous_state=second, llm_call_count=2)
    new_phase = _assessment(previous_state=second, phase="inference")
    exhausted_new_phase = _assessment(
        previous_state=exhausted,
        phase="inference",
        llm_call_count=2,
    )

    assert exhausted["review_limit_reached"] is True
    assert exhausted["review_required"] is False
    assert new_phase["mismatch_windows"] == 1
    assert exhausted_new_phase["mismatch_windows"] == 1
    assert exhausted_new_phase["review_limit_reached"] is True
    assert exhausted_new_phase["review_required"] is False


def test_completion_grace_is_non_sliding_within_a_phase() -> None:
    protected = _assessment(completion_protected=True, eta_sec=240.0)
    later = assess_resource_efficiency(
        previous_state=protected,
        cpu_set="168-183",
        process_tree_cpu={"available": True, "total_cpu_pct": 100.0},
        assigned_gpu_count=0,
        gpu_expected=False,
        gpu_active=False,
        gpu_sample_available=False,
        runtime_sec=3901.0,
        phase="validation",
        eta_to_deliverable_sec=240.0,
        eta_confidence="high",
        remaining_useful_budget_sec=7200.0,
        phase_completion_protected=True,
        metric_useful=False,
        deliverable_validity="none",
        progress_source="stdout_pattern",
        progress_evidence_trust="agent_reported",
        progress_interval_samples=2,
        llm_call_count=0,
    )

    assert protected["phase_completion_protected"] is True
    assert later["completion_grace_expired"] is True
    assert later["phase_completion_protected"] is False


def test_partial_submission_is_not_verified_value_progress(tmp_path) -> None:
    progress = {
        "runtime_sec": 1800.0,
        "artifact_updates": [{"path": "submission.csv"}],
        "artifact_last_update_age_sec": 5.0,
        "metric_history_line_count": 0,
        "meaningful_stdout": False,
        "near_submission": False,
    }
    partial = build_execution_facts(
        resource_snapshot={},
        progress_snapshot={**progress, "deliverable_validity": "none"},
    )
    valid = build_execution_facts(
        resource_snapshot={},
        progress_snapshot={**progress, "deliverable_validity": "produced_valid"},
    )
    observer, _ = make_observer(tmp_path)
    artifact = {
        "path": "submission.csv",
        "size_bytes": 100,
        "stability": "stable",
        "artifact_scope": "current_run",
    }

    assert partial["has_recent_useful_artifact"] is False
    assert valid["has_recent_useful_artifact"] is True
    assert observer._artifact_update_is_value_bearing(artifact) is False
    assert observer._artifact_update_is_value_bearing(
        artifact,
        deliverable_validity="produced_valid",
    ) is True


def test_efficiency_review_bypasses_only_low_priority_llm_defer(tmp_path) -> None:
    observer, _ = make_observer(tmp_path)
    job_id = "W00:bash:efficiency"
    observer._job_llm_call_count[job_id] = 1
    proposal = {
        "command_id": job_id,
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "high",
            "resource_efficiency": {
                "review_required": True,
                "review_limit_reached": False,
                "kill_authority": "review_only",
            },
        },
        "budget_priority": {"level": "low"},
        "suggested_actions": ["OBSERVE_MORE"],
    }

    priority = observer._budget_priority_for_proposal(proposal)

    assert "bounded_resource_efficiency_review" in priority["reasons"]
    assert priority["level"] in {"medium", "high"}
    assert observer._llm_budget_defer_reason(proposal, job_id, advisory=False) == ""


def test_efficiency_signal_triggers_review_when_generic_periodic_review_is_off(
    tmp_path,
) -> None:
    observer, _ = make_observer(
        tmp_path,
        arbiter_enabled=True,
        arbiter_periodic_review_enabled=False,
    )
    job_id = observer.job_created(
        command="python3 slow_validation.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=7200.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    job = observer._jobs[job_id]
    observer._promote(job, reason="elapsed_threshold", elapsed_sec=1800.0)
    job.resource_efficiency_state = {
        "review_required": True,
        "review_limit_reached": False,
        "kill_authority": "review_only",
        "reason_codes": ["cpu_capacity_underutilized", "long_eta"],
    }

    decision = observer._periodic_efficiency_review_decision(
        job,
        {
            "elapsed_sec": 1800.0,
            "resource_efficiency": job.resource_efficiency_state,
            "process_tree_cpu": {"available": True, "total_cpu_pct": 100.0},
        },
    )

    assert decision["enabled"] is True
    proposal = decision["proposal"]
    assert proposal["reason_code"] == "resource_efficiency:sustained_resource_mismatch"
    assert "sustained_resource_efficiency_mismatch" in proposal["trigger_reasons"]


def test_efficiency_window_growth_does_not_churn_llm_generation() -> None:
    efficiency = {
        "status": "inefficient",
        "review_required": True,
        "review_generation": 1,
        "review_limit_reached": False,
        "mismatch_windows": 2,
    }
    first = build_resource_state_generation({
        "progress_snapshot": {"resource_efficiency": efficiency},
    })
    later = build_resource_state_generation({
        "progress_snapshot": {
            "resource_efficiency": {**efficiency, "mismatch_windows": 200},
        },
    })
    next_review = build_resource_state_generation({
        "progress_snapshot": {
            "resource_efficiency": {**efficiency, "review_generation": 2},
        },
    })

    assert later.control_generation_key == first.control_generation_key
    assert later.feedback_generation_key == first.feedback_generation_key
    assert next_review.control_generation_key != first.control_generation_key
    assert next_review.feedback_generation_key == first.feedback_generation_key


def test_pre_boundary_llm_call_does_not_consume_efficiency_budget() -> None:
    first = _assessment(llm_call_count=0)
    triggered = _assessment(previous_state=first, llm_call_count=1)
    one_review = _assessment(previous_state=triggered, llm_call_count=2)
    exhausted = _assessment(previous_state=one_review, llm_call_count=3)

    assert triggered["baseline_llm_call_count"] == 1
    assert triggered["efficiency_review_llm_calls"] == 0
    assert triggered["review_required"] is True
    assert one_review["efficiency_review_llm_calls"] == 1
    assert one_review["review_required"] is True
    assert exhausted["efficiency_review_llm_calls"] == 2
    assert exhausted["review_limit_reached"] is True
    assert exhausted["review_required"] is False
