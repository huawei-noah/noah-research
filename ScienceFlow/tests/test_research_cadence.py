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

import csv

from scienceflow.solver.lnr.resource_runtime.review.research_cadence import (
    build_research_cadence_facts,
    is_comparable_live_metric,
    normalize_validation_protocol,
    route_metric_evidence,
    validation_protocols_comparable,
)
from scienceflow.solver.lnr.prompts import build_first_user_prompt
from scienceflow.solver.lnr.stage.metric_semantics import classify_metric_semantics
from tests.lnr_resource_test_utils import make_observer


def test_simple_direct_full_stays_within_first_metric_budget() -> None:
    facts = build_research_cadence_facts(
        enabled=True,
        eligible=True,
        elapsed_sec=300.0,
        observe_sec=300.0,
        first_metric_budget_sec=900.0,
        proven_route_metric_budget_sec=1800.0,
        execution_scale="direct_full",
        route_key="linear_v1",
        route_evidence={},
        current_comparable_metric=False,
        eta_to_next_comparable_metric_sec=300.0,
        eta_confidence="high",
    )

    assert facts["state"] == "within_budget"
    assert facts["violation"] is False
    assert facts["projected_metric_elapsed_sec"] == 600.0


def test_unproven_route_exceeding_first_metric_eta_requests_review() -> None:
    facts = build_research_cadence_facts(
        enabled=True,
        eligible=True,
        elapsed_sec=300.0,
        observe_sec=300.0,
        first_metric_budget_sec=900.0,
        proven_route_metric_budget_sec=1800.0,
        execution_scale="direct_full",
        route_key="resnet_v1",
        route_evidence={},
        current_comparable_metric=False,
        eta_to_next_comparable_metric_sec=3600.0,
        eta_confidence="medium",
    )

    assert facts["violation"] is True
    assert facts["reason_code"] == "next_comparable_metric_eta_exceeds_budget"
    assert facts["metric_budget_kind"] == "first_comparable_metric"


def test_proven_route_gets_larger_window_but_explicit_pilot_does_not() -> None:
    common = {
        "enabled": True,
        "eligible": True,
        "elapsed_sec": 1000.0,
        "observe_sec": 300.0,
        "first_metric_budget_sec": 900.0,
        "proven_route_metric_budget_sec": 1800.0,
        "route_key": "resnet_v1",
        "route_evidence": {"route_metric_proven": True, "comparable_metric_count": 1},
        "current_comparable_metric": False,
        "eta_to_next_comparable_metric_sec": 300.0,
        "eta_confidence": "high",
    }

    full = build_research_cadence_facts(execution_scale="full_after_pilot", **common)
    pilot = build_research_cadence_facts(execution_scale="pilot", **common)

    assert full["violation"] is False
    assert full["metric_budget_sec"] == 1800.0
    assert pilot["violation"] is True
    assert pilot["metric_budget_sec"] == 900.0


def test_route_metric_evidence_accepts_comparable_pilot_without_submission(tmp_path) -> None:
    path = tmp_path / "lhr_stage_performance.csv"
    fields = [
        "route_id",
        "metric_value",
        "metric_name",
        "lower_is_better",
        "validation_ok",
        "val_score_type",
        "selection_eligible",
        "selection_score",
        "metric_validity",
        "candidate_ready",
        "submission_status",
        "stage_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "route_id": "resnet_v1",
            "metric_value": "0.81",
            "metric_name": "val_auc",
            "lower_is_better": "0",
            "validation_ok": "1",
            "val_score_type": "holdout",
            "selection_eligible": "1",
            "selection_score": "0.81",
            "metric_validity": "high",
            "candidate_ready": "0",
            "submission_status": "missing_submission",
            "stage_id": "S01",
        })

    evidence = route_metric_evidence(path, "resnet_v1")

    assert evidence["route_metric_proven"] is True
    assert evidence["comparable_metric_count"] == 1
    assert evidence["latest_stage_id"] == "S01"


def test_loss_or_artifact_activity_is_not_a_comparable_metric() -> None:
    assert is_comparable_live_metric({"metric_name": "train_loss", "phase": "train"}) is False
    assert is_comparable_live_metric({"metric_name": "train_auc", "phase": "train"}) is False
    assert is_comparable_live_metric({"metric_name": "val_auc", "phase": "validation"}) is True


def test_metric_semantics_capture_scale_and_route_from_value_hint() -> None:
    semantics = classify_metric_semantics(
        metric_value=0.81,
        data={
            "bash_cmd": (
                "SCIENCEFLOW_RESOURCE_VALUE_HINT='"
                '{"execution_scale":"pilot","route_id":"resnet_v1"}'
                "' python train.py"
            ),
            "stdout_tail": "Final Validation Score: 0.81",
        },
        workspace_source_text="",
    )

    assert semantics["execution_scale"] == "pilot"
    assert semantics["route_id"] == "resnet_v1"


def test_ml_prompt_renders_pilot_value_hint_as_literal_json() -> None:
    prompt = build_first_user_prompt("Train a model.", wall_clock_budget_sec=3600)

    assert "bounded pilot" in prompt
    assert 'SCIENCEFLOW_RESOURCE_VALUE_HINT=\'{"execution_scale":"pilot","route_id":"stable_route_name"}\'' in prompt


def test_observer_emits_existing_value_review_for_cadence_violation(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        kill_mode="arbiter",
        arbiter_enabled=True,
        review_state_enabled=False,
        research_cadence_enabled=True,
        research_cadence_observe_sec=300.0,
        first_comparable_metric_budget_sec=900.0,
        proven_route_metric_budget_sec=1800.0,
    )
    job_id = observer.job_created(
        command="python train.py --epochs 10",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=[],
        timeout_sec=3600.0,
        workspace_dir=tmp_path,
        value_hint_override={"execution_scale": "direct_full", "route_id": "resnet_v1"},
    )
    assert job_id is not None
    observer._jobs[job_id].visible = True

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=901.0,
        stdout_age_sec=1.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="train",
    )

    assert decision["arbiter_review"] is True
    assert decision["proposal"]["proposal_type"] == "periodic_efficiency_review"
    assert decision["proposal"]["reason_code"] == "first_comparable_metric_budget_exceeded"
    assert decision["proposal"]["research_cadence"]["route_metric_proven"] is False


def test_validation_protocol_aliases_are_canonical_and_unknown_is_not_comparable() -> None:
    assert normalize_validation_protocol("group_cv") == "cv"
    assert normalize_validation_protocol("OOF-CV") == "cv"
    assert normalize_validation_protocol("validation holdout") == "holdout"
    assert validation_protocols_comparable("group_cv", "cv") is True
    assert validation_protocols_comparable("heldout", "holdout") is True
    assert validation_protocols_comparable("", "holdout") is False
    assert validation_protocols_comparable("mystery", "cv") is False
    assert validation_protocols_comparable("holdout", "cv") is False
