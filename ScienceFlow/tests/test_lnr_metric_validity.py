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

import asyncio
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scienceflow.solver.lnr.stage.metric_adjudication import (
    MetricValidityJudgment,
    adjudicate_metric_validity,
    parse_metric_output_interpretation_text,
    parse_metric_validity_judgment_text,
)
from scienceflow.solver.lnr.prompts import (
    build_metric_output_interpreter_prompt,
    build_metric_output_interpreter_system_prompt,
    build_metric_validity_adjudicator_prompt,
    build_metric_validity_adjudicator_system_prompt,
)
from scienceflow.solver.lnr.stage.score_summary import build_stage_performance_score_summary
from scienceflow.solver.lnr.solver import LnrSolver


FIELDS = [
    "row_order",
    "candidate_id",
    "worker_id",
    "stage_id",
    "metric_value",
    "metric_name",
    "lower_is_better",
    "validation_ok",
    "val_score_type",
    "selection_eligible",
    "selection_score",
    "metric_source_note",
    "metric_validity",
    "metric_validity_note",
    "metric_validity_reason_code",
    "metric_validity_source",
    "evaluator_backend",
    "evaluator_status",
    "brief",
    "why",
    "route_evidence",
    "submission_snapshot",
    "candidate_ready",
    "submission_status",
    "snapshot_path",
]


def _write_perf(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_schema_fix_metric_validity_low_excluded_from_valid_best(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.842050",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.842050",
                "metric_source_note": "type=holdout; eval=validation; train_data=train_only",
                "brief": "valid held-out baseline",
                "why": "validation rows were not used for training",
                "route_evidence": "verdict=continue; reason=valid held-out score; next=improve features",
                "submission_snapshot": "snapshots/W00/s01.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S01",
            },
            {
                "row_order": 2,
                "candidate_id": "W00:L01:S02",
                "worker_id": "W00",
                "stage_id": "S02",
                "metric_value": "1.000000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "1.000000",
                "metric_source_note": "type=holdout; execution=predict_only_reuse_artifacts",
                "brief": "submission produced after final train+val retrain",
                "why": "pipeline works but validation report is not comparable",
                "route_evidence": "verdict=schema_fix_needed; reason=predict.py computes val AUC on retrained train+val model; next=load early-stop model for held-out score",
                "submission_snapshot": "snapshots/W00/s02.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S02",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W01")

    assert summary["best_score"]["value"] == 1.0
    assert summary["best_score"]["metric_validity"] == "low"
    assert summary["best_score"]["validity"] == "risk_flagged_metric"
    assert summary["valid_best_score"]["value"] == 0.84205
    assert summary["valid_best_score"]["metric_validity"] == "high"
    assert summary["valid_record_count"] == 1
    assert summary["capture_gap"] is True
    assert summary["recommended_action"] == "metric_source_review"


def test_same_validation_reason_code_excluded_from_valid_best_even_with_evaluator_ok(tmp_path: Path) -> None:
    perf = tmp_path / "lhr_stage_performance.csv"
    _write_perf(
        perf,
        [
            {
                "row_order": 1,
                "candidate_id": "W00:L02:S26",
                "worker_id": "W00",
                "stage_id": "S26",
                "metric_value": "0.708000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.708000",
                "metric_validity": "high",
                "metric_validity_reason_code": "comparable_holdout",
                "metric_validity_note": "comparable_holdout: no validation meta-fit evidence",
                "evaluator_backend": "local_csv",
                "evaluator_status": "ok",
                "metric_source_note": "Submission is valid.",
                "brief": "valid DINOv2-S ensemble baseline",
                "why": "Uniform ensemble evaluated on held-out validation data.",
                "submission_snapshot": "snapshots/W00/s26.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S26",
            },
            {
                "row_order": 2,
                "candidate_id": "W00:L05:S12",
                "worker_id": "W00",
                "stage_id": "S12",
                "metric_value": "0.967800",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "val_score_type": "holdout",
                "selection_eligible": "1",
                "selection_score": "0.967800",
                "metric_validity": "medium",
                "metric_validity_reason_code": "same_validation_meta_fit",
                "metric_validity_note": "same_validation_meta_fit: validation set tuned ensemble weights",
                "metric_validity_source": "metric_validity_llm",
                "evaluator_backend": "local_csv",
                "evaluator_status": "ok",
                "metric_source_note": "Submission is valid.",
                "brief": "contaminated high validation score",
                "why": "Full-data models trained on validation gave a spuriously high score.",
                "submission_snapshot": "snapshots/W00/s12.csv",
                "candidate_ready": "1",
                "submission_status": "ready",
                "snapshot_path": "/abs/snapshots/W00/S12",
            },
        ],
    )

    summary = build_stage_performance_score_summary(perf, current_worker_id="W01")

    assert summary["best_score"]["value"] == 0.9678
    assert summary["best_score"]["validity"] == "risk_flagged_metric"
    assert summary["best_score"]["metric_validity"] == "low"
    assert summary["valid_best_score"]["value"] == 0.708
    assert summary["valid_best_score"]["stage_id"] == "S26"
    assert summary["valid_record_count"] == 1
    assert summary["capture_gap"] is True


def test_metric_adjudicator_system_veto_overrides_explicit_high_same_validation_meta() -> None:
    fields = {
        "metric_value": 0.964891,
        "metric_name": "Final Validation Score",
        "validation_ok": "1",
        "val_score_type": "cv",
        "selection_eligible": "1",
        "candidate_ready": "1",
        "metric_validity": "high",
        "metric_source_note": "type=cv; protocol=unknown; eval=unknown",
        "why": "Meta-stacking/calibration attempted but not held-out valid.",
        "route_evidence": "verdict=useful_negative; reason=meta stack not held-out valid; next=ignore score",
    }

    adjudicated = adjudicate_metric_validity(fields)

    assert adjudicated["metric_validity"] == "low"
    assert adjudicated["selection_eligible"] is False
    assert adjudicated["metric_validity_source"] == "metric_validity_system_veto"
    assert adjudicated["metric_validity_reason_code"] == "same_validation_meta_fit"


def test_metric_adjudicator_llm_can_downgrade_explicit_high() -> None:
    fields = {
        "metric_value": 0.91,
        "metric_name": "Final Validation Score",
        "validation_ok": "1",
        "val_score_type": "cv",
        "selection_eligible": "1",
        "candidate_ready": "1",
        "metric_validity": "high",
        "why": "Short stage note with ambiguous validation semantics.",
    }
    judgment = MetricValidityJudgment(
        metric_validity="medium",
        selection_eligible=False,
        reason_code="unknown_protocol",
        reason="No positive evidence that the reported score is comparable.",
        confidence="high",
        source="metric_validity_llm",
    )

    adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)

    assert adjudicated["metric_validity"] == "medium"
    assert adjudicated["selection_eligible"] is False
    assert adjudicated["metric_validity_source"] == "metric_validity_llm"


def test_opt_solver_evaluator_ok_is_authoritative_even_if_llm_says_unknown() -> None:
    fields = {
        "task_profile": "opt_solver",
        "evaluator_backend": "artifact_command",
        "evaluator_status": "ok",
        "metric_value": 820.828603,
        "metric_name": "tomato_mass_kg",
        "lower_is_better": False,
        "validation_ok": "1",
        "val_score_type": "benchmark",
        "selection_eligible": "1",
        "candidate_ready": "1",
        "metric_validity": "medium",
        "why": "T1 refinement improved the official artifact score.",
    }
    judgment = MetricValidityJudgment(
        metric_validity="medium",
        selection_eligible=False,
        reason_code="unknown_protocol",
        reason="No CV or held-out protocol is documented.",
        confidence="high",
        source="metric_validity_llm",
    )

    adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)

    assert adjudicated["metric_validity"] == "high"
    assert adjudicated["selection_eligible"] is True
    assert adjudicated["metric_validity_reason_code"] == "opt_solver_evaluator_ok"
    assert adjudicated["metric_validity_source"] == "metric_validity_system"


def test_declared_authoritative_evaluator_ignores_ml_protocol_downgrade() -> None:
    fields = {
        "task_profile": "scientific_design",
        "evaluator_backend": "task_package",
        "evaluator_status": "ok",
        "metric_authoritative": True,
        "metric_value": 1.0,
        "metric_name": "state_fidelity",
        "lower_is_better": False,
        "validation_ok": True,
        "val_score_type": "simulator_reward",
        "selection_eligible": True,
        "candidate_ready": True,
        "metric_validity": "high",
    }
    judgment = MetricValidityJudgment(
        metric_validity="medium",
        selection_eligible=False,
        reason_code="unknown_protocol",
        reason="No held-out or CV protocol is documented.",
        confidence="high",
        source="metric_validity_llm",
    )

    adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)

    assert adjudicated["metric_validity"] == "high"
    assert adjudicated["selection_eligible"] is True
    assert adjudicated["metric_validity_reason_code"] == "authoritative_evaluator_ok"
    assert adjudicated["metric_validity_source"] == "metric_validity_system"


@pytest.mark.parametrize(
    "why",
    [
        "The added prototype feature is slightly overfitting.",
        "This reproduction rules out overfitting as the explanation.",
    ],
)
def test_authoritative_evaluator_ignores_overfit_route_language(why: str) -> None:
    fields = {
        "task_profile": "sci_modeling_bench",
        "evaluator_backend": "task_package",
        "evaluator_status": "ok",
        "metric_authoritative": True,
        "metric_value": 0.235,
        "metric_name": "global_ndcg",
        "lower_is_better": False,
        "validation_ok": True,
        "selection_eligible": True,
        "candidate_ready": True,
        "submission_status": "ok",
        "metric_validity": "high",
        "why": why,
    }

    adjudicated = adjudicate_metric_validity(fields)

    assert adjudicated["metric_validity"] == "high"
    assert adjudicated["selection_eligible"] is True
    assert adjudicated["metric_validity_reason_code"] == "authoritative_evaluator_ok"


def test_authoritative_evaluator_does_not_override_invalid_submission() -> None:
    fields = {
        "task_profile": "sci_modeling_bench",
        "evaluator_backend": "task_package",
        "evaluator_status": "ok",
        "metric_authoritative": True,
        "metric_value": 0.9,
        "lower_is_better": False,
        "validation_ok": True,
        "submission_validation_ok": False,
        "selection_eligible": True,
        "candidate_ready": True,
        "submission_status": "invalid_submission",
        "why": "official score",
    }

    adjudicated = adjudicate_metric_validity(fields)

    assert adjudicated["metric_validity"] == "low"
    assert adjudicated["selection_eligible"] is False
    assert adjudicated["metric_validity_reason_code"] == "invalid_submission"


def test_evaluator_verified_medium_keeps_selection_eligible() -> None:
    fields = {
        "task_profile": "custom_solver",
        "evaluator_backend": "artifact_command",
        "evaluator_status": "ok",
        "metric_value": 120.5,
        "metric_name": "solver_score",
        "lower_is_better": False,
        "validation_ok": "1",
        "val_score_type": "benchmark",
        "selection_eligible": "1",
        "candidate_ready": "1",
        "submission_status": "ready",
        "metric_validity": "medium",
    }
    judgment = MetricValidityJudgment(
        metric_validity="medium",
        selection_eligible=False,
        reason_code="weak_provenance",
        reason="The score provenance is not strong enough to label high.",
        confidence="high",
        source="metric_validity_llm",
    )

    adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)

    assert adjudicated["metric_validity"] == "medium"
    assert adjudicated["selection_eligible"] is True
    assert adjudicated["metric_validity_source"] == "metric_validity_llm"


def test_opt_solver_evaluator_ok_without_metric_direction_is_not_high() -> None:
    fields = {
        "task_profile": "opt_solver",
        "evaluator_backend": "artifact_command",
        "evaluator_status": "ok",
        "metric_value": 820.828603,
        "metric_name": "tomato_mass_kg",
        "lower_is_better": None,
        "validation_ok": "1",
        "val_score_type": "benchmark",
        "selection_eligible": "1",
        "candidate_ready": "1",
    }

    adjudicated = adjudicate_metric_validity(fields)

    assert adjudicated["metric_validity"] == "medium"
    assert adjudicated["selection_eligible"] is False
    assert adjudicated["metric_validity_reason_code"] == "uncertain"
    assert "opt_solver_metric_direction_missing" in adjudicated["metric_validity_note"]


def test_metric_adjudicator_llm_high_cannot_upgrade_unknown_protocol() -> None:
    fields = {
        "metric_value": 0.91,
        "metric_name": "Final Validation Score",
        "validation_ok": "1",
        "val_score_type": "unknown",
        "selection_eligible": "1",
        "candidate_ready": "1",
        "metric_validity": "medium",
        "metric_source_note": "type=cv; protocol=unknown; eval=unknown",
    }
    judgment = MetricValidityJudgment(
        metric_validity="high",
        selection_eligible=True,
        reason_code="comparable_cv",
        reason="Claims CV, but structured fields are still unknown.",
        confidence="high",
        source="metric_validity_llm",
    )

    adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)

    assert adjudicated["metric_validity"] == "medium"
    assert adjudicated["selection_eligible"] is False
    assert "conservative_merge_from_high_to_medium" in adjudicated["metric_validity_note"]


def test_metric_validity_adjudicator_parses_direction_fields() -> None:
    judgment = parse_metric_validity_judgment_text(
        '{"metric_validity":"medium","selection_eligible":false,'
        '"reason_code":"unknown_protocol","reason":"protocol unclear",'
        '"confidence":"high","expected_lower_is_better":false,'
        '"lower_is_better_ok":false,"metric_direction_reason":"Kendall tau is higher better"}'
    )

    assert judgment is not None
    assert judgment.expected_lower_is_better is False
    assert judgment.lower_is_better_ok is False
    assert judgment.metric_direction_reason == "Kendall tau is higher better"


def test_metric_validity_adjudicator_prompts_render() -> None:
    system = build_metric_validity_adjudicator_system_prompt()
    user = build_metric_validity_adjudicator_prompt({"metric_value": 0.5})

    assert "MetricValidityAdjudicator" in system
    assert "Return JSON only" in system
    assert '"metric_value": 0.5' in user


def test_metric_output_interpreter_parses_and_renders() -> None:
    interpretation = parse_metric_output_interpretation_text(
        '{"metric_found":true,"metric_name":"weighted_auc","metric_value":0.641136,'
        '"split":"validation","is_final":true,'
        '"evidence_line":"Best Validation wAUC: 0.641136",'
        '"confidence":"high","reason":"explicit best validation metric"}'
    )

    assert interpretation is not None
    assert interpretation.metric_found is True
    assert interpretation.metric_value == 0.641136
    assert interpretation.evidence_line == "Best Validation wAUC: 0.641136"
    system = build_metric_output_interpreter_system_prompt()
    user = build_metric_output_interpreter_prompt({"stdout_tail": "Best Validation wAUC: 0.641136"})
    assert "MetricOutputInterpreter" in system
    assert "untrusted" in system
    assert "Best Validation wAUC" in user


def test_metric_output_interpreter_callback_uses_isolated_feedback_llm() -> None:
    class FakeFeedbackLlm:
        def __init__(self) -> None:
            self.call: dict[str, object] = {}

        async def ask_tool_stream(self, **kwargs: object) -> SimpleNamespace:
            self.call = dict(kwargs)
            return SimpleNamespace(
                content=(
                    '{"metric_found":true,"metric_name":"weighted_auc",'
                    '"metric_value":0.641136,"split":"validation","is_final":true,'
                    '"evidence_line":"best_val_wauc=0.641136",'
                    '"confidence":"high","reason":"explicit best validation metric"}'
                )
            )

    fake_llm = FakeFeedbackLlm()
    solver = LnrSolver.__new__(LnrSolver)
    solver.lhr = SimpleNamespace(
        metric_validity_adjudicator_enabled=True,
        metric_validity_adjudicator_timeout_sec=60.0,
    )
    solver.task_desc = "Evaluation metric is weighted AUC; higher is better."
    solver._metric_validity_feedback_llm = fake_llm
    solver._jsonl = lambda *_args, **_kwargs: None
    solver._accumulate_ephemeral_tokens = lambda *_args, **_kwargs: None

    result = asyncio.run(
        solver._metric_output_interpretation_callback(
            agent=SimpleNamespace(),
            stdout="best_val_wauc=0.641136\n",
            script_label="solution.py",
        )
    )

    assert result is not None
    assert result["metric_value"] == 0.641136
    assert result["evidence_line"] == "best_val_wauc=0.641136"
    assert "lower_is_better" not in result
    assert fake_llm.call["tools"] == []
    assert fake_llm.call["tool_choice"] == "none"
    prompt = fake_llm.call["messages"][0].content
    assert "best_val_wauc=0.641136" in prompt
    assert "weighted AUC" in prompt
