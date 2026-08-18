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

from scienceflow.config.settings import apply_profile_overrides, load_cfg
from scienceflow.gates.evaluator import EvalContext, EvaluatorManager
from scienceflow.gates.evaluator.adapters import (
    merge_adjudicated_stage_facts,
    merge_primary_stage_facts,
    metric_event_to_stage_facts,
)
from scienceflow.gates.evaluator.models import (
    EvalContext,
    EvaluationRequest,
    GateDecision,
    MetricEvent,
)
from scienceflow.gates.service import GateService
from scienceflow.solver.lnr.prompts import build_first_user_prompt


def test_default_evaluator_config_is_task_neutral() -> None:
    cfg = load_cfg(cli_args=False)

    assert cfg.evaluator.enabled is True
    assert cfg.evaluator.expose_wall_clock_remaining_sec is False
    assert cfg.evaluator.backend == "auto"
    assert cfg.evaluator.task_profile == "auto"
    assert cfg.evaluator.stage_source_mode == "primary"
    assert cfg.evaluator.candidate.artifact == ""
    assert cfg.evaluator.metric.name == "metric"


def test_task_package_profile_overrides_generic_default() -> None:
    cfg = load_cfg(cli_args=False)
    cfg.exp_id = "ratio-minimization"

    apply_profile_overrides(cfg)

    assert cfg.evaluator.task_profile == "opt_solver"
    assert cfg.evaluator.backend == "task_package"
    assert cfg.evaluator.candidate.artifact == "artifacts/best_solution.json"
    assert cfg.evaluator.metric.name == "inv_ratio_squared"


def test_unknown_task_does_not_fall_back_to_mlebench() -> None:
    cfg = load_cfg(cli_args=False)
    cfg.exp_id = "unregistered-task"

    apply_profile_overrides(cfg)

    assert cfg.evaluator.task_profile == "auto"
    assert cfg.evaluator.backend == "auto"
    assert cfg.evaluator.candidate.artifact == ""

    ctx = EvalContext(
        task_profile=cfg.evaluator.task_profile,
        task_id=cfg.exp_id,
        task_root=Path("."),
        workspace=Path("."),
        worker_id="W00",
        cfg=cfg,
    )
    assert EvaluatorManager.default().backend_name_for_context(ctx) == ""


def test_evaluator_config_loads_nested_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
evaluator:
  task_profile: opt_solver
  backend: artifact_command
  stage_source_mode: primary
  candidate:
    artifact: artifacts/best_solution.json
    require_sha: true
  metric:
    name: cost
    lower_is_better: true
  command:
    evaluator_command: python eval.py --candidate {artifact_path}
    python_executable: /opt/envs/kttsp/bin/python
    environment_path: /opt/envs/kttsp
""",
        encoding="utf-8",
    )

    cfg = load_cfg(config, cli_args=False)

    assert cfg.evaluator.backend == "artifact_command"
    assert cfg.evaluator.candidate.require_sha is True
    assert cfg.evaluator.metric.name == "cost"
    assert cfg.evaluator.metric.lower_is_better is True
    assert cfg.evaluator.command.python_executable == "/opt/envs/kttsp/bin/python"
    assert cfg.evaluator.command.environment_path == "/opt/envs/kttsp"


def test_metric_event_to_stage_facts_maps_common_fields() -> None:
    event = MetricEvent(
        candidate_id="W00:S01",
        worker_id="W00",
        stage_id="S01",
        metric_value=1.5,
        metric_name="cost",
        lower_is_better=True,
        validation_ok=True,
        candidate_ready=True,
        selection_eligible=True,
        metric_validity="high",
        metric_validity_reason_code="ok",
        artifact_path="artifacts/best.json",
        artifact_sha="abc",
        evaluator_backend="artifact_command",
        evaluator_status="ok",
        metric_type="benchmark",
        task_profile="opt_solver",
        metric_note="parsed",
        run_time_sec=2.0,
        extra={"artifact_kind": "json_solution", "metric_authoritative": True},
    )

    facts = metric_event_to_stage_facts(event)

    assert facts["metric_value"] == 1.5
    assert facts["artifact_path"] == "artifacts/best.json"
    assert facts["artifact_sha"] == "abc"
    assert facts["artifact_kind"] == "json_solution"
    assert facts["evaluator_backend"] == "artifact_command"
    assert facts["task_profile"] == "opt_solver"
    assert facts["val_score_type"] == "benchmark"
    assert facts["metric_authoritative"] is True


def test_metric_event_adapter_exposes_only_explicit_agent_visible_extra() -> None:
    event = MetricEvent(
        candidate_id="candidate",
        worker_id="W00",
        stage_id="S01",
        metric_value=0.5,
        metric_name="score",
        lower_is_better=False,
        validation_ok=True,
        candidate_ready=True,
        selection_eligible=True,
        metric_validity="high",
        metric_validity_reason_code="ok",
        artifact_path="artifacts/submission.json",
        artifact_sha="abc",
        evaluator_backend="task_package",
        evaluator_status="ok",
        metric_type="benchmark",
        extra={
            "agent_visible": {"normalized_enrichment": 0.7},
            "queries_remaining": 3,
            "query_limit": 10,
            "query_budget_exhausted": False,
            "hidden_lookup_size": 60_000,
        },
    )

    facts = metric_event_to_stage_facts(event)

    assert facts["extra"] == {
        "normalized_enrichment": 0.7,
        "queries_remaining": 3,
        "query_limit": 10,
        "query_budget_exhausted": False,
    }
    assert "hidden_lookup_size" not in facts["extra"]


def test_gate_wall_clock_disclosure_is_opt_in(tmp_path: Path) -> None:
    class _GateManager:
        @staticmethod
        def decide_with_trace(_ctx, _event, *, trigger):
            _ = trigger
            return (
                GateDecision(
                    action="accept",
                    accepted=True,
                    candidate_ready=True,
                    selection_eligible=True,
                    reason_code="ok",
                ),
                {"policy": "test"},
            )

    event = MetricEvent(
        candidate_id="candidate",
        worker_id="W00",
        stage_id="S01",
        metric_value=0.5,
        metric_name="score",
        lower_is_better=False,
        validation_ok=True,
        candidate_ready=True,
        selection_eligible=True,
        metric_validity="high",
        metric_validity_reason_code="ok",
        artifact_path="artifacts/submission.json",
        artifact_sha="abc",
        evaluator_backend="task_package",
        evaluator_status="ok",
        metric_type="benchmark",
        extra={
            "wall_clock_remaining_sec": 999,
            "agent_visible": {
                "wall_clock_remaining_sec": 998,
                "safe_task_metric": 0.7,
            },
        },
    )
    service = GateService(gate_manager=_GateManager())

    for enabled in (False, True):
        cfg = load_cfg(cli_args=False)
        cfg.evaluator.expose_wall_clock_remaining_sec = enabled
        request = EvaluationRequest(
            context=EvalContext(
                task_profile="test",
                task_id="test",
                task_root=tmp_path,
                workspace=tmp_path,
                worker_id="W00",
                cfg=cfg,
                wall_clock_remaining_sec=42.9,
            ),
        )
        outcome = service._evaluate_event(request, event)
        if enabled:
            assert outcome.event.extra["wall_clock_remaining_sec"] == 42
        else:
            assert "wall_clock_remaining_sec" not in outcome.event.extra
        assert outcome.event.extra["agent_visible"] == {"safe_task_metric": 0.7}


def test_adjudicated_merge_preserves_legacy_metric() -> None:
    legacy = {"metric_value": 0.9, "selection_eligible": True}
    facts = {"metric_value": 0.1, "validation_ok": False, "selection_eligible": False}

    merged = merge_adjudicated_stage_facts(legacy, facts)

    assert merged["metric_value"] == 0.9
    assert merged["validation_ok"] is False
    assert merged["selection_eligible"] is False


def test_primary_merge_uses_evaluator_metric() -> None:
    legacy = {"metric_value": 0.9, "solution_sha": "src"}
    facts = {"metric_value": 0.1, "artifact_sha": "artifact"}

    merged = merge_primary_stage_facts(legacy, facts)

    assert merged["metric_value"] == 0.1
    assert merged["solution_sha"] == "src"
    assert merged["artifact_sha"] == "artifact"


def test_first_user_prompt_includes_evaluator_contract() -> None:
    prompt = build_first_user_prompt(
        "Solve task.",
        wall_clock_budget_sec=120,
        evaluator_contract="Save artifact at `artifacts/best_solution.json`.",
    )

    assert "Evaluator artifact contract" in prompt
    assert "artifacts/best_solution.json" in prompt
