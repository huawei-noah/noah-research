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

from scienceflow.core.agent.run_control.embedded_fullrun import (
    _score_contract_errors,
    _validate_llm_metric_interpretation,
)


def test_run_control_accepts_parseable_score_that_is_not_last_stdout_line() -> None:
    hard, soft = _score_contract_errors(
        "training ok\nFinal Validation Score: 0.67067\nSubmission is valid.\n",
    )

    assert hard is None
    assert soft is not None
    assert "last non-empty" in soft


def test_run_control_keeps_missing_score_as_hard_error() -> None:
    hard, soft = _score_contract_errors("training ok\nSubmission is valid.\n")

    assert "missing required" in (hard or "")
    assert soft is None


def test_run_control_keeps_nonfinite_score_as_hard_error() -> None:
    hard, soft = _score_contract_errors("Final Validation Score: nan\n")

    assert "not finite" in (hard or "")
    assert soft is None


def test_llm_metric_interpretation_requires_verbatim_numeric_evidence() -> None:
    stdout = "Best Validation wAUC: 0.641136\n"
    interpretation = {
        "metric_found": True,
        "metric_name": "weighted_auc",
        "metric_value": 0.641136,
        "split": "validation",
        "is_final": True,
        "evidence_line": "Best Validation wAUC: 0.641136",
        "confidence": "high",
    }

    value, name, error = _validate_llm_metric_interpretation(stdout, interpretation)

    assert error is None
    assert value == 0.641136
    assert name == "weighted_auc"


def test_llm_metric_interpretation_rejects_hallucinated_value() -> None:
    stdout = "Best Validation wAUC: 0.641136\n"
    interpretation = {
        "metric_found": True,
        "metric_name": "weighted_auc",
        "metric_value": 0.673,
        "split": "validation",
        "is_final": True,
        "evidence_line": "Best Validation wAUC: 0.641136",
        "confidence": "high",
    }

    value, _, error = _validate_llm_metric_interpretation(stdout, interpretation)

    assert value is None
    assert "does not occur" in (error or "")


def test_llm_metric_interpretation_rejects_test_score() -> None:
    stdout = "best_private_test_auc=0.99\n"
    interpretation = {
        "metric_found": True,
        "metric_name": "auc",
        "metric_value": 0.99,
        "split": "validation",
        "is_final": True,
        "evidence_line": "best_private_test_auc=0.99",
        "confidence": "high",
    }

    value, _, error = _validate_llm_metric_interpretation(stdout, interpretation)

    assert value is None
    assert "test or leaderboard" in (error or "")


def test_llm_metric_interpretation_rejects_out_of_range_auc() -> None:
    stdout = "Best Validation AUC: 1.5\n"
    interpretation = {
        "metric_found": True,
        "metric_name": "auc",
        "metric_value": 1.5,
        "split": "validation",
        "is_final": True,
        "evidence_line": "Best Validation AUC: 1.5",
        "confidence": "high",
    }

    value, _, error = _validate_llm_metric_interpretation(stdout, interpretation)

    assert value is None
    assert "expected validation range" in (error or "")


def test_llm_metric_interpretation_rejects_source_template() -> None:
    stdout = 'print("Best Validation wAUC: 0.641136")\n'
    interpretation = {
        "metric_found": True,
        "metric_name": "weighted_auc",
        "metric_value": 0.641136,
        "split": "validation",
        "is_final": True,
        "evidence_line": 'print("Best Validation wAUC: 0.641136")',
        "confidence": "high",
    }

    value, _, error = _validate_llm_metric_interpretation(stdout, interpretation)

    assert value is None
    assert "source-code template" in (error or "")
