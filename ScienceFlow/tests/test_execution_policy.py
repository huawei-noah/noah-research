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

"""Unit tests for execution_policy (no real training)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scienceflow.safety.execution_policy import (
    FULL_RUN_STAMP_VERSION,
    FULL_RUN_SOURCE_SAFETY,
    _extract_metric_from_stdout,
    _has_timeout_marker,
    _should_skip_full_run,
    _write_full_run_stamp,
    final_validation_score_contract_error,
    format_score_contract_repair_feedback,
    parse_lower_is_better_from_result_md,
    resolve_lower_is_better_for_bare_snapshot,
)


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("[safety] TIMEOUT after 5s\n", "", True),
        ("", "[external-runner] TIMEOUT after 120s\n", True),
        ("TIMEOUT after 5s\n", "", False),
        ("log mentions [runner] TIMEOUT after 5s inline\n", "", False),
    ],
)
def test_has_timeout_marker(stdout: str, stderr: str, expected: bool) -> None:
    assert _has_timeout_marker(stdout, stderr) is expected


def test_should_not_skip_when_quick_style_artifacts_present(tmp_path: Path) -> None:
    """Quick-test can leave artifacts and a metric; the safety policy must still run."""
    (tmp_path / "solution.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "submission.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "result.md").write_text(
        "## Results\nmetric_value: 0.5\nlower_is_better: false\n",
        encoding="utf-8",
    )
    skip, reason = _should_skip_full_run(tmp_path)
    assert skip is False
    assert reason == ""


def test_should_skip_without_solution(tmp_path: Path) -> None:
    (tmp_path / "result.md").write_text("## Results\nmetric_value: 0.5\n", encoding="utf-8")
    skip, reason = _should_skip_full_run(tmp_path)
    assert skip is True
    assert "solution" in reason.lower()


def test_should_skip_when_agent_stamp_present(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("print(1)", encoding="utf-8")
    _write_full_run_stamp(tmp_path, "agent", metric_value=0.1)
    skip, reason = _should_skip_full_run(tmp_path)
    assert skip is True
    assert "stamp" in reason.lower()


def test_metricless_stamp_does_not_skip(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("print(1)", encoding="utf-8")
    logs = tmp_path / ".logs"
    logs.mkdir(parents=True)
    (logs / "full_run_stamp.json").write_text(
        json.dumps({"version": FULL_RUN_STAMP_VERSION, "source": "agent", "unix_ts": 1.0}),
        encoding="utf-8",
    )
    skip, _ = _should_skip_full_run(tmp_path)
    assert skip is False


def test_invalid_stamp_does_not_skip(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("print(1)", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "full_run_stamp.json").write_text(
        json.dumps({"version": 99, "source": "agent", "unix_ts": 1.0}),
        encoding="utf-8",
    )
    skip, _ = _should_skip_full_run(tmp_path)
    assert skip is False

    (logs / "full_run_stamp.json").write_text(
        json.dumps({"version": FULL_RUN_STAMP_VERSION, "source": "bogus", "unix_ts": 1.0}),
        encoding="utf-8",
    )
    skip, _ = _should_skip_full_run(tmp_path)
    assert skip is False


def test_should_not_skip_without_submission(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "result.md").write_text(
        "## Results\nmetric_value: 0.5\n",
        encoding="utf-8",
    )
    skip, _ = _should_skip_full_run(tmp_path)
    assert skip is False


def test_extract_average_r2() -> None:
    out = "Overall metrics\nAverage R²: 0.1234\n"
    v, name, lower = _extract_metric_from_stdout(out)
    assert v == pytest.approx(0.1234)
    assert "R²" in name
    assert lower is False


def test_extract_final_validation_score_priority_over_rmse() -> None:
    """Contract line must win over earlier RMSE / sub-metrics."""
    out = "RMSE: 0.039\nSome log\nFinal Validation Score: 0.068463\n"
    v, name, lower = _extract_metric_from_stdout(out)
    assert v == pytest.approx(0.068463)
    assert name == "Final Validation Score"
    assert lower is True


def test_extract_final_validation_score_with_auc_logs_is_higher_better() -> None:
    """FVS line is minimization by default, but AUC training logs imply maximize."""
    out = (
        "[EPOCH] 1/30 train_loss=0.33 val_auc=0.89\n"
        "Final Validation Score: 0.93706\n"
    )
    v, name, lower = _extract_metric_from_stdout(out)
    assert v == pytest.approx(0.93706)
    assert name == "Final Validation Score"
    assert lower is False


def test_extract_final_validation_score_with_validation_auc_colon_is_higher_better() -> None:
    """Human-readable Validation AUC logs must also imply maximize."""
    out = (
        "[EVAL] Validation AUC: 0.528999\n"
        "Final Validation Score: 0.5289994855967078\n"
    )
    v, name, lower = _extract_metric_from_stdout(out)
    assert v == pytest.approx(0.5289994855967078)
    assert name == "Final Validation Score"
    assert lower is False


def test_extract_final_validation_score_with_mcc_logs_is_higher_better() -> None:
    out = (
        "[EVAL] Found best threshold: 0.6200 with MCC: 0.774093\n"
        "Final Validation Score: 0.774093\n"
    )
    v, name, lower = _extract_metric_from_stdout(out)
    assert v == pytest.approx(0.774093)
    assert name == "Final Validation Score"
    assert lower is False


def test_extract_legacy_parenthesized_final_validation_score() -> None:
    out = "FINAL VALIDATION SCORE (MCC): 0.61540\n"
    v, name, lower = _extract_metric_from_stdout(out)
    assert v == pytest.approx(0.61540)
    assert name == "Final Validation Score"
    assert lower is False
    assert final_validation_score_contract_error(out) is None


def test_score_contract_ignores_source_template_mentions() -> None:
    out = '207|print(f"FINAL VALIDATION SCORE (MCC): {best_mcc:.5f}")\n'
    v, _name, _lower = _extract_metric_from_stdout(out)
    assert v is None
    assert final_validation_score_contract_error(out) is None
    assert "missing required" in (
        final_validation_score_contract_error(out, require_present=True) or ""
    )


def test_extract_final_validation_score_rejects_mcc_out_of_range() -> None:
    out = (
        "[EVAL] Best threshold: 0.0952, Val MCC: 6.001707\n"
        "Final Validation Score: 6.001706997220164\n"
    )
    v, name, lower = _extract_metric_from_stdout(out)
    assert v is None
    assert name == "Final Validation Score"
    assert lower is False
    err = final_validation_score_contract_error(out)
    assert err is not None
    assert "MCC" in err
    assert "expected validation range" in err


def test_bad_final_validation_score_does_not_fall_back_to_rmse() -> None:
    out = "valid_0's rmse: 0.0325365\nFinal Validation Score: inf\n"
    v, name, lower = _extract_metric_from_stdout(out)
    assert v is None
    assert name == "Final Validation Score"
    assert lower is True
    assert "not finite" in (final_validation_score_contract_error(out) or "")


def test_unparseable_final_validation_score_does_not_fall_back_to_rmse() -> None:
    out = "RMSE: 0.012\nFinal Validation Score: N/A\n"
    v, name, _lower = _extract_metric_from_stdout(out)
    assert v is None
    assert name == "Final Validation Score"
    assert "not parseable" in (final_validation_score_contract_error(out) or "")


def test_score_contract_can_require_final_stdout_line() -> None:
    out = "Final Validation Score: 0.12\nextra diagnostic after score\n"
    assert final_validation_score_contract_error(out) is None
    assert "last non-empty" in (
        final_validation_score_contract_error(out, require_final_line=True) or ""
    )


def test_score_contract_repair_feedback_prefers_existing_artifacts() -> None:
    text = format_score_contract_repair_feedback(
        "missing required `Final Validation Score: <finite_float>` line",
        script_label="train.py",
    )

    assert "[SCORE-CONTRACT-INVALID]" in text
    assert "do not retrain from scratch" in text
    assert "`predict.py` / `score_existing.py`" in text
    assert "loads the existing artifacts" in text
    assert "root `submission.csv`" in text
    assert "cheapest finalization command" in text


def test_resolve_lower_is_better_prefers_context_when_result_md_has_no_direction(
    tmp_path: Path,
) -> None:
    """No explicit ``lower_is_better`` line in result.md → use context.json."""
    (tmp_path / "result.md").write_text(
        "## Results\nmetric_value: 0.5\n",
        encoding="utf-8",
    )
    out = "Final Validation Score: 0.71\n"
    ctx = {"lower_is_better": False}
    lb = resolve_lower_is_better_for_bare_snapshot(
        workspace=tmp_path,
        context_json=ctx,
        stdout=out,
    )
    assert lb is False


def test_resolve_lower_is_better_prefers_explicit_result_md_over_conflicting_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.md").write_text(
        "## Results\nlower_is_better: false\n",
        encoding="utf-8",
    )
    ctx = {"lower_is_better": True}
    out = "Final Validation Score: 0.71\n"
    lb = resolve_lower_is_better_for_bare_snapshot(
        workspace=tmp_path,
        context_json=ctx,
        stdout=out,
    )
    assert lb is False


def test_resolve_lower_is_better_custom_env_before_stdout_heuristic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env overrides stdout when result.md and context do not fix direction."""
    monkeypatch.setenv("CUSTOM_IS_LOWER_BETTER", "false")
    (tmp_path / "result.md").write_text(
        "## Results\nmetric_value: 0.5\n",
        encoding="utf-8",
    )
    out = "Final Validation Score: 0.71\n"
    lb = resolve_lower_is_better_for_bare_snapshot(
        workspace=tmp_path,
        context_json={},
        stdout=out,
    )
    assert lb is False


def test_resolve_lower_is_better_stdout_mcc_overrides_generic_context(tmp_path: Path) -> None:
    (tmp_path / "result.md").write_text(
        "## Results\nmetric_value: 0.5\n",
        encoding="utf-8",
    )
    out = "[EVAL] Val MCC: 0.42\nFinal Validation Score: 0.42\n"
    lb = resolve_lower_is_better_for_bare_snapshot(
        workspace=tmp_path,
        context_json={"lower_is_better": True},
        stdout=out,
    )
    assert lb is False


def test_resolve_lower_is_better_prefers_result_md_over_stdout_heuristic(tmp_path: Path) -> None:
    (tmp_path / "result.md").write_text(
        "## Results\nlower_is_better: false\n",
        encoding="utf-8",
    )
    out = "Final Validation Score: 0.71\n"
    lb = resolve_lower_is_better_for_bare_snapshot(
        workspace=tmp_path,
        context_json={},
        stdout=out,
    )
    assert lb is False


def test_parse_lower_is_better_from_result_md_bold_list(tmp_path: Path) -> None:
    (tmp_path / "result.md").write_text(
        "## Results\n- **lower_is_better**: true\n",
        encoding="utf-8",
    )
    assert parse_lower_is_better_from_result_md(tmp_path / "result.md") is True


@pytest.mark.asyncio
async def test_ensure_full_execution_runs_even_when_artifacts_present(tmp_path: Path) -> None:
    from scienceflow.safety.execution_policy import ensure_full_execution

    (tmp_path / "solution.py").write_text(
        'print("Final Validation Score: 0.9")\n',
        encoding="utf-8",
    )
    (tmp_path / "submission.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "result.md").write_text(
        "## Results\nmetric_value: 0.9\nlower_is_better: false\nMETRIC: 0.9\n",
        encoding="utf-8",
    )
    prior_md = (tmp_path / "result.md").read_text(encoding="utf-8")
    r = await ensure_full_execution(tmp_path, timeout_sec=5.0, extra_env=None)
    assert r.skipped is False
    assert r.executed is True
    assert r.exit_code == 0
    assert r.metric_value == pytest.approx(0.9)
    assert (tmp_path / "result.md").read_text(encoding="utf-8") == prior_md
    stamp_path = tmp_path / ".logs" / "full_run_stamp.json"
    assert stamp_path.is_file()
    data = json.loads(stamp_path.read_text(encoding="utf-8"))
    assert data["version"] == FULL_RUN_STAMP_VERSION
    assert data["source"] == FULL_RUN_SOURCE_SAFETY
    assert data.get("metric_value") == pytest.approx(0.9)
    assert data.get("metric_name") == "Final Validation Score"
    assert data.get("lower_is_better") is True


@pytest.mark.asyncio
async def test_ensure_full_execution_no_stamp_on_nonfinite_final_score(tmp_path: Path) -> None:
    from scienceflow.safety.execution_policy import ensure_full_execution

    (tmp_path / "solution.py").write_text(
        "print(\"valid_0's rmse: 0.0325365\")\nprint('Final Validation Score: inf')\n",
        encoding="utf-8",
    )
    r = await ensure_full_execution(tmp_path, timeout_sec=5.0, extra_env=None)
    assert r.executed is True
    assert r.exit_code == 0
    assert r.reason == "score_contract_failed"
    assert r.metric_value is None
    assert "not finite" in r.score_contract_error
    assert "do not retrain from scratch" in r.stderr
    assert "predict.py" in r.stderr
    assert not (tmp_path / ".logs" / "full_run_stamp.json").is_file()


@pytest.mark.asyncio
async def test_ensure_full_execution_no_stamp_on_mcc_out_of_range(tmp_path: Path) -> None:
    from scienceflow.safety.execution_policy import ensure_full_execution

    (tmp_path / "solution.py").write_text(
        "print('[EVAL] Val MCC: 6.001707')\n"
        "print('Final Validation Score: 6.001706997220164')\n",
        encoding="utf-8",
    )
    r = await ensure_full_execution(tmp_path, timeout_sec=5.0, extra_env=None)
    assert r.executed is True
    assert r.exit_code == 0
    assert r.reason == "score_contract_failed"
    assert r.metric_value is None
    assert "MCC" in r.score_contract_error
    assert not (tmp_path / ".logs" / "full_run_stamp.json").is_file()


@pytest.mark.asyncio
async def test_ensure_full_execution_no_stamp_on_nonzero_exit(tmp_path: Path) -> None:
    from scienceflow.safety.execution_policy import ensure_full_execution

    (tmp_path / "solution.py").write_text(
        "import sys\nsys.exit(1)\n",
        encoding="utf-8",
    )
    r = await ensure_full_execution(tmp_path, timeout_sec=5.0, extra_env=None)
    assert r.executed is True
    assert r.exit_code == 1
    assert not (tmp_path / ".logs" / "full_run_stamp.json").is_file()


@pytest.mark.asyncio
async def test_ensure_full_execution_skips_when_stamp_present(tmp_path: Path) -> None:
    from scienceflow.safety.execution_policy import ensure_full_execution

    (tmp_path / "solution.py").write_text('print("METRIC: 1.0")\n', encoding="utf-8")
    _write_full_run_stamp(tmp_path, "agent", metric_value=1.0)
    r = await ensure_full_execution(tmp_path, timeout_sec=5.0, extra_env=None)
    assert r.skipped is True
    assert r.executed is False
    assert "stamp" in r.reason.lower()
