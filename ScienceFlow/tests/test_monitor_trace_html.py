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
import json
from pathlib import Path

from scienceflow.ui.monitor_trace.builder import _format_llm_display, build_monitor_trace_report
from scienceflow.ui.monitor_trace.runner import run_monitor_trace


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, workspace_base: Path) -> None:
    path.write_text(
        "defaults:\n"
        "  gpu_list: none\n"
        "tasks:\n"
        "- exp_id: demo-task\n"
        "  run_id: run-a\n"
        f"  workspace_base: {workspace_base}\n",
        encoding="utf-8",
    )


def test_monitor_trace_builds_html_and_cache_from_existing_logs(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces"
    task_root = workspace_base / "run-a" / "demo-task"
    manifest = tmp_path / "monitor.yaml"
    _write_manifest(manifest, workspace_base)
    _write_csv(
        task_root / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "row_order": "1",
                "candidate_id": "W00:L01:S001",
                "worker_id": "W00",
                "stage_id": "S001",
                "metric_value": "0.40",
                "metric_name": "score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
                "created_at_utc": "2026-06-25T01:00:00Z",
            },
            {
                "row_order": "2",
                "candidate_id": "W00:L01:S002",
                "worker_id": "W00",
                "stage_id": "S002",
                "metric_value": "0.50",
                "metric_name": "score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "0",
                "metric_validity": "medium",
                "created_at_utc": "2026-06-25T02:00:00Z",
            },
            {
                "row_order": "3",
                "candidate_id": "W00:L01:S003",
                "worker_id": "W00",
                "stage_id": "S003",
                "metric_value": "0.70",
                "metric_name": "score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
                "created_at_utc": "2026-06-25T03:00:00Z",
            },
        ],
    )
    events = task_root / "task_logs" / "lhr_events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp": 1782356400,
                "payload": {"action": "switch_stage"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "trace.html"
    cache = tmp_path / "trace.json"
    run_monitor_trace(manifest_path=manifest, output_path=output, cache_path=cache, refresh_sec=30, once=True)

    html = output.read_text(encoding="utf-8")
    data = json.loads(cache.read_text(encoding="utf-8"))
    task = data["tasks"][0]
    assert "ScienceFlow Monitor Trace" in html
    assert "<svg" in html
    assert "stage metric points" in html
    assert "opacity:.6" in html
    assert "Y range" in html
    assert "Valid range" in html
    embedded = html.split('id="monitor-trace-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["tasks"][0]["exp_id"] == "demo-task"
    assert task["exp_id"] == "demo-task"
    assert task["lower_is_better"] is False
    assert task["x_mode"] == "wall_clock"
    assert task["best_points"][-1]["metric"] == 0.70
    assert task["best_points"][-1]["kind"] == "valid_best"
    assert task["events"][0]["kind"] == "estra"


def test_monitor_trace_does_not_fallback_to_raw_best_without_verified_points(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces"
    task_root = workspace_base / "run-a" / "demo-task"
    manifest = tmp_path / "monitor.yaml"
    _write_manifest(manifest, workspace_base)
    _write_csv(
        task_root / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "row_order": "1",
                "candidate_id": "W00:L01:S001",
                "worker_id": "W00",
                "stage_id": "S001",
                "metric_value": "0.40",
                "metric_name": "score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "0",
                "metric_validity": "medium",
                "created_at_utc": "2026-06-25T01:00:00Z",
            },
            {
                "row_order": "2",
                "candidate_id": "W00:L01:S002",
                "worker_id": "W00",
                "stage_id": "S002",
                "metric_value": "0.90",
                "metric_name": "score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "0",
                "metric_validity": "medium",
                "created_at_utc": "2026-06-25T02:00:00Z",
            },
        ],
    )

    report = build_monitor_trace_report(manifest)
    task = report.tasks[0]

    assert len(task.points) == 2
    assert task.best_points == []
    assert task.summary["best_kind"] == "none"


def test_monitor_trace_uses_latest_stage_direction_when_rows_conflict(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces"
    task_root = workspace_base / "run-a" / "demo-task"
    manifest = tmp_path / "monitor.yaml"
    _write_manifest(manifest, workspace_base)
    _write_csv(
        task_root / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "row_order": "1",
                "candidate_id": "W00:L01:S001",
                "worker_id": "W00",
                "stage_id": "S001",
                "metric_value": "0.55",
                "metric_name": "accuracy",
                "lower_is_better": "1",
                "candidate_ready": "0",
                "selection_eligible": "0",
                "metric_validity": "medium",
            },
            {
                "row_order": "2",
                "candidate_id": "W00:L01:S002",
                "worker_id": "W00",
                "stage_id": "S002",
                "metric_value": "0.30",
                "metric_name": "accuracy",
                "lower_is_better": "0",
                "candidate_ready": "0",
                "selection_eligible": "0",
                "metric_validity": "medium",
            },
        ],
    )

    task = build_monitor_trace_report(manifest).tasks[0]

    assert task.lower_is_better is False
    assert task.summary["metric_direction_source"] == "metric_text_after_conflict"
    assert task.summary["metric_direction_conflict"] is True
    assert task.summary["best_raw_metric"] == 0.55


def test_monitor_trace_rejects_latest_wrong_direction_for_accuracy(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces"
    task_root = workspace_base / "run-a" / "demo-task"
    manifest = tmp_path / "monitor.yaml"
    _write_manifest(manifest, workspace_base)
    _write_csv(
        task_root / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "row_order": "1",
                "candidate_id": "W00:L01:S001",
                "worker_id": "W00",
                "stage_id": "S001",
                "metric_value": "0.60",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "brief": "Holdout accuracy baseline",
                "candidate_ready": "0",
                "selection_eligible": "0",
                "metric_validity": "medium",
            },
            {
                "row_order": "2",
                "candidate_id": "W01:L01:S001",
                "worker_id": "W01",
                "stage_id": "S001",
                "metric_value": "0.48",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "brief": "Quick validation accuracy",
                "candidate_ready": "0",
                "selection_eligible": "0",
                "metric_validity": "medium",
            },
            {
                "row_order": "3",
                "candidate_id": "W01:L01:S002",
                "worker_id": "W01",
                "stage_id": "S002",
                "metric_value": "0.63",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "brief": "Improved held-out validation accuracy",
                "candidate_ready": "0",
                "selection_eligible": "0",
                "metric_validity": "high",
            },
        ],
    )

    task = build_monitor_trace_report(manifest).tasks[0]

    assert task.lower_is_better is False
    assert task.summary["metric_direction_source"] == "metric_text_after_conflict"
    assert task.summary["best_raw_metric"] == 0.63


def test_monitor_trace_prefers_task_description_over_stage_majority(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces"
    task_root = workspace_base / "run-a" / "demo-task"
    manifest = tmp_path / "monitor.yaml"
    _write_manifest(manifest, workspace_base)
    description = task_root / "workers" / "w00" / "workspace" / "description.md"
    description.parent.mkdir(parents=True)
    description.write_text("The metric is categorization accuracy.\n", encoding="utf-8")
    _write_csv(
        task_root / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "row_order": str(index),
                "candidate_id": f"W00:L01:S00{index}",
                "worker_id": "W00",
                "stage_id": f"S00{index}",
                "metric_value": str(value),
                "metric_name": "Final Validation Score",
                "lower_is_better": lower,
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
            for index, value, lower in (
                (1, 0.60, "0"),
                (2, 0.45, "1"),
                (3, 0.65, "1"),
            )
        ],
    )

    task = build_monitor_trace_report(manifest).tasks[0]

    assert task.lower_is_better is False
    assert task.summary["metric_direction_source"] == "task_description"
    assert task.summary["metric_direction_conflict"] is True
    assert task.summary["best_raw_metric"] == 0.65
    assert task.summary["best_valid_metric"] == 0.65


def test_monitor_trace_builder_handles_missing_stage_csv(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces"
    (workspace_base / "run-a" / "demo-task" / "task_logs").mkdir(parents=True)
    manifest = tmp_path / "monitor.yaml"
    _write_manifest(manifest, workspace_base)

    report = build_monitor_trace_report(manifest)

    assert len(report.tasks) == 1
    assert report.tasks[0].points == []
    assert report.tasks[0].summary["candidate_stage_count"] == 0


def test_monitor_trace_llm_display_hides_endpoint_and_route_details() -> None:
    text = _format_llm_display(
        {
            "llm_config": {
                "code": {
                    "model": "deepseek-v4-pro",
                    "base_url": "https://api.deepseek.com",
                    "routing_mode": "sticky_failover",
                },
                "feedback": {
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com",
                    "routing_mode": "sticky_failover",
                },
            },
            "llm_config_text": "code=old url=https://example route=sticky",
        }
    )

    assert text == "code=deepseek-v4-pro\nfeedback=deepseek-v4-flash"
    assert "url=" not in text
    assert "route=" not in text
