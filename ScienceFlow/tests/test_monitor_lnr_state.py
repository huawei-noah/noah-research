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

import csv
import json
import os
import time
from pathlib import Path

from scienceflow.ui.monitor.dashboard import MonitorDashboard
from scienceflow.ui.monitor.helpers import _load_state
from scienceflow.ui.monitor.lnr_state import (
    _display_status,
    _elapsed_from_live_process_summary,
    _elapsed_from_resource_range,
    _load_live_run_started_at,
    _status_from_final_state,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_monitor_state_falls_back_to_lnr_task_logs(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.80",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            },
            {
                "candidate_id": "W01:L01:S01",
                "worker_id": "W01",
                "metric_value": "0.75",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "0",
                "metric_validity": "medium",
            },
        ],
    )
    _write_csv(
        task / "task_logs" / "scienceflow_time_trace.csv",
        [
            {
                "category": "llm_api",
                "operation": "ask_tool_stream",
                "tokens_input": "1000",
                "tokens_output": "50",
                "tokens_cached": "900",
                "ttft_sec": "1.2",
                "tpot_ms": "35.5",
                "llm_cost_usd": "0.00123",
            }
        ],
    )
    resource_path = task / "task_logs" / "resource" / "resource_events.jsonl"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_text(
        json.dumps(
            {
                "event_type": "resource_review_outcome",
                "worker_id": "W00",
                "command_id": "W00:bash:00001",
                "payload": {
                    "execution_outcome": "NO_ACTION",
                    "resource_review_signal": {
                        "active_work": True,
                        "useful_progress": True,
                        "gpu_bucket": "busy",
                        "cpu_bucket": "medium",
                        "metric_changed": True,
                    },
                    "progress_snapshot": {"runtime_sec": 123, "stdout_lines": 10},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    estra_path = task / "task_logs" / "lhr_events.jsonl"
    estra_path.write_text(
        json.dumps({"event": "text_only_estra_check", "payload": {"trigger_source": "text_only"}})
        + "\n"
        + json.dumps({"event": "context_limit_estra_check", "payload": {"trigger_source": "context_limit"}})
        + "\n"
        + json.dumps({"event": "cache_hygiene_compact_triggered", "payload": {"trigger_source": "context_hygiene"}})
        + "\n"
        + json.dumps({"event": "estra_decision", "payload": {"action": "keep_current", "startpoint": "current_workspace", "intent": "continue", "compact": True}})
        + "\n"
        + json.dumps({"event": "estra_decision", "payload": {"action": "keep_but_redirect", "startpoint": "current_workspace", "intent": "redirect", "compact": True}})
        + "\n"
        + json.dumps({"event": "estra_decision", "payload": {"action": "switch_stage", "startpoint": "previous_stage", "intent": "redirect", "target_stage": "S01"}})
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["monitor_kind"] == "lnr_task"
    assert state["stage_count"] == 2
    assert state["best_raw_metric"] == 0.80
    assert state["best_valid_metric"] == 0.80
    assert state["resource_outcomes"] == {"NO_ACTION": 1}
    assert state["resource_actionable_outcomes"] == {}
    assert state["active_job_count"] == 1
    assert state["llm_cache_rate"] == 0.9
    assert state["total_llm_cost_usd"] == 0.00123
    assert state["llm_cost_known_calls"] == 1
    assert state["max_ttft_sec"] == 1.2
    assert state["max_tpot_ms"] == 35.5
    assert state["estra_decision_count"] == 3
    assert state["estra_continue_count"] == 1
    assert state["estra_redirect_count"] == 1
    assert state["estra_switch_count"] == 1
    assert state["estra_current_continue_count"] == 1
    assert state["estra_current_redirect_count"] == 1
    assert state["estra_stage_continue_count"] == 0
    assert state["estra_stage_redirect_count"] == 1
    assert state["estra_compact_count"] == 2
    assert state["estra_text_only_check_count"] == 1
    assert state["estra_context_check_count"] == 2
    assert state["estra_text_only_trigger_count"] == 0
    assert state["estra_context_trigger_count"] == 0



def test_monitor_estra_counts_deduplicate_repeated_stage_decisions(tmp_path: Path) -> None:
    task = tmp_path / "estra-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "1.0",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    event_lines = [
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp": 1000.0,
                "worker_id": "W00",
                "payload": {
                    "action": "keep_but_redirect",
                    "startpoint": "current_workspace",
                    "intent": "redirect",
                    "trigger_source": "text_only",
                    "latest_stage": "S01",
                    "target_stage": "S01",
                    "reason": "first wording",
                },
            }
        ),
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp": 1010.0,
                "worker_id": "W00",
                "payload": {
                    "action": "keep_but_redirect",
                    "startpoint": "current_workspace",
                    "intent": "redirect",
                    "trigger_source": "text_only",
                    "latest_stage": "S01",
                    "target_stage": "S01",
                    "reason": "same decision with different wording",
                },
            }
        ),
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp": 1020.0,
                "worker_id": "W00",
                "payload": {
                    "action": "keep_current",
                    "startpoint": "current_workspace",
                    "intent": "continue",
                    "trigger_source": "text_only",
                    "latest_stage": "S01",
                },
            }
        ),
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp": 1030.0,
                "worker_id": "W00",
                "payload": {
                    "action": "switch_stage",
                    "startpoint": "previous_stage",
                    "intent": "redirect",
                    "trigger_source": "text_only",
                    "latest_stage": "S02",
                    "target_stage": "S01",
                },
            }
        ),
    ]
    events = task / "task_logs" / "workers" / "w00" / "lhr_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("\n".join(event_lines) + "\n", encoding="utf-8")
    (events.parent / "lhr_state.json").write_text(
        json.dumps(
            {
                "estra_decisions": 99,
                "estra_redirect_count": 99,
                "estra_continue_count": 99,
                "estra_switch_count": 99,
            }
        ),
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["estra_raw_decision_count"] == 4
    assert state["estra_repeat_decision_count"] == 1
    assert state["estra_decision_count"] == 3
    assert state["estra_redirect_count"] == 1
    assert state["estra_continue_count"] == 1
    assert state["estra_switch_count"] == 1
    assert state["estra_text_only_trigger_count"] == 3


def test_monitor_stage_count_deduplicates_resume_lineage_candidates(tmp_path: Path) -> None:
    task = tmp_path / "resume-task"
    rows = []
    for candidate_id, worker_id, stage_id, metric in [
        ("W00:L01:S01", "W00", "S01", "1.0"),
        ("W00:L02:S01", "W00", "S01", "1.1"),
        ("W00:L02:S02", "W00", "S02", "0.9"),
        ("W01:L01:S01", "W01", "S01", "2.0"),
    ]:
        rows.append(
            {
                "candidate_id": candidate_id,
                "worker_id": worker_id,
                "stage_id": stage_id,
                "metric_value": metric,
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        )
    _write_csv(task / "task_logs" / "lhr_stage_performance.csv", rows)

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["stage_count"] == 3
    assert state["candidate_stage_count"] == 4
    assert state["stages_by_worker"] == {"W00": 2, "W01": 1}
    assert state["candidate_stages_by_worker"] == {"W00": 3, "W01": 1}
    assert MonitorDashboard()._lnr_stage_cell(state) == "stage=3\ncand=4/4\nelig=4"


def test_monitor_best_cell_shows_valid_before_candidate() -> None:
    cell = MonitorDashboard()._lnr_best_cell(
        {
            "best_raw_metric": 2.480694929,
            "best_raw_candidate": "W01:L07:S38",
            "best_raw_validity": "high",
            "best_valid_metric": 116.911984,
            "best_valid_candidate": "W00:L33:S35",
            "best_valid_validity": "high",
        }
    )

    assert cell.splitlines()[0].startswith("valid 116.9120")
    assert cell.splitlines()[1].startswith("cand 2.4807")


def test_monitor_state_displays_sanitized_llm_config_and_seen_models(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.80",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    _write_csv(
        task / "task_logs" / "scienceflow_time_trace.csv",
        [
            {
                "category": "llm_api",
                "operation": "ask_tool_stream",
                "tokens_input": "10",
                "tokens_output": "2",
                "tokens_cached": "0",
                "detail": "llm_role=code;model=glm-5.2;token_scope=llm_api",
            }
        ],
    )
    log_file = tmp_path / "run.log"
    log_file.write_text(
        "2026-06-23 | INFO | scienceflow | [LLM] glm-5.2: PooledLLM with 1 keys\n"
        "2026-06-23 | INFO | scienceflow | [LLM] deepseek-v4-flash: PooledLLM with 1 keys\n",
        encoding="utf-8",
    )
    (task / "resolved_config.yaml").write_text(
        """agent:
  code:
    model: glm-5.2
    api_key: sk-secret
    base_url: https://user:pass@llm.example/v1?token=secret
    api_routing_mode: sticky_failover
  feedback:
    model: deepseek-v4-flash
    api_keys: [sk-a, sk-b]
    base_urls: [https://fb.example/v1]
""",
        encoding="utf-8",
    )
    (task / "task_logs" / "state.json").write_text(
        json.dumps({"status": "failed", "elapsed_sec": 1.0, "log_file": str(log_file)}),
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["llm_config"]["code"]["model"] == "glm-5.2"
    assert state["llm_config"]["code"]["base_url"] == "https://llm.example/v1"
    assert state["llm_config"]["feedback"]["key_count"] == 2
    assert state["observed_llm_models"] == ["glm-5.2", "deepseek-v4-flash"]
    assert "sk-secret" not in state["llm_config_text"]
    assert "pass" not in state["llm_config_text"]
    assert "token=secret" not in state["llm_config_text"]
    dashboard = MonitorDashboard()
    cell = dashboard._lnr_llm_cell(state)
    assert cell == "code=glm-5.2\nfeedback=deepseek-v4-flash"
    task_cell = dashboard._lnr_task_cell("demo-task-run", state)
    assert "demo-task-run" in task_cell
    assert "code=glm-5.2" in task_cell
    assert "feedback=deepseek-v4-flash" in task_cell
    assert "seen=" not in task_cell


def test_monitor_state_prefers_running_state_llm_config_over_stale_resolved_config(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.80",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    (task / "resolved_config.yaml").write_text(
        """agent:
  code:
    model: gpt-4o
  feedback:
    model: deepseek-v4-flash
""",
        encoding="utf-8",
    )
    (task / "task_logs" / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "elapsed_sec": 1.0,
                "llm_config": {
                    "code": {
                        "model": "",
                        "models": ["deepseek-v4-pro", "glm-5.1"],
                    },
                    "feedback": {"model": "deepseek-v4-flash"},
                },
            }
        ),
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["llm_config"]["code"]["models"] == ["deepseek-v4-pro", "glm-5.1"]
    assert MonitorDashboard()._lnr_llm_cell(state) == "code=deepseek-v4-pro+2\nfeedback=deepseek-v4-flash"


def test_live_monitor_elapsed_prefers_latest_run_start_over_recent_event_window(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    event_path = task / "task_logs" / "lhr_events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    started_at = now - 1000.0
    previous_started_at = now - 10_000.0
    event_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "multi_worker_start", "timestamp": previous_started_at}),
                json.dumps({"event": "worker_start", "timestamp": previous_started_at + 2.0}),
                json.dumps({"event": "multi_worker_start", "timestamp": started_at}),
                json.dumps({"event": "worker_start", "timestamp": started_at + 2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _load_live_run_started_at(task) == started_at

    elapsed = _elapsed_from_resource_range(
        {
            "first_event_at": now - 100.0,
            "last_event_at": now - 50.0,
        },
        live=True,
        run_started_at=started_at,
    )

    assert elapsed is not None
    assert 990.0 <= elapsed <= 1010.0


def test_finished_monitor_state_uses_final_elapsed_and_not_stale(tmp_path: Path) -> None:
    task = tmp_path / "finished-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.80",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    (task / "resolved_config.yaml").write_text("lnr:\n  wall_clock_budget_sec: 43200\n", encoding="utf-8")
    state_dir = task / "task_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "lhr_state.json").write_text(json.dumps({"run_status": "finished"}), encoding="utf-8")
    (state_dir / "state.json").write_text(
        json.dumps({"status": "completed", "elapsed_sec": 43250.3, "charged_elapsed_sec": 43250.3}),
        encoding="utf-8",
    )
    old_ts = 1_700_000_000
    for old_path in [state_dir / "lhr_stage_performance.csv", state_dir / "lhr_state.json", state_dir / "state.json"]:
        os.utime(old_path, (old_ts, old_ts))

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["status"] == "finished"
    assert state["elapsed_sec"] == 43250.3
    assert state["remaining_sec"] == 0.0
    assert state["progress_ratio"] == 1.0
    status_cell = MonitorDashboard()._status_cell(state)
    assert "finished" in status_cell
    assert "STALE" not in status_cell


def test_lnr_budget_cell_displays_elapsed_over_total_budget() -> None:
    cell = MonitorDashboard()._lnr_budget_cell(
        {
            "elapsed_sec": 3600.0,
            "remaining_sec": 7200.0,
            "total_sec": 10800.0,
            "progress_ratio": 1.0 / 3.0,
        }
    )

    assert "1h00/3h00" in cell
    assert "33.3%" in cell
    assert "rem 2h00" in cell
    assert "1h00/2h00" not in cell


def test_live_resume_elapsed_includes_prior_charged_time() -> None:
    now = time.time()
    elapsed = _elapsed_from_live_process_summary(
        {"oldest_start_ts": now - 60.0},
        {"resume_prior_charged_elapsed_sec": 120.0},
    )

    assert elapsed is not None
    assert 175.0 <= elapsed <= 185.0


def test_monitor_state_uses_score_summary_direction_votes_despite_bad_stage_flag(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.684373",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
                "brief": "baseline route",
                "why": "validated submission",
            },
            {
                "candidate_id": "W01:L01:S03",
                "worker_id": "W01",
                "metric_value": "0.742407",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
                "brief": "ensemble route",
                "why": "validated submission",
            },
            {
                "candidate_id": "W02:L01:S02",
                "worker_id": "W02",
                "metric_value": "0.701111",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
                "brief": "stale direction flag",
                "why": "single bad row should not invert the task-level vote",
            },
        ],
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["lower_is_better"] is False
    assert state["best_raw_metric"] == 0.742407
    assert state["best_valid_metric"] == 0.742407
    assert state["best_valid_candidate"] == "W01:L01:S03"


def test_monitor_state_uses_worker_lhr_events_for_progress(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    (task / "resolved_config.yaml").write_text("lnr:\n  wall_clock_budget_sec: 3600\n", encoding="utf-8")
    state_path = task / "task_logs" / "lhr_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"run_status": "running"}), encoding="utf-8")
    events = task / "task_logs" / "workers" / "w00" / "lhr_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    event_lines = [
        json.dumps(
            {
                "event": "worker_status",
                "timestamp": 1000.0,
                "worker_id": "W00",
                "payload": {"mode": "single_worker"},
            }
        ),
        json.dumps(
            {
                "event": "progress_heartbeat",
                "timestamp": 1060.0,
                "worker_id": "W00",
                "payload": {
                    "job_id": "W00:bash:00001",
                    "elapsed_sec": 60.0,
                    "metric_history_line_count": 2,
                    "stdout_lines": 10,
                    "signals": {"metrics": {"rmse": 0.12}},
                },
            }
        ),
        json.dumps(
            {
                "event": "text_only_estra_check",
                "timestamp": 1070.0,
                "worker_id": "W00",
                "payload": {"trigger_source": "text_only"},
            }
        ),
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp": 1080.0,
                "worker_id": "W00",
                "payload": {
                    "action": "keep_but_redirect",
                    "startpoint": "current_workspace",
                    "intent": "redirect",
                    "compact": True,
                    "trigger_source": "text_only",
                },
            }
        ),
    ]
    event_text = "\n".join(event_lines) + "\n"
    events.write_text(event_text, encoding="utf-8")
    (task / "task_logs" / "lhr_events.jsonl").write_text(event_text, encoding="utf-8")

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["status"] == "run"
    assert state["progress_ratio"] is not None
    assert state["active_job_count"] == 1
    assert state["resource_counts"]["progress_heartbeat"] == 1
    assert state["estra_decision_count"] == 1
    assert state["estra_redirect_count"] == 1
    assert state["estra_text_only_check_count"] == 1
    assert state["estra_context_check_count"] == 0
    assert state["estra_text_only_trigger_count"] == 1
    assert state["estra_context_trigger_count"] == 0
    assert state["active_jobs"][0]["command_id"] == "W00:bash:00001"



def test_monitor_active_job_cell_shows_structured_heartbeat_progress(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    now = time.time()
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "resource_source_hint_detected",
                        "timestamp": now - 2,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "command_excerpt": "python eval.py --fold 0",
                            "entrypoint": "eval.py",
                            "resource_class": "heavy_cpu_eval",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "progress_heartbeat",
                        "timestamp": now - 1,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "job_id": "W00:bash:00001",
                            "process_alive": True,
                            "elapsed_sec": 40.0,
                            "stdout_lines": 12,
                            "signals": {
                                "heartbeat": {
                                    "progress": {"current": 80, "total": 100},
                                    "phase": "validation",
                                },
                                "metrics": {"kendall_tau": 0.774},
                            },
                            "structured_progress": {
                                "current": 80,
                                "total": 100,
                                "unit": "items",
                                "advanced": True,
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_job_count"] == 1
    job = state["active_jobs"][0]
    assert job["progress_current"] == 80.0
    assert job["progress_total"] == 100.0
    assert job["metric_name"] == "kendall_tau"
    assert job["metric_value"] == 0.774

    cell = MonitorDashboard()._lnr_active_jobs_cell(
        {"running_process_count": 2, "active_jobs": state["active_jobs"]}
    )
    assert "proc=2 job=1" in cell
    assert "W00" in cell
    assert "VAL" in cell
    assert "80%" in cell
    assert "kt=.774" in cell
    assert "hb=" in cell


def test_monitor_active_jobs_drop_finished_commands(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    events = task / "task_logs" / "workers" / "w00" / "lhr_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "resource_monitor_heartbeat",
                        "timestamp": 1000.0,
                        "worker_id": "W00",
                        "payload": {
                            "job_id": "W00:bash:00001",
                            "process_alive": True,
                            "elapsed_sec": 30.0,
                            "stdout_lines": 4,
                            "cpu_bucket": "active",
                            "gpu_bucket": "idle_or_none",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "resource_job_finished",
                        "timestamp": 1010.0,
                        "worker_id": "W00",
                        "payload": {
                            "job_id": "W00:bash:00001",
                            "elapsed_sec": 40.0,
                            "returncode": 0,
                            "filtered_signal": {"stdout_lines": 5},
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "resource_monitor_heartbeat",
                        "timestamp": 1020.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00002",
                        "payload": {
                            "process_alive": True,
                            "elapsed_sec": 15.0,
                            "stdout_lines": 0,
                            "cpu_bucket": "active",
                            "gpu_bucket": "idle_or_none",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_job_count"] == 1
    assert state["active_jobs"][0]["command_id"] == "W00:bash:00002"
    assert state["active_jobs"][0]["alive"] is True
    assert state["active_jobs"][0]["runtime_sec"] == 15.0
    assert state["active_jobs"][0]["cpu_bucket"] == "active"


def test_monitor_active_jobs_drop_killed_commands(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "resource_monitor_heartbeat",
                        "timestamp": 1000.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "process_alive": True,
                            "elapsed_sec": 160.0,
                            "stdout_lines": 0,
                            "cpu_bucket": "active",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "resource_kill_executed",
                        "timestamp": 1010.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "job_id": "W00:bash:00001",
                            "execution_outcome": "KILL",
                            "elapsed_sec": 170.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_job_count"] == 0
    assert state["active_jobs"] == []
    assert state["worker_states"] == []


def test_monitor_active_jobs_keep_newest_display_job_but_status_keeps_long_worker_state(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    now = time.time()
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(6):
        rows.append(
            json.dumps(
                {
                    "event_type": "resource_monitor_heartbeat",
                    "timestamp": now - 300.0 - idx,
                    "worker_id": "W00",
                    "command_id": f"W00:bash:{idx:05d}",
                    "payload": {
                        "process_alive": True,
                        "elapsed_sec": 3600.0 + idx,
                        "stdout_lines": 1,
                        "cpu_bucket": "active",
                    },
                }
            )
        )
    rows.append(
        json.dumps(
            {
                "event_type": "resource_monitor_heartbeat",
                "timestamp": now - 1.0,
                "worker_id": "W00",
                "command_id": "W00:bash:99999",
                "payload": {
                    "process_alive": True,
                    "elapsed_sec": 20.0,
                    "stdout_lines": 2,
                    "cpu_bucket": "active",
                },
            }
        )
    )
    events.write_text("\n".join(rows) + "\n", encoding="utf-8")

    state = _load_state(task / "logs" / "monitor_state.json")

    assert any(job["command_id"] == "W00:bash:99999" for job in state["active_jobs"])
    assert state["worker_states"][0]["command_id"].startswith("W00:bash:000")
    assert state["worker_states"][0]["label"] == "LRUN"
    assert state["worker_states"][0]["runtime_sec"] >= 3600.0


def test_monitor_drops_stale_real_epoch_active_job(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    now = time.time()
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps(
            {
                "event_type": "resource_monitor_heartbeat",
                "timestamp": now - 7200.0,
                "worker_id": "W00",
                "command_id": "W00:bash:00013",
                "payload": {
                    "process_alive": True,
                    "elapsed_sec": 135.0,
                    "stdout_lines": 17,
                    "cpu_bucket": "idle",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_jobs"] == []
    assert state["worker_states"] == []


def test_monitor_does_not_inherit_stale_heartbeat_for_reused_command_id(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    now = time.time()
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "progress_heartbeat",
                        "timestamp": now - 4000.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "job_id": "W00:bash:00001",
                            "elapsed_sec": 120.0,
                            "signals": {
                                "heartbeat": {
                                    "progress": {"current": 8, "total": 10},
                                    "phase": "train",
                                }
                            },
                            "structured_progress": {
                                "current": 8,
                                "total": 10,
                                "unit": "items",
                                "advanced": True,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "resource_monitor_heartbeat",
                        "timestamp": now - 1.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "process_alive": True,
                            "elapsed_sec": 10.0,
                            "stdout_lines": 0,
                            "cpu_bucket": "active",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_job_count"] == 1
    job = state["active_jobs"][0]
    assert job["command_id"] == "W00:bash:00001"
    assert "heartbeat_observed_at" not in job
    assert "progress_current" not in job
    cell = MonitorDashboard()._lnr_active_jobs_cell(
        {"running_process_count": 1, "active_jobs": state["active_jobs"]}
    )
    assert "stale_hb=" not in cell
    assert "obs=" in cell


def test_monitor_source_hint_does_not_refresh_stale_reused_command_heartbeat(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    now = time.time()
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "progress_heartbeat",
                        "timestamp": now - 4000.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00002",
                        "payload": {
                            "job_id": "W00:bash:00002",
                            "elapsed_sec": 120.0,
                            "signals": {
                                "heartbeat": {
                                    "progress": {"current": 47},
                                    "phase": "training",
                                }
                            },
                            "structured_progress": {
                                "current": 47,
                                "unit": "step",
                                "advanced": True,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "resource_source_hint_detected",
                        "timestamp": now - 2.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00002",
                        "payload": {
                            "command_excerpt": "python new_probe.py",
                            "entrypoint": "new_probe.py",
                            "resource_class": "light_cpu",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "resource_monitor_heartbeat",
                        "timestamp": now - 1.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00002",
                        "payload": {
                            "process_alive": True,
                            "elapsed_sec": 30.0,
                            "stdout_lines": 0,
                            "cpu_bucket": "active",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_job_count"] == 1
    job = state["active_jobs"][0]
    assert job["command_id"] == "W00:bash:00002"
    assert job.get("entrypoint") == "new_probe.py"
    assert "heartbeat_observed_at" not in job
    assert "progress_current" not in job
    cell = MonitorDashboard()._lnr_active_jobs_cell(
        {"running_process_count": 1, "active_jobs": state["active_jobs"]}
    )
    assert "stale_hb=" not in cell
    assert "step=47" not in cell
    assert "obs=" in cell


def test_monitor_active_job_cell_marks_stale_heartbeat() -> None:
    cell = MonitorDashboard()._lnr_active_jobs_cell(
        {
            "running_process_count": 1,
            "active_jobs": [
                {
                    "worker_id": "W00",
                    "command_id": "W00:bash:00001",
                    "heartbeat_observed_at": time.time() - 400.0,
                    "observed_at": time.time() - 400.0,
                    "runtime_sec": 1200.0,
                    "heartbeat_phase": "train",
                    "progress_current": 3,
                    "progress_total": 10,
                }
            ],
        }
    )

    assert "STALL" in cell
    assert "stale_hb=" in cell


def test_monitor_status_exposes_worker_current_states(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    events = task / "task_logs" / "resource" / "resource_events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "resource_source_hint_detected",
                        "timestamp": 1000.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "command_excerpt": "python train.py --epochs 10",
                            "entrypoint": "train.py",
                            "resource_class": "heavy_gpu_train",
                            "source_hint": {"command_train_evidence": True},
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "managed_resource_wait_offered",
                        "timestamp": 1010.0,
                        "worker_id": "W01",
                        "command_id": "W01:bash:00002",
                        "payload": {
                            "job_id": "W01:bash:00002",
                            "worker_id": "W01",
                            "reason": "gpu_slot_unavailable",
                            "resource_class": "heavy_gpu_train",
                            "queue_position": 1,
                            "queue_len": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "resource_monitor_heartbeat",
                        "timestamp": 1020.0,
                        "worker_id": "W00",
                        "command_id": "W00:bash:00001",
                        "payload": {
                            "process_alive": True,
                            "elapsed_sec": 1200.0,
                            "stdout_lines": 20,
                            "cpu_bucket": "active",
                            "gpu_bucket": "active",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["active_job_count"] == 1
    assert state["worker_states"][0]["worker_id"] == "W00"
    assert state["worker_states"][0]["label"] == "LTRN"
    assert state["worker_states"][0]["runtime_sec"] == 1200.0
    assert state["worker_states"][1]["worker_id"] == "W01"
    assert state["worker_states"][1]["label"] == "WAIT"

    status_cell = MonitorDashboard()._status_cell(
        {"status": "run", "worker_states": state["worker_states"]}
    )
    assert "W00:LTRN/20m" in status_cell
    assert "W01:WAIT" in status_cell


def test_monitor_status_cell_prioritizes_long_running_workers() -> None:
    cell = MonitorDashboard()._status_cell(
        {
            "status": "run",
            "worker_states": [
                {"worker_id": "W00", "label": "RUN", "runtime_sec": 60.0},
                {"worker_id": "W01", "label": "RUN", "runtime_sec": 120.0},
                {"worker_id": "W02", "label": "RUN", "runtime_sec": 180.0},
                {"worker_id": "W03", "label": "LRUN", "runtime_sec": 4200.0},
            ],
        }
    )

    assert cell.split("\n", 1)[1].startswith("W03:LRUN/1h10")


def test_monitor_resource_cell_separates_current_waits_from_event_totals(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    resource_dir = task / "task_logs" / "resource"
    resource_dir.mkdir(parents=True, exist_ok=True)
    (resource_dir / "resource_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "managed_resource_wait_offered",
                        "timestamp": 1000.0,
                        "worker_id": "W01",
                        "command_id": "W01:bash:00002",
                        "payload": {
                            "job_id": "W01:bash:00002",
                            "worker_id": "W01",
                            "reason": "gpu_slot_unavailable",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "admission_pending",
                        "timestamp": 1010.0,
                        "worker_id": "W01",
                        "command_id": "W01:bash:00002",
                        "payload": {"job_id": "W01:bash:00002", "worker_id": "W01"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (resource_dir / "gpu_leases.json").write_text(
        json.dumps({"leases": {}, "waiters": {}}), encoding="utf-8"
    )
    (resource_dir / "resource_state.json").write_text(
        json.dumps({"leases": {}, "waiters": {}}), encoding="utf-8"
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    assert state["resource_current"]["active_waiter_count"] == 0
    assert all(worker.get("kind") != "wait" for worker in state["worker_states"])
    cell = MonitorDashboard()._lnr_resource_cell(state)
    assert "wait_total=1" in cell
    assert "pend_total=1" in cell
    assert "wait_now=" not in cell


def test_resource_cell_treats_no_action_as_noop_not_ok() -> None:
    dashboard = MonitorDashboard()

    no_action_cell = dashboard._lnr_resource_cell(
        {
            "resource_outcomes": {"NO_ACTION": 7},
            "resource_counts": {"resource_monitor_heartbeat": 4},
        }
    )
    assert no_action_cell == "hb=4"
    assert "ok=" not in no_action_cell

    action_cell = dashboard._lnr_resource_cell(
        {
            "resource_actionable_outcomes": {"KILL": 2, "TIMEBOX": 1},
            "resource_counts": {},
        }
    )
    assert action_cell == "kill_req=2 tb=1"

    exec_cell = dashboard._lnr_resource_cell(
        {
            "resource_actionable_outcomes": {"KILL": 5},
            "resource_counts": {"resource_kill_executed": 1},
        }
    )
    assert "kill_req=5" in exec_cell
    assert "kill_exec=1" in exec_cell
    assert "kill=5" not in exec_cell


def test_monitor_recent_events_flatten_advisory_and_gate_fields(tmp_path: Path) -> None:
    task = tmp_path / "demo-task"
    _write_csv(
        task / "task_logs" / "lhr_stage_performance.csv",
        [
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "metric_value": "0.12",
                "lower_is_better": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        ],
    )
    resource_path = task / "task_logs" / "resource" / "resource_events.jsonl"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_text(
        json.dumps(
            {
                "event_type": "execution",
                "worker_id": "W00",
                "command_id": "W00:bash:00050",
                "payload": {
                    "execution_outcome": "KILL",
                    "raw_action": "KILL_AND_REPLAN",
                    "status": "pending_terminate",
                    "reason": "main-agent advisory and high-confidence resource facts agree",
                    "strict_gate_result": "allow",
                    "strict_gate_reason": "",
                    "advisory_status": "captured",
                    "advisory_preference": "replan",
                    "advisory_confidence": "high",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = _load_state(task / "logs" / "monitor_state.json")

    recent = state["resource_recent"][-1]
    assert recent["event_type"] == "execution"
    assert recent["outcome"] == "KILL"
    assert recent["preference"] == "replan"
    assert recent["advisory_status"] == "captured"
    assert recent["advisory_confidence"] == "high"
    assert recent["strict_gate_result"] == "allow"


def test_monitor_live_process_overrides_historical_terminal_status() -> None:
    assert _display_status(
        run_status="finished",
        running_process_count=1,
        active_jobs=[],
    ) == "run_resume"


def test_monitor_status_displays_stopped_by_user() -> None:
    assert _status_from_final_state({"status": "stopped_by_user"}) == "stopped_by_user"
    assert (
        _display_status(
            run_status="stopped_by_user",
            running_process_count=0,
            active_jobs=[],
        )
        == "stopped_by_user"
    )


def test_monitor_status_cell_marks_resume_run() -> None:
    cell = MonitorDashboard()._status_cell({"status": "run_resume"})
    assert "run" in cell
    assert "resume" in cell
    assert "finished" not in cell
