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

import pytest

from scienceflow.config.settings import load_cfg


def test_config_include_relative_paths_and_local_override(tmp_path: Path):
    base = tmp_path / "base.yaml"
    tool = tmp_path / "tool.yaml"
    lnr = tmp_path / "lnr.yaml"
    tool.write_text("tool:\n  scienceflow_stdout_max_chars: 1234\n", encoding="utf-8")
    lnr.write_text("lnr:\n  num_workers: 3\n  wall_clock_budget_sec: 111\n", encoding="utf-8")
    base.write_text(
        "include:\n"
        "  - tool.yaml\n"
        "  - lnr.yaml\n"
        "lnr:\n"
        "  num_workers: 5\n",
        encoding="utf-8",
    )

    cfg = load_cfg(base, cli_args=False)

    assert cfg.tool.scienceflow_stdout_max_chars == 1234
    assert cfg.lnr.wall_clock_budget_sec == 111
    assert cfg.lnr.num_workers == 5
    assert cfg.config_source_path == str(base.resolve())


def test_config_recursive_include_fails(tmp_path: Path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("include: b.yaml\n", encoding="utf-8")
    b.write_text("include: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Recursive config include"):
        load_cfg(a, cli_args=False)


def test_default_resource_control_mode_is_full_smart_monitoring():
    cfg = load_cfg(None, cli_args=False)

    assert cfg.lnr.stage_commit_output_format == "text"
    assert cfg.lnr.stage_commit_context_mode == "compact"
    assert cfg.lnr.stage_commit_tool_choice == "none"
    assert cfg.lnr.stage_commit_experiment_state_enabled is False
    assert cfg.lnr.protected_eda_mode == "facts"
    assert cfg.lnr.protected_eda_summary_max_chars == 8000
    assert cfg.evaluator.query_budget_scope == "task"
    assert cfg.evaluator.stop_on_query_budget_exhausted is False
    assert cfg.lnr.final_artifact_mode == "workspace"
    assert cfg.lnr.merge_required_finals == 3
    assert cfg.lnr.merge_max_finals == 3
    assert cfg.lnr.resource_control_mode == "resource_smart_llm"
    assert cfg.lnr.resource_monitor_enabled is True
    assert cfg.lnr.resource_runtime_enabled is True
    assert cfg.lnr.resource_gpu_queue_enabled is True
    assert cfg.lnr.resource_gpu_admission_queue_enabled is True
    assert cfg.lnr.resource_admission_llm_enabled is True
    assert cfg.lnr.resource_arbiter_enabled is True
    assert cfg.lnr.resource_arbiter_mode == "llm"
    assert cfg.lnr.resource_monitor_kill_mode == "arbiter"
    assert cfg.lnr.resource_main_agent_advisory_enabled is True
    assert cfg.lnr.resource_gpu_share_enabled is True


def test_resource_control_mode_overlay_overrides_included_defaults(tmp_path: Path):
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    base.write_text("lnr:\n  resource_control_mode: resource_smart_llm\n", encoding="utf-8")
    overlay.write_text(
        "include: base.yaml\n"
        "lnr:\n"
        "  resource_control_mode: off\n",
        encoding="utf-8",
    )

    cfg = load_cfg(overlay, cli_args=False)

    assert cfg.lnr.resource_control_mode == "off"
    assert cfg.lnr.resource_monitor_enabled is False
    assert cfg.lnr.resource_runtime_enabled is False
    assert cfg.lnr.resource_gpu_queue_enabled is False
    assert cfg.lnr.resource_gpu_admission_queue_enabled is False
    assert cfg.lnr.resource_admission_llm_enabled is False
    assert cfg.lnr.resource_arbiter_enabled is False
    assert cfg.lnr.resource_main_agent_advisory_enabled is False
    assert cfg.lnr.resource_gpu_share_enabled is False


def test_resource_control_mode_keeps_same_payload_expert_override(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "lnr:\n"
        "  resource_control_mode: resource_smart_policy\n"
        "  resource_admission_llm_enabled: true\n",
        encoding="utf-8",
    )

    cfg = load_cfg(cfg_path, cli_args=False)

    assert cfg.lnr.resource_control_mode == "resource_smart_policy"
    assert cfg.lnr.resource_arbiter_enabled is True
    assert cfg.lnr.resource_arbiter_mode == "policy"
    assert cfg.lnr.resource_main_agent_advisory_enabled is False
    assert cfg.lnr.resource_gpu_share_enabled is False
    assert cfg.lnr.resource_admission_llm_enabled is True


def test_invalid_resource_control_mode_fails(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("lnr:\n  resource_control_mode: mystery\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resource_control_mode"):
        load_cfg(cfg_path, cli_args=False)
