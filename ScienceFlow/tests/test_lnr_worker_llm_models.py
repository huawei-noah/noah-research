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

import json
from typing import Any

from scienceflow.config.settings import Config, apply_parallel_manifest_agent_overrides
from scienceflow.solver.lnr.solver import LnrSolver


def _solver_for_worker(cfg: Config, *, worker_index: int, worker_count: int = 4):
    solver = LnrSolver.__new__(LnrSolver)
    solver.cfg = cfg
    solver.worker_index = worker_index
    solver.worker_count = worker_count
    solver.worker_id = f"W{worker_index:02d}"
    events: list[dict[str, Any]] = []
    solver._jsonl = lambda _name, payload: events.append(payload)
    solver._worker_uid_prefix = lambda: solver.worker_id
    return solver, events


def test_worker_code_models_pin_primary_and_keep_failover_pool() -> None:
    cfg = Config()
    cfg.exp_id = "exp-a"
    cfg.agent.code.model = "fallback-model"
    cfg.agent.code.models = ["model-a", "model-b", "model-c"]
    cfg.agent.code.api_keys = ["key-a", "key-b", "key-c"]
    cfg.agent.code.base_urls = ["https://a.example/v1", "https://b.example/v1", "https://c.example/v1"]
    cfg.agent.code.api_routing_mode = "sticky_failover"

    solver, events = _solver_for_worker(cfg, worker_index=1)
    stage = solver._worker_llm_stage_override()

    assert stage is not None
    assert stage.model == "model-b"
    assert stage.models == ["model-b", "model-c", "model-a"]
    assert stage.api_key == "key-b"
    assert stage.api_keys == ["key-b", "key-c", "key-a"]
    assert stage.base_url == "https://b.example/v1"
    assert stage.base_urls == ["https://b.example/v1", "https://c.example/v1", "https://a.example/v1"]
    assert stage.api_sticky_id == "exp-a:W01"
    assert stage.api_sticky_primary_index == 0
    assert events[-1]["event"] == "worker_model_endpoint_assigned"
    assert events[-1]["model_index"] == 1
    assert events[-1]["api_key_index"] == 1
    assert events[-1]["base_url_index"] == 1
    assert events[-1]["failover_pool_size"] == 3


def test_worker_code_models_cycle_shorter_lists_and_broadcast_single_url() -> None:
    cfg = Config()
    cfg.agent.code.models = ["model-a", "model-b"]
    cfg.agent.code.api_keys = ["key-a", "key-b", "key-c"]
    cfg.agent.code.base_urls = ["https://shared.example/v1"]

    solver, _events = _solver_for_worker(cfg, worker_index=3)
    stage = solver._worker_llm_stage_override()

    assert stage is not None
    assert stage.model == "model-b"
    assert stage.models == ["model-b", "model-a", "model-b", "model-a", "model-b", "model-a"]
    assert stage.api_key == "key-a"
    assert stage.api_keys == ["key-a", "key-b", "key-c", "key-a", "key-b", "key-c"]
    assert stage.base_url == "https://shared.example/v1"
    assert stage.base_urls == ["https://shared.example/v1"] * 6


def test_worker_without_models_preserves_existing_sticky_key_pool_behavior() -> None:
    cfg = Config()
    cfg.exp_id = "exp-a"
    cfg.agent.code.api_keys = ["key-a", "key-b", "key-c"]
    cfg.agent.code.api_routing_mode = "sticky_failover"

    solver, events = _solver_for_worker(cfg, worker_index=2)
    stage = solver._worker_llm_stage_override()

    assert stage is not None
    assert stage.model == cfg.agent.code.model
    assert stage.api_keys == ["key-a", "key-b", "key-c"]
    assert stage.api_sticky_id == "exp-a:W02"
    assert stage.api_sticky_primary_index == 2
    assert events[-1]["event"] == "worker_sticky_primary_assigned"


def test_worker_feedback_models_can_pin_primary_endpoint() -> None:
    cfg = Config()
    cfg.exp_id = "exp-a"
    cfg.agent.feedback.models = ["judge-a", "judge-b", "judge-c"]
    cfg.agent.feedback.api_keys = ["fb-key-a", "fb-key-b", "fb-key-c"]
    cfg.agent.feedback.base_urls = [
        "https://fb-a.example/v1",
        "https://fb-b.example/v1",
        "https://fb-c.example/v1",
    ]

    solver, events = _solver_for_worker(cfg, worker_index=2)
    stage = solver._worker_llm_stage_override("feedback")

    assert stage is not None
    assert stage.model == "judge-c"
    assert stage.api_key == "fb-key-c"
    assert stage.base_url == "https://fb-c.example/v1"
    assert stage.api_sticky_id == "exp-a:W02"
    assert stage.api_sticky_primary_index == 0
    assert events[-1]["stage_role"] == "feedback"
    assert events[-1]["model_index"] == 2
    assert events[-1]["api_key_index"] == 2
    assert events[-1]["base_url_index"] == 2


def test_parallel_manifest_nested_agent_stage_overrides_apply(monkeypatch) -> None:
    cfg = Config()
    monkeypatch.setenv("NF_TEST_CODE_KEY_A", "key-a")
    monkeypatch.setenv("NF_TEST_CODE_URL_A", "https://a.example/v1")
    monkeypatch.setenv(
        "SCIENCEFLOW_PARALLEL_AGENT_JSON",
        json.dumps(
            {
                "code": {
                    "models": ["code-a", "code-b"],
                    "api_keys": ["${NF_TEST_CODE_KEY_A}", "key-b"],
                    "base_urls": ["${NF_TEST_CODE_URL_A}", "https://b.example/v1"],
                    "api_routing_mode": "sticky_failover",
                },
                "feedback": {
                    "model": "judge-a",
                    "api_key": "judge-key",
                    "base_url": "https://judge.example/v1",
                    "temp": 0.1,
                },
            },
        ),
    )

    apply_parallel_manifest_agent_overrides(cfg)

    assert cfg.agent.code.models == ["code-a", "code-b"]
    assert cfg.agent.code.api_keys == ["key-a", "key-b"]
    assert cfg.agent.code.base_urls == ["https://a.example/v1", "https://b.example/v1"]
    assert cfg.agent.code.api_routing_mode == "sticky_failover"
    assert cfg.agent.feedback.model == "judge-a"
    assert cfg.agent.feedback.api_key == "judge-key"
    assert cfg.agent.feedback.base_url == "https://judge.example/v1"
    assert cfg.agent.feedback.temp == 0.1
