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

import httpx
from openai import BadRequestError

from scienceflow.core import key_pool
from scienceflow.core.key_pool import PooledLLM
from scienceflow.config.settings import Config, _apply_env


class DummyLLM:
    def __init__(self, name: str, *, fail_first_calls: int = 0) -> None:
        self.name = name
        self.base_url = "https://llm.example/v1"
        self.api_key = name
        self.model = "dummy"
        self.fail_first_calls = fail_first_calls
        self.calls = 0
        self._last_call_input_tokens = 1
        self._last_call_output_tokens = 1
        self._last_call_input_cached_tokens = 0
        self._last_call_ttft = 0.1
        self._last_call_tpot = 1.0
        self._last_finish_reason = "stop"

    async def ask(self, *_args, **_kwargs) -> str:
        self.calls += 1
        if self.calls <= self.fail_first_calls:
            raise httpx.ReadTimeout(f"{self.name} timeout")
        return self.name


def test_sticky_failover_returns_to_primary_after_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(key_pool, "_SOFT_RETRY_DELAY_SEC", 0.0)
    primary = DummyLLM("primary", fail_first_calls=2)
    secondary = DummyLLM("secondary")
    pool = PooledLLM(
        [primary, secondary],
        routing_mode="sticky_failover",
        sticky_primary_index=0,
        connection_cooldown_sec=0.01,
        rate_limit_cooldown_sec=0.01,
    )

    assert asyncio.run(pool.ask([])) == "secondary"
    assert primary.calls == 2
    assert secondary.calls == 1
    assert pool._last_call_pool_index == 1
    assert pool._last_call_failover_count == 1

    import time

    time.sleep(0.02)

    assert asyncio.run(pool.ask([])) == "primary"
    assert primary.calls == 3
    assert secondary.calls == 1
    assert pool._last_call_pool_index == 0


def test_round_robin_mode_still_rotates(monkeypatch) -> None:
    monkeypatch.setattr(key_pool, "_SOFT_RETRY_DELAY_SEC", 0.0)
    first = DummyLLM("first")
    second = DummyLLM("second")
    pool = PooledLLM([first, second])

    assert asyncio.run(pool.ask([])) == "first"
    assert asyncio.run(pool.ask([])) == "second"
    assert first.calls == 1
    assert second.calls == 1




def test_from_endpoints_keeps_per_endpoint_models(monkeypatch) -> None:
    class ConstructedLLM:
        def __init__(self, *, base_url: str, api_key: str, model: str, **_kwargs) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self._last_call_input_tokens = 1
            self._last_call_output_tokens = 1
            self._last_call_input_cached_tokens = 0
            self._last_call_ttft = 0.1
            self._last_call_tpot = 1.0
            self._last_finish_reason = "stop"

    monkeypatch.setattr(key_pool, "OnlineLLM", ConstructedLLM)

    pool = PooledLLM.from_endpoints(
        [("https://a.example/v1", "key-a"), ("https://b.example/v1", "key-b")],
        endpoint_models=["model-a", "model-b"],
        model="fallback",
        max_tokens=128,
    )

    assert [inst.model for inst in pool._instances] == ["model-a", "model-b"]
    assert [inst.api_key for inst in pool._instances] == ["key-a", "key-b"]

def test_apply_env_loads_code_models_pool(monkeypatch) -> None:
    monkeypatch.setenv("CODE_MODELS", "model-a,model-b model-c")
    monkeypatch.setenv("FEEDBACK_MODELS", "judge-a,judge-b")
    cfg = Config()

    _apply_env(cfg)

    assert cfg.agent.code.models == ["model-a", "model-b", "model-c"]
    assert cfg.agent.feedback.models == ["judge-a", "judge-b"]


def test_apply_env_loads_generic_sticky_routing(monkeypatch) -> None:
    monkeypatch.setenv("SCIENCEFLOW_LLM_ROUTING_MODE", "sticky_failover")
    monkeypatch.setenv("SCIENCEFLOW_LLM_STICKY_ID", "run-a")
    monkeypatch.setenv("SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX", "0")
    cfg = Config()

    _apply_env(cfg)

    assert cfg.agent.code.api_routing_mode == "sticky_failover"
    assert cfg.agent.code.api_sticky_id == "run-a"
    assert cfg.agent.code.api_sticky_primary_index == 0
    assert cfg.agent.feedback.api_routing_mode == "sticky_failover"
    assert cfg.agent.feedback.api_sticky_id == "run-a"
    assert cfg.agent.feedback.api_sticky_primary_index == 0


def _reasoning_replay_bad_request() -> BadRequestError:
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        "Error code: 400 - {'error': {'message': 'The `reasoning_content` "
        "in the thinking mode must be passed back to the API.'}}",
        response=response,
        body={
            "error": {
                "message": (
                    "The `reasoning_content` in the thinking mode must be "
                    "passed back to the API."
                ),
            },
        },
    )


def test_reasoning_replay_error_is_not_failed_over(monkeypatch) -> None:
    monkeypatch.setattr(key_pool, "_SOFT_RETRY_DELAY_SEC", 0.0)

    class ReasoningFailLLM(DummyLLM):
        async def ask(self, *_args, **_kwargs) -> str:
            self.calls += 1
            raise _reasoning_replay_bad_request()

    first = ReasoningFailLLM("first")
    second = DummyLLM("second")
    pool = PooledLLM([first, second])

    try:
        asyncio.run(pool.ask([]))
    except BadRequestError as exc:
        assert "reasoning_content" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected BadRequestError")

    assert first.calls == 1
    assert second.calls == 0

