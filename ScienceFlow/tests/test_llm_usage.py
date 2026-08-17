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

"""``read_llm_last_usage_tokens`` prefers OnlineLLM CallStats over legacy names."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scienceflow.core.llm_usage import read_llm_last_usage_tokens


def test_read_prefers_last_call_stats():
    llm = SimpleNamespace(
        _last_call_input_tokens=100,
        _last_call_output_tokens=20,
        last_tokens_input=1,
        last_tokens_output=1,
    )
    assert read_llm_last_usage_tokens(llm) == (100, 20)


def test_read_falls_back_to_last_tokens():
    llm = SimpleNamespace(
        _last_call_input_tokens=0,
        _last_call_output_tokens=0,
        last_tokens_input=5,
        last_tokens_output=2,
    )
    assert read_llm_last_usage_tokens(llm) == (5, 2)


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def _install_fake_completion_client(llm):
    completions = _FakeCompletions()
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return completions


@pytest.mark.asyncio
async def test_online_llm_ask_omits_default_top_p():
    from deepcraft_core.llm.online import OnlineLLM

    llm = OnlineLLM(
        model="unit-test-model",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        stream=False,
    )
    completions = _install_fake_completion_client(llm)

    text = await llm._stream_request(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        temperature=None,
    )

    assert text == "ok"
    assert completions.kwargs is not None
    assert "top_p" not in completions.kwargs


@pytest.mark.asyncio
async def test_online_llm_ask_forwards_explicit_top_p():
    from deepcraft_core.llm.online import OnlineLLM

    llm = OnlineLLM(
        model="unit-test-model",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        stream=False,
        top_p=0.9,
    )
    completions = _install_fake_completion_client(llm)

    await llm._stream_request(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        temperature=None,
    )

    assert completions.kwargs is not None
    assert completions.kwargs["top_p"] == 0.9
