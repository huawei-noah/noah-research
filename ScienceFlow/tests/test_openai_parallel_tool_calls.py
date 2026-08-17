#!/usr/bin/env python3
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

"""OpenAI 兼容 API：多工具调用真网冒烟（可选）。

``OnlineLLM.ask_tool`` 在 ``parallel_tool_calls=None``（默认）时**不**把 ``parallel_tool_calls`` 写入请求体，
以便智谱 GLM 等网关拒绝该 OpenAI 专有字段时仍能调用（见 ``deepcraft_core/llm/online.py``）。

本文件验证行为（不依赖上游是否实现 ``parallel_tool_calls`` 开关）：

1. 用户只要求调用一个 function 时，本轮 **≤1** 个 tool call。
2. 用户明确要求同轮两个 function 时，**是否**出现 **≥2** 个 tool call（取决于模型与网关）。

**运行**（需有效 ``agent.code`` 的 key/url/model，见 ``default.yaml`` 或环境变量）::

    cd scienceflow
    SCIENCEFLOW_PARALLEL_TOOL_SMOKE=1 .venv/bin/python -m pytest tests/test_openai_parallel_tool_calls.py -v -s

未设置 ``SCIENCEFLOW_PARALLEL_TOOL_SMOKE=1`` 时测试会 **skip**，不消耗网络。

与 ``tests/test_llm_tool_call.py`` 一样，默认 ``pyproject.toml`` 会 **ignore** 本文件；显式路径运行即可。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from deepcraft_core import Message

from scienceflow.config.settings import Config, StageConfig, load_cfg
from scienceflow.core.llm_http import aclose_llm_clients
from scienceflow.core.agent_runtime import _build_llm


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")


def _smoke_enabled() -> bool:
    return os.environ.get("SCIENCEFLOW_PARALLEL_TOOL_SMOKE", "").strip() == "1"


def _stage(cfg: Config, name: str) -> StageConfig:
    if name not in ("code", "feedback"):
        raise ValueError(name)
    return getattr(cfg.agent, name)


def _stage_ready(stage: StageConfig) -> tuple[bool, str]:
    if stage.api_keys:
        pass
    elif not (stage.api_key or "").strip():
        return False, "missing API_KEY / API_KEYS or agent.<stage>.api_key(s)"
    if stage.base_urls:
        pass
    elif not (stage.base_url or "").strip():
        return False, "missing BASE_URL / BASE_URLS or agent.<stage>.base_url(s)"
    return True, ""


def _two_probe_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "tool_alpha",
                "description": "First parallel probe.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "v": {"type": "string", "description": "payload"},
                    },
                    "required": ["v"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_beta",
                "description": "Second parallel probe.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "v": {"type": "string", "description": "payload"},
                    },
                    "required": ["v"],
                },
            },
        },
    ]


_SYS = (
    "You are an API tool-call probe. When the user asks for multiple tool calls in one turn, "
    "you must respond with that many tool_calls in a single assistant message (no explanatory text)."
)


def _count_tool_calls(msg: Any) -> int:
    tcs = getattr(msg, "tool_calls", None) or []
    return len(tcs)


def _names(msg: Any) -> list[str]:
    out: list[str] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        n = getattr(fn, "name", None)
        if n:
            out.append(str(n))
    return out


@pytest.mark.asyncio
async def test_parallel_tool_calls_false_at_most_one() -> None:
    """单工具 prompt：期望本轮至多 1 个 tool call（不传 ``parallel_tool_calls``，兼容 GLM 等）。"""
    if not _smoke_enabled():
        pytest.skip("set SCIENCEFLOW_PARALLEL_TOOL_SMOKE=1 to run real API smoke test")
    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)

    llm = _build_llm(stage)
    tools = _two_probe_tools()
    user = (
        "In this single response, call ONLY tool_alpha once with v=\"x\". "
        "Do not call tool_beta."
    )
    try:
        msg = await llm.ask_tool(
            messages=[Message.user_message(user)],
            system_msgs=[Message.system_message(_SYS)],
            tools=tools,
            tool_choice="auto",
            timeout=120,
        )
    finally:
        await aclose_llm_clients(llm)

    n = _count_tool_calls(msg)
    print(f"\n[single-tool prompt] tool_calls count={n} names={_names(msg)!r}", file=sys.stderr)
    assert n <= 1, (
        f"expected at most 1 tool call for single-tool prompt, got {n}: {_names(msg)}"
    )


@pytest.mark.asyncio
async def test_parallel_tool_calls_true_may_emit_two() -> None:
    """双工具 prompt：要求同轮两个 function；若只得 1 个则 skip（不传 ``parallel_tool_calls``，兼容 GLM 等）。"""
    if not _smoke_enabled():
        pytest.skip("set SCIENCEFLOW_PARALLEL_TOOL_SMOKE=1 to run real API smoke test")
    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)

    llm = _build_llm(stage)
    tools = _two_probe_tools()
    user = (
        "In this single response, emit exactly two tool calls: "
        "first tool_alpha with v=\"a\", then tool_beta with v=\"b\". "
        "No prose before or after."
    )
    try:
        msg = await llm.ask_tool(
            messages=[Message.user_message(user)],
            system_msgs=[Message.system_message(_SYS)],
            tools=tools,
            tool_choice="auto",
            timeout=120,
        )
    finally:
        await aclose_llm_clients(llm)

    n = _count_tool_calls(msg)
    names = _names(msg)
    print(f"\n[dual-tool prompt] tool_calls count={n} names={names!r}", file=sys.stderr)

    if n < 2:
        pytest.skip(
            f"only {n} tool call(s) returned; API/model may still support multi-tool but "
            f"this run did not observe two (names={names})",
        )

    assert set(names) >= {"tool_alpha", "tool_beta"}, f"unexpected tool names: {names}"


@pytest.mark.asyncio
async def test_parallel_tool_calls_true_print_only_for_manual_inspection() -> None:
    """与上一测试相同，但 **不** skip：用于人工看 stderr 行数（可能失败）。"""
    if not _smoke_enabled():
        pytest.skip("set SCIENCEFLOW_PARALLEL_TOOL_SMOKE=1 to run real API smoke test")
    if os.environ.get("SCIENCEFLOW_PARALLEL_TOOL_STRICT", "").strip() != "1":
        pytest.skip("set SCIENCEFLOW_PARALLEL_TOOL_STRICT=1 to assert len>=2 (may fail on weak models)")

    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)

    llm = _build_llm(stage)
    tools = _two_probe_tools()
    user = (
        "In this single response, emit exactly two tool calls: "
        "first tool_alpha with v=\"a\", then tool_beta with v=\"b\"."
    )
    try:
        msg = await llm.ask_tool(
            messages=[Message.user_message(user)],
            system_msgs=[Message.system_message(_SYS)],
            tools=tools,
            tool_choice="auto",
            timeout=120,
        )
    finally:
        await aclose_llm_clients(llm)

    n = _count_tool_calls(msg)
    names = _names(msg)
    print(f"\n[STRICT dual-tool prompt] count={n} names={names!r}", file=sys.stderr)
    assert n >= 2, f"expected >=2 tool calls, got {n}: {names}"
    assert set(names) >= {"tool_alpha", "tool_beta"}
