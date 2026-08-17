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

"""LLM tool-call integration checks (same stack as ScienceAgent / REPL).

Validates that the configured model returns **non-empty** JSON arguments for ``bash``
(and optionally ``read``), matching what :class:`~scienceflow.core.agent.ScienceAgent`
expects — catches regressions like ``bash {}`` / missing ``command``.

**Manual run** (requires API keys, same as the ``scienceflow`` CLI)::

    cd /path/to/scienceflow
    python tests/test_llm_tool_call.py
    python tests/test_llm_tool_call.py -c scienceflow/config/default.yaml --stage code

**Pytest** (opt-in; file is ignored in default ``addopts``)::

    SCIENCEFLOW_LLM_TOOL_CALL_TEST=1 pytest tests/test_llm_tool_call.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from deepcraft_core import Message
from deepcraft_core.llm.base import StreamHandle

from scienceflow.config.settings import Config, StageConfig, load_cfg
from scienceflow.core.llm_http import aclose_llm_clients
from scienceflow.core.agent_runtime import _build_llm
from scienceflow.core.tools import BashTool, create_tool_collection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _stage(cfg: Config, name: str) -> StageConfig:
    if name not in ("code", "feedback"):
        raise SystemExit(f"unknown stage: {name!r}, expected code|feedback")
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


def _summarize_stage(stage: StageConfig) -> str:
    keys_n = len(stage.api_keys) if stage.api_keys else (1 if stage.api_key else 0)
    urls_n = len(stage.base_urls) if stage.base_urls else (1 if stage.base_url else 0)
    url_preview = (stage.base_urls[0] if stage.base_urls else stage.base_url or "")[:48]
    return (
        f"model={stage.model!r}, endpoints≈{urls_n} url(s) ({url_preview!r}…), "
        f"keys={keys_n}, max_tokens={stage.max_tokens}"
    )


def _first_tool_call(msg: Any) -> tuple[str, dict[str, Any], str] | None:
    """Parse first tool call from ``ask_tool`` or ``ask_tool_stream`` return value."""
    tcs = getattr(msg, "tool_calls", None) or []
    if not tcs:
        return None
    tc = tcs[0]
    fn = tc.function
    name = fn.name
    raw = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        args = {}
    tid = getattr(tc, "id", None) or "call_0"
    return name, args, tid


def _assert_bash_nonempty(label: str, msg: Any) -> dict[str, Any]:
    parsed = _first_tool_call(msg)
    assert parsed is not None, f"{label}: expected at least one tool call, got none"
    name, args, _ = parsed
    assert name == "bash", f"{label}: expected tool name 'bash', got {name!r}"
    cmd = args.get("command")
    assert cmd is not None and str(cmd).strip(), f"{label}: bash args missing non-empty command: {args!r}"
    return args


def _integration_enabled() -> bool:
    return os.environ.get("SCIENCEFLOW_LLM_TOOL_CALL_TEST", "").lower() in ("1", "true", "yes")


def _sys_prompt_tools() -> Message:
    return Message.system_message(
        "You are a helpful coding assistant. When listing files or running shell commands, "
        "you MUST call the provided tools with valid JSON arguments (e.g. bash with a non-empty command string).",
    )


async def _drain_stream_to_string(handle: StreamHandle) -> str:
    parts: list[str] = []
    while True:
        chunk = await handle.queue.get()
        if chunk is None:
            break
        if chunk:
            parts.append(chunk)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Test implementations (async; used by CLI and pytest)
# ---------------------------------------------------------------------------


async def run_test_a_ask_tool(llm: Any, bash_param: dict[str, Any]) -> None:
    msg = await llm.ask_tool(
        messages=[
            Message.user_message(
                "Use the bash tool exactly once to list files in the current directory. "
                "The command must be: ls -la",
            ),
        ],
        system_msgs=[_sys_prompt_tools()],
        tools=[bash_param],
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout=300,
    )
    _assert_bash_nonempty("Test A (ask_tool)", msg)


async def run_test_b_ask_tool_stream(llm: Any, bash_param: dict[str, Any]) -> tuple[str, Any]:
    handle = StreamHandle()

    async def drain() -> str:
        return await _drain_stream_to_string(handle)

    consumer = asyncio.create_task(drain())
    try:
        msg = await llm.ask_tool_stream(
            messages=[
                Message.user_message(
                    "Use the bash tool to print the word OK via: echo OK",
                ),
            ],
            system_msgs=[_sys_prompt_tools()],
            tools=[bash_param],
            tool_choice="auto",
            parallel_tool_calls=False,
            timeout=300,
            handle=handle,
        )
    finally:
        streamed = await consumer

    _assert_bash_nonempty("Test B (ask_tool_stream)", msg)
    assert "bash" in streamed.lower() or "→" in streamed, (
        f"Test B: expected stream hint for bash tool, got preview: {streamed[:200]!r}"
    )
    return streamed, msg


async def run_test_c_read_tool(llm: Any, tmp_path: Path, tools_params: list[dict[str, Any]]) -> None:
    (tmp_path / "task_desc.txt").write_text("hello from task_desc\n", encoding="utf-8")
    msg = await llm.ask_tool(
        messages=[
            Message.user_message(
                "Read the file task_desc.txt in the workspace using the read tool. "
                "Do not use bash cat.",
            ),
        ],
        system_msgs=[_sys_prompt_tools()],
        tools=tools_params,
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout=300,
    )
    parsed = _first_tool_call(msg)
    assert parsed is not None, "Test C: expected a tool call"
    name, args, _ = parsed
    assert name == "read", f"Test C: expected read tool, got {name!r}"
    path = args.get("path")
    assert path is not None and str(path).strip(), f"Test C: read args missing path: {args!r}"


async def run_test_d_multi_turn(llm: Any, bash_param: dict[str, Any], tmp_path: Path) -> None:
    """Two-turn flow: bash echo hello → tool result → second request (bash pwd or text)."""
    handle = StreamHandle()

    async def drain() -> str:
        return await _drain_stream_to_string(handle)

    consumer = asyncio.create_task(drain())
    try:
        msg1 = await llm.ask_tool_stream(
            messages=[Message.user_message("Call bash once with command: echo hello")],
            system_msgs=[_sys_prompt_tools()],
            tools=[bash_param],
            tool_choice="auto",
            parallel_tool_calls=False,
            timeout=300,
            handle=handle,
        )
    finally:
        await consumer

    _assert_bash_nonempty("Test D round 1", msg1)
    coll = create_tool_collection(tmp_path)
    tc = (msg1.tool_calls or [])[0]
    fn = tc.function
    raw = fn.arguments if isinstance(fn.arguments, str) else str(fn.arguments)
    tool_input = json.loads(raw or "{}")
    tr = await coll.execute(name="bash", tool_input=tool_input)
    feedback = (tr.output or "") if not tr.error else f"Error: {tr.error}"

    tc = (msg1.tool_calls or [])[0]
    assistant_dict: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(msg1, "content", "") or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": (
                        tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else json.dumps(tc.function.arguments)
                    ),
                },
            },
        ],
    }
    tool_dict = {
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.function.name,
        "content": feedback[:8000],
    }
    msg2 = await llm.ask_tool(
        messages=[
            assistant_dict,
            tool_dict,
            Message.user_message(
                "Good. Now call bash once with command: pwd",
            ).to_dict(),
        ],
        system_msgs=[_sys_prompt_tools()],
        tools=[bash_param],
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout=300,
    )

    tcs = getattr(msg2, "tool_calls", None) or []
    if tcs:
        _assert_bash_nonempty("Test D round 2", msg2)
    else:
        text = (getattr(msg2, "content", None) or "").strip()
        assert text, "Test D round 2: expected tool call or non-empty assistant text"


async def _run_all_tests(cfg: Config, stage_name: str, tmp_path: Path) -> list[tuple[str, bool, str]]:
    stage = _stage(cfg, stage_name)
    ok, err = _stage_ready(stage)
    if not ok:
        raise SystemExit(err)

    print(f"[test_llm_tool_call] stage={stage_name} {_summarize_stage(stage)}")
    bash_tool = BashTool(workspace_dir=tmp_path, max_output_chars=4000)
    bash_param = bash_tool.to_param()
    tools_all = create_tool_collection(tmp_path).to_params()

    llm = _build_llm(stage)
    results: list[tuple[str, bool, str]] = []

    try:
        try:
            await run_test_a_ask_tool(llm, bash_param)
            results.append(("A ask_tool", True, ""))
        except Exception as e:
            results.append(("A ask_tool", False, f"{type(e).__name__}: {e}"))

        try:
            await run_test_b_ask_tool_stream(llm, bash_param)
            results.append(("B ask_tool_stream", True, ""))
        except Exception as e:
            results.append(("B ask_tool_stream", False, f"{type(e).__name__}: {e}"))

        try:
            await run_test_c_read_tool(llm, tmp_path, tools_all)
            results.append(("C read (full tools)", True, ""))
        except Exception as e:
            results.append(("C read (full tools)", False, f"{type(e).__name__}: {e}"))

        try:
            await run_test_d_multi_turn(llm, bash_param, tmp_path)
            results.append(("D multi-turn", True, ""))
        except Exception as e:
            results.append(("D multi-turn", False, f"{type(e).__name__}: {e}"))

    finally:
        await aclose_llm_clients(llm)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM tool-call integration test (bash/read).")
    parser.add_argument("--config", "-c", default=None, help="YAML path; default scienceflow/config/default.yaml")
    parser.add_argument("--stage", "-s", default="code", choices=("code", "feedback"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _bootstrap_env()
    cfg = load_cfg(Path(args.config) if args.config else None)

    import tempfile

    from tests.fs_root import TEST_WORKSPACE_ROOT

    TEST_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(TEST_WORKSPACE_ROOT)) as td:
        tmp_path = Path(td)

        async def _go() -> int:
            rows = await _run_all_tests(cfg, args.stage, tmp_path)
            print("\n--- results ---")
            failed = 0
            for name, ok, err in rows:
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {name}")
                if not ok:
                    print(f"         {err}")
                    failed += 1
            print("---------------")
            if failed:
                print(f"FAILED: {failed}/{len(rows)}")
                return 1
            print(f"OK: {len(rows)}/{len(rows)}")
            return 0

        try:
            return asyncio.run(_go())
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 1
        except Exception as e:
            print(f"[test_llm_tool_call] fatal: {e}", file=sys.stderr)
            if args.verbose:
                logging.getLogger("test_llm_tool_call").exception("detail")
            return 1


# ---------------------------------------------------------------------------
# Pytest (opt-in)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ws(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.skipif(not _integration_enabled(), reason="set SCIENCEFLOW_LLM_TOOL_CALL_TEST=1 to run")
async def test_llm_tool_call_integration_a(tmp_ws: Path) -> None:
    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)
    llm = _build_llm(stage)
    bash_param = BashTool(workspace_dir=tmp_ws, max_output_chars=4000).to_param()
    try:
        await run_test_a_ask_tool(llm, bash_param)
    finally:
        await aclose_llm_clients(llm)


@pytest.mark.asyncio
@pytest.mark.skipif(not _integration_enabled(), reason="set SCIENCEFLOW_LLM_TOOL_CALL_TEST=1 to run")
async def test_llm_tool_call_integration_b(tmp_ws: Path) -> None:
    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)
    llm = _build_llm(stage)
    bash_param = BashTool(workspace_dir=tmp_ws, max_output_chars=4000).to_param()
    try:
        await run_test_b_ask_tool_stream(llm, bash_param)
    finally:
        await aclose_llm_clients(llm)


@pytest.mark.asyncio
@pytest.mark.skipif(not _integration_enabled(), reason="set SCIENCEFLOW_LLM_TOOL_CALL_TEST=1 to run")
async def test_llm_tool_call_integration_c(tmp_ws: Path) -> None:
    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)
    llm = _build_llm(stage)
    tools_all = create_tool_collection(tmp_ws).to_params()
    try:
        await run_test_c_read_tool(llm, tmp_ws, tools_all)
    finally:
        await aclose_llm_clients(llm)


@pytest.mark.asyncio
@pytest.mark.skipif(not _integration_enabled(), reason="set SCIENCEFLOW_LLM_TOOL_CALL_TEST=1 to run")
async def test_llm_tool_call_integration_d(tmp_ws: Path) -> None:
    _bootstrap_env()
    cfg = load_cfg(None)
    stage = _stage(cfg, "code")
    ok, err = _stage_ready(stage)
    if not ok:
        pytest.skip(err)
    llm = _build_llm(stage)
    bash_param = BashTool(workspace_dir=tmp_ws, max_output_chars=4000).to_param()
    try:
        await run_test_d_multi_turn(llm, bash_param, tmp_ws)
    finally:
        await aclose_llm_clients(llm)


if __name__ == "__main__":
    raise SystemExit(main())
