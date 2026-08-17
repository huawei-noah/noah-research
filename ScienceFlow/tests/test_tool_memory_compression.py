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

"""Unit tests for ScienceAgent tool memory compression (args + tool outputs)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

from deepcraft_core import Message
from deepcraft_core.tool import ToolResult

from scienceflow.core.agent import (
    _DEFAULT_SYSTEM,
    _compress_tool_call_for_memory,
    _looks_like_write_placeholder_mimicry,
)
from scienceflow.core.agent.shared.constants import _SCRUBBED_WRITE_PLACEHOLDER_NOTE
from scienceflow.core.agent.memory.memory_utils import (
    _assistant_message_from_api,
    scrub_last_assistant_write_tool_call_content,
)
from scienceflow.core.agent_runtime import create_agent_memory
from scienceflow.core.mem.memory_context import (
    MemoryContextManager,
    _bash_is_pure_read_only,
    bash_command_dumps_python_source,
    _message_char_len,
    _READ_OVERLAP_COACHING,
    _AUTO_SNAPSHOT_PREFIX,
    _build_write_auto_snapshot_block,
    compress_bash_tool_output_for_memory,
    compress_edit_success_output_for_memory,
    compress_read_output_for_memory,
)


def test_compress_write_tool_call_preserves_large_body_for_cache() -> None:
    body = "x" * 500
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": body}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out.function.name == "write"
    args = json.loads(out.function.arguments)
    assert args["path"] == "a.py"
    assert args["content"] == body


def test_compress_write_tool_call_keeps_placeholder_attempt_for_storage() -> None:
    mimic = "[file content omitted - 51 chars written to disk]"
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": mimic}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out.function.name == "write"
    args = json.loads(out.function.arguments)
    assert args["content"] == mimic


def test_compress_write_tool_call_keeps_new_angle_placeholder_attempt_for_storage() -> None:
    mimic = (
        "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>"
    )
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": mimic}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out.function.name == "write"
    args = json.loads(out.function.arguments)
    assert args["content"] == mimic


def test_compress_write_short_body_unchanged() -> None:
    short = "x" * 100
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": short}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    args = json.loads(out.function.arguments)
    assert args["content"] == short


def test_compress_tool_call_disabled_returns_original() -> None:
    tc = SimpleNamespace(
        id="call1",
        type="function",
        function=SimpleNamespace(
            name="write",
            arguments=json.dumps({"path": "a.py", "content": "x" * 100}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=False)
    assert out is tc


def test_compress_edit_tool_call_long_old_and_new_str() -> None:
    long_new = "y" * 300
    long_old = "o" * 240
    tc = SimpleNamespace(
        id="c2",
        type="function",
        function=SimpleNamespace(
            name="edit",
            arguments=json.dumps(
                {"path": "b.py", "old_str": long_old, "new_str": long_new},
            ),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    args = json.loads(out.function.arguments)
    assert args["old_str"].startswith("o" * 60)
    assert args["old_str"].endswith("o" * 60)
    assert (
        "# <<<MEMORY_COMPRESSED: edit.old_str=240bytes — NOT CODE — real content on disk>>>"
        in args["old_str"]
    )
    assert len(args["new_str"]) < len(long_new)
    assert (
        "# <<<MEMORY_COMPRESSED: omitted 180 chars from new_str — NOT CODE>>>" in args["new_str"]
    )
    assert args["new_str"].startswith("y" * 60)
    assert args["new_str"].endswith("y" * 60)


def test_compress_edit_success_output_strips_context() -> None:
    obs = (
        "Edited foo.py: 1 replacement OK.\n"
        "(context around edit near line 5)\n"
        ">>     1|line\n"
        "(42 lines, sha256~abcdef0123456789)"
    )
    c = compress_edit_success_output_for_memory(obs)
    assert "context around edit" not in c
    assert c == (
        "Edited foo.py: 1 replacement OK.\n"
        "(42 lines, sha256~abcdef0123456789)"
    )


def test_compress_edit_success_relaxed_regex_long_hash_and_trailing() -> None:
    """Last line may have longer hex and optional trailing metadata."""
    long_hash = "a" * 64
    obs = (
        "Edited foo.py: 1 replacement OK.\n"
        "(context)\n"
        f"(42 lines, sha256~{long_hash}) [verified]"
    )
    c = compress_edit_success_output_for_memory(obs)
    assert "context" not in c
    assert c == (
        "Edited foo.py: 1 replacement OK.\n"
        f"(42 lines, sha256~{long_hash}) [verified]"
    )


def test_compress_bash_dedup_repeated_lines_before_tail() -> None:
    spam = "[LightGBM] [Warning] noisy"
    body = "\n".join([spam] * 50 + [f"L{i:03d}" for i in range(30)])
    obs = "[exit=0, 1.0s]\n" + body
    c = compress_bash_tool_output_for_memory(obs, tail_lines=40, dedup_enabled=True)
    assert spam in c
    assert c.count(spam) == 1
    assert "log-dedup" in c
    assert "L029" in c
    assert len(c) < len(obs)


def test_compress_bash_success_keeps_tail_only() -> None:
    body = "\n".join(f"L{i:03d}" for i in range(30))
    obs = "[exit=0, 1.0s]\n" + body
    c = compress_bash_tool_output_for_memory(obs, tail_lines=5)
    assert "30 lines total" in c
    assert "L029" in c
    assert "L000" not in c
    # Header must reconstruct outer brackets only (not str.strip("[]") charset stripping).
    assert c.splitlines()[0].startswith("[exit=0, 1.0s]")


def test_compress_bash_short_output_unchanged() -> None:
    obs = "[exit=0, 0.1s]\nhello\nworld"
    assert compress_bash_tool_output_for_memory(obs, tail_lines=20) == obs


def test_record_bare_solution_bash_success_uses_short_tail(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mbare", "Draft", 200)
    ws = tmp_path / "wbare"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, bash_success_tail_lines=20)
    body = "\n".join([f"L{i:02d}" for i in range(20)] + ["Final Validation Score: 0.123"])
    obs = ctx.record_tool_result(
        "bash",
        {"command": "python3 solution.py"},
        ToolResult(output="[exit=0, 4.8s]\n" + body),
    )
    assert "bash_training_signal_v2" in obs
    assert "L00" not in obs
    assert "L19" in obs
    assert "Final Validation Score: 0.123" in obs


def test_record_bare_solution_bash_failure_preserves_traceback(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mtrace", "Draft", 200)
    ws = tmp_path / "wtrace"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, bash_success_tail_lines=20)
    body = "\n".join(
        [f"[DATA] Fold {i % 5} metric={0.08 + i / 10000:.5f}" for i in range(24)]
        + [
            "[DATASET] Generating submission...",
            "Traceback (most recent call last):",
            '  File "solution.py", line 256, in <module>',
            '    sub_df[target] = test_preds[target].values.astype("float64")',
            "AttributeError: 'numpy.ndarray' object has no attribute 'values'",
        ]
    )
    obs = ctx.record_tool_result(
        "bash",
        {"command": "python3 solution.py"},
        ToolResult(output="[exit=1, 3.0s]\n" + body),
    )
    assert "bash_training_signal_v2" in obs
    assert "preserved error/traceback lines" in obs
    assert "AttributeError" in obs
    assert "test_preds[target].values" in obs


def _record_successful_bash_with_tail_config(
    tmp_path,
    *,
    command: str,
    line_count: int = 20,
) -> str:
    mem = create_agent_memory(tmp_path / ("m_" + str(abs(hash(command)))), "Draft", 200)
    ws = tmp_path / ("w_" + str(abs(hash(command))))
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=6,
        bash_success_tail_lines_solution=3,
        bash_success_tail_lines_test=9,
        bash_success_tail_lines_readonly=4,
        bash_success_tail_lines_install=2,
    )
    body = "\n".join(f"L{i:02d}" for i in range(line_count))
    return ctx.record_tool_result(
        "bash",
        {"command": command},
        ToolResult(output="[exit=0, 1.0s]\n" + body),
    )


def test_record_bash_dynamic_tail_solution_positive_and_substring_taskset(tmp_path) -> None:
    obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="python3 solution.py",
        line_count=30,
    )
    assert "bash_training_signal_v2" in obs
    assert "L00" not in obs
    assert "L29" in obs

    taskset_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="taskset -c 0 python3 solution.py",
        line_count=30,
    )
    assert "bash_training_signal_v2" in taskset_obs
    assert "L00" not in taskset_obs
    assert "L29" in taskset_obs


def test_record_bash_dynamic_tail_sampled_solution_uses_fallback(tmp_path) -> None:
    obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="NROWS=100 python3 solution.py",
    )
    assert "showing last 6" in obs
    assert "L13" not in obs
    assert "L14" in obs


def test_record_bash_dynamic_tail_test_readonly_install_and_fallback(tmp_path) -> None:
    pytest_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="CUDA_VISIBLE_DEVICES=0 pytest tests/test_x.py",
        line_count=130,
    )
    assert "bash_pytest_summary_v2" in pytest_obs

    py3_pytest_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="python3 -m pytest tests/test_x.py",
        line_count=130,
    )
    assert "bash_pytest_summary_v2" in py3_pytest_obs

    readonly_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="cat notes.txt",
    )
    assert "showing last 4" in readonly_obs

    install_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="pip install x",
    )
    assert "bash_install_summary_v2" in install_obs

    uv_run_install_obs = _record_successful_bash_with_tail_config(
        tmp_path,
        command="uv run pip install x",
    )
    assert "showing last 6" in uv_run_install_obs


def test_message_char_len_includes_tool_calls_json() -> None:
    m = Message(
        role="assistant",
        content="hi",
        tool_calls=[
            {
                "id": "x",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{"path":"p","content":"[file content omitted - 3 chars written to disk]"}',
                },
            },
        ],
    )
    assert _message_char_len(m) > len("hi")


def test_tool_call_invalid_json_unchanged() -> None:
    tc = SimpleNamespace(
        id="c",
        type="function",
        function=SimpleNamespace(name="write", arguments="not json"),
    )
    assert _compress_tool_call_for_memory(tc, enabled=True) is tc


def test_compress_edit_success_returns_obs_on_internal_error() -> None:
    obs = "Edited x: 1 replacement OK.\n(1 lines, sha256~ab)"
    with mock.patch("scienceflow.core.mem.memory_context.re.match", side_effect=RuntimeError("boom")):
        assert compress_edit_success_output_for_memory(obs) == obs


def test_compress_bash_returns_obs_on_internal_error() -> None:
    """If compression fails, return original *obs* unchanged (never raise)."""

    class BoomSplitlines(str):
        def splitlines(self, *args, **kwargs):  # noqa: ANN001, ANN002
            raise RuntimeError("boom")

    body = "\n".join(f"L{i}" for i in range(25))
    obs = BoomSplitlines(f"[exit=0, 1.0s]\n{body}")
    out = compress_bash_tool_output_for_memory(obs, tail_lines=5)
    assert out is obs


def test_compress_tool_call_swallows_exception_returns_tc() -> None:
    class Boom:
        def __getattribute__(self, name: str):
            raise RuntimeError("boom")

    tc = SimpleNamespace(function=Boom())
    assert _compress_tool_call_for_memory(tc, enabled=True) is tc


# ---------- compress_read_output_for_memory ----------


def test_compress_read_short_file_unchanged() -> None:
    header = "[solution.py: 20 lines total]"
    body = "\n".join(f"    {i}|line {i}" for i in range(1, 21))
    obs = header + "\n" + body
    assert compress_read_output_for_memory(obs, max_lines=80) == obs


def test_compress_read_long_file_truncated() -> None:
    header = "[solution.py: 200 lines total, showing 1-200]"
    body_lines = [f"    {i}|line {i}" for i in range(1, 201)]
    obs = header + "\n" + "\n".join(body_lines)
    out = compress_read_output_for_memory(obs, max_lines=30)
    out_lines = out.splitlines()
    # Header preserved
    assert out_lines[0] == header
    # Only 30 content lines kept + truncation notice
    assert len(out_lines) == 32  # header + 30 body + truncation
    assert "truncated for LLM context" in out_lines[-1]
    assert "first 30 of 200" in out_lines[-1]


def test_compress_read_no_header_unchanged() -> None:
    """If first line doesn't start with '[', return unchanged."""
    obs = "some weird output\nline2"
    assert compress_read_output_for_memory(obs, max_lines=1) == obs


def test_compress_read_returns_obs_on_internal_error() -> None:
    class BoomSplitlines(str):
        def splitlines(self, *args, **kwargs):
            raise RuntimeError("boom")

    obs = BoomSplitlines("[file.py: 100 lines total]\nstuff")
    out = compress_read_output_for_memory(obs, max_lines=5)
    assert out is obs


def test_record_read_default_full_python_uses_code_map(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mreadmap", "Draft", 200)
    ws = tmp_path / "wreadmap"
    ws.mkdir()
    lines = [
        "import pandas as pd",
        "",
        "def engineer_features(df):",
        "    df = df.copy()",
        "    return df",
        "",
        "def train_predict(train, test):",
        "    return engineer_features(test)",
    ]
    lines += [f"# filler {i}" for i in range(80)]
    (ws / "solution.py").write_text("\n".join(lines), encoding="utf-8")
    read_body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(lines[:60], start=1))
    raw = f"[solution.py: {len(lines)} lines total, showing 1-60]\n{read_body}"
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        read_success_max_lines=20,
    )

    obs = ctx.record_tool_result(
        "read",
        {"path": "solution.py"},
        ToolResult(output=raw),
        raw_id="tool_000001_read.txt",
    )

    assert "read_code_map_v2" in obs
    assert "[tool-summary code-map: solution.py]" in obs
    assert "engineer_features(df):" in obs
    assert "# filler 70" not in obs
    assert "tool_000001_read.txt" in obs


def test_record_read_default_full_uses_snapshot_ref_after_write(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mreadref", "Draft", 200)
    ws = tmp_path / "wreadref"
    ws.mkdir()
    lines = ["def main():", "    return 1"]
    (ws / "solution.py").write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    ctx.record_tool_result(
        "write",
        {"path": "solution.py"},
        ToolResult(output="Written solution.py (2 lines, 40 bytes, sha256~abc123)"),
    )
    raw = "[solution.py: 2 lines total, showing 1-2]\n     1|def main():\n     2|    return 1"

    obs = ctx.record_tool_result(
        "read",
        {"path": "solution.py"},
        ToolResult(output=raw),
        raw_id="tool_000002_read.txt",
    )

    assert "snapshot_ref_v1" in obs
    assert "[see current snapshot: solution.py sha~" in obs
    assert "def main():" not in obs
    assert "tool_000002_read.txt" in obs


def test_record_grep_uses_snapshot_ref_for_high_volume_matches(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mgrepref", "Draft", 200)
    ws = tmp_path / "wgrepref"
    ws.mkdir()
    lines = [f"target_{i} = {i}" for i in range(12)]
    (ws / "solution.py").write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    ctx.record_tool_result(
        "write",
        {"path": "solution.py"},
        ToolResult(output="Written solution.py (12 lines, 200 bytes, sha256~abc123)"),
    )
    raw = "\n".join(f"solution.py:{i}:target_{i} = {i}" for i in range(1, 13))

    obs = ctx.record_tool_result(
        "grep",
        {"pattern": "target"},
        ToolResult(output=raw),
        raw_id="tool_000003_grep.txt",
    )

    assert "snapshot_ref_v1" in obs
    assert "solution.py: 12 matches (lines 1,2,3,4,5,6,7,8,9,10,...)" in obs
    assert "target_1 = 1" not in obs
    assert "tool_000003_grep.txt" in obs


# ---------- bash tool call args compression ----------


def test_compress_bash_tool_call_short_command_unchanged() -> None:
    tc = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"command": "ls -la"}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out is tc


def test_compress_bash_tool_call_long_command_preserved_for_cache() -> None:
    long_cmd = "echo " + "x" * 600
    tc = SimpleNamespace(
        id="c2",
        type="function",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"command": long_cmd}),
        ),
    )
    out = _compress_tool_call_for_memory(tc, enabled=True)
    assert out is tc


def test_assistant_message_from_api_preserves_large_write_tool_call() -> None:
    body = "print('hello')\n" + ("x" * 600)
    syn = SimpleNamespace(
        content="",
        reasoning_content="write file",
        tool_calls=[
            SimpleNamespace(
                id="call_write",
                type="function",
                function=SimpleNamespace(
                    name="write",
                    arguments=json.dumps({"path": "solution.py", "content": body}),
                ),
            ),
        ],
    )
    msg = _assistant_message_from_api(syn)
    assert msg.reasoning_content == "write file"
    tc = msg.tool_calls[0]
    fn = tc["function"] if isinstance(tc, dict) else tc.function
    raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
    args = json.loads(raw)
    assert args["path"] == "solution.py"
    assert args["content"] == body


def test_default_system_has_reasoning_discipline_and_placeholder_ban() -> None:
    assert "## Reasoning discipline" in _DEFAULT_SYSTEM
    assert "placeholder" in _DEFAULT_SYSTEM.lower()
    assert "MEMORY_COMPRESSED" in _DEFAULT_SYSTEM
    assert "thought" in _DEFAULT_SYSTEM.lower()
    assert "required" in _DEFAULT_SYSTEM.lower() or "`thought`" in _DEFAULT_SYSTEM
    assert "1–2 concise sentences" in _DEFAULT_SYSTEM
    assert "2–3 sentences" not in _DEFAULT_SYSTEM
    assert "Two to three" not in _DEFAULT_SYSTEM


def test_default_system_forbids_confirmatory_read_after_write_or_edit() -> None:
    assert "successful **write** or **edit**" in _DEFAULT_SYSTEM
    assert "double-check the write took effect" in _DEFAULT_SYSTEM


def test_default_system_explains_chat_memory_write_omit() -> None:
    assert "Historical write tool call omitted from executable LLM context" in _DEFAULT_SYSTEM
    assert "not a tool call and is not file content" in _DEFAULT_SYSTEM
    # Legacy formats still mentioned for backward-compat
    assert "WRITE_OK memory-compression" in _DEFAULT_SYSTEM
    assert "DO NOT COPY" in _DEFAULT_SYSTEM
    assert "chat-memory" in _DEFAULT_SYSTEM


def test_llm_view_keeps_large_write_tool_call_for_cache(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj", "Draft", 200)
    ws = tmp_path / "wproj"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    body = "print('hello')\n" + ("x" * 500)
    raw_args = {"path": "solution.py", "content": body, "thought": "write full file"}
    mem.add_message(
        Message(
            role="assistant",
            content="writing solution",
            tool_calls=[
                {
                    "id": "tc_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps(raw_args),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Written solution.py (12 lines, 520 bytes, sha256~abc123)",
            "write",
            "tc_write",
        ),
    )

    stored = mem.chat_history_memory.retrieve(window_size=None)
    stored_tc = stored[0].memory_record.message.tool_calls[0]
    stored_fn = stored_tc["function"] if isinstance(stored_tc, dict) else stored_tc.function
    stored_name = stored_fn["name"] if isinstance(stored_fn, dict) else stored_fn.name
    stored_raw = stored_fn["arguments"] if isinstance(stored_fn, dict) else stored_fn.arguments
    stored_args = json.loads(stored_raw)
    assert stored_name == "write"
    assert stored_args["content"] == body

    visible = ctx.build_messages_for_llm()
    assert visible[0].role == "assistant"
    assert getattr(visible[0], "tool_calls", None)
    visible_tc = visible[0].tool_calls[0]
    visible_fn = visible_tc["function"] if isinstance(visible_tc, dict) else visible_tc.function
    visible_raw = visible_fn["arguments"] if isinstance(visible_fn, dict) else visible_fn.arguments
    visible_args = json.loads(visible_raw)
    assert visible_args["path"] == "solution.py"
    assert visible_args["content"] == body
    assert "thought" not in visible_args
    assert visible[1].role == "tool"
    assert "Written solution.py" in (visible[1].content or "")


def test_llm_view_projects_placeholder_write_tool_call(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj_placeholder", "Draft", 200)
    ws = tmp_path / "wproj_placeholder"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    mimic = "[WRITE_OK memory-compression: 13617 chars -> sha256:abc123; full body applied to disk; NOT file content]"
    mem.add_message(
        Message(
            role="assistant",
            content="writing solution",
            tool_calls=[
                {
                    "id": "tc_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": mimic}),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Write rejected: content looked like a placeholder",
            "write",
            "tc_write",
        ),
    )

    visible = ctx.build_messages_for_llm()
    assert visible[0].role == "assistant"
    assert not getattr(visible[0], "tool_calls", None)
    assert "unsafe historical write payloads" in (visible[0].content or "")
    assert visible[1].role == "user"
    assert "Historical write tool call omitted" in (visible[1].content or "")


def test_llm_view_keeps_large_write_and_snapshot_result(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj_map", "Draft", 200)
    ws = tmp_path / "wproj_map"
    ws.mkdir()
    sol = ws / "solution.py"
    body = "\n".join(
        [
            "import argparse",
            "import xgboost as xgb",
            "import pandas as pd",
            "",
            "def engineer_features(df):",
            "    return df",
            "",
            "def train_and_evaluate(args):",
            "    pd.DataFrame({'id': [1]}).to_csv('submission.csv', index=False)",
            "",
            "def main():",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--epochs', type=int, default=1)",
            "    args = parser.parse_args()",
            "    train_and_evaluate(args)",
        ],
    )
    sol.write_text(body, encoding="utf-8")
    snap = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=1200,
    )
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    mem.add_message(
        Message(
            role="assistant",
            content="writing solution",
            tool_calls=[
                {
                    "id": "tc_write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": body}),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Written solution.py (15 lines, 500 bytes, sha256~abc123)\n\n" + snap,
            "write",
            "tc_write",
        ),
    )

    visible = ctx.build_messages_for_llm()
    text = "\n".join(str(m.content or "") for m in visible)
    assert getattr(visible[0], "tool_calls", None)
    visible_tc = visible[0].tool_calls[0]
    visible_fn = visible_tc["function"] if isinstance(visible_tc, dict) else visible_tc.function
    visible_args = json.loads(
        visible_fn["arguments"] if isinstance(visible_fn, dict) else visible_fn.arguments
    )
    assert visible_args["content"] == body
    assert "[tool-summary code-map: solution.py]" in text
    assert "engineer_features(df):" in text
    assert "train_and_evaluate(args):" in text
    assert "--epochs" in text
    assert "submission.csv" in text


def test_llm_view_regular_tool_projection_keeps_tool_call_serializable(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mproj2", "Draft", 200)
    ws = tmp_path / "wproj2"
    ws.mkdir()
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    mem.add_message(
        Message(
            role="assistant",
            content="checking files",
            tool_calls=[
                {
                    "id": "tc_bash",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "ls dataset", "thought": "internal note"},
                        ),
                    },
                },
            ],
        ),
    )
    mem.add_message(Message.tool_message("dataset files listed", "bash", "tc_bash"))

    visible = ctx.build_messages_for_llm()
    assert visible[0].tool_calls
    assert hasattr(visible[0].tool_calls[0], "model_dump")
    serialized = visible[0].to_dict()
    args = json.loads(serialized["tool_calls"][0]["function"]["arguments"])
    assert args == {"command": "ls dataset"}


def test_write_auto_snapshot_uses_semantic_layers_for_python(tmp_path) -> None:
    sol = tmp_path / "solution.py"
    before = "\n".join(
        [
            "import pandas as pd",
            "",
            "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:",
            '    """Feature engineering pipeline."""',
            "    df = df.copy()",
            "    df['ratio'] = df['a'] / df['b'].clip(lower=1)",
            "    df['ratio_log'] = df['ratio']",
            "    return df",
            "",
            "def train_predict(train_df, test_df):",
            "    model = object()",
            "    features = engineer_features(train_df)",
            "    return features",
        ],
    )
    after = before.replace(
        "    df['ratio_log'] = df['ratio']",
        "    df['ratio_log'] = df['ratio'].clip(0, 100)",
    )
    sol.write_text(after, encoding="utf-8")

    block = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=8000,
        previous_text=before,
        tool_name="write",
        args={"path": "solution.py"},
    )

    assert len(block) <= 8000
    assert "[tool-summary code-map: solution.py]" in block
    assert "[changed-range excerpt: solution.py::engineer_features lines 3-8]" in block
    assert "df['ratio_log'] = df['ratio'].clip(0, 100)" in block
    assert "[symbol-level summary: solution.py]" in block
    assert "def train_predict(train_df,test_df):" in block
    assert "[source-excerpt: head]" not in block
    assert "middle lines omitted" not in block


def test_llm_view_projects_large_edit_payload_as_summary(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "meditproj", "Draft", 200)
    ws = tmp_path / "weditproj"
    ws.mkdir()
    sol = ws / "solution.py"
    body = "\n".join(
        [
            "def engineer_features(df):",
            "    df = df.copy()",
            "    df['x'] = df['x'].clip(0, 10)",
            "    return df",
        ],
    )
    sol.write_text(body, encoding="utf-8")
    snap = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=8000,
        tool_name="edit",
        args={"path": "solution.py", "new_str": "df['x'] = df['x'].clip(0, 10)"},
    )
    old_payload = "old-anchor\n" + ("x = 1\n" * 120)
    new_payload = "new-anchor\n" + ("x = 2\n" * 120)
    mem.add_message(
        Message(
            role="assistant",
            content="editing solution",
            tool_calls=[
                {
                    "id": "tc_edit",
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps(
                            {
                                "path": "solution.py",
                                "old_str": old_payload,
                                "new_str": new_payload,
                            },
                        ),
                    },
                },
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            "Edited solution.py: 1 replacement OK.\n\n" + snap,
            "edit",
            "tc_edit",
        ),
    )
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)

    visible = ctx.build_messages_for_llm()
    text = "\n".join(str(m.content or "") for m in visible)
    assert visible[0].role == "assistant"
    assert not getattr(visible[0], "tool_calls", None)
    assert "edit payloads" in (visible[0].content or "")
    assert "Historical edit tool call omitted" in text
    assert "Changed range: [changed-range excerpt: solution.py::engineer_features" in text
    assert "[tool-summary code-map: solution.py]" in text
    assert old_payload not in text
    assert new_payload not in text


def test_write_auto_snapshot_prefers_code_map_under_small_budget(tmp_path) -> None:
    sol = tmp_path / "solution.py"
    sol.write_text(
        "\n".join(
            [
                "import argparse",
                "import xgboost as xgb",
                "import pandas as pd",
                "",
                "def engineer_features(df):",
                "    return df",
                "",
                "def train_and_evaluate(args):",
                "    pd.DataFrame({'id': [1]}).to_csv('submission.csv', index=False)",
                "",
                "def main():",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--epochs', type=int, default=1)",
                "    args = parser.parse_args()",
                "    train_and_evaluate(args)",
            ],
        ),
        encoding="utf-8",
    )

    block = _build_write_auto_snapshot_block(
        "solution.py",
        sol,
        max_lines=400,
        max_chars=900,
    )
    assert len(block) <= 900
    assert "[tool-summary code-map: solution.py]" in block
    assert "engineer_features(df):" in block
    assert "train_and_evaluate(args):" in block
    assert "--epochs" in block
    assert "submission.csv" in block
    assert "Do NOT call" not in block


def test_looks_like_write_placeholder_mimicry() -> None:
    assert _looks_like_write_placeholder_mimicry("<4884 chars>")
    assert _looks_like_write_placeholder_mimicry("  <12 chars>  ")
    assert _looks_like_write_placeholder_mimicry("<0 chars>")
    assert _looks_like_write_placeholder_mimicry(
        "[file content omitted - 500 chars written to disk]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "  [file content omitted - 51 chars written to disk]  ",
    )
    assert _looks_like_write_placeholder_mimicry(
        "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>",
    )
    assert _looks_like_write_placeholder_mimicry(
        "[interaction-log: write body omitted; 13617 chars applied by tool]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "# [chat-memory: write payload omitted (500 bytes); "
        "following write tool result confirms on-disk file.]",
    )
    # Legacy "DO NOT COPY" format (still recognised for backward-compat)
    assert _looks_like_write_placeholder_mimicry(
        "[DO NOT COPY - display-only stub: real 13617-char body applied to disk]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "# [DO NOT COPY - display-only stub: real 500-byte solution body on disk; "
        "see following write tool result.]",
    )
    # New "WRITE_OK memory-compression" format (current)
    assert _looks_like_write_placeholder_mimicry(
        "[WRITE_OK memory-compression: 13617 chars -> sha256:abc1234567890abc; "
        "full body applied to disk; NOT file content]",
    )
    assert _looks_like_write_placeholder_mimicry(
        "# [WRITE_OK memory-compression: 500 bytes -> sha256:def0987654321fed; "
        "the full file is on disk. This comment line is NOT file content; "
        "do NOT copy it as a new write payload. Use read tool only when you need the on-disk text "
        "to craft an edit or verify a crash.]",
    )
    mem_blob = (
        "x" * 60
        + "\n\n# [MEMORY_COMPRESSED: full file is 500 bytes; real content on disk]\n\n"
        + "x" * 60
    )
    assert _looks_like_write_placeholder_mimicry(mem_blob)
    mem_blob_v2 = (
        "x" * 60
        + "\n\n# <<<MEMORY_COMPRESSED: prior successful write, 500 bytes already on disk — "
        "do NOT rewrite to 'fix' this; use read if you need the full current content>>>\n\n"
        + "x" * 60
    )
    assert _looks_like_write_placeholder_mimicry(mem_blob_v2)
    mem_blob_write_ok = (
        "x" * 60
        + "\n\n# <<<WRITE_OK: 500 bytes written and syntax-validated — "
        "file is complete on disk. No read needed.>>>\n\n"
        + "x" * 60
    )
    assert _looks_like_write_placeholder_mimicry(mem_blob_write_ok)
    assert _looks_like_write_placeholder_mimicry(_SCRUBBED_WRITE_PLACEHOLDER_NOTE)
    assert not _looks_like_write_placeholder_mimicry("print('hi')")
    assert not _looks_like_write_placeholder_mimicry("<not digits chars>")
    assert not _looks_like_write_placeholder_mimicry(
        "[file content omitted - not-a-number chars written to disk]",
    )
    long_with_marker = (
        "# real file\n" * 200 + "# [MEMORY_COMPRESSED: not a real marker line]\n" + "z\n" * 200
    )
    assert not _looks_like_write_placeholder_mimicry(long_with_marker)


def test_scrub_last_assistant_write_tool_call_content() -> None:
    bad = "[file content omitted - 51 chars written to disk]"
    rec = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": bad}),
                    },
                },
            ],
        },
    }
    memory = SimpleNamespace(
        chat_history_memory=SimpleNamespace(
            storage=SimpleNamespace(memory_list=[rec]),
        ),
    )
    scrub_last_assistant_write_tool_call_content(memory, tool_call_id="tc1")
    assert rec["message"]["tool_calls"][0]["function"]["name"] == "write"
    args = json.loads(rec["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["content"] == bad


def test_bash_is_pure_read_only_whitelist() -> None:
    assert _bash_is_pure_read_only("cat a.py") is True
    assert _bash_is_pure_read_only("head -n 50 a.py | grep foo") is True
    assert _bash_is_pure_read_only("ENV=1 head a.py") is True
    assert _bash_is_pure_read_only("cat a.py | sed -n '1,5p'") is True
    assert _bash_is_pure_read_only("sed -i 's/a/b/' x.py") is False
    assert _bash_is_pure_read_only("cat > solution.py << 'PY'\nprint(1)\nPY") is False
    assert _bash_is_pure_read_only("grep foo solution.py > matches.txt") is False
    assert _bash_is_pure_read_only("cat a.py && ls") is False
    assert _bash_is_pure_read_only("python3 a.py") is False


def test_bash_command_dumps_python_source_heuristic() -> None:
    assert bash_command_dumps_python_source("cat -n solution.py") is True
    assert bash_command_dumps_python_source("cat -n solution.py | head -174") is True
    assert bash_command_dumps_python_source("cat > solution.py << 'PY'\nprint(1)\nPY") is False
    assert bash_command_dumps_python_source("head -5 dataset/train.csv") is False
    assert bash_command_dumps_python_source("grep foo solution.py") is False


def test_record_tool_result_bash_cat_solution_appends_read_coaching(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mreadco", "Draft", 200)
    ws = tmp_path / "wreadco"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=5,
        read_success_max_lines=50,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i:03d}" for i in range(80))
    o1 = ctx.record_tool_result("bash", {"command": "cat -n solution.py"}, ToolResult(output=long_out))
    assert "showing last" in o1
    assert "`read` tool" in o1


def test_record_tool_result_repeat_bash_cat_second_call_uncompressed(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m", "Draft", 200)
    ws = tmp_path / "w"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=5,
        read_success_max_lines=50,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i:03d}" for i in range(80))
    tr = ToolResult(output=long_out)
    cmd = {"command": "cat solution.py"}
    o1 = ctx.record_tool_result("bash", cmd, tr)
    assert "bash:" in o1 and "showing last" in o1
    mem.add_message(Message.tool_message(o1, "bash", "c1"))
    o2 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" not in o2
    assert "L000" in o2 and "L079" in o2


def test_record_tool_result_repeat_read_second_call_uncompressed(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m2", "Draft", 200)
    ws = tmp_path / "w2"
    ws.mkdir()
    p = ws / "f.txt"
    body = "\n".join(f"line-{i}" for i in range(300))
    p.write_text(
        "[f.txt: 300 lines total, showing 1-300]\n" + body,
        encoding="utf-8",
    )
    raw_read = p.read_text(encoding="utf-8")
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=5,
        read_success_max_lines=40,
    )
    args = {"path": "f.txt"}
    o1 = ctx.record_tool_result("read", args, ToolResult(output=raw_read))
    assert "read output truncated" in o1
    mem.add_message(Message.tool_message(o1, "read", "r1"))
    o2 = ctx.record_tool_result("read", args, ToolResult(output=raw_read))
    assert "read output truncated" not in o2
    assert "line-0" in o2 and "line-299" in o2


def test_record_tool_result_read_different_offset_not_treated_as_repeat(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m3", "Draft", 200)
    ws = tmp_path / "w3"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        read_success_max_lines=20,
    )
    n = 80
    a = f"[a: {n} lines total, showing 1-{n}]\n" + "\n".join(f"row-{i}" for i in range(n))
    o1 = ctx.record_tool_result("read", {"path": "a", "offset": 1}, ToolResult(output=a))
    assert "read output truncated" not in o1
    assert "row-79" in o1
    mem.add_message(Message.tool_message(o1, "read", "x1"))
    o2 = ctx.record_tool_result("read", {"path": "a", "offset": 2}, ToolResult(output=a))
    assert "read output truncated" not in o2


def test_record_tool_result_non_readonly_bash_repeat_still_compressed(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "m4", "Draft", 200)
    ws = tmp_path / "w4"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=4,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i}" for i in range(40))
    tr = ToolResult(output=long_out)
    cmd = {"command": "ls -la /tmp"}
    o1 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" in o1
    mem.add_message(Message.tool_message(o1, "bash", "b1"))
    o2 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" in o2


def test_gc_supersedes_first_compressed_after_different_tool(tmp_path) -> None:
    from deepcraft_core.tool import ToolCall
    from deepcraft_core.tool.base import Function

    mem = create_agent_memory(tmp_path / "mgc", "Draft", 200)
    ws = tmp_path / "wgc"
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=3,
        bash_success_tail_lines_readonly=3,
    )
    long_out = "[exit=0, 0.1s]\n" + "\n".join(f"L{i}" for i in range(25))
    tr = ToolResult(output=long_out)
    cmd = {"command": "cat solution.py"}

    def _asst_bash(tcid: str) -> None:
        mem.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=tcid,
                        function=Function(
                            name="bash",
                            arguments=json.dumps(cmd),
                        ),
                    ),
                ],
            ),
        )

    _asst_bash("t1")
    o1 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" in o1
    mem.add_message(Message.tool_message(o1, "bash", "t1"))

    _asst_bash("t2")
    o2 = ctx.record_tool_result("bash", cmd, tr)
    assert "showing last" not in o2
    mem.add_message(Message.tool_message(o2, "bash", "t2"))

    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="te",
                    function=Function(
                        name="edit",
                        arguments=json.dumps({"path": "x.py", "old_str": "a", "new_str": "b"}),
                    ),
                ),
            ],
        ),
    )
    ctx.record_tool_result(
        "edit",
        {"path": "x.py", "old_str": "a", "new_str": "b"},
        ToolResult(output="Edited x.py: 1 replacement OK.\ncontext\n(1 lines, sha256~abc)"),
    )

    records = mem.chat_history_memory.storage.load()
    first_tool = next(
        r
        for r in records
        if isinstance(r, dict)
        and isinstance(r.get("message"), dict)
        and r["message"].get("role") == "tool"
        and r["message"].get("tool_call_id") == "t1"
    )
    content = str(first_tool["message"].get("content") or "")
    assert "superseded" in content.lower()


def test_record_tool_write_appends_auto_snapshot_from_disk(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn", "Draft", 200)
    ws = tmp_path / "w_sn"
    ws.mkdir()
    (ws / "solution.py").write_text("print(1)\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    raw = "Written solution.py (1 lines, 10 bytes, sha256~deadbeef)\n     1|not-on-disk"
    obs = ctx.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))
    assert _AUTO_SNAPSHOT_PREFIX in obs
    assert "print(1)" in obs
    assert "not-on-disk" not in obs

    ctx2 = MemoryContextManager(
        mem, ws, tool_memory_compression=True, write_auto_snapshot_enabled=False
    )
    obs2 = ctx2.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))
    assert _AUTO_SNAPSHOT_PREFIX not in obs2


def test_record_tool_write_auto_snapshots_any_python_file_with_signatures(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn_py", "Draft", 200)
    ws = tmp_path / "w_sn_py"
    ws.mkdir()
    (ws / "utils.py").write_text(
        "def load_data(path, limit=100, *, strict=False):\n"
        "    return path\n\n"
        "class Trainer:\n"
        "    def fit(self, X, y=None):\n"
        "        return self\n",
        encoding="utf-8",
    )
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)

    obs = ctx.record_tool_result(
        "write",
        {"path": "utils.py"},
        ToolResult(output="Written utils.py (6 lines, 120 bytes, sha256~deadbeef)"),
    )

    assert _AUTO_SNAPSHOT_PREFIX in obs
    assert "[tool-summary code-map: utils.py]" in obs
    assert "load_data(path,limit=100,*,strict=False)" in obs
    assert "class Trainer" in obs
    assert "fit(self,X,y=None)" in obs


def test_record_tool_write_does_not_snapshot_non_code_outputs(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn_csv", "Draft", 200)
    ws = tmp_path / "w_sn_csv"
    ws.mkdir()
    (ws / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)

    obs = ctx.record_tool_result(
        "write",
        {"path": "submission.csv"},
        ToolResult(output="Written submission.csv (2 lines, 16 bytes, sha256~deadbeef)"),
    )

    assert _AUTO_SNAPSHOT_PREFIX not in obs


def test_record_tool_write_dedupes_unchanged_auto_snapshot(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mwsn_dedup", "Draft", 200)
    ws = tmp_path / "w_sn_dedup"
    ws.mkdir()
    (ws / "solution.py").write_text("print(1)\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    raw = "Written solution.py (1 lines, 10 bytes, sha256~deadbeef)\n     1|not-on-disk"

    obs1 = ctx.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))
    obs2 = ctx.record_tool_result("write", {"path": "solution.py"}, ToolResult(output=raw))

    assert obs1.count(_AUTO_SNAPSHOT_PREFIX) == 1
    assert _AUTO_SNAPSHOT_PREFIX not in obs2


def test_record_tool_edit_strips_raw_snapshot_and_appends_canonical_once(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "medsn", "Draft", 200)
    ws = tmp_path / "e_sn"
    ws.mkdir()
    (ws / "solution.py").write_text("a = 1\nb = 99\n", encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True)
    raw = (
        "Edited solution.py: 1 replacement OK.\n"
        "context around edit\n"
        "(2 lines, sha256~deadbeef)\n"
        "--- Current file snapshot (after edit) ---\n"
        "     1|raw tool snapshot should be dropped\n"
    )

    obs = ctx.record_tool_result("edit", {"path": "solution.py"}, ToolResult(output=raw))

    assert "context around edit" not in obs
    assert "--- Current file snapshot (after edit) ---" not in obs
    assert "raw tool snapshot should be dropped" not in obs
    assert obs.count(_AUTO_SNAPSHOT_PREFIX) == 1
    assert "     2|b = 99" in obs

    obs2 = ctx.record_tool_result("edit", {"path": "solution.py"}, ToolResult(output=raw))
    assert _AUTO_SNAPSHOT_PREFIX not in obs2
    assert "--- Current file snapshot (after edit) ---" not in obs2


def test_record_tool_read_overlap_nudge_subrange(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mrov", "Draft", 200)
    ws = tmp_path / "wrov"
    ws.mkdir()
    p = ws / "x.txt"
    all_lines = [f"line{i}" for i in range(80)]
    p.write_text("\n".join(all_lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, read_success_max_lines=200)

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = 80
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = all_lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[x.txt: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    o1 = ctx.record_tool_result(
        "read",
        {"path": "x.txt", "offset": 1, "limit": 40},
        ToolResult(output=_fmt(1, 40)),
    )
    assert _READ_OVERLAP_COACHING not in o1
    o2 = ctx.record_tool_result(
        "read",
        {"path": "x.txt", "offset": 20, "limit": 5},
        ToolResult(output=_fmt(20, 5)),
    )
    assert _READ_OVERLAP_COACHING in o2


def test_record_tool_redundant_read_solution_py_uses_coverage_summary(tmp_path) -> None:
    """Tier 2: redundant read of solution.py records coverage, not repeated source."""
    mem = create_agent_memory(tmp_path / "msil", "Draft", 200)
    ws = tmp_path / "wsil"
    ws.mkdir()
    sol = ws / "solution.py"
    sol_lines = [f"# line {i}" for i in range(40)]
    sol.write_text("\n".join(sol_lines), encoding="utf-8")
    ctx = MemoryContextManager(
        mem, ws,
        tool_memory_compression=True,
        read_success_max_lines=200,
        write_auto_snapshot_enabled=True,
        write_auto_snapshot_paths=("solution.py",),
        write_auto_snapshot_max_lines=400,
        write_auto_snapshot_max_chars=30_000,
    )

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = 40
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = sol_lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[solution.py: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    o1 = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 1, "limit": 40},
        ToolResult(output=_fmt(1, 40)),
    )
    assert _AUTO_SNAPSHOT_PREFIX not in o1, "first read must not get a snapshot block"
    assert _READ_OVERLAP_COACHING not in o1
    o2 = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 5, "limit": 10},
        ToolResult(output=_fmt(5, 10)),
    )
    assert _AUTO_SNAPSHOT_PREFIX not in o2
    assert "[read-coverage summary: solution.py]" in o2
    assert "requested lines: 5-14" in o2
    assert "already covered ranges: 1-40" in o2
    assert "# line 5" not in o2
    assert _READ_OVERLAP_COACHING not in o2, (
        "coverage replacement must not also append the neutral marker"
    )
    # Forbidden phrases (super loop integrity): no prescriptive language must leak into chat.
    forbidden = (
        "Do not re-read",
        "do not call `read`",
        "Do NOT call",
        "POST-TELEPORT",
        "workspace was swapped",
        "treat as the only source of truth",
    )
    for ph in forbidden:
        assert ph not in o2, f"silent intercept must not leak forbidden phrase: {ph!r}"
    # Counter increments for the redundant read path.
    assert ctx._redundant_read_count_by_path.get("solution.py") == 1


def test_record_tool_read_overlap_uses_symbol_coverage_raw_id(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "msym", "Draft", 200)
    ws = tmp_path / "wsym"
    ws.mkdir()
    sol = ws / "solution.py"
    lines = [
        "def foo(x):",
        '    """Foo transform."""',
        "    total = x",
        "    total += 1",
        "    total += 2",
        "    return total",
        "",
        "def bar(y):",
        "    value = y",
        "    return value",
    ]
    sol.write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        read_success_max_lines=200,
        write_auto_snapshot_enabled=True,
        write_auto_snapshot_paths=("solution.py",),
    )

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = len(lines)
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[solution.py: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    first = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 1, "limit": 6},
        ToolResult(output=_fmt(1, 6)),
        raw_id="tool_000001_read.txt",
    )
    assert "[read-coverage summary:" not in first

    second = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 3, "limit": 2},
        ToolResult(output=_fmt(3, 2)),
        raw_id="tool_000002_read.txt",
    )
    assert "[read-coverage summary: solution.py::foo]" in second
    assert "previous full read raw_id: tool_000001_read.txt" in second
    assert "covered symbol range: 1-6" in second
    assert "total += 1" not in second

    lines[8] = "    value = y + 1"
    sol.write_text("\n".join(lines), encoding="utf-8")
    ctx.record_tool_result(
        "write",
        {"path": "solution.py"},
        ToolResult(output="Written solution.py (10 lines, 150 bytes, sha256~def456)"),
    )
    third = ctx.record_tool_result(
        "read",
        {"path": "solution.py", "offset": 4, "limit": 1},
        ToolResult(output=_fmt(4, 1)),
        raw_id="tool_000003_read.txt",
    )
    assert "[read-coverage summary: solution.py::foo]" in third
    assert "previous full read raw_id: tool_000001_read.txt" in third
    assert "total += 1" not in third


def test_record_tool_redundant_read_other_path_uses_neutral_marker(tmp_path) -> None:
    """Tier 2: redundant read on a non-snapshot path uses the neutral marker, no coaching."""
    mem = create_agent_memory(tmp_path / "mneut", "Draft", 200)
    ws = tmp_path / "wneut"
    ws.mkdir()
    p = ws / "other.txt"
    all_lines = [f"line{i}" for i in range(20)]
    p.write_text("\n".join(all_lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, read_success_max_lines=200)

    def _fmt(off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        n = 20
        end = n if lim <= 0 else min(n, s0 + lim)
        chunk = all_lines[s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[other.txt: {n} lines total, showing {s0 + 1}-{end}]\n" + body

    ctx.record_tool_result(
        "read",
        {"path": "other.txt", "offset": 1, "limit": 20},
        ToolResult(output=_fmt(1, 20)),
    )
    o2 = ctx.record_tool_result(
        "read",
        {"path": "other.txt", "offset": 5, "limit": 5},
        ToolResult(output=_fmt(5, 5)),
    )
    # Non-snapshot path: keep body, append neutral marker (no canonical block).
    assert _AUTO_SNAPSHOT_PREFIX not in o2
    assert _READ_OVERLAP_COACHING in o2
    assert "Do not re-read" not in o2
    assert "Use that prior context" not in o2


def test_record_tool_write_clears_read_overlap_state(tmp_path) -> None:
    mem = create_agent_memory(tmp_path / "mclr", "Draft", 200)
    ws = tmp_path / "wclr"
    ws.mkdir()
    p = ws / "x.txt"
    lines = [f"row{i}" for i in range(10)]
    p.write_text("\n".join(lines), encoding="utf-8")
    ctx = MemoryContextManager(mem, ws, tool_memory_compression=True, read_success_max_lines=20)

    def _fmt(nl: int, off: int, lim: int) -> str:
        s0 = max(0, off - 1)
        if lim <= 0:
            end = nl
        else:
            end = min(nl, s0 + lim)
        chunk = lines[:nl][s0:end]
        body = "\n".join(f"{i:>6}|{ln}" for i, ln in enumerate(chunk, start=s0 + 1))
        return f"[x.txt: {nl} lines total, showing {s0 + 1}-{end}]\n" + body

    o1 = ctx.record_tool_result(
        "read", {"path": "x.txt", "offset": 1, "limit": 5}, ToolResult(output=_fmt(10, 1, 5))
    )
    assert _READ_OVERLAP_COACHING not in o1
    # rewrite same file path — clears read coverage for x.txt
    lines = [f"new{i}" for i in range(10)]
    p.write_text("\n".join(lines), encoding="utf-8")
    wout = "Written x.txt (10 lines, 100 bytes, sha256~a)\n" + "\n".join(
        f"  {i:4d}|{lines[i-1]}" for i in range(1, 11)
    )
    ctx.record_tool_result("write", {"path": "x.txt"}, ToolResult(output=wout))
    o2 = ctx.record_tool_result(
        "read", {"path": "x.txt", "offset": 1, "limit": 5}, ToolResult(output=_fmt(10, 1, 5))
    )
    assert _READ_OVERLAP_COACHING not in o2
    o3 = ctx.record_tool_result(
        "read", {"path": "x.txt", "offset": 1, "limit": 5}, ToolResult(output=_fmt(10, 1, 5))
    )
    assert _READ_OVERLAP_COACHING in o3


def test_scrub_last_assistant_write_tool_call_content_new_placeholder() -> None:
    bad = (
        "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>"
    )
    rec = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "solution.py", "content": bad}),
                    },
                },
            ],
        },
    }
    memory = SimpleNamespace(
        chat_history_memory=SimpleNamespace(
            storage=SimpleNamespace(memory_list=[rec]),
        ),
    )
    scrub_last_assistant_write_tool_call_content(memory, tool_call_id="tc1")
    assert rec["message"]["tool_calls"][0]["function"]["name"] == "write"
    args = json.loads(rec["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["content"] == bad
