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

"""Tests for clone-continue inherited memory compression."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepcraft_core import Message
from deepcraft_core.tool import ToolCall
from deepcraft_core.tool.base import Function

from scienceflow.core.agent_runtime import create_agent_memory
from scienceflow.core.mem.memory_context import (
    MemoryContextManager,
    apply_clone_inherit_compression,
    apply_clone_minimal_slice_to_memory,
    apply_clone_non_a_memory_budget,
    build_shallow_continuity_capsule,
    compress_inherited_tool_message_content,
    slice_clone_minimal_inherited_messages,
    trim_tool_feedback_for_llm_context,
    _compile_clone_inherit_signal_patterns,
)


def _cfg(**overrides: object) -> SimpleNamespace:
    base = {
        "clone_inherit_compress": True,
        "clone_inherit_read_max_lines": 5,
        "clone_inherit_bash_tail_lines": 3,
        "clone_inherit_omit_solution_py_reads": True,
        "clone_inherit_edit_compact": True,
        "clone_inherit_write_compact": True,
        "clone_inherit_tool_args_head_tail_chars": 120,
        "clone_inherit_last_state_write_exempt": True,
        "clone_inherit_signal_patterns": None,
        "clone_non_a_inherit_budget_chars": 1200,
        "clone_bdeep_shallow_capsule_max_chars": 2000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_compress_read_keeps_signal_line_from_dropped_middle() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    header = "[dataset/foo.csv: 50 lines total, showing 1-50]"
    body = [f"{i}|row" for i in range(1, 51)]
    body[25] = "26|Final Validation Score: 0.42"
    content = header + "\n" + "\n".join(body)
    out = compress_inherited_tool_message_content(
        content,
        tool_name="read",
        tool_args={"path": "dataset/foo.csv"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=False,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
    )
    assert "truncated for LLM context" in out
    assert "Final Validation Score: 0.42" in out


def test_compress_bash_success_preserves_metric_outside_tail() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    lines = ["[exit=0, 0.1s]"]
    lines.extend([f"log {i}" for i in range(15)])
    lines.insert(3, "Final Validation Score: 0.88")
    lines.extend([f"tail {i}" for i in range(10)])
    content = "\n".join(lines)
    out = compress_inherited_tool_message_content(
        content,
        tool_name="bash",
        tool_args=None,
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
    )
    assert "Final Validation Score: 0.88" in out
    assert "tail 9" in out


def test_compress_write_success_short_line() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    raw = "Written solution.py (10 lines, 200 bytes, sha256~abc)\n--- snapshot ---\n"
    out = compress_inherited_tool_message_content(
        raw,
        tool_name="write",
        tool_args={"path": "solution.py"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
    )
    assert "File `solution.py` written successfully" in out
    assert "10 lines" in out
    assert "sha256~abc" in out


def test_compress_edit_success_two_lines() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    raw = (
        "Edited solution.py: 1 replacement OK.\n"
        "context\n"
        "(200 lines, sha256~deadbeef01234567)"
    )
    out = compress_inherited_tool_message_content(
        raw,
        tool_name="edit",
        tool_args={"path": "solution.py"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
    )
    lines = out.splitlines()
    assert len(lines) == 2
    assert "Edited solution.py" in lines[0]
    assert "sha256~" in lines[1]


def test_omit_solution_py_read_stub() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    header = "[solution.py: 5 lines total, showing 1-5]"
    body = "\n".join(f"{i}|c" for i in range(1, 6))
    content = header + "\n" + body
    out = compress_inherited_tool_message_content(
        content,
        tool_name="read",
        tool_args={"path": "solution.py"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
    )
    assert "read omitted" in out


def test_apply_skips_inherited_fullrun_synthetic(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "memff", "Draft", 100)
    long_bash = "[exit=0, 0.1s]\n" + "\n".join(f"L{i}" for i in range(50))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=Function(name="bash", arguments=json.dumps({"command": "true"})),
                ),
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(long_bash, "bash", "inherited_fullrun_abc"),
    )
    stats = apply_clone_inherit_compression(mem, _cfg())
    assert stats["tool_msgs"] == 0
    rec = mem.chat_history_memory.retrieve(window_size=None)[-1].memory_record.message
    assert len((rec.content or "").splitlines()) > 40


def test_apply_truncates_write_tool_call_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    mem_dir = tmp_path / "mem"
    mem = create_agent_memory(mem_dir, "Draft", 200)
    big = "X" * 400
    args = json.dumps({"path": "f.py", "content": big}, ensure_ascii=False)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w1", function=Function(name="write", arguments=args)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("ok", "write", "w1"))
    stats = apply_clone_inherit_compression(mem, _cfg(clone_inherit_tool_args_head_tail_chars=80))
    assert stats["assistant_tc_args"] == 1
    m0 = mem.chat_history_memory.retrieve(window_size=None)[0].memory_record.message
    assert m0.tool_calls
    raw = m0.tool_calls[0].function.arguments
    d = json.loads(raw)
    assert len(d["content"]) < len(big)
    assert "truncated on inherit" in d["content"]


def test_preserve_initial_eda_keeps_prefix_before_first_code_write(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "eda", "Draft", 200)
    read_content = "[dataset/train.csv: 50 lines total, showing 1-50]\n" + "\n".join(
        f"{i}|EDA_FULL_LINE_{i}" for i in range(50)
    )
    scratch_content = "E" * 400
    post_solution_bash = "[exit=0, 0.1s]\n" + "\n".join(f"post {i}" for i in range(50))

    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="read_eda",
                    function=Function(
                        name="read",
                        arguments=json.dumps({"path": "dataset/train.csv"}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message(read_content, "read", "read_eda"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="write_scratch",
                    function=Function(
                        name="write",
                        arguments=json.dumps(
                            {"path": "eda_notes.py", "content": scratch_content},
                        ),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written eda_notes.py", "write", "write_scratch"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="write_solution",
                    function=Function(
                        name="write",
                        arguments=json.dumps({"path": "solution.py", "content": "S" * 400}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written solution.py", "write", "write_solution"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="post_bash",
                    function=Function(
                        name="bash",
                        arguments=json.dumps({"command": "python3 solution.py"}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message(post_solution_bash, "bash", "post_bash"))

    stats = apply_clone_inherit_compression(
        mem,
        _cfg(clone_inherit_tool_args_head_tail_chars=80),
        preserve_initial_eda=True,
    )

    recs = mem.chat_history_memory.retrieve(window_size=None)
    preserved_read = recs[1].memory_record.message.content or ""
    preserved_scratch_args = recs[2].memory_record.message.tool_calls[0].function.arguments or "{}"
    post_bash = recs[-1].memory_record.message.content or ""
    assert stats["tool_msgs"] == 1
    assert "EDA_FULL_LINE_49" in preserved_read
    assert "truncated for LLM context" not in preserved_read
    assert "truncated on inherit" in json.loads(preserved_scratch_args)["content"]
    assert len(post_bash.splitlines()) < len(post_solution_bash.splitlines())
    assert "post 49" in post_bash


def test_preserve_initial_eda_compresses_stale_fork_prompts_after_write(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "stale_prompts", "Draft", 200)
    eda_note = "EDA details should stay verbatim"
    mem.add_message(Message(role="user", content=eda_note))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="write_solution",
                    function=Function(
                        name="write",
                        arguments=json.dumps({"path": "solution.py", "content": "print(1)"}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written solution.py", "write", "write_solution"))
    old_backtrack = "## Trajectory branch history (top 5 shown in detail)\n" + ("old\n" * 200)
    old_guard = "[LNR] Detected repeated single-`read` rounds.\n" + ("guard\n" * 100)
    mem.add_message(Message(role="user", content=old_backtrack))
    mem.add_message(Message(role="user", content=old_guard))

    stats = apply_clone_inherit_compression(
        mem,
        _cfg(),
        preserve_initial_eda=True,
    )

    recs = mem.chat_history_memory.retrieve(window_size=None)
    contents = [r.memory_record.message.content or "" for r in recs]
    assert contents[0] == eda_note
    assert stats["user_msgs"] == 2
    assert "prior C_BACKTRACK trajectory" in contents[-2]
    assert "prior repeated-read warning" in contents[-1]
    assert "old\nold\nold" not in contents[-2]


def test_preserve_initial_eda_compresses_stale_peer_prompt_after_write(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "stale_peer", "Draft", 200)
    mem.add_message(Message(role="user", content="EDA details should stay verbatim"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="write_solution",
                    function=Function(
                        name="write",
                        arguments=json.dumps({"path": "solution.py", "content": "print(1)"}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written solution.py", "write", "write_solution"))
    old_peer = "## Reference peer trajectories\n" + ("peer detail\n" * 200)
    mem.add_message(Message(role="user", content=old_peer))

    stats = apply_clone_inherit_compression(mem, _cfg(), preserve_initial_eda=True)

    recs = mem.chat_history_memory.retrieve(window_size=None)
    content = recs[-1].memory_record.message.content or ""
    assert stats["user_msgs"] == 1
    assert "prior B_ENSEMBLE peer list" in content
    assert "peer detail\npeer detail" not in content


def test_bdeep_budget_preserves_eda_and_latest_state_capsule(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "bdeep_budget", "Draft", 200)
    mem.add_message(Message(role="user", content="EDA_SIGNAL keep this dataset observation"))
    big_old = "[exit=0, 0.1s]\n" + ("old log\n" * 300)
    code = "print('current')\n"
    mem.add_message(
        Message(
            role="assistant",
            content="Decision: keep shallow feature pipeline; deep should tune epochs.",
            tool_calls=[
                ToolCall(
                    id="write_solution",
                    function=Function(
                        name="write",
                        arguments=json.dumps({"path": "solution.py", "content": code}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            (
                "File `solution.py` written successfully (1 lines, sha256~abc123).\n\n"
                "[auto-snapshot after successful write: solution.py]\n"
                "[solution.py: 1 lines total, sha256~abc123, showing 1-1]\n"
                "     1|print('current')\n"
            ),
            "write",
            "write_solution",
        ),
    )
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="old_bash",
                    function=Function(name="bash", arguments=json.dumps({"command": "python old.py"})),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message(big_old, "bash", "old_bash"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="run_current",
                    function=Function(
                        name="bash",
                        arguments=json.dumps({"command": "python3 solution.py"}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(
        Message.tool_message("[exit=0, 1.0s]\nFinal Validation Score: 0.12\n", "bash", "run_current"),
    )

    capsule = build_shallow_continuity_capsule(mem, max_chars=1200)
    stats = apply_clone_non_a_memory_budget(
        mem,
        _cfg(clone_non_a_inherit_budget_chars=700),
        fork_class="B_DEEP",
    )

    recs = mem.chat_history_memory.retrieve(window_size=None)
    text = "\n".join((r.memory_record.message.content or "") for r in recs)
    assert stats["changed"] is True
    assert "EDA_SIGNAL" in text
    assert "print('current')" in text
    assert "Final Validation Score: 0.12" in text
    assert "old log\nold log" not in text
    assert "Recent shallow continuity capsule" in capsule
    assert "Final Validation Score: 0.12" in capsule


def test_preserve_initial_eda_keeps_only_latest_solution_snapshot(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "latest_snapshot", "Draft", 200)
    mem.add_message(Message(role="user", content="EDA details should stay verbatim"))

    def add_solution_write(tid: str, sha: str, line: str) -> None:
        mem.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=tid,
                        function=Function(
                            name="write",
                            arguments=json.dumps(
                                {"path": "solution.py", "content": f"print({line!r})"},
                            ),
                        ),
                    ),
                ],
            ),
        )
        mem.add_message(
            Message.tool_message(
                (
                    f"File `solution.py` written successfully (2 lines, sha256~{sha}).\n\n"
                    "[auto-snapshot after successful write: solution.py]\n"
                    f"[solution.py: 2 lines total, sha256~{sha}, showing 1-2]\n"
                    f"     1|{line}\n"
                    "     2|done\n"
                ),
                "write",
                tid,
            ),
        )

    add_solution_write("w1", "aaa111", "old-code")
    mem.add_message(Message(role="user", content="## Trajectory branch history\nold branch"))
    add_solution_write("w2", "bbb222", "new-code")

    stats = apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_write_snapshot_max_lines=20,
            clone_inherit_write_snapshot_max_chars=2000,
        ),
        preserve_initial_eda=True,
    )

    recs = mem.chat_history_memory.retrieve(window_size=None)
    contents = [r.memory_record.message.content or "" for r in recs]
    old_tool = contents[2]
    new_tool = contents[-1]
    assert stats["snapshots_dropped"] == 1
    assert "old-code" not in old_tool
    assert "auto-snapshot omitted" in old_tool
    assert "new-code" in new_tool
    assert "[auto-snapshot after successful write:" in new_tool


def test_preserve_initial_eda_keeps_latest_snapshot_per_python_file(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "latest_snapshot_per_file", "Draft", 200)
    mem.add_message(Message(role="user", content="EDA details should stay verbatim"))

    def add_write(tid: str, path: str, sha: str, line: str) -> None:
        mem.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=tid,
                        function=Function(
                            name="write",
                            arguments=json.dumps({"path": path, "content": line}),
                        ),
                    ),
                ],
            ),
        )
        mem.add_message(
            Message.tool_message(
                (
                    f"File `{path}` written successfully (1 lines, sha256~{sha}).\n\n"
                    f"[auto-snapshot after successful write: {path}]\n"
                    "[current `{path}` snapshot summary; deterministic tool-result memory]\n"
                    f"[tool-summary code-map: {path}]\n- lines: 1\n- sha256:{sha}\n"
                    f"[{path}: 1 lines total, sha256~{sha}]\n"
                    f"     1|{line}\n"
                ),
                "write",
                tid,
            ),
        )

    add_write("s1", "solution.py", "aaa111", "old-solution")
    add_write("u1", "utils.py", "uuu111", "current-utils")
    mem.add_message(Message(role="user", content="## Trajectory branch history\nold branch"))
    add_write("s2", "solution.py", "bbb222", "new-solution")

    stats = apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_write_snapshot_max_lines=20,
            clone_inherit_write_snapshot_max_chars=2000,
        ),
        preserve_initial_eda=True,
    )

    text = "\n".join(
        str(r.memory_record.message.content or "")
        for r in mem.chat_history_memory.retrieve(window_size=None)
    )
    assert stats["snapshots_dropped"] == 1
    assert "old-solution" not in text
    assert "new-solution" in text
    assert "current-utils" in text


def test_preserve_initial_eda_compresses_last_solution_write_args(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "last_args", "Draft", 200)
    mem.add_message(Message(role="user", content="EDA details should stay verbatim"))
    big_code = "print('current')\n" * 200
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="w_current",
                    function=Function(
                        name="write",
                        arguments=json.dumps(
                            {"path": "solution.py", "content": big_code},
                        ),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(
        Message.tool_message(
            (
                "File `solution.py` written successfully (200 lines, sha256~abc123).\n\n"
                "[auto-snapshot after successful write: solution.py]\n"
                "[solution.py: 200 lines total, sha256~abc123, snapshot truncated for memory budget]\n"
                "     1|print('current')\n"
            ),
            "write",
            "w_current",
        ),
    )

    stats = apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_tool_args_head_tail_chars=120,
            clone_inherit_last_state_write_exempt=True,
        ),
        preserve_initial_eda=True,
    )

    recs = mem.chat_history_memory.retrieve(window_size=None)
    raw = recs[1].memory_record.message.tool_calls[0].function.arguments
    args = json.loads(raw)
    assert stats["assistant_tc_args"] == 1
    assert len(args["content"]) < len(big_code)
    assert "truncated on inherit" in args["content"]


def test_last_state_write_exempt_does_not_apply_outside_repl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Last state-write exemption is ignored outside REPL; snapshots are the code source of truth."""
    monkeypatch.chdir(tmp_path)
    mem = create_agent_memory(tmp_path / "msol", "Draft", 200)
    small_budget = 80
    big1 = "A" * 400
    big2 = "B" * 400
    args1 = json.dumps(
        {"path": "solution.py", "content": big1},
        ensure_ascii=False,
    )
    args2 = json.dumps(
        {"path": "solution.py", "content": big2},
        ensure_ascii=False,
    )
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w1", function=Function(name="write", arguments=args1)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("ok", "write", "w1"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w2", function=Function(name="write", arguments=args2)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("ok", "write", "w2"))
    stats = apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_tool_args_head_tail_chars=small_budget,
            clone_inherit_last_state_write_exempt=True,
        ),
    )
    assert stats["assistant_tc_args"] == 2
    recs = mem.chat_history_memory.retrieve(window_size=None)
    c0 = json.loads(recs[0].memory_record.message.tool_calls[0].function.arguments or "{}")["content"]
    c1 = json.loads(recs[2].memory_record.message.tool_calls[0].function.arguments or "{}")["content"]
    assert "truncated on inherit" in c0
    assert len(c0) < len(big1)
    assert "truncated on inherit" in c1
    assert len(c1) < len(big2)


def test_last_state_write_exempt_outside_repl_two_writes_in_one_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Last state-write exemption is ignored outside REPL even within a bundled message."""
    monkeypatch.chdir(tmp_path)
    mem = create_agent_memory(tmp_path / "m2s", "Draft", 200)
    small_budget = 80
    big1 = "C" * 400
    big2 = "D" * 400
    args1 = json.dumps({"path": "solution.py", "content": big1}, ensure_ascii=False)
    args2 = json.dumps({"path": "solution.py", "content": big2}, ensure_ascii=False)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w1", function=Function(name="write", arguments=args1)),
                ToolCall(id="w2", function=Function(name="write", arguments=args2)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("ok", "write", "w1"))
    mem.add_message(Message.tool_message("ok", "write", "w2"))
    stats = apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_tool_args_head_tail_chars=small_budget,
            clone_inherit_last_state_write_exempt=True,
        ),
    )
    assert stats["assistant_tc_args"] == 1
    m = mem.chat_history_memory.retrieve(window_size=None)[0].memory_record.message
    c0 = json.loads(m.tool_calls[0].function.arguments or "{}")["content"]
    c1 = json.loads(m.tool_calls[1].function.arguments or "{}")["content"]
    assert "truncated on inherit" in c0
    assert len(c0) < len(big1)
    assert "truncated on inherit" in c1
    assert len(c1) < len(big2)


def test_last_state_write_exempt_off_truncates_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the flag is False, all write contents are middle-truncated."""
    monkeypatch.chdir(tmp_path)
    mem = create_agent_memory(tmp_path / "moff", "Draft", 200)
    small_budget = 80
    big1 = "E" * 400
    big2 = "F" * 400
    args1 = json.dumps({"path": "solution.py", "content": big1}, ensure_ascii=False)
    args2 = json.dumps({"path": "solution.py", "content": big2}, ensure_ascii=False)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w1", function=Function(name="write", arguments=args1)),
                ToolCall(id="w2", function=Function(name="write", arguments=args2)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("ok", "write", "w1"))
    mem.add_message(Message.tool_message("ok", "write", "w2"))
    apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_tool_args_head_tail_chars=small_budget,
            clone_inherit_last_state_write_exempt=False,
        ),
    )
    m = mem.chat_history_memory.retrieve(window_size=None)[0].memory_record.message
    c0 = json.loads(m.tool_calls[0].function.arguments or "{}")["content"]
    c1 = json.loads(m.tool_calls[1].function.arguments or "{}")["content"]
    assert "truncated on inherit" in c0
    assert "truncated on inherit" in c1
    assert len(c1) < len(big2)


def test_repl_mode_preserves_read_verbatim(tmp_path: Path) -> None:
    """REPL mode: read/write/edit tool messages pass through verbatim; no stub replacement."""
    mem = create_agent_memory(tmp_path / "mrepl_read", "Draft", 200)
    long_read_body = "\n".join(f"{i:>6}|line {i}" for i in range(1, 51))
    long_read = "[solution.py: 50 lines total, showing 1-50]\n" + long_read_body
    edit_ok = (
        "Edited solution.py: 1 replacement OK.\n"
        "context line\n"
        "(50 lines, sha256~abc123)"
    )
    write_ok = (
        "Written solution.py (50 lines, 2000 bytes, sha256~abc123)\n"
        "--- Current file snapshot (after write) ---\n"
        "extra-line"
    )
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="r1", function=Function(name="read", arguments=json.dumps({"path": "solution.py"}))),
            ],
        ),
    )
    mem.add_message(Message.tool_message(long_read, "read", "r1"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="e1", function=Function(name="edit", arguments=json.dumps({"path": "solution.py"}))),
            ],
        ),
    )
    mem.add_message(Message.tool_message(edit_ok, "edit", "e1"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="w1",
                    function=Function(name="write", arguments=json.dumps({"path": "solution.py", "content": "x"})),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message(write_ok, "write", "w1"))

    stats = apply_clone_inherit_compression(mem, _cfg(), repl_mode=True)

    assert stats["tool_msgs"] == 0
    recs = mem.chat_history_memory.retrieve(window_size=None)
    bodies = [r.memory_record.message.content for r in recs if r.memory_record.message.role == "tool"]
    assert long_read in bodies, "read result must be verbatim in repl_mode"
    assert edit_ok in bodies, "edit ok must be verbatim in repl_mode"
    assert write_ok in bodies, "write ok must be verbatim in repl_mode"
    for b in bodies:
        assert "read omitted" not in b
        assert "written successfully" not in b


def test_repl_mode_bash_error_signal_line_extraction(tmp_path: Path) -> None:
    """REPL mode: inherited bash errors keep signal lines (Error type, traceback) but body is trimmed."""
    mem = create_agent_memory(tmp_path / "mrepl_err", "Draft", 200)
    # Build a long bash error: 60 body lines, with a KeyError signal buried in the middle
    body_lines = [f"log line {i}" for i in range(60)]
    body_lines[30] = "KeyError: 'spacegroup'"
    body_lines[29] = "Traceback (most recent call last):"
    long_bash_err = "[exit=1, 5.0s]\n" + "\n".join(body_lines)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="b1", function=Function(name="bash", arguments=json.dumps({"command": "python3 solution.py"}))),
            ],
        ),
    )
    mem.add_message(Message.tool_message(long_bash_err, "bash", "b1"))

    # bash_tail_lines=3: keeps last 3 body lines + any signal lines from dropped middle
    stats = apply_clone_inherit_compression(mem, _cfg(clone_inherit_bash_tail_lines=3), repl_mode=True)

    recs = mem.chat_history_memory.retrieve(window_size=None)
    bash_body = next(
        r.memory_record.message.content
        for r in recs
        if r.memory_record.message.role == "tool"
    )
    # The body must be compressed (not the original 60-line error)
    assert bash_body != long_bash_err
    # Signal lines must be preserved
    assert "Traceback" in bash_body or "KeyError" in bash_body
    # Tail lines (last 3) must be present
    assert "log line 59" in bash_body
    assert stats["tool_msgs"] == 1


def test_repl_mode_bash_success_preserved(tmp_path: Path) -> None:
    """REPL mode: successful bash results (exit=0) are also signal-line compressed if long."""
    mem = create_agent_memory(tmp_path / "mrepl_ok", "Draft", 200)
    long_bash_ok = "[exit=0, 2.0s]\n" + "\n".join(f"L{i}" for i in range(40))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="b2", function=Function(name="bash", arguments=json.dumps({"command": "ls"}))),
            ],
        ),
    )
    mem.add_message(Message.tool_message(long_bash_ok, "bash", "b2"))

    stats = apply_clone_inherit_compression(mem, _cfg(clone_inherit_bash_tail_lines=3), repl_mode=True)

    recs = mem.chat_history_memory.retrieve(window_size=None)
    bash_body = next(
        r.memory_record.message.content
        for r in recs
        if r.memory_record.message.role == "tool"
    )
    # Success path: compressed to tail
    assert bash_body != long_bash_ok
    assert "L39" in bash_body
    assert stats["tool_msgs"] == 1


def test_trim_tool_feedback_preserves_write_auto_snapshot() -> None:
    snapshot_lines = "\n".join(f"{i:>6}|line {i}" for i in range(1, 121))
    feedback = (
        "File `solution.py` written successfully (120 lines, sha256~abc). "
        "Syntax check passed.\n\n"
        "[auto-snapshot after successful write: solution.py]\n"
        "File is complete on disk.\n"
        "[solution.py: 120 lines total, sha256~abc, showing 1-120]\n"
        + snapshot_lines
    )

    out = trim_tool_feedback_for_llm_context(feedback, max_chars=500)

    assert out.startswith("[Tool feedback trimmed for LLM context:")
    assert "auto-snapshot preserved" in out
    assert "     1|line 1" in out
    assert "    60|line 60" in out
    assert "   120|line 120" in out


def test_repl_mode_still_truncates_huge_assistant_args_and_exempts_last_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REPL mode: assistant tool_call middle-truncation still runs; last code/state write is exempt."""
    monkeypatch.chdir(tmp_path)
    mem = create_agent_memory(tmp_path / "mrepl2", "Draft", 200)
    big1 = "A" * 400
    big2 = "B" * 400
    args1 = json.dumps({"path": "solution.py", "content": big1}, ensure_ascii=False)
    args2 = json.dumps({"path": "solution.py", "content": big2}, ensure_ascii=False)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w1", function=Function(name="write", arguments=args1)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written solution.py (1 lines, 1 bytes, sha256~aaa)", "write", "w1"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="w2", function=Function(name="write", arguments=args2)),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written solution.py (1 lines, 1 bytes, sha256~bbb)", "write", "w2"))

    stats = apply_clone_inherit_compression(
        mem,
        _cfg(
            clone_inherit_tool_args_head_tail_chars=80,
            clone_inherit_last_state_write_exempt=True,
        ),
        repl_mode=True,
    )

    assert stats["assistant_tc_args"] == 1
    assert stats["tool_msgs"] == 0
    recs = mem.chat_history_memory.retrieve(window_size=None)
    c0 = json.loads(recs[0].memory_record.message.tool_calls[0].function.arguments or "{}")["content"]
    c1 = json.loads(recs[2].memory_record.message.tool_calls[0].function.arguments or "{}")["content"]
    assert "truncated on inherit" in c0
    assert len(c0) < len(big1)
    assert c1 == big2
    tool_bodies = [r.memory_record.message.content for r in recs if r.memory_record.message.role == "tool"]
    for b in tool_bodies:
        assert b.startswith("Written solution.py")


def test_replay_state_rebuilds_read_coverage(tmp_path: Path) -> None:
    src = "\n".join(f"print({i})" for i in range(1, 301))
    (tmp_path / "solution.py").write_text(src, encoding="utf-8")
    mem = create_agent_memory(tmp_path / "mreplay", "Draft", 200)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="r1",
                    function=Function(
                        name="read",
                        arguments=json.dumps({"path": "solution.py", "offset": 1, "limit": 100}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("[solution.py: 300 lines total, showing 1-100]", "read", "r1"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="r2",
                    function=Function(
                        name="read",
                        arguments=json.dumps({"path": "solution.py", "offset": 101, "limit": 100}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("[solution.py: 300 lines total, showing 101-200]", "read", "r2"))
    mgr = MemoryContextManager(mem, tmp_path)

    stats = mgr.replay_state_from_inherited_memory()

    assert stats["read_coverage"] == 2
    assert stats["tool_signatures"] == 2
    sha, intervals = mgr._read_coverage["solution.py"]
    assert sha
    assert intervals == [(1, 200)]


def test_replay_state_post_write_clears_read_coverage_and_seeds_snapshot(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("print('new')\n", encoding="utf-8")
    mem = create_agent_memory(tmp_path / "mreplay_write", "Draft", 200)
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="r1",
                    function=Function(
                        name="read",
                        arguments=json.dumps({"path": "solution.py", "offset": 1, "limit": 10}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("[solution.py: 1 lines total, showing 1-1]", "read", "r1"))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="w1",
                    function=Function(
                        name="write",
                        arguments=json.dumps({"path": "solution.py", "content": "print('new')\n"}),
                    ),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message("Written solution.py (1 lines, 13 bytes, sha256~abc)", "write", "w1"))
    mgr = MemoryContextManager(mem, tmp_path)

    stats = mgr.replay_state_from_inherited_memory()

    assert "solution.py" not in mgr._read_coverage
    assert "solution.py" in mgr._file_snapshots
    assert stats["file_snapshots"] == 1


def test_apply_clone_minimal_slice_on_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    mem = create_agent_memory(tmp_path / "m", "Draft", 200)
    u0 = Message.user_message("task")
    a0 = Message.assistant_message("old")
    t0 = Message.tool_message("x", "bash", "1")
    a1 = Message.assistant_message("new")
    t1 = Message.tool_message("y", "bash", "2")
    for m in (u0, a0, t0, a1, t1):
        mem.add_message(m)
    assert apply_clone_minimal_slice_to_memory(mem) is True
    out = [r.memory_record.message for r in mem.chat_history_memory.retrieve(window_size=None)]
    assert out == [u0, a1, t1]


def test_slice_clone_minimal_is_identity_when_no_assistant() -> None:
    u0 = Message.user_message("only")
    msgs = [u0]
    assert slice_clone_minimal_inherited_messages(msgs) is msgs


def test_apply_clone_inherit_compression_long_term_matches_storage_rows(tmp_path: Path) -> None:
    """Rewriting inherited memory must truncate long_term.jsonl (no duplicate append)."""
    mem_dir = tmp_path / "memlt"
    mem = create_agent_memory(mem_dir, "Draft", 200)
    long_bash = "[exit=0, 0.1s]\n" + "\n".join(f"L{i}" for i in range(40))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    function=Function(name="bash", arguments=json.dumps({"command": "true"})),
                ),
            ],
        ),
    )
    mem.add_message(Message.tool_message(long_bash, "bash", "tc1"))
    lt_path = mem_dir / "Draft" / "long_term.jsonl"
    n_before = sum(1 for ln in lt_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    n_storage = len(mem.chat_history_memory.retrieve(window_size=None))
    assert n_before == n_storage == 2
    stats = apply_clone_inherit_compression(mem, _cfg())
    assert stats["changed"] is True
    rows = [ln for ln in lt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_after_storage = len(mem.chat_history_memory.retrieve(window_size=None))
    assert len(rows) == n_after_storage == 2
    tcids = [
        json.loads(ln)["message"].get("tool_call_id")
        for ln in rows
        if json.loads(ln).get("message", {}).get("tool_call_id")
    ]
    assert len(tcids) == len(set(tcids))


def test_apply_clone_minimal_slice_long_term_matches_storage_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    mem_dir = tmp_path / "mslice"
    mem = create_agent_memory(mem_dir, "Draft", 200)
    u0 = Message.user_message("task")
    a0 = Message.assistant_message("old")
    t0 = Message.tool_message("x", "bash", "1")
    a1 = Message.assistant_message("new")
    t1 = Message.tool_message("y", "bash", "2")
    for m in (u0, a0, t0, a1, t1):
        mem.add_message(m)
    lt_path = mem_dir / "Draft" / "long_term.jsonl"
    assert sum(1 for ln in lt_path.read_text(encoding="utf-8").splitlines() if ln.strip()) == 5
    assert apply_clone_minimal_slice_to_memory(mem) is True
    rows = [ln for ln in lt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == len(mem.chat_history_memory.retrieve(window_size=None)) == 3


def test_compress_inherited_write_preserves_write_auto_snapshot() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    snap = (
        "File `solution.py` written successfully (1 lines, sha256~abc). OK.\n\n"
        "[auto-snapshot after successful write: solution.py]\n"
        "File is complete on disk.\n"
        "[solution.py: 1 lines total, sha256~abc, showing 1-1]\n"
        "     1|print(1)\n"
    )
    out = compress_inherited_tool_message_content(
        snap,
        tool_name="write",
        tool_args={"path": "solution.py"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
        write_snapshot_inherit_max_lines=80,
        write_snapshot_inherit_max_chars=5000,
    )
    assert "[auto-snapshot after successful write:" in out
    assert "print(1)" in out


def test_compress_inherited_trimmed_write_auto_snapshot_respects_cap() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    snapshot_lines = "\n".join(f"{i:>6}|line {i}" for i in range(1, 120))
    content = (
        "[Tool feedback trimmed for LLM context: 5000 chars, 120 lines; auto-snapshot preserved]\n"
        "File `solution.py` written successfully (119 lines, sha256~abc123). "
        "Syntax check passed.\n\n"
        "[auto-snapshot after successful write: solution.py]\n"
        "[current `solution.py` snapshot summary; deterministic tool-result memory]\n"
        "[tool-summary code-map: solution.py]\n- lines: 119\n- sha256:abc123\n"
        "[solution.py: 119 lines total, sha256~abc123, snapshot truncated for memory budget]\n"
        + snapshot_lines
    )
    out = compress_inherited_tool_message_content(
        content,
        tool_name="write",
        tool_args={"path": "solution.py"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
        write_snapshot_inherit_max_lines=8,
        write_snapshot_inherit_max_chars=700,
    )
    assert not out.startswith("[Tool feedback trimmed")
    assert "File `solution.py` written successfully (119 lines, sha256~abc123)" in out
    assert "[auto-snapshot after successful write:" in out
    assert "inherited snapshot truncated" in out or "[truncated]" in out
    assert len(out) < len(content)


def test_compress_inherited_trimmed_edit_auto_snapshot_respects_cap() -> None:
    patterns = _compile_clone_inherit_signal_patterns(None)
    snapshot_lines = "\n".join(f"{i:>6}|line {i}" for i in range(1, 120))
    content = (
        "[Tool feedback trimmed for LLM context: 5000 chars, 120 lines; auto-snapshot preserved]\n"
        "Edited solution.py: 1 replacement OK.\n"
        "(119 lines, sha256~abc123)\n\n"
        "[auto-snapshot after successful write: solution.py]\n"
        "[current `solution.py` snapshot summary; deterministic tool-result memory]\n"
        "[tool-summary code-map: solution.py]\n- lines: 119\n- sha256:abc123\n"
        "[solution.py: 119 lines total, sha256~abc123, snapshot truncated for memory budget]\n"
        + snapshot_lines
    )
    out = compress_inherited_tool_message_content(
        content,
        tool_name="edit",
        tool_args={"path": "solution.py"},
        read_max_lines=5,
        bash_tail_lines=3,
        omit_solution_py_reads=True,
        write_compact=True,
        edit_compact=True,
        patterns=patterns,
        write_snapshot_inherit_max_lines=8,
        write_snapshot_inherit_max_chars=700,
    )
    assert not out.startswith("[Tool feedback trimmed")
    assert "Edited solution.py: 1 replacement OK." in out
    assert "(119 lines, sha256~abc123)" in out
    assert "[auto-snapshot after successful write:" in out
    assert len(out) < len(content)


def test_preserve_initial_eda_compresses_duplicate_user_prose_only(tmp_path: Path) -> None:
    mem = create_agent_memory(tmp_path / "m_userdup", "Draft", 200)
    initial_user = (
        "Design a **gold** strong single pipeline for this node.\n\n"
        "Why this matters: this root draft policy is repeated by child fork prompts "
        "and should not be inherited verbatim.\n\n"
        "# Nomad2018\n\n"
        "## Task objective\n"
        "Predict transparent conductors from the provided competition files.\n"
    )
    mem.add_message(Message.user_message(initial_user))
    mem.add_message(Message.user_message("[fresh-workspace] parent unpacked a fresh copy here."))
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="eda_read",
                    function=Function(
                        name="read",
                        arguments=json.dumps({"path": "train.csv", "limit": 3}),
                    ),
                )
            ],
        )
    )
    eda_body = "[train.csv: 3 lines total, showing 1-3]\na,b\n1,2\n3,4"
    mem.add_message(Message.tool_message(eda_body, "read", "eda_read"))
    solution = "print('current solution')\n"
    mem.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="w",
                    function=Function(
                        name="write",
                        arguments=json.dumps({"path": "solution.py", "content": solution}),
                    ),
                )
            ],
        )
    )
    mem.add_message(Message.tool_message("Written solution.py (1 lines, 26 bytes, sha256~abc)", "write", "w"))

    stats = apply_clone_inherit_compression(mem, _cfg(), preserve_initial_eda=True)

    recs = [r.memory_record.message for r in mem.chat_history_memory.retrieve(window_size=None)]
    assert stats["user_msgs"] == 2
    assert "Inherited task brief (compressed)" in recs[0].content
    assert "Why this matters" not in recs[0].content
    assert "Predict transparent conductors" in recs[0].content
    assert "inherited setup hint omitted" in recs[1].content
    assert recs[3].content == eda_body
