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

"""Tests for write/edit full snapshot feedback and stale snapshot collapse in memory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from deepcraft_core.tool import ToolResult

from scienceflow.core.agent.memory.memory_utils import (
    collapse_stale_file_snapshots_before_add,
    extract_path_from_tool_feedback_text,
    maybe_collapse_stale_snapshots,
)
from scienceflow.core.tools.edit_tool import EditTool
from scienceflow.core.tools.write_tool import WriteTool, format_file_snapshot_for_tool_return
from scienceflow.core.mem.memory_context import compress_edit_success_output_for_memory


@pytest.mark.asyncio
async def test_write_rejects_log_placeholder_content(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    r = await w.execute(
        path="solution.py",
        content="[file content omitted - 51 chars written to disk]",
    )
    assert r.error is not None
    assert "placeholder" in (r.error or "").lower()
    assert "Syntax check failed" not in (r.error or "")
    assert not (tmp_path / "solution.py").is_file()


@pytest.mark.asyncio
async def test_write_rejects_new_angle_write_placeholder_content(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    r = await w.execute(
        path="solution.py",
        content=(
            "<<<WRITE_PLACEHOLDER: 51 chars on disk — NOT real file content — use read tool>>>"
        ),
    )
    assert r.error is not None
    assert "placeholder" in (r.error or "").lower()
    assert not (tmp_path / "solution.py").is_file()


@pytest.mark.asyncio
async def test_write_rejects_legacy_angle_placeholder_content(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    r = await w.execute(path="x.py", content="<1234 chars>")
    assert r.error is not None
    assert "placeholder" in (r.error or "").lower()
    assert not (tmp_path / "x.py").is_file()


@pytest.mark.asyncio
async def test_write_rejects_memory_compressed_snippet(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    blob = (
        "x" * 60
        + "\n\n# [MEMORY_COMPRESSED: full file is 500 bytes; real content on disk]\n\n"
        + "x" * 60
    )
    r = await w.execute(path="solution.py", content=blob)
    assert r.error is not None
    assert "placeholder" in (r.error or "").lower() or "compressed" in (r.error or "").lower()
    assert not (tmp_path / "solution.py").is_file()


@pytest.mark.asyncio
async def test_write_rejects_memory_compressed_snippet_v2_marker(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    blob = (
        "x" * 60
        + "\n\n# <<<MEMORY_COMPRESSED: prior successful write, 500 bytes already on disk — "
        "do NOT rewrite to 'fix' this; use read if you need the full current content>>>\n\n"
        + "x" * 60
    )
    r = await w.execute(path="solution.py", content=blob)
    assert r.error is not None
    assert "placeholder" in (r.error or "").lower() or "compressed" in (r.error or "").lower()
    assert not (tmp_path / "solution.py").is_file()


@pytest.mark.asyncio
async def test_write_rejects_write_ok_marker_snippet(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    blob = (
        "x" * 60
        + "\n\n# <<<WRITE_OK: 500 bytes written and syntax-validated — "
        "file is complete on disk. No read needed.>>>\n\n"
        + "x" * 60
    )
    r = await w.execute(path="solution.py", content=blob)
    assert r.error is not None
    assert "placeholder" in (r.error or "").lower() or "compressed" in (r.error or "").lower()
    assert not (tmp_path / "solution.py").is_file()


@pytest.mark.asyncio
async def test_write_idempotent_same_content_is_noop(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    body = "a = 1\n" * 20
    r1 = await w.execute(path="n.py", content=body)
    assert r1.error is None
    assert "Written" in (r1.output or "")
    r2 = await w.execute(path="n.py", content=body)
    assert r2.error is None
    assert "No-op:" in (r2.output or "")
    assert "sha256~" in (r2.output or "")


@pytest.mark.asyncio
async def test_write_small_returns_full_numbered(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    body = "\n".join(f"# L{i}" for i in range(50)) + "\n"
    r = await w.execute(path="s.py", content=body)
    assert r.error is None
    out = r.output or ""
    assert "     1|# L0" in out
    assert "    50|# L49" in out
    assert "omitted" not in out


def test_format_snapshot_large_uses_head_tail() -> None:
    lines = [f"x={i}" for i in range(260)]
    text = "\n".join(lines) + "\n"
    snap = format_file_snapshot_for_tool_return(
        text,
        max_chars=12000,
        max_lines=250,
        head_tail_lines=40,
    )
    assert "lines omitted" in snap
    assert "     1|x=0" in snap
    assert "   260|x=259" in snap


@pytest.mark.asyncio
async def test_edit_success_includes_file_snapshot_block(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    await w.execute(path="e.py", content="a = 1\nb = 2\nc = 3\n")
    ed = EditTool(workspace_dir=tmp_path)
    r = await ed.execute(path="e.py", old_str="b = 2", new_str="b = 99")
    assert r.error is None
    out = r.output or ""
    assert "--- Current file snapshot (after edit) ---" in out
    assert "     2|b = 99" in out


@pytest.mark.asyncio
async def test_edit_not_found_includes_top_candidates(tmp_path) -> None:
    w = WriteTool(workspace_dir=tmp_path)
    await w.execute(path="f.py", content="line1\nline2\nline3\n")
    ed = EditTool(workspace_dir=tmp_path, failure_top_k_candidates=3)
    r = await ed.execute(path="f.py", old_str="nope", new_str="x")
    assert r.error is not None
    assert "Candidate 1" in (r.error or "")
    assert "No close line match" in (r.error or "") or "Closest line match" in (r.error or "")


def test_compress_edit_drops_raw_snapshot_block() -> None:
    obs = (
        "Edited x.py: 1 replacement OK.\nctx\n(3 lines, sha256~abc)\n"
        "--- Current file snapshot (after edit) ---\n     1|a\n"
    )
    out = compress_edit_success_output_for_memory(obs)
    assert out == "Edited x.py: 1 replacement OK.\n(3 lines, sha256~abc)"
    assert "--- Current file snapshot" not in out
    assert "     1|a" not in out


def test_extract_path_from_feedback() -> None:
    t = "Written foo/bar.py (3 lines, 10 bytes, sha256~ab)\n     1|x"
    assert extract_path_from_tool_feedback_text(t, "write") == "foo/bar.py"
    t1b = "File `foo/bar.py` written successfully (3 lines, sha256~ab). More text.\n[auto"
    assert extract_path_from_tool_feedback_text(t1b, "write") == "foo/bar.py"
    t2 = "Edited baz.py: 1 replacement OK.\n"
    assert extract_path_from_tool_feedback_text(t2, "edit") == "baz.py"


def test_collapse_stale_snapshot_same_file() -> None:
    old_full = (
        "Written a.py (2 lines, 20 bytes, sha256~dead)\n"
        "     1|x\n"
        "     2|y"
    )
    memory_list = [
        {
            "message": {
                "role": "tool",
                "name": "write",
                "content": old_full,
            },
        },
    ]
    mem = SimpleNamespace(
        chat_history_memory=SimpleNamespace(storage=SimpleNamespace(memory_list=memory_list)),
    )
    collapse_stale_file_snapshots_before_add(
        mem,
        tool_name="write",
        rel_path="a.py",
        enabled=True,
    )
    c = memory_list[0]["message"]["content"]
    assert "superseded by a later write/edit" in c
    assert "\n     1|" not in c


def test_maybe_collapse_skips_on_error(tmp_path) -> None:
    memory_list: list = []
    mem = SimpleNamespace(
        chat_history_memory=SimpleNamespace(storage=SimpleNamespace(memory_list=memory_list)),
    )
    tr = ToolResult(error="fail")
    maybe_collapse_stale_snapshots(
        mem,
        "write",
        {"path": "a.py"},
        tr,
        enabled=True,
    )
    assert memory_list == []


def test_collapse_different_paths_noop(tmp_path) -> None:
    memory_list = [
        {
            "message": {
                "role": "tool",
                "name": "write",
                "content": "Written b.py (1 lines, 5 bytes, sha256~x)\n     1|z",
            },
        },
    ]
    mem = SimpleNamespace(
        chat_history_memory=SimpleNamespace(storage=SimpleNamespace(memory_list=memory_list)),
    )
    collapse_stale_file_snapshots_before_add(
        mem,
        tool_name="write",
        rel_path="a.py",
        enabled=True,
    )
    assert "     1|z" in memory_list[0]["message"]["content"]
