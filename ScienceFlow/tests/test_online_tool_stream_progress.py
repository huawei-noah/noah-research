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

"""Unit tests for OnlineLLM tool-arg streaming progress (post-preview silence)."""

from __future__ import annotations

from deepcraft_core.llm.online import OnlineLLM


def test_tool_arg_stream_progress_empty_when_under_cap() -> None:
    slot: dict = {"name": "write", "arguments": "x" * 500}
    assert OnlineLLM._tool_arg_stream_progress_chunks(slot, 1200, 500) == []


def test_tool_arg_stream_progress_skips_non_write_edit() -> None:
    slot: dict = {"name": "bash", "arguments": "x" * 9000}
    assert OnlineLLM._tool_arg_stream_progress_chunks(slot, 4000, 9000) == []


def test_tool_arg_stream_progress_milestones_write() -> None:
    slot: dict = {"name": "write", "arguments": ""}
    cap = 1200
    step = 6144
    # First chunk crosses first milestone (cap + step = 7344)
    c1 = OnlineLLM._tool_arg_stream_progress_chunks(slot, cap, 8000, step=step)
    assert len(c1) == 1
    assert "8000" in c1[0]
    c2 = OnlineLLM._tool_arg_stream_progress_chunks(slot, cap, 15_000, step=step)
    assert len(c2) == 1
    assert "15000" in c2[0]


def test_tool_arg_stream_progress_multiple_in_one_jump() -> None:
    slot: dict = {"name": "edit", "arguments": ""}
    cap = 2500
    step = 1000
    total = cap + 3500
    lines = OnlineLLM._tool_arg_stream_progress_chunks(slot, cap, total, step=step)
    assert len(lines) == 3
