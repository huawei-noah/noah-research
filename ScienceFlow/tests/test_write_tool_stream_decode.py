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

"""Regression tests for write-tool JSON content streaming (online.py)."""

from __future__ import annotations

import asyncio
import json

import pytest

from deepcraft_core.llm.base import StreamHandle
from deepcraft_core.llm.online import _WriteToolArgStreamDecoder


async def _drain_handle(h: StreamHandle) -> str:
    parts: list[str] = []
    while True:
        c = await h.queue.get()
        if c is None:
            break
        parts.append(c)
    return "".join(parts)


@pytest.mark.asyncio
async def test_write_decoder_decodes_content_and_path_header() -> None:
    inner = "line1\n\"quoted\"\\" + "\u0041" + "end"
    args = json.dumps({"path": "x.py", "content": inner}, ensure_ascii=False)
    h = StreamHandle()
    cons = asyncio.create_task(_drain_handle(h))
    slot: dict = {"name": "write", "arguments": ""}
    for i in range(0, len(args), 7):
        slot["arguments"] = args[: i + 7]
        await _WriteToolArgStreamDecoder.feed(slot, h)
    h.finish()
    out = await cons
    assert "[write -> x.py]" in out
    assert "line1\n" in out
    assert '"quoted"' in out
    assert "Aend" in out


@pytest.mark.asyncio
async def test_write_decoder_survives_split_unicode_escape() -> None:
    inner = "a\u1234b"
    args = json.dumps({"path": "p", "content": inner}, ensure_ascii=True)
    assert "\\u1234" in args
    h = StreamHandle()
    cons = asyncio.create_task(_drain_handle(h))
    slot: dict = {"name": "write", "arguments": ""}
    # Split inside \u1234 (after \u)
    cut = args.index("\\u")
    slot["arguments"] = args[: cut + 2]
    await _WriteToolArgStreamDecoder.feed(slot, h)
    slot["arguments"] = args
    await _WriteToolArgStreamDecoder.feed(slot, h)
    h.finish()
    out = await cons
    assert "\u1234" in out
