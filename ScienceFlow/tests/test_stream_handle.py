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

"""Tests for StreamHandle and the REPL interrupt/truncation mechanism."""

import asyncio
import pytest

from deepcraft_core.llm.base import StreamHandle


# ---------------------------------------------------------------
# StreamHandle unit tests
# ---------------------------------------------------------------


class TestStreamHandleBasic:
    """Core lifecycle: put, finish, consume."""

    @pytest.mark.asyncio
    async def test_normal_stream(self):
        h = StreamHandle()
        await h.put("Hello ")
        await h.put("world")
        h.finish()

        chunks = []
        while True:
            c = await h.queue.get()
            if c is None:
                break
            chunks.append(c)
        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_interrupted_property_initially_false(self):
        h = StreamHandle()
        assert h.interrupted is False

    @pytest.mark.asyncio
    async def test_finish_sends_none_sentinel(self):
        h = StreamHandle()
        h.finish()
        val = await asyncio.wait_for(h.queue.get(), timeout=1.0)
        assert val is None

    @pytest.mark.asyncio
    async def test_finish_is_noop_after_interrupt(self):
        h = StreamHandle()
        h.interrupt()
        sentinel_1 = await h.queue.get()
        assert sentinel_1 is None
        assert h.queue.empty(), "finish() after interrupt should not add another None"


class TestStreamHandleInterrupt:
    """Interrupt semantics: idempotent, clears queue, marks interrupted."""

    @pytest.mark.asyncio
    async def test_interrupt_sets_flag(self):
        h = StreamHandle()
        h.interrupt()
        assert h.interrupted is True

    @pytest.mark.asyncio
    async def test_interrupt_clears_pending_chunks(self):
        h = StreamHandle()
        await h.put("a")
        await h.put("b")
        assert h.queue.qsize() == 2
        h.interrupt()
        val = await asyncio.wait_for(h.queue.get(), timeout=1.0)
        assert val is None
        assert h.queue.empty()

    @pytest.mark.asyncio
    async def test_interrupt_is_idempotent(self):
        h = StreamHandle()
        h.interrupt()
        h.interrupt()
        assert h.interrupted is True
        val = await h.queue.get()
        assert val is None
        assert h.queue.empty()

    @pytest.mark.asyncio
    async def test_put_after_interrupt_is_noop(self):
        h = StreamHandle()
        h.interrupt()
        _ = await h.queue.get()
        await h.put("late chunk")
        assert h.queue.empty()


class TestStreamHandleStop:
    """stop(): end stream without draining queued previews (ScienceAgent tool stream)."""

    @pytest.mark.asyncio
    async def test_stop_preserves_pending_chunks(self):
        h = StreamHandle()
        await h.put("a")
        await h.put("b")
        assert h.queue.qsize() == 2
        h.stop()
        assert h.interrupted is True
        val1 = await asyncio.wait_for(h.queue.get(), timeout=1.0)
        val2 = await asyncio.wait_for(h.queue.get(), timeout=1.0)
        val3 = await asyncio.wait_for(h.queue.get(), timeout=1.0)
        assert val1 == "a"
        assert val2 == "b"
        assert val3 is None
        assert h.queue.empty()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        h = StreamHandle()
        h.stop()
        h.stop()
        assert h.interrupted is True
        val = await h.queue.get()
        assert val is None
        assert h.queue.empty()


class TestStreamHandleIsolation:
    """Multiple handles must be independent."""

    @pytest.mark.asyncio
    async def test_two_handles_independent(self):
        h1 = StreamHandle()
        h2 = StreamHandle()

        await h1.put("from h1")
        await h2.put("from h2")

        h1.interrupt()
        assert h1.interrupted is True
        assert h2.interrupted is False

        val = await h2.queue.get()
        assert val == "from h2"


# ---------------------------------------------------------------
# Simulated REPL interrupt flow
# ---------------------------------------------------------------

_BLOCK_RE_PATTERN = r"```(?:bash|python)\n(.*?)```"


class TestREPLInterruptFlow:
    """Simulate the producer/consumer pattern used in _ask_llm + _drain_and_execute."""

    @pytest.mark.asyncio
    async def test_single_action_interrupt_truncates(self):
        """Producer pushes chunks forming two code blocks.
        Consumer interrupts after detecting the first block."""
        import re

        handle = StreamHandle()
        full_text = (
            "Here is step 1:\n"
            "```bash\nls -la\n```\n"
            "Here is step 2:\n"
            "```bash\necho hello\n```\n"
        )

        async def producer():
            for ch in full_text:
                if handle.interrupted:
                    break
                await handle.put(ch)
                await asyncio.sleep(0)
            handle.finish()

        consumed = []
        truncate_pos = None

        async def consumer():
            nonlocal truncate_pos
            buf = ""
            detected = 0
            while True:
                chunk = await handle.queue.get()
                if chunk is None:
                    break
                buf += chunk
                consumed.append(chunk)
                blocks = list(re.finditer(r"```(?:bash|python)\n(.*?)```", buf, re.DOTALL))
                if len(blocks) > detected:
                    detected += 1
                    truncate_pos = blocks[-1].end()
                    handle.interrupt()

        await asyncio.gather(producer(), consumer())

        result = "".join(consumed)
        if truncate_pos is not None:
            result = result[:truncate_pos]

        assert "ls -la" in result
        assert "echo hello" not in result
        assert handle.interrupted is True

    @pytest.mark.asyncio
    async def test_no_interrupt_without_single_action_mode(self):
        """When not in single-action mode, all blocks should pass through."""
        handle = StreamHandle()
        full_text = "```bash\nls\n```\n```bash\necho hi\n```\n"

        async def producer():
            for ch in full_text:
                await handle.put(ch)
                await asyncio.sleep(0)
            handle.finish()

        buf = ""

        async def consumer():
            nonlocal buf
            while True:
                chunk = await handle.queue.get()
                if chunk is None:
                    break
                buf += chunk

        await asyncio.gather(producer(), consumer())
        assert "ls" in buf
        assert "echo hi" in buf
        assert handle.interrupted is False


# ---------------------------------------------------------------
# Concurrent handles (simulating parallel LLM calls)
# ---------------------------------------------------------------


class TestConcurrentHandles:

    @pytest.mark.asyncio
    async def test_parallel_streams_no_interference(self):
        """Two parallel 'ask()' calls with separate handles don't cross-talk."""
        h1 = StreamHandle()
        h2 = StreamHandle()

        async def stream(h: StreamHandle, prefix: str, n: int):
            for i in range(n):
                await h.put(f"{prefix}-{i} ")
                await asyncio.sleep(0)
            h.finish()

        async def collect(h: StreamHandle) -> str:
            parts = []
            while True:
                c = await h.queue.get()
                if c is None:
                    break
                parts.append(c)
            return "".join(parts)

        await asyncio.gather(stream(h1, "A", 5), stream(h2, "B", 5))
        r1, r2 = await asyncio.gather(collect(h1), collect(h2))

        assert all(f"A-{i}" in r1 for i in range(5))
        assert all(f"B-{i}" in r2 for i in range(5))
        assert "B-" not in r1
        assert "A-" not in r2

    @pytest.mark.asyncio
    async def test_interrupt_one_does_not_affect_other(self):
        h1 = StreamHandle()
        h2 = StreamHandle()

        await h1.put("x")
        await h2.put("y")
        h1.interrupt()

        assert h1.interrupted is True
        assert h2.interrupted is False
        val = await h2.queue.get()
        assert val == "y"
