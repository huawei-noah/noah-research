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

"""Sliding-window priority: prefer errors / read output over successful bash tails."""

from __future__ import annotations

from deepcraft_core import Message

from scienceflow.core.mem.memory_context import (
    _best_suffix_for_budget_priority,
    _message_priority_score,
)


def test_message_priority_error_above_success_bash() -> None:
    ok = Message.tool_message("[exit=0, 1.0s]\nok\n", "bash", "a")
    bad = Message.tool_message("[exit=1, 1.0s]\nTraceback\n", "bash", "b")
    assert _message_priority_score(bad) > _message_priority_score(ok)


def test_message_priority_read_above_edit_success() -> None:
    read_m = Message.tool_message(
        "[x.csv: 3 lines total, showing 1-3]\na,b",
        "read",
        "r",
    )
    ed = Message.tool_message(
        "Edited solution.py — 1 replacement OK.\n(lines, sha256~ab)",
        "edit",
        "e",
    )
    assert _message_priority_score(read_m) > _message_priority_score(ed)


def test_best_suffix_prefers_recent_high_priority_under_tight_budget() -> None:
    """Two tool messages: long ok-bash (old) + short error (new). Budget fits only one → keep error."""
    old = Message.tool_message(
        "[exit=0, 1s]\n" + ("line\n" * 40),
        "bash",
        "o",
    )
    new = Message.tool_message("[exit=1, 0.1s]\nfail", "bash", "n")
    rest = [old, new]
    budget = 80
    out, _ = _best_suffix_for_budget_priority(rest, budget)
    assert len(out) == 1
    assert "exit=1" in (out[0].content or "")


def test_best_suffix_full_list_when_fits() -> None:
    a = Message.tool_message("a", "t", "1")
    b = Message.tool_message("b", "t", "2")
    rest = [a, b]
    out, _ = _best_suffix_for_budget_priority(rest, 1000)
    assert len(out) == 2


# ── New tests for monotone-k and constant placeholder ──────────────────────


def test_best_suffix_min_k_enforced() -> None:
    """min_k=2 must never return a suffix starting earlier than index 2."""
    msgs = [Message.tool_message(f"msg{i}", "bash", str(i)) for i in range(6)]
    # Large budget: without min_k would choose k=0 (full list).
    out, k = _best_suffix_for_budget_priority(msgs, 100_000, min_k=2)
    assert k >= 2, f"expected k >= 2 but got k={k}"
    assert out == msgs[k:]


def test_best_suffix_k_monotone_across_two_calls() -> None:
    """Simulates two consecutive build calls; k must not decrease."""
    msgs = [Message.tool_message(f"x{i}" * 20, "bash", str(i)) for i in range(8)]
    # First call: tight budget forces k=4
    out1, k1 = _best_suffix_for_budget_priority(msgs, 160)
    # Second call: same budget but pass previous k as min_k
    out2, k2 = _best_suffix_for_budget_priority(msgs, 160, min_k=k1)
    assert k2 >= k1, f"k regressed: {k2} < {k1}"


def test_best_suffix_min_k_gt_len_resets_to_zero() -> None:
    """If min_k > len(rest) (e.g. after agent.compact()), function must not crash."""
    msgs = [Message.tool_message("x", "bash", "1")]
    out, k = _best_suffix_for_budget_priority(msgs, 1000, min_k=999)
    # min_k safely clamped, whole list returned
    assert k <= len(msgs)
    assert len(out) == len(msgs)


def test_omitted_placeholder_is_constant_text() -> None:
    """The omitted-messages placeholder injected into LLM context must not contain
    a changing count — otherwise prefix cache always misses."""
    import pathlib
    import re
    import tempfile

    from deepcraft_core import Memory
    from scienceflow.core.mem.memory_context import MemoryContextManager

    with tempfile.TemporaryDirectory() as tmp:
        ws = pathlib.Path(tmp)
        mem = Memory()
        # Use a very tight budget so earlier messages are guaranteed to be omitted.
        ctx = MemoryContextManager(mem, ws, budget_chars=200)
        for i in range(20):
            ctx._memory.add_message(Message.user_message(f"user-message-number-{i:03d}: " + "x" * 30))
            ctx._memory.add_message(Message.tool_message(f"tool-result-{i:03d}: " + "y" * 30, "bash", str(i)))

        msgs1, omitted1 = ctx.build_messages_for_llm_with_stats()
        assert omitted1 > 0, "test requires omitted messages"

        placeholder_texts = [
            m.content or ""
            for m in msgs1
            if "[Context budget:" in (m.content or "")
        ]
        assert placeholder_texts, "placeholder not found in result"
        placeholder = placeholder_texts[0]
        # The placeholder must NOT contain any digit representing omitted count.
        assert not re.search(r"\d+ earlier message", placeholder), (
            f"placeholder contains changing count: {placeholder!r}"
        )


def test_leading_tool_cleanup_does_not_count_as_budget_omission(tmp_path) -> None:
    """Dropping leading tool records after compact is format cleanup, not budget loss."""
    from deepcraft_core import Memory
    from scienceflow.core.mem.memory_context import MemoryContextManager

    mem = Memory(max_messages=20)
    mem.add_message(Message.tool_message("old ls output", "ls", "tool-1"))
    mem.add_message(Message.tool_message("old glob output", "glob", "tool-2"))
    mem.add_message(Message.user_message("[Compacted conversation summary] current plan fits"))

    ctx = MemoryContextManager(mem, tmp_path, budget_chars=10_000)
    msgs, omitted = ctx.build_messages_for_llm_with_stats()

    assert omitted == 0
    assert msgs
    assert msgs[0].role != "tool"
    assert not any("[Context budget:" in (m.content or "") for m in msgs)


def test_leading_user_prefix_survives_sliding_window(tmp_path) -> None:
    """REPL/LNR first user turns must stay byte-stable when the tail overflows."""
    from deepcraft_core import Memory
    from scienceflow.core.mem.memory_context import MemoryContextManager

    mem = Memory(max_messages=200)
    mem.add_message(Message.user_message("FIRST USER QUERY: keep me verbatim"))
    mem.add_message(Message.user_message("[Compacted conversation summary]\nkeep this slot too"))
    for i in range(24):
        mem.add_message(Message.assistant_message(f"assistant-{i}: " + "x" * 80))
        mem.add_message(Message.user_message(f"dynamic-tail-{i}: " + "y" * 80))

    ctx = MemoryContextManager(mem, tmp_path, budget_chars=500)
    ctx.pin_message(Message.user_message("## Task description\n\nfixed task"))

    msgs, omitted = ctx.build_messages_for_llm_with_stats()
    contents = [m.content or "" for m in msgs]

    assert omitted > 0
    assert contents[0] == "## Task description\n\nfixed task"
    assert contents[1] == "FIRST USER QUERY: keep me verbatim"
    assert contents[2] == "[Compacted conversation summary]\nkeep this slot too"
    assert contents.count("FIRST USER QUERY: keep me verbatim") == 1
    assert any("[Context budget:" in c for c in contents)
