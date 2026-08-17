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

"""Unit tests for ScienceAgent solution.py write nudges (quick-test + inherited-copy warning)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from deepcraft_core.tool import ToolResult


def _bare_agent(tmp_path: Path):
    from scienceflow.core.agent import ScienceAgent

    agent = ScienceAgent.__new__(ScienceAgent)
    agent._workspace_dir = tmp_path
    agent._consecutive_solution_writes = 0
    agent._initial_solution_sha = None
    agent._log_info = lambda *_a, **_k: None  # type: ignore[assignment]
    return agent


def test_bash_resets_consecutive_writes(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    agent._consecutive_solution_writes = 5
    out = agent._maybe_append_solution_write_nudge(
        "bash",
        {"command": "echo hi"},
        ToolResult(output="ok"),
        "bash out",
    )
    assert out == "bash out"
    assert agent._consecutive_solution_writes == 0


def test_write_non_solution_no_nudge(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    out = agent._maybe_append_solution_write_nudge(
        "write",
        {"path": "foo.py"},
        ToolResult(output="ok"),
        "w",
    )
    assert out == "w"
    assert agent._consecutive_solution_writes == 0


def test_write_solution_error_no_increment(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    out = agent._maybe_append_solution_write_nudge(
        "write",
        {"path": "solution.py"},
        ToolResult(output="err", error="write failed"),
        "e",
    )
    assert out == "e"
    assert agent._consecutive_solution_writes == 0


def test_identical_to_inherited_warning(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    content = b"parent code\n"
    (tmp_path / "solution.py").write_bytes(content)
    agent._initial_solution_sha = hashlib.sha256(content).hexdigest()
    out = agent._maybe_append_solution_write_nudge(
        "write",
        {"path": "solution.py"},
        ToolResult(output="ok"),
        "Written.",
    )
    assert "WARNING" in out
    assert "identical to the inherited version" in out
    assert agent._consecutive_solution_writes == 1


def test_first_write_different_from_initial_no_extra_nudge(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    (tmp_path / "solution.py").write_text("print(2)", encoding="utf-8")
    agent._initial_solution_sha = hashlib.sha256(b"print(1)").hexdigest()
    out = agent._maybe_append_solution_write_nudge(
        "write",
        {"path": "solution.py"},
        ToolResult(output="ok"),
        "ok",
    )
    assert out == "ok"
    assert "WARNING" not in out
    assert agent._consecutive_solution_writes == 1


def test_second_write_no_consecutive_nudge(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    (tmp_path / "solution.py").write_text("print(3)", encoding="utf-8")
    agent._initial_solution_sha = hashlib.sha256(b"print(0)").hexdigest()
    agent._consecutive_solution_writes = 1
    out = agent._maybe_append_solution_write_nudge(
        "write",
        {"path": "solution.py"},
        ToolResult(output="ok"),
        "ok",
    )
    assert out == "ok"
    assert agent._consecutive_solution_writes == 2


def test_write_failure_syntax_coaching_appended(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    err = "Syntax check failed: invalid syntax (<unknown>, line 1)"
    fb = agent._maybe_append_write_failure_coaching(
        "write",
        {"path": "solution.py", "content": "def oops("},
        ToolResult(output="", error=err),
        "Tool failed.",
    )
    assert "Tool failed." in fb
    assert "content" in fb.lower()
    assert "complete" in fb.lower() or "syntactically" in fb.lower()
    assert "assistant" in fb.lower()


def test_write_failure_no_coaching_on_success(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    fb = agent._maybe_append_write_failure_coaching(
        "write",
        {"path": "solution.py", "content": "print(1)"},
        ToolResult(output="ok"),
        "ok",
    )
    assert fb == "ok"


def test_write_failure_short_content_hint_without_syntax_phrase(tmp_path: Path) -> None:
    agent = _bare_agent(tmp_path)
    fb = agent._maybe_append_write_failure_coaching(
        "write",
        {"path": "solution.py", "content": "x"},
        ToolResult(output="", error="disk full"),
        "e",
    )
    assert "too short" in fb.lower()


def test_extract_unfenced_with_trailing_prose() -> None:
    """Code followed by natural-language explanation should still be extracted."""
    from scienceflow.core.agent.prompts.write_coaching import (
        _extract_python_code_from_assistant_text,
    )

    pad = "x\n" * 100
    body = "    print('hello')\n" * 30
    text = (
        pad
        + "import os\n"
        + "def main():\n"
        + body
        + "\nThis code prints hello."
    )
    result = _extract_python_code_from_assistant_text(text)
    assert result is not None
    ast.parse(result)
    assert "import os" in result
    assert "def main" in result


def test_extract_unfenced_multiple_candidates() -> None:
    """If first candidate fails, try the next import/def start line."""
    from scienceflow.core.agent.prompts.write_coaching import (
        _extract_python_code_from_assistant_text,
    )

    text = (
        ("Some prose.\n" * 20)
        + "import broken(\n"
        + "def real_func():\n"
        + "    return 42\n" * 30
    )
    result = _extract_python_code_from_assistant_text(text)
    assert result is not None
    assert "real_func" in result
    assert "broken" not in result
    ast.parse(result)


# ---------------------------------------------------------------------------
# WriteNudgeGuard (tool_guards.py) — streak nudge and bash reset
# ---------------------------------------------------------------------------

def _make_write_nudge_guard(tmp_path: Path, initial_sha: str | None = None):
    """Build a WriteNudgeGuard with a tmp workspace."""
    from scienceflow.core.agent.tools.tool_guards import WriteNudgeGuard

    def current_sha():
        p = tmp_path / "solution.py"
        if not p.exists():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def initial():
        return initial_sha

    return WriteNudgeGuard(
        get_current_solution_sha=current_sha,
        get_initial_solution_sha=initial,
    )


def test_write_nudge_guard_bash_resets_streak(tmp_path: Path) -> None:
    from deepcraft_core.tool import ToolResult

    guard = _make_write_nudge_guard(tmp_path)
    guard._consecutive_solution_writes = 5
    guard.on_tool_result("bash", {"command": "python3 solution.py"}, ToolResult(output="ok"))
    assert guard._consecutive_solution_writes == 0


def test_write_nudge_guard_no_nudge_before_threshold(tmp_path: Path) -> None:
    from deepcraft_core.tool import ToolResult
    from scienceflow.core.agent.shared.constants import _WRITE_STREAK_NUDGE_THRESHOLD

    (tmp_path / "solution.py").write_text("print(1)", encoding="utf-8")
    guard = _make_write_nudge_guard(tmp_path, initial_sha=hashlib.sha256(b"other").hexdigest())
    for _ in range(_WRITE_STREAK_NUDGE_THRESHOLD - 1):
        result = guard.on_tool_result(
            "write", {"path": "solution.py"}, ToolResult(output="ok")
        )
        assert result is None or "times in a row" not in (result or "")


def test_write_nudge_guard_streak_nudge_at_threshold(tmp_path: Path) -> None:
    from deepcraft_core.tool import ToolResult
    from scienceflow.core.agent.shared.constants import _WRITE_STREAK_NUDGE_THRESHOLD

    (tmp_path / "solution.py").write_text("print(42)", encoding="utf-8")
    guard = _make_write_nudge_guard(tmp_path, initial_sha=hashlib.sha256(b"other").hexdigest())
    guard._consecutive_solution_writes = _WRITE_STREAK_NUDGE_THRESHOLD - 1
    result = guard.on_tool_result(
        "write", {"path": "solution.py"}, ToolResult(output="ok")
    )
    assert result is not None
    assert "times in a row" in result
    assert "python3 solution.py" in result
    assert "sha256~" in result
    assert "argparse" in result or "epochs" in result


def test_write_nudge_guard_streak_resets_after_bash(tmp_path: Path) -> None:
    from deepcraft_core.tool import ToolResult
    from scienceflow.core.agent.shared.constants import _WRITE_STREAK_NUDGE_THRESHOLD

    (tmp_path / "solution.py").write_text("print(0)", encoding="utf-8")
    guard = _make_write_nudge_guard(tmp_path, initial_sha=hashlib.sha256(b"other").hexdigest())
    guard._consecutive_solution_writes = _WRITE_STREAK_NUDGE_THRESHOLD
    # bash resets counter
    guard.on_tool_result("bash", {"command": "python3 solution.py"}, ToolResult(output="ok"))
    assert guard._consecutive_solution_writes == 0
    # next write is below threshold — no streak nudge
    result = guard.on_tool_result(
        "write", {"path": "solution.py"}, ToolResult(output="ok")
    )
    assert result is None or "times in a row" not in (result or "")
