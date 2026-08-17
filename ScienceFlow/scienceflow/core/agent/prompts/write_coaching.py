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

"""Write nudges, failure coaching, and assistant-text code recovery."""

from __future__ import annotations

import ast
import re
from typing import Any

from deepcraft_core.tool import ToolResult

from scienceflow.core.agent.tools.bash_utils import _looks_like_write_placeholder_mimicry
from scienceflow.core.agent.shared.constants import (
    _UNFENCED_PYTHON_MAX_TAIL_TRIM,
    _UNFENCED_PYTHON_MIN_CHARS,
    _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD,
)


def _line_looks_like_python_start(line: str) -> bool:
    """First line of a module body (after prose): import/def/class or decorator."""
    ll = line.lstrip()
    if not ll or ll.startswith("#"):
        return False
    if ll.startswith("@") and len(ll) > 1 and (ll[1].isalpha() or ll[1] == "_"):
        return True
    return ll.startswith(("import ", "from ", "def ", "class "))


def _try_trim_tail_for_parse(lines_slice: list[str], min_chars: int) -> str | None:
    """Drop up to ``_UNFENCED_PYTHON_MAX_TAIL_TRIM`` trailing lines until ``ast.parse`` succeeds."""
    max_drop = min(len(lines_slice), _UNFENCED_PYTHON_MAX_TAIL_TRIM + 1)
    for drop in range(1, max_drop):
        candidate = "\n".join(lines_slice[:-drop])
        if len(candidate) < min_chars:
            return None
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return None


def _extract_python_code_from_assistant_text(text: str) -> str | None:
    """Best-effort extract a Python file body from assistant text.

    1) Fenced blocks (```python ... ``` or ``` ... ```).
    2) Unfenced: whole text parses as Python (models often emit raw code with no fences).
    3) Unfenced: trim prose above the first import/def/class/@ line and parse.
    """
    if not (text or "").strip():
        return None
    m = re.search(r"```(?:python|py)\s*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
        if len(code) >= 40:
            return code
    m2 = re.search(r"```\s*\r?\n(.*?)```", text, re.DOTALL)
    if m2:
        code = m2.group(1).strip()
        if len(code) >= 40 and (
            "def " in code or "import " in code or "class " in code
        ):
            return code

    stripped = text.strip()
    if len(stripped) >= _UNFENCED_PYTHON_MIN_CHARS:
        try:
            ast.parse(stripped)
            return stripped
        except SyntaxError:
            pass
        lines = stripped.splitlines()
        for i, line in enumerate(lines):
            if not _line_looks_like_python_start(line):
                continue
            tail = lines[i:]
            candidate = "\n".join(tail)
            if len(candidate) < _UNFENCED_PYTHON_MIN_CHARS:
                # Later python-start lines have strictly shorter tails; stop searching.
                break
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                trimmed = _try_trim_tail_for_parse(tail, _UNFENCED_PYTHON_MIN_CHARS)
                if trimmed is not None:
                    return trimmed
                continue
    return None


class WriteCoachingMixin:
    """Hints after writes and recovery from assistant prose."""

    def _maybe_append_solution_write_nudge(
        self,
        name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
        feedback: str,
    ) -> str:
        """Compatibility helper for tests and legacy paths."""
        if name == "bash":
            self._consecutive_solution_writes = 0
            return feedback

        if name != "write" or tool_result.error:
            return feedback

        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel.endswith("solution.py"):
            return feedback

        self._consecutive_write_syntax_fails = 0
        self._consecutive_solution_writes = int(
            getattr(self, "_consecutive_solution_writes", 0),
        ) + 1
        current_sha = self._sha256_of_solution()
        ini = getattr(self, "_initial_solution_sha", None)

        extra = ""
        if current_sha is not None and ini is not None and current_sha == ini:
            extra = (
                "\n\n[Guard] WARNING: solution.py is identical to the inherited version — "
                "you must modify it before testing. After your changes, verify with: "
                "`python3 solution.py` (or pass argparse knobs like `--epochs 1` for a faster smoke pass)."
            )

        self._log_info(
            "[write-nudge] consecutive_writes=%d",
            int(getattr(self, "_consecutive_solution_writes", 0)),
        )
        return feedback + extra

    def _maybe_append_write_failure_coaching(
        self,
        name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
        feedback: str,
    ) -> str:
        """Compatibility helper: append hard coaching after failed writes."""
        if name != "write" or not tool_result.error:
            return feedback
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel.endswith("solution.py"):
            return feedback

        err = str(tool_result.error or "")
        extra_parts: list[str] = []
        raw = args.get("content")
        if isinstance(raw, str) and _looks_like_write_placeholder_mimicry(raw):
            extra_parts.append(
                "[Guard] Your `write` **`content`** looks like a **placeholder or compressed "
                "memory snippet** (e.g. `<N chars>`, `<<<WRITE_PLACEHOLDER: …>>>`, legacy "
                "`[file content omitted - …]`, or `# <<<MEMORY_COMPRESSED:` / `# [MEMORY_COMPRESSED:`) "
                "— that is **not** a complete file. "
                "The `content` field must contain the **full source file text**; use **read** if needed."
            )
        if "Syntax check failed" in err:
            extra_parts.append(
                "[Guard] Next `write` to solution.py must pass a **complete, syntactically valid** "
                "Python file in the **`content`** field — do not resend a tiny placeholder. "
                "The runnable code must appear in the tool argument, not only in your assistant text."
            )
        if isinstance(raw, str) and len(raw) < _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD:
            extra_parts.append(
                "[Guard] The submitted `content` looks **too short** for a full solution — "
                "confirm you are passing the entire file body in `content`."
            )

        if not extra_parts:
            return feedback
        injected = "\n\n".join(extra_parts)
        self._log_info(
            "[write-failure-coaching] path=%s syntax=%s short_content_hint=%s "
            "placeholder_mimicry=%s appended_chars=%d",
            rel,
            "Syntax check failed" in err,
            isinstance(raw, str) and len(raw) < _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD,
            isinstance(raw, str) and _looks_like_write_placeholder_mimicry(raw),
            len(injected),
        )
        self._log_info("%s", "[write-failure-coaching-body]\n" + injected)
        return feedback + "\n\n" + injected

    def _solution_write_syntax_or_short_failure(
        self,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> bool:
        """Eligibility for write-recovery fallback (syntax / tiny payload)."""
        err = str(tool_result.error or "")
        if "Syntax check failed" in err:
            return True
        raw = args.get("content")
        if isinstance(raw, str) and len(raw) < _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD:
            return True
        return False

    async def _try_write_solution_from_assistant_text(
        self,
        args: dict[str, Any],
        assistant_text: str,
    ) -> ToolResult | None:
        """Re-run ``write`` using code extracted from assistant text (truncated tool args)."""
        extracted = _extract_python_code_from_assistant_text(assistant_text)
        if not extracted:
            self._log_info("[write-recover] no usable python block in assistant text")
            return None
        try:
            ast.parse(extracted)
        except SyntaxError:
            self._log_info("[write-recover] extracted code failed ast.parse, skipping")
            return None
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/") or "solution.py"
        self._log_info(
            "[write-recover] retrying write from assistant text chars=%d path=%s",
            len(extracted),
            rel,
        )
        tr = await self.availableTools.execute(
            name="write",
            tool_input={"path": rel, "content": extracted},
        )
        if not isinstance(tr, ToolResult):
            tr = ToolResult(output=str(tr))
        return tr
