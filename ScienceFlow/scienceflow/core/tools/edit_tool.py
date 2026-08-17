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

"""Exact-match search/replace with uniqueness and post-edit syntax check."""

from __future__ import annotations

import ast
from difflib import SequenceMatcher
from pathlib import Path

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.tools.file_utils import PathGuard, atomic_write, validate_syntax
from scienceflow.core.tools.write_tool import (
    DEFAULT_RETURN_FULL_MAX_CHARS,
    DEFAULT_RETURN_FULL_MAX_LINES,
    DEFAULT_RETURN_HEAD_TAIL_LINES,
    format_file_snapshot_for_tool_return,
)


class EditTool(BaseTool):
    name: str = "edit"
    description: str = (
        "Replace exactly one occurrence of old_str with new_str in a workspace file. "
        "old_str must match the file byte-for-byte once; include enough context to be unique."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace."},
            "old_str": {
                "type": "string",
                "description": "Exact text to find (must appear exactly once).",
            },
            "new_str": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_str", "new_str"],
    }
    workspace_dir: Path = Field(...)
    sandbox: bool = Field(default=True)
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    return_full_max_chars: int = Field(default=DEFAULT_RETURN_FULL_MAX_CHARS)
    return_full_max_lines: int = Field(default=DEFAULT_RETURN_FULL_MAX_LINES)
    return_head_tail_lines: int = Field(default=DEFAULT_RETURN_HEAD_TAIL_LINES)
    return_change_ctx_lines: int = Field(default=8)
    failure_top_k_candidates: int = Field(default=3)
    failure_diag_max_chars: int = Field(default=2000)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _diagnose_not_found(content: str, old_str: str) -> str:
        first_line = old_str.strip().split("\n")[0].strip()
        if not first_line:
            return "old_str is empty or whitespace-only."
        best_line, best_ratio, best_no = "", 0.0, 0
        for i, line in enumerate(content.splitlines(), 1):
            ratio = SequenceMatcher(None, line.strip(), first_line).ratio()
            if ratio > best_ratio:
                best_ratio, best_line, best_no = ratio, line.rstrip(), i
        if best_ratio > 0.5:
            return (
                f"Closest line match (line {best_no}, {best_ratio:.0%} similar): {best_line!r}"
            )
        return "No close line match; re-read the file and copy the exact span."

    @staticmethod
    def _top_k_line_block_matches(
        content: str,
        old_str: str,
        *,
        k: int,
    ) -> list[tuple[float, int, str]]:
        """Sliding windows with same line count as *old_str*; score by ratio to full old_str."""
        old_lines = old_str.splitlines()
        if not old_lines:
            return []
        n = len(old_lines)
        file_lines = content.splitlines()
        if len(file_lines) < n:
            return []
        candidates: list[tuple[float, int, str]] = []
        for i in range(0, len(file_lines) - n + 1):
            block = "\n".join(file_lines[i : i + n])
            ratio = SequenceMatcher(None, old_str, block).ratio()
            candidates.append((ratio, i + 1, block))
        candidates.sort(key=lambda x: -x[0])
        return candidates[: max(1, k)]

    @staticmethod
    def _context_around_lines(
        file_lines: list[str],
        start_1based: int,
        *,
        n_lines: int,
        ctx: int,
    ) -> str:
        """start_1based: first line of the n_lines block (1-based)."""
        i0 = start_1based - 1
        lo = max(0, i0 - ctx)
        hi = min(len(file_lines), i0 + n_lines + ctx)
        out: list[str] = [
            f"(lines {i0 + 1}-{i0 + n_lines} context; lines {lo + 1}-{hi})"
        ]
        for j in range(lo, hi):
            prefix = ">> " if i0 <= j < i0 + n_lines else "   "
            out.append(f"{prefix}{j + 1:6}|{file_lines[j]}")
        return "\n".join(out)

    @staticmethod
    def _python_symbol_for_line(content: str, line_no: int) -> str:
        try:
            tree = ast.parse(content or "\n")
        except SyntaxError:
            return ""
        best: ast.AST | None = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            if start <= line_no <= end:
                if best is None:
                    best = node
                    continue
                best_start = int(getattr(best, "lineno", 0) or 0)
                best_end = int(getattr(best, "end_lineno", best_start) or best_start)
                if (end - start) <= (best_end - best_start):
                    best = node
        if best is None:
            return ""
        start = int(getattr(best, "lineno", 0) or 0)
        end = int(getattr(best, "end_lineno", start) or start)
        kind = "class" if isinstance(best, ast.ClassDef) else "function"
        name = getattr(best, "name", "(unknown)")
        return f"{kind} {name} lines {start}-{end}"

    @staticmethod
    def _suggest_old_str_anchor(
        content: str,
        file_lines: list[str],
        start_1based: int,
        *,
        n_lines: int,
        ctx: int,
        path: str,
    ) -> str:
        i0 = start_1based - 1
        lo = max(0, i0 - ctx)
        hi = min(len(file_lines), i0 + n_lines + ctx)
        anchor = "\n".join(file_lines[lo:hi])
        if len(anchor) > 2000:
            anchor = anchor[:1000].rstrip() + "\n...[anchor truncated]...\n" + anchor[-1000:]
        lines = [
            f"[edit-anchor suggestion: {path} lines {lo + 1}-{hi}]",
        ]
        if Path(path).suffix == ".py":
            symbol = EditTool._python_symbol_for_line(content, start_1based)
            if symbol:
                lines.append(f"Enclosing symbol: {symbol}")
        lines.extend(
            [
                "Suggested old_str anchor (copy exactly between markers):",
                "```text",
                anchor,
                "```",
            ],
        )
        return "\n".join(lines)

    @staticmethod
    def _change_context(
        text: str,
        target: str,
        ctx: int = 3,
        hint_offset: int = -1,
    ) -> str:
        """Show context lines around *target* in *text*, marking the span with ``>>``."""
        if hint_offset >= 0 and text[hint_offset : hint_offset + len(target)] == target:
            idx = hint_offset
        else:
            idx = text.find(target)
        if idx < 0:
            return ""
        before = text[:idx]
        line_no = before.count("\n") + 1
        lines = text.splitlines()
        acc = 0
        start_line = 0
        for i, ln in enumerate(lines):
            if acc + len(ln) >= idx:
                start_line = i
                break
            acc += len(ln) + 1
        end_line = start_line + target.count("\n")
        lo = max(0, start_line - ctx)
        hi = min(len(lines), end_line + ctx + 1)
        out: list[str] = [f"(context around edit near line {line_no})"]
        for j in range(lo, hi):
            prefix = ">> " if start_line <= j <= end_line else "   "
            out.append(f"{prefix}{j + 1:6}|{lines[j]}")
        return "\n".join(out)

    async def execute(
        self,
        path: str,
        old_str: str,
        new_str: str,
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        try:
            resolved = PathGuard(
                self.workspace_dir,
                enabled=self.sandbox,
                extra_roots=self.path_guard_extra_roots,
                denied_prefixes=self.path_guard_denied_prefixes,
            ).resolve(path)
        except ValueError as e:
            return ToolResult(error=str(e))
        if not resolved.is_file():
            return ToolResult(error=f"File not found or not a file: {path}")

        content = resolved.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_str)
        if count == 0:
            diag = self._diagnose_not_found(content, old_str)
            tops = self._top_k_line_block_matches(
                content,
                old_str,
                k=int(self.failure_top_k_candidates),
            )
            file_lines = content.splitlines()
            parts: list[str] = []
            n_old_lines = len(old_str.splitlines())
            ctx = 2
            for rank, (ratio, start_ln, _blk) in enumerate(tops, 1):
                parts.append(
                    f"\n--- Candidate {rank} ({ratio:.0%} similar to old_str, "
                    f"starts at line {start_ln}) ---"
                )
                parts.append(
                    self._context_around_lines(
                        file_lines,
                        start_ln,
                        n_lines=n_old_lines,
                        ctx=ctx,
                    ),
                )
                parts.append(
                    self._suggest_old_str_anchor(
                        content,
                        file_lines,
                        start_ln,
                        n_lines=n_old_lines,
                        ctx=ctx,
                        path=path,
                    ),
                )
            extra = "".join(parts)
            cap = int(self.failure_diag_max_chars)
            if len(extra) > cap:
                extra = extra[: cap // 2] + "\n...[truncated]...\n" + extra[-cap // 2 :]
            return ToolResult(
                error=(
                    f"old_str not found in {path}.\n{diag}{extra}"
                ),
            )
        if count > 1:
            return ToolResult(
                error=(
                    f"old_str matches {count} locations in {path}; "
                    "add more surrounding lines so the match is unique."
                )
            )

        new_content = content.replace(old_str, new_str, 1)
        err = validate_syntax(resolved, new_content)
        if err:
            return ToolResult(error=f"Edit would break syntax/format: {err}")

        info = atomic_write(resolved, new_content)
        edit_offset = content.find(old_str)
        ctx = max(3, int(self.return_change_ctx_lines))
        preview = self._change_context(
            new_content,
            new_str,
            ctx=ctx,
            hint_offset=edit_offset,
        )
        snapshot = format_file_snapshot_for_tool_return(
            new_content,
            max_chars=int(self.return_full_max_chars),
            max_lines=int(self.return_full_max_lines),
            head_tail_lines=int(self.return_head_tail_lines),
        )
        snap_block = f"\n--- Current file snapshot (after edit) ---\n{snapshot}" if snapshot else ""
        return ToolResult(
            output=(
                f"Edited {path}: 1 replacement OK.\n{preview}\n"
                f"({info['lines']} lines, sha256~{info['sha256_short']}){snap_block}"
            )
        )
