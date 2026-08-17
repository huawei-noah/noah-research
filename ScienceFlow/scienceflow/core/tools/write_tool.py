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

"""Full-file write with syntax pre-check and atomic replace."""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Any

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.tools.file_utils import (
    PathGuard,
    atomic_write,
    display_under_root,
    validate_syntax,
)
from scienceflow.core.tools.write_placeholder import looks_like_write_placeholder_mimicry

logger = logging.getLogger(__name__)

_CONTENT_ALIASES = ("body", "text", "file_contents")

# Defaults aligned with Config (see default.yaml); overridable per WriteTool instance.
DEFAULT_RETURN_FULL_MAX_CHARS = 12000
DEFAULT_RETURN_FULL_MAX_LINES = 250
DEFAULT_RETURN_HEAD_TAIL_LINES = 40


def _py_ast_smoke_hints(source: str) -> str | None:
    """Lightweight AST hints (non-blocking): unused import-ish names at module scope."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = (alias.asname or alias.name).split(".")[0]
                imported.add(base)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    base = (alias.asname or alias.name).split(".")[0]
                    imported.add(base)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    unused = sorted(n for n in imported if n not in used and not n.startswith("_"))
    if not unused:
        return None
    return (
        "Possible unused top-level imports (heuristic, not exhaustive): "
        + ", ".join(unused[:12])
        + (" …" if len(unused) > 12 else "")
    )


def format_file_snapshot_for_tool_return(
    content: str,
    *,
    max_chars: int = DEFAULT_RETURN_FULL_MAX_CHARS,
    max_lines: int = DEFAULT_RETURN_FULL_MAX_LINES,
    head_tail_lines: int = DEFAULT_RETURN_HEAD_TAIL_LINES,
) -> str:
    """Numbered snapshot for LLM tool feedback: full file when within budget, else head+tail.

    Used by ``write`` and ``edit`` so the model sees disk truth for the next ``edit`` call.
    """
    if not content:
        return ""
    lines_list = content.splitlines()
    total = len(lines_list)
    if total == 0:
        return ""
    if len(content) <= max_chars and total <= max_lines:
        return "\n".join(f"{i:>6}|{line}" for i, line in enumerate(lines_list, 1))
    ht = max(1, head_tail_lines)
    if total <= 2 * ht + 2:
        return "\n".join(f"{i:>6}|{line}" for i, line in enumerate(lines_list, 1))
    head = "\n".join(f"{i:>6}|{line}" for i, line in enumerate(lines_list[:ht], 1))
    tail = "\n".join(
        f"{i:>6}|{line}"
        for i, line in enumerate(
            lines_list[-ht:],
            total - ht + 1,
        )
    )
    omitted = total - 2 * ht
    return f"{head}\n  ... ({omitted} lines omitted) ...\n{tail}"


class WriteTool(BaseTool):
    name: str = "write"
    description: str = (
        "Create or overwrite a file under the workspace with the given full content. "
        "Syntax is validated for .py / .json / .yaml before write."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to workspace root.",
            },
            "content": {
                "type": "string",
                "description": "Complete new file contents.",
            },
        },
        "required": ["path", "content"],
    }
    workspace_dir: Path = Field(...)
    sandbox: bool = Field(default=True)
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    return_full_max_chars: int = Field(default=DEFAULT_RETURN_FULL_MAX_CHARS)
    return_full_max_lines: int = Field(default=DEFAULT_RETURN_FULL_MAX_LINES)
    return_head_tail_lines: int = Field(default=DEFAULT_RETURN_HEAD_TAIL_LINES)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        path: str | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        kwargs.pop("config", None)
        if content is None:
            for alt in _CONTENT_ALIASES:
                if alt in kwargs and kwargs[alt] is not None:
                    content = kwargs.pop(alt)
                    break
        path_str = str(path).strip() if path is not None else ""
        if not path_str:
            return ToolResult(
                error="write requires non-empty 'path' (relative to workspace root).",
            )
        if content is None:
            extra_keys = sorted(k for k in kwargs if k not in _CONTENT_ALIASES)
            logger.warning(
                "write missing 'content'; path=%r; other_kwarg_keys=%s",
                path_str,
                extra_keys,
            )
            return ToolResult(
                error=(
                    "write requires 'content' (complete file body). "
                    "If content is missing, the model may have omitted it in the tool JSON, "
                    "used a non-standard key, or the tool arguments were truncated by the API."
                ),
            )
        if looks_like_write_placeholder_mimicry(content):
            return ToolResult(
                error=(
                    "write rejected: `content` looks like a log/memory placeholder or compressed "
                    "memory snippet (e.g. `<<<WRITE_PLACEHOLDER: ...>>>`, `<<<WRITE_CONTENT: ...>>>`, "
                    "`[interaction-log: write body omitted; N chars applied by tool]`, "
                    "`# [chat-memory: write payload omitted (N bytes); ...]`, "
                    "`<N chars>`, or lines with `# <<<WRITE_OK:` / `# <<<MEMORY_COMPRESSED:` / "
                    "`# [MEMORY_COMPRESSED:`), "
                    "not real file text. "
                    "Put the **complete** file body in the `content` field (use **read** if needed)."
                ),
            )
        path = path_str
        try:
            resolved = PathGuard(
                self.workspace_dir,
                enabled=self.sandbox,
                extra_roots=self.path_guard_extra_roots,
                denied_prefixes=self.path_guard_denied_prefixes,
            ).resolve(path)
        except ValueError as e:
            return ToolResult(error=str(e))
        ext = resolved.suffix.lower()
        if ext == ".py":
            try:
                ast.parse(content, filename=str(resolved))
            except SyntaxError as e:
                return ToolResult(error=f"Syntax check failed (ast.parse): {e}")
            ast_hints = _py_ast_smoke_hints(content)
        else:
            ast_hints = None
            err = validate_syntax(resolved, content)
            if err:
                return ToolResult(error=f"Syntax check failed: {err}")
        display_path = display_under_root(self.workspace_dir, resolved)
        prev_lines: int | None = None
        if resolved.exists():
            new_raw = content.encode("utf-8")
            existing_raw = resolved.read_bytes()
            if hashlib.sha256(new_raw).digest() == hashlib.sha256(existing_raw).digest():
                existing_sha = hashlib.sha256(existing_raw).hexdigest()[:16]
                n_lines = len(content.splitlines())
                return ToolResult(
                    output=(
                        f"No-op: {display_path} already has exactly this content "
                        f"({n_lines} lines, sha256~{existing_sha}). "
                        "File was not modified. Use bash to test when the code is already correct."
                    )
                )
            try:
                prev_lines = len(existing_raw.decode("utf-8", errors="replace").splitlines())
            except Exception:
                prev_lines = None
        info = atomic_write(resolved, content)
        numbered = format_file_snapshot_for_tool_return(
            content,
            max_chars=int(self.return_full_max_chars),
            max_lines=int(self.return_full_max_lines),
            head_tail_lines=int(self.return_head_tail_lines),
        )
        summary_block = f"\n{numbered}" if numbered else ""
        hint_block = f"\n[ast-smoke] {ast_hints}" if ast_hints else ""
        # Soft hint when writing an existing file with small delta — prefer edit next time.
        edit_hint_block = ""
        if prev_lines is not None and prev_lines > 0:
            new_lines = info["lines"]
            ratio_changed = abs(new_lines - prev_lines) / prev_lines
            if ratio_changed <= 0.30:
                edit_hint_block = (
                    f"\n[hint] Only ~{ratio_changed:.0%} of lines changed vs the previous file. "
                    "Next time prefer `edit` (old_string/new_string) for surgical changes — "
                    "it is faster (lower ttft) and easier to review."
                )
        return ToolResult(
            output=(
                f"Written {display_path} ({info['lines']} lines, {info['bytes']} bytes, "
                f"sha256~{info['sha256_short']}){hint_block}{edit_hint_block}{summary_block}"
            )
        )
