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

"""Read text files with line numbers and paging; list directories."""

from __future__ import annotations

from pathlib import Path

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.tools.file_utils import PathGuard, display_from_request


def _list_dir_tree(root: Path, label: str, *, max_entries: int = 400, max_depth: int = 4) -> str:
    lines: list[str] = [f"[directory listing: {label}]"]
    count = 0

    def walk(current: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            lines.append(f"  (cannot read: {e})")
            return
        for p in entries:
            if count >= max_entries:
                lines.append("  ... (truncated)")
                return
            if p.name.startswith(".") and p.name not in (".", ".."):
                continue
            rel = p.relative_to(root)
            indent = "  " * depth
            if p.is_dir():
                lines.append(f"{indent}{rel.as_posix()}/")
                count += 1
                walk(p, depth + 1)
            else:
                try:
                    sz = p.stat().st_size
                except OSError:
                    sz = -1
                lines.append(f"{indent}{rel.as_posix()} ({sz} bytes)")
                count += 1

    walk(root, 0)
    return "\n".join(lines)


class ReadTool(BaseTool):
    name: str = "read"
    description: str = (
        "Read a text file with line numbers, or list a directory tree. "
        "Use offset/limit to page through large files (default 200 lines per call). "
        "Always provide path on every call; when paging, repeat the same path with the new offset."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to workspace. Required on every call, including paging calls.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based start line. Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return; default 200. Use 0 for entire file (careful).",
            },
        },
        "required": ["path"],
    }
    workspace_dir: Path = Field(...)
    sandbox: bool = Field(default=True)
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    display_paths_relative: bool = Field(default=True)
    display_root: Path | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int = 200,
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        path_str = str(path).strip() if path is not None else ""
        if not path_str:
            return ToolResult(
                error=(
                    "read requires non-empty 'path' (relative to workspace root). "
                    "When paging with offset/limit, repeat the same path; there is no implicit previous file."
                ),
            )
        guard = PathGuard(
            self.workspace_dir,
            enabled=self.sandbox,
            extra_roots=self.path_guard_extra_roots,
            denied_prefixes=self.path_guard_denied_prefixes,
        )
        try:
            resolved = guard.resolve(path_str)
        except ValueError as e:
            return ToolResult(error=str(e))
        if not resolved.exists():
            return ToolResult(error=f"Path not found: {path_str}")
        if resolved.is_dir():
            label = display_from_request(
                guard.root,
                resolved,
                path_str,
                logical_paths=bool(self.display_paths_relative),
                display_root=self.display_root,
            )
            return ToolResult(output=_list_dir_tree(resolved, label))

        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        total = len(lines)
        start = max(0, int(offset) - 1)
        lim = int(limit)
        if lim <= 0:
            end = total
        else:
            end = min(total, start + lim)
        selected = lines[start:end]
        numbered = "\n".join(
            f"{i:>6}|{line}" for i, line in enumerate(selected, start=start + 1)
        )
        rel = display_from_request(
            guard.root,
            resolved,
            path_str,
            logical_paths=bool(self.display_paths_relative),
            display_root=self.display_root,
        )
        meta = f"[{rel}: {total} lines total"
        if end < total or start > 0:
            meta += f", showing {start + 1}-{end if end >= start else start}"
        meta += "]"
        return ToolResult(output=f"{meta}\n{numbered}")
