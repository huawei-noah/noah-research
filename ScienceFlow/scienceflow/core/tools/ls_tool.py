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

"""Structured workspace directory listing tool."""

from __future__ import annotations

from pathlib import Path

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.tools.file_utils import (
    PathGuard,
    display_from_request,
    join_display_path,
)


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix() or "."
    except ValueError:
        return path.name or "."


def _display_entry(base_label: str, listing_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(listing_root).as_posix()
    except ValueError:
        return _display_path(listing_root, path)
    return join_display_path(base_label, rel)


def _is_hidden(path: Path, base: Path) -> bool:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts if part not in ("", "."))


def _entry_kind(path: Path) -> str:
    if path.is_symlink():
        try:
            if path.is_dir():
                return "DIRLINK"
            if path.is_file():
                return "FILELINK"
        except OSError:
            return "LINK"
        return "LINK"
    if path.is_dir():
        return "DIR"
    if path.is_file():
        return "FILE"
    return "OTHER"


class LsTool(BaseTool):
    name: str = "ls"
    description: str = (
        "List files and directories under a workspace path with stable ordering. "
        "Use recursive=true and depth to inspect nested directories."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory path relative to workspace. Default '.'.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to recurse into subdirectories. Default false.",
            },
            "depth": {
                "type": "integer",
                "description": "Maximum recursion depth when recursive=true. Default 2.",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include dotfiles and dot directories. Default false.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum entries to return. Default 200.",
            },
        },
        "required": [],
    }
    workspace_dir: Path = Field(...)
    sandbox: bool = Field(default=True)
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    display_paths_relative: bool = Field(default=True)
    display_root: Path | None = Field(default=None)
    max_results: int = Field(default=1000)
    max_depth: int = Field(default=8)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        path: str = ".",
        recursive: bool = False,
        depth: int = 2,
        include_hidden: bool = False,
        limit: int = 200,
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        guard = PathGuard(
            self.workspace_dir,
            enabled=self.sandbox,
            extra_roots=self.path_guard_extra_roots,
            denied_prefixes=self.path_guard_denied_prefixes,
        )
        try:
            resolved = guard.resolve(path)
        except ValueError as e:
            return ToolResult(error=str(e))
        if not resolved.exists():
            return ToolResult(error=f"Path not found: {path}")

        cap = max(1, min(int(limit or 200), int(self.max_results)))
        label = display_from_request(
            guard.root,
            resolved,
            path,
            logical_paths=bool(self.display_paths_relative),
            display_root=self.display_root,
        )

        if resolved.is_file():
            try:
                size = resolved.stat().st_size
            except OSError:
                size = -1
            return ToolResult(output=f"[ls: {label}]\nFILE {label} ({size} bytes)")

        if not resolved.is_dir():
            return ToolResult(error=f"Path is not a file or directory: {path}")

        max_depth = max(0, min(int(depth or 0), int(self.max_depth)))
        if not recursive:
            max_depth = 0
        lines: list[str] = [f"[ls: {label}]"]
        count = 0
        truncated = False

        def allowed(candidate: Path) -> bool:
            if guard.is_denied(candidate):
                return False
            try:
                guard.resolve(candidate)
                return True
            except ValueError:
                return False

        def walk(current: Path, rel_depth: int) -> None:
            nonlocal count, truncated
            if truncated:
                return
            try:
                entries = sorted(
                    current.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except OSError as e:
                lines.append(f"{'  ' * rel_depth}(cannot read: {e})")
                return

            for entry in entries:
                if count >= cap:
                    truncated = True
                    lines.append(f"... [more entries omitted, limit={cap}]")
                    return
                if not include_hidden and _is_hidden(entry, resolved):
                    continue
                if guard.is_denied(entry):
                    continue
                if not allowed(entry):
                    lines.append(
                        f"{'  ' * rel_depth}BLOCKED {_display_entry(label, resolved, entry)}"
                    )
                    count += 1
                    continue
                kind = _entry_kind(entry)
                shown = _display_entry(label, resolved, entry)
                suffix = "/" if kind in {"DIR", "DIRLINK"} else ""
                size = ""
                if kind in {"FILE", "FILELINK"}:
                    try:
                        size = f" ({entry.stat().st_size} bytes)"
                    except OSError:
                        size = " (size unknown)"
                lines.append(f"{'  ' * rel_depth}{kind} {shown}{suffix}{size}")
                count += 1
                if recursive and rel_depth < max_depth and kind in {"DIR", "DIRLINK"}:
                    walk(entry, rel_depth + 1)

        walk(resolved, 0)
        if not truncated:
            lines.append(f"[{count} entries]")
        return ToolResult(output="\n".join(lines))
