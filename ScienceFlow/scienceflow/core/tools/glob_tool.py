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

"""Structured workspace file discovery via glob patterns."""

from __future__ import annotations

from pathlib import Path

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.tools.file_utils import (
    PathGuard,
    display_from_request,
    join_display_path,
)


def _has_parent_ref(pattern: str) -> bool:
    parts = pattern.replace("\\", "/").split("/")
    return any(part == ".." for part in parts)


def _has_hidden_part(path: Path, base: Path) -> bool:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts if part not in ("", "."))


class GlobTool(BaseTool):
    name: str = "glob"
    description: str = (
        "Find files and optionally directories under the workspace using a glob pattern. "
        "Use patterns like '*.py', '**/*.csv', or 'dataset/**/*.json'."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern relative to path, e.g. '*.py' or '**/*.csv'.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search, relative to workspace. Default '.'.",
            },
            "include_dirs": {
                "type": "boolean",
                "description": "Whether to include matching directories. Default false.",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include dotfiles and files inside dot directories. Default false.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum matches to return. Default 200.",
            },
        },
        "required": ["pattern"],
    }
    workspace_dir: Path = Field(...)
    sandbox: bool = Field(default=True)
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    display_paths_relative: bool = Field(default=True)
    display_root: Path | None = Field(default=None)
    max_results: int = Field(default=1000)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        include_dirs: bool = False,
        include_hidden: bool = False,
        limit: int = 200,
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        pattern = str(pattern or "").strip()
        if not pattern:
            return ToolResult(error="glob tool requires non-empty 'pattern' argument")
        if Path(pattern).is_absolute() or _has_parent_ref(pattern):
            return ToolResult(error="glob pattern must be relative and must not contain '..'")

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
        if not resolved.is_dir():
            return ToolResult(error=f"Path is not a directory: {path}")

        cap = max(1, min(int(limit or 200), int(self.max_results)))
        matches: list[Path] = []
        truncated = False
        try:
            iterator = resolved.glob(pattern)
            for candidate in iterator:
                if guard.is_denied(candidate):
                    continue
                try:
                    guard.resolve(candidate)
                except ValueError:
                    continue
                if not include_hidden and _has_hidden_part(candidate, resolved):
                    continue
                if candidate.is_dir() and not include_dirs:
                    continue
                if not candidate.is_dir() and not candidate.is_file():
                    continue
                if len(matches) >= cap:
                    truncated = True
                    break
                matches.append(candidate)
        except (OSError, RuntimeError, ValueError) as e:
            return ToolResult(error=f"glob failed: {e}")

        label = display_from_request(
            guard.root,
            resolved,
            path,
            logical_paths=bool(self.display_paths_relative),
            display_root=self.display_root,
        )
        if not matches:
            return ToolResult(output=f"[glob: {pattern!r} under {label}]\nNo matches")

        def shown_path(p: Path) -> str:
            try:
                rel = p.relative_to(resolved).as_posix()
            except ValueError:
                rel = p.name
            out = join_display_path(label, rel)
            if p.is_dir():
                out += "/"
            return out

        matches.sort(key=lambda p: shown_path(p).lower())
        lines = [f"[glob: {pattern!r} under {label}]"]
        for p in matches:
            lines.append(shown_path(p))
        if truncated:
            lines.append(f"... [more matches omitted, limit={cap}]")
        else:
            lines.append(f"[{len(matches)} matches]")
        return ToolResult(output="\n".join(lines))
