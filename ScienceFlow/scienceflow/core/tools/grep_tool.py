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

"""Structured workspace text search via ripgrep."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from deepcraft_core.tool import BaseTool, ToolResult
from pydantic import ConfigDict, Field

from scienceflow.core.tools.file_utils import (
    PathGuard,
    display_from_request,
    join_display_path,
)

_RG_MAX_COLUMNS = 20_000
_RG_STREAM_LIMIT = 1024 * 1024


class GrepTool(BaseTool):
    name: str = "grep"
    description: str = (
        "Search for a regex pattern under the workspace. "
        "Uses ripgrep (rg) and returns matching lines with file paths and line numbers."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {
                "type": "string",
                "description": "File or directory relative to workspace. Default '.'",
            },
            "include": {
                "type": "string",
                "description": "Optional glob passed to rg -g, e.g. '*.py'.",
            },
            "file_type": {
                "type": "string",
                "description": "Optional ripgrep file type passed to rg -t, e.g. 'py'.",
            },
        },
        "required": ["pattern"],
    }
    workspace_dir: Path = Field(...)
    max_results_lines: int = Field(default=50)
    sandbox: bool = Field(default=True)
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    display_paths_relative: bool = Field(default=True)
    display_root: Path | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        file_type: str = "",
        **kwargs,
    ) -> ToolResult:
        kwargs.pop("config", None)
        if not pattern.strip():
            return ToolResult(error="Empty search pattern")
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
        rg = shutil.which("rg")
        if rg is None:
            return ToolResult(error="rg binary not found on PATH")

        logical_label = display_from_request(
            guard.root,
            resolved,
            path,
            logical_paths=bool(self.display_paths_relative),
            display_root=self.display_root,
        )
        remap_rg_paths = False
        search_target_is_file = resolved.is_file()
        try:
            rel = resolved.relative_to(guard.root)
            search_cwd = guard.root
            search_path = "." if str(rel) == "." else rel.as_posix()
        except ValueError:
            remap_rg_paths = True
            if search_target_is_file:
                search_cwd = resolved.parent
                search_path = resolved.name
            else:
                search_cwd = resolved
                search_path = "."

        cmd: list[str] = [
            rg,
            "--line-number",
            "--with-filename",
            "--color=never",
            "--max-columns",
            str(_RG_MAX_COLUMNS),
            "--max-columns-preview",
        ]
        # Double insurance: rg already respects ignore files and skips hidden dirs by
        # default, but these guards help for workspaces without complete ignore rules
        # and extra roots outside the repository.
        for excluded in (
            "**/.git/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "**/.mypy_cache/**",
            "**/.ruff_cache/**",
        ):
            cmd.extend(["-g", f"!{excluded}"])
        for prefix in self.path_guard_denied_prefixes:
            clean = str(prefix or "").strip().strip("/")
            if clean:
                cmd.extend(["-g", f"!{clean}/**", "-g", f"!**/{clean}/**"])
        if file_type.strip():
            cmd.extend(["-t", file_type.strip()])
        if include.strip():
            cmd.extend(["-g", include.strip()])
        cmd.extend(["--", pattern, search_path])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(search_cwd),
            limit=_RG_STREAM_LIMIT,
        )
        cap = max(1, int(self.max_results_lines))
        raw_lines: list[str] = []
        killed_for_limit = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    return ToolResult(error="Search timed out after 30s")
                assert proc.stdout is not None
                try:
                    line_b = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                except ValueError:
                    proc.kill()
                    await proc.wait()
                    return ToolResult(
                        error=(
                            "Search output line exceeded the internal read limit even after "
                            f"rg preview truncation at {_RG_MAX_COLUMNS} columns; narrow the search."
                        )
                    )
                if not line_b:
                    break
                line = line_b.decode(errors="replace").rstrip("\n")
                if not line.strip():
                    continue
                if remap_rg_paths:
                    line = _rewrite_rg_line_path(
                        line,
                        logical_label=logical_label,
                        target_is_file=search_target_is_file,
                    )
                raw_lines.append(line)
                if len(raw_lines) > cap:
                    killed_for_limit = True
                    proc.kill()
                    break
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(error="Search timed out after 30s")

        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        stderr_b = b""
        if proc.stderr is not None:
            try:
                stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=1)
            except asyncio.TimeoutError:
                stderr_b = b""
        stderr = stderr_b.decode(errors="replace").strip()
        # exit 1 = no matches for ripgrep. Negative rc can be intentional after
        # we stop once the memory-facing result limit is reached.
        if not killed_for_limit and proc.returncode not in (0, 1):
            err = stderr or f"rg failed with code {proc.returncode}"
            return ToolResult(error=err)

        if not raw_lines:
            return ToolResult(output=f"No matches for pattern: {pattern!r}")

        shown = raw_lines[:cap]
        out = "\n".join(shown)
        total = len(raw_lines)
        if killed_for_limit:
            out += f"\n... [more matches omitted, stopped after limit={cap}]"
        elif total > cap:
            out += f"\n... [{total - cap} more matches omitted, {total} total]"
        else:
            out += f"\n[{total} matches]"
        return ToolResult(output=out)


def _rewrite_rg_line_path(
    line: str,
    *,
    logical_label: str,
    target_is_file: bool,
) -> str:
    """Rewrite ripgrep's filename prefix to the tool-request logical path."""
    if ":" not in line:
        return line
    file_part, rest = line.split(":", 1)
    shown = file_part[2:] if file_part.startswith("./") else file_part
    if target_is_file:
        rel = logical_label
    else:
        rel = "." if shown in ("", ".") else shown
        rel = join_display_path(logical_label, rel)
    return f"{rel}:{rest}"
