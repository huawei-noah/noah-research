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

"""ScienceFlow workspace tools for ScienceAgent (bash / file IO / search)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from scienceflow.core.tools.file_utils import _normalize_extra_roots
from scienceflow.core.skills.registry import SkillRegistry

from scienceflow.core.tools.bash_tool import BashTool
from scienceflow.core.tools.tool_collection import ToolCollection
from scienceflow.core.tools.edit_tool import EditTool
from scienceflow.core.tools.file_utils import (
    PathGuard,
    atomic_write,
    display_under_root,
    file_summary,
    validate_syntax,
)
from scienceflow.core.tools.grep_tool import GrepTool
from scienceflow.core.tools.glob_tool import GlobTool
from scienceflow.core.tools.ls_tool import LsTool
from scienceflow.core.tools.read_tool import ReadTool
from scienceflow.core.tools.resource_wait_tool import ResourceWaitTool
from scienceflow.core.tools.skill_tool import SkillTool
from scienceflow.core.tools.write_tool import WriteTool

__all__ = [
    "PathGuard",
    "display_under_root",
    "atomic_write",
    "file_summary",
    "validate_syntax",
    "BashTool",
    "WriteTool",
    "EditTool",
    "ReadTool",
    "GrepTool",
    "GlobTool",
    "LsTool",
    "SkillTool",
    "ResourceWaitTool",
    "create_tool_collection",
]


def create_tool_collection(
    workspace_dir: str | Path,
    *,
    sandbox: bool = True,
    path_guard_extra_roots: Sequence[str | Path] | None = None,
    max_bash_output_chars: int = 8000,
    max_bash_stream_line_chars: int = 2400,
    bash_observation_summary_enabled: bool = False,
    bash_dedup_min_repeat: int = 3,
    bash_distill_tracebacks: bool = True,
    bash_timeout_sec: float = 1800.0,
    bash_timeout_slow_sec: float = 1800.0,
    extra_env: Optional[dict[str, str]] = None,
    readonly_dirs: Sequence[str] | None = None,
    resource_observer: Any | None = None,
    skill_registry: SkillRegistry | None = None,
    task_type: str | None = None,
    skill_allow_names: Sequence[str] | None = None,
    skill_tool_mode: str = "all",
    skill_allow_generic_wildcard: bool = True,
    skill_visible_max: int = 0,
    include_write_edit_tools: bool = True,
    tool_display_paths_relative: bool = True,
    tool_display_root: str | Path | None = None,
    write_return_full_max_chars: int = 12000,
    write_return_full_max_lines: int = 250,
    write_return_head_tail_lines: int = 40,
    edit_return_full_max_chars: int = 12000,
    edit_return_full_max_lines: int = 250,
    edit_return_head_tail_lines: int = 40,
    edit_return_change_ctx_lines: int = 8,
    edit_failure_top_k_candidates: int = 3,
    edit_failure_diag_max_chars: int = 2000,
    grep_max_results_lines: int = 50,
) -> ToolCollection:
    """Build the default tool set rooted at *workspace_dir*."""
    ws = Path(workspace_dir).resolve()
    pg = list(_normalize_extra_roots(path_guard_extra_roots))
    ro = list(readonly_dirs) if readonly_dirs else []
    display_root = Path(tool_display_root).resolve() if tool_display_root else None
    bash_description = BashTool.model_fields["description"].default
    bash_parameters = BashTool.model_fields["parameters"].default
    if not include_write_edit_tools:
        bash_description = (
            "Execute a shell command with cwd already set to the task workspace. "
            "Use this tool for compact inspection/search/data summaries, package installs, "
            "validation/training runs, artifact preservation, and file creation or "
            "file modification. Put complete file content or the exact rewrite "
            "operation in the command. Use relative paths (e.g. dataset/) or run "
            "`pwd` first; do not assume Kaggle paths like /mnt/data exist on this "
            "machine. IMPORTANT: cwd is already the workspace — do NOT prefix "
            "commands with `cd /home`, `cd /home/user`, or any other absolute-path "
            "cd. Run workspace commands directly, such as `python3 solution.py` "
            "or `python3 train.py` when that is the relevant entrypoint. For "
            "workspace facts, prefer `read`, `grep`, `glob`, and `ls` over raw "
            "bash dumps. For large files or directories, do not dump raw content; "
            "run a small summary command/script that scans the full target and "
            "prints compact statistics such as counts, dtypes, missing values, "
            "grouped file counts, and representative head/tail samples. If a "
            "command may emit many lines or run for a long time, write verbose "
            "logs to workspace files and print only key metrics, exit status, "
            "artifact paths, and a compact tail. "
            "Do not pipe verbose training or validation directly to `tail` as the "
            "only output record; save the full log first. During iterative "
            "optimization, update a workspace ledger immediately after each "
            "substantive attempt, preserve improved best-known artifacts before "
            "starting the next risky experiment, and promote the best known artifact "
            "to the expected final path before finishing. "
            "A compact or truncated bash result is not complete ground truth; recover "
            "exact facts with narrower commands or the dedicated workspace tools."
        )
        bash_parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run (bash -c semantics via subprocess shell). "
                        "For file changes, include the complete content or exact rewrite "
                        "operation in this command. cwd is already set to the workspace — "
                        "do NOT use `cd /home/...` or any absolute path prefix before "
                        "running scripts. Use workspace-relative paths (e.g. dataset/) "
                        "or run `pwd` to confirm location. For large observations, "
                        "prefer full-scan summaries over raw dumps. If output is "
                        "summarized or truncated, use a narrower command, `grep`, or "
                        "paged `read` calls to recover exact details. For verbose "
                        "training or validation, save a full workspace log first; "
                        "do not use `command | tail` as the only output record."
                    ),
                },
            },
            "required": ["command"],
        }

    tools: list[Any] = [
        BashTool(
            workspace_dir=ws,
            description=bash_description,
            parameters=bash_parameters,
            max_output_chars=max_bash_output_chars,
            max_stream_line_chars=max_bash_stream_line_chars,
            observation_summary_enabled=bool(bash_observation_summary_enabled),
            dedup_min_repeat=bash_dedup_min_repeat,
            distill_tracebacks=bash_distill_tracebacks,
            bash_timeout_sec=bash_timeout_sec,
            bash_timeout_slow_sec=bash_timeout_slow_sec,
            extra_env=extra_env or None,
            readonly_dirs=ro,
            path_guard_extra_roots=pg,
            resource_observer=resource_observer,
        ),
    ]
    if include_write_edit_tools:
        tools.extend(
            [
                WriteTool(
                    workspace_dir=ws,
                    sandbox=sandbox,
                    path_guard_extra_roots=pg,
                    return_full_max_chars=write_return_full_max_chars,
                    return_full_max_lines=write_return_full_max_lines,
                    return_head_tail_lines=write_return_head_tail_lines,
                ),
                EditTool(
                    workspace_dir=ws,
                    sandbox=sandbox,
                    path_guard_extra_roots=pg,
                    return_full_max_chars=edit_return_full_max_chars,
                    return_full_max_lines=edit_return_full_max_lines,
                    return_head_tail_lines=edit_return_head_tail_lines,
                    return_change_ctx_lines=edit_return_change_ctx_lines,
                    failure_top_k_candidates=edit_failure_top_k_candidates,
                    failure_diag_max_chars=edit_failure_diag_max_chars,
                ),
            ],
        )
    tools.extend(
        [
            ReadTool(
                workspace_dir=ws,
                sandbox=sandbox,
                path_guard_extra_roots=pg,
                display_paths_relative=bool(tool_display_paths_relative),
                display_root=display_root,
            ),
            GrepTool(
                workspace_dir=ws,
                sandbox=sandbox,
                path_guard_extra_roots=pg,
                max_results_lines=int(grep_max_results_lines),
                display_paths_relative=bool(tool_display_paths_relative),
                display_root=display_root,
            ),
            GlobTool(
                workspace_dir=ws,
                sandbox=sandbox,
                path_guard_extra_roots=pg,
                display_paths_relative=bool(tool_display_paths_relative),
                display_root=display_root,
            ),
            LsTool(
                workspace_dir=ws,
                sandbox=sandbox,
                path_guard_extra_roots=pg,
                display_paths_relative=bool(tool_display_paths_relative),
                display_root=display_root,
            ),
        ],
    )
    if resource_observer is not None:
        tools.append(ResourceWaitTool(resource_observer=resource_observer))
    if skill_registry is not None:
        tools.append(
            SkillTool(
                registry=skill_registry,
                task_type=task_type,
                mode=skill_tool_mode,
                allow_names=tuple(skill_allow_names or ()),
                allow_generic_wildcard=bool(skill_allow_generic_wildcard),
                visible_max=int(skill_visible_max or 0),
            ),
        )
    return ToolCollection(*tools)
