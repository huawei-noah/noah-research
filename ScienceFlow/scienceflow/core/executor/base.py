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

"""Unified execution base: CodeRunner ABC and ExecutionResult."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("scienceflow")

MAX_OUTPUT_CHARS = 4000
# Larger cap when execution failed so long tracebacks (e.g. pandas) keep the root cause.
MAX_OUTPUT_CHARS_FAILURE = 16_000


@dataclass
class ExecutionResult:
    """Unified result from any code/shell execution.

    Fields kept compatible with ScienceAgent execution-result consumers.
    (term_out, exec_time, exc_type, exc_info, exc_stack).
    """

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    exec_time: float = 0.0

    term_out: list[str] = field(default_factory=list)
    all_term_out: list[str] = field(default_factory=list)
    metric_lines: list[str] = field(default_factory=list)
    exc_type: str | None = None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    output: Any = None
    # When set, caps successful stdout/term_out in ``display`` (e.g. prep bash exploration).
    display_cap_success: int | None = None

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def display(self) -> str:
        """Human-readable output for feeding back to the LLM."""
        if self.stderr and not self.success:
            text = self.stderr
        elif self.stdout:
            text = self.stdout
        elif self.term_out:
            text = "\n".join(self.term_out)
        else:
            return "[No output]"
        if not self.success:
            limit = MAX_OUTPUT_CHARS_FAILURE
        else:
            limit = (
                self.display_cap_success
                if self.display_cap_success is not None
                else MAX_OUTPUT_CHARS
            )
        if len(text) > limit:
            text = text[:limit] + "\n...[Output truncated]"
        return text


class CodeRunner(ABC):
    """Abstract base for shell and python execution.

    ``workspace_dir`` is bound at construction time and serves as:
      - temp file directory (for PythonRunner)
      - subprocess cwd
      - output artifact location

    On construction the class auto-detects the project's uv-managed
    ``.venv`` so that subprocesses use the correct Python interpreter and
    ``PATH``.
    """

    _python_exec: str
    _venv_bin: str | None

    def __init__(self, workspace_dir: str | Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._python_exec, self._venv_bin = self._resolve_project_venv()

    @staticmethod
    def _resolve_project_venv() -> tuple[str, str | None]:
        """Find project Python and bin dir via :func:`resolve_project_python`."""
        from scienceflow.core.runtime_env import resolve_project_python

        python_exec, bin_dir = resolve_project_python()
        return python_exec, bin_dir

    def _build_run_env(self, extra: dict | None = None) -> dict[str, str]:
        """Return a subprocess env dict with the venv ``bin/`` prepended to PATH."""
        import os
        run_env = dict(os.environ)
        if self._venv_bin:
            run_env["PATH"] = self._venv_bin + ":" + run_env.get("PATH", "")
        if extra:
            run_env.update(extra)
        return run_env

    @abstractmethod
    async def run(self, code: str, *, timeout: float = 30,
                  env: dict | None = None) -> ExecutionResult:
        ...
