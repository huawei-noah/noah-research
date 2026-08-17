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

"""PythonRunner — async subprocess Python execution (no MLE sandbox).

Uses the project's uv-managed ``.venv`` Python (resolved in ``CodeRunner``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time

from scienceflow.core.executor.base import CodeRunner, ExecutionResult
from scienceflow.core.subprocess_utils import _LIVE_PGIDS, spawn_exec, terminate_tree
from scienceflow.core.executor.output_parser import (
    extract_metric_lines_from_stdout,
    format_compact_exc_stack,
    parse_traceback,
    strip_env_token_notes,
    strip_working_dir_from_content,
)

logger = logging.getLogger("scienceflow")


class PythonRunner(CodeRunner):
    """Execute Python code in an isolated subprocess.

    Each ``run()`` call writes code to a temp .py file inside
    ``workspace_dir``, spawns ``python temp.py`` with ``cwd=workspace_dir``,
    and returns a structured ``ExecutionResult``.
    """

    async def run(
        self,
        code: str,
        *,
        timeout: float = 120,
        env: dict | None = None,
        pre_code: str = "",
    ) -> ExecutionResult:
        full_code = (pre_code + code) if pre_code else code
        working_dir = str(self.workspace_dir)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=working_dir, encoding="utf-8",
        ) as f:
            f.write(full_code)
            temp_file = f.name

        run_env = self._build_run_env(env)

        start = time.time()
        try:
            proc = await spawn_exec(
                self._python_exec, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=run_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                await terminate_tree(proc)
                elapsed = time.time() - start
                return ExecutionResult(
                    stderr=f"TimeoutError: execution exceeded {timeout}s",
                    returncode=-1,
                    exec_time=elapsed,
                    term_out=[f"TimeoutError: Execution exceeded {timeout}s"],
                    exc_type="TimeoutError",
                    exc_info={"msg": f"Execution exceeded {timeout} seconds"},
                )
            finally:
                _LIVE_PGIDS.discard(proc.pid)

            elapsed = time.time() - start
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            rc = proc.returncode
            if rc is None:
                rc = 0
            return self._build_result(stdout, stderr, elapsed, working_dir, rc)
        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    @staticmethod
    def _build_result(
        stdout: str,
        stderr: str,
        exec_time: float,
        working_dir: str,
        returncode: int,
    ) -> ExecutionResult:
        all_term_out: list[str] = []
        if stdout:
            all_term_out.append(stdout)
        if stderr:
            all_term_out.append("\n[stderr]\n" + stderr)

        metric_lines = extract_metric_lines_from_stdout(stdout)

        exc_type, exc_info, exc_stack = None, None, None
        term_out: list[str] = [f"Execution time: {exec_time:.1f}s"]

        if stderr.strip():
            exc_type, exc_info, exc_stack = parse_traceback(
                stderr, returncode=returncode,
            )
            if working_dir and exc_info and isinstance(exc_info.get("msg"), str):
                exc_info["msg"] = strip_working_dir_from_content(
                    exc_info["msg"], working_dir,
                )

        if exc_type:
            msg = (exc_info or {}).get("msg", "")
            term_out.append(f"{exc_type}: {msg}")
            compact = format_compact_exc_stack(exc_stack)
            if compact:
                term_out.append(compact)
        elif stdout.strip():
            for line in stdout.strip().split("\n"):
                term_out.append(line)

        if metric_lines:
            blob_so_far = "\n".join(term_out)
            extra = [ln for ln in metric_lines if ln.strip() and ln not in blob_so_far]
            if extra:
                term_out.append("\n[metric lines from stdout]\n" + "\n".join(extra))

        term_out = [
            strip_env_token_notes(strip_working_dir_from_content(s, working_dir))
            for s in term_out
        ]
        metric_lines = [
            strip_env_token_notes(strip_working_dir_from_content(s, working_dir))
            for s in metric_lines
        ]

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            exec_time=exec_time,
            term_out=term_out,
            all_term_out=all_term_out,
            metric_lines=metric_lines,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
        )
