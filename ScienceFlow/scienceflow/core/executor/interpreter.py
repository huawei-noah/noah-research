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

"""AsyncInterpreter — MLE-specific wrapper around PythonRunner.

Adds sandbox features (submission rename, data truncation, CPU/GPU affinity)
on top of the generic async Python execution base.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from scienceflow.core.executor.base import ExecutionResult
from scienceflow.core.executor.python_runner import PythonRunner
from scienceflow.core.executor.sandbox import (
    rename_submission,
    replace_model_path_in_code,
    replace_submission_in_code,
    strip_node_id_from_exec_result,
    truncate_data_wrapper,
)
from scienceflow.utils.resource_utils import (
    parse_cpu_list,
    parse_gpu_list,
    parse_gpu_list_auto,
    select_least_used_gpu,
)

logger = logging.getLogger("scienceflow")


_DEFAULT_TIMEOUT_SEC = 1800.0
_FAST_DEBUG_TIMEOUT_SEC = 600.0


class AsyncInterpreter:
    """Execute Python code with MLE sandbox in isolated subprocesses.

    Thin wrapper: delegates actual subprocess management to ``PythonRunner``,
    adds submission/model-path renaming, CPU/GPU affinity,
    and fast-debug data truncation.
    """

    def __init__(self, cfg):
        self.workspace_dir = Path(cfg.workspace_dir).resolve()
        self._runner = PythonRunner(self.workspace_dir)

        self.fast_debug_max_samples = cfg.exec.fast_debug_max_samples
        self.submission_dir = Path(cfg.submission_dir).resolve()
        self.use_filtered = bool(getattr(cfg.exec, "use_filtered", False))
        self.exp_id = getattr(cfg, "exp_id", "")
        self.cpu_list = parse_cpu_list(cfg.exec.cpu_list)
        self.gpu_list = self._resolve_gpu_list(cfg.exec.gpu_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gpu_list(raw: str) -> list[int]:
        """Resolve ``gpu_list`` config value, supporting ``auto:N[:candidates]`` syntax.

        Resolution order for auto mode:
        1. If ``CUDA_VISIBLE_DEVICES`` is already set (e.g. by ``parallel_runner``),
           parse its physical IDs and use them for round-robin — avoids conflicting
           with the parent's device mask.
        2. Otherwise query ``nvidia-smi`` for the N least-loaded GPUs.
        3. Fallback: clear ``CUDA_VISIBLE_DEVICES`` and return an empty list so
           CUDA auto-selects without restriction.
        """
        is_auto, auto_count, auto_candidates = parse_gpu_list_auto(raw)
        if not is_auto:
            return parse_gpu_list(raw)

        env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_cvd:
            try:
                return [int(g.strip()) for g in env_cvd.split(",") if g.strip()]
            except ValueError:
                logger.warning(
                    "interpreter: cannot parse CUDA_VISIBLE_DEVICES=%r; "
                    "falling back to nvidia-smi query",
                    env_cvd,
                )

        chosen = select_least_used_gpu(
            candidates=auto_candidates or None,
            count=auto_count,
        )
        if chosen:
            os.environ["CUDA_VISIBLE_DEVICES"] = chosen
            return [int(g.strip()) for g in chosen.split(",") if g.strip()]

        logger.warning(
            "interpreter: gpu_list=%r but no GPU available; "
            "clearing CUDA_VISIBLE_DEVICES",
            raw,
        )
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        code: str,
        node_id: str,
        process_id: int = 0,
        fast_debug: bool = False,
        *,
        timeout_override: float | None = None,
    ) -> ExecutionResult:
        """Execute code with MLE sandbox, then post-process result."""
        code, timeout = self._apply_sandbox(code, node_id, fast_debug)
        if timeout_override is not None:
            timeout = float(timeout_override)
        pre_code = self._build_pre_code(process_id, fast_debug)
        env = self._build_env(process_id)

        result = await self._runner.run(
            code, timeout=timeout, env=env, pre_code=pre_code,
        )

        strip_node_id_from_exec_result(result, node_id)
        sub_dir = self.submission_dir
        sub_prefix = "submission"
        rename_submission(
            self.workspace_dir, sub_dir, str(node_id), file_prefix=sub_prefix,
        )

        return result

    # ------------------------------------------------------------------
    # MLE-specific helpers
    # ------------------------------------------------------------------

    def _apply_sandbox(
        self, code: str, node_id: str, fast_debug: bool,
    ) -> tuple[str, float]:
        """Apply submission/model-path renaming and pick timeout."""
        if fast_debug:
            code = replace_submission_in_code(code, node_id, "submission")
            code = replace_model_path_in_code(code, node_id, "fast_debug")
            timeout = _FAST_DEBUG_TIMEOUT_SEC
        else:
            code = replace_submission_in_code(code, node_id, "submission")
            code = replace_model_path_in_code(code, node_id, "default")
            timeout = _DEFAULT_TIMEOUT_SEC
        return code, timeout

    def _build_env(self, process_id: int) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.gpu_list:
            gpu_id = self.gpu_list[process_id % len(self.gpu_list)]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return env

    def _build_pre_code(self, process_id: int, fast_debug: bool) -> str:
        parts: list[str] = ["import os\n"]

        if self.cpu_list:
            total = len(self.cpu_list)
            parallel = max(1, len(self.gpu_list)) if self.gpu_list else 1
            per_session = max(1, total // parallel)
            local_id = process_id % parallel
            start = local_id * per_session
            cpu_set = set(self.cpu_list[start : start + per_session])
            if cpu_set:
                parts.append(f"os.sched_setaffinity(0, {cpu_set})\n")

        if self.gpu_list:
            gpu_id = self.gpu_list[process_id % len(self.gpu_list)]
            parts.append(f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"\n')

        if fast_debug and self.fast_debug_max_samples > 0:
            parts.append(f"DEBUG_MAX_SAMPLES = {self.fast_debug_max_samples}\n")
            parts.append(truncate_data_wrapper(self.fast_debug_max_samples))

        return "".join(parts)
