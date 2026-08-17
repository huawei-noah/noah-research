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

"""Pluggable stop/continue policies for :class:`~scienceflow.core.agent.ScienceAgent`.

REPL uses :class:`AutoContinuePolicy` to avoid returning on text-only turns.
LNR solver sessions use :class:`AutoContinuePolicy`; result.md recovery stays in the generic default policy and agent gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_MD_RECOVERY_PROMPT = (
    "You have used all available rounds. "
    "result.md is MISSING. You MUST write it NOW.\n\n"
    "Stay in the task workspace and use relative paths like `dataset/`. "
    "Do NOT use `/mnt/data`, `/kaggle/`, or other paths that do not exist here.\n\n"
    "Create `result.md` with the available file-write mechanism. If this runtime "
    "has no `write` tool, use exactly one bash command that writes `result.md` "
    "without inspecting files first. Use this exact format:\n\n"
    "BRIEF: <one concise sentence, <=50 words, describing concrete solution "
    "changes vs the parent if known: model, features, hyperparameters, target "
    "transforms, validation/refit behavior, or data handling. Do not include "
    "metric/score/delta numbers.>\n\n"
    "## Solution Design\n"
    "<2-4 concise sentences: model family, per-target or multi-output setup, "
    "target transforms, key features/training choices, and final prediction/refit "
    "behavior. No command logs, raw diffs, debugging narrative, or improvement "
    "suggestions.>\n\n"
    "## Results\n"
    "metric_name: <name or unknown>\n"
    "metric_value: <float or null>\n"
    "lower_is_better: <true or false>\n"
    "exec_time: <seconds or 0>\n\n"
    "METRIC: <same value as metric_value>\n\n"
    "If you have no metric, set metric_value and METRIC to null."
)

RESULT_MD_AFTER_SUCCESS_PROMPT = (
    "The last bare `python3 solution.py` run passed run-control checks. "
    "write result.md now using the just-seen bash output, current solution, "
    "and inherited parent context if present. Do not inspect files or run "
    "validation commands first. Use exactly one file-write action to path "
    "`result.md`; if no `write` tool is available, use exactly one bash command "
    "that writes `result.md`.\n\n"
    "Required format:\n\n"
    "BRIEF: <one concise sentence, <=50 words, describing concrete changes vs "
    "parent if known: model, features, hyperparameters, target transforms, "
    "validation/refit behavior, or data handling. No metric/score/delta numbers; "
    "hyperparameter values are allowed when they identify the change.>\n\n"
    "## Solution Design\n"
    "<2-4 concise sentences: model family, per-target or multi-output setup, "
    "target transforms, key features/training choices, and final prediction/refit "
    "behavior. No command logs, raw diffs, debugging narrative, or improvement "
    "suggestions.>\n\n"
    "## Results\n"
    "metric_name: <name or Final Validation Score>\n"
    "metric_value: <float from the last successful run>\n"
    "lower_is_better: <true or false>\n"
    "exec_time: <seconds from the last run, or 0>\n\n"
    "METRIC: <same value as metric_value>\n"
)

RESULT_MD_AFTER_SUCCESS_RETRY_PROMPT = (
    "[Guard] The previous response did not create `result.md`. No other tool "
    "action is allowed after a run-control-ready bare `python3 solution.py` run until "
    "`result.md` exists. Use the existing conversation history and the just-seen "
    "successful bash output; do not inspect files or run commands.\n\n"
    + RESULT_MD_AFTER_SUCCESS_PROMPT
)

_TEXT_ONLY_NUDGE = (
    "Do not just describe plans — use a tool (bash/read/write/edit/grep/glob/ls) "
    "to take the next concrete step right now."
)


@dataclass(frozen=True)
class RoundContext:
    """Context for one policy decision inside ``ScienceAgent.run()``."""

    round_idx: int
    max_steps: int
    assistant_text: str
    workspace_dir: Path


class RunPolicy:
    """Pluggable stop/continue policy for ``ScienceAgent.run()``."""

    def reset_for_new_run(self) -> None:
        """Called at the start of each ``run()``; override for per-run state."""

    def on_text_only(self, ctx: RoundContext) -> tuple[bool, str | None]:
        """LLM returned text without ``tool_calls``.

        Returns:
            ``(False, None)`` — stop ``run()`` and return assistant text.
            ``(True, msg)`` — keep assistant message, inject *msg* as user, continue.
            ``(True, None)`` — continue without injection (unused by default).
        """
        return (False, None)

    def on_tool_calls(self, ctx: RoundContext) -> None:
        """LLM returned executable tool calls; reset any consecutive text-only state."""

    def on_loop_exhausted(self, ctx: RoundContext) -> tuple[int, str | None]:
        """``max_steps`` rounds finished without early text-only return.

        Returns:
            ``(0, None)`` — no recovery; emit limit message.
            ``(N, prompt)`` — inject *prompt* and run up to *N* recovery LLM rounds.
        """
        return (0, None)

    def on_round_start(self, ctx: RoundContext) -> bool:
        """Start of each main-loop round; return False to break out early."""
        return True

    def expand_round_budget_after_tool_error(self, **kwargs: Any) -> int:
        """After a tool finishes, optionally raise the session round cap.

        Parameters include: ``effective_max``, ``initial_max``, ``tool_error``,
        ``tool_error_type`` (``"runtime"`` | ``"protocol"`` -- protocol skips expansion),
        ``debug_dynamic_steps_enabled``, ``debug_step_boost``, ``debug_max_steps_cap``.

        Returns:
            New effective max rounds (must be >= previous ``effective_max``).
        """
        return int(kwargs["effective_max"])


class DefaultPolicy(RunPolicy):
    """Default: text-only stops the run; optional ``result.md`` recovery after limit.

    ``recovery_rounds=3`` matches the historical ``ScienceAgent`` default when not using REPL.
    """

    def __init__(self, recovery_rounds: int = 0) -> None:
        self._recovery_rounds = max(0, int(recovery_rounds))

    def on_loop_exhausted(self, ctx: RoundContext) -> tuple[int, str | None]:
        if self._recovery_rounds <= 0:
            return (0, None)
        result_md = ctx.workspace_dir / "result.md"
        if result_md.exists():
            return (0, None)
        return (self._recovery_rounds, RESULT_MD_RECOVERY_PROMPT)


class AutoContinuePolicy(RunPolicy):
    """REPL/LNR: retry a few times when the model returns text without tools."""

    def __init__(self, max_text_only_retries: int = 2) -> None:
        self._max = max(0, int(max_text_only_retries))
        self._text_only_count = 0

    def reset_for_new_run(self) -> None:
        self._text_only_count = 0

    def on_tool_calls(self, ctx: RoundContext) -> None:
        self._text_only_count = 0

    def on_text_only(self, ctx: RoundContext) -> tuple[bool, str | None]:
        if self._max <= 0:
            return (False, None)
        if self._text_only_count >= self._max:
            self._text_only_count = 0
            return (False, None)
        self._text_only_count += 1
        return (True, _TEXT_ONLY_NUDGE)

    def on_loop_exhausted(self, ctx: RoundContext) -> tuple[int, str | None]:
        return (0, None)
