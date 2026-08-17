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

"""Modular tool-guard framework for coaching and safety rails."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from deepcraft_core.tool import ToolResult
from scienceflow.core.agent.tools.bash_utils import (
    _looks_like_write_placeholder_mimicry,
    classify_runtime_error,
)
from scienceflow.core.agent.shared.constants import (
    _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD,
    _WRITE_STREAK_NUDGE_THRESHOLD,
)
from scienceflow.core.mem.memory_context import bash_command_dumps_python_source


def build_explore_streak_inject_message(
    *,
    category_skill_text: str = "",
    streak_n: int = 0,
    max_skill_chars: int = 12000,
) -> str:
    skill = (category_skill_text or "").strip()
    if max_skill_chars > 0 and len(skill) > max_skill_chars:
        skill = skill[:max_skill_chars].rstrip()
    note = f" after {streak_n} read/search-only rounds" if streak_n else ""
    msg = (
        "You have been inspecting files without changing code" + note + ". "
        "Make a concrete implementation or experiment change next, then run it."
    )
    return msg + ("\n\n" + skill if skill else "")


class ToolGuard(ABC):
    """Base class for tool-level protective guards."""

    name: str = "tool_guard"

    @abstractmethod
    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        """Return coaching text to append, or ``None``."""

    def on_round_complete(
        self,
        tool_names: list[str] | None,
        **kwargs: Any,
    ) -> list[str]:
        """Round-level hook returning user messages to inject."""
        return []

    def should_block_budget_expansion(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> bool:
        """Whether this guard blocks dynamic round-budget expansion."""
        return False

    def should_terminate_run(self) -> bool:
        """Whether this guard requests ending the current run early."""
        return False

    def reset(self) -> None:
        """Reset run-scoped state at the start of each run."""


class GuardManager:
    """Run and aggregate multiple tool guards."""

    def __init__(self, guards: list[ToolGuard], *, phase_header: str = "") -> None:
        self._guards = list(guards)
        self._phase_header = (phase_header or "").strip()
        self._write_success_count = 0
        self._edit_success_count = 0
        # True after any successful write/edit in the current LLM tool round (cleared in process_round_complete).
        self._round_write_edit_success: bool = False

    def process_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str:
        """Process one tool result and aggregate coaching text."""
        if not result.error:
            if tool_name == "write":
                self._write_success_count += 1
            elif tool_name == "edit":
                self._edit_success_count += 1
            if tool_name in {"write", "edit"}:
                self._round_write_edit_success = True
        parts: list[str] = []
        for guard in self._guards:
            msg = guard.on_tool_result(tool_name, args, result)
            if isinstance(msg, str) and msg.strip():
                parts.append(msg.strip())
        return "\n\n".join(parts)

    def process_round_complete(self, tool_names: list[str] | None) -> list[str]:
        """Run round-complete hooks and return user-message injections."""
        out: list[str] = []
        np_guard: ToolGuard | None = None
        others: list[ToolGuard] = []
        for guard in self._guards:
            if getattr(guard, "name", "") == "no_progress_hard_stop_guard":
                np_guard = guard
            else:
                others.append(guard)
        round_kw: dict[str, Any] = {"write_edit_success": bool(self._round_write_edit_success)}
        peer_msgs: list[str] = []
        for guard in others:
            msgs = guard.on_round_complete(tool_names, **round_kw)
            if not msgs:
                continue
            for msg in msgs:
                if isinstance(msg, str) and msg.strip():
                    peer_msgs.append(msg.strip())
        out.extend(peer_msgs)
        if np_guard is not None:
            np_msgs = np_guard.on_round_complete(
                tool_names,
                peer_injected=bool(peer_msgs),
                **round_kw,
            )
            if np_msgs:
                for msg in np_msgs:
                    if isinstance(msg, str) and msg.strip():
                        out.append(msg.strip())
            # Next round: suppress hardstop once after any non–no-progress LNR guard inject.
            grace_n = int(
                getattr(np_guard, "grace_rounds_after_peer_inject", 0) or 0,
            )
            if grace_n > 0 and peer_msgs:
                np_guard.arm_cross_round_grace(grace_n)
        self._round_write_edit_success = False
        if not self._phase_header:
            return out
        return [
            f"{self._phase_header}\n\n{msg}" if msg.strip() else msg
            for msg in out
        ]

    def signal_no_progress_hardstop_grace(self, rounds: int = 1) -> None:
        """LNR hooks (e.g. periodic no-solution inject) can defer no-progress hardstop."""
        for guard in self._guards:
            if getattr(guard, "name", "") != "no_progress_hard_stop_guard":
                continue
            g = guard
            n = max(0, int(rounds))
            if n <= 0:
                return
            if hasattr(g, "arm_cross_round_grace"):
                g.arm_cross_round_grace(n)
            return

    def edit_failure_streak(self) -> int:
        """Current consecutive ``edit`` failure streak (0 if none / unknown)."""
        for guard in self._guards:
            if getattr(guard, "name", "") == "edit_failure_guard":
                return int(getattr(guard, "failure_streak", 0))
        return 0

    def any_blocks_budget_expansion(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> bool:
        """Whether any guard blocks dynamic budget expansion for this tool result."""
        for guard in self._guards:
            if guard.should_block_budget_expansion(tool_name, args, result):
                return True
        return False

    def should_terminate_run(self) -> bool:
        """Whether any guard requests ending the current run."""
        for guard in self._guards:
            if guard.should_terminate_run():
                return True
        return False

    def productivity_counters(self) -> dict[str, int]:
        """Successful write/edit counters for current run."""
        return {
            "write_success_count": int(self._write_success_count),
            "edit_success_count": int(self._edit_success_count),
        }

    def reset_all(self) -> None:
        """Reset all guards for a new run."""
        self._write_success_count = 0
        self._edit_success_count = 0
        self._round_write_edit_success = False
        for guard in self._guards:
            guard.reset()


_ERROR_SIGNATURE_LINE_RE = re.compile(
    r"(syntaxerror:|traceback|[a-z_]+error:|exception:)",
    re.IGNORECASE,
)


class BashRepeatFailureGuard(ToolGuard):
    """Detect repeated bash failures with the same signature and inject coaching."""

    name = "bash_repeat_failure_guard"

    def __init__(
        self,
        *,
        soft_threshold: int = 3,
        hard_threshold: int = 5,
        block_budget_threshold: int = 4,
    ) -> None:
        self._soft_threshold = int(soft_threshold)
        self._hard_threshold = int(hard_threshold)
        self._block_budget_threshold = int(block_budget_threshold)
        self._consecutive_failures = 0
        self._last_error_signature: str | None = None
        self._last_coaching_level = 0

    @staticmethod
    def _extract_signature(result: ToolResult) -> str:
        text = str(result.output or "").strip()
        if "[stderr]" in text:
            text = text.split("[stderr]")[-1].strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if _ERROR_SIGNATURE_LINE_RE.search(ln):
                return " ".join(ln.split())[:240]
        if lines:
            return " ".join(lines[-1].split())[:240]
        err = str(result.error or "").strip()
        return " ".join(err.split())[:240]

    def reset(self) -> None:
        self._consecutive_failures = 0
        self._last_error_signature = None
        self._last_coaching_level = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name != "bash":
            return None

        if not result.error:
            self.reset()
            return None

        signature = self._extract_signature(result)
        if not signature:
            signature = "<unknown-bash-error>"

        if signature == self._last_error_signature:
            self._consecutive_failures += 1
        else:
            self._last_error_signature = signature
            self._consecutive_failures = 1
            self._last_coaching_level = 0

        cmd = str(args.get("command") or "").strip()
        heredoc_maybe_truncated = cmd.startswith("cat << 'EOF'") and len(cmd) < 500

        if self._consecutive_failures >= self._hard_threshold and self._last_coaching_level < 2:
            self._last_coaching_level = 2
            extra = (
                "[Guard] Bash failed repeatedly with the same error (5+ times). Stop retrying the same "
                "command. Use `write` to save the full file content, then run `bash` only for execution "
                "(for example `python3 test_fe.py`)."
            )
            if heredoc_maybe_truncated:
                extra += (
                    "\n[Guard] Current heredoc command appears truncated. Avoid heredoc for long code; "
                    "prefer `write` with full content."
                )
            return extra

        if self._consecutive_failures >= self._soft_threshold and self._last_coaching_level < 1:
            self._last_coaching_level = 1
            extra = (
                "[Guard] Bash hit the same error repeatedly (3+ times). Recheck the full command before "
                "retrying. For long code blocks, prefer `write` full-file content over heredoc."
            )
            if heredoc_maybe_truncated:
                extra += (
                    "\n[Guard] Detected a short heredoc payload; it may be truncated and syntactically "
                    "incomplete."
                )
            return extra

        return None

    def should_block_budget_expansion(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> bool:
        return (
            tool_name == "bash"
            and bool(result.error)
            and self._consecutive_failures >= self._block_budget_threshold
        )


class EditFailureGuard(ToolGuard):
    """Escalate guidance when edit fails repeatedly in one run."""

    name = "edit_failure_guard"

    def __init__(
        self,
        *,
        soft_threshold: int = 2,
        hard_threshold: int = 4,
        short_old_str_chars: int = 40,
    ) -> None:
        self._soft_threshold = int(soft_threshold)
        self._hard_threshold = int(hard_threshold)
        self._short_old_str_chars = int(short_old_str_chars)
        self._streak = 0

    @property
    def failure_streak(self) -> int:
        return int(self._streak)

    def reset(self) -> None:
        self._streak = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name != "edit":
            return None
        if not result.error:
            self._streak = 0
            return None
        self._streak += 1
        err = str(result.error or "")
        err_lower = err.strip().lower()
        old_str = str(args.get("old_str") or "")
        parts: list[str] = []
        if err_lower.startswith("edit blocked:"):
            parts.append(
                "[Guard] `edit` was blocked (stale read / file changed). "
                "Run `read` on that path (enough context) before another `edit`. "
                "Only use `write` for a full replacement if you truly need a different file body — "
                "do not spam the same blocked `edit`."
            )
        if "old_str not found" in err.lower() and len(old_str) < self._short_old_str_chars:
            parts.append(
                f"[Guard] old_str is very short (<{self._short_old_str_chars} chars); this is ambiguous. "
                "Re-read the region and pass a longer, unique span (≥2 lines)."
            )
        if self._streak >= self._hard_threshold:
            parts.append(
                "[Guard] `edit` failed 4+ times in a row. Stop retrying guessed `old_str` spans. "
                "Do a full `read` of the file NOW. If the code is already what you need, test with `bash` "
                "— do NOT `write` the same content again. Only call `write` after you verify a real change."
            )
        elif self._streak >= self._soft_threshold:
            parts.append(
                "[Guard] `edit` failed 2+ times in a row. `read` the current file, then use a unique "
                "`old_str` span. If the file is already correct, run it with `bash` rather than rewriting. "
                "Use `write` only when a full-file change is required."
            )
        if not parts:
            return None
        return "\n\n".join(parts)


class WriteRepeatGuard(ToolGuard):
    """Warn when the same path is written with identical content (sha) again in one run,
    or when a path is written again without any ``bash`` in between (second write nudge)."""

    name = "write_repeat_guard"

    def __init__(self) -> None:
        self._write_history: dict[str, list[str]] = {}
        self._written_since_bash: set[str] = set()

    def reset(self) -> None:
        self._write_history.clear()
        self._written_since_bash.clear()

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name == "bash":
            self._written_since_bash.clear()
            return None

        if tool_name != "write" or result.error:
            return None
        out = str(getattr(result, "output", None) or result)
        first_line = out.splitlines()[0] if out else ""
        if first_line.strip().lower().startswith("no-op:"):
            return None
        m = re.search(r"sha256~([0-9a-f]+)", out, re.IGNORECASE)
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel:
            return None

        if rel in self._written_since_bash:
            history = self._write_history.setdefault(rel, [])
            if m:
                history.append(m.group(1))
            return (
                f"[Guard] WARNING: `{rel}` was rewritten without running it first. "
                "Stop rewriting — execute it NOW with bash "
                f"(e.g. `python3 {rel}`). "
                "Only rewrite after you see the output and find a bug."
            )
        self._written_since_bash.add(rel)

        if not m:
            return None
        sha = m.group(1)
        history = self._write_history.setdefault(rel, [])
        if sha in history:
            c = int(history.count(sha)) + 1
            return (
                f"[Guard] WARNING: `{rel}` was written with identical content again "
                f"({c} time(s) with sha256~{sha}). The on-disk file did not change. "
                "Do not keep rewriting. If the file is correct, use bash. "
                "If you need a real change, read the file and edit precisely."
            )
        history.append(sha)
        return None


_RUNTIME_ERR_SIG_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$")


def _bash_runtime_error_signature(err: str) -> str | None:
    """Stable key for the last exception-like line in bash tool error text."""
    t = (err or "").strip()
    if not t:
        return None
    lines = [x.rstrip() for x in t.splitlines() if x.strip()]
    if not lines:
        return None
    for line in reversed(lines[-12:]):
        s = line.strip()
        if _RUNTIME_ERR_SIG_LINE.match(s):
            return s[:800]
    return lines[-1].strip()[:800]


class RepeatedRuntimeErrorGuard(ToolGuard):
    """Inject a strong nudge when the same bash runtime error repeats."""

    name = "repeated_runtime_error_guard"

    def __init__(self, *, same_error_threshold: int = 2) -> None:
        t = int(same_error_threshold)
        self._threshold = 0 if t <= 0 else max(2, t)
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._counts.clear()

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if self._threshold <= 0:
            return None
        if tool_name != "bash":
            return None
        if not result.error:
            self._counts.clear()
            return None
        sig = _bash_runtime_error_signature(str(result.error or ""))
        if not sig:
            self._counts.clear()
            return None
        self._counts[sig] = self._counts.get(sig, 0) + 1
        if self._counts[sig] >= self._threshold:
            self._counts.clear()
            return (
                "[Guard] The same runtime error has occurred again. Stop editing blindly; "
                "reproduce it with a minimal `python3 -c` snippet on a tiny slice of the "
                "failing data/column, confirm the fix, then re-run the full pipeline."
            )
        return None


class WriteNudgeGuard(ToolGuard):
    """Nudge after successful writes to solution.py."""

    name = "write_nudge_guard"

    def __init__(
        self,
        *,
        get_current_solution_sha: Callable[[], str | None],
        get_initial_solution_sha: Callable[[], str | None],
        on_solution_write_success: Callable[[], None] | None = None,
    ) -> None:
        self._get_current_solution_sha = get_current_solution_sha
        self._get_initial_solution_sha = get_initial_solution_sha
        self._on_solution_write_success = on_solution_write_success
        self._consecutive_solution_writes = 0

    def reset(self) -> None:
        self._consecutive_solution_writes = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name == "bash":
            self._consecutive_solution_writes = 0
            return None
        if tool_name != "write" or result.error:
            return None
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel.endswith("solution.py"):
            return None
        if self._on_solution_write_success is not None:
            self._on_solution_write_success()
        self._consecutive_solution_writes += 1
        current_sha = self._get_current_solution_sha()
        initial_sha = self._get_initial_solution_sha()
        if current_sha is not None and initial_sha is not None and current_sha == initial_sha:
            return (
                "[Guard] WARNING: solution.py is identical to the inherited version — you must modify "
                "it before testing. After your changes, verify with: `python3 solution.py` "
                "(or pass argparse knobs like `--epochs 1` for a faster smoke pass)."
            )
        if self._consecutive_solution_writes >= _WRITE_STREAK_NUDGE_THRESHOLD:
            short_sha = current_sha[:16] if current_sha else "?"
            return (
                f"[Guard] You have written solution.py {self._consecutive_solution_writes} times "
                "in a row without running it. "
                "This is fine if you are tuning hyperparameters or layering small refinements — "
                f"the on-disk file is complete (sha256~{short_sha}). "
                "Before further `write`/`edit`, consider running `python3 solution.py` to test it "
                "end-to-end (for a faster smoke pass, use the argparse knobs your solution exposes, "
                "e.g. `python3 solution.py --epochs 1` or `python3 solution.py --num_samples 20`). "
                "Real exec evidence guides the next change far better than re-reads or guesses. "
                "Earlier `# [WRITE_OK …]` lines in chat history are memory-compression stubs, not the disk content."
            )
        return None


class WriteFailureGuard(ToolGuard):
    """Coaching for failed writes of solution.py."""

    name = "write_failure_guard"

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name != "write" or not result.error:
            return None
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel.endswith("solution.py"):
            return None

        err = str(result.error or "")
        raw = args.get("content")
        extra_parts: list[str] = []
        if isinstance(raw, str) and _looks_like_write_placeholder_mimicry(raw):
            extra_parts.append(
                "[Guard] Your `write` **`content`** looks like a **length placeholder** (e.g. "
                "`<N chars>`) copied from chat metadata — that is **not** valid Python. "
                "The `content` field must contain the **full source file text**, not a size summary."
            )
        if "Syntax check failed" in err:
            extra_parts.append(
                "[Guard] Next `write` to solution.py must pass a **complete, syntactically valid** "
                "Python file in the **`content`** field — do not resend a tiny placeholder. "
                "The runnable code must appear in the tool argument, not only in your assistant text."
            )
        if isinstance(raw, str) and len(raw) < _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD:
            extra_parts.append(
                "[Guard] The submitted `content` looks **too short** for a full solution — "
                "confirm you are passing the entire file body in `content`."
            )
        if not extra_parts:
            return None
        return "\n\n".join(extra_parts)


def _distinct_runtime_key(err_type: str | None) -> str:
    return err_type if err_type is not None else "__unclassified__"


def _dtype_diagnostic_hint(output: str) -> str:
    """Append a concrete pandas dtype inspection command when output looks dtype-related."""
    s = (output or "").lower()
    if not any(
        k in s
        for k in (
            "dtypes must be",
            "intcastingnanerror",
            "could not convert string to float",
        )
    ):
        return ""
    return (
        "\n\n[Guard] **dtype / casting hint**: run a lightweight inspection before patching, e.g.\n"
        '`python3 -c "import pandas as pd; df=pd.read_csv(\'dataset/train.csv\', nrows=200); '
        "print(df.dtypes.value_counts()); print(df.select_dtypes('object').head())\"`"
    )


class RuntimeErrorGuard(ToolGuard):
    """Coaching for repeated runtime crashes while running solution.py."""

    name = "runtime_error_guard"

    def __init__(
        self,
        *,
        rotating_runtime_error_threshold: int = 3,
        rotating_distinct_types_min: int = 2,
    ) -> None:
        self._rotating_threshold = int(rotating_runtime_error_threshold)
        self._rotating_distinct_min = int(rotating_distinct_types_min)
        self._consecutive_solution_runtime_errors = 0
        self._last_solution_runtime_error_type: str | None = None
        self._any_type_streak = 0
        self._distinct_types_seen: set[str] = set()
        self._rotating_emitted = False

    def reset(self) -> None:
        self._consecutive_solution_runtime_errors = 0
        self._last_solution_runtime_error_type = None
        self._any_type_streak = 0
        self._distinct_types_seen = set()
        self._rotating_emitted = False

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name != "bash":
            return None

        cmd = (args.get("command") or "").strip()
        low = cmd.lower()

        if not result.error:
            if "solution.py" in cmd and ("python3" in low or "python" in low):
                self.reset()
            return None

        if "solution.py" not in cmd:
            return None
        if "python3" not in low and "python" not in low:
            return None

        output = str(result.output or "")
        err_type = classify_runtime_error(output)
        dkey = _distinct_runtime_key(err_type)

        self._any_type_streak += 1
        self._distinct_types_seen.add(dkey)

        if err_type == self._last_solution_runtime_error_type:
            self._consecutive_solution_runtime_errors += 1
        else:
            self._consecutive_solution_runtime_errors = 1
            self._last_solution_runtime_error_type = err_type

        err_label = err_type if err_type is not None else "Unknown"

        type_hint = ""
        if err_type in ("IndexError",):
            type_hint = (
                "\n- For **IndexError** / shape issues: verify batch dimensions, collate, and "
                "DataLoader output shapes."
            )
        elif err_type == "TypeError":
            type_hint = (
                "\n- For **TypeError**: verify function signatures, argument types, and calls "
                "match definitions."
            )
        elif err_type == "KeyError":
            type_hint = (
                "\n- For **KeyError**: verify DataFrame columns, dict keys, and file/schema "
                "consistency."
            )
        elif err_type == "IntCastingNaNError":
            type_hint = (
                "\n- For **IntCastingNaNError**: check for NaNs / strings in columns you cast to int; "
                "use nullable integer dtypes or fill/drop before astype."
            )

        coaching = (
            f"[Guard] `solution.py` crashed with **{err_label}**. "
            "Before fixing, **inspect the solution broadly**:\n"
            "1. Review `solution.py` enough to check that data shapes, "
            "types, and variable names are consistent across all functions.\n"
            "2. Verify the data pipeline: loading -> preprocessing -> model input -> "
            "model output -> postprocessing -> submission.\n"
            "3. Check tensor/array dimensions at each stage boundary.\n"
            "4. Only then fix the specific crash site with awareness of the full context.\n"
            "Use compact source inspection plus the available file-change mechanism; "
            "do not guess at a fix from the traceback alone."
            f"{type_hint}"
        )
        coaching += _dtype_diagnostic_hint(output)

        rotating_ready = (
            not self._rotating_emitted
            and self._any_type_streak >= self._rotating_threshold
            and len(self._distinct_types_seen) >= self._rotating_distinct_min
        )
        if rotating_ready:
            self._rotating_emitted = True
            coaching += (
                "\n\n[Guard] Rotating errors detected: multiple consecutive failures on "
                "`python3 solution.py` with **different exception types**. Stop patching blindly — "
                "run a small diagnostic bash first (inspect data shapes/dtypes), then fix preprocessing "
                "/ categorical columns before retraining."
            )

        if self._consecutive_solution_runtime_errors >= 2:
            coaching += (
                "\n\n[Guard] This is the 2nd consecutive runtime crash. You MUST inspect "
                "`solution.py` broadly before the next change; do NOT guess at a fix from "
                "the traceback alone."
            )
        return coaching


class NoSuccessfulSolutionRunGuard(ToolGuard):
    """Coaching when many rounds pass without a successful ``python3 solution.py`` run."""

    name = "no_successful_solution_run_guard"

    def __init__(
        self,
        *,
        soft_threshold: int = 6,
        hard_threshold: int = 12,
    ) -> None:
        self._soft_threshold = int(soft_threshold)
        self._hard_threshold = int(hard_threshold)
        self._rounds_since_last_success = 0
        self._ever_attempted_solution_run = False
        self._had_success_this_round = False
        self._emitted_soft = False
        self._emitted_hard = False

    @staticmethod
    def _looks_like_solution_py_run(cmd: str) -> bool:
        s = (cmd or "").strip()
        if not s or "solution.py" not in s:
            return False
        low = s.lower()
        return "python3" in low or "python " in low

    def reset(self) -> None:
        self._rounds_since_last_success = 0
        self._ever_attempted_solution_run = False
        self._had_success_this_round = False
        self._emitted_soft = False
        self._emitted_hard = False

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name != "bash":
            return None
        cmd = str(args.get("command") or "").strip()
        if not self._looks_like_solution_py_run(cmd):
            return None
        self._ever_attempted_solution_run = True
        if not result.error:
            self._had_success_this_round = True
        return None

    def on_round_complete(
        self,
        tool_names: list[str] | None,
        **kwargs: Any,
    ) -> list[str]:
        if not self._ever_attempted_solution_run:
            return []
        msgs: list[str] = []
        if self._had_success_this_round:
            self._rounds_since_last_success = 0
            self._emitted_soft = False
            self._emitted_hard = False
        else:
            self._rounds_since_last_success += 1
            if self._rounds_since_last_success >= self._hard_threshold and not self._emitted_hard:
                self._emitted_hard = True
                self._emitted_soft = True
                msgs.append(
                    "[Guard] Long window without a successful `python3 solution.py` run. "
                    "Strongly consider `write` a **minimal runnable** `solution.py` first (tiny data "
                    "slice / stub model), verify it runs end-to-end, then expand."
                )
            elif self._rounds_since_last_success >= self._soft_threshold and not self._emitted_soft:
                self._emitted_soft = True
                n = self._rounds_since_last_success
                msgs.append(
                    f"[Guard] No successful `python3 solution.py` in the last {n} round(s). "
                    "Verify the pipeline with a minimal command first, e.g. "
                    "`NROWS=20 python3 solution.py` or "
                    '`python3 -c \'from solution import *; print("ok")\'`.'
                )
        self._had_success_this_round = False
        return msgs


class SingleReadStreakGuard(ToolGuard):
    """Inject nudge when agent keeps doing one read per round."""

    name = "single_read_streak_guard"

    def __init__(
        self,
        *,
        threshold: int,
        parallel_llm_tool_calls_enabled: bool,
    ) -> None:
        self._threshold = int(threshold)
        self._parallel_llm_tool_calls_enabled = bool(parallel_llm_tool_calls_enabled)
        self._streak = 0

    def reset(self) -> None:
        self._streak = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        return None

    def on_round_complete(
        self,
        tool_names: list[str] | None,
        **kwargs: Any,
    ) -> list[str]:
        if self._threshold <= 0 or not self._parallel_llm_tool_calls_enabled:
            self._streak = 0
            return []
        if tool_names and len(tool_names) == 1 and tool_names[0] == "read":
            self._streak += 1
            if self._streak >= self._threshold:
                self._streak = 0
                return [
                    "[Guard] Detected repeated single-`read` rounds. Stop read-only paging; "
                    "use a compact bash/search summary for the remaining question or make the "
                    "next targeted change and validate it.",
                ]
            return []
        self._streak = 0
        return []


class BashPythonSourceDumpGuard(ToolGuard):
    """Nudge when bash is used to slice/dump Python source (cat/head/sed) instead of ``read``."""

    name = "bash_python_source_dump_guard"

    def __init__(self, *, parallel_threshold: int = 2, consecutive_rounds: int = 2) -> None:
        self._parallel_threshold = max(2, int(parallel_threshold))
        self._consecutive_rounds = max(2, int(consecutive_rounds))
        self._round_bash_cmds: list[str] = []
        self._single_bash_code_streak = 0

    def reset(self) -> None:
        self._round_bash_cmds.clear()
        self._single_bash_code_streak = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        if tool_name != "bash" or result.error:
            return None
        self._round_bash_cmds.append(str(args.get("command") or ""))
        return None

    def on_round_complete(
        self,
        tool_names: list[str] | None,
        **kwargs: Any,
    ) -> list[str]:
        cmds = self._round_bash_cmds
        self._round_bash_cmds = []

        if not tool_names:
            return []
        if not all(t == "bash" for t in tool_names):
            self._single_bash_code_streak = 0
            return []
        n_bash = len(tool_names)
        if n_bash == 0 or len(cmds) != n_bash:
            self._single_bash_code_streak = 0
            return []
        flags = [bash_command_dumps_python_source(c) for c in cmds]
        if not any(flags):
            self._single_bash_code_streak = 0
            return []

        if n_bash >= self._parallel_threshold and all(flags):
            self._single_bash_code_streak = 0
            return [
                "[Guard] Multiple **bash** source dumps in one turn (e.g. `cat`/`head`/`sed` slices of "
                "`*.py`). Replace raw dumps with one compact bounded source-inspection command "
                "or a targeted `read` only when exact line-numbered context is needed.",
            ]

        if n_bash == 1 and flags[0]:
            self._single_bash_code_streak += 1
        else:
            self._single_bash_code_streak = 0
            return []
        if self._single_bash_code_streak >= self._consecutive_rounds:
            self._single_bash_code_streak = 0
            return [
                "[Guard] Repeated **bash** dumps of `*.py` / `solution.py`. Stop chaining `cat`/`head`/`sed` "
                "across rounds. Use compact search/snippets, targeted `read` only for exact anchors, "
                "or move on to a code change and validation.",
            ]
        return []


class ExploreStreakGuard(ToolGuard):
    """Inject explore-streak guidance when no coding tools are used for many rounds."""

    name = "explore_streak_guard"

    def __init__(
        self,
        *,
        threshold: int,
        get_current_solution_sha: Callable[[], str | None],
        get_initial_solution_sha: Callable[[], str | None],
        category_skill_text: str,
        max_skill_chars: int,
    ) -> None:
        self._threshold = int(threshold)
        self._get_current_solution_sha = get_current_solution_sha
        self._get_initial_solution_sha = get_initial_solution_sha
        self._category_skill_text = str(category_skill_text or "")
        self._max_skill_chars = int(max_skill_chars)
        self._streak = 0

    def reset(self) -> None:
        self._streak = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        return None

    def on_round_complete(
        self,
        tool_names: list[str] | None,
        **kwargs: Any,
    ) -> list[str]:
        if self._threshold <= 0 or tool_names is None or not tool_names:
            return []
        write_edit_success = bool(kwargs.get("write_edit_success"))
        if write_edit_success:
            self._streak = 0
            return []
        if any(t in {"write", "edit"} for t in tool_names):
            # Coding tool ran but did not succeed — not an explore-only streak.
            return []
        if all(t in {"bash", "read", "grep"} for t in tool_names):
            self._streak += 1
        else:
            self._streak = 0
            return []
        current_sha = self._get_current_solution_sha()
        initial_sha = self._get_initial_solution_sha()
        sha_changed = (
            (current_sha is not None and initial_sha is None)
            or (current_sha is not None and initial_sha is not None and current_sha != initial_sha)
            or (current_sha is None and initial_sha is not None)
        )
        if sha_changed:
            self._streak = 0
            return []
        if self._streak < self._threshold:
            return []
        msg = build_explore_streak_inject_message(
            category_skill_text=self._category_skill_text,
            streak_n=self._streak,
            max_skill_chars=self._max_skill_chars,
        )
        self._streak = 0
        return [msg]


class NoProgressHardStopGuard(ToolGuard):
    """Hard-stop a run after too many no-progress rounds.

    Two-stage behavior: when the no-progress streak first reaches *threshold*, inject
    a user-visible warning and reset the streak so the next LLM turn can see it.
    Only if the streak reaches *threshold* again (still no write/edit / SHA / metric
    progress) do we set ``should_terminate_run``.
    """

    name = "no_progress_hard_stop_guard"

    _WARN_USER_MSG = (
        "[Guard] No meaningful progress detected for multiple rounds "
        "(no successful file change, unchanged solution.py, no new embedded metric). "
        "Modify `solution.py` with the available file-change mechanism on the next round. "
        "If this stalls again after this warning, the run will stop."
    )

    _FINAL_USER_MSG = (
        "[Guard] No meaningful progress detected again after the prior warning. "
        "Stopping this run early to avoid repetitive exploration."
    )

    def __init__(
        self,
        *,
        threshold: int,
        get_current_solution_sha: Callable[[], str | None],
        get_initial_solution_sha: Callable[[], str | None],
        get_embedded_metric_token: Callable[[], str | None],
        grace_rounds_after_peer_inject: int = 1,
    ) -> None:
        self._threshold = int(threshold)
        self._get_current_solution_sha = get_current_solution_sha
        self._get_initial_solution_sha = get_initial_solution_sha
        self._get_embedded_metric_token = get_embedded_metric_token
        self.grace_rounds_after_peer_inject = max(0, int(grace_rounds_after_peer_inject))
        self._streak = 0
        self._terminate = False
        self._warning_issued = False
        self._last_metric_token: str | None = None
        self._cross_round_grace_remaining = 0

    def arm_cross_round_grace(self, rounds: int) -> None:
        """Skip streak accumulation for the next *rounds* ``on_round_complete`` calls."""
        n = max(0, int(rounds))
        if n <= 0:
            return
        self._cross_round_grace_remaining = max(self._cross_round_grace_remaining, n)

    def reset(self) -> None:
        self._streak = 0
        self._terminate = False
        self._warning_issued = False
        self._last_metric_token = self._get_embedded_metric_token()
        self._cross_round_grace_remaining = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> str | None:
        return None

    def on_round_complete(
        self,
        tool_names: list[str] | None,
        **kwargs: Any,
    ) -> list[str]:
        if self._threshold <= 0 or tool_names is None or not tool_names:
            return []
        if self._terminate:
            return []

        if self._cross_round_grace_remaining > 0:
            self._cross_round_grace_remaining -= 1
            self._streak = 0
            self._last_metric_token = self._get_embedded_metric_token()
            return []

        peer_injected = bool(kwargs.get("peer_injected"))
        if peer_injected:
            self._streak = 0
            self._last_metric_token = self._get_embedded_metric_token()
            return []

        current_sha = self._get_current_solution_sha()
        initial_sha = self._get_initial_solution_sha()
        sha_changed = (
            (current_sha is not None and initial_sha is None)
            or (current_sha is not None and initial_sha is not None and current_sha != initial_sha)
            or (current_sha is None and initial_sha is not None)
        )

        metric_token = self._get_embedded_metric_token()
        metric_progress = metric_token is not None and metric_token != self._last_metric_token
        self._last_metric_token = metric_token

        write_edit_success = bool(kwargs.get("write_edit_success"))
        made_progress = bool(sha_changed or metric_progress or write_edit_success)

        if not made_progress:
            self._streak += 1
        else:
            self._streak = 0
            self._warning_issued = False

        if self._streak < self._threshold:
            return []

        if not self._warning_issued:
            self._warning_issued = True
            self._streak = 0
            return [self._WARN_USER_MSG]

        self._terminate = True
        return [self._FINAL_USER_MSG]

    def should_terminate_run(self) -> bool:
        return self._terminate
