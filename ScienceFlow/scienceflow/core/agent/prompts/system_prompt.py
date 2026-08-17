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

"""Default system prompt loading and round-budget formatting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from deepcraft_core import Message

if TYPE_CHECKING:
    pass


_PROMPTS_DIR = Path(__file__).resolve().parent
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _load_prompt_file(name: str) -> str:
    path = _PROMPTS_DIR / "L0" / name
    raw = path.read_text(encoding="utf-8")
    text = raw.lstrip("\ufeff")
    if text.startswith("---"):
        text = _FRONTMATTER_RE.sub("", text, count=1)
    return text.strip() + "\n"


def _default_system_prompt(
    *,
    parallel_bash_enabled: bool,
    bash_file_write_mode: bool = False,
) -> str:
    """Default system prompt; tool-batch policy depends on parallel_bash_enabled."""
    if bash_file_write_mode:
        policy_name = (
            "tool_policy_parallel_bash_write.md"
            if parallel_bash_enabled
            else "tool_policy_single_bash_write.md"
        )
        body_name = "default_system_bash_write.md"
    else:
        policy_name = (
            "tool_policy_parallel.md" if parallel_bash_enabled else "tool_policy_single.md"
        )
        body_name = "default_system.md"
    policy = _load_prompt_file(policy_name)
    body = _load_prompt_file(body_name)
    return body.format(policy=policy)


def _code_agent_core_prompt(*, bash_file_write_mode: bool = False) -> str:
    """Stable generic REPL code-agent core prompt."""
    if bash_file_write_mode:
        return _load_prompt_file("code_agent_core_bash_write.md")
    return _load_prompt_file("code_agent_core.md")


# Backwards-compatible alias (single-tool policy; prefer :func:`_default_system_prompt`).
_DEFAULT_SYSTEM = _default_system_prompt(parallel_bash_enabled=False)


class SystemPromptMixin:
    """Build system prompt with optional hook and round budget."""

    def _build_system_prompt(self) -> str:
        base = self.systemPrompt or ""
        if self._system_prompt_hook is not None:
            base = self._system_prompt_hook(base)
        mem = getattr(self, "_memory_ctx", None)
        # Under teleport_mode the system prefix must stay bit-identical across the
        # entire search (KV cache friendly). [Workspace file state] varies as files
        # are written / edited, so we drop it from the system prompt; the same
        # information is conveyed by the tool results (read / write / edit returns).
        _teleport_on = bool(getattr(self, "_teleport_mode", "off") not in ("", "off"))
        _stable_system = bool(getattr(self, "_stable_system_prompt", False))
        if not (_teleport_on or _stable_system) and mem is not None:
            fss = mem.file_state_summary
            if fss:
                base += (
                    "\n\n[Workspace file state — use read/grep/glob/ls tools for full content]\n" + fss
                )
        # Teleport relies on a bit-identical system prompt across A-class
        # continuations. Node/phase budget is already supplied through LNR
        # dynamic user content, so keep Round Budget out of the system prefix.
        if (
            not (_teleport_on or _stable_system)
            and self._round_budget_enabled
            and self._effective_max_steps > 0
        ):
            base += self._format_round_budget()
        return base

    def _build_system_messages(self) -> list[Message]:
        """Return stable system message layers for the next LLM request."""
        runtime = self._build_system_prompt()
        core = str(getattr(self, "_system_prompt_core", "") or "").strip()
        if not core:
            return [Message.system_message(runtime)]
        return [
            Message.system_message(core),
            Message.system_message(runtime),
        ]

    def _format_round_budget(self) -> str:
        # ``_search_round_offset`` (default 0) accumulates rounds spent in prior ScienceAgent
        # instances within the same tree-search run. The formatter remains available for
        # legacy/non-teleport prompts and tests; teleport mode deliberately does not
        # append this dynamic section to the system prompt.
        offset = int(getattr(self, "_search_round_offset", 0) or 0)
        current = self._current_round + 1 + offset
        eff = self._effective_max_steps + offset
        cap = self._round_budget_prompt_cap
        if cap <= 0:
            total = eff
            remaining = max(0, total - current)
            section = (
                f"\n\n## Round Budget\nRound {current}/{total} (remaining: {remaining})\n"
            )
        else:
            soft = min(int(cap), self._effective_max_steps) if self._effective_max_steps > 0 else int(cap)
            soft = max(1, soft)
            soft_with_offset = soft + offset
            display_current = min(current, soft_with_offset)
            display_total = soft_with_offset
            display_remaining = max(0, soft_with_offset - current)
            section = (
                f"\n\n## Round Budget\n"
                f"Round {display_current}/{display_total} (remaining: {display_remaining})\n"
            )
            if current > soft_with_offset and self._effective_max_steps > 0:
                remaining_real = max(0, eff - current)
                if bool(getattr(self, "_include_write_edit_tools", True)):
                    action_hint = "start **`write`** now"
                else:
                    action_hint = "start a bash file-modifying command now"
                section += (
                    f"\n[Extended budget] You have used **{current}** LLM rounds "
                    f"(the soft display cap was {soft_with_offset}). "
                    f"**Real remaining rounds: {remaining_real}** of **{eff}**. "
                    f"If `solution.py` is not written yet, {action_hint} - "
                    "exploration-only rounds waste the real budget.\n"
                )
        return section
