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

"""Legacy hook surface retained as no-op for REPL/LNR agents."""

from __future__ import annotations

from typing import Any

from deepcraft_core import Message


class LNRHooksMixin:
    def _lnr_maybe_fresh_workspace_hint_first_round(self, round_idx: int) -> None:
        return None

    def _lnr_maybe_periodic_inject_at_round_start(self, round_idx: int) -> None:
        return None

    def _lnr_on_round_complete(self, tool_names: list[str] | None) -> None:
        guard_mgr = getattr(self, "_guard_manager", None)
        if guard_mgr is None:
            return None
        process = getattr(guard_mgr, "process_round_complete", None)
        if not callable(process):
            return None
        for msg in process(tool_names) or []:
            if isinstance(msg, str) and msg.strip():
                self.memory.add_message(Message.user_message(msg.strip()))
        return None

    def _maybe_log_lnr_llm_turn(self, round_idx: int) -> None:
        return None

    def _maybe_log_sft_turn(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _lnr_get_last_assistant_text(self) -> str:
        return ""

    def _lnr_should_attach_search_phase_hard_constraint_reminder(self) -> bool:
        return False

    def _lnr_guard_inject_search_phase_suffix(self) -> str:
        return ""

    def _lnr_maybe_build_search_phase_constraint_correction(self) -> str | None:
        return None

    def _lnr_maybe_inject_requested_skill(self) -> None:
        return None

def _adapt_write_tool_hint_for_bash_only(text: str) -> str:
    """Rewrite legacy write-tool nudges for bash-write REPL sessions."""
    out = str(text or "")
    replacements = {
        "**`write` `solution.py`**": "create or update `solution.py` with a single bash command",
        "`write` `solution.py`": "create or update `solution.py` with a single bash command",
        "write `solution.py`": "create or update `solution.py` with a single bash command",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out
