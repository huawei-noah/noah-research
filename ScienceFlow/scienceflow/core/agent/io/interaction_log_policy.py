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

"""Resolved policy for ``interaction.log`` verbosity (level + legacy overrides)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InteractionLogPolicy:
    """Effective logging policy for workspace ``interaction.log``."""

    level: str
    tool_call_full: bool
    llm_stream_to_file: bool
    bash_preview_max_chars: int | None
    tool_result_max_lines: int | None
    write_edit_tool_result_verbose: bool
    # True: ``[tool-result]`` mirrors full stdout (legacy ``interaction_log_full`` / verbose level).
    tool_result_unlimited: bool
    # Bash ``[tool-result]`` / stream: collapse consecutive duplicate lines (see ``bash_output_dedup_*`` in config).
    bash_output_dedup_enabled: bool = True
    bash_output_dedup_min_repeat: int = 3
    bash_output_dedup_summary_prefix: str = "[log-dedup]"

    @classmethod
    def from_legacy_full(cls, full: bool) -> InteractionLogPolicy:
        """Map old ``interaction_log_full`` bool to a policy for tests / callers."""
        if full:
            return resolve_interaction_log_policy(
                level="verbose",
                legacy_full=False,
                legacy_llm_stream=False,
            )
        return resolve_interaction_log_policy(
            level="normal",
            legacy_full=False,
            legacy_llm_stream=False,
        )


def normalize_interaction_log_level(raw: str | None) -> str:
    s = (raw or "normal").strip().lower()
    aliases = {"m": "minimal", "n": "normal", "v": "verbose"}
    if s in aliases:
        s = aliases[s]
    if s not in ("minimal", "normal", "verbose"):
        return "normal"
    return s


def resolve_interaction_log_policy(
    *,
    level: str,
    legacy_full: bool = False,
    legacy_llm_stream: bool = False,
    bash_output_dedup_enabled: bool = True,
    bash_output_dedup_min_repeat: int = 3,
    bash_output_dedup_summary_prefix: str = "[log-dedup]",
) -> InteractionLogPolicy:
    """Combine ``scienceflow_interaction_log_level`` with legacy boolean overrides.

    Legacy:
        ``scienceflow_interaction_log_full`` — OR into ``tool_call_full`` (enable edit/bash bodies).
        ``scienceflow_interaction_log_llm_stream`` — OR into ``llm_stream_to_file`` (enable mirror).
    """
    lv = normalize_interaction_log_level(level)
    if lv == "minimal":
        base = InteractionLogPolicy(
            level=lv,
            tool_call_full=False,
            llm_stream_to_file=False,
            bash_preview_max_chars=80,
            tool_result_max_lines=5,
            write_edit_tool_result_verbose=False,
            tool_result_unlimited=False,
            bash_output_dedup_enabled=bash_output_dedup_enabled,
            bash_output_dedup_min_repeat=bash_output_dedup_min_repeat,
            bash_output_dedup_summary_prefix=bash_output_dedup_summary_prefix,
        )
    elif lv == "verbose":
        base = InteractionLogPolicy(
            level=lv,
            tool_call_full=True,
            llm_stream_to_file=True,
            bash_preview_max_chars=None,
            tool_result_max_lines=None,
            write_edit_tool_result_verbose=True,
            tool_result_unlimited=True,
            bash_output_dedup_enabled=bash_output_dedup_enabled,
            bash_output_dedup_min_repeat=bash_output_dedup_min_repeat,
            bash_output_dedup_summary_prefix=bash_output_dedup_summary_prefix,
        )
    else:
        base = InteractionLogPolicy(
            level="normal",
            tool_call_full=False,
            llm_stream_to_file=False,
            bash_preview_max_chars=120,
            tool_result_max_lines=5,
            write_edit_tool_result_verbose=False,
            tool_result_unlimited=False,
            bash_output_dedup_enabled=bash_output_dedup_enabled,
            bash_output_dedup_min_repeat=bash_output_dedup_min_repeat,
            bash_output_dedup_summary_prefix=bash_output_dedup_summary_prefix,
        )
    return InteractionLogPolicy(
        level=base.level,
        tool_call_full=base.tool_call_full or bool(legacy_full),
        llm_stream_to_file=base.llm_stream_to_file or bool(legacy_llm_stream),
        bash_preview_max_chars=base.bash_preview_max_chars,
        tool_result_max_lines=base.tool_result_max_lines,
        write_edit_tool_result_verbose=base.write_edit_tool_result_verbose or bool(legacy_full),
        tool_result_unlimited=base.tool_result_unlimited or bool(legacy_full),
        bash_output_dedup_enabled=base.bash_output_dedup_enabled,
        bash_output_dedup_min_repeat=base.bash_output_dedup_min_repeat,
        bash_output_dedup_summary_prefix=base.bash_output_dedup_summary_prefix,
    )
