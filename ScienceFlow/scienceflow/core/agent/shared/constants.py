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

"""Shared caps and regex for the core agent package."""

from __future__ import annotations

# Workspace logs/interaction.log caps (when interaction_log_full is False).
_WS_LOG_TOOL_CALL_SINGLE_CAP = 1000
_WS_LOG_BODY_CAP = 500
_WS_LOG_TOOL_RESULT_CAP = 2000
_WS_LOG_USER_CAP = 2000
# Truncation cap for logging assistant text in result.md recovery rounds.
_WS_LOG_RECOVERY_CAP = 500
_EMBEDDED_FAIL_STDERR_CAP = 12000
# Append coaching when ``write`` of ``solution.py`` fails with very short ``content``.
_WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD = 200

# Unfenced assistant text must be at least this long to treat as a full Python file.
_UNFENCED_PYTHON_MIN_CHARS = 200
# When trailing prose breaks ast.parse, drop at most this many tail lines before retrying.
_UNFENCED_PYTHON_MAX_TAIL_TRIM = 30

# Tools that may be batched in one LLM round and executed concurrently (read-only).
PARALLEL_READONLY_TOOLS = frozenset({"read", "grep", "glob", "ls"})

# Large write/edit JSON args are redundant in chat memory (tool result + read suffice).
_NEW_STR_MEMORY_EDGE_CHARS = 80
# Head/tail bytes kept when compressing write/edit bodies into LLM memory (not interaction.log).
_MEMORY_COMPRESSION_EDGE_CHARS = 60
# Replace mimicked log placeholders in memory with this (also rejected by WriteTool if echoed).
_SCRUBBED_WRITE_PLACEHOLDER_NOTE = (
    "# <<<REJECTED: placeholder mimicry — DO NOT reuse — call read for real content>>>"
)

# Consecutive successful solution.py writes (without any bash in between) that trigger a nudge
# encouraging the model to run the file before continuing with more edits/writes.
_WRITE_STREAK_NUDGE_THRESHOLD = 3
