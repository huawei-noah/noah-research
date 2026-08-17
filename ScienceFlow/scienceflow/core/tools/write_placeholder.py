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

"""Detect when models paste log/memory placeholders as ``write`` ``content``."""

from __future__ import annotations

import re

# Legacy chat metadata used ``<N chars>``; interaction.log uses natural-language placeholders.
INTERACTION_LOG_WRITE_BODY_OMITTED_RE = re.compile(
    r"^\[interaction-log: write body omitted; \d+ chars applied by tool\]$",
    re.IGNORECASE,
)
CHAT_MEMORY_WRITE_PAYLOAD_OMITTED_RE = re.compile(
    r"^# \[chat-memory: write payload omitted \(\d+ bytes\); "
    r"following write tool result confirms on-disk file\.\]$",
    re.IGNORECASE,
)
# Legacy "DO NOT COPY" variants — kept for backward-compat with old chat history / node resumes.
NEW_DISPLAY_STUB_INTERACTION_RE = re.compile(
    r"^\[DO NOT COPY - display-only stub: real \d+-char body applied to disk\]$",
    re.IGNORECASE,
)
NEW_DISPLAY_STUB_CHAT_MEMORY_RE = re.compile(
    r"^# \[DO NOT COPY - display-only stub: real \d+-byte solution body on disk;",
    re.IGNORECASE,
)
# New "WRITE_OK memory-compression" variants — positive framing with embedded sha256.
WRITE_OK_INTERACTION_RE = re.compile(
    r"^\[WRITE_OK memory-compression: \d+ chars",
    re.IGNORECASE,
)
WRITE_OK_CHAT_MEMORY_RE = re.compile(
    r"^# \[WRITE_OK memory-compression: \d+ bytes",
    re.IGNORECASE,
)
WRITE_PLACEHOLDER_ANCHOR_RE = re.compile(
    r"^(?:"
    r"<\d+\s+chars>"
    r"|\[file content omitted - \d+ chars written to disk\]"
    r"|<<<WRITE_PLACEHOLDER:[^>]*>>>"
    r"|<<<WRITE_CONTENT:[^>]*>>>"
    r")$",
    re.IGNORECASE,
)

# Memory compression uses this prefix in tool-call args shown to the LLM; must not be written to disk.
MEMORY_COMPRESSED_MARKER = "# [MEMORY_COMPRESSED:"
MEMORY_COMPRESSED_MARKER_V2 = "# <<<MEMORY_COMPRESSED:"
# Success marker in compressed write args (replaces old MEMORY_COMPRESSED text).
WRITE_OK_MARKER = "# <<<WRITE_OK:"

# Back-compat alias for tests / external imports.
WRITE_PLACEHOLDER_MIMICRY_RE = WRITE_PLACEHOLDER_ANCHOR_RE

# Reject memory-compressed snippets pasted as a "full file" (head+marker+tail is ~200–400 chars).
_MAX_MEMORY_LIKE_WRITE_CHARS = 768


def looks_like_write_placeholder_mimicry(content: str) -> bool:
    """True if *content* is a known placeholder / memory-compressed snippet, not a real file body."""
    if not isinstance(content, str):
        return False
    s = content.strip()
    if not s:
        return False
    if WRITE_PLACEHOLDER_ANCHOR_RE.match(s):
        return True
    if INTERACTION_LOG_WRITE_BODY_OMITTED_RE.match(s):
        return True
    if CHAT_MEMORY_WRITE_PAYLOAD_OMITTED_RE.match(s):
        return True
    if NEW_DISPLAY_STUB_INTERACTION_RE.match(s):
        return True
    if NEW_DISPLAY_STUB_CHAT_MEMORY_RE.match(s):
        return True
    if WRITE_OK_INTERACTION_RE.match(s):
        return True
    if WRITE_OK_CHAT_MEMORY_RE.match(s):
        return True
    if s.startswith("# [REJECTED: placeholder mimicry") or s.startswith("# <<<REJECTED:"):
        return True
    if (
        (
            MEMORY_COMPRESSED_MARKER in s
            or MEMORY_COMPRESSED_MARKER_V2 in s
            or WRITE_OK_MARKER in s
        )
        and len(s) <= _MAX_MEMORY_LIKE_WRITE_CHARS
    ):
        return True
    return False
