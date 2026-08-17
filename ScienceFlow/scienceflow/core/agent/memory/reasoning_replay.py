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

"""Provider compatibility for replaying assistant reasoning metadata."""

from __future__ import annotations

import re
from typing import Any

from deepcraft_core import Message


def messages_with_synthetic_reasoning_replay(
    messages: list[Any],
    *,
    purpose: str,
) -> tuple[list[Any], int]:
    """Return a non-mutating request projection with synthetic assistant reasoning.

    Some thinking-mode OpenAI-compatible providers require historical assistant
    messages to include ``reasoning_content`` on replay. Old resumed memories may
    not have this provider-specific metadata. This helper adapts only the outbound
    LLM-visible request and never rewrites stored memory.
    """

    clean_purpose = re.sub(r"\s+", " ", str(purpose or "resume")).strip()
    placeholder = (
        f"Synthetic replay placeholder for {clean_purpose}."
        if clean_purpose
        else "Synthetic replay placeholder."
    )
    out: list[Any] = []
    changed = 0
    for msg in messages:
        if _needs_synthetic_reasoning(msg):
            out.append(_clone_message_with_reasoning_content(msg, placeholder))
            changed += 1
        else:
            out.append(msg)
    return out, changed


def _needs_synthetic_reasoning(message: Any) -> bool:
    if getattr(message, "role", None) != "assistant":
        return False
    existing = getattr(message, "reasoning_content", None)
    return not (isinstance(existing, str) and existing.strip())


def _clone_message_with_reasoning_content(message: Any, reasoning_content: str) -> Message:
    return Message(
        role=getattr(message, "role", "assistant"),
        content=getattr(message, "content", None),
        tool_calls=getattr(message, "tool_calls", None),
        name=getattr(message, "name", None),
        tool_call_id=getattr(message, "tool_call_id", None),
        reasoning_content=reasoning_content,
    )
