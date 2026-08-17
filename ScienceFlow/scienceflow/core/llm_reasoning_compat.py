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

"""Compatibility helpers for reasoning-aware LLM providers."""

from __future__ import annotations


_THINKING_REASONING_ERROR_PATTERNS = (
    "reasoning_content",
    "thinking mode",
)


def is_missing_reasoning_replay_error(exc: BaseException) -> bool:
    """Return True when a provider requires assistant ``reasoning_content`` replay."""

    text = str(exc).lower()
    return all(pattern in text for pattern in _THINKING_REASONING_ERROR_PATTERNS)
