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

"""Read per-call token usage from BaseLLM / OnlineLLM after ``ask`` / tool streams."""

from __future__ import annotations

from typing import Any


def read_llm_last_usage_tokens(llm: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from the last API call, best-effort.

    Production ``OnlineLLM`` stores usage on :class:`CallStats` via
    ``_last_call_input_tokens`` / ``_last_call_output_tokens`` (not ``last_tokens_*``).
    """
    def _i(name: str) -> int:
        if not hasattr(llm, name):
            return 0
        v = getattr(llm, name)
        try:
            from unittest.mock import Mock

            if isinstance(v, Mock):
                return 0
        except Exception:
            pass
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    ti, to = _i("_last_call_input_tokens"), _i("_last_call_output_tokens")
    if ti or to:
        return ti, to
    ti, to = _i("last_tokens_input"), _i("last_tokens_output")
    if ti or to:
        return ti, to
    return _i("last_prompt_tokens"), _i("last_completion_tokens")
