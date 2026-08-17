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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LnrContextMemoryPolicyResult:
    original_max_messages: int
    effective_max_messages: int
    changed: bool
    reason: str


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def ensure_lnr_context_memory_budget(cfg: Any, lhr: Any) -> LnrContextMemoryPolicyResult:
    """Keep LNR chat memory from truncating before the context compactor can run."""
    original = max(1, _safe_int(getattr(cfg, "max_messages", None), 100))
    if not bool(getattr(lhr, "compact_on_context_limit", False)):
        return LnrContextMemoryPolicyResult(
            original_max_messages=original,
            effective_max_messages=original,
            changed=False,
            reason="compact_on_context_limit_disabled",
        )

    floor = _safe_int(getattr(lhr, "context_limit_min_messages", None), 0)
    if floor <= 0:
        return LnrContextMemoryPolicyResult(
            original_max_messages=original,
            effective_max_messages=original,
            changed=False,
            reason="context_limit_min_messages_disabled",
        )

    effective = max(original, floor)
    if effective != original:
        setattr(cfg, "max_messages", effective)
    return LnrContextMemoryPolicyResult(
        original_max_messages=original,
        effective_max_messages=effective,
        changed=effective != original,
        reason="context_limit_memory_floor" if effective != original else "already_sufficient",
    )
