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

from typing import Any


def format_magent_recommendations(
    packets: list[dict[str, Any]],
    *,
    max_items: int = 3,
    max_chars: int = 1200,
) -> str:
    rows = [x for x in packets if isinstance(x, dict)]
    if not rows:
        return ""
    header = "magent_recommendations:"
    budget = max(0, int(max_chars or 0))
    if budget <= 0 or len(header) > budget:
        return ""
    lines = [header]
    used = len(header)
    emitted = 0
    for packet in rows:
        if emitted >= max(0, int(max_items or 0)):
            break
        item_lines = _format_magent_recommendation_item(packet)
        if not item_lines:
            continue
        item_text = "\n".join(item_lines)
        additional = 1 + len(item_text)
        if used + additional > budget:
            continue
        lines.extend(item_lines)
        used += additional
        emitted += 1
    if emitted == 0:
        return ""
    return "\n".join(lines)


def _format_magent_recommendation_item(packet: dict[str, Any]) -> list[str]:
    sidecar_id = _compact(packet.get("sidecar_id"), 40)
    quality = _compact(packet.get("quality_gate"), 24)
    parent_summary = _compact(packet.get("summary_for_parent"), 260)
    artifacts = packet.get("artifact_refs") if isinstance(packet.get("artifact_refs"), list) else []
    lines = [f"  - sidecar_id: {sidecar_id}; quality_gate: {quality}"]
    if parent_summary:
        lines.append(f"    parent_summary: {parent_summary}")
    if artifacts:
        refs = []
        for raw in artifacts[:3]:
            if not isinstance(raw, dict):
                continue
            path = _compact(raw.get("path"), 120)
            kind = _compact(raw.get("type") or "artifact", 40)
            if path:
                refs.append(f"{kind}:{path}")
        if refs:
            lines.append("    artifacts: " + "; ".join(refs))
    return lines


def _compact(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text
