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

"""Append-only evaluator event log helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def append_evaluator_event(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSONL evaluator event, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": SCHEMA_VERSION, **dict(payload or {})}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_evaluator_events(path: Path, *, max_events: int | None = None) -> list[dict[str, Any]]:
    """Read valid evaluator JSONL records, ignoring malformed lines."""

    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_events is not None and max_events >= 0:
        lines = lines[-max_events:]
    out: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def latest_event_by_cache_key(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest event for every non-empty evaluator cache key."""

    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        key = str(event.get("cache_key") or "").strip()
        if key:
            latest[key] = dict(event)
    return latest
