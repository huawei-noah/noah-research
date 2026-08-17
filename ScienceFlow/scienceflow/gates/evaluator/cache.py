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

"""Small file-backed cache helpers for evaluator results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scienceflow.gates.evaluator.event_log import append_evaluator_event, read_evaluator_events


def evaluator_cache_key(
    *,
    backend: str,
    artifact_sha: str,
    command_digest: str = "",
    metric_digest: str = "",
) -> str:
    """Build a stable cache key for an evaluated artifact/config pair."""

    parts = [
        str(backend or "").strip(),
        str(artifact_sha or "").strip(),
        str(command_digest or "").strip(),
        str(metric_digest or "").strip(),
    ]
    raw = "\0".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def digest_mapping(data: dict[str, Any]) -> str:
    """Hash a config mapping deterministically."""

    encoded = json.dumps(data or {}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EvaluatorCache:
    """Append-only JSONL cache indexed by evaluator cache key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._loaded = False
        self._items: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        self._load()
        item = self._items.get(str(key or ""))
        return dict(item) if item else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        clean_key = str(key or "").strip()
        if not clean_key:
            return
        payload = {"event": "evaluator_cache_put", "cache_key": clean_key, "value": dict(value or {})}
        append_evaluator_event(self.path, payload)
        self._items[clean_key] = payload["value"]
        self._loaded = True

    def _load(self) -> None:
        if self._loaded:
            return
        self._items = {}
        for record in read_evaluator_events(self.path):
            key = str(record.get("cache_key") or "").strip()
            value = record.get("value")
            if key and isinstance(value, dict):
                self._items[key] = dict(value)
        self._loaded = True
