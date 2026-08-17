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

import hashlib
import json
import math
from typing import Any


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _artifact_path(progress: dict[str, Any]) -> str:
    recoverability = progress.get("recoverability") if isinstance(progress.get("recoverability"), dict) else {}
    path = str(recoverability.get("artifact_path") or "").strip()
    if path:
        return path
    updates = progress.get("artifact_updates") if isinstance(progress.get("artifact_updates"), list) else []
    for item in reversed(updates):
        if isinstance(item, dict) and str(item.get("path") or "").strip():
            return str(item.get("path") or "").strip()
    return ""


def build_kill_intent_snapshot(
    progress: dict[str, Any] | None,
    metric: dict[str, Any] | None,
    *,
    kill_basis: str = "",
) -> dict[str, Any]:
    progress_facts = dict(progress or {})
    metric_facts = dict(metric or {})
    payload = {
        "kill_basis": str(kill_basis or "").strip().lower(),
        "metric_scope_key": str(metric_facts.get("metric_scope_key") or progress_facts.get("metric_scope_key") or ""),
        "metric_history_line_count": _int_or_zero(progress_facts.get("metric_history_line_count")),
        "metric_value": _float_or_none(metric_facts.get("value")),
        "metric_status": str(metric_facts.get("status") or ""),
        "metric_useful": metric_facts.get("useful") if isinstance(metric_facts.get("useful"), bool) else None,
        "recoverable_artifact_on_disk": _as_bool(progress_facts.get("recoverable_artifact_on_disk")),
        "artifact_path": _artifact_path(progress_facts),
        "progress_unit": str(progress_facts.get("progress_unit") or ""),
        "progress_units_done": _float_or_none(progress_facts.get("progress_units_done")),
        "progress_units_total": _float_or_none(progress_facts.get("progress_units_total")),
        "structured_progress_recent": _as_bool(progress_facts.get("structured_progress_recent")),
        "known_stage": str(progress_facts.get("known_stage") or ""),
        "near_submission": _as_bool(progress_facts.get("near_submission")),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    payload["snapshot_id"] = hashlib.sha1(serialized.encode("utf-8", errors="replace")).hexdigest()[:16]
    return payload


def revalidate_kill_intent(
    original: dict[str, Any] | None,
    fresh: dict[str, Any] | None,
    *,
    hard_safety: bool = False,
) -> dict[str, Any]:
    original_facts = dict(original or {})
    fresh_facts = dict(fresh or {})
    if hard_safety:
        return {"allow_kill": True, "reason": "hard_safety", "protective_changes": []}
    if not original_facts.get("snapshot_id"):
        return {"allow_kill": True, "reason": "legacy_snapshot_missing", "protective_changes": []}

    changes: list[str] = []
    rebased_changes: list[str] = []
    value_stagnation = str(original_facts.get("kill_basis") or "").strip().lower() == "value_stagnation"
    old_scope = str(original_facts.get("metric_scope_key") or "")
    new_scope = str(fresh_facts.get("metric_scope_key") or "")
    if old_scope and new_scope and old_scope != new_scope:
        changes.append("metric_scope_changed")

    old_value = _float_or_none(original_facts.get("metric_value"))
    new_value = _float_or_none(fresh_facts.get("metric_value"))
    metric_advanced = bool(
        fresh_facts.get("metric_useful") is True
        and (
            _int_or_zero(fresh_facts.get("metric_history_line_count"))
            > _int_or_zero(original_facts.get("metric_history_line_count"))
            or (new_value is not None and new_value != old_value)
        )
    )
    if metric_advanced:
        changes.append("useful_metric_advanced")

    old_recoverable = _as_bool(original_facts.get("recoverable_artifact_on_disk"))
    new_recoverable = _as_bool(fresh_facts.get("recoverable_artifact_on_disk"))
    old_artifact = str(original_facts.get("artifact_path") or "")
    new_artifact = str(fresh_facts.get("artifact_path") or "")
    if new_recoverable and (not old_recoverable or (new_artifact and new_artifact != old_artifact)):
        target = rebased_changes if value_stagnation else changes
        target.append("recoverable_checkpoint_created")

    old_done = _float_or_none(original_facts.get("progress_units_done"))
    new_done = _float_or_none(fresh_facts.get("progress_units_done"))
    old_total = _float_or_none(original_facts.get("progress_units_total"))
    new_total = _float_or_none(fresh_facts.get("progress_units_total"))
    same_total = bool(
        old_total is not None
        and new_total is not None
        and abs(new_total - old_total) <= max(1e-9, abs(old_total) * 1e-9)
    )
    same_progress_contract = bool(
        str(original_facts.get("progress_unit") or "")
        and str(original_facts.get("progress_unit") or "") == str(fresh_facts.get("progress_unit") or "")
        and same_total
    )
    if (
        same_progress_contract
        and old_done is not None
        and new_done is not None
        and new_done > old_done
        and _as_bool(fresh_facts.get("structured_progress_recent"))
    ):
        target = rebased_changes if value_stagnation else changes
        target.append("structured_progress_advanced")
    if _as_bool(fresh_facts.get("near_submission")) and not _as_bool(original_facts.get("near_submission")):
        changes.append("deliverable_became_ready")

    return {
        "allow_kill": not changes,
        "reason": (
            "stale_kill_intent"
            if changes
            else "kill_intent_rebased"
            if rebased_changes
            else "kill_intent_current"
        ),
        "protective_changes": changes,
        "rebased_changes": rebased_changes,
        "kill_basis": str(original_facts.get("kill_basis") or ""),
        "original_snapshot_id": str(original_facts.get("snapshot_id") or ""),
        "fresh_snapshot_id": str(fresh_facts.get("snapshot_id") or ""),
    }
