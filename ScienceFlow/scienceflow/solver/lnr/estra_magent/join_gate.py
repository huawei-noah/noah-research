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

import json
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.estra_magent.models import ACCEPTED_QUALITY_GATES, JoinPacket


def build_join_packet(
    report: dict[str, Any],
    *,
    parent_worker_id: str,
    inject_parent: bool = True,
    inject_estra: bool = True,
    inject_resource_context: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    quality_gate = str(report.get("quality_gate") or "").strip().lower()
    if quality_gate not in ACCEPTED_QUALITY_GATES:
        return None
    sidecar_id = str(report.get("sidecar_id") or "").strip()
    parent_job_id = str(report.get("parent_job_id") or "").strip()
    if not sidecar_id or not parent_job_id:
        return None

    observations = _compact_list(report.get("observations"), max_items=3)
    recommended = _compact_text(report.get("recommended_next_action"), 180)
    artifact_refs = _artifact_refs(report)
    summary_bits = []
    if observations:
        summary_bits.append("; ".join(observations))
    if recommended:
        summary_bits.append("next: " + recommended)
    if artifact_refs:
        summary_bits.append(f"{len(artifact_refs)} helper artifact(s) ready")
    summary = ". ".join(summary_bits) or "sidecar evidence is ready for review"

    packet = JoinPacket(
        sidecar_id=sidecar_id,
        parent_worker_id=str(parent_worker_id or ""),
        parent_job_id=parent_job_id,
        quality_gate=quality_gate,
        summary_for_parent=summary,
        summary_for_estra=_compact_text(summary, 260),
        artifact_refs=artifact_refs,
        inject={
            "parent_next_prompt": bool(inject_parent),
            "estra_evidence": bool(inject_estra),
            "resource_context": bool(inject_resource_context),
        },
    )
    return packet.to_dict()


def write_join_packet(
    resource_dir: str | Path,
    report: dict[str, Any],
    *,
    parent_worker_id: str,
    inject_parent: bool = True,
    inject_estra: bool = True,
    inject_resource_context: bool = True,
) -> dict[str, Any] | None:
    packet = build_join_packet(
        report,
        parent_worker_id=parent_worker_id,
        inject_parent=inject_parent,
        inject_estra=inject_estra,
        inject_resource_context=inject_resource_context,
    )
    if packet is None:
        return None
    root = Path(resource_dir)
    sidecar_id = str(packet.get("sidecar_id") or "")
    sidecar_dir = root / "sidecars" / sidecar_id
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    packet_path = sidecar_dir / "join_packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_path = root / "estra_magent_join_packets.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(packet, ensure_ascii=True, sort_keys=True) + "\n")
    packet["packet_path"] = packet_path.relative_to(root).as_posix()
    return packet


def load_join_packets(
    resource_dir: str | Path,
    *,
    parent_worker_id: str = "",
    max_packets: int = 3,
) -> list[dict[str, Any]]:
    root = Path(resource_dir)
    sidecars_dir = root / "sidecars"
    if not sidecars_dir.is_dir():
        return []
    candidates = sorted(
        sidecars_dir.glob("*/join_packet.json"),
        key=lambda p: _mtime(p),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    wanted_worker = str(parent_worker_id or "").strip()
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("quality_gate") or "").lower() not in ACCEPTED_QUALITY_GATES:
            continue
        inject = data.get("inject") if isinstance(data.get("inject"), dict) else {}
        if inject and not bool(inject.get("resource_context", True)):
            continue
        packet_worker = str(data.get("parent_worker_id") or "").strip()
        if wanted_worker and packet_worker and packet_worker != wanted_worker:
            continue
        data["packet_path"] = path.relative_to(root).as_posix()
        out.append(data)
        if len(out) >= max(0, int(max_packets or 0)):
            break
    return out


def _artifact_refs(report: dict[str, Any]) -> list[dict[str, str]]:
    sidecar_id = str(report.get("sidecar_id") or "").strip()
    refs: list[dict[str, str]] = []
    raw_artifacts = report.get("useful_artifacts")
    if not isinstance(raw_artifacts, list):
        return refs
    for raw in raw_artifacts[:6]:
        if not isinstance(raw, dict):
            continue
        rel = str(raw.get("path") or "").strip().replace("\\", "/")
        kind = str(raw.get("type") or "artifact").strip() or "artifact"
        if not rel:
            continue
        if not rel.startswith("sidecars/") and sidecar_id:
            rel = f"sidecars/{sidecar_id}/workspace/{rel.lstrip('/')}"
        refs.append({"path": rel, "type": kind})
    return refs


def _compact_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[: max(0, int(max_items))]:
        text = _compact_text(item, 160)
        if text:
            out.append(text)
    return out


def _compact_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
