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

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.stage.score_summary import (
    ScoreRecord,
    normalize_score_record_directions,
    score_record_from_stage_performance_row,
)


@dataclass(frozen=True)
class PeerRouteEvidence:
    text: str
    peer_best_worker: str = ""
    peer_best_stage: str = ""
    peer_best_metric: float | None = None
    current_best_stage: str = ""
    current_best_metric: float | None = None
    peer_delta_to_current: float | None = None
    evidence_strength: str = "none"
    peer_record_count: int = 0

    @property
    def present(self) -> bool:
        return bool(self.text.strip())

    def audit_fields(self) -> dict[str, Any]:
        return {
            "peer_route_evidence_present": self.present,
            "peer_best_worker": self.peer_best_worker,
            "peer_best_stage": self.peer_best_stage,
            "peer_best_metric": self.peer_best_metric,
            "peer_delta_to_current": self.peer_delta_to_current,
            "peer_evidence_strength": self.evidence_strength,
            "peer_record_count": self.peer_record_count,
        }


@dataclass(frozen=True)
class _PeerStageRecord:
    rec: ScoreRecord
    selection_eligible: bool | None
    candidate_ready: bool | None


def build_peer_route_evidence_from_csv(
    path: str | Path,
    *,
    current_worker_id: str,
    max_chars: int = 1400,
    min_delta_ratio: float = 0.0,
) -> PeerRouteEvidence:
    csv_path = Path(path)
    if not csv_path.is_file():
        return PeerRouteEvidence(text="")
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return PeerRouteEvidence(text="")
    return build_peer_route_evidence(
        rows,
        current_worker_id=current_worker_id,
        max_chars=max_chars,
        min_delta_ratio=min_delta_ratio,
    )


def build_peer_route_evidence(
    rows: list[dict[str, Any]],
    *,
    current_worker_id: str,
    max_chars: int = 1400,
    min_delta_ratio: float = 0.0,
) -> PeerRouteEvidence:
    raw_items = [
        _PeerStageRecord(
            rec=rec,
            selection_eligible=_boolish(row.get("selection_eligible"), default=None),
            candidate_ready=_boolish(row.get("candidate_ready"), default=None),
        )
        for row in rows
        if (rec := score_record_from_stage_performance_row(row)) is not None
    ]
    normalized_records = normalize_score_record_directions([item.rec for item in raw_items])
    records = [
        _PeerStageRecord(
            rec=rec,
            selection_eligible=item.selection_eligible,
            candidate_ready=item.candidate_ready,
        )
        for item, rec in zip(raw_items, normalized_records, strict=False)
    ]
    current_worker = str(current_worker_id or "").strip() or "W00"
    peer_records = [item for item in records if item.rec.worker_id and item.rec.worker_id != current_worker]
    reliable_peers = [item for item in peer_records if _record_is_reliable_peer(item)]
    if not reliable_peers:
        return PeerRouteEvidence(text="", peer_record_count=len(peer_records))

    peer_best = _best_record(reliable_peers)
    if peer_best is None:
        return PeerRouteEvidence(text="", peer_record_count=len(peer_records))
    current_records = [item for item in records if item.rec.worker_id == current_worker]
    current_best = _best_record([item for item in current_records if _record_is_reliable_peer(item)])
    if current_best is None:
        current_best = _best_record(current_records)
    peer = peer_best.rec
    current = current_best.rec if current_best is not None else None

    delta = _delta_to_current(peer, current)
    if delta is not None and delta <= 0 and min_delta_ratio >= 0:
        ratio = abs(delta) / max(abs(current.value) if current else 1.0, 1e-12)
        if ratio > min_delta_ratio:
            return PeerRouteEvidence(
                text="",
                peer_best_worker=peer.worker_id,
                peer_best_stage=peer.stage_id,
                peer_best_metric=peer.value,
                current_best_stage=current.stage_id if current else "",
                current_best_metric=current.value if current else None,
                peer_delta_to_current=delta,
                evidence_strength="weaker_peer",
                peer_record_count=len(peer_records),
            )

    strength = _evidence_strength(peer, current, delta)
    lines = [
        (
            f"- Best reliable peer: {peer.worker_id}:{peer.stage_id} "
            f"metric={_fmt_metric(peer.value)} ({_direction_text(peer)}); "
            f"status={_status_text(peer_best)}; method={_compact_method(peer.method_text)}"
        ),
    ]
    if current is not None:
        lines.append(
            f"- This worker best: {current.worker_id}:{current.stage_id} "
            f"metric={_fmt_metric(current.value)}; peer_delta={_fmt_delta(delta)}."
        )
    else:
        lines.append("- This worker has no metric-backed stage yet.")
    lines.append(
        "- Treat this as search evidence: compare adapting the peer route, restoring a local stage to test it, or justifying an independent route."
    )
    text = _truncate("\n".join(lines), max_chars=max_chars, marker="peer route evidence truncated")
    return PeerRouteEvidence(
        text=text,
        peer_best_worker=peer.worker_id,
        peer_best_stage=peer.stage_id,
        peer_best_metric=peer.value,
        current_best_stage=current.stage_id if current else "",
        current_best_metric=current.value if current else None,
        peer_delta_to_current=delta,
        evidence_strength=strength,
        peer_record_count=len(peer_records),
    )


def _record_is_reliable_peer(item: _PeerStageRecord) -> bool:
    rec = item.rec
    if rec.validation_ok is False:
        return False
    if str(rec.metric_validity or "").strip().lower() == "low":
        return False
    if item.selection_eligible is False:
        return False
    return True


def _best_record(records: list[_PeerStageRecord]) -> _PeerStageRecord | None:
    best: _PeerStageRecord | None = None
    for item in records:
        rec = item.rec
        if best is None:
            best = item
            continue
        best_rec = best.rec
        is_better = rec.value < best_rec.value if rec.lower_is_better else rec.value > best_rec.value
        if is_better:
            best = item
    return best


def _delta_to_current(peer: ScoreRecord, current: ScoreRecord | None) -> float | None:
    if current is None:
        return None
    delta = current.value - peer.value if peer.lower_is_better else peer.value - current.value
    return delta if math.isfinite(delta) else None


def _evidence_strength(peer: ScoreRecord, current: ScoreRecord | None, delta: float | None) -> str:
    if current is None:
        return "peer_only"
    if delta is None:
        return "unknown_delta"
    if delta > 0:
        return "peer_better"
    if delta == 0:
        return "peer_tied"
    return "peer_weaker"


def _status_text(item: _PeerStageRecord) -> str:
    rec = item.rec
    bits = []
    if rec.validation_ok is not None:
        bits.append(f"validation_ok={rec.validation_ok}")
    if rec.metric_validity:
        bits.append(f"metric_validity={rec.metric_validity}")
    if item.selection_eligible is not None:
        bits.append(f"selection_eligible={item.selection_eligible}")
    if item.candidate_ready is not None:
        bits.append(f"candidate_ready={item.candidate_ready}")
    return ",".join(bits)


def _boolish(value: Any, *, default: bool | None = None) -> bool | None:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _direction_text(rec: ScoreRecord) -> str:
    return "lower is better" if rec.lower_is_better else "higher is better"


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6g}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "unknown"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.6g}"


def _compact_method(text: str, *, max_chars: int = 120) -> str:
    method = " ".join(str(text or "").split())
    if not method:
        return "unknown"
    return _truncate(method, max_chars=max_chars, marker="truncated")


def _truncate(text: str, *, max_chars: int, marker: str) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = f"... [{marker}]"
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix
