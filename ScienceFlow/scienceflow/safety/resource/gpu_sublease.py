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

import re
from dataclasses import dataclass
from typing import Any


_MULTI_GPU_PATTERNS = (
    re.compile(r"\btorchrun\b", re.IGNORECASE),
    re.compile(r"\bdeepspeed\b", re.IGNORECASE),
    re.compile(r"\baccelerate\s+launch\b", re.IGNORECASE),
    re.compile(r"\bmpirun\b|\bmpiexec\b", re.IGNORECASE),
    re.compile(r"\bpython(?:\d+(?:\.\d+)?)?\s+-m\s+torch\.distributed\b", re.IGNORECASE),
    re.compile(r"\bDistributedDataParallel\b|\bDataParallel\b", re.IGNORECASE),
)
_NPROC_RE = re.compile(r"(?:--nproc_per_node|--num_processes|--num-gpus|--gpus)\s*[= ]\s*(\d+)", re.IGNORECASE)
_CUDA_VISIBLE_RE = re.compile(r"\bCUDA_VISIBLE_DEVICES\s*=\s*([0-9,\s]+)")


@dataclass(frozen=True)
class GPUSubleasePlan:
    requested_gpu_count: int
    inferred_needed_gpu_count: int
    candidate_gpu_ids: list[str]
    explicit_request: bool
    multi_gpu_evidence: list[str]
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "requested_gpu_count": self.requested_gpu_count,
            "inferred_needed_gpu_count": self.inferred_needed_gpu_count,
            "candidate_gpu_ids": list(self.candidate_gpu_ids),
            "explicit_request": self.explicit_request,
            "multi_gpu_evidence": list(self.multi_gpu_evidence),
            "reason": self.reason,
        }


def plan_gpu_sublease(
    *,
    candidate_gpu_ids: list[str],
    requested_gpu_count: int | None = None,
    default_request: int = 1,
    max_request: int = 1,
    command: str = "",
    metadata: dict[str, Any] | None = None,
) -> GPUSubleasePlan:
    """Choose how many GPUs one command should lease from a task-local pool.

    The helper is intentionally generic: it only uses explicit request counts
    and multi-GPU launch evidence. It does not inspect task names or special-case
    entrypoints.
    """

    candidates = _clean_ids(candidate_gpu_ids)
    available = max(1, len(candidates) or 1)
    max_allowed = max(1, min(_int_or_one(max_request), available))
    explicit = _explicit_count(requested_gpu_count, metadata)
    if explicit is not None:
        requested = max(1, min(explicit, max_allowed))
        return GPUSubleasePlan(
            requested_gpu_count=requested,
            inferred_needed_gpu_count=requested,
            candidate_gpu_ids=candidates,
            explicit_request=True,
            multi_gpu_evidence=["explicit_gpu_request"] if requested > 1 else [],
            reason="explicit_gpu_request",
        )

    evidence = multi_gpu_evidence(command=command, metadata=metadata)
    if evidence:
        inferred = max(2, _int_or_one(default_request))
        requested = max(1, min(inferred, max_allowed))
        return GPUSubleasePlan(
            requested_gpu_count=requested,
            inferred_needed_gpu_count=requested,
            candidate_gpu_ids=candidates,
            explicit_request=False,
            multi_gpu_evidence=evidence,
            reason="multi_gpu_launch_evidence",
        )

    return GPUSubleasePlan(
        requested_gpu_count=1,
        inferred_needed_gpu_count=1,
        candidate_gpu_ids=candidates,
        explicit_request=False,
        multi_gpu_evidence=[],
        reason="single_gpu_until_multi_gpu_evidence",
    )


def multi_gpu_evidence(*, command: str = "", metadata: dict[str, Any] | None = None) -> list[str]:
    text_parts = [str(command or "")]
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("command", "command_excerpt", "entrypoint"):
        raw = meta.get(key)
        if raw:
            text_parts.append(str(raw))
    source_hint = meta.get("source_hint") if isinstance(meta.get("source_hint"), dict) else {}
    for key in ("raw_text", "command_text", "labels"):
        raw = source_hint.get(key)
        if raw:
            text_parts.append(str(raw))
    text = "\n".join(text_parts)
    evidence: list[str] = []
    for pattern in _MULTI_GPU_PATTERNS:
        if pattern.search(text):
            evidence.append(pattern.pattern)
            break
    match = _NPROC_RE.search(text)
    if match and _int_or_zero(match.group(1)) > 1:
        evidence.append(f"process_count={match.group(1)}")
    cuda = _CUDA_VISIBLE_RE.search(text)
    if cuda:
        visible = [x.strip() for x in cuda.group(1).split(",") if x.strip()]
        if len(visible) > 1:
            evidence.append("cuda_visible_devices_multi")
    return evidence


def _explicit_count(value: int | None, metadata: dict[str, Any] | None) -> int | None:
    direct = _int_or_zero(value)
    if direct > 0:
        return direct
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("gpu_request_count", "requested_gpu_count"):
        parsed = _int_or_zero(meta.get(key))
        if parsed > 0:
            return parsed
    source_hint = meta.get("source_hint") if isinstance(meta.get("source_hint"), dict) else {}
    for key in ("requested_gpu_count", "source_gpu_request"):
        parsed = _int_or_zero(source_hint.get(key))
        if parsed > 0:
            return parsed
    return None


def _clean_ids(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        gpu_id = str(value).strip()
        if gpu_id and gpu_id not in seen:
            out.append(gpu_id)
            seen.add(gpu_id)
    return out


def _int_or_one(value: Any) -> int:
    parsed = _int_or_zero(value)
    return parsed if parsed > 0 else 1


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
