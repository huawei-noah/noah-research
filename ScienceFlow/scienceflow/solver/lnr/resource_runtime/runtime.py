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

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scienceflow.safety.resource.gpu_sublease import plan_gpu_sublease
from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_GPU_EXEC,
)

_CPU_SUPPORT_CLASSES = [RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_READONLY_CPU, RESOURCE_LIGHT_CPU]
_GPU_PRESSURE_SUPPORT_CLASSES = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, *_CPU_SUPPORT_CLASSES]
_SHARE_OVERRIDE_CANDIDATE_CLASSES = {
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
}
_SHARE_OVERRIDE_PRIMARY_CLASSES = {
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
}
_TRIAL_SHARE_CANDIDATE_CLASSES = {
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
}

from scienceflow.solver.lnr.resource_feedback_contract import resource_feedback_text

from .admission_safety import (
    build_admission_safety_deferred_result,
    evaluate_admission_safety,
    evaluate_yellow_safety_hold,
)
from .gpu_lease_store import GPULeaseStore
from .history_store import ResourceHistoryStore
from .models import GPUQueueConfig
from .pressure_state import GPUPressureStateStore
from .priority import calculate_admission_priority
from .unified_store import UnifiedResourceStore
from .utilization import sample_nvidia_smi


@dataclass(frozen=True)
class _GPUResourcePolicy:
    resource_class: str
    slot_weight: float
    max_per_gpu: int
    incompatible_classes: tuple[str, ...]


class ResourceRuntime:
    """LHR resource facade for queue and lease coordination."""

    def __init__(
        self,
        *,
        worker_id: str,
        resource_dir: Path,
        gpu_queue: GPUQueueConfig,
    ) -> None:
        self.worker_id = str(worker_id or "W00")
        self.resource_dir = Path(resource_dir)
        self.gpu_queue = gpu_queue
        self.gpu_store = GPULeaseStore(
            self.resource_dir / "gpu_leases.json",
            lease_ttl_sec=self.gpu_queue.lease_ttl_sec,
            admission_queue_enabled=bool(self.gpu_queue.admission_queue_enabled),
            waiter_ttl_sec=float(self.gpu_queue.admission_waiter_ttl_sec or 900.0),
            idle_release_admission_mode=str(self.gpu_queue.idle_release_admission_mode or "strict_exclusive"),
        )
        self.pressure_store = GPUPressureStateStore(
            self.resource_dir / "gpu_pressure.json",
            default_cooldown_sec=max(120.0, float(self.gpu_queue.max_wait_sec or 0.0)),
            duplicate_digest_cooldown_sec=float(self.gpu_queue.duplicate_digest_cooldown_sec or 600.0),
            duplicate_digest_threshold=int(self.gpu_queue.duplicate_digest_threshold or 2),
            yellow_hold_sec=float(self.gpu_queue.pressure_yellow_hold_sec or 120.0),
            red_to_yellow_sec=float(self.gpu_queue.pressure_red_to_yellow_sec or 120.0),
        )
        self.history_store = ResourceHistoryStore(self.resource_dir / "gpu_runtime_history.json")
        self.unified_store = UnifiedResourceStore(self.resource_dir)
        self._queue_started: set[str] = set()
        self._last_heartbeat: dict[str, float] = {}
        self._leased_gpu_ids: dict[str, list[str]] = {}
        self._admission_backoff_cache: dict[str, dict[str, Any]] = {}
        self._resource_wait_seq = 0
        self._resource_wait_tokens: dict[str, dict[str, Any]] = {}
        self._resource_wait_suppressed_until: dict[str, float] = {}

    def sync_resource_state(self, *, reason: str = "sync") -> dict[str, Any]:
        try:
            leases = self.gpu_store.snapshot_active()
            pressure = self.pressure_store.snapshot()
            history = self.history_store.snapshot()
            return self.unified_store.sync_legacy_gpu_state(
                leases=leases,
                pressure=pressure,
                runtime_history=history,
                reason=str(reason or "sync"),
            )
        except Exception as exc:
            return {"synced": False, "reason": "resource_state_sync_failed", "error": str(exc)}

    def record_resource_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        trace_id: str = "",
        proposal_id: str = "",
        decision_id: str = "",
        task_id: str = "",
        command_id: str = "",
        lease_id: str = "",
    ) -> dict[str, Any]:
        self.sync_resource_state(reason=f"event:{event_type}")
        try:
            return self.unified_store.append_event(
                event_type,
                payload=payload,
                trace_id=trace_id,
                proposal_id=proposal_id,
                decision_id=decision_id,
                task_id=task_id,
                worker_id=self.worker_id,
                command_id=command_id,
                lease_id=lease_id,
            )
        except Exception as exc:
            return {"recorded": False, "reason": "resource_event_append_failed", "error": str(exc)}

    def update_active_resource_proposal(self, proposal_id: str, proposal: dict[str, Any] | None) -> dict[str, Any]:
        try:
            return self.unified_store.update_active_proposal(proposal_id, proposal)
        except Exception as exc:
            return {"updated": False, "reason": "active_proposal_update_failed", "error": str(exc)}

    def clear_active_resource_proposals_for_command(
        self,
        command_id: str,
        *,
        proposal_type: str = "",
        reason_code: str = "",
    ) -> dict[str, Any]:
        try:
            return self.unified_store.clear_active_proposals_for_command(
                command_id,
                proposal_type=proposal_type,
                reason_code=reason_code,
            )
        except Exception as exc:
            return {"cleared": False, "reason": "active_proposal_clear_failed", "error": str(exc)}

    def _capacity_slots(self) -> float:
        try:
            capacity = float(self.gpu_queue.capacity_slots)
        except (TypeError, ValueError):
            capacity = 1.0
        return capacity if capacity > 0 else 1.0

    def _policy_for_resource_class(self, resource_class: str) -> _GPUResourcePolicy | None:
        cls = str(resource_class or "").strip()
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
            return _GPUResourcePolicy(
                resource_class=RESOURCE_HEAVY_GPU_TRAIN,
                slot_weight=1.0,
                max_per_gpu=max(1, int(self.gpu_queue.max_heavy_per_gpu or 1)),
                incompatible_classes=(
                    RESOURCE_GPU_TT_LIGHT,
                    RESOURCE_GPU_FEATURE_EXTRACT,
                    RESOURCE_GPU_LIGHT_TRAIN,
                    RESOURCE_UNKNOWN_GPU_EXEC,
                ),
            )
        if cls == RESOURCE_GPU_TT_LIGHT:
            incompatible = [RESOURCE_UNKNOWN_GPU_EXEC]
            if not bool(self.gpu_queue.share_tt_with_train):
                incompatible.extend([RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE])
            return _GPUResourcePolicy(
                resource_class=RESOURCE_GPU_TT_LIGHT,
                slot_weight=0.25,
                max_per_gpu=max(1, int(self.gpu_queue.gpu_tt_max_per_gpu or 1)),
                incompatible_classes=tuple(incompatible),
            )
        if cls == RESOURCE_GPU_FEATURE_EXTRACT:
            return _GPUResourcePolicy(
                resource_class=RESOURCE_GPU_FEATURE_EXTRACT,
                slot_weight=0.5,
                max_per_gpu=max(1, int(self.gpu_queue.gpu_feature_max_per_gpu or 1)),
                incompatible_classes=(
                    RESOURCE_HEAVY_GPU_TRAIN,
                    RESOURCE_HEAVY_GPU_CANDIDATE,
                    RESOURCE_UNKNOWN_GPU_EXEC,
                ),
            )
        if cls == RESOURCE_GPU_LIGHT_TRAIN:
            incompatible = [RESOURCE_UNKNOWN_GPU_EXEC]
            if not bool(self.gpu_queue.share_tt_with_train):
                incompatible.extend([RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE])
            return _GPUResourcePolicy(
                resource_class=RESOURCE_GPU_LIGHT_TRAIN,
                slot_weight=0.5,
                max_per_gpu=1,
                incompatible_classes=tuple(incompatible),
            )
        if cls == RESOURCE_UNKNOWN_GPU_EXEC:
            return _GPUResourcePolicy(
                resource_class=RESOURCE_UNKNOWN_GPU_EXEC,
                slot_weight=1.0,
                max_per_gpu=1,
                incompatible_classes=(
                    RESOURCE_HEAVY_GPU_TRAIN,
                    RESOURCE_HEAVY_GPU_CANDIDATE,
                    RESOURCE_GPU_TT_LIGHT,
                    RESOURCE_GPU_FEATURE_EXTRACT,
                    RESOURCE_GPU_LIGHT_TRAIN,
                ),
            )
        return None

    def should_queue_gpu(self, *, resource_class: str, gpu_ids: list[str]) -> bool:
        if not self.gpu_queue.enabled:
            return False
        if self._policy_for_resource_class(resource_class) is None:
            return False
        if [x for x in (gpu_ids or []) if str(x).strip()]:
            return True
        return self._assignment_mode() == "lease" and bool(self._configured_gpu_pool())

    def _assignment_mode(self) -> str:
        mode = str(self.gpu_queue.assignment or "env_only").strip().lower()
        return mode if mode in {"env_only", "lease"} else "env_only"

    def lease_assignment_enabled(self) -> bool:
        return self._assignment_mode() == "lease"

    def _configured_gpu_pool(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in self.gpu_queue.gpu_pool or []:
            gpu_id = str(raw).strip()
            if gpu_id and gpu_id not in seen:
                seen.add(gpu_id)
                out.append(gpu_id)
        return out

    def _pressure_gpu_ids(self, gpu_ids: list[str]) -> list[str]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if ids:
            return ids
        if self._assignment_mode() == "lease":
            return self._configured_gpu_pool()
        return ids

    def _request_count(self, available_count: int, *, requested: int | None = None) -> int:
        max_request = max(1, int(self.gpu_queue.max_request or 1))
        default_request = max(1, int(self.gpu_queue.default_request or 1))
        desired = max(1, int(requested or default_request))
        return max(1, min(desired, max_request, max(1, int(available_count or 1))))

    def _gpu_sublease_plan(
        self,
        *,
        candidate_gpu_ids: list[str],
        request_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = metadata if isinstance(metadata, dict) else {}
        plan = plan_gpu_sublease(
            candidate_gpu_ids=candidate_gpu_ids,
            requested_gpu_count=request_count,
            default_request=int(self.gpu_queue.default_request or 1),
            max_request=int(self.gpu_queue.max_request or 1),
            command=str(meta.get("command") or meta.get("command_excerpt") or ""),
            metadata=meta,
        )
        return plan.to_json()

    def _estimated_gpu_footprint(self, resource_class: str) -> dict[str, Any]:
        cls = str(resource_class or "").strip()
        estimates = {
            RESOURCE_GPU_TT_LIGHT: (6.0, "medium"),
            RESOURCE_GPU_FEATURE_EXTRACT: (12.0, "medium"),
            RESOURCE_GPU_LIGHT_TRAIN: (18.0, "low"),
            RESOURCE_HEAVY_GPU_CANDIDATE: (28.0, "low"),
            RESOURCE_HEAVY_GPU_TRAIN: (28.0, "low"),
            RESOURCE_UNKNOWN_GPU_EXEC: (32.0, "low"),
        }
        peak, confidence = estimates.get(cls, (24.0, "low"))
        return {
            "estimated_peak_mem_gb": float(peak),
            "confidence": confidence,
            "source": "resource_class_default",
        }

    @staticmethod
    def _sample_rows_by_gpu(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if gpu_id:
                out[gpu_id] = row
        return out

    @staticmethod
    def _row_free_mem_gb(row: dict[str, Any]) -> float | None:
        try:
            used = float(row.get("memory_used_mb"))
            total = float(row.get("memory_total_mb"))
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        return max(0.0, (total - used) / 1024.0)

    def admission_opportunity_facts(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str],
        request_count: int | None = None,
        selected_gpu_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        policy = self._policy_for_resource_class(resource_class)
        if policy is None or self._assignment_mode() != "lease":
            return {"enabled": False, "reason": "lease_admission_not_applicable"}
        raw_candidates = selected_gpu_ids or gpu_ids or self._configured_gpu_pool()
        candidates = [str(x) for x in (raw_candidates or []) if str(x).strip()]
        if not candidates:
            return {"enabled": False, "reason": "no_candidate_gpu_ids"}
        requested = self._request_count(len(candidates), requested=request_count)
        sample = sample_nvidia_smi(candidates)
        rows = self._sample_rows_by_gpu(sample if isinstance(sample, dict) else {})
        availability = self.gpu_store.availability(
            candidate_gpu_ids=candidates,
            request_count=requested,
            max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
            resource_class=policy.resource_class,
            slot_weight=policy.slot_weight,
            capacity_slots=self._capacity_slots(),
            max_per_gpu=policy.max_per_gpu,
            incompatible_classes=list(policy.incompatible_classes),
        )
        internally_available = {str(x) for x in (availability.get("available_gpu_ids") or []) if str(x).strip()}
        footprint = self._estimated_gpu_footprint(policy.resource_class)
        try:
            estimated_peak = float(footprint.get("estimated_peak_mem_gb") or 0.0)
        except (TypeError, ValueError):
            estimated_peak = 0.0
        reserve = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        buffer = max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0))
        hard_threshold = reserve + buffer if reserve > 0 else 0.0
        footprint_threshold = estimated_peak + max(2.0, min(8.0, buffer if buffer > 0 else 4.0)) if estimated_peak > 0 else hard_threshold
        rows_out: list[dict[str, Any]] = []
        grantable: list[str] = []
        partial: list[dict[str, Any]] = []
        for gpu_id in candidates:
            row = rows.get(gpu_id, {})
            free_gb = self._row_free_mem_gb(row) if row else None
            try:
                util = float(row.get("utilization_gpu_pct") or 0.0) if row else None
            except (TypeError, ValueError):
                util = None
            internal_ok = gpu_id in internally_available
            hard_mem_ok = free_gb is None or hard_threshold <= 0 or free_gb >= hard_threshold
            footprint_ok = free_gb is None or footprint_threshold <= 0 or free_gb >= footprint_threshold
            candidate = {
                "gpu_id": gpu_id,
                "internal_lease_available": bool(internal_ok),
                "free_mem_gb": (round(float(free_gb), 3) if free_gb is not None else None),
                "utilization_gpu_pct": (round(float(util), 3) if util is not None else None),
                "hard_mem_ok": bool(hard_mem_ok),
                "footprint_ok": bool(footprint_ok),
            }
            rows_out.append(candidate)
            if internal_ok and hard_mem_ok:
                grantable.append(gpu_id)
                if free_gb is not None and free_gb >= max(hard_threshold, min(16.0, footprint_threshold)):
                    partial.append(candidate)
        return {
            "enabled": True,
            "lease_grantable_by_llm": bool(grantable),
            "full_request_grantable": len(grantable) >= requested,
            "grantable_gpu_ids": grantable,
            "partial_gpu_candidates": partial,
            "observed_gpus": rows_out,
            "requested_gpu_count": requested,
            "footprint": footprint,
            "free_mem_hard_threshold_gb": hard_threshold,
            "estimated_footprint_threshold_gb": round(float(footprint_threshold), 3),
            "availability": {
                "available": bool(availability.get("available")),
                "available_gpu_ids": list(availability.get("available_gpu_ids") or []),
                "requested_count": availability.get("requested_count"),
            },
            "sample_available": bool(isinstance(sample, dict) and sample.get("available") is not False),
        }

    def grant_llm_admission_lease(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        request_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        observe_then_run: bool = False,
        observe_sec: float = 0.0,
    ) -> dict[str, Any]:
        policy = self._policy_for_resource_class(resource_class)
        if policy is None:
            return {"enabled": False, "acquired": False, "reason": "resource_class_not_queueable"}
        if self._assignment_mode() != "lease":
            return {"enabled": False, "acquired": False, "reason": "lease_assignment_disabled"}
        candidates = [str(x) for x in (gpu_ids or self._configured_gpu_pool()) if str(x).strip()]
        if not candidates:
            return {"enabled": True, "acquired": False, "status": "PENDING", "reason": "no_candidate_gpu_ids"}
        sublease = self._gpu_sublease_plan(
            candidate_gpu_ids=candidates,
            request_count=request_count,
            metadata=metadata,
        )
        requested = int(sublease.get("requested_gpu_count") or 1)
        min_free = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        if min_free > 0:
            sample = sample_nvidia_smi(candidates)
            safety = evaluate_admission_safety(
                sample=sample if isinstance(sample, dict) else {},
                gpu_ids=candidates,
                resource_class=policy.resource_class,
                request_count=requested,
                assignment="lease",
                min_free_mem_gb=min_free,
                free_mem_buffer_gb=max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0)),
            )
            if safety.get("blocked"):
                return {
                    "enabled": True,
                    "acquired": False,
                    "status": "PENDING",
                    "reason": str(safety.get("reason") or "gpu_memory_below_admission_reserve"),
                    "safety": safety,
                    "candidate_physical_gpus": candidates,
                    "requested_gpu_count": requested,
                    "gpu_sublease": dict(sublease),
                }
            safe_ids = [str(x) for x in (safety.get("safe_gpu_ids") or candidates) if str(x).strip()]
            if safe_ids:
                candidates = safe_ids
                requested = self._request_count(len(candidates), requested=requested)
                sublease = {**sublease, "candidate_gpu_ids": list(candidates), "requested_gpu_count": requested}
        meta = {
            **dict(metadata or {}),
            **GPULeaseStore.current_owner_metadata(),
            "assignment": "lease",
            "resource_class": str(resource_class or ""),
            "policy_resource_class": policy.resource_class,
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "max_per_gpu": policy.max_per_gpu,
            "incompatible_classes": list(policy.incompatible_classes),
            "gpu_sublease": dict(sublease),
            "llm_admission_grant": True,
            "observe_then_run": bool(observe_then_run),
            "observe_sec": max(0.0, float(observe_sec or 0.0)),
        }
        acquired, details = self.gpu_store.try_acquire_any(
            job_id=str(job_id or ""),
            worker_id=self.worker_id,
            candidate_gpu_ids=candidates,
            request_count=requested,
            max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
            metadata=meta,
            resource_class=policy.resource_class,
            slot_weight=policy.slot_weight,
            capacity_slots=self._capacity_slots(),
            max_per_gpu=policy.max_per_gpu,
            incompatible_classes=list(policy.incompatible_classes),
        )
        lease = details.get("lease") if isinstance(details.get("lease"), dict) else {}
        assigned = [str(x) for x in (lease.get("gpu_ids") or details.get("assigned_gpu_ids") or []) if str(x).strip()]
        env_updates = {}
        if acquired and assigned:
            self._leased_gpu_ids[str(job_id or "")] = assigned
            env_updates = {
                "CUDA_VISIBLE_DEVICES": ",".join(assigned),
                "SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL": ",".join(assigned),
                "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL": ",".join(str(i) for i, _ in enumerate(assigned)),
            }
            pool = self._configured_gpu_pool()
            if pool:
                env_updates["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] = ",".join(pool)
        result = {
            "enabled": True,
            "acquired": bool(acquired),
            "status": "GRANTED" if acquired else "PENDING",
            "admission_action": "RUN_NOW" if acquired else "PENDING",
            "reason": str(details.get("reason") or ("llm_admission_lease_granted" if acquired else "llm_admission_lease_unavailable")),
            "gpu_ids": assigned or candidates,
            "assigned_physical_gpus": assigned,
            "candidate_physical_gpus": candidates,
            "allowed_physical_gpus": self._configured_gpu_pool() or candidates,
            "requested_gpu_count": requested,
            "gpu_sublease": dict(sublease),
            "env_updates": env_updates,
            "details": details,
            "policy_resource_class": policy.resource_class,
            "resource_class": str(resource_class or ""),
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "admission_llm_lease_attempted": True,
            "admission_observe_then_run": bool(observe_then_run and acquired),
            "admission_observe_sec": max(0.0, float(observe_sec or 0.0)),
        }
        self.record_resource_event(
            "admission_llm_lease_grant" if acquired else "admission_llm_lease_grant_failed",
            payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return result

    def _cold_eta_sec(self) -> float:
        return max(300.0, min(float(self.gpu_queue.max_wait_sec or 0.0) + 300.0, 1800.0))

    def _admission_priority(self, value_hint: dict[str, Any]) -> dict[str, float]:
        return calculate_admission_priority(value_hint)

    @staticmethod
    def _resource_wait_scope_key(*, resource_class: str, gpu_ids: list[str], holder_job_ids: list[str]) -> str:
        gpu_key = ",".join(sorted({str(x) for x in (gpu_ids or []) if str(x).strip()}))
        holder_key = ",".join(sorted({str(x) for x in (holder_job_ids or []) if str(x).strip()}))
        return f"{str(resource_class or 'unknown')}|{gpu_key}|{holder_key}"

    def create_resource_wait_option(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        holder_job_ids: list[str] | None = None,
        queue_position: int | None = None,
        queue_len: int | None = None,
        command_digest: str = "",
        reason: str = "resource_busy",
        max_wait_sec: float | None = None,
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        holders = [str(x) for x in (holder_job_ids or []) if str(x).strip()]
        scope_key = self._resource_wait_scope_key(resource_class=resource_class, gpu_ids=ids, holder_job_ids=holders)
        generation = self._current_pressure_generation()
        suppressed_until = float(self._resource_wait_suppressed_until.get(scope_key) or 0.0)
        if suppressed_until > time.time():
            return {"offered": False, "reason": "wait_suppressed_after_timeout", "scope_key": scope_key, "suppressed_until": suppressed_until}
        self._resource_wait_seq += 1
        token = f"rw-{self.worker_id}-{int(time.time() * 1000)}-{self._resource_wait_seq:04d}"
        wait_max = max(30.0, min(float(max_wait_sec or 600.0), 1800.0))
        record = {
            "wait_token": token,
            "worker_id": self.worker_id,
            "job_id": str(job_id or ""),
            "resource_class": str(resource_class or ""),
            "gpu_ids": ids,
            "holder_job_ids": holders,
            "queue_position": int(queue_position or 0),
            "queue_len": int(queue_len or 0),
            "command_digest": str(command_digest or ""),
            "reason": str(reason or "resource_busy"),
            "scope_key": scope_key,
            "start_pressure_generation": generation,
            "created_at": time.time(),
            "max_wait_sec": wait_max,
        }
        self._resource_wait_tokens[token] = record
        self.record_resource_event(
            "managed_resource_wait_offered",
            payload={**record, "cpu_support_preferred": True, "wait_is_optional": True},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return {"offered": True, **record}

    @staticmethod
    def _resource_wait_extra_facts(wait_option: dict[str, Any] | None) -> dict[str, str]:
        if not isinstance(wait_option, dict) or not wait_option.get("offered"):
            return {}
        return {
            "action_options": "cpu_support,resource_wait",
            "cpu_support_preferred": "true",
            "wait_is_optional": "true",
            "bash_sleep_allowed": "false",
            "wait_tool": "resource_wait",
            "wait_token": str(wait_option.get("wait_token") or ""),
            "wait_max_sec": "%.0f" % max(0.0, float(wait_option.get("max_wait_sec") or 0.0)),
            "wait_reason": str(wait_option.get("reason") or "resource_busy"),
        }

    @staticmethod
    def _duplicate_summary(details: dict[str, Any]) -> dict[str, Any]:
        raw = details.get("duplicate_digest_summary") if isinstance(details.get("duplicate_digest_summary"), dict) else {}
        active = [str(x) for x in (raw.get("duplicate_active_job_ids") or []) if str(x).strip()]
        waiters = [str(x) for x in (raw.get("duplicate_waiter_job_ids") or []) if str(x).strip()]
        try:
            oldest_wait = float(raw.get("same_digest_oldest_wait_sec") or 0.0)
        except (TypeError, ValueError):
            oldest_wait = 0.0
        return {
            "command_digest": str(raw.get("command_digest") or ""),
            "duplicate_active_job_ids": active,
            "duplicate_waiter_job_ids": waiters,
            "duplicate_count": len(active) + len(waiters),
            "same_digest_oldest_wait_sec": max(0.0, oldest_wait),
        }

    def _admission_action_for_blocked(
        self,
        *,
        policy: _GPUResourcePolicy,
        requested_count: int,
        details: dict[str, Any],
        priority: dict[str, float],
        eta_next_train_sec: float,
    ) -> tuple[str, str]:
        duplicate = self._duplicate_summary(details)
        priority_score = float(priority.get("admission_priority_score") or 0.0)
        duplicate_count = int(duplicate.get("duplicate_count") or 0)
        oldest_same_digest_wait = float(duplicate.get("same_digest_oldest_wait_sec") or 0.0)
        is_train = policy.resource_class in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}
        if duplicate_count > 0 and is_train and priority_score < 0.75:
            return "REPLAN", "duplicate_gpu_command_under_contention"
        if requested_count > 1 and oldest_same_digest_wait >= float(self.gpu_queue.admission_waiter_ttl_sec or 900.0):
            return "REPLAN", "multi_gpu_pending_wait_exceeded"
        # ETA is informational on the first blocked request. A multi-GPU task becomes
        # REPLAN only after the same pending command has actually waited too long.
        return "PENDING", str(details.get("reason") or "gpu_slot_unavailable")

    def _admission_feedback(
        self,
        *,
        action: str,
        reason: str,
        policy_resource_class: str,
        gpu_ids: list[str],
        queue_position: int,
        queue_len: int,
        eta_next_train_sec: float,
        eta_confidence: str,
        priority_score: float,
        holder_job_id: str = "",
        retry_after_sec: float | None = None,
        post_feedback_action: str = "",
        cached_backoff: bool = False,
        wait_option: dict[str, Any] | None = None,
    ) -> str:
        extra = {"admission_priority_score": f"{priority_score:.4f}"}
        extra.update(self._resource_wait_extra_facts(wait_option))
        if retry_after_sec is not None:
            extra["retry_after_sec"] = f"{max(0.0, float(retry_after_sec or 0.0)):.0f}"
        if post_feedback_action:
            extra["post_feedback_action"] = str(post_feedback_action)
        if cached_backoff:
            extra["cached_backoff"] = "true"
        return resource_feedback_text(
            status=action,
            reason=reason,
            scope="per_gpu",
            resource_mode="YELLOW",
            blocked_class=policy_resource_class,
            gpu_ids=gpu_ids,
            holder_job_id=holder_job_id,
            queue_position=queue_position,
            queue_len=queue_len,
            eta_next_train_sec=eta_next_train_sec,
            eta_confidence=eta_confidence,
            unlock_condition="holder_released" if holder_job_id else "resource_slot_available",
            blocked_until_unlock=True,
            extra_facts=extra,
        )

    @staticmethod
    def _blocker_job_ids(details: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        def add(raw: Any) -> None:
            job_id = str(raw or "").strip()
            if job_id and job_id not in seen:
                seen.add(job_id)
                out.append(job_id)
        for owners in (details.get("blockers") or {}).values() if isinstance(details.get("blockers"), dict) else []:
            for owner in owners or []:
                add(owner)
        blocker_details = details.get("blocker_details") if isinstance(details.get("blocker_details"), dict) else {}
        for raw in blocker_details.values():
            if not isinstance(raw, dict):
                continue
            for owner in raw.get("owners") or []:
                add(owner)
        return out

    @staticmethod
    def _share_override_blocker_reasons(details: dict[str, Any]) -> list[str]:
        reasons: set[str] = set()
        blocker_details = details.get("blocker_details") if isinstance(details.get("blocker_details"), dict) else {}
        for raw in blocker_details.values():
            if not isinstance(raw, dict):
                continue
            for reason in raw.get("reasons") or []:
                clean = str(reason or "").strip()
                if clean in {"slot_capacity_exceeded", "class_limit_exceeded", "incompatible_active_class"}:
                    reasons.add(clean)
        return sorted(reasons)

    def _admission_share_override_candidate(
        self,
        *,
        job_id: str,
        policy: _GPUResourcePolicy,
        details: dict[str, Any],
        priority: dict[str, float],
        value_hint: dict[str, Any],
    ) -> dict[str, Any]:
        if policy.resource_class not in _SHARE_OVERRIDE_CANDIDATE_CLASSES:
            return {"candidate": False, "reason": "secondary_class_not_share_override_candidate"}
        trial_share = policy.resource_class in _TRIAL_SHARE_CANDIDATE_CLASSES
        if trial_share:
            duplicate = self._duplicate_summary(details)
            if int(duplicate.get("duplicate_count") or 0) > 0:
                return {"candidate": False, "reason": "duplicate_gpu_command_not_trial_share_candidate"}
        block_reasons = self._share_override_blocker_reasons(details)
        trigger_reasons = {"incompatible_active_class", "slot_capacity_exceeded", "class_limit_exceeded"} if trial_share else {"incompatible_active_class"}
        if not (set(block_reasons) & trigger_reasons):
            return {"candidate": False, "reason": "no_share_override_block_reason", "block_reasons": block_reasons}
        blocker_ids = self._blocker_job_ids(details)
        if not blocker_ids:
            return {"candidate": False, "reason": "no_active_blocker"}
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception:
            snapshot = {}
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        primary_ids: list[str] = []
        primary_classes: dict[str, str] = {}
        for blocker_id in blocker_ids:
            raw = leases.get(str(blocker_id)) if isinstance(leases.get(str(blocker_id)), dict) else {}
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            primary_class = str(meta.get("policy_resource_class") or meta.get("resource_class") or "")
            primary_classes[str(blocker_id)] = primary_class
            if primary_class in _SHARE_OVERRIDE_PRIMARY_CLASSES:
                primary_ids.append(str(blocker_id))
        if not primary_ids:
            return {
                "candidate": False,
                "reason": "no_heavy_primary_blocker",
                "blocker_job_ids": blocker_ids,
                "primary_classes": primary_classes,
                "block_reasons": block_reasons,
            }
        waiter = details.get("waiter") if isinstance(details.get("waiter"), dict) else {}
        required_gates = [
            "memory_high_water_mark_headroom",
            "cpu_isolation_or_low_secondary_cpu",
            "primary_progress_not_stalled",
            "primary_kill_replan_candidate_false",
            "secondary_revocation_available",
        ]
        if any(primary_classes.get(pid) not in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN} for pid in primary_ids):
            required_gates.append("non_heavy_holder_low_util_trial")
        if policy.resource_class == RESOURCE_UNKNOWN_GPU_EXEC:
            required_gates.extend([
                "unknown_gpu_conservative_memory_estimate",
                "unknown_gpu_short_initial_window",
                "unknown_gpu_requires_early_runtime_evidence",
            ])
        if trial_share:
            required_gates.extend([
                "arbiter_llm_grant_required",
                "trial_secondary_stop_first",
                "bounded_trial_initial_window",
            ])
        trial_initial_observe_sec = 90.0 if trial_share and policy.resource_class == RESOURCE_UNKNOWN_GPU_EXEC else 180.0 if trial_share else 0.0
        return {
            "candidate": True,
            "reason": "resource_contention_trial_share_review_required" if trial_share else "incompatible_active_class_share_review_required",
            "proposal_type": "task_gpu_share_review",
            "trial_share": trial_share,
            "trial_mode": "revocable_secondary_trial" if trial_share else "",
            "trial_initial_observe_sec": trial_initial_observe_sec,
            "trial_priority": "primary_protected" if trial_share else "",
            "primary_job_ids": primary_ids,
            "blocker_job_ids": blocker_ids,
            "primary_classes": primary_classes,
            "waiter_job_id": str(job_id or ""),
            "waiter_resource_class": policy.resource_class,
            "waiter_slot_weight": float(policy.slot_weight or 0.0),
            "waiter_effective_slot_weight": 0.5 if trial_share else float(policy.slot_weight or 0.0),
            "waiter_gpu_ids": [
                str(x)
                for x in (
                    waiter.get("gpu_ids")
                    or waiter.get("candidate_gpu_ids")
                    or details.get("candidate_gpu_ids")
                    or []
                )
                if str(x).strip()
            ],
            "waiter_priority_score": float(priority.get("admission_priority_score") or 0.0),
            "waiter_value_hint": dict(value_hint or {}),
            "block_reasons": block_reasons,
            "required_gates": required_gates,
        }

    def _eta_for_blocked(
        self,
        *,
        resource_class: str,
        details: dict[str, Any],
        command_digest: str = "",
        entrypoint: str = "",
    ) -> dict[str, Any]:
        cold = self._cold_eta_sec()
        summary = self.history_store.summary(
            resource_class=str(resource_class or ""),
            command_digest=str(command_digest or ""),
            entrypoint=str(entrypoint or ""),
        )
        count = int(summary.get("count") or 0)
        if count > 0:
            expected = max(60.0, float(summary.get("p80_sec") or summary.get("mean_sec") or cold))
            source = "runtime_history"
            confidence = "high" if count >= 8 else "medium" if count >= 3 else "low"
        else:
            expected = cold
            source = "cold_start"
            confidence = "low"
        blocker_ids = self._blocker_job_ids(details)
        active = self.gpu_store.snapshot_active()
        now = float(active.get("now") or time.time())
        leases = active.get("leases") if isinstance(active.get("leases"), dict) else {}
        ages: list[float] = []
        remaining: list[float] = []
        for job_id in blocker_ids:
            raw = leases.get(job_id) if isinstance(leases.get(job_id), dict) else {}
            if not raw:
                continue
            acquired_at = float(raw.get("acquired_at") or raw.get("heartbeat_at") or now)
            age = max(0.0, now - acquired_at)
            ages.append(age)
            floor = 60.0 if source == "runtime_history" else 300.0
            remaining.append(max(floor, expected - age))
        eta = min(remaining) if remaining else expected
        return {
            "eta_next_train_sec": max(0.0, float(eta)),
            "eta_confidence": confidence,
            "eta_source": source,
            "runtime_history_count": count,
            "runtime_avg_sec": float(summary.get("mean_sec") or 0.0),
            "runtime_p80_sec": float(summary.get("p80_sec") or 0.0),
            "active_blocker_count": len(blocker_ids),
            "active_blocker_job_ids": blocker_ids,
            "active_blocker_age_sec_max": max(ages or [0.0]),
        }


    def _admission_backoff_key(self, *, policy_resource_class: str, gpu_ids: list[str]) -> str:
        gpu_key = ",".join(sorted({str(x) for x in (gpu_ids or []) if str(x).strip()}))
        return f"{self.worker_id}:{str(policy_resource_class or '')}:{gpu_key}"

    def _current_pressure_generation(self) -> int:
        try:
            snap = self.pressure_store.snapshot()
            return int(snap.get("generation") or 0)
        except Exception:
            return 0

    def _waiter_wait_age_sec(self, details: dict[str, Any]) -> float:
        waiter = details.get("waiter") if isinstance(details.get("waiter"), dict) else {}
        try:
            submitted_at = float(waiter.get("submitted_at") or 0.0)
        except (TypeError, ValueError):
            submitted_at = 0.0
        if submitted_at <= 0:
            return 0.0
        return max(0.0, time.time() - submitted_at)

    def _admission_backoff_grace_sec(self) -> float:
        heartbeat = max(0.0, float(self.gpu_queue.heartbeat_sec or 0.0))
        return max(0.25, min(10.0, heartbeat * 4.0 if heartbeat > 0 else 10.0))

    def _admission_backoff_retry_sec(self, eta_next_train_sec: float) -> float:
        eta = max(0.0, float(eta_next_train_sec or 0.0))
        if eta > 0:
            return max(30.0, min(300.0, eta))
        return 120.0

    def _blockers_still_active(self, blocker_job_ids: list[str]) -> bool:
        holders = [str(x) for x in (blocker_job_ids or []) if str(x).strip()]
        if not holders:
            return False
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception:
            return False
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        released_idle = snapshot.get("released_idle") if isinstance(snapshot.get("released_idle"), dict) else {}
        return any(job_id in leases or job_id in released_idle for job_id in holders)

    def _cached_admission_backoff(
        self,
        *,
        job_id: str,
        policy: _GPUResourcePolicy,
        target_gpu_ids: list[str],
        priority: dict[str, float],
    ) -> dict[str, Any] | None:
        if policy.resource_class not in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC}:
            return None
        key = self._admission_backoff_key(policy_resource_class=policy.resource_class, gpu_ids=target_gpu_ids)
        cached = self._admission_backoff_cache.get(key)
        if not isinstance(cached, dict):
            return None
        now = time.time()
        retry_until = float(cached.get("retry_until") or 0.0)
        if retry_until <= now:
            self._admission_backoff_cache.pop(key, None)
            return None
        if int(cached.get("pressure_generation") or 0) != self._current_pressure_generation():
            self._admission_backoff_cache.pop(key, None)
            return None
        blocker_ids = [str(x) for x in (cached.get("holder_job_ids") or []) if str(x).strip()]
        if not self._blockers_still_active(blocker_ids):
            self._admission_backoff_cache.pop(key, None)
            return None
        retry_after = max(1.0, retry_until - now)
        target_ids = [str(x) for x in (target_gpu_ids or cached.get("gpu_ids") or []) if str(x).strip()]
        opportunity = self.admission_opportunity_facts(
            resource_class=policy.resource_class,
            gpu_ids=target_ids,
            request_count=int(cached.get("requested_gpu_count") or 1),
        )
        if opportunity.get("lease_grantable_by_llm"):
            self._admission_backoff_cache.pop(key, None)
            return None
        priority_score = float(priority.get("admission_priority_score") or cached.get("admission_priority_score") or 0.0)
        feedback = self._admission_feedback(
            action="DENIED_REPLAN",
            reason="cached_gpu_admission_backoff",
            policy_resource_class=policy.resource_class,
            gpu_ids=target_ids,
            queue_position=int(cached.get("queue_position") or 1),
            queue_len=int(cached.get("queue_len") or 0),
            eta_next_train_sec=float(cached.get("eta_next_train_sec") or retry_after),
            eta_confidence=str(cached.get("eta_confidence") or "low"),
            priority_score=priority_score,
            holder_job_id=blocker_ids[0] if blocker_ids else "",
            retry_after_sec=retry_after,
            post_feedback_action="cpu_support",
            cached_backoff=True,
            wait_option=None,
        )
        release = self.gpu_store.release(job_id=str(job_id or ""))
        result = {
            "enabled": True,
            "acquired": False,
            "status": "DENIED_REPLAN",
            "admission_action": "DENIED_REPLAN",
            "reason": "cached_gpu_admission_backoff",
            "resource_mode": "YELLOW",
            "max_wait_sec": float(self.gpu_queue.max_wait_sec),
            "heartbeat_sec": float(self.gpu_queue.heartbeat_sec),
            "gpu_ids": target_ids,
            "assigned_physical_gpus": [],
            "allowed_physical_gpus": [str(x) for x in (self._configured_gpu_pool() or target_ids) if str(x).strip()],
            "candidate_physical_gpus": target_ids,
            "requested_gpu_count": int(cached.get("requested_gpu_count") or 1),
            "resource_class": str(cached.get("resource_class") or policy.resource_class),
            "policy_resource_class": policy.resource_class,
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "details": {"cached_backoff": cached, "release": release},
            "queue_started_first": False,
            "queue_position": int(cached.get("queue_position") or 1),
            "queue_len": int(cached.get("queue_len") or 0),
            "top_waiter_job_id": str(cached.get("top_waiter_job_id") or ""),
            "eta_next_train_sec": float(cached.get("eta_next_train_sec") or retry_after),
            "eta_confidence": str(cached.get("eta_confidence") or "low"),
            "holder_job_ids": blocker_ids,
            "allowed_classes": _GPU_PRESSURE_SUPPORT_CLASSES,
            "feedback": feedback,
            "admission_cached_backoff": True,
            "retry_after_sec": retry_after,
            "blocked_until_unlock": True,
            "unlock_condition": "holder_released",
            "post_feedback_action": "cpu_support",
            **priority,
        }
        self.record_resource_event(
            "admission_cached_backoff",
            payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return result

    def _remember_admission_backoff(
        self,
        *,
        key: str,
        policy: _GPUResourcePolicy,
        target_gpu_ids: list[str],
        result: dict[str, Any],
        holder_job_ids: list[str],
        eta_next_train_sec: float,
        eta_confidence: str,
        pressure_generation: int,
    ) -> None:
        if policy.resource_class not in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC}:
            return
        retry_after = self._admission_backoff_retry_sec(eta_next_train_sec)
        self._admission_backoff_cache[key] = {
            "created_at": time.time(),
            "retry_until": time.time() + retry_after,
            "retry_after_sec": retry_after,
            "pressure_generation": int(pressure_generation or 0),
            "gpu_ids": [str(x) for x in (target_gpu_ids or []) if str(x).strip()],
            "holder_job_ids": [str(x) for x in (holder_job_ids or []) if str(x).strip()],
            "resource_class": str(result.get("resource_class") or policy.resource_class),
            "policy_resource_class": policy.resource_class,
            "queue_position": int(result.get("queue_position") or 1),
            "queue_len": int(result.get("queue_len") or 0),
            "top_waiter_job_id": str(result.get("top_waiter_job_id") or ""),
            "requested_gpu_count": int(result.get("requested_gpu_count") or 1),
            "eta_next_train_sec": float(eta_next_train_sec or 0.0),
            "eta_confidence": str(eta_confidence or "low"),
            "admission_priority_score": float(result.get("admission_priority_score") or 0.0),
        }

    def _resource_wait_wake_reason(self, record: dict[str, Any]) -> str:
        start_generation = int(record.get("start_pressure_generation") or 0)
        current_generation = self._current_pressure_generation()
        if current_generation != start_generation:
            return "pressure_generation_changed"
        holders = [str(x) for x in (record.get("holder_job_ids") or []) if str(x).strip()]
        if holders and not self._blockers_still_active(holders):
            return "holder_released"
        gpu_ids = [str(x) for x in (record.get("gpu_ids") or []) if str(x).strip()]
        if gpu_ids:
            try:
                snapshot = self.gpu_store.snapshot_active()
                leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
                busy = False
                for lease in leases.values():
                    if not isinstance(lease, dict):
                        continue
                    lease_gpu_ids = {str(x) for x in (lease.get("gpu_ids") or []) if str(x).strip()}
                    if lease_gpu_ids.intersection(set(gpu_ids)):
                        busy = True
                        break
                if not busy:
                    return "gpu_slot_released"
            except Exception:
                return ""
        return ""

    def _resource_wait_feedback(
        self,
        *,
        status: str,
        reason: str,
        record: dict[str, Any] | None = None,
        extra_facts: dict[str, Any] | None = None,
    ) -> str:
        rec = record or {}
        return resource_feedback_text(
            status=status,
            reason=reason,
            scope="per_gpu",
            resource_mode="YELLOW",
            blocked_class=str(rec.get("resource_class") or "resource_wait"),
            gpu_ids=[str(x) for x in (rec.get("gpu_ids") or []) if str(x).strip()],
            holder_job_id=([str(x) for x in (rec.get("holder_job_ids") or []) if str(x).strip()] or [""])[0],
            queue_position=(int(rec.get("queue_position") or 0) if rec.get("queue_position") is not None else None),
            queue_len=(int(rec.get("queue_len") or 0) if rec.get("queue_len") is not None else None),
            unlock_condition="resource_state_changed" if status == "RESOURCE_AVAILABLE" else "resource_state_change_required",
            blocked_until_unlock=(False if status == "RESOURCE_AVAILABLE" else True),
            extra_facts=extra_facts or {},
        )

    def resource_wait_instruction_for_bash_sleep(
        self,
        *,
        command: str = "",
        planned_sleep_sec: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        records: list[dict[str, Any]] = []
        for record in self._resource_wait_tokens.values():
            if not isinstance(record, dict):
                continue
            try:
                created_at = float(record.get("created_at") or 0.0)
                max_wait = float(record.get("max_wait_sec") or 0.0)
            except (TypeError, ValueError):
                continue
            if created_at <= 0:
                continue
            # Tokens are intended for the immediate RESOURCE_FEEDBACK turn. If
            # an old token survives past its wait window, do not hijack unrelated
            # sleep commands.
            if max_wait > 0 and now - created_at > max_wait:
                continue
            records.append(record)
        if not records:
            return {"available": False, "reason": "no_active_resource_wait_option"}

        record = max(records, key=lambda r: float(r.get("created_at") or 0.0))
        wait_max = max(0.05, min(float(record.get("max_wait_sec") or 600.0), 1800.0))
        if planned_sleep_sec is not None:
            try:
                wait_max = max(0.05, min(wait_max, float(planned_sleep_sec)))
            except (TypeError, ValueError):
                pass
        payload = {
            **record,
            "command": str(command or ""),
            "planned_sleep_sec": planned_sleep_sec,
            "recommended_tool": "resource_wait",
        }
        self.record_resource_event(
            "managed_resource_wait_bash_sleep_blocked",
            payload=payload,
            command_id=str(record.get("job_id") or ""),
            lease_id=str(record.get("job_id") or ""),
        )
        feedback = self._resource_wait_feedback(
            status="RESOURCE_WAIT_REQUIRED",
            reason="use resource_wait tool instead of bash sleep",
            record=record,
            extra_facts={
                "retry_allowed": "false",
                "same_command_retry_allowed": "false",
                "action_options": "resource_wait,cpu_support",
                "cpu_support_preferred": "true",
                "wait_is_optional": "true",
                "bash_sleep_allowed": "false",
                "wait_tool": "resource_wait",
                "wait_token": str(record.get("wait_token") or ""),
                "wait_max_sec": "%.0f" % wait_max,
                "wait_reason": str(record.get("reason") or "resource_busy"),
            },
        )
        return {
            "available": True,
            "wait_token": str(record.get("wait_token") or ""),
            "wait_max_sec": wait_max,
            "feedback": feedback,
        }

    async def managed_resource_wait(
        self,
        *,
        wait_token: str,
        max_wait_sec: float | None = None,
        reason: str = "resource_busy",
    ) -> dict[str, Any]:
        token = str(wait_token or "").strip()
        record = self._resource_wait_tokens.get(token)
        if not isinstance(record, dict):
            feedback = self._resource_wait_feedback(
                status="RESOURCE_WAIT_NOT_AVAILABLE",
                reason="no active resource wait option",
                extra_facts={"retry_allowed": "false"},
            )
            self.record_resource_event("managed_resource_wait_invalid", payload={"wait_token": token, "reason": "invalid_or_expired"})
            return {"status": "RESOURCE_WAIT_NOT_AVAILABLE", "reason": "invalid_or_expired", "feedback": feedback}
        wait_max = max(0.05, min(float(max_wait_sec or record.get("max_wait_sec") or 600.0), float(record.get("max_wait_sec") or 600.0), 1800.0))
        started_at = time.monotonic()
        self.record_resource_event(
            "managed_resource_wait_started",
            payload={**record, "requested_max_wait_sec": wait_max, "reason": str(reason or record.get("reason") or "resource_busy")},
            command_id=str(record.get("job_id") or ""),
            lease_id=str(record.get("job_id") or ""),
        )
        while True:
            wake_reason = self._resource_wait_wake_reason(record)
            if wake_reason:
                self._resource_wait_tokens.pop(token, None)
                elapsed = max(0.0, time.monotonic() - started_at)
                payload = {
                    **record,
                    "elapsed_sec": elapsed,
                    "wake_reason": wake_reason,
                    "wake_pressure_generation": self._current_pressure_generation(),
                }
                self.record_resource_event(
                    "managed_resource_wait_woken",
                    payload=payload,
                    command_id=str(record.get("job_id") or ""),
                    lease_id=str(record.get("job_id") or ""),
                )
                feedback = self._resource_wait_feedback(
                    status="RESOURCE_AVAILABLE",
                    reason=wake_reason,
                    record=record,
                    extra_facts={
                        "retry_allowed": "true",
                        "same_command_retry_allowed": "true",
                        "action_options": "retry_after_feedback_change",
                        "wake_reason": wake_reason,
                        "delta_since_wait_started": wake_reason,
                    },
                )
                return {"status": "RESOURCE_AVAILABLE", "reason": wake_reason, "elapsed_sec": elapsed, "feedback": feedback}
            elapsed = time.monotonic() - started_at
            if elapsed >= wait_max:
                self._resource_wait_tokens.pop(token, None)
                scope_key = str(record.get("scope_key") or "")
                if scope_key:
                    suppress_sec = max(60.0, min(float(record.get("max_wait_sec") or wait_max or 600.0), 600.0))
                    self._resource_wait_suppressed_until[scope_key] = time.time() + suppress_sec
                payload = {**record, "elapsed_sec": elapsed, "timeout_fuse": True}
                self.record_resource_event(
                    "managed_resource_wait_timeout",
                    payload=payload,
                    command_id=str(record.get("job_id") or ""),
                    lease_id=str(record.get("job_id") or ""),
                )
                feedback = self._resource_wait_feedback(
                    status="RESOURCE_WAIT_TIMEOUT",
                    reason="no material resource state change",
                    record=record,
                    extra_facts={
                        "retry_allowed": "false",
                        "same_command_retry_allowed": "false",
                        "resource_wait_allowed": "false",
                        "action_options": "cpu_support,replan,retry_after_feedback_change",
                    },
                )
                return {"status": "RESOURCE_WAIT_TIMEOUT", "reason": "timeout", "elapsed_sec": elapsed, "feedback": feedback}
            await asyncio.sleep(min(0.1, max(0.05, wait_max - elapsed)))

    def queue_try_acquire(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        request_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        policy = self._policy_for_resource_class(resource_class)
        if policy is None or not self.should_queue_gpu(resource_class=resource_class, gpu_ids=ids):
            return {"enabled": False, "acquired": True}
        assignment = self._assignment_mode()
        meta = {
            **dict(metadata or {}),
            **GPULeaseStore.current_owner_metadata(),
            "assignment": assignment,
            "resource_class": str(resource_class or ""),
            "policy_resource_class": policy.resource_class,
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "max_per_gpu": policy.max_per_gpu,
            "incompatible_classes": list(policy.incompatible_classes),
        }
        value_hint = meta.get("value_hint") if isinstance(meta.get("value_hint"), dict) else {}
        priority = self._admission_priority(value_hint)
        meta["admission_priority"] = priority
        if assignment == "lease":
            candidate_ids = self._configured_gpu_pool() or ids
            sublease = self._gpu_sublease_plan(
                candidate_gpu_ids=candidate_ids,
                request_count=request_count,
                metadata=meta,
            )
            requested_count = int(sublease.get("requested_gpu_count") or 1)
            meta["gpu_sublease"] = dict(sublease)
        else:
            candidate_ids = ids
            requested_count = max(1, int(request_count or len(candidate_ids) or 1))
            sublease = {
                "requested_gpu_count": requested_count,
                "candidate_gpu_ids": list(candidate_ids),
                "reason": "direct_assignment",
            }

        cached_backoff = self._cached_admission_backoff(
            job_id=str(job_id or ""),
            policy=policy,
            target_gpu_ids=candidate_ids or ids,
            priority=priority,
        )
        if cached_backoff is not None:
            return cached_backoff

        min_free = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        if min_free > 0 and candidate_ids:
            safety_sample = sample_nvidia_smi(candidate_ids)
            safety = evaluate_admission_safety(
                sample=safety_sample if isinstance(safety_sample, dict) else {},
                gpu_ids=candidate_ids,
                resource_class=policy.resource_class,
                request_count=requested_count,
                assignment=assignment,
                min_free_mem_gb=min_free,
                free_mem_buffer_gb=max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0)),
            )
            if safety.get("blocked"):
                queue_started_first = False
                if job_id not in self._queue_started:
                    self._queue_started.add(str(job_id))
                    queue_started_first = True
                pressure_ids = [str(x) for x in (safety.get("blocked_gpu_ids") or candidate_ids) if str(x).strip()]
                pressure = self.pressure_store.record_yellow(
                    gpu_ids=self._pressure_gpu_ids(pressure_ids),
                    worker_id=self.worker_id,
                    job_id=str(job_id),
                    resource_class=policy.resource_class,
                    reason=str(safety.get("reason") or "gpu_memory_below_admission_reserve"),
                    queue_len=0,
                    queue_position=1,
                    metadata={
                        "free_mem_threshold_gb": safety.get("free_mem_threshold_gb"),
                        "blocked_gpu_ids": safety.get("blocked_gpu_ids") or [],
                    },
                )
                deferred = build_admission_safety_deferred_result(
                    resource_class=str(resource_class or ""),
                    policy_resource_class=policy.resource_class,
                    gpu_ids=pressure_ids or candidate_ids,
                    safety=safety,
                    max_wait_sec=float(self.gpu_queue.max_wait_sec),
                    heartbeat_sec=float(self.gpu_queue.heartbeat_sec),
                    slot_weight=policy.slot_weight,
                    capacity_slots=self._capacity_slots(),
                    details={
                        "reason": str(safety.get("reason") or "gpu_memory_below_admission_reserve"),
                        "candidate_gpu_ids": candidate_ids,
                        "blocked_gpu_ids": safety.get("blocked_gpu_ids") or [],
                        "safety": safety,
                        "gpu_sublease": dict(sublease),
                    },
                    value_hint=value_hint,
                    priority=priority,
                    queue_started_first=queue_started_first,
                    pressure=pressure,
                    eta_next_train_sec=self._cold_eta_sec(),
                )
                allowed_pool_ids = [str(x) for x in ((self._configured_gpu_pool() or ids or candidate_ids) if assignment == "lease" else ids) if str(x).strip()]
                opportunity = self.admission_opportunity_facts(
                    resource_class=policy.resource_class,
                    gpu_ids=candidate_ids or allowed_pool_ids,
                    request_count=requested_count,
                )
                deferred.update({
                    "assigned_physical_gpus": [],
                    "allowed_physical_gpus": allowed_pool_ids,
                    "candidate_physical_gpus": candidate_ids,
                    "requested_gpu_count": requested_count,
                    "gpu_sublease": dict(sublease),
                    "admission_opportunity": opportunity,
                    "lease_grantable_by_llm": bool(opportunity.get("lease_grantable_by_llm")),
                })
                return deferred
            if assignment == "lease":
                safe_candidate_ids = [str(x) for x in (safety.get("safe_gpu_ids") or candidate_ids) if str(x).strip()]
                if safe_candidate_ids:
                    candidate_ids = safe_candidate_ids
                    requested_count = self._request_count(len(candidate_ids), requested=requested_count)
                    sublease = {**sublease, "candidate_gpu_ids": list(candidate_ids), "requested_gpu_count": requested_count}
                    meta["gpu_sublease"] = dict(sublease)

        if assignment == "lease":
            acquired, details = self.gpu_store.try_acquire_any(
                job_id=str(job_id),
                worker_id=self.worker_id,
                candidate_gpu_ids=candidate_ids,
                request_count=requested_count,
                max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
                metadata=meta,
                resource_class=policy.resource_class,
                slot_weight=policy.slot_weight,
                capacity_slots=self._capacity_slots(),
                max_per_gpu=policy.max_per_gpu,
                incompatible_classes=list(policy.incompatible_classes),
            )
        else:
            acquired, details = self.gpu_store.try_acquire(
                job_id=str(job_id),
                worker_id=self.worker_id,
                gpu_ids=ids,
                max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
                metadata={**meta, "request_count": request_count},
                resource_class=policy.resource_class,
                slot_weight=policy.slot_weight,
                capacity_slots=self._capacity_slots(),
                max_per_gpu=policy.max_per_gpu,
                incompatible_classes=list(policy.incompatible_classes),
            )
        lease = details.get("lease") if isinstance(details.get("lease"), dict) else {}
        assigned_gpu_ids = [str(x) for x in (lease.get("gpu_ids") or details.get("assigned_gpu_ids") or ids) if str(x).strip()]
        allowed_physical_gpu_ids = [str(x) for x in ((self._configured_gpu_pool() or ids or candidate_ids) if assignment == "lease" else ids) if str(x).strip()]
        if acquired and assignment == "lease" and allowed_physical_gpu_ids and assigned_gpu_ids and not set(assigned_gpu_ids).issubset(set(allowed_physical_gpu_ids)):
            release = self.gpu_store.release(job_id=str(job_id or ""))
            self._queue_started.discard(str(job_id or ""))
            self._last_heartbeat.pop(str(job_id or ""), None)
            self._leased_gpu_ids.pop(str(job_id or ""), None)
            boundary_result = {
                "enabled": True,
                "acquired": False,
                "status": "REPLAN",
                "admission_action": "REPLAN",
                "reason": "assigned_gpu_outside_task_pool",
                "resource_mode": "BOUNDARY_VIOLATION",
                "max_wait_sec": float(self.gpu_queue.max_wait_sec),
                "heartbeat_sec": float(self.gpu_queue.heartbeat_sec),
                "gpu_ids": assigned_gpu_ids,
                "assigned_physical_gpus": assigned_gpu_ids,
                "allowed_physical_gpus": allowed_physical_gpu_ids,
                "candidate_physical_gpus": candidate_ids,
                "resource_class": str(resource_class or ""),
                "policy_resource_class": policy.resource_class,
                "slot_weight": policy.slot_weight,
                "capacity_slots": self._capacity_slots(),
                "details": {**details, "release": release},
                "queue_started_first": False,
                "queue_position": 0,
                "queue_len": 0,
                "allowed_classes": [RESOURCE_PURE_TT_CPU, *_CPU_SUPPORT_CLASSES],
                "feedback": (
                    "RESOURCE_FEEDBACK: REPLAN because assigned_gpu_outside_task_pool; "
                    f"assigned_gpu={','.join(assigned_gpu_ids)}; allowed_gpu={','.join(allowed_physical_gpu_ids)}.\n"
                ),
                "value_hint": value_hint,
                **priority,
            }
            self.record_resource_event(
                "admission_boundary_violation",
                payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": boundary_result},
                command_id=str(job_id or ""),
                lease_id=str(job_id or ""),
            )
            return boundary_result
        if acquired and assigned_gpu_ids:
            self._leased_gpu_ids[str(job_id)] = assigned_gpu_ids
        env_updates = {}
        if acquired and assignment == "lease" and assigned_gpu_ids:
            env_updates = {
                "CUDA_VISIBLE_DEVICES": ",".join(assigned_gpu_ids),
                "SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL": ",".join(assigned_gpu_ids),
                "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL": ",".join(str(i) for i, _ in enumerate(assigned_gpu_ids)),
            }
            if allowed_physical_gpu_ids:
                env_updates["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] = ",".join(allowed_physical_gpu_ids)
        result = {
            "enabled": True,
            "acquired": bool(acquired),
            "reason": str(details.get("reason") or ""),
            "max_wait_sec": float(self.gpu_queue.max_wait_sec),
            "heartbeat_sec": float(self.gpu_queue.heartbeat_sec),
            "gpu_ids": assigned_gpu_ids or ids,
            "assigned_physical_gpus": assigned_gpu_ids,
            "allowed_physical_gpus": allowed_physical_gpu_ids,
            "candidate_physical_gpus": candidate_ids,
            "requested_gpu_count": requested_count,
            "gpu_sublease": dict(sublease),
            "env_updates": env_updates,
            "resource_class": str(resource_class or ""),
            "policy_resource_class": policy.resource_class,
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "details": details,
            "queue_started_first": False,
            "queue_position": details.get("queue_position"),
            "queue_len": details.get("queue_len"),
            "top_waiter_job_id": details.get("top_waiter_job_id"),
            "value_hint": value_hint,
            **priority,
        }
        if acquired:
            result["status"] = "GRANTED"
            result["admission_action"] = "RUN_NOW"
            result["queue_position"] = int(details.get("queue_position") or 1)
            result["queue_len"] = int(details.get("queue_len") or 0)
            lease_meta = lease.get("metadata") if isinstance(lease.get("metadata"), dict) else {}
            if str(lease_meta.get("lease_mode") or "") == "shared_secondary":
                self.record_resource_event(
                    "shared_gpu_lease_consumed",
                    payload={
                        "job_id": str(job_id or ""),
                        "primary_job_id": str(lease_meta.get("shared_primary_job_id") or ""),
                        "gpu_ids": assigned_gpu_ids,
                        "reason": str(details.get("reason") or "already_acquired"),
                        "lease": lease,
                    },
                    command_id=str(job_id or ""),
                    lease_id=str(job_id or ""),
                )
            self.record_resource_event(
                "admission_granted",
                payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
                command_id=str(job_id or ""),
                lease_id=str(job_id or ""),
            )
            return result
        if job_id not in self._queue_started:
            self._queue_started.add(job_id)
            result["queue_started_first"] = True
        eta_info = self._eta_for_blocked(
            resource_class=policy.resource_class,
            details=details,
            command_digest=str(meta.get("command_digest") or ""),
            entrypoint=str(meta.get("entrypoint") or ""),
        )
        share_override_candidate = self._admission_share_override_candidate(
            job_id=str(job_id or ""),
            policy=policy,
            details=details,
            priority=priority,
            value_hint=value_hint,
        )
        share_override_record = {"updated": False}
        if share_override_candidate.get("candidate"):
            try:
                share_override_record = self.gpu_store.annotate_waiter_share_override(
                    job_id=str(job_id or ""),
                    share_override=share_override_candidate,
                )
            except Exception as exc:
                share_override_record = {"updated": False, "reason": f"share_override_record_failed:{type(exc).__name__}"}
        eta = float(eta_info.get("eta_next_train_sec") or self._cold_eta_sec())
        eta_confidence = str(eta_info.get("eta_confidence") or "low")
        eta_source = str(eta_info.get("eta_source") or "cold_start")
        pressure_ids = assigned_gpu_ids or ids or [
            str(x) for x in (details.get("candidate_gpu_ids") or details.get("blocked_gpu_ids") or []) if str(x).strip()
        ]
        yellow_pressure = self.pressure_store.record_yellow(
            gpu_ids=self._pressure_gpu_ids(pressure_ids),
            worker_id=self.worker_id,
            job_id=str(job_id),
            resource_class=policy.resource_class,
            reason=str(details.get("reason") or "resource_admission_deferred"),
            queue_len=int(result.get("queue_len") or 0),
            queue_position=int(result.get("queue_position") or 1),
            metadata={"top_waiter_job_id": str(result.get("top_waiter_job_id") or "")},
        )
        admission_action, admission_reason = self._admission_action_for_blocked(
            policy=policy,
            requested_count=requested_count,
            details=details,
            priority=priority,
            eta_next_train_sec=eta,
        )
        queue_position = int(result.get("queue_position") or 1)
        queue_len = int(result.get("queue_len") or 0)
        target_gpu_ids = assigned_gpu_ids or ids or self._pressure_gpu_ids(gpu_ids)
        blocker_job_ids = [str(x) for x in (eta_info.get("active_blocker_job_ids") or []) if str(x).strip()]
        backoff_key = self._admission_backoff_key(policy_resource_class=policy.resource_class, gpu_ids=target_gpu_ids)
        backoff_due = (
            admission_action == "PENDING"
            and not bool(share_override_candidate.get("candidate"))
            and self._waiter_wait_age_sec(details) >= self._admission_backoff_grace_sec()
            and policy.resource_class in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC}
        )
        retry_after_sec: float | None = None
        if admission_action == "REPLAN" or backoff_due:
            self.gpu_store.release(job_id=str(job_id or ""))
            self._queue_started.discard(str(job_id or ""))
            result["queue_started_first"] = False
        if backoff_due:
            admission_action = "DENIED_REPLAN"
            admission_reason = "cached_gpu_admission_backoff"
            retry_after_sec = self._admission_backoff_retry_sec(eta)
        wait_option: dict[str, Any] = {}
        if admission_action == "PENDING" and not backoff_due:
            wait_option = self.create_resource_wait_option(
                job_id=str(job_id or ""),
                resource_class=policy.resource_class,
                gpu_ids=target_gpu_ids,
                holder_job_ids=blocker_job_ids,
                queue_position=queue_position,
                queue_len=queue_len,
                command_digest=str(meta.get("command_digest") or ""),
                reason=admission_reason,
                max_wait_sec=min(600.0, max(60.0, eta if eta > 0 else 600.0)),
            )
        result.update({
            "status": admission_action,
            "admission_action": admission_action,
            "reason": admission_reason,
            "resource_mode": "YELLOW",
            "queue_position": queue_position,
            "queue_len": queue_len,
            "top_waiter_job_id": str(result.get("top_waiter_job_id") or ""),
            "eta_next_train_sec": eta,
            "eta_confidence": eta_confidence,
            "holder_job_ids": blocker_job_ids,
            "allowed_classes": _GPU_PRESSURE_SUPPORT_CLASSES,
            "pressure": yellow_pressure,
            "duplicate_digest_summary": self._duplicate_summary(details),
            "share_override_candidate": bool(share_override_candidate.get("candidate")),
            "share_override": share_override_candidate,
            "share_override_record": share_override_record,
            "admission_cached_backoff": bool(backoff_due),
            "retry_after_sec": retry_after_sec,
            "blocked_until_unlock": bool(backoff_due),
            "unlock_condition": "holder_released" if backoff_due else "",
            "post_feedback_action": "cpu_support" if backoff_due else "",
            "resource_wait_option": wait_option,
            **eta_info,
            "feedback": self._admission_feedback(
                action=admission_action,
                reason=admission_reason,
                policy_resource_class=policy.resource_class,
                gpu_ids=target_gpu_ids,
                queue_position=queue_position,
                queue_len=queue_len,
                eta_next_train_sec=eta,
                eta_confidence=eta_confidence,
                priority_score=float(priority.get("admission_priority_score") or 0.0),
                holder_job_id=(blocker_job_ids[0] if blocker_job_ids else ""),
                retry_after_sec=retry_after_sec,
                post_feedback_action="cpu_support" if backoff_due else "",
                cached_backoff=bool(backoff_due),
                wait_option=wait_option,
            ),
        })
        opportunity = self.admission_opportunity_facts(
            resource_class=policy.resource_class,
            gpu_ids=target_gpu_ids or candidate_ids,
            request_count=requested_count,
        )
        result["admission_opportunity"] = opportunity
        result["lease_grantable_by_llm"] = bool(opportunity.get("lease_grantable_by_llm"))
        if backoff_due:
            self._remember_admission_backoff(
                key=backoff_key,
                policy=policy,
                target_gpu_ids=target_gpu_ids,
                result=result,
                holder_job_ids=blocker_job_ids,
                eta_next_train_sec=eta,
                eta_confidence=eta_confidence,
                pressure_generation=int(yellow_pressure.get("generation") or self._current_pressure_generation()),
            )
            self.record_resource_event(
                "admission_cached_backoff",
                payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
                command_id=str(job_id or ""),
                lease_id=str(job_id or ""),
            )
        if share_override_candidate.get("candidate"):
            self.record_resource_event(
                "admission_share_override_candidate",
                payload={
                    "job_id": str(job_id or ""),
                    "resource_type": "gpu",
                    "result": result,
                    "share_override": share_override_candidate,
                    "share_override_record": share_override_record,
                },
                command_id=str(job_id or ""),
                lease_id=str(job_id or ""),
            )
        self.record_resource_event(
            "admission_pending" if admission_action == "PENDING" else "admission_replan",
            payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return result

    def queue_wait_heartbeat(self, *, job_id: str, elapsed_sec: float) -> dict[str, Any]:
        if not job_id or job_id not in self._queue_started:
            return {"emit": False}
        now = time.monotonic()
        last = self._last_heartbeat.get(job_id, 0.0)
        if now - last < max(1.0, float(self.gpu_queue.heartbeat_sec)):
            return {"emit": False}
        self._last_heartbeat[job_id] = now
        return {"emit": True, "elapsed_sec": float(elapsed_sec or 0.0)}

    def queue_timeout(self, *, job_id: str, elapsed_sec: float, reason: str) -> dict[str, Any]:
        self._queue_started.discard(str(job_id or ""))
        self._last_heartbeat.pop(str(job_id or ""), None)
        self._leased_gpu_ids.pop(str(job_id or ""), None)
        release = self.gpu_store.release(job_id=str(job_id or ""))
        result = {
            "released": bool(release.get("released")),
            "elapsed_sec": float(elapsed_sec or 0.0),
            "reason": str(reason or "queue_timeout"),
            "release": release,
        }
        self.record_resource_event(
            "queue_timeout",
            payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return result

    def record_queue_timeout_pressure(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        elapsed_sec: float,
        reason: str,
        command_digest: str = "",
    ) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        return self.pressure_store.record_queue_timeout(
            gpu_ids=ids,
            worker_id=self.worker_id,
            job_id=str(job_id or ""),
            resource_class=str(resource_class or ""),
            elapsed_sec=float(elapsed_sec or 0.0),
            reason=str(reason or "queue_timeout"),
            command_digest=str(command_digest or ""),
        )

    def record_policy_gate_digest(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        command_digest: str,
        reason: str,
    ) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        return self.pressure_store.record_digest_failure(
            gpu_ids=ids,
            worker_id=self.worker_id,
            job_id=str(job_id or ""),
            resource_class=str(resource_class or ""),
            command_digest=str(command_digest or ""),
            reason=str(reason or "resource_policy_gate"),
        )

    def _yellow_safety_hold(self, *, gpu_ids: list[str]) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not ids:
            return {"blocked": False, "gpu_ids": []}
        sample = sample_nvidia_smi(ids)
        reserve = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        return evaluate_yellow_safety_hold(
            sample=sample if isinstance(sample, dict) else {},
            gpu_ids=ids,
            util_exit_threshold_pct=50.0,
            min_free_mem_gb=reserve,
        )

    @staticmethod
    def _lease_record_gpu_ids(raw: dict[str, Any]) -> set[str]:
        return {str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()} if isinstance(raw, dict) else set()

    def _active_gpu_reservations(self, *, gpu_ids: list[str]) -> dict[str, Any]:
        ids = {str(x) for x in (gpu_ids or []) if str(x).strip()}
        if not ids:
            return {"active": False, "gpu_ids": []}
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception as exc:
            return {"active": True, "reason": "lease_snapshot_failed", "error": str(exc), "gpu_ids": sorted(ids)}
        active: list[str] = []
        released_idle: list[str] = []
        for job_id, raw in (snapshot.get("leases") or {}).items():
            if ids & self._lease_record_gpu_ids(raw if isinstance(raw, dict) else {}):
                active.append(str(job_id))
        for job_id, raw in (snapshot.get("released_idle") or {}).items():
            if ids & self._lease_record_gpu_ids(raw if isinstance(raw, dict) else {}):
                released_idle.append(str(job_id))
        return {
            "active": bool(active or released_idle),
            "gpu_ids": sorted(ids),
            "active_lease_job_ids": active,
            "released_idle_job_ids": released_idle,
        }

    def reconcile_free_gpu_pressure(
        self,
        *,
        gpu_ids: list[str],
        reason: str = "observed_free_no_active_lease",
        sample: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        if not ids:
            return {"cleared": False, "reason": "no_gpu_ids", "gpu_ids": []}
        reservations = self._active_gpu_reservations(gpu_ids=ids)
        if reservations.get("active"):
            return {"cleared": False, "reason": "active_gpu_reservation_present", "gpu_ids": ids, "reservations": reservations}
        observed = sample if isinstance(sample, dict) else sample_nvidia_smi(ids)
        if not isinstance(observed, dict) or observed.get("available") is False:
            return {"cleared": False, "reason": "gpu_sample_unavailable", "gpu_ids": ids, "observed": observed or {}}
        rows = observed.get("gpus") if isinstance(observed.get("gpus"), list) else []
        by_id = {str(row.get("gpu_id") or row.get("index") or ""): row for row in rows if isinstance(row, dict)}
        missing = [gpu_id for gpu_id in ids if gpu_id not in by_id]
        if missing:
            return {"cleared": False, "reason": "gpu_sample_missing_ids", "gpu_ids": ids, "missing_gpu_ids": missing, "observed": observed}
        busy: list[dict[str, Any]] = []
        for gpu_id in ids:
            row = by_id[gpu_id]
            try:
                util = float(row.get("utilization_gpu_pct") or 0.0)
            except (TypeError, ValueError):
                util = 0.0
            try:
                used_mb = float(row.get("memory_used_mb") or 0.0)
            except (TypeError, ValueError):
                used_mb = 0.0
            if util > 1.0 or used_mb > 1024.0:
                busy.append({"gpu_id": gpu_id, "utilization_gpu_pct": util, "memory_used_mb": used_mb})
        if busy:
            return {"cleared": False, "reason": "gpu_not_observed_free", "gpu_ids": ids, "busy_gpus": busy, "observed": observed}
        cleared = self.pressure_store.clear_global_pressure(
            gpu_ids=ids,
            reason=str(reason or "observed_free_no_active_lease"),
            observed={"sample": observed, "reservations": reservations},
        )
        if cleared.get("cleared"):
            self.record_resource_event(
                "gpu_pressure_reconciled_free",
                payload={"gpu_ids": ids, "reason": str(reason or "observed_free_no_active_lease"), "result": cleared},
            )
        return cleared

    def pressure_gate_decision(self, *, resource_class: str, gpu_ids: list[str], command_digest: str = "") -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        if not ids:
            return {"blocked": False, "gpu_ids": [], "pressure": {}}
        duplicate = self.pressure_store.duplicate_gate_decision(gpu_ids=ids, command_digest=str(command_digest or ""))
        if isinstance(duplicate, dict) and duplicate.get("blocked"):
            return {
                **duplicate,
                "allowed_after_gate": "changed command or plan",
                "allowed_classes": ["pure_tt_cpu", "readonly_cpu", "light_cpu"],
                "resource_mode": "DUPLICATE_COOLDOWN",
            }
        reconcile = self.reconcile_free_gpu_pressure(gpu_ids=ids, reason="observed_free_before_pressure_gate")
        snapshot = self.pressure_store.snapshot(gpu_ids=ids)
        if reconcile.get("cleared"):
            snapshot = self.pressure_store.snapshot(gpu_ids=ids)
        red = {
            gpu_id: entry
            for gpu_id, entry in (snapshot.get("gpus") or {}).items()
            if isinstance(entry, dict) and entry.get("mode") == "RED"
        }
        cls = str(resource_class or "")
        if not red:
            yellow = {
                gpu_id: entry
                for gpu_id, entry in (snapshot.get("gpus") or {}).items()
                if isinstance(entry, dict) and entry.get("mode") == "YELLOW"
            }
            if yellow and cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
                safety = self._yellow_safety_hold(gpu_ids=sorted(yellow))
                if safety.get("blocked"):
                    return {
                        "blocked": True,
                        "status": "PENDING",
                        "reason": "yellow_pressure_exit_guard",
                        "allowed_after_gate": "support work until GPU leaves yellow safety hold",
                        "allowed_classes": _GPU_PRESSURE_SUPPORT_CLASSES,
                        "gpu_ids": ids,
                        "red_gpu_ids": sorted(safety.get("blocked_gpu_ids") or sorted(yellow)),
                        "max_queue_timeout_count": max(int((entry or {}).get("queue_timeout_count") or 0) for entry in yellow.values()),
                        "cooldown_remaining_sec": max(float((entry or {}).get("yellow_remaining_sec") or 0.0) for entry in yellow.values()),
                        "eta_next_train_sec": max(300.0, max(float((entry or {}).get("yellow_remaining_sec") or 0.0) for entry in yellow.values())),
                        "eta_confidence": "low",
                        "resource_mode": "YELLOW",
                        "pressure": snapshot,
                        "safety": safety,
                    }
            return {"blocked": False, "gpu_ids": ids, "pressure": snapshot}
        counts = [int((entry or {}).get("queue_timeout_count") or 0) for entry in red.values()]
        max_count = max(counts or [0])
        if max_count >= 2:
            blocked = cls not in set(_GPU_PRESSURE_SUPPORT_CLASSES)
            reason = "tt_only_after_gpu_pressure"
            allowed = "pure_tt_cpu,gpu_tt_light,heavy_cpu_candidate,readonly_cpu,light_cpu"
            allowed_classes = _GPU_PRESSURE_SUPPORT_CLASSES
        else:
            blocked = cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}
            reason = "block_train_after_gpu_pressure"
            allowed = "non-training work"
            allowed_classes = _GPU_PRESSURE_SUPPORT_CLASSES
        cooldown = max(float((entry or {}).get("cooldown_remaining_sec") or 0.0) for entry in red.values())
        return {
            "blocked": bool(blocked),
            "status": "DENIED_REPLAN",
            "reason": reason,
            "allowed_after_gate": allowed,
            "allowed_classes": allowed_classes,
            "gpu_ids": ids,
            "red_gpu_ids": sorted(red),
            "max_queue_timeout_count": max_count,
            "cooldown_remaining_sec": cooldown,
            "eta_next_train_sec": cooldown + 300.0,
            "eta_confidence": "low",
            "resource_mode": "RED",
            "pressure": snapshot,
        }

    def record_runtime_pressure(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        elapsed_sec: float,
        reason: str,
        command_digest: str = "",
    ) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        return self.pressure_store.record_runtime_pressure(
            gpu_ids=ids,
            worker_id=self.worker_id,
            job_id=str(job_id or ""),
            resource_class=str(resource_class or ""),
            elapsed_sec=float(elapsed_sec or 0.0),
            reason=str(reason or "runtime_pressure"),
            command_digest=str(command_digest or ""),
        )

    def pressure_generation(self) -> int:
        snapshot = self.pressure_store.snapshot()
        return int(snapshot.get("generation") or 0)

    def pressure_snapshot(self, *, gpu_ids: list[str] | None = None) -> dict[str, Any]:
        return self.pressure_store.snapshot(gpu_ids=gpu_ids)

    def has_active_lease(self, *, job_id: str) -> bool:
        return str(job_id or "") in self._leased_gpu_ids

    def persisted_lease(self, *, job_id: str) -> dict[str, Any]:
        if not job_id:
            return {}
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception:
            return {}
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        raw = leases.get(str(job_id)) if isinstance(leases.get(str(job_id)), dict) else {}
        return dict(raw or {})

    def grant_shared_gpu_lease(
        self,
        *,
        primary_job_id: str,
        secondary_job_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.gpu_store.grant_shared_secondary(
            primary_job_id=str(primary_job_id or ""),
            secondary_job_id=str(secondary_job_id or ""),
            metadata=dict(metadata or {}),
            max_secondary_per_primary=1,
        )
        lease = result.get("lease") if isinstance(result.get("lease"), dict) else {}
        assigned = [str(x) for x in (lease.get("gpu_ids") or result.get("assigned_gpu_ids") or []) if str(x).strip()]
        if result.get("acquired") and assigned:
            self.pressure_store.bump_generation(reason="shared_gpu_lease_granted")
            self.record_resource_event(
                "gpu_share_decision",
                payload={
                    "proposal_type": "task_gpu_share_review",
                    "action": "GRANT_SHARED_GPU_LEASE",
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "result": result,
                },
                command_id=str(primary_job_id or ""),
                lease_id=str(primary_job_id or ""),
            )
            self.record_resource_event(
                "shared_gpu_lease_granted",
                payload={
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "gpu_ids": assigned,
                    "lease_mode": "shared_secondary",
                    "result": result,
                },
                command_id=str(secondary_job_id or ""),
                lease_id=str(secondary_job_id or ""),
            )
        elif not result.get("acquired"):
            self.record_resource_event(
                "gpu_share_decision",
                payload={
                    "proposal_type": "task_gpu_share_review",
                    "action": "DENY_SHARE_USE_CPU_SUPPORT",
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "reason": str(result.get("reason") or "shared_lease_denied"),
                    "result": result,
                },
                command_id=str(primary_job_id or ""),
                lease_id=str(primary_job_id or ""),
            )
        return {**result, "assigned_gpu_ids": assigned}

    def revoke_shared_gpu_lease(
        self,
        *,
        primary_job_id: str,
        secondary_job_id: str,
        reason: str,
    ) -> dict[str, Any]:
        release = self.gpu_store.release(job_id=str(secondary_job_id or ""))
        if release.get("released"):
            self.pressure_store.bump_generation(reason="shared_gpu_lease_revoked")
            self.record_resource_event(
                "shared_gpu_secondary_stopped",
                payload={
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "reason": str(reason or "shared_runtime_review"),
                    "release": release,
                },
                command_id=str(secondary_job_id or ""),
                lease_id=str(secondary_job_id or ""),
            )
            self.record_resource_event(
                "shared_gpu_lease_revoked",
                payload={
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "reason": str(reason or "shared_runtime_review"),
                    "release": release,
                },
                command_id=str(secondary_job_id or ""),
                lease_id=str(secondary_job_id or ""),
            )
        return release

    def resource_available_for(self, *, resource_class: str, gpu_ids: list[str] | None = None) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        cls = str(resource_class or RESOURCE_HEAVY_GPU_TRAIN)
        policy = self._policy_for_resource_class(cls)
        if policy is None:
            return {"available": True, "reason": "unmanaged_resource_class", "gpu_ids": ids}
        candidates = ids or self._configured_gpu_pool()
        if not candidates:
            return {"available": True, "reason": "resource_changed_no_gpu_candidates", "gpu_ids": [], "generation": self.pressure_generation()}
        pressure = self.pressure_gate_decision(resource_class=policy.resource_class, gpu_ids=candidates)
        if isinstance(pressure, dict) and pressure.get("blocked"):
            return {
                "available": False,
                "reason": str(pressure.get("reason") or "resource_pressure"),
                "gpu_ids": candidates,
                "pressure": pressure.get("pressure") or {},
                "generation": self.pressure_generation(),
            }
        availability = self.gpu_store.availability(
            candidate_gpu_ids=candidates,
            request_count=self._request_count(len(candidates), requested=None),
            max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
            resource_class=policy.resource_class,
            slot_weight=policy.slot_weight,
            capacity_slots=self._capacity_slots(),
            max_per_gpu=policy.max_per_gpu,
            incompatible_classes=list(policy.incompatible_classes),
        )
        return {
            **availability,
            "reason": "resource_available" if availability.get("available") else "gpu_slot_unavailable",
            "gpu_ids": candidates,
            "generation": self.pressure_generation(),
        }

    def release_idle_lease(
        self,
        *,
        job_id: str,
        elapsed_sec: float = 0.0,
        reason: str = "idle_gpu_lease",
        resident_mem_gb: float = 0.0,
    ) -> dict[str, Any]:
        """Release a GPU lease without treating the running process as finished."""
        jid = str(job_id or "")
        self._queue_started.discard(jid)
        self._last_heartbeat.pop(jid, None)
        self._leased_gpu_ids.pop(jid, None)
        release = self.gpu_store.release_idle(
            job_id=jid,
            elapsed_sec=float(elapsed_sec or 0.0),
            reason=str(reason or "idle_gpu_lease"),
            resident_mem_gb=max(0.0, float(resident_mem_gb or 0.0)),
        )
        result = {
            "released": bool(release.get("released")),
            "elapsed_sec": float(elapsed_sec or 0.0),
            "reason": str(reason or "idle_gpu_lease"),
            "release": release,
        }
        if release.get("released"):
            generation = self.pressure_store.bump_generation(reason="idle_lease_released")
            result["pressure_generation"] = generation
            self.record_resource_event(
                "resource_idle_lease_released",
                payload={"job_id": jid, "resource_type": "gpu", "result": result, "idle_release_admission_mode": self.gpu_store.idle_release_admission_mode},
                command_id=jid,
                lease_id=jid,
            )
            self.record_resource_event(
                "resource_release",
                payload={
                    "job_id": jid,
                    "resource_type": "gpu",
                    "status": "idle_lease_released",
                    "reason": str(reason or "idle_gpu_lease"),
                    "release": release,
                },
                command_id=jid,
                lease_id=jid,
            )
        else:
            self.sync_resource_state(reason="idle_lease_release_not_found")
        return result

    def release(self, *, job_id: str, elapsed_sec: float = 0.0, status: str = "finished") -> dict[str, Any]:
        self._queue_started.discard(str(job_id or ""))
        self._last_heartbeat.pop(str(job_id or ""), None)
        self._leased_gpu_ids.pop(str(job_id or ""), None)
        release = self.gpu_store.release(job_id=str(job_id or ""))
        if release.get("released"):
            generation = self.pressure_store.bump_generation(reason="lease_released")
            release["pressure_generation"] = generation
            lease = release.get("lease") if isinstance(release.get("lease"), dict) else {}
            meta = lease.get("metadata") if isinstance(lease.get("metadata"), dict) else {}
            resource_class = str(meta.get("policy_resource_class") or meta.get("resource_class") or "")
            history = self.history_store.record_completion(
                job_id=str(job_id or ""),
                worker_id=self.worker_id,
                resource_class=resource_class,
                gpu_ids=[str(x) for x in (lease.get("gpu_ids") or []) if str(x).strip()],
                elapsed_sec=float(elapsed_sec or 0.0),
                status=str(status or "finished"),
                command_digest=str(meta.get("command_digest") or ""),
                entrypoint=str(meta.get("entrypoint") or ""),
            )
            release["runtime_history"] = history
            released_gpu_ids = [str(x) for x in (lease.get("gpu_ids") or []) if str(x).strip()]
            if released_gpu_ids:
                release["pressure_reconcile"] = self.reconcile_free_gpu_pressure(
                    gpu_ids=released_gpu_ids,
                    reason="lease_released_observed_free",
                )
        if release.get("released"):
            self.record_resource_event(
                "resource_release",
                payload={"job_id": str(job_id or ""), "resource_type": "gpu", "status": str(status or "finished"), "release": release},
                command_id=str(job_id or ""),
                lease_id=str(job_id or ""),
            )
        else:
            self.sync_resource_state(reason="resource_release_not_found")
        return release

    def env_updates(self, *, job_id: str) -> dict[str, str]:
        ids = self._leased_gpu_ids.get(str(job_id or "")) or []
        if self._assignment_mode() != "lease" or not ids:
            return {}
        pool = self._configured_gpu_pool() or ids
        return {
            "CUDA_VISIBLE_DEVICES": ",".join(ids),
            "SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL": ",".join(ids),
            "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL": ",".join(str(i) for i, _ in enumerate(ids)),
            "SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL": ",".join(pool),
        }

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out == out else None

    def _record_yellow_from_util_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        if not rows:
            return {"recorded": False, "reason": "no_gpu_rows", "gpu_ids": []}
        util_threshold = max(0.0, float(self.gpu_queue.pressure_yellow_util_pct or 0.0))
        min_free = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        free_threshold = min_free + max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0))
        yellow_ids: list[str] = []
        reasons: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if not gpu_id:
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            used_mb = self._float_or_none(row.get("memory_used_mb"))
            total_mb = self._float_or_none(row.get("memory_total_mb"))
            reason_parts: list[str] = []
            if util is not None and util_threshold > 0 and util >= util_threshold:
                reason_parts.append("util_near_limit")
            if used_mb is not None and total_mb is not None and total_mb > 0 and free_threshold > 0:
                free_gb = max(0.0, (total_mb - used_mb) / 1024.0)
                if free_gb < free_threshold:
                    reason_parts.append("free_mem_near_reserve")
            if reason_parts:
                yellow_ids.append(gpu_id)
                reasons[gpu_id] = "+".join(reason_parts)
        if not yellow_ids:
            return {"recorded": False, "reason": "below_yellow_thresholds", "gpu_ids": []}
        return self.pressure_store.record_yellow(
            gpu_ids=yellow_ids,
            worker_id=self.worker_id,
            job_id="gpu_util_sample",
            resource_class="gpu_util_sample",
            reason="gpu_util_or_memory_near_limit",
            metadata={
                "gpu_reasons": reasons,
                "util_threshold_pct": util_threshold,
                "free_mem_threshold_gb": free_threshold,
            },
        )

    def sample_gpu_util(self, *, gpu_ids: list[str]) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        sample = sample_nvidia_smi(ids)
        if isinstance(sample, dict) and sample.get("available") is not False:
            pressure = self._record_yellow_from_util_sample(sample)
            if pressure.get("recorded"):
                sample = {**sample, "pressure": pressure}
        return sample
