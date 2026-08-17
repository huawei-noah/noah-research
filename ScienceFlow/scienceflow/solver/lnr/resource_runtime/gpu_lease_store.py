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

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import ResourceLease


class GPULeaseStore:
    """Atomic task-level GPU lease and admission-waiter file.

    The file lives under task-level logs, not inside an agent workspace, so it is
    not exposed to the model or captured by workspace source checkpoints.
    """

    def __init__(
        self,
        path: Path,
        *,
        lease_ttl_sec: float = 7200.0,
        admission_queue_enabled: bool = True,
        waiter_ttl_sec: float = 900.0,
        idle_release_admission_mode: str = "strict_exclusive",
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.lease_ttl_sec = max(0.0, float(lease_ttl_sec or 0.0))
        self.admission_queue_enabled = bool(admission_queue_enabled)
        self.waiter_ttl_sec = max(30.0, float(waiter_ttl_sec or 900.0))
        mode = str(idle_release_admission_mode or "strict_exclusive").strip().lower()
        self.idle_release_admission_mode = mode if mode in {"strict_exclusive", "measured_share"} else "strict_exclusive"

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"version": 4, "leases": {}, "waiters": {}, "released_idle": {}, "reaped": [], "reaped_waiters": []}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        if not isinstance(data.get("leases"), dict):
            data["leases"] = {}
        if not isinstance(data.get("waiters"), dict):
            data["waiters"] = {}
        if not isinstance(data.get("released_idle"), dict):
            data["released_idle"] = {}
        data.setdefault("version", 4)
        data.setdefault("reaped", [])
        data.setdefault("reaped_waiters", [])
        return data

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=True, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _pid_start_time(pid: int) -> str:
        if pid <= 0:
            return ""
        try:
            text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        right = text.rfind(")")
        if right < 0:
            return ""
        fields = text[right + 2 :].split()
        # /proc/<pid>/stat field 22 is starttime; after removing pid+comm,
        # field 3 is at index 0, so field 22 is index 19.
        return fields[19] if len(fields) > 19 else ""

    @classmethod
    def current_owner_metadata(cls) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "owner_pid": pid,
            "owner_start_time": cls._pid_start_time(pid),
        }

    @classmethod
    def _lease_stale_reason(cls, lease: ResourceLease, *, now: float, lease_ttl_sec: float) -> str:
        meta = lease.metadata or {}
        owner_pid_raw = meta.get("owner_pid")
        if owner_pid_raw is not None and str(owner_pid_raw).strip():
            try:
                owner_pid = int(owner_pid_raw)
            except (TypeError, ValueError):
                owner_pid = 0
            if owner_pid > 0:
                if not cls._pid_alive(owner_pid):
                    return "owner_pid_dead"
                expected_start = str(meta.get("owner_start_time") or "")
                current_start = cls._pid_start_time(owner_pid)
                if expected_start and current_start and expected_start != current_start:
                    return "owner_pid_reused"
        if lease_ttl_sec > 0 and now - lease.heartbeat_at >= lease_ttl_sec:
            return "lease_ttl_expired"
        return ""

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            state = self._read_unlocked()
            try:
                yield state
            finally:
                self._write_unlocked(state)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def _reap_stale_unlocked(self, state: dict[str, Any], *, now: float) -> list[ResourceLease]:
        reaped: list[ResourceLease] = []
        leases = dict(state.get("leases") or {})
        for job_id, raw in leases.items():
            lease = ResourceLease.from_json(raw if isinstance(raw, dict) else {})
            if not lease.job_id:
                lease.job_id = str(job_id)
            reason = self._lease_stale_reason(lease, now=now, lease_ttl_sec=self.lease_ttl_sec)
            if reason:
                lease.metadata = {
                    **dict(lease.metadata or {}),
                    "reaped_reason": reason,
                    "reaped_at": now,
                }
                reaped.append(lease)
                state["leases"].pop(job_id, None)
        if reaped:
            state.setdefault("reaped", []).extend([lease.to_json() for lease in reaped])
        self._reap_stale_waiters_unlocked(state, now=now)
        return reaped

    def _reap_stale_waiters_unlocked(self, state: dict[str, Any], *, now: float) -> list[dict[str, Any]]:
        waiters = state.setdefault("waiters", {})
        reaped: list[dict[str, Any]] = []
        for job_id, raw in list(waiters.items()):
            if not isinstance(raw, dict):
                waiters.pop(job_id, None)
                continue
            updated = float(raw.get("updated_at") or raw.get("submitted_at") or now)
            if now - updated >= self.waiter_ttl_sec:
                reaped.append(dict(raw))
                waiters.pop(job_id, None)
        if reaped:
            state.setdefault("reaped_waiters", []).extend(reaped)
        return reaped

    @staticmethod
    def _lease_resource_class(lease: ResourceLease) -> str:
        return str((lease.metadata or {}).get("policy_resource_class") or (lease.metadata or {}).get("resource_class") or "")

    @staticmethod
    def _lease_slot_weight(lease: ResourceLease) -> float:
        try:
            value = float((lease.metadata or {}).get("slot_weight", 1.0))
        except (TypeError, ValueError):
            value = 1.0
        return max(0.0, value)

    @staticmethod
    def _lease_incompatible_classes(lease: ResourceLease) -> set[str]:
        raw = (lease.metadata or {}).get("incompatible_classes") or []
        if not isinstance(raw, list):
            return set()
        return {str(x) for x in raw if str(x).strip()}

    @staticmethod
    def _lease_mode(lease: ResourceLease) -> str:
        mode = str((lease.metadata or {}).get("lease_mode") or "exclusive").strip()
        return mode or "exclusive"

    @staticmethod
    def _shared_secondary_job_ids(lease: ResourceLease) -> list[str]:
        raw = (lease.metadata or {}).get("shared_secondary_job_ids") or []
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw if str(x).strip()]

    @staticmethod
    def _set_shared_primary_metadata(primary: ResourceLease, *, secondary_job_id: str, now: float) -> ResourceLease:
        meta = dict(primary.metadata or {})
        secondaries = [x for x in GPULeaseStore._shared_secondary_job_ids(primary) if x != str(secondary_job_id)]
        secondaries.append(str(secondary_job_id))
        meta["lease_mode"] = "shared_primary"
        meta["shared_secondary_job_ids"] = secondaries
        meta["shared_gpu_updated_at"] = now
        primary.metadata = meta
        primary.heartbeat_at = now
        return primary

    @staticmethod
    def _remove_shared_secondary_metadata(primary: ResourceLease, *, secondary_job_id: str, now: float) -> ResourceLease:
        meta = dict(primary.metadata or {})
        secondaries = [x for x in GPULeaseStore._shared_secondary_job_ids(primary) if x != str(secondary_job_id)]
        if secondaries:
            meta["lease_mode"] = "shared_primary"
            meta["shared_secondary_job_ids"] = secondaries
        else:
            meta["lease_mode"] = "exclusive"
            meta.pop("shared_secondary_job_ids", None)
        meta["shared_gpu_updated_at"] = now
        primary.metadata = meta
        primary.heartbeat_at = now
        return primary

    @staticmethod
    def _waiter_priority(raw: dict[str, Any]) -> float:
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        pri = meta.get("admission_priority") if isinstance(meta.get("admission_priority"), dict) else {}
        try:
            base = float(raw.get("priority_score", pri.get("admission_priority_score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        try:
            submitted_at = float(raw.get("submitted_at") or 0.0)
        except (TypeError, ValueError):
            submitted_at = 0.0
        wait_age = max(0.0, time.time() - submitted_at) if submitted_at > 0 else 0.0
        starvation_bonus = min(0.35, wait_age / 900.0 * 0.35)
        try:
            request_count = int(raw.get("request_count") or 1)
        except (TypeError, ValueError):
            request_count = 1
        gang_bonus = 0.05 if request_count > 1 else 0.0
        return base + starvation_bonus + gang_bonus

    @staticmethod
    def _waiter_sort_key(raw: dict[str, Any]) -> tuple[float, float, str]:
        return (-GPULeaseStore._waiter_priority(raw), float(raw.get("submitted_at") or 0.0), str(raw.get("job_id") or ""))

    @staticmethod
    def _metadata_command_digest(metadata: dict[str, Any] | None) -> str:
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("command_digest") or "").strip()

    def _duplicate_digest_summary_unlocked(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        metadata: dict[str, Any] | None,
        now: float,
    ) -> dict[str, Any]:
        digest = self._metadata_command_digest(metadata)
        if not digest:
            return {"command_digest": "", "duplicate_active_job_ids": [], "duplicate_waiter_job_ids": []}
        active: list[str] = []
        waiters: list[str] = []
        oldest_wait = 0.0
        for owner_id, raw in (state.get("leases") or {}).items():
            if str(owner_id) == str(job_id) or not isinstance(raw, dict):
                continue
            lease = ResourceLease.from_json(raw)
            if self._metadata_command_digest(lease.metadata) != digest:
                continue
            active.append(str(owner_id))
        for waiter_id, raw in (state.get("waiters") or {}).items():
            if str(waiter_id) == str(job_id) or not isinstance(raw, dict):
                continue
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            if self._metadata_command_digest(meta) != digest:
                continue
            waiters.append(str(waiter_id))
            try:
                submitted_at = float(raw.get("submitted_at") or raw.get("updated_at") or now)
            except (TypeError, ValueError):
                submitted_at = now
            oldest_wait = max(oldest_wait, max(0.0, now - submitted_at))
        return {
            "command_digest": digest,
            "duplicate_active_job_ids": active,
            "duplicate_waiter_job_ids": waiters,
            "duplicate_count": len(active) + len(waiters),
            "same_digest_oldest_wait_sec": oldest_wait,
        }

    @staticmethod
    def _waiter_gpu_ids(raw: dict[str, Any]) -> set[str]:
        return {str(x) for x in (raw.get("gpu_ids") or raw.get("candidate_gpu_ids") or []) if str(x).strip()}

    def _matching_waiters(
        self,
        waiters: dict[str, Any],
        *,
        gpu_ids: list[str],
        resource_class: str,
    ) -> list[dict[str, Any]]:
        wanted = {str(x) for x in (gpu_ids or []) if str(x).strip()}
        cls = str(resource_class or "")
        out: list[dict[str, Any]] = []
        for raw in waiters.values():
            if not isinstance(raw, dict):
                continue
            if cls and str(raw.get("resource_class") or "") != cls:
                continue
            waiter_ids = self._waiter_gpu_ids(raw)
            if wanted and waiter_ids and not (wanted & waiter_ids):
                continue
            out.append(dict(raw))
        out.sort(key=self._waiter_sort_key)
        return out

    def _waiter_details(
        self,
        waiters: dict[str, Any],
        *,
        job_id: str,
        gpu_ids: list[str],
        resource_class: str,
        virtual: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = {str(k): dict(v) for k, v in waiters.items() if isinstance(v, dict)}
        if virtual is not None:
            merged[str(job_id)] = dict(virtual)
        matching = self._matching_waiters(merged, gpu_ids=gpu_ids, resource_class=resource_class)
        ids = [str(x.get("job_id") or "") for x in matching if str(x.get("job_id") or "")]
        position = ids.index(str(job_id)) + 1 if str(job_id) in ids else len(ids) + 1
        return {
            "queue_position": position,
            "queue_len": len(ids),
            "waiter_job_ids": ids,
            "top_waiter_job_id": ids[0] if ids else "",
        }

    def _waiter_record(
        self,
        *,
        job_id: str,
        worker_id: str,
        gpu_ids: list[str],
        request_count: int,
        metadata: dict[str, Any],
        resource_class: str,
        slot_weight: float,
        capacity_slots: float,
        max_per_gpu: int,
        incompatible_classes: set[str],
        now: float,
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = dict(metadata or {})
        pri = meta.get("admission_priority") if isinstance(meta.get("admission_priority"), dict) else {}
        submitted = float((existing or {}).get("submitted_at") or now)
        return {
            "job_id": str(job_id),
            "worker_id": str(worker_id or ""),
            "gpu_ids": [str(x) for x in (gpu_ids or []) if str(x).strip()],
            "request_count": max(1, int(request_count or 1)),
            "resource_class": str(resource_class or ""),
            "slot_weight": max(0.0, float(slot_weight or 0.0)),
            "capacity_slots": max(0.0, float(capacity_slots or 0.0)) or 1.0,
            "max_per_gpu": max(1, int(max_per_gpu or 1)),
            "incompatible_classes": sorted(incompatible_classes),
            "priority_score": self._waiter_priority({"metadata": {"admission_priority": pri}}),
            "metadata": meta,
            "submitted_at": submitted,
            "updated_at": now,
        }

    def _upsert_waiter_unlocked(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        worker_id: str,
        gpu_ids: list[str],
        request_count: int,
        metadata: dict[str, Any],
        resource_class: str,
        slot_weight: float,
        capacity_slots: float,
        max_per_gpu: int,
        incompatible_classes: set[str],
        now: float,
    ) -> dict[str, Any]:
        waiters = state.setdefault("waiters", {})
        existing = waiters.get(job_id) if isinstance(waiters.get(job_id), dict) else None
        record = self._waiter_record(
            job_id=job_id,
            worker_id=worker_id,
            gpu_ids=gpu_ids,
            request_count=request_count,
            metadata=metadata,
            resource_class=resource_class,
            slot_weight=slot_weight,
            capacity_slots=capacity_slots,
            max_per_gpu=max_per_gpu,
            incompatible_classes=incompatible_classes,
            now=now,
            existing=existing,
        )
        waiters[str(job_id)] = record
        return record

    def annotate_waiter_share_override(self, *, job_id: str, share_override: dict[str, Any]) -> dict[str, Any]:
        clean_job = str(job_id or "")
        if not clean_job:
            return {"updated": False, "reason": "missing_job_id"}
        if not isinstance(share_override, dict) or not share_override.get("candidate"):
            return {"updated": False, "reason": "not_share_override_candidate"}
        now = time.time()
        with self._locked_state() as state:
            self._reap_stale_waiters_unlocked(state, now=now)
            waiters = state.setdefault("waiters", {})
            raw = waiters.get(clean_job)
            if not isinstance(raw, dict):
                return {"updated": False, "reason": "waiter_not_found"}
            meta = dict(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {})
            clean_override = dict(share_override)
            clean_override["requested_at"] = now
            meta["share_override"] = clean_override
            meta["share_override_requested_at"] = now
            raw["metadata"] = meta
            raw["share_override"] = clean_override
            raw["share_override_requested_at"] = now
            raw["updated_at"] = now
            waiters[clean_job] = raw
        return {"updated": True, "reason": "share_override_recorded", "requested_at": now}

    def _admission_queue_decision_unlocked(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        worker_id: str,
        gpu_ids: list[str],
        request_count: int,
        metadata: dict[str, Any],
        resource_class: str,
        slot_weight: float,
        capacity_slots: float,
        max_per_gpu: int,
        incompatible_classes: set[str],
        now: float,
    ) -> tuple[bool, dict[str, Any]]:
        if not self.admission_queue_enabled:
            return True, {"queue_position": 1, "queue_len": 0}
        self._reap_stale_waiters_unlocked(state, now=now)
        waiters = state.setdefault("waiters", {})
        existing = waiters.get(job_id) if isinstance(waiters.get(job_id), dict) else None
        virtual = self._waiter_record(
            job_id=job_id,
            worker_id=worker_id,
            gpu_ids=gpu_ids,
            request_count=request_count,
            metadata=metadata,
            resource_class=resource_class,
            slot_weight=slot_weight,
            capacity_slots=capacity_slots,
            max_per_gpu=max_per_gpu,
            incompatible_classes=incompatible_classes,
            now=now,
            existing=existing,
        )
        details = self._waiter_details(
            waiters,
            job_id=job_id,
            gpu_ids=gpu_ids,
            resource_class=resource_class,
            virtual=virtual,
        )
        if int(details.get("queue_position") or 1) <= 1:
            waiters.pop(str(job_id), None)
            return True, details
        waiters[str(job_id)] = virtual
        return False, {**details, "reason": "admission_waiter_ahead"}

    def _gpu_fit_details(
        self,
        leases: dict[str, Any],
        *,
        gpu_id: str,
        resource_class: str,
        slot_weight: float,
        capacity_slots: float,
        max_per_gpu: int,
        incompatible_classes: set[str],
        released_idle: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        owners: list[str] = []
        class_owners: list[str] = []
        incompatible_owners: list[str] = []
        released_idle_owners: list[str] = []
        released_idle_resident_mem_gb = 0.0
        used_weight = 0.0
        for owner_job_id, raw in leases.items():
            lease = ResourceLease.from_json(raw if isinstance(raw, dict) else {})
            if gpu_id not in lease.gpu_ids:
                continue
            owners.append(str(owner_job_id))
            active_class = self._lease_resource_class(lease)
            used_weight += self._lease_slot_weight(lease)
            if active_class == resource_class:
                class_owners.append(str(owner_job_id))
            active_incompatible = self._lease_incompatible_classes(lease)
            if active_class in incompatible_classes or resource_class in active_incompatible:
                incompatible_owners.append(str(owner_job_id))
        for owner_job_id, raw in (released_idle or {}).items():
            if not isinstance(raw, dict):
                continue
            lease = ResourceLease.from_json(raw)
            if gpu_id not in lease.gpu_ids:
                continue
            released_idle_owners.append(str(owner_job_id))
            meta = lease.metadata or {}
            try:
                released_idle_resident_mem_gb += max(0.0, float(meta.get("released_idle_gpu_resident_mem_gb") or 0.0))
            except (TypeError, ValueError):
                pass
            if self.idle_release_admission_mode == "strict_exclusive":
                owners.append(str(owner_job_id))
        reasons: list[str] = []
        if used_weight + max(0.0, slot_weight) > max(0.0, capacity_slots) + 1e-9:
            reasons.append("slot_capacity_exceeded")
        if len(class_owners) >= max(1, int(max_per_gpu or 1)):
            reasons.append("class_limit_exceeded")
        if incompatible_owners:
            reasons.append("incompatible_active_class")
        if released_idle_owners and self.idle_release_admission_mode == "strict_exclusive":
            reasons.append("released_idle_residue_reserved")
        details = {
            "owners": owners,
            "class_owners": class_owners,
            "incompatible_owners": incompatible_owners,
            "released_idle_owners": released_idle_owners,
            "released_idle_gpu_resident_mem_gb": released_idle_resident_mem_gb,
            "idle_release_admission_mode": self.idle_release_admission_mode,
            "used_slot_weight": used_weight,
            "requested_slot_weight": max(0.0, slot_weight),
            "capacity_slots": max(0.0, capacity_slots),
            "resource_class": resource_class,
            "reasons": reasons,
        }
        return not reasons, details

    def _reacquire_released_idle_unlocked(
        self,
        state: dict[str, Any],
        *,
        job_id: str,
        now: float,
    ) -> ResourceLease | None:
        released_idle = state.setdefault("released_idle", {})
        raw = released_idle.pop(str(job_id), None)
        if not isinstance(raw, dict):
            return None
        lease = ResourceLease.from_json(raw)
        meta = dict(lease.metadata or {})
        meta["lease_mode"] = "exclusive"
        meta["released_idle_reacquired_at"] = now
        meta.pop("released_idle", None)
        lease.metadata = meta
        lease.heartbeat_at = now
        state.setdefault("leases", {})[str(job_id)] = lease.to_json()
        return lease

    def try_acquire(
        self,
        *,
        job_id: str,
        worker_id: str,
        gpu_ids: list[str],
        max_heavy_per_gpu: int,
        metadata: dict[str, Any] | None = None,
        resource_class: str = "heavy_gpu_candidate",
        slot_weight: float = 1.0,
        capacity_slots: float = 1.0,
        max_per_gpu: int | None = None,
        incompatible_classes: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not job_id or not ids:
            return True, {"reason": "no_gpu_request", "reaped": []}
        limit = max(1, int(max_per_gpu or max_heavy_per_gpu or 1))
        capacity = max(0.0, float(capacity_slots or 0.0)) or 1.0
        weight = max(0.0, float(slot_weight or 0.0))
        incompatible = {str(x) for x in (incompatible_classes or []) if str(x).strip()}
        now = time.time()
        with self._locked_state() as state:
            reaped = self._reap_stale_unlocked(state, now=now)
            leases = state.setdefault("leases", {})
            existing = leases.get(job_id)
            if isinstance(existing, dict):
                lease = ResourceLease.from_json(existing)
                lease.heartbeat_at = now
                leases[job_id] = lease.to_json()
                state.setdefault("waiters", {}).pop(str(job_id), None)
                return True, {
                    "reason": "already_acquired",
                    "lease": lease.to_json(),
                    "queue_position": 1,
                    "queue_len": 0,
                    "reaped": [x.to_json() for x in reaped],
                }
            reacquired = self._reacquire_released_idle_unlocked(state, job_id=str(job_id), now=now)
            if reacquired is not None:
                state.setdefault("waiters", {}).pop(str(job_id), None)
                return True, {
                    "reason": "reacquired_released_idle",
                    "lease": reacquired.to_json(),
                    "queue_position": 1,
                    "queue_len": 0,
                    "reaped": [x.to_json() for x in reaped],
                }

            blockers: dict[str, list[str]] = {}
            blocker_details: dict[str, Any] = {}
            for gpu_id in ids:
                ok, details = self._gpu_fit_details(
                    leases,
                    gpu_id=gpu_id,
                    resource_class=str(resource_class or ""),
                    slot_weight=weight,
                    capacity_slots=capacity,
                    max_per_gpu=limit,
                    incompatible_classes=incompatible,
                    released_idle=state.setdefault("released_idle", {}),
                )
                if not ok:
                    blockers[gpu_id] = [str(x) for x in details.get("owners") or []]
                    blocker_details[gpu_id] = details
            meta = dict(metadata or {})
            duplicate_summary = self._duplicate_digest_summary_unlocked(state, job_id=str(job_id), metadata=meta, now=now)
            if blockers:
                waiter = self._upsert_waiter_unlocked(
                    state,
                    job_id=str(job_id),
                    worker_id=str(worker_id or ""),
                    gpu_ids=ids,
                    request_count=len(ids),
                    metadata=meta,
                    resource_class=str(resource_class or ""),
                    slot_weight=weight,
                    capacity_slots=capacity,
                    max_per_gpu=limit,
                    incompatible_classes=incompatible,
                    now=now,
                )
                q = self._waiter_details(
                    state.setdefault("waiters", {}),
                    job_id=str(job_id),
                    gpu_ids=ids,
                    resource_class=str(resource_class or ""),
                )
                return False, {
                    "reason": "gpu_slot_unavailable",
                    "blocked_gpu_ids": sorted(blockers),
                    "blockers": blockers,
                    "blocker_details": blocker_details,
                    "duplicate_digest_summary": duplicate_summary,
                    "waiter": waiter,
                    **q,
                    "reaped": [x.to_json() for x in reaped],
                }

            queue_ok, q = self._admission_queue_decision_unlocked(
                state,
                job_id=str(job_id),
                worker_id=str(worker_id or ""),
                gpu_ids=ids,
                request_count=len(ids),
                metadata=meta,
                resource_class=str(resource_class or ""),
                slot_weight=weight,
                capacity_slots=capacity,
                max_per_gpu=limit,
                incompatible_classes=incompatible,
                now=now,
            )
            if not queue_ok:
                return False, {**q, "duplicate_digest_summary": duplicate_summary, "reaped": [x.to_json() for x in reaped]}

            meta.setdefault("resource_class", str(resource_class or ""))
            meta.setdefault("policy_resource_class", str(resource_class or ""))
            meta["slot_weight"] = weight
            meta["capacity_slots"] = capacity
            meta["max_per_gpu"] = limit
            meta["incompatible_classes"] = sorted(incompatible)
            lease = ResourceLease(
                job_id=str(job_id),
                worker_id=str(worker_id or ""),
                gpu_ids=ids,
                acquired_at=now,
                heartbeat_at=now,
                metadata=meta,
            )
            leases[job_id] = lease.to_json()
            state.setdefault("waiters", {}).pop(str(job_id), None)
            return True, {
                "reason": "acquired",
                "lease": lease.to_json(),
                "duplicate_digest_summary": duplicate_summary,
                **q,
                "reaped": [x.to_json() for x in reaped],
            }

    def try_acquire_any(
        self,
        *,
        job_id: str,
        worker_id: str,
        candidate_gpu_ids: list[str],
        request_count: int,
        max_heavy_per_gpu: int,
        metadata: dict[str, Any] | None = None,
        resource_class: str = "heavy_gpu_candidate",
        slot_weight: float = 1.0,
        capacity_slots: float = 1.0,
        max_per_gpu: int | None = None,
        incompatible_classes: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        candidates = [str(x) for x in (candidate_gpu_ids or []) if str(x).strip()]
        seen: set[str] = set()
        candidates = [x for x in candidates if not (x in seen or seen.add(x))]
        if not job_id or not candidates:
            return True, {"reason": "no_gpu_request", "reaped": []}
        limit = max(1, int(max_per_gpu or max_heavy_per_gpu or 1))
        capacity = max(0.0, float(capacity_slots or 0.0)) or 1.0
        weight = max(0.0, float(slot_weight or 0.0))
        incompatible = {str(x) for x in (incompatible_classes or []) if str(x).strip()}
        request = max(1, min(int(request_count or 1), len(candidates)))
        now = time.time()
        with self._locked_state() as state:
            reaped = self._reap_stale_unlocked(state, now=now)
            leases = state.setdefault("leases", {})
            existing = leases.get(job_id)
            if isinstance(existing, dict):
                lease = ResourceLease.from_json(existing)
                lease.heartbeat_at = now
                leases[job_id] = lease.to_json()
                state.setdefault("waiters", {}).pop(str(job_id), None)
                return True, {
                    "reason": "already_acquired",
                    "lease": lease.to_json(),
                    "queue_position": 1,
                    "queue_len": 0,
                    "reaped": [x.to_json() for x in reaped],
                }
            reacquired = self._reacquire_released_idle_unlocked(state, job_id=str(job_id), now=now)
            if reacquired is not None:
                state.setdefault("waiters", {}).pop(str(job_id), None)
                return True, {
                    "reason": "reacquired_released_idle",
                    "lease": reacquired.to_json(),
                    "assigned_gpu_ids": reacquired.gpu_ids,
                    "queue_position": 1,
                    "queue_len": 0,
                    "reaped": [x.to_json() for x in reaped],
                }

            owners_by_gpu: dict[str, list[str]] = {gpu_id: [] for gpu_id in candidates}
            fit_details: dict[str, Any] = {}
            available: list[str] = []
            for gpu_id in candidates:
                ok, details = self._gpu_fit_details(
                    leases,
                    gpu_id=gpu_id,
                    resource_class=str(resource_class or ""),
                    slot_weight=weight,
                    capacity_slots=capacity,
                    max_per_gpu=limit,
                    incompatible_classes=incompatible,
                    released_idle=state.setdefault("released_idle", {}),
                )
                owners_by_gpu[gpu_id] = [str(x) for x in details.get("owners") or []]
                fit_details[gpu_id] = details
                if ok:
                    available.append(gpu_id)
            meta = dict(metadata or {})
            duplicate_summary = self._duplicate_digest_summary_unlocked(state, job_id=str(job_id), metadata=meta, now=now)
            if len(available) < request:
                waiter = self._upsert_waiter_unlocked(
                    state,
                    job_id=str(job_id),
                    worker_id=str(worker_id or ""),
                    gpu_ids=candidates,
                    request_count=request,
                    metadata=meta,
                    resource_class=str(resource_class or ""),
                    slot_weight=weight,
                    capacity_slots=capacity,
                    max_per_gpu=limit,
                    incompatible_classes=incompatible,
                    now=now,
                )
                q = self._waiter_details(
                    state.setdefault("waiters", {}),
                    job_id=str(job_id),
                    gpu_ids=candidates,
                    resource_class=str(resource_class or ""),
                )
                return False, {
                    "reason": "gpu_slot_unavailable",
                    "candidate_gpu_ids": candidates,
                    "available_gpu_ids": available,
                    "requested_count": request,
                    "blockers": owners_by_gpu,
                    "blocker_details": fit_details,
                    "duplicate_digest_summary": duplicate_summary,
                    "waiter": waiter,
                    **q,
                    "reaped": [x.to_json() for x in reaped],
                }

            queue_ok, q = self._admission_queue_decision_unlocked(
                state,
                job_id=str(job_id),
                worker_id=str(worker_id or ""),
                gpu_ids=candidates,
                request_count=request,
                metadata=meta,
                resource_class=str(resource_class or ""),
                slot_weight=weight,
                capacity_slots=capacity,
                max_per_gpu=limit,
                incompatible_classes=incompatible,
                now=now,
            )
            if not queue_ok:
                return False, {
                    **q,
                    "duplicate_digest_summary": duplicate_summary,
                    "candidate_gpu_ids": candidates,
                    "available_gpu_ids": available,
                    "requested_count": request,
                    "reaped": [x.to_json() for x in reaped],
                }

            def _assignment_rank(gpu_id: str) -> tuple[float, int, int, int]:
                detail = fit_details.get(gpu_id) if isinstance(fit_details.get(gpu_id), dict) else {}
                try:
                    used_weight = float(detail.get("used_slot_weight") or 0.0)
                except (TypeError, ValueError):
                    used_weight = 0.0
                owners = detail.get("owners") if isinstance(detail.get("owners"), list) else []
                class_owners = detail.get("class_owners") if isinstance(detail.get("class_owners"), list) else []
                return (
                    used_weight,
                    len(class_owners),
                    len(owners),
                    candidates.index(gpu_id) if gpu_id in candidates else 10_000,
                )

            assigned = sorted(available, key=_assignment_rank)[:request]
            meta.setdefault("resource_class", str(resource_class or ""))
            meta.setdefault("policy_resource_class", str(resource_class or ""))
            meta["slot_weight"] = weight
            meta["capacity_slots"] = capacity
            meta["max_per_gpu"] = limit
            meta["incompatible_classes"] = sorted(incompatible)
            lease = ResourceLease(
                job_id=str(job_id),
                worker_id=str(worker_id or ""),
                gpu_ids=assigned,
                acquired_at=now,
                heartbeat_at=now,
                metadata=meta,
            )
            leases[job_id] = lease.to_json()
            state.setdefault("waiters", {}).pop(str(job_id), None)
            return True, {
                "reason": "acquired",
                "lease": lease.to_json(),
                "assigned_gpu_ids": assigned,
                "duplicate_digest_summary": duplicate_summary,
                **q,
                "reaped": [x.to_json() for x in reaped],
            }

    def availability(
        self,
        *,
        candidate_gpu_ids: list[str],
        request_count: int,
        max_heavy_per_gpu: int,
        resource_class: str,
        slot_weight: float,
        capacity_slots: float,
        max_per_gpu: int | None = None,
        incompatible_classes: list[str] | None = None,
    ) -> dict[str, Any]:
        candidates = [str(x) for x in (candidate_gpu_ids or []) if str(x).strip()]
        if not candidates:
            return {"available": True, "available_gpu_ids": [], "requested_count": 0}
        limit = max(1, int(max_per_gpu or max_heavy_per_gpu or 1))
        capacity = max(0.0, float(capacity_slots or 0.0)) or 1.0
        weight = max(0.0, float(slot_weight or 0.0))
        incompatible = {str(x) for x in (incompatible_classes or []) if str(x).strip()}
        request = max(1, min(int(request_count or 1), len(candidates)))
        now = time.time()
        with self._locked_state() as state:
            self._reap_stale_unlocked(state, now=now)
            leases = state.setdefault("leases", {})
            available: list[str] = []
            fit_details: dict[str, Any] = {}
            for gpu_id in candidates:
                ok, details = self._gpu_fit_details(
                    leases,
                    gpu_id=gpu_id,
                    resource_class=str(resource_class or ""),
                    slot_weight=weight,
                    capacity_slots=capacity,
                    max_per_gpu=limit,
                    incompatible_classes=incompatible,
                    released_idle=state.setdefault("released_idle", {}),
                )
                fit_details[gpu_id] = details
                if ok:
                    available.append(gpu_id)
            return {
                "available": len(available) >= request,
                "available_gpu_ids": available,
                "requested_count": request,
                "fit_details": fit_details,
                "waiters": [dict(x) for x in self._matching_waiters(state.setdefault("waiters", {}), gpu_ids=candidates, resource_class=str(resource_class or ""))],
            }

    def snapshot_active(self) -> dict[str, Any]:
        now = time.time()
        with self._locked_state() as state:
            reaped = self._reap_stale_unlocked(state, now=now)
            leases = state.setdefault("leases", {})
            waiters = state.setdefault("waiters", {})
            released_idle = state.setdefault("released_idle", {})
            return {
                "now": now,
                "leases": {
                    str(job_id): ResourceLease.from_json(raw if isinstance(raw, dict) else {}).to_json()
                    for job_id, raw in leases.items()
                },
                "waiters": {str(job_id): dict(raw) for job_id, raw in waiters.items() if isinstance(raw, dict)},
                "released_idle": {str(job_id): dict(raw) for job_id, raw in released_idle.items() if isinstance(raw, dict)},
                "reaped": [x.to_json() for x in reaped],
            }

    def grant_shared_secondary(
        self,
        *,
        primary_job_id: str,
        secondary_job_id: str,
        metadata: dict[str, Any] | None = None,
        max_secondary_per_primary: int = 1,
    ) -> dict[str, Any]:
        primary_id = str(primary_job_id or "").strip()
        secondary_id = str(secondary_job_id or "").strip()
        if not primary_id or not secondary_id:
            return {"acquired": False, "reason": "missing_job_id"}
        if primary_id == secondary_id:
            return {"acquired": False, "reason": "primary_secondary_same_job"}
        now = time.time()
        with self._locked_state() as state:
            self._reap_stale_unlocked(state, now=now)
            leases = state.setdefault("leases", {})
            primary_raw = leases.get(primary_id)
            if not isinstance(primary_raw, dict):
                return {"acquired": False, "reason": "primary_lease_not_found"}
            primary = ResourceLease.from_json(primary_raw)
            existing_raw = leases.get(secondary_id)
            if isinstance(existing_raw, dict):
                existing = ResourceLease.from_json(existing_raw)
                if self._lease_mode(existing) == "shared_secondary":
                    existing.heartbeat_at = now
                    leases[secondary_id] = existing.to_json()
                    state.setdefault("waiters", {}).pop(secondary_id, None)
                    return {
                        "acquired": True,
                        "reason": "already_shared_secondary",
                        "lease": existing.to_json(),
                        "primary_lease": primary.to_json(),
                        "assigned_gpu_ids": existing.gpu_ids,
                    }
                return {"acquired": False, "reason": "secondary_already_has_non_shared_lease"}

            active_secondaries = [
                job_id
                for job_id in self._shared_secondary_job_ids(primary)
                if isinstance(leases.get(job_id), dict)
            ]
            if len(active_secondaries) >= max(1, int(max_secondary_per_primary or 1)):
                return {"acquired": False, "reason": "shared_secondary_limit_reached"}

            waiter = state.setdefault("waiters", {}).get(secondary_id)
            if not isinstance(waiter, dict):
                return {"acquired": False, "reason": "secondary_waiter_not_found"}
            primary_gpu_ids = [str(x) for x in primary.gpu_ids if str(x).strip()]
            waiter_gpu_ids = [str(x) for x in (waiter.get("gpu_ids") or waiter.get("candidate_gpu_ids") or []) if str(x).strip()]
            waiter_set = set(waiter_gpu_ids)
            assigned = [gpu_id for gpu_id in primary_gpu_ids if not waiter_set or gpu_id in waiter_set]
            if not assigned:
                return {"acquired": False, "reason": "no_shared_gpu_intersection"}

            meta = dict(waiter.get("metadata") if isinstance(waiter.get("metadata"), dict) else {})
            meta.update(dict(metadata or {}))
            meta.setdefault("resource_class", str(waiter.get("resource_class") or ""))
            meta.setdefault("policy_resource_class", str(waiter.get("resource_class") or ""))
            effective_slot = meta.get("shared_effective_slot_weight")
            meta["slot_weight"] = float(effective_slot if effective_slot is not None else (waiter.get("slot_weight") or meta.get("slot_weight") or 0.0))
            meta["lease_mode"] = "shared_secondary"
            meta["shared_primary_job_id"] = primary_id
            meta["shared_granted_at"] = now
            meta["shared_grant_reason"] = str(meta.get("shared_grant_reason") or "task_gpu_share_review")
            lease = ResourceLease(
                job_id=secondary_id,
                worker_id=str(waiter.get("worker_id") or ""),
                gpu_ids=assigned,
                acquired_at=now,
                heartbeat_at=now,
                metadata=meta,
            )
            primary = self._set_shared_primary_metadata(primary, secondary_job_id=secondary_id, now=now)
            leases[primary_id] = primary.to_json()
            leases[secondary_id] = lease.to_json()
            state.setdefault("waiters", {}).pop(secondary_id, None)
            return {
                "acquired": True,
                "reason": "shared_gpu_lease_granted",
                "lease": lease.to_json(),
                "primary_lease": primary.to_json(),
                "assigned_gpu_ids": assigned,
            }

    def release_idle(
        self,
        *,
        job_id: str,
        resident_mem_gb: float = 0.0,
        elapsed_sec: float = 0.0,
        reason: str = "idle_gpu_lease",
    ) -> dict[str, Any]:
        if not job_id:
            return {"released": False, "reason": "missing_job_id"}
        with self._locked_state() as state:
            state.setdefault("waiters", {}).pop(str(job_id), None)
            now = time.time()
            leases = state.setdefault("leases", {})
            released_idle = state.setdefault("released_idle", {})
            raw = leases.pop(str(job_id), None)
            if not isinstance(raw, dict):
                if isinstance(released_idle.get(str(job_id)), dict):
                    return {"released": True, "reason": "already_released_idle", "lease": dict(released_idle[str(job_id)])}
                return {"released": False, "reason": "not_found"}
            lease = ResourceLease.from_json(raw)
            mode = self._lease_mode(lease)
            if mode in {"shared_primary", "shared_secondary"}:
                leases[str(job_id)] = lease.to_json()
                return {"released": False, "reason": f"{mode}_not_idle_releasable", "lease": lease.to_json()}
            meta = dict(lease.metadata or {})
            meta.update({
                "released_idle": True,
                "lease_mode": "released_idle",
                "released_idle_reason": str(reason or "idle_gpu_lease"),
                "released_idle_released_at": now,
                "released_idle_elapsed_sec": float(elapsed_sec or 0.0),
                "released_idle_gpu_resident_mem_gb": max(0.0, float(resident_mem_gb or 0.0)),
                "idle_release_admission_mode": self.idle_release_admission_mode,
            })
            lease.metadata = meta
            lease.heartbeat_at = now
            released_idle[str(job_id)] = lease.to_json()
            return {"released": True, "reason": "idle_lease_released", "lease": lease.to_json()}

    def release(self, *, job_id: str) -> dict[str, Any]:
        if not job_id:
            return {"released": False, "reason": "missing_job_id"}
        with self._locked_state() as state:
            state.setdefault("waiters", {}).pop(str(job_id), None)
            now = time.time()
            leases = state.setdefault("leases", {})
            released_idle = state.setdefault("released_idle", {})
            raw = leases.pop(job_id, None)
            release_source = "active"
            if not isinstance(raw, dict):
                raw = released_idle.pop(job_id, None)
                release_source = "released_idle"
            if not isinstance(raw, dict):
                return {"released": False, "reason": "not_found"}
            lease = ResourceLease.from_json(raw)
            result: dict[str, Any] = {"released": True, "lease": lease.to_json(), "release_source": release_source}
            meta = lease.metadata or {}
            if self._lease_mode(lease) == "shared_secondary":
                primary_id = str(meta.get("shared_primary_job_id") or "")
                primary_raw = leases.get(primary_id)
                if isinstance(primary_raw, dict):
                    primary = self._remove_shared_secondary_metadata(
                        ResourceLease.from_json(primary_raw),
                        secondary_job_id=lease.job_id,
                        now=now,
                    )
                    leases[primary_id] = primary.to_json()
                    result["primary_update"] = primary.to_json()
            else:
                revoked: list[dict[str, Any]] = []
                for secondary_id in self._shared_secondary_job_ids(lease):
                    secondary_raw = leases.pop(secondary_id, None)
                    if isinstance(secondary_raw, dict):
                        revoked.append(ResourceLease.from_json(secondary_raw).to_json())
                if revoked:
                    result["revoked_shared_secondaries"] = revoked
            return result
