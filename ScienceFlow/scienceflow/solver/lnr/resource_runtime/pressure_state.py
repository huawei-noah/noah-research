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


class GPUPressureStateStore:
    """Small shared per-GPU pressure state file for LHR workers."""

    def __init__(
        self,
        path: Path,
        *,
        default_cooldown_sec: float = 120.0,
        duplicate_digest_cooldown_sec: float = 600.0,
        duplicate_digest_threshold: int = 2,
        yellow_hold_sec: float = 120.0,
        red_to_yellow_sec: float = 120.0,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.default_cooldown_sec = max(0.0, float(default_cooldown_sec or 0.0))
        self.duplicate_digest_cooldown_sec = max(0.0, float(duplicate_digest_cooldown_sec or 0.0))
        self.duplicate_digest_threshold = max(1, int(duplicate_digest_threshold or 2))
        self.yellow_hold_sec = max(0.0, float(yellow_hold_sec or 0.0))
        self.red_to_yellow_sec = max(0.0, float(red_to_yellow_sec or 0.0))

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"version": 3, "generation": 0, "gpus": {}}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        if not isinstance(data.get("gpus"), dict):
            data["gpus"] = {}
        data.setdefault("version", 3)
        data.setdefault("generation", 0)
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

    def _record_digest_failure_unlocked(
        self,
        raw: dict[str, Any],
        *,
        command_digest: str,
        now: float,
        worker_id: str,
        job_id: str,
        resource_class: str,
        reason: str,
    ) -> dict[str, Any]:
        digest = str(command_digest or "").strip()
        if not digest:
            return {}
        failures = raw.get("digest_failures") if isinstance(raw.get("digest_failures"), dict) else {}
        entry = dict(failures.get(digest) if isinstance(failures.get(digest), dict) else {})
        count = int(entry.get("count") or 0) + 1
        cooldown_until = float(entry.get("cooldown_until") or 0.0)
        if count >= self.duplicate_digest_threshold:
            cooldown_until = max(cooldown_until, now + self.duplicate_digest_cooldown_sec)
        entry.update({
            "command_digest": digest,
            "count": count,
            "last_failure_at": now,
            "last_worker_id": str(worker_id or ""),
            "last_job_id": str(job_id or ""),
            "last_resource_class": str(resource_class or ""),
            "last_reason": str(reason or ""),
            "cooldown_until": cooldown_until,
        })
        failures[digest] = entry
        raw["digest_failures"] = failures
        raw["last_command_digest"] = digest
        raw["last_digest_failure_count"] = count
        raw["last_digest_cooldown_until"] = cooldown_until
        return entry

    def record_queue_timeout(
        self,
        *,
        gpu_ids: list[str],
        worker_id: str,
        job_id: str,
        resource_class: str,
        elapsed_sec: float,
        reason: str,
        cooldown_sec: float | None = None,
        command_digest: str = "",
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not ids:
            return {"recorded": False, "reason": "no_gpu_ids", "gpu_ids": []}
        now = time.time()
        cooldown = max(0.0, float(self.default_cooldown_sec if cooldown_sec is None else cooldown_sec))
        with self._locked_state() as state:
            state["generation"] = int(state.get("generation") or 0) + 1
            gpus = state.setdefault("gpus", {})
            for gpu_id in ids:
                raw = gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {}
                count = int(raw.get("queue_timeout_count") or 0) + 1
                red_until = now + cooldown
                yellow_until = max(float(raw.get("yellow_until") or 0.0), red_until + self.red_to_yellow_sec)
                if command_digest:
                    self._record_digest_failure_unlocked(
                        raw,
                        command_digest=command_digest,
                        now=now,
                        worker_id=worker_id,
                        job_id=job_id,
                        resource_class=resource_class,
                        reason=reason,
                    )
                gpus[gpu_id] = {
                    **raw,
                    "gpu_id": gpu_id,
                    "mode": "RED",
                    "queue_timeout_count": count,
                    "last_timeout_at": now,
                    "red_until": red_until,
                    "yellow_until": yellow_until,
                    "last_reason": str(reason or "queue_timeout"),
                    "last_worker_id": str(worker_id or ""),
                    "last_job_id": str(job_id or ""),
                    "last_resource_class": str(resource_class or ""),
                    "last_elapsed_sec": float(elapsed_sec or 0.0),
                    "updated_at": now,
                }
            out_gpus: dict[str, Any] = {}
            for gpu_id in ids:
                entry = dict(gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {})
                red_until = float(entry.get("red_until") or 0.0)
                entry["cooldown_remaining_sec"] = max(0.0, red_until - now)
                out_gpus[gpu_id] = entry
            return {
                "recorded": True,
                "generation": int(state.get("generation") or 0),
                "mode": "RED",
                "gpu_ids": ids,
                "gpus": out_gpus,
                "cooldown_sec": cooldown,
                "red_until": now + cooldown,
                "yellow_until": now + cooldown + self.red_to_yellow_sec,
            }

    def record_digest_failure(
        self,
        *,
        gpu_ids: list[str],
        worker_id: str,
        job_id: str,
        resource_class: str,
        command_digest: str,
        reason: str,
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        digest = str(command_digest or "").strip()
        if not ids or not digest:
            return {"recorded": False, "reason": "missing_gpu_or_digest", "gpu_ids": ids}
        now = time.time()
        with self._locked_state() as state:
            state["generation"] = int(state.get("generation") or 0) + 1
            gpus = state.setdefault("gpus", {})
            out_gpus: dict[str, Any] = {}
            for gpu_id in ids:
                raw = gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {"gpu_id": gpu_id}
                entry = self._record_digest_failure_unlocked(
                    raw,
                    command_digest=digest,
                    now=now,
                    worker_id=worker_id,
                    job_id=job_id,
                    resource_class=resource_class,
                    reason=reason,
                )
                raw["gpu_id"] = gpu_id
                raw["updated_at"] = now
                gpus[gpu_id] = raw
                out_gpus[gpu_id] = {**raw, "current_digest_failure": entry}
            return {
                "recorded": True,
                "generation": int(state.get("generation") or 0),
                "gpu_ids": ids,
                "gpus": out_gpus,
            }



    def record_yellow(
        self,
        *,
        gpu_ids: list[str],
        worker_id: str,
        job_id: str,
        resource_class: str,
        reason: str,
        hold_sec: float | None = None,
        queue_len: int | None = None,
        queue_position: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not ids:
            return {"recorded": False, "reason": "no_gpu_ids", "gpu_ids": []}
        now = time.time()
        hold = max(0.0, float(self.yellow_hold_sec if hold_sec is None else hold_sec))
        yellow_until = now + hold
        with self._locked_state() as state:
            state["generation"] = int(state.get("generation") or 0) + 1
            gpus = state.setdefault("gpus", {})
            out_gpus: dict[str, Any] = {}
            for gpu_id in ids:
                raw = gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {}
                red_until = float(raw.get("red_until") or 0.0)
                count = int(raw.get("yellow_count") or 0) + 1
                effective_yellow_until = max(float(raw.get("yellow_until") or 0.0), yellow_until)
                mode = "RED" if red_until > now else "YELLOW"
                updated = {
                    **raw,
                    "gpu_id": gpu_id,
                    "mode": mode,
                    "yellow_count": count,
                    "yellow_until": effective_yellow_until,
                    "last_yellow_at": now,
                    "last_yellow_reason": str(reason or "yellow_pressure"),
                    "last_reason": str(raw.get("last_reason") or reason or "yellow_pressure"),
                    "last_worker_id": str(worker_id or ""),
                    "last_job_id": str(job_id or ""),
                    "last_resource_class": str(resource_class or ""),
                    "updated_at": now,
                }
                if queue_len is not None:
                    updated["queue_len"] = max(0, int(queue_len or 0))
                if queue_position is not None:
                    updated["queue_position"] = max(0, int(queue_position or 0))
                if metadata:
                    updated["yellow_metadata"] = dict(metadata)
                gpus[gpu_id] = updated
                entry = dict(updated)
                entry["yellow_remaining_sec"] = max(0.0, effective_yellow_until - now)
                entry["cooldown_remaining_sec"] = max(0.0, red_until - now)
                out_gpus[gpu_id] = entry
            return {
                "recorded": True,
                "generation": int(state.get("generation") or 0),
                "mode": "YELLOW",
                "gpu_ids": ids,
                "gpus": out_gpus,
                "yellow_until": yellow_until,
                "hold_sec": hold,
                "reason": str(reason or "yellow_pressure"),
            }

    def bump_generation(self, *, reason: str = "resource_changed") -> dict[str, Any]:
        now = time.time()
        with self._locked_state() as state:
            state["generation"] = int(state.get("generation") or 0) + 1
            state["last_generation_reason"] = str(reason or "resource_changed")
            state["updated_at"] = now
            return {
                "generation": int(state.get("generation") or 0),
                "reason": str(reason or "resource_changed"),
                "updated_at": now,
            }

    def clear_global_pressure(
        self,
        *,
        gpu_ids: list[str],
        reason: str,
        observed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Clear per-GPU RED/YELLOW holds without touching digest cooldowns."""

        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not ids:
            return {"cleared": False, "reason": "no_gpu_ids", "gpu_ids": []}
        now = time.time()
        changed: dict[str, Any] = {}
        with self._locked_state() as state:
            gpus = state.setdefault("gpus", {})
            for gpu_id in ids:
                raw = gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {}
                if not raw:
                    continue
                red_until = float(raw.get("red_until") or 0.0)
                yellow_until = float(raw.get("yellow_until") or 0.0)
                was_active = red_until > now or yellow_until > now or str(raw.get("mode") or "").upper() in {"RED", "YELLOW"}
                if not was_active:
                    continue
                previous_mode = "RED" if red_until > now else "YELLOW" if yellow_until > now else str(raw.get("mode") or "")
                updated = dict(raw)
                updated["red_until"] = 0.0
                updated["yellow_until"] = 0.0
                updated["mode"] = "GREEN"
                updated["cooldown_remaining_sec"] = 0.0
                updated["yellow_remaining_sec"] = 0.0
                updated["last_reconciled_at"] = now
                updated["last_reconciled_reason"] = str(reason or "observed_free_no_active_lease")
                updated["previous_pressure_mode"] = previous_mode
                updated["updated_at"] = now
                if observed:
                    updated["last_reconcile_observed"] = dict(observed)
                gpus[gpu_id] = updated
                changed[gpu_id] = updated
            if not changed:
                return {
                    "cleared": False,
                    "reason": "no_active_global_pressure",
                    "gpu_ids": ids,
                    "generation": int(state.get("generation") or 0),
                }
            state["generation"] = int(state.get("generation") or 0) + 1
            state["last_generation_reason"] = str(reason or "observed_free_no_active_lease")
            state["updated_at"] = now
            return {
                "cleared": True,
                "reason": str(reason or "observed_free_no_active_lease"),
                "gpu_ids": sorted(changed),
                "generation": int(state.get("generation") or 0),
                "gpus": changed,
                "observed": dict(observed or {}),
            }

    def record_runtime_pressure(
        self,
        *,
        gpu_ids: list[str],
        worker_id: str,
        job_id: str,
        resource_class: str,
        elapsed_sec: float,
        reason: str,
        cooldown_sec: float | None = None,
        command_digest: str = "",
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not ids:
            return {"recorded": False, "reason": "no_gpu_ids", "gpu_ids": []}
        now = time.time()
        cooldown = max(0.0, float(self.default_cooldown_sec if cooldown_sec is None else cooldown_sec))
        with self._locked_state() as state:
            state["generation"] = int(state.get("generation") or 0) + 1
            gpus = state.setdefault("gpus", {})
            for gpu_id in ids:
                raw = gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {}
                count = int(raw.get("runtime_pressure_count") or 0) + 1
                red_until = now + cooldown
                yellow_until = max(float(raw.get("yellow_until") or 0.0), red_until + self.red_to_yellow_sec)
                if command_digest:
                    self._record_digest_failure_unlocked(
                        raw,
                        command_digest=command_digest,
                        now=now,
                        worker_id=worker_id,
                        job_id=job_id,
                        resource_class=resource_class,
                        reason=reason,
                    )
                gpus[gpu_id] = {
                    **raw,
                    "gpu_id": gpu_id,
                    "mode": "RED",
                    "runtime_pressure_count": count,
                    "last_runtime_pressure_at": now,
                    "red_until": red_until,
                    "yellow_until": yellow_until,
                    "last_reason": str(reason or "runtime_pressure"),
                    "last_worker_id": str(worker_id or ""),
                    "last_job_id": str(job_id or ""),
                    "last_resource_class": str(resource_class or ""),
                    "last_elapsed_sec": float(elapsed_sec or 0.0),
                    "updated_at": now,
                }
            out_gpus: dict[str, Any] = {}
            for gpu_id in ids:
                entry = dict(gpus.get(gpu_id) if isinstance(gpus.get(gpu_id), dict) else {})
                red_until = float(entry.get("red_until") or 0.0)
                entry["cooldown_remaining_sec"] = max(0.0, red_until - now)
                out_gpus[gpu_id] = entry
            return {
                "recorded": True,
                "generation": int(state.get("generation") or 0),
                "mode": "RED",
                "gpu_ids": ids,
                "gpus": out_gpus,
                "cooldown_sec": cooldown,
                "red_until": now + cooldown,
                "yellow_until": now + cooldown + self.red_to_yellow_sec,
                "reason": str(reason or "runtime_pressure"),
            }

    def duplicate_gate_decision(self, *, gpu_ids: list[str], command_digest: str) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        digest = str(command_digest or "").strip()
        if not ids or not digest:
            return {"blocked": False, "gpu_ids": ids}
        now = time.time()
        snapshot = self.snapshot(gpu_ids=ids)
        blocked_gpus: dict[str, Any] = {}
        for gpu_id, raw in (snapshot.get("gpus") or {}).items():
            if not isinstance(raw, dict):
                continue
            failures = raw.get("digest_failures") if isinstance(raw.get("digest_failures"), dict) else {}
            entry = failures.get(digest) if isinstance(failures.get(digest), dict) else {}
            cooldown_until = float(entry.get("cooldown_until") or 0.0)
            if cooldown_until > now:
                blocked_gpus[str(gpu_id)] = {
                    "gpu_id": str(gpu_id),
                    "command_digest": digest,
                    "duplicate_digest_count": int(entry.get("count") or 0),
                    "cooldown_remaining_sec": max(0.0, cooldown_until - now),
                    "cooldown_until": cooldown_until,
                    "last_reason": str(entry.get("last_reason") or ""),
                }
        if not blocked_gpus:
            return {"blocked": False, "gpu_ids": ids, "pressure": snapshot}
        cooldown = max(float(x.get("cooldown_remaining_sec") or 0.0) for x in blocked_gpus.values())
        return {
            "blocked": True,
            "reason": "duplicate_digest_cooldown",
            "status": "DENIED_DUPLICATE",
            "gpu_ids": ids,
            "red_gpu_ids": sorted(blocked_gpus),
            "duplicate_digest_count": max(int(x.get("duplicate_digest_count") or 0) for x in blocked_gpus.values()),
            "cooldown_remaining_sec": cooldown,
            "pressure": snapshot,
            "duplicate_gpus": blocked_gpus,
        }

    def snapshot(self, *, gpu_ids: list[str] | None = None) -> dict[str, Any]:
        wanted = {str(x) for x in (gpu_ids or []) if str(x).strip()}
        now = time.time()
        with self._locked_state() as state:
            gpus = state.setdefault("gpus", {})
            out: dict[str, Any] = {}
            for gpu_id, raw in list(gpus.items()):
                entry = dict(raw if isinstance(raw, dict) else {})
                if wanted and str(gpu_id) not in wanted:
                    continue
                red_until = float(entry.get("red_until") or 0.0)
                yellow_until = float(entry.get("yellow_until") or 0.0)
                if red_until > now:
                    mode = "RED"
                elif yellow_until > now:
                    mode = "YELLOW"
                else:
                    mode = "GREEN"
                entry["mode"] = mode
                entry["cooldown_remaining_sec"] = max(0.0, red_until - now)
                entry["yellow_remaining_sec"] = max(0.0, yellow_until - now)
                failures = entry.get("digest_failures") if isinstance(entry.get("digest_failures"), dict) else {}
                for digest, failure in list(failures.items()):
                    if not isinstance(failure, dict):
                        failures.pop(digest, None)
                        continue
                    cooldown_until = float(failure.get("cooldown_until") or 0.0)
                    failure["cooldown_remaining_sec"] = max(0.0, cooldown_until - now)
                entry["digest_failures"] = failures
                out[str(gpu_id)] = entry
            return {
                "generation": int(state.get("generation") or 0),
                "gpus": out,
                "now": now,
            }
