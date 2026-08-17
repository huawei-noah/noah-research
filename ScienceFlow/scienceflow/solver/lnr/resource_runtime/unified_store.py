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
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class UnifiedResourceStore:
    """Run-level generic resource state and append-only audit events."""

    def __init__(self, resource_dir: Path) -> None:
        self.resource_dir = Path(resource_dir)
        self.state_path = self.resource_dir / "resource_state.json"
        self.events_path = self.resource_dir / "resource_events.jsonl"
        self.state_lock_path = self.state_path.with_name(f"{self.state_path.name}.lock")
        self.events_lock_path = self.events_path.with_name(f"{self.events_path.name}.lock")

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": 0.0,
            "leases": {},
            "waiters": {},
            "released_idle": {},
            "pressure": {},
            "runtime_history": {},
            "active_proposals": {},
            "cooldowns": {},
            "ttl": {},
            "control_plane": {},
            "legacy": {},
        }

    def _read_state_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        state = self._empty_state()
        state.update(data)
        return state

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.state_path.name}.", suffix=".tmp", dir=str(self.resource_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=True, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, self.state_path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        with self.state_lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            state = self._read_state_unlocked()
            try:
                yield state
            finally:
                state["updated_at"] = time.time()
                self._write_state_unlocked(state)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def update_state(self, **patch: Any) -> dict[str, Any]:
        with self._locked_state() as state:
            for key, value in patch.items():
                if value is not None:
                    state[key] = value
            return dict(state)

    def sync_legacy_gpu_state(
        self,
        *,
        leases: dict[str, Any],
        pressure: dict[str, Any],
        runtime_history: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        legacy = {
            "gpu_leases": dict(leases or {}),
            "gpu_pressure": dict(pressure or {}),
            "gpu_runtime_history": dict(runtime_history or {}),
            "sync_reason": str(reason or "sync"),
            "synced_at": time.time(),
        }
        generic_leases = dict((leases or {}).get("leases") or {})
        generic_waiters = dict((leases or {}).get("waiters") or {})
        generic_released_idle = dict((leases or {}).get("released_idle") or {})
        generic_pressure = {"gpu": dict((pressure or {}).get("gpus") or {}), "generation": int((pressure or {}).get("generation") or 0)}
        return self.update_state(
            leases=generic_leases,
            waiters=generic_waiters,
            released_idle=generic_released_idle,
            pressure=generic_pressure,
            runtime_history=dict(runtime_history or {}),
            legacy=legacy,
        )

    @staticmethod
    def _proposal_matches(
        proposal: dict[str, Any],
        *,
        command_id: str = "",
        proposal_type: str = "",
        reason_code: str = "",
    ) -> bool:
        preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
        candidate_command = str(
            proposal.get("command_id")
            or proposal.get("job_id")
            or preview.get("job_id")
            or ""
        ).strip()
        if command_id and candidate_command != command_id:
            return False
        if proposal_type and str(proposal.get("proposal_type") or "").strip() != proposal_type:
            return False
        if reason_code and str(proposal.get("reason_code") or "").strip() != reason_code:
            return False
        return True

    def update_active_proposal(self, proposal_id: str, proposal: dict[str, Any] | None) -> dict[str, Any]:
        pid = str(proposal_id or "").strip()
        if not pid:
            return self.update_state()
        with self._locked_state() as state:
            active = state.setdefault("active_proposals", {})
            if proposal is None:
                active.pop(pid, None)
            else:
                new_proposal = dict(proposal)
                command_id = str(new_proposal.get("command_id") or new_proposal.get("job_id") or "").strip()
                proposal_type = str(new_proposal.get("proposal_type") or "").strip()
                reason_code = str(new_proposal.get("reason_code") or "").strip()
                if command_id and proposal_type and reason_code:
                    stale = [
                        old_pid
                        for old_pid, old in active.items()
                        if old_pid != pid
                        and isinstance(old, dict)
                        and self._proposal_matches(
                            old,
                            command_id=command_id,
                            proposal_type=proposal_type,
                            reason_code=reason_code,
                        )
                    ]
                    for old_pid in stale:
                        active.pop(old_pid, None)
                active[pid] = new_proposal
            return dict(state)

    def clear_active_proposals_for_command(
        self,
        command_id: str,
        *,
        proposal_type: str = "",
        reason_code: str = "",
    ) -> dict[str, Any]:
        command = str(command_id or "").strip()
        if not command:
            return self.update_state()
        with self._locked_state() as state:
            active = state.setdefault("active_proposals", {})
            stale = [
                pid
                for pid, proposal in active.items()
                if isinstance(proposal, dict)
                and self._proposal_matches(
                    proposal,
                    command_id=command,
                    proposal_type=str(proposal_type or "").strip(),
                    reason_code=str(reason_code or "").strip(),
                )
            ]
            for pid in stale:
                active.pop(pid, None)
            return dict(state)

    def append_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        trace_id: str = "",
        proposal_id: str = "",
        decision_id: str = "",
        task_id: str = "",
        worker_id: str = "",
        command_id: str = "",
        lease_id: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        event = {
            "version": 1,
            "event_id": f"re_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(now))}_{uuid.uuid4().hex[:10]}",
            "event_type": str(event_type or "resource_event"),
            "created_at": now,
            "trace_id": str(trace_id or ""),
            "proposal_id": str(proposal_id or ""),
            "decision_id": str(decision_id or ""),
            "task_id": str(task_id or ""),
            "worker_id": str(worker_id or ""),
            "command_id": str(command_id or ""),
            "lease_id": str(lease_id or ""),
            "payload": dict(payload or {}),
        }
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        with self.events_lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                with self.events_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
                    fh.write("\n")
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return event


def find_resource_event_files(root: str | Path, *, max_files: int = 256) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists():
        return []
    if base.is_file():
        return [base] if base.name == "resource_events.jsonl" else []
    direct = base / "resource_events.jsonl"
    if direct.exists():
        return [direct]
    files: list[Path] = []
    for path in base.rglob("resource_events.jsonl"):
        if len(files) >= max(1, int(max_files or 256)):
            break
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        files.append(path)
    return sorted(files)


def read_resource_events(path: str | Path, *, max_events: int | None = None) -> tuple[list[dict[str, Any]], int]:
    event_path = Path(path)
    if not event_path.exists() or not event_path.is_file():
        return [], 0
    events: list[dict[str, Any]] = []
    malformed = 0
    try:
        with event_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if max_events is not None and len(events) >= max(0, int(max_events)):
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(data, dict):
                    events.append(data)
                else:
                    malformed += 1
    except OSError:
        return [], malformed + 1
    return events, malformed


def summarize_resource_event_file(path: str | Path, *, max_events: int | None = None) -> dict[str, Any]:
    event_path = Path(path)
    events, malformed = read_resource_events(event_path, max_events=max_events)
    event_counts: Counter[str] = Counter()
    decision_actions: Counter[str] = Counter()
    guard_actions: Counter[str] = Counter()
    admission_actions: Counter[str] = Counter()
    workers: set[str] = set()
    commands: set[str] = set()
    first_created: float | None = None
    last_created: float | None = None
    boundary_violations = 0
    gpu_orphan_cleanups = 0
    gpu_cleanup_skipped = 0
    latest_boundary_event: dict[str, Any] | None = None
    idle_release_candidates = 0

    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        event_counts[event_type] += 1
        if event.get("worker_id"):
            workers.add(str(event.get("worker_id")))
        if event.get("command_id"):
            commands.add(str(event.get("command_id")))
        try:
            created = float(event.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created = 0.0
        if created > 0:
            first_created = created if first_created is None else min(first_created, created)
            last_created = created if last_created is None else max(last_created, created)

        if event_type.startswith("admission_"):
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            action = str(result.get("admission_action") or result.get("status") or event_type.removeprefix("admission_")).upper()
            admission_actions[action] += 1
        if event_type == "decision":
            decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            outcome = decision.get("value_review_outcome") if isinstance(decision.get("value_review_outcome"), dict) else {}
            action = str(outcome.get("outcome") or decision.get("execution_outcome") or decision.get("action") or payload.get("action") or "").upper()
            if action:
                decision_actions[action] += 1
        if event_type == "execution":
            action = str(payload.get("action") or "").upper()
            if action:
                guard_actions[action] += 1
        if event_type == "resource_guard_action":
            action = str(payload.get("action") or "").upper()
            if action:
                guard_actions[action] += 1
        action_lower = str(payload.get("action") or "").strip().lower()
        reason_lower = str(payload.get("reason") or "").strip().lower()
        if event_type == "resource_idle_lease_release_candidate" or str(payload.get("action") or "").upper() == "RELEASE_IDLE_LEASE":
            idle_release_candidates += 1
        if action_lower == "gpu_orphan_cleanup":
            gpu_orphan_cleanups += 1
        if action_lower == "gpu_cleanup_skipped":
            gpu_cleanup_skipped += 1
        boundary_hit = (
            action_lower == "stop_boundary_violation"
            or event_type == "admission_boundary_violation"
            or ("boundary_violation" in reason_lower and action_lower != "gpu_orphan_cleanup")
        )
        if boundary_hit:
            boundary_violations += 1
            latest_boundary_event = _compact_boundary_event(event_type=event_type, event=event, payload=payload)

    state = _read_resource_state_for_events(event_path)
    leases = state.get("leases") if isinstance(state.get("leases"), dict) else {}
    waiters = state.get("waiters") if isinstance(state.get("waiters"), dict) else {}
    label = _resource_event_file_label(event_path)
    return {
        "label": label,
        "event_path": str(event_path),
        "event_count": len(events),
        "malformed_events": malformed,
        "workers": sorted(workers),
        "worker_count": len(workers),
        "command_count": len(commands),
        "first_created_at": first_created or 0.0,
        "last_created_at": last_created or 0.0,
        "duration_sec": max(0.0, (last_created or 0.0) - (first_created or 0.0)),
        "event_counts": dict(event_counts),
        "admission_actions": dict(admission_actions),
        "decision_actions": dict(decision_actions),
        "guard_actions": dict(guard_actions),
        "boundary_violations": boundary_violations,
        "gpu_orphan_cleanups": gpu_orphan_cleanups,
        "gpu_cleanup_skipped": gpu_cleanup_skipped,
        "latest_boundary_event": latest_boundary_event or {},
        "idle_release_candidates": idle_release_candidates,
        "active_lease_count": len(leases),
        "waiter_count": len(waiters),
    }


def summarize_resource_run(root: str | Path, *, max_files: int = 256, max_events_per_file: int | None = None) -> dict[str, Any]:
    files = find_resource_event_files(root, max_files=max_files)
    tasks = [summarize_resource_event_file(path, max_events=max_events_per_file) for path in files]
    totals = {
        "event_count": sum(int(task.get("event_count") or 0) for task in tasks),
        "malformed_events": sum(int(task.get("malformed_events") or 0) for task in tasks),
        "boundary_violations": sum(int(task.get("boundary_violations") or 0) for task in tasks),
        "gpu_orphan_cleanups": sum(int(task.get("gpu_orphan_cleanups") or 0) for task in tasks),
        "gpu_cleanup_skipped": sum(int(task.get("gpu_cleanup_skipped") or 0) for task in tasks),
        "idle_release_candidates": sum(int(task.get("idle_release_candidates") or 0) for task in tasks),
        "active_lease_count": sum(int(task.get("active_lease_count") or 0) for task in tasks),
        "waiter_count": sum(int(task.get("waiter_count") or 0) for task in tasks),
    }
    for key in ("admission_actions", "decision_actions", "guard_actions", "event_counts"):
        counter: Counter[str] = Counter()
        for task in tasks:
            counter.update(task.get(key) if isinstance(task.get(key), dict) else {})
        totals[key] = dict(counter)
    return {"root": str(Path(root)), "resource_event_files": len(files), "tasks": tasks, "totals": totals}


def format_resource_run_summary(summary: dict[str, Any]) -> str:
    tasks = summary.get("tasks") if isinstance(summary.get("tasks"), list) else []
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    lines = [f"Resource summary: {summary.get('root')}", f"resource_event_files: {summary.get('resource_event_files', 0)}"]
    lines.append("")
    lines.append(_format_resource_table(tasks))
    lines.append("")
    lines.append("Totals:")
    lines.append(
        "  events={events} malformed={malformed} active_leases={leases} waiters={waiters} "
        "boundary={boundary} orphan={orphan} cleanup_skipped={skipped} idle_release_candidates={idle}".format(
            events=int(totals.get("event_count") or 0),
            malformed=int(totals.get("malformed_events") or 0),
            leases=int(totals.get("active_lease_count") or 0),
            waiters=int(totals.get("waiter_count") or 0),
            boundary=int(totals.get("boundary_violations") or 0),
            orphan=int(totals.get("gpu_orphan_cleanups") or 0),
            skipped=int(totals.get("gpu_cleanup_skipped") or 0),
            idle=int(totals.get("idle_release_candidates") or 0),
        )
    )
    lines.append(f"  admission={_compact_counter(totals.get('admission_actions'))}")
    lines.append(f"  decisions={_compact_counter(totals.get('decision_actions'))}")
    lines.append(f"  guard_actions={_compact_counter(totals.get('guard_actions'))}")
    return "\n".join(lines).rstrip() + "\n"


def _compact_boundary_event(*, event_type: str, event: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    placement = payload.get("placement") if isinstance(payload.get("placement"), dict) else {}
    violations = placement.get("violations") if isinstance(placement.get("violations"), list) else []
    allowed = placement.get("allowed_gpu_ids") or placement.get("allowed_physical_gpus") or []
    used = sorted({str(row.get("gpu_id") or "") for row in violations if isinstance(row, dict) and str(row.get("gpu_id") or "").strip()})
    return {
        "event_type": str(event_type or ""),
        "created_at": event.get("created_at") or event.get("timestamp") or 0.0,
        "worker_id": str(event.get("worker_id") or ""),
        "command_id": str(event.get("command_id") or payload.get("job_id") or ""),
        "action": str(payload.get("action") or ""),
        "reason": str(payload.get("reason") or ""),
        "actual_gpu_ids": used,
        "allowed_gpu_ids": [str(x) for x in allowed if str(x).strip()],
    }


def _read_resource_state_for_events(event_path: Path) -> dict[str, Any]:
    state_path = event_path.with_name("resource_state.json")
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resource_event_file_label(event_path: Path) -> str:
    try:
        if event_path.parent.name == "resource" and event_path.parent.parent.name == "logs":
            return event_path.parent.parent.parent.name
    except IndexError:
        pass
    return event_path.parent.name or event_path.name


def _compact_counter(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    items = sorted(((str(k), int(v or 0)) for k, v in value.items()), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}:{v}" for k, v in items[:8])


def _format_resource_table(tasks: list[Any]) -> str:
    headers = ["task", "events", "run", "pending", "replan", "kill", "no_action", "boundary", "orphan", "skip", "idle_rel", "leases", "waiters"]
    rows: list[list[str]] = []
    for raw in tasks:
        task = raw if isinstance(raw, dict) else {}
        admission = task.get("admission_actions") if isinstance(task.get("admission_actions"), dict) else {}
        decisions = task.get("decision_actions") if isinstance(task.get("decision_actions"), dict) else {}
        rows.append([
            str(task.get("label") or "resource"),
            str(int(task.get("event_count") or 0)),
            str(int(admission.get("RUN_NOW") or admission.get("GRANTED") or 0)),
            str(int(admission.get("PENDING") or 0)),
            str(int(admission.get("REPLAN") or admission.get("DENIED_REPLAN") or 0)),
            str(
                int(decisions.get("KILL") or 0)
                + int(decisions.get("KILL_AND_REPLAN") or 0)
                + int(decisions.get("APPROVE_KILL") or 0)
                + int(decisions.get("APPROVE_KILL_STALLED") or 0)
            ),
            str(
                int(decisions.get("NO_ACTION") or 0)
                + int(decisions.get("DENY_KILL") or 0)
                + int(decisions.get("OBSERVE_MORE") or 0)
                + int(decisions.get("CONTINUE") or 0)
                + int(decisions.get("MARK_STALLED_NO_KILL") or 0)
            ),
            str(int(task.get("boundary_violations") or 0)),
            str(int(task.get("gpu_orphan_cleanups") or 0)),
            str(int(task.get("gpu_cleanup_skipped") or 0)),
            str(int(task.get("idle_release_candidates") or 0)),
            str(int(task.get("active_lease_count") or 0)),
            str(int(task.get("waiter_count") or 0)),
        ])
    if not rows:
        rows = [["(none)"] + ["0"] * (len(headers) - 1)]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], min(42, len(cell)))
    def fmt(row: list[str]) -> str:
        clipped = [cell if len(cell) <= widths[idx] else cell[: max(0, widths[idx] - 1)] + "…" for idx, cell in enumerate(row)]
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(clipped))
    return "\n".join([fmt(headers), fmt(["-" * w for w in widths]), *(fmt(row) for row in rows)])
