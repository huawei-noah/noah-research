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

from dataclasses import dataclass
from typing import Any

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
)


PHASES = {"off", "observe", "tt_share", "feature_share", "light_train"}
POLICY_PROFILES = {"conservative", "balanced"}
MEMORY_PROFILES = {"conservative", "balanced"}
CPU_POLICIES = {"conservative", "balanced"}
TRIAL_ADMISSION_POLICIES = {"llm_grant", "deterministic_grant_when_hard_gates_pass"}


@dataclass(frozen=True)
class GPUShareConfig:
    enabled: bool = False
    phase: str = "observe"
    policy_profile: str = "conservative"
    memory_profile: str = "conservative"
    cpu_policy: str = "conservative"
    min_windows: int = 3
    warmup_sec: float = 300.0
    trial_warmup_sec: float = 60.0
    util_p90_pct: float = 35.0
    min_free_mem_gb: float = 8.0
    max_mem_used_ratio: float = 0.60
    safety_margin_gb: float = 4.0
    secondary_estimated_peak_gb_default: float = 8.0
    secondary_estimated_peak_gb_tt: float = 2.0
    secondary_estimated_peak_gb_feature: float = 6.0
    secondary_estimated_peak_gb_light_train: float = 8.0
    secondary_max_slot_weight: float = 0.5
    monitor_interval_sec: float = 30.0
    revoke_min_windows: int = 2
    secondary_warmup_sec: float = 180.0
    trial_admission_policy: str = "llm_grant"

    @property
    def observe_active(self) -> bool:
        return self.enabled and self.phase in {"observe", "tt_share", "feature_share", "light_train"}

    @property
    def grant_active(self) -> bool:
        return self.enabled and self.phase in {"tt_share", "feature_share", "light_train"}


def build_gpu_share_config(
    *,
    enabled: bool,
    phase: str,
    policy_profile: str,
    memory_profile: str,
    cpu_policy: str,
    trial_admission_policy: str = "llm_grant",
) -> GPUShareConfig:
    clean_phase = _choice(phase, PHASES, "observe")
    clean_policy = _choice(policy_profile, POLICY_PROFILES, "conservative")
    clean_memory = _choice(memory_profile, MEMORY_PROFILES, "conservative")
    clean_cpu = _choice(cpu_policy, CPU_POLICIES, "conservative")
    clean_trial_policy = _choice(trial_admission_policy, TRIAL_ADMISSION_POLICIES, "llm_grant")
    if clean_phase == "off":
        enabled = False

    min_windows = 2 if clean_policy == "balanced" else 3
    util_p90 = 45.0 if clean_policy == "balanced" else 35.0
    safety_margin = 2.0 if clean_memory == "balanced" else 4.0
    min_free = 6.0 if clean_memory == "balanced" else 8.0
    max_ratio = 0.70 if clean_memory == "balanced" else 0.60

    return GPUShareConfig(
        enabled=bool(enabled),
        phase=clean_phase,
        policy_profile=clean_policy,
        memory_profile=clean_memory,
        cpu_policy=clean_cpu,
        min_windows=min_windows,
        util_p90_pct=util_p90,
        min_free_mem_gb=min_free,
        max_mem_used_ratio=max_ratio,
        safety_margin_gb=safety_margin,
        trial_admission_policy=clean_trial_policy,
    )


def isolated_active_work_signal(process_tree_cpu: dict[str, Any], *, cpu_isolated: bool) -> bool:
    if not cpu_isolated or not isinstance(process_tree_cpu, dict) or not process_tree_cpu.get("available", True):
        return False
    child_cpu = _float(process_tree_cpu.get("child_cpu_pct"), 0.0)
    total_cpu = _float(process_tree_cpu.get("total_cpu_pct"), child_cpu)
    busy_children = int(_float(process_tree_cpu.get("busy_child_count"), 0.0))
    return bool(child_cpu >= 50.0 or total_cpu >= 50.0 or busy_children >= 1)


def classify_cpu_pressure(process_tree_cpu: dict[str, Any], *, cpu_policy: str = "conservative") -> str:
    if not isinstance(process_tree_cpu, dict) or not process_tree_cpu.get("available", True):
        return "unknown"
    child_cpu = _float(process_tree_cpu.get("child_cpu_pct"), 0.0)
    total_cpu = _float(process_tree_cpu.get("total_cpu_pct"), child_cpu)
    busy_children = int(_float(process_tree_cpu.get("busy_child_count"), 0.0))
    high_child = 200.0 if cpu_policy == "conservative" else 300.0
    medium_child = 100.0 if cpu_policy == "conservative" else 150.0
    if child_cpu >= high_child or total_cpu >= high_child or busy_children >= 2:
        return "high"
    if child_cpu >= medium_child or total_cpu >= medium_child or busy_children >= 1:
        return "medium"
    return "low"


def gpu_memory_summary(sample: dict[str, Any], gpu_ids: list[str], *, previous_peak_gb: float = 0.0) -> dict[str, Any]:
    wanted = {str(x) for x in (gpu_ids or []) if str(x).strip()}
    rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
    used: list[float] = []
    total: list[float] = []
    util: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
        if wanted and gpu_id not in wanted:
            continue
        used_mb = _float_or_none(row.get("memory_used_mb"))
        total_mb = _float_or_none(row.get("memory_total_mb"))
        util_pct = _float_or_none(row.get("utilization_gpu_pct"))
        if used_mb is not None:
            used.append(max(0.0, used_mb / 1024.0))
        if total_mb is not None:
            total.append(max(0.0, total_mb / 1024.0))
        if util_pct is not None:
            util.append(max(0.0, util_pct))
    current = max(used or [0.0])
    total_gb = min(total or [0.0])
    free_gb = max(0.0, total_gb - current) if total_gb > 0 else 0.0
    peak = max(float(previous_peak_gb or 0.0), current)
    ratio = (current / total_gb) if total_gb > 0 else 0.0
    return {
        "gpu_mem_current_gb": current,
        "gpu_mem_peak_gb": peak,
        "gpu_mem_total_gb": total_gb,
        "gpu_free_mem_gb": free_gb,
        "gpu_mem_used_ratio": ratio,
        "gpu_util_p50_pct": _percentile(util, 0.50),
        "gpu_util_p90_pct": _percentile(util, 0.90),
        "sample_available": bool(rows),
    }


def secondary_estimated_peak_gb(resource_class: str, cfg: GPUShareConfig) -> float:
    cls = str(resource_class or "")
    if cls == RESOURCE_GPU_TT_LIGHT:
        return cfg.secondary_estimated_peak_gb_tt
    if cls == RESOURCE_GPU_FEATURE_EXTRACT:
        return cfg.secondary_estimated_peak_gb_feature
    if cls == RESOURCE_GPU_LIGHT_TRAIN:
        return cfg.secondary_estimated_peak_gb_light_train
    if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
        return max(float(cfg.secondary_estimated_peak_gb_default or 0.0), 16.0)
    return cfg.secondary_estimated_peak_gb_default


def share_warmup_sec(cfg: GPUShareConfig, *, trial_share: bool) -> float:
    base = max(0.0, float(cfg.warmup_sec or 0.0))
    if not trial_share:
        return base
    trial = max(0.0, float(cfg.trial_warmup_sec or 0.0))
    return min(base, trial) if base > 0 else trial


def evaluate_admission_share_trial(
    *,
    cfg: GPUShareConfig,
    admission_result: dict[str, Any],
    snapshot: dict[str, Any],
    gpu_sample: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate immediate shared-trial eligibility from the waiter admission path.

    This is intentionally narrower than the primary-owned share review: it can
    start only a revocable secondary trial when admission already observed an
    active same-GPU blocker and recorded a share override. Route value and kill
    decisions remain in the arbiter/primary monitor paths.
    """
    if not cfg.grant_active:
        return {"enabled": False, "reason": "gpu_share_grant_disabled"}
    override = admission_result.get("share_override") if isinstance(admission_result.get("share_override"), dict) else {}
    if not override.get("candidate"):
        return {"enabled": False, "reason": "no_share_override_candidate"}
    primary_ids = [str(x) for x in (override.get("primary_job_ids") or []) if str(x).strip()]
    if not primary_ids:
        return {"enabled": False, "reason": "missing_primary_job_id"}
    primary_id = primary_ids[0]
    waiter_id = str(override.get("waiter_job_id") or admission_result.get("job_id") or "").strip()
    if not waiter_id:
        return {"enabled": False, "reason": "missing_waiter_job_id"}

    leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
    waiters = snapshot.get("waiters") if isinstance(snapshot.get("waiters"), dict) else {}
    primary = leases.get(primary_id) if isinstance(leases.get(primary_id), dict) else {}
    waiter = waiters.get(waiter_id) if isinstance(waiters.get(waiter_id), dict) else {}
    if not primary:
        return {"enabled": False, "reason": "primary_lease_not_found", "primary_job_id": primary_id, "waiter_job_id": waiter_id}
    if not waiter:
        return {"enabled": False, "reason": "waiter_record_not_found", "primary_job_id": primary_id, "waiter_job_id": waiter_id}

    primary_gpu_ids = [str(x) for x in (primary.get("gpu_ids") or admission_result.get("gpu_ids") or []) if str(x).strip()]
    if not primary_gpu_ids:
        return {"enabled": False, "reason": "primary_gpu_missing", "primary_job_id": primary_id, "waiter_job_id": waiter_id}
    primary_meta = primary.get("metadata") if isinstance(primary.get("metadata"), dict) else {}
    waiter_meta = waiter.get("metadata") if isinstance(waiter.get("metadata"), dict) else {}
    waiter_override = waiter.get("share_override") if isinstance(waiter.get("share_override"), dict) else waiter_meta.get("share_override") if isinstance(waiter_meta.get("share_override"), dict) else {}
    waiter = dict(waiter)
    waiter["share_override"] = dict(waiter_override or override)
    waiter["trial_share"] = bool(waiter.get("trial_share") or override.get("trial_share") or waiter_override.get("trial_share"))
    waiter["trial_mode"] = str(waiter.get("trial_mode") or override.get("trial_mode") or waiter_override.get("trial_mode") or "")
    waiter["trial_initial_observe_sec"] = _float(
        waiter.get("trial_initial_observe_sec") or override.get("trial_initial_observe_sec") or waiter_override.get("trial_initial_observe_sec"),
        0.0,
    )
    waiter["job_id"] = str(waiter.get("job_id") or waiter_id)
    secondary_class = str(waiter.get("resource_class") or override.get("waiter_resource_class") or admission_result.get("policy_resource_class") or admission_result.get("resource_class") or "")
    slot_weight = _float(override.get("waiter_effective_slot_weight") or waiter.get("slot_weight") or admission_result.get("slot_weight"), 1.0)

    cpu_isolation = cpu_isolation_summary(snapshot, primary_job_id=primary_id, waiter=waiter)
    cpu_isolated = bool(cpu_isolation.get("isolated"))
    allowed_secondary = _secondary_allowed(
        cfg,
        secondary_class,
        slot_weight=slot_weight,
        cpu_pressure="unknown",
        cpu_isolated=cpu_isolated,
        waiter=waiter,
    )
    memory = gpu_memory_summary(gpu_sample if isinstance(gpu_sample, dict) else {}, primary_gpu_ids, previous_peak_gb=0.0)
    secondary_peak = secondary_estimated_peak_gb(secondary_class, cfg)
    trial_share = bool(waiter.get("trial_share"))
    unknown_primary_spike_gb = secondary_peak if trial_share else 0.0
    required_headroom = secondary_peak + unknown_primary_spike_gb + cfg.safety_margin_gb
    coarse_memory_ok = (
        memory["sample_available"]
        and memory["gpu_free_mem_gb"] >= cfg.min_free_mem_gb
        and (memory["gpu_mem_used_ratio"] <= cfg.max_mem_used_ratio or memory["gpu_mem_total_gb"] <= 0)
    )
    memory_headroom_ok = bool(coarse_memory_ok and memory["gpu_free_mem_gb"] >= required_headroom)
    util_ok = bool(memory["sample_available"] and memory["gpu_util_p90_pct"] <= cfg.util_p90_pct)
    existing_secondaries = [
        str(job_id)
        for job_id, raw in leases.items()
        if isinstance(raw, dict)
        and isinstance(raw.get("metadata"), dict)
        and str(raw["metadata"].get("shared_primary_job_id") or "") == primary_id
    ]
    trial_decision_path_available = (
        (not trial_share)
        or bool(allowed_secondary.get("requires_llm_policy_decision"))
        or cfg.trial_admission_policy == "deterministic_grant_when_hard_gates_pass"
    )
    hard_gates = {
        "primary_lease_exists": True,
        "same_task_gpu_waiter_exists": True,
        "secondary_limit_available": len(existing_secondaries) < 1,
        "gpu_util_p90_lte_threshold": util_ok,
        "memory_headroom_for_secondary": memory_headroom_ok,
        "secondary_allowed": bool(allowed_secondary.get("allowed")),
        "trial_share_requires_llm_grant": trial_decision_path_available,
        "trial_share_decision_path_available": trial_decision_path_available,
    }
    share_eligible = bool(all(hard_gates.values()))
    return {
        "enabled": True,
        "reason": "admission_share_eligible" if share_eligible else "admission_share_not_eligible",
        "phase": cfg.phase,
        "observe_only": not cfg.grant_active,
        "primary_job_id": primary_id,
        "primary_worker_id": str(primary.get("worker_id") or primary_meta.get("worker_id") or ""),
        "primary_resource_class": str(primary_meta.get("policy_resource_class") or primary_meta.get("resource_class") or ""),
        "primary_gpu_ids": primary_gpu_ids,
        "waiter_job_id": waiter_id,
        "share_candidate": True,
        "share_eligible": share_eligible,
        "cpu_pressure": "unknown",
        "cpu_isolation": cpu_isolation,
        "memory": {**memory, "required_headroom_gb": required_headroom, "secondary_estimated_peak_gb": secondary_peak, "unknown_primary_spike_gb": unknown_primary_spike_gb},
        "hard_gates": hard_gates,
        "waiter": _compact_waiter(waiter),
        "waiters": [_compact_waiter(waiter)],
        "secondary_allowed": allowed_secondary,
        "trial_share": {
            "enabled": trial_share,
            "mode": str(waiter.get("trial_mode") or ""),
            "initial_observe_sec": _float(waiter.get("trial_initial_observe_sec"), 0.0),
            "primary_protected": trial_share,
            "admission_immediate_review": True,
        },
        "admission_result": {
            "status": str(admission_result.get("status") or ""),
            "reason": str(admission_result.get("reason") or ""),
            "queue_position": admission_result.get("queue_position"),
            "queue_len": admission_result.get("queue_len"),
            "share_override_candidate": bool(admission_result.get("share_override_candidate")),
        },
    }


def share_waiters_for_primary(snapshot: dict[str, Any], *, primary_job_id: str, primary_gpu_ids: list[str]) -> list[dict[str, Any]]:
    wanted = {str(x) for x in (primary_gpu_ids or []) if str(x).strip()}
    waiters = snapshot.get("waiters") if isinstance(snapshot.get("waiters"), dict) else {}
    leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
    rows: list[dict[str, Any]] = []
    now = _float(snapshot.get("now"), 0.0)
    clean_primary = str(primary_job_id or "")
    for waiter_id, raw in waiters.items():
        clean_waiter_id = str(waiter_id)
        if clean_waiter_id == clean_primary or not isinstance(raw, dict):
            continue
        if isinstance(leases.get(clean_waiter_id), dict):
            continue
        gpu_ids = {str(x) for x in (raw.get("gpu_ids") or raw.get("candidate_gpu_ids") or []) if str(x).strip()}
        if wanted and gpu_ids and not (wanted & gpu_ids):
            continue
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        override = raw.get("share_override") if isinstance(raw.get("share_override"), dict) else meta.get("share_override") if isinstance(meta.get("share_override"), dict) else {}
        primary_ids = {str(x) for x in (override.get("primary_job_ids") or []) if str(x).strip()} if isinstance(override, dict) else set()
        requested_at = _float(raw.get("share_override_requested_at") or meta.get("share_override_requested_at") or (override or {}).get("requested_at"), 0.0)
        row = dict(raw)
        row["job_id"] = str(row.get("job_id") or clean_waiter_id)
        row["queue_age_sec"] = max(0.0, now - _float(row.get("submitted_at"), now)) if now > 0 else 0.0
        row["share_override"] = dict(override) if isinstance(override, dict) else {}
        row["share_override_primary_match"] = bool(clean_primary and clean_primary in primary_ids)
        row["share_override_requested_at"] = requested_at
        row["trial_share"] = bool((override or {}).get("trial_share"))
        row["trial_mode"] = str((override or {}).get("trial_mode") or "")
        row["trial_initial_observe_sec"] = _float((override or {}).get("trial_initial_observe_sec"), 0.0)
        rows.append(row)
    rows.sort(key=lambda item: (
        0 if bool(item.get("share_override_primary_match")) else 1,
        -_float(item.get("share_override_requested_at"), 0.0),
        -_float(item.get("priority_score"), 0.0),
        _float(item.get("submitted_at"), 0.0),
        str(item.get("job_id") or ""),
    ))
    return rows


def cpu_isolation_summary(snapshot: dict[str, Any], *, primary_job_id: str, waiter: dict[str, Any]) -> dict[str, Any]:
    leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
    primary = leases.get(str(primary_job_id)) if isinstance(leases.get(str(primary_job_id)), dict) else {}
    primary_meta = primary.get("metadata") if isinstance(primary.get("metadata"), dict) else {}
    waiter_meta = waiter.get("metadata") if isinstance(waiter.get("metadata"), dict) else {}
    primary_cpus = _parse_cpu_set(primary_meta.get("cpu_set"))
    waiter_cpus = _parse_cpu_set(waiter_meta.get("cpu_set"))
    primary_text = str(primary_meta.get("cpu_set") or "")
    waiter_text = str(waiter_meta.get("cpu_set") or "")
    if primary_cpus and waiter_cpus:
        overlap = sorted(primary_cpus & waiter_cpus)
        if not overlap:
            return {
                "mode": "isolated",
                "isolated": True,
                "primary_cpu_set": primary_text,
                "waiter_cpu_set": waiter_text,
                "reason": "disjoint_cpuset",
            }
        return {
            "mode": "shared",
            "isolated": False,
            "primary_cpu_set": primary_text,
            "waiter_cpu_set": waiter_text,
            "overlap_count": len(overlap),
            "reason": "overlapping_cpuset",
        }
    return {
        "mode": "unknown",
        "isolated": False,
        "primary_cpu_set": primary_text,
        "waiter_cpu_set": waiter_text,
        "reason": "missing_cpuset",
    }


def evaluate_share_phase_a(
    *,
    cfg: GPUShareConfig,
    primary_job_id: str,
    primary_resource_class: str,
    primary_gpu_ids: list[str],
    elapsed_sec: float,
    progress_snapshot: dict[str, Any],
    gpu_sample: dict[str, Any],
    previous_mem_peak_gb: float,
    process_tree_cpu: dict[str, Any],
    snapshot: dict[str, Any],
    primary_kill_replan_candidate: dict[str, Any],
) -> dict[str, Any]:
    if not cfg.observe_active:
        return {"enabled": False, "reason": "gpu_share_disabled"}
    waiters = share_waiters_for_primary(snapshot, primary_job_id=primary_job_id, primary_gpu_ids=primary_gpu_ids)
    if not waiters:
        return {"enabled": False, "reason": "no_same_task_gpu_waiter", "waiters": []}
    memory = gpu_memory_summary(gpu_sample, primary_gpu_ids, previous_peak_gb=previous_mem_peak_gb)
    primary_kill = bool(primary_kill_replan_candidate.get("share_blocking_candidate", primary_kill_replan_candidate.get("candidate")))
    cpu_pressure = classify_cpu_pressure(process_tree_cpu, cpu_policy=cfg.cpu_policy)
    best_waiter = waiters[0]
    cpu_isolation = cpu_isolation_summary(snapshot, primary_job_id=primary_job_id, waiter=best_waiter)
    cpu_isolated = bool(cpu_isolation.get("isolated"))
    secondary_class = str(best_waiter.get("resource_class") or "")
    slot_weight = _float(best_waiter.get("slot_weight"), 1.0)
    trial_share = bool(best_waiter.get("trial_share"))
    effective_warmup_sec = share_warmup_sec(cfg, trial_share=trial_share)
    secondary_peak = secondary_estimated_peak_gb(secondary_class, cfg)
    required_headroom = secondary_peak + max(0.0, memory["gpu_mem_peak_gb"] - memory["gpu_mem_current_gb"]) + cfg.safety_margin_gb
    coarse_memory_ok = (
        memory["gpu_free_mem_gb"] >= cfg.min_free_mem_gb
        and (memory["gpu_mem_used_ratio"] <= cfg.max_mem_used_ratio or memory["gpu_mem_total_gb"] <= 0)
    )
    memory_headroom_ok = bool(coarse_memory_ok and memory["gpu_free_mem_gb"] >= required_headroom)
    artifact_updates = progress_snapshot.get("artifact_updates") if isinstance(progress_snapshot.get("artifact_updates"), list) else []
    artifact_recent = bool(artifact_updates and _float(progress_snapshot.get("artifact_last_update_age_sec"), 1e9) < 300.0)
    stdout_age_raw = progress_snapshot.get("stdout_last_line_age_sec", progress_snapshot.get("stdout_age_sec"))
    stdout_recent = bool(
        _float(progress_snapshot.get("stdout_lines"), 0.0) > 0
        and (stdout_age_raw is None or _float(stdout_age_raw, 0.0) < 300.0)
    )
    recent_signal = bool(
        artifact_recent
        or stdout_recent
        or _float(progress_snapshot.get("metric_last_update_age_sec"), 1e9) < 300.0
    )
    active_work_signal = isolated_active_work_signal(process_tree_cpu, cpu_isolated=cpu_isolated)
    productive_primary_signal = bool(recent_signal or active_work_signal)
    progress_signal = str(progress_snapshot.get("progress_signal") or "unknown").lower()
    allowed_secondary = _secondary_allowed(
        cfg,
        secondary_class,
        slot_weight=slot_weight,
        cpu_pressure=cpu_pressure,
        cpu_isolated=cpu_isolated,
        waiter=best_waiter,
    )
    share_candidate = bool(elapsed_sec >= effective_warmup_sec and not primary_kill)
    trial_decision_path_available = (
        (not trial_share)
        or bool(allowed_secondary.get("requires_llm_policy_decision"))
        or cfg.trial_admission_policy == "deterministic_grant_when_hard_gates_pass"
    )
    hard_gates = {
        "same_task_gpu_waiter_exists": True,
        "primary_runtime_sec_gte_warmup": elapsed_sec >= effective_warmup_sec,
        "primary_kill_replan_candidate_false": not primary_kill,
        "gpu_util_p90_lte_threshold": memory["gpu_util_p90_pct"] <= cfg.util_p90_pct,
        "memory_headroom_for_secondary": memory_headroom_ok,
        "progress_signal_not_stalled": progress_signal != "stalled",
        "productive_primary_signal": productive_primary_signal,
        "cpu_pressure_share_safe": cpu_pressure != "high" or cpu_isolated,
        "secondary_allowed": allowed_secondary["allowed"],
        "trial_share_requires_llm_grant": trial_decision_path_available,
        "trial_share_decision_path_available": trial_decision_path_available,
    }
    share_eligible = bool(share_candidate and all(hard_gates.values()))
    return {
        "enabled": True,
        "reason": "share_eligible" if share_eligible else "share_candidate_observed" if share_candidate else "share_not_candidate",
        "phase": cfg.phase,
        "observe_only": not cfg.grant_active,
        "share_warmup_sec": effective_warmup_sec,
        "standard_warmup_sec": float(cfg.warmup_sec or 0.0),
        "primary_job_id": str(primary_job_id),
        "primary_resource_class": str(primary_resource_class or ""),
        "primary_gpu_ids": [str(x) for x in primary_gpu_ids],
        "primary_kill_replan_candidate": primary_kill_replan_candidate,
        "share_candidate": share_candidate,
        "share_eligible": share_eligible,
        "cpu_pressure": cpu_pressure,
        "cpu_isolation": cpu_isolation,
        "memory": {**memory, "required_headroom_gb": required_headroom, "secondary_estimated_peak_gb": secondary_peak},
        "primary_signal": {
            "recent_artifact_or_metric_or_stdout": recent_signal,
            "isolated_active_work_signal": active_work_signal,
            "productive_primary_signal": productive_primary_signal,
        },
        "hard_gates": hard_gates,
        "waiter": _compact_waiter(best_waiter),
        "waiters": [_compact_waiter(row) for row in waiters[:3]],
        "secondary_allowed": allowed_secondary,
        "trial_share": {
            "enabled": trial_share,
            "mode": str(best_waiter.get("trial_mode") or ""),
            "initial_observe_sec": _float(best_waiter.get("trial_initial_observe_sec"), 0.0),
            "primary_protected": trial_share,
        },
    }


def _secondary_allowed(
    cfg: GPUShareConfig,
    resource_class: str,
    *,
    slot_weight: float,
    cpu_pressure: str,
    cpu_isolated: bool = False,
    waiter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cls = str(resource_class or "")
    if cls == RESOURCE_UNKNOWN_GPU_EXEC:
        if not _is_revocable_trial_train_secondary_candidate(cfg, cls, waiter=waiter):
            return {"allowed": False, "reason": "unknown_gpu_secondary_requires_revocable_trial"}
        if not cpu_isolated:
            return {"allowed": False, "reason": "unknown_gpu_trial_requires_cpu_isolation"}
        return {
            "allowed": True,
            "reason": "unknown_gpu_revocable_trial_allowed",
            "effective_secondary_class": RESOURCE_UNKNOWN_GPU_EXEC,
            "effective_slot_weight": min(float(cfg.secondary_max_slot_weight or 0.5), 0.5),
            "requires_llm_policy_decision": False,
            "policy_gate": "unknown_gpu_revocable_trial",
            "trial_share": True,
            "risk_class": "unknown",
        }
    if not cls:
        return {"allowed": False, "reason": "missing_gpu_secondary_class"}
    if _is_revocable_trial_train_secondary_candidate(cfg, cls, waiter=waiter):
        if not cpu_isolated:
            return {"allowed": False, "reason": "trial_heavy_secondary_requires_cpu_isolation"}
        return {
            "allowed": True,
            "reason": "revocable_heavy_train_trial_allowed",
            "effective_secondary_class": RESOURCE_HEAVY_GPU_TRAIN,
            "effective_slot_weight": min(float(cfg.secondary_max_slot_weight or 0.5), 0.5),
            "requires_llm_policy_decision": True,
            "policy_gate": "revocable_heavy_train_trial",
            "trial_share": True,
        }
    if _is_low_footprint_train_secondary_candidate(cfg, cls, waiter=waiter):
        if cpu_pressure == "high" and not cpu_isolated:
            return {"allowed": False, "reason": "cpu_pressure_high_unisolated"}
        return {
            "allowed": True,
            "reason": "low_footprint_train_secondary_allowed_cpu_isolated" if cpu_pressure == "high" else "low_footprint_train_secondary_allowed",
            "effective_secondary_class": RESOURCE_GPU_LIGHT_TRAIN,
            "effective_slot_weight": min(float(cfg.secondary_max_slot_weight or 0.5), 0.5),
            "requires_llm_policy_decision": True,
            "policy_gate": "bounded_train_secondary",
        }
    if _is_isolated_train_secondary_candidate(cfg, cls, cpu_isolated=cpu_isolated, waiter=waiter):
        return {
            "allowed": True,
            "reason": "isolated_train_secondary_allowed",
            "effective_secondary_class": RESOURCE_HEAVY_GPU_TRAIN,
            "effective_slot_weight": min(float(cfg.secondary_max_slot_weight or 0.5), 0.5),
            "requires_llm_policy_decision": True,
            "policy_gate": "bounded_heavy_train_secondary",
        }
    if slot_weight > cfg.secondary_max_slot_weight:
        return {"allowed": False, "reason": "secondary_slot_weight_too_high"}
    if cls == RESOURCE_GPU_FEATURE_EXTRACT and cfg.phase not in {"feature_share", "light_train"}:
        return {"allowed": False, "reason": "feature_secondary_phase_disabled"}
    if cls == RESOURCE_GPU_LIGHT_TRAIN and cfg.phase not in {"feature_share", "light_train"}:
        return {"allowed": False, "reason": "light_train_phase_disabled"}
    if cls not in {RESOURCE_GPU_TT_LIGHT, RESOURCE_GPU_FEATURE_EXTRACT, RESOURCE_GPU_LIGHT_TRAIN}:
        return {"allowed": False, "reason": "secondary_class_not_allowed"}
    if cpu_pressure == "high" and not cpu_isolated:
        return {"allowed": False, "reason": "cpu_pressure_high_unisolated"}
    if cpu_pressure == "medium" and cls != RESOURCE_GPU_TT_LIGHT and not cpu_isolated:
        return {"allowed": False, "reason": "cpu_pressure_medium_feature_denied"}
    if cls == RESOURCE_GPU_LIGHT_TRAIN and cfg.phase == "feature_share":
        return {
            "allowed": True,
            "reason": "light_train_policy_deferred_to_arbiter",
            "effective_secondary_class": RESOURCE_GPU_LIGHT_TRAIN,
            "effective_slot_weight": min(float(cfg.secondary_max_slot_weight or 0.5), 0.5),
            "requires_llm_policy_decision": True,
            "policy_gate": "light_train_in_feature_share_phase",
        }
    if cpu_pressure in {"medium", "high"} and cpu_isolated:
        return {"allowed": True, "reason": "allowed_cpu_isolated"}
    return {"allowed": True, "reason": "allowed"}


def _is_revocable_trial_train_secondary_candidate(
    cfg: GPUShareConfig,
    resource_class: str,
    *,
    waiter: dict[str, Any] | None,
) -> bool:
    if cfg.phase not in {"feature_share", "light_train"}:
        return False
    if resource_class not in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
        return False
    override = waiter.get("share_override") if isinstance(waiter, dict) and isinstance(waiter.get("share_override"), dict) else {}
    meta = waiter.get("metadata") if isinstance(waiter, dict) and isinstance(waiter.get("metadata"), dict) else {}
    meta_override = meta.get("share_override") if isinstance(meta.get("share_override"), dict) else {}
    return bool(waiter and (waiter.get("trial_share") or override.get("trial_share") or meta_override.get("trial_share")))


def _is_isolated_train_secondary_candidate(
    cfg: GPUShareConfig,
    resource_class: str,
    *,
    cpu_isolated: bool,
    waiter: dict[str, Any] | None,
) -> bool:
    if not cpu_isolated or cfg.phase not in {"feature_share", "light_train"}:
        return False
    if resource_class not in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
        return False
    meta = waiter.get("metadata") if isinstance(waiter, dict) and isinstance(waiter.get("metadata"), dict) else {}
    source_hint = meta.get("source_hint") if isinstance(meta.get("source_hint"), dict) else {}
    if source_hint:
        return bool(source_hint.get("command_train_evidence") or source_hint.get("source_train_evidence"))
    return True


def _is_low_footprint_train_secondary_candidate(
    cfg: GPUShareConfig,
    resource_class: str,
    *,
    waiter: dict[str, Any] | None,
) -> bool:
    if cfg.phase not in {"feature_share", "light_train"}:
        return False
    if resource_class not in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
        return False
    meta = waiter.get("metadata") if isinstance(waiter, dict) and isinstance(waiter.get("metadata"), dict) else {}
    source_hint = meta.get("source_hint") if isinstance(meta.get("source_hint"), dict) else {}
    if not source_hint:
        return False
    explicit_gpu = any(
        bool(source_hint.get(key))
        for key in (
            "command_gpu_compute_evidence",
            "command_gpu_evidence",
            "command_gpu_request",
            "source_gpu_request",
        )
    )
    if explicit_gpu:
        return False
    return bool(source_hint.get("command_train_evidence") or source_hint.get("source_train_evidence"))


def _parse_cpu_set(value: Any) -> set[int]:
    text = str(value or "").strip()
    if not text:
        return set()
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left.strip())
                end = int(right.strip())
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            out.update(range(max(0, start), max(0, end) + 1))
            continue
        try:
            out.add(max(0, int(part)))
        except ValueError:
            continue
    return out


def _compact_waiter(raw: dict[str, Any]) -> dict[str, Any]:
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    value_hint = meta.get("value_hint") if isinstance(meta.get("value_hint"), dict) else {}
    override = raw.get("share_override") if isinstance(raw.get("share_override"), dict) else meta.get("share_override") if isinstance(meta.get("share_override"), dict) else {}
    return {
        "job_id": str(raw.get("job_id") or ""),
        "worker_id": str(raw.get("worker_id") or ""),
        "resource_class": str(raw.get("resource_class") or ""),
        "slot_weight": _float(raw.get("slot_weight"), 1.0),
        "gpu_ids": [str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()],
        "cpu_set": str(meta.get("cpu_set") or ""),
        "queue_age_sec": _float(raw.get("queue_age_sec"), 0.0),
        "priority_score": _float(raw.get("priority_score"), 0.0),
        "share_override_primary_match": bool(raw.get("share_override_primary_match")),
        "share_override_requested_at": _float(raw.get("share_override_requested_at"), 0.0),
        "trial_share": bool(raw.get("trial_share") or override.get("trial_share")),
        "trial_mode": str(raw.get("trial_mode") or override.get("trial_mode") or ""),
        "trial_initial_observe_sec": _float(raw.get("trial_initial_observe_sec") or override.get("trial_initial_observe_sec"), 0.0),
        "expected_value_score": _float(value_hint.get("expected_value_score"), 0.0),
        "near_submission_score": _float(value_hint.get("near_submission_score"), 0.0),
    }


def _choice(value: str, allowed: set[str], default: str) -> str:
    clean = str(value or "").strip().lower()
    return clean if clean in allowed else default


def _float(value: Any, default: float) -> float:
    out = _float_or_none(value)
    return default if out is None else out


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    index = int(round((len(rows) - 1) * max(0.0, min(1.0, q))))
    return rows[index]
