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
import copy
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from deepcraft_core import Message
from deepcraft_core.llm import StreamHandle
from deepcraft_core.tool import ToolResult

from scienceflow.core.agent.run_policy import AutoContinuePolicy, RoundContext
from scienceflow.config.settings import Config
from scienceflow.core.llm_http import aclose_llm_clients
from scienceflow.core.task_package import find_task_package
from scienceflow.core.agent_runtime import (
    _build_llm,
    select_prefix_safe_agent_memory_records,
    write_agent_memory_record_files,
)
from scienceflow.gates.evaluator import (
    EvalContext,
    EvaluationRequest,
)
from scienceflow.gates.evaluator.backends.command_env import (
    task_command_env,
    task_python_executable,
)
from scienceflow.gates import (
    GateService,
    format_invalid_evaluator_feedback,
    legacy_score_contract_enabled,
)
from scienceflow.gates.evaluator.adapters import (
    merge_adjudicated_stage_facts,
    merge_primary_stage_facts,
    metric_event_to_stage_facts,
)
from scienceflow.core.tools.resource_classifier import normalize_shell_command
from scienceflow.core.tools.file_utils import atomic_write
from scienceflow.core.skills.paths import default_skill_library_dir
from scienceflow.core.skills.registry import SkillRegistry
from scienceflow.solver.lnr.prompts import (
    build_first_user_prompt,
    build_metric_output_interpreter_prompt,
    build_metric_output_interpreter_system_prompt,
    build_metric_validity_adjudicator_prompt,
    build_metric_validity_adjudicator_system_prompt,
    build_estra_resume_prompt,
    build_estra_prompt,
    build_estra_archive_summary_prompt,
    build_stage_commit_judgment_prompt,
    build_keep_current_compact_prompt,
)
from scienceflow.solver.lnr.context_hygiene import (
    evaluate_context_hygiene_compact,
    large_code_file_touch_counts,
    large_tool_output_count,
)
from scienceflow.solver.lnr.context_memory_policy import ensure_lnr_context_memory_budget
from scienceflow.solver.lnr.init_workspace import initialize_workspace_from_path
from scienceflow.solver.lnr.stage.metric_semantics import classify_metric_semantics
from scienceflow.solver.lnr.stage.metric_adjudication import (
    adjudicate_metric_validity,
    build_metric_validity_fields,
    parse_metric_output_interpretation_text,
    parse_metric_validity_judgment_text,
)
from scienceflow.solver.lnr.stage.metric_validity import infer_metric_validity
from scienceflow.solver.lnr.stage.stage_files import format_stage_files
from scienceflow.solver.lnr.estra_magent.join_gate import load_join_packets
from scienceflow.solver.lnr.estra_magent.prompt_blocks import format_magent_recommendations
from scienceflow.solver.lnr.resource_observer import LHRResourceObserver
from scienceflow.solver.lnr.resource_advisory import (
    INLINE_RESOURCE_ADVISORY_MODE,
    build_inline_resource_advisory_prompt,
    normalize_inline_resource_advisory,
    parse_inline_resource_advisory_response,
    safe_inline_resource_advisory_boundary,
)
from scienceflow.solver.lnr.resource_runtime.review.arbiter import (
    build_resource_arbiter_prompt,
    fallback_policy_decision,
    normalize_arbiter_decision,
    parse_arbiter_decision_text,
)
from scienceflow.solver.lnr.resource_runtime.admission import build_resource_admission_prompt
from scienceflow.solver.lnr.resume import resume_loaded_agent_from_memory
from scienceflow.solver.lnr.global_merge import metric_float, run_global_merge
from scienceflow.solver.lnr.global_merge.candidate_evidence import (
    apply_candidate_evidence,
    load_peer_candidate_evidence,
    recover_candidate_artifact,
)
from scienceflow.solver.lnr.submission_links import (
    canonicalize_worker_workspace_artifacts,
    refresh_submission_links,
)
from scienceflow.solver.lnr.stage.score_summary import (
    build_stage_performance_score_summary,
    format_score_summary_context,
    infer_stage_rows_lower_is_better,
    metric_lower_is_better_hint,
)
from scienceflow.solver.lnr.stage.peer_context import (
    PeerRouteEvidence,
    build_peer_route_evidence_from_csv,
)
from scienceflow.solver.lnr.snapshot_store import SnapshotStore, StageSnapshot
from scienceflow.solver.lnr.stage_logs import (
    append_stage_event,
    attach_stage_interaction_handlers,
    ensure_stage_log_dir,
    reset_stage_log_dir,
    stage_log_dir,
)
from scienceflow.solver.lnr.stage_memory import build_stage_memory_view, sync_current_segment
from scienceflow.solver.lnr.stage.stage_ledger import (
    StageCard,
    append_archived_trajectory_summary,
    append_stage_event_summary,
    next_stage_id,
    normalize_stage_id,
    parse_stage_cards,
    read_ledger,
    render_stage_cards,
    stage_cards_with_overrides,
    tail_summary_from_cards,
    tail_summary_after_cards,
    validate_append_only_stage_commit,
)
from scienceflow.solver.lnr.state_machine import LHR_EVENTS_JSONL, LHRStateMachineStore
from scienceflow.solver.lnr.worker_layout import ensure_lnr_worker_layout, lnr_worker_layout
from scienceflow.utils.node_paths import find_node_log_path
from scienceflow.utils.resource_utils import parse_cpu_list
from scienceflow.utils.workspace_interaction_log import (
    attach_workspace_interaction_logger,
    close_workspace_interaction_logger,
)
from scienceflow.utils.workspace_git import (
    archive_workspace_candidate_artifact,
    auto_checkpoint_workspace_source,
    ensure_workspace_source_git,
    normalize_workspace_git_track_globs,
    workspace_source_changed,
)

logger = logging.getLogger("scienceflow")


def _effective_lnr_bash_timeout_sec(
    configured_sec: float,
    remaining_sec: float,
) -> float:
    """Return the per-command cap bounded by the task hard fuse."""
    configured = max(1.0, float(configured_sec or 1.0))
    remaining = max(1.0, float(remaining_sec or 1.0))
    return min(configured, remaining)


_RESOURCE_FEEDBACK_SYSTEM_PROTOCOL = """Resource feedback is runtime context, not a normal shell error.
Rules:
- PENDING/REPLAN/policy blocked: update the plan; optional standalone sleep/backoff; do not retry the same blocked GPU command unchanged.
- recommend_stop/recommend_release: resource system found inefficient work; stop, wait, or rerun only with a changed lighter/checkpointed plan.
- terminated feedback: auto mode already stopped the command; inspect artifacts/logs and replan.
- OOM/memory pressure: reduce batch, image size, model size, workers, or precision cost; CPU/light work is valid.
- RED resource mode: avoid new heavy GPU training unless clearly worth waiting for; sleep/backoff is valid.
- STOP_BOUNDARY_VIOLATION: remove physical GPU-id overrides; use runner-provided visible devices and rerun after patching.
- Resource truth: use ResourceContext/RESOURCE_FEEDBACK, not nvidia-smi or torch.cuda probes, before heavy GPU work.
- Value hint: optionally prefix valuable GPU work with SCIENCEFLOW_RESOURCE_VALUE_HINT={"expected_value_score":0.8,"near_submission_score":0.7}.
- Resource intent: prefix bash with SCIENCEFLOW_RESOURCE_INTENT=cpu_support|gpu_train|light_train|gpu_tt|readonly_cpu when the command purpose is clear; this is a hint and GPU evidence overrides CPU hints.
- Heartbeat: any command, script phase, loop, or optimization process expected to run over 2 minutes must print flushed SCIENCEFLOW_HB lines every 30-60s.
- Use SCIENCEFLOW_HB for all long phases: data preparation, search/optimization, evaluation, inference, aggregation, export, and artifact writing.
- Heartbeat format: SCIENCEFLOW_HB v=1 phase=<phase> tick=<n> elapsed_s=<sec> progress=<done>/<total_or_unknown> unit=<unit> metric=<name:value_or_na> loss=<value_or_na> artifact=<path_or_none>.
- If no metric is available, use metric=na; when a deliverable is written, print artifact=<path>.
- Bypass: do not obfuscate commands or code to evade resource classification; write normal code and let scheduling work.
- Stale context: if preflight returns stale_resource_context or newer RESOURCE_FEEDBACK, treat it as current truth and replan.
The resource manager reports constraints and observations; the main agent chooses the scientific plan using feedback, artifacts, scores, and remaining time.
"""

_LNR_MERGE_PAYLOAD_SYSTEM_BOUNDARY = """Optional reduction payload boundary. The final reducer targets three distinct defensible artifacts. During normal research, preserve materially different validated candidate artifacts when they are produced, and retain reusable prediction-level outputs under `merge_payload/` with compact metadata describing IDs, shape, class order or output semantics, and validation provenance. Do not create duplicate or cosmetically perturbed artifacts merely to reach three. For expensive inference, reuse retained predictions instead of rerunning solely for reduction; do not save model weights only for the reducer. The controller snapshots candidate artifacts and merge payloads for final cross-stage reduction.
"""


_LNR_LEDGER_SYSTEM_BOUNDARY = """LNR ledger boundary. Record experiment facts only in ordinary workspace artifacts or scratch notes such as `tmp/experiment_notes.md`, `tmp/run_notes.md`, or task-local artifacts when useful. Do not create extra planning systems, stage-like ledgers, or authoritative status files. Do not read or rewrite the hidden stage ledger unless the controller explicitly starts a stage-commit bookkeeping turn. The hidden stage ledger is append-only and controller-owned.
"""


def _append_lnr_main_agent_protocols(text: str) -> str:
    out = str(text or "").rstrip()
    for marker, block in (
        ("Resource feedback is runtime context", _RESOURCE_FEEDBACK_SYSTEM_PROTOCOL),
        ("LNR ledger boundary", _LNR_LEDGER_SYSTEM_BOUNDARY),
        ("Optional reduction payload boundary", _LNR_MERGE_PAYLOAD_SYSTEM_BOUNDARY),
    ):
        if marker not in out:
            out = (out + "\n\n" if out else "") + block.strip()
    return out + ("\n" if out else "")

LHR_STAGE_PERFORMANCE_CSV = "lhr_stage_performance.csv"
LHR_UNIFIED_ONLY_JSONL = {
    "lhr_stage_events.jsonl",
    "lhr_stage_commit_events.jsonl",
    "lhr_estra_events.jsonl",
    "lhr_estras.jsonl",
    "lhr_coordinator_events.jsonl",
    "lhr_context_events.jsonl",
}

_LHR_EDA_FACT_RE = re.compile(
    r"(train|test|valid|validation|sample|submission|shape|rows|columns|dtype|"
    r"missing|null|nan|unique|target|spacegroup|metric|rmsle|id,|percent_|"
    r"lattice_|formation_energy|bandgap_energy|min|max|mean|std|corr|skew)",
    flags=re.IGNORECASE,
)
_LHR_EXPERIMENT_NOISE_RE = re.compile(
    r"(Final Validation Score|Validation RMSLE|Individual model|Total models|"
    r"RandomForest|ExtraTrees|XGBoost|LightGBM|CatBoost|MLPRegressor|Ridge|"
    r"n_estimators|max_depth|num_leaves|learning_rate|avg=|form=|band=)",
    flags=re.IGNORECASE,
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def _validation_leakage_reason(solution_src: str, stdout_tail: str) -> str:
    src = str(solution_src or "")
    out = str(stdout_tail or "")
    src_lower = src.lower()
    out_lower = out.lower()
    concat_match = re.search(
        r"pd\s*\.\s*concat\s*\(\s*\[\s*(?:train|train_df|df_train)\s*,\s*(?:val|valid|validation|val_df|valid_df)\b",
        src,
        flags=re.IGNORECASE,
    )
    if not concat_match:
        return ""
    score_pos = src_lower.rfind("final validation score")
    if score_pos >= 0 and concat_match.start() > score_pos:
        return ""
    if "optimizing ensemble weights on validation" in out_lower:
        return "train_validation_concat_before_validation_weight_optimization"
    if "training final model" in out_lower and "validation" in out_lower and "final validation score" in out_lower:
        return "train_validation_concat_before_reported_metric"
    return ""

LHR_STAGE_PERFORMANCE_COLUMNS = [
    "row_order",
    "candidate_id",
    "worker_id",
    "worker_index",
    "worker_stage_order",
    "stage_id",
    "lineage_id",
    "node_uid",
    "parent_candidate_id",
    "parent_stage_id",
    "restored_from_candidate_id",
    "restored_from_stage",
    "restored_from_node_uid",
    "metric_value",
    "metric_name",
    "lower_is_better",
    "validation_ok",
    "validation_issue",
    "reported_val_score",
    "val_score_type",
    "selection_eligible",
    "selection_score",
    "selection_note",
    "metric_source_note",
    "metric_validity",
    "metric_validity_note",
    "metric_validity_reason_code",
    "metric_validity_confidence",
    "metric_validity_source",
    "task_profile",
    "metric_protocol",
    "train_data_used",
    "metric_eval_data",
    "artifacts_reused_from",
    "training_rows",
    "validation_rows",
    "execution_mode",
    "is_best_so_far",
    "brief",
    "why",
    "route_evidence",
    "solution_sha",
    "submission_sha",
    "source_commit_sha",
    "source_changed",
    "semantic_source_changed",
    "capture_type",
    "workspace_git_stage_id",
    "submission_snapshot",
    "artifact_path",
    "artifact_sha",
    "artifact_kind",
    "evaluator_backend",
    "evaluator_status",
    "gate_metric_validity",
    "gate_policy",
    "gate_policy_version",
    "gate_action",
    "gate_accepted",
    "gate_reason_code",
    "submission_changed",
    "candidate_ready",
    "submission_status",
    "duplicate_submission_of_stage",
    "duplicate_submission_of_snapshot_id",
    "workspace_git_ready",
    "workspace_git_message",
    "solution_run_sec",
    "elapsed_min",
    "main_llm_calls",
    "main_tokens_input",
    "main_tokens_output",
    "main_tokens_cached",
    "main_cache_rate",
    "stage_commit_llm_calls",
    "estra_llm_calls",
    "estra_decision_count_before",
    "estra_continue_count_before",
    "estra_redirect_count_before",
    "estra_switch_count_before",
    "estra_current_continue_count_before",
    "estra_current_redirect_count_before",
    "estra_stage_continue_count_before",
    "estra_stage_redirect_count_before",
    "estra_switch_stage_count_before",
    "snapshot_id",
    "snapshot_path",
    "created_at_utc",
    "route_id",
    "execution_scale",
]


class _WallClockAutoContinuePolicy(AutoContinuePolicy):
    def __init__(self, *, deadline_monotonic: float, max_text_only_retries: int = 2) -> None:
        super().__init__(max_text_only_retries=max_text_only_retries)
        self.deadline_monotonic = float(deadline_monotonic)

    def on_round_start(self, ctx: RoundContext) -> bool:
        _ = ctx
        return time.monotonic() < self.deadline_monotonic


class LnrSolver:
    solver_name = "lnr"

    def __init__(
        self,
        *,
        task_desc: str,
        cfg: Config,
        orchestrator: Any,
        worker_id: str = "",
        worker_index: int = 0,
        worker_count: int = 1,
        worker_extra_env: dict[str, str] | None = None,
        task_root_dir: str | Path | None = None,
    ) -> None:
        self.task_desc = str(task_desc or "")
        self._task_metric_lower_is_better = metric_lower_is_better_hint(self.task_desc)
        self.cfg = cfg
        self.lhr = cfg.lnr
        memory_policy = ensure_lnr_context_memory_budget(self.cfg, self.lhr)
        if memory_policy.changed:
            logger.info(
                "[lnr] raised max_messages from %d to %d (%s)",
                memory_policy.original_max_messages,
                memory_policy.effective_max_messages,
                memory_policy.reason,
            )
        self.orchestrator = orchestrator
        self.worker_id = str(worker_id or "")
        self.worker_index = int(worker_index or 0)
        self.worker_count = max(1, int(worker_count or 1))
        self.worker_extra_env = dict(worker_extra_env or {})
        self.root_dir = Path(cfg.task_workspace_root_dir).resolve()
        self.task_root_dir = (
            Path(task_root_dir).resolve()
            if task_root_dir is not None
            else self.root_dir
        )
        self.global_log_dir = self.task_root_dir / "task_logs"
        self.workspace_dir = Path(cfg.workspace_dir).resolve()
        configured_log_dir = Path(cfg.log_dir).resolve()
        self.log_dir = configured_log_dir if self.worker_id else self.global_log_dir
        self.ledger_filename = self._safe_ledger_filename(self.lhr.ledger_filename)
        self.ledger_path = self.workspace_dir / self.ledger_filename
        self.memory_dir = self.workspace_dir / ".agent_memory"
        self.snapshot_store = SnapshotStore(
            root_dir=self.root_dir,
            workspace_dir=self.workspace_dir,
            snapshot_dirname="snapshots",
            archive_dirname=str(
                getattr(self.lhr, "archive_dirname", "") or "snapshots/archives"
            ),
            control_log_dir=self.log_dir,
            metadata_dirname="logs",
            strict_layout=True,
            memory_dir=self.memory_dir,
            workspace_snapshot_enabled=bool(
                getattr(self.lhr, "workspace_snapshot_enabled", False)
            ),
            workspace_snapshot_verify_objects=bool(
                getattr(self.lhr, "workspace_snapshot_verify_objects", True)
            ),
        )
        self.deadline = time.monotonic() + max(
            60, int(self.lhr.wall_clock_budget_sec or 0)
        )
        self.stage_snapshots: dict[str, StageSnapshot] = {}
        self.archived_stage_snapshots: dict[str, StageSnapshot] = {}
        self.duplicate_submission_skip_keys: set[str] = set()
        self.last_captured_solution_sha = ""
        self.initial_workspace_state = ""
        self.last_captured_run_signature = ""
        self.last_stage_commit_ts = 0.0
        self._s01_eda_prefix_end_index: int | None = None
        self.last_estra_stage_count = 0
        self.last_estra_observation_key = ""
        self.last_force_estra_observation_count = 0
        self.last_force_estra_observation_key = ""
        self.last_context_limit_estra_generation = ""
        self.context_limit_estra_restore_keys: set[str] = set()
        self.estra_decisions = 0
        self.current_lineage_no = 1
        self.current_lineage_id = "L01"
        self.current_restored_from_node_uid = ""
        self.pending_estra: dict[str, Any] | None = None
        self.pending_stage_commit_transaction: dict[str, Any] | None = None
        self.main_tokens_in = 0
        self.main_tokens_out = 0
        self.main_tokens_cached = 0
        self.main_llm_calls = 0
        self.context_hygiene_cache_rates: list[float] = []
        self.context_hygiene_last_stage_tokens_in = 0
        self.context_hygiene_last_stage_tokens_cached = 0
        self.context_hygiene_last_compact_tokens_in = 0
        self.context_hygiene_last_compact_stage_count = 0
        self.context_hygiene_last_compact_ts = time.time()
        self.stage_tokens_in = 0
        self.stage_tokens_out = 0
        self.stage_tokens_cached = 0
        self.stage_llm_calls = 0
        self.estra_tokens_in = 0
        self.estra_tokens_out = 0
        self.estra_tokens_cached = 0
        self.estra_llm_calls = 0
        self.current_restored_from_stage = ""
        self.started_at = time.time()
        self.state_machine = LHRStateMachineStore(
            log_dir=self.log_dir,
            worker_id=self.worker_id or "W00",
            worker_index=self.worker_index,
            worker_count=self.worker_count,
            ledger_filename=self.ledger_filename,
        )
        self._resource_arbiter_agent: Any | None = None
        self._resource_admission_agent: Any | None = None
        self._metric_validity_feedback_llm: Any | None = None
        self._resource_main_agent_ref: Any | None = None
        self._live_agent: Any | None = None
        self._merge_owner_solver: LnrSolver | None = None
        self.gate_service = GateService.default()
        # Compatibility for agent hooks and older integrations.
        self.evaluation_service = self.gate_service
        self.evaluator_manager = self.evaluation_service.manager
        self.skill_registry: SkillRegistry | None = None
        self.skill_task_category = ""
        self.skill_allow_names: tuple[str, ...] = ()
        self.skill_tool_mode = str(
            getattr(self.lhr, "lnr_skill_tool_mode", "category_only") or "category_only"
        )
        self.skill_allow_generic_wildcard = bool(
            getattr(self.lhr, "lnr_skill_allow_generic_wildcard", False)
        )
        self.skill_visible_max = int(getattr(self.lhr, "lnr_skill_visible_max", 1) or 1)
        self.skill_category_status = "disabled"
        self._configure_lnr_category_skill()
        self.resource_observer = self._make_resource_observer()

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _skill_library_dir(self) -> Path:
        return default_skill_library_dir(self._repo_root())

    def _lnr_skill_category_source_path(self) -> Path:
        configured = str(
            getattr(self.lhr, "lnr_skill_category_source", "tasks/ml/mlebench/competition_categories.json") or ""
        ).strip()
        path = Path(configured or "tasks/ml/mlebench/competition_categories.json")
        if path.is_absolute():
            return path
        return self._repo_root() / path

    def _load_lnr_task_category_label(self) -> str:
        exp_id = str(getattr(self.cfg, "exp_id", "") or "").strip()
        if not exp_id:
            return ""
        path = self._lnr_skill_category_source_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        label = raw.get(exp_id) if isinstance(raw, dict) else ""
        return str(label or "").strip()

    def _configure_lnr_category_skill(self) -> None:
        self.skill_registry = None
        self.skill_task_category = ""
        self.skill_allow_names = ()
        self.skill_category_status = "disabled"
        if not bool(getattr(self.lhr, "lnr_skill_tool_enabled", False)):
            return
        mode = str(getattr(self.lhr, "lnr_skill_tool_mode", "category_only") or "category_only").strip()
        self.skill_tool_mode = mode or "category_only"
        if self.skill_tool_mode != "category_only":
            self.skill_category_status = "unsupported_mode"
            return
        self.skill_allow_generic_wildcard = bool(getattr(self.lhr, "lnr_skill_allow_generic_wildcard", False))
        self.skill_visible_max = max(1, int(getattr(self.lhr, "lnr_skill_visible_max", 1) or 1))
        label_field = str(getattr(self.lhr, "lnr_skill_category_label_field", "category_label") or "category_label").strip()
        if label_field != "category_label":
            self.skill_category_status = "unsupported_category_label_field"
            return
        label = self._load_lnr_task_category_label()
        self.skill_task_category = label
        if not label:
            self.skill_category_status = "skill_category_missing"
            return
        registry = getattr(self.orchestrator, "skill_registry", None)
        if registry is None:
            registry = SkillRegistry()
            try:
                registry.load_all(self._skill_library_dir())
            except Exception:
                self.skill_category_status = "skill_registry_load_failed"
                return
        candidates = registry.get_by_category_label(label)
        if not candidates:
            self.skill_category_status = "skill_category_unmapped"
            return
        selected = candidates[: self.skill_visible_max]
        selected_names: list[str] = []
        for skill in selected:
            name = skill.metadata.name
            if name and name not in selected_names:
                selected_names.append(name)
        exp_id = str(getattr(self.cfg, "exp_id", "") or "").strip()
        if exp_id:
            try:
                task_skill = registry.get_by_name_or_alias(exp_id)
            except Exception:
                task_skill = None
            if (
                task_skill is not None
                and task_skill.metadata.category == "tasks"
                and task_skill.metadata.name
                and task_skill.metadata.name not in selected_names
            ):
                selected_names.append(task_skill.metadata.name)
        self.skill_registry = registry
        self.skill_allow_names = tuple(selected_names)
        self.skill_visible_max = max(self.skill_visible_max, len(self.skill_allow_names))
        self.skill_category_status = "enabled" if self.skill_allow_names else "skill_category_unmapped"

    @staticmethod
    def _extract_lnr_auto_skill_hint(rendered: str) -> str:
        """Return a short auto-load hint section when a skill provides one."""
        lines = str(rendered or "").splitlines()
        start: int | None = None
        for idx, line in enumerate(lines):
            normalized = line.strip().lower().replace("_", "-")
            if normalized in {"## auto-load hint", "## auto-load summary", "## auto load hint"}:
                start = idx + 1
                break
        if start is None:
            return str(rendered or "").strip()
        collected: list[str] = []
        for line in lines[start:]:
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                break
            collected.append(line)
        return "\n".join(collected).strip()

    @staticmethod
    def _lnr_compact_skill_rendered(meta: Any, rendered: str, *, max_chars: int) -> str:
        text = str(rendered or "").strip()
        if not text:
            return ""
        if str(getattr(meta, "category", "") or "") != "categories":
            return text
        if len(text) <= max_chars:
            return text
        excerpt = text[: max(200, max_chars)].rstrip()
        return excerpt + "\n\n[category excerpt only; use `skill read` for the full category skill]"

    def _lnr_auto_read_skill_block(self) -> str:
        if not bool(getattr(self.lhr, "lnr_skill_auto_read", False)):
            return ""
        if self.skill_registry is None or not self.skill_allow_names:
            return ""
        max_chars = max(500, int(getattr(self.lhr, "lnr_skill_auto_read_max_chars", 6000) or 6000))
        category_max_chars = max(400, int(getattr(self.lhr, "lnr_skill_category_auto_read_max_chars", 800) or 800))
        chunks: list[str] = []
        remaining = max_chars
        skill_entries: list[Any] = []
        for name in self.skill_allow_names:
            try:
                skill = self.skill_registry.get_by_name_or_alias(name)
                if skill is None and hasattr(self.skill_registry, "get_by_name"):
                    skill = self.skill_registry.get_by_name(name)
            except Exception:
                skill = None
            if skill is None:
                continue
            skill_entries.append(skill)
        skill_entries.sort(
            key=lambda skill: 0
            if str(getattr(skill.metadata, "category", "") or "") == "tasks"
            else 1
        )
        for skill in skill_entries:
            meta = skill.metadata
            tags = ",".join(getattr(meta, "tags", []) or []) or "-"
            rendered = str(skill.render() or "").strip()
            auto_hint = self._extract_lnr_auto_skill_hint(rendered)
            compacted = self._lnr_compact_skill_rendered(
                meta,
                auto_hint,
                max_chars=category_max_chars,
            )
            if not compacted:
                continue
            if compacted != auto_hint:
                source = "category_excerpt"
            elif auto_hint != rendered:
                source = "auto_hint"
            else:
                source = "full_compact"
            block = (
                f"[skill:{meta.name}]\n"
                f"category={getattr(meta, 'category', '') or '-'} tags={tags} source={source}\n"
                f"{compacted}\n"
            )
            if len(block) > remaining:
                if remaining < 200:
                    break
                chunks.append(block[: remaining - 3].rstrip() + "...")
                break
            chunks.append(block)
            remaining -= len(block)
        if not chunks:
            return ""
        return (
            "\nAuto-loaded compact skill hints from enabled category/task skills "
            "because lnr_skill_auto_read=true. Use `skill read <name>` for full guidance:\n"
            + "\n".join(chunks).strip()
            + "\n"
        )
    def _lnr_skill_hint(self) -> str:
        if self.skill_registry is None or not self.skill_allow_names:
            return ""
        if any(name.startswith("task_") for name in self.skill_allow_names):
            base = "Category- and task-specific skills are available. Use `skill list` if you want guidance for this task."
        else:
            base = "A category-specific skill is available. Use `skill list` if you want guidance for this task type."
        return base + self._lnr_auto_read_skill_block()

    def _make_resource_arbiter_decider(self):
        if not bool(getattr(self.lhr, "resource_arbiter_enabled", False)):
            return None
        mode = str(getattr(self.lhr, "resource_arbiter_mode", "policy") or "policy").strip().lower()
        if mode != "llm":
            return None

        async def _decide(proposal: dict[str, Any]) -> dict[str, Any]:
            try:
                agent = self._resource_arbiter_agent
                if agent is None:
                    hook = self.orchestrator.make_llm_call_tracer(
                        node_id="resource_arbiter",
                        process_id=f"{getattr(self.cfg, 'exp_id', '') or 'lhr'}:resource_arbiter",
                        detail_prefix="mode=lnr;role=resource_arbiter;llm_role=feedback",
                    )
                    agent = self.orchestrator.create_science_agent(
                        task_description=None,
                        run_policy=_WallClockAutoContinuePolicy(deadline_monotonic=self.deadline, max_text_only_retries=0),
                        system_prompt=(
                            "You are Resource Arbiter. You are isolated from the main science agent. "
                            "Return JSON only. Do not call tools. Decide resource kill proposals using evidence. "
                            "Recent metric, checkpoint, submission, or training-progress stdout supports DENY_KILL; CPU busy, log churn, or artifact mtime alone does not, especially when declared GPU work is observed CPU-only while assigned GPU is idle."
                        ),
                        append_repl_system_prompt=False,
                        pin_task_description=False,
                        max_steps_override=1,
                        on_llm_call=hook,
                        memory_dir_override=self.global_log_dir / "resource" / "arbiter_memory",
                        load_existing_memory=False,
                        memory_agent_name="ResourceArbiter",
                        teleport_mode="off",
                        repl_bash_write_mode=False,
                        stable_system_prompt=True,
                        pin_environment_context=False,
                        workspace_git_enabled=False,
                        workspace_git_auto_checkpoint=False,
                        bash_observation_summary_override=True,
                        llm_stage_override=(
                            self._worker_llm_stage_override("feedback")
                            or self.cfg.agent.feedback
                        ),
                    )
                    self._resource_arbiter_agent = agent
                prompt = build_resource_arbiter_prompt(proposal)
                text = await agent.run_ephemeral_agentic_route_prompt(
                    prompt,
                    trigger="resource_arbiter",
                    base_messages=[],
                )
                parsed = parse_arbiter_decision_text(text)
                if not parsed:
                    parsed = fallback_policy_decision(proposal)
                    parsed["reason"] = "arbiter LLM returned unparsable output; used conservative fallback"
                return normalize_arbiter_decision(parsed, proposal=proposal, source="resource_arbiter_llm")
            except Exception as exc:
                fallback = fallback_policy_decision(proposal)
                fallback["action"] = "OBSERVE_MORE"
                fallback["reason"] = f"arbiter LLM failed: {type(exc).__name__}"
                fallback["confidence"] = "low"
                return normalize_arbiter_decision(fallback, proposal=proposal, source="llm_error_fallback")

        return _decide

    @staticmethod
    def _resource_advisory_prompt(proposal: dict[str, Any]) -> str:
        return build_inline_resource_advisory_prompt(proposal)

    @staticmethod
    def _drop_dangling_tool_call_tail(messages: list[Any]) -> list[Any]:
        out = list(messages or [])
        if not out:
            return out
        tail = out[-1]
        role = str(getattr(tail, "role", "") or "").lower()
        tool_calls = getattr(tail, "tool_calls", None) or []
        if role == "assistant" and tool_calls:
            return out[:-1]
        return out

    def _make_resource_main_agent_advisory_decider(self):
        if not bool(getattr(self.lhr, "resource_main_agent_advisory_enabled", False)):
            return None

        async def _decide(proposal: dict[str, Any]) -> dict[str, Any]:
            agent = self._resource_main_agent_ref
            mode = str(getattr(self.lhr, "resource_advisory_mode", INLINE_RESOURCE_ADVISORY_MODE) or INLINE_RESOURCE_ADVISORY_MODE)
            if mode != INLINE_RESOURCE_ADVISORY_MODE:
                return {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": f"unsupported_resource_advisory_mode:{mode}",
                    "ttl_sec": 300,
                    "advisory_mode": mode,
                }
            if agent is None or getattr(agent, "llm", None) is None:
                return {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": "main_agent_context_unavailable",
                    "ttl_sec": 300,
                    "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                }
            safe, unsafe_reason = safe_inline_resource_advisory_boundary(agent)
            prompt = self._resource_advisory_prompt(proposal)
            if hasattr(getattr(agent, "llm", None), "ask_tool_stream"):
                t0 = time.time()
                blocked_state = not safe
                try:
                    base_messages = list(getattr(agent, "_resource_advisory_base_messages", []) or [])
                    if not base_messages and hasattr(agent, "_memory_ctx"):
                        base_messages = self._drop_dangling_tool_call_tail(
                            agent._memory_ctx.build_messages_for_llm()
                        )
                    else:
                        base_messages = self._drop_dangling_tool_call_tail(base_messages)
                    messages = list(base_messages)
                    messages.append(Message.user_message(prompt))
                    system_msgs = list(getattr(agent, "_resource_advisory_base_system_msgs", []) or [])
                    if not system_msgs and hasattr(agent, "_build_system_messages"):
                        system_msgs = list(agent._build_system_messages())
                    system_msgs.append(Message.system_message(
                        "You are answering a resource advisory for your own current command. "
                        "Return exactly one RESOURCE_ADVISORY_RESPONSE block. "
                        "Do not call tools, write files, or continue experiments."
                    ))
                    assistant_msg = await self._ask_agent_tool_stream_guarded(
                        agent,
                        messages=messages,
                        system_msgs=system_msgs,
                        timeout=float(getattr(self.lhr, "resource_main_agent_advisory_timeout_sec", 60.0) or 60.0),
                        tools=[],
                        tool_choice="none",
                        parallel_tool_calls=False,
                        collect_all_tool_calls=True,
                    )
                    text = (getattr(assistant_msg, "content", None) or "").strip()
                    if not text:
                        text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
                    advisory, block_text, parse_reason = parse_inline_resource_advisory_response(text)
                    if parse_reason:
                        advisory = normalize_inline_resource_advisory({
                            "preference": "unknown",
                            "confidence": "low",
                            "reason": parse_reason,
                        })
                    advisory.update({
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "advisory_status": "captured" if not parse_reason else "captured_with_parse_issue",
                        "blocked_state": blocked_state,
                        "memory_edit_applied": True,
                        "memory_edit_removed_tokens": 0,
                        "advisory_tool_call_rejected": False,
                    })
                    advisory["_audit"] = {
                        "event_version": 1,
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "status": advisory["advisory_status"],
                        "blocked_state": blocked_state,
                        "defer_reason": unsafe_reason if blocked_state else "cache_friendly_direct_advisory",
                        "proposal_id": proposal.get("proposal_id"),
                        "proposal_type": proposal.get("proposal_type"),
                        "reason_code": proposal.get("reason_code"),
                        "request": prompt[:6000],
                        "raw_response": str(text or "")[:6000],
                        "parsed_response": {k: v for k, v in advisory.items() if not str(k).startswith("_")},
                        "block_text": block_text[:2000],
                        "parse_reason": parse_reason,
                        "tool_call_rejected": False,
                        "memory_edit_applied": True,
                        "memory_edit_removed_tokens": 0,
                        "final_feedback_emitted": False,
                        "duration_sec": round(time.time() - t0, 3),
                    }
                    if hasattr(agent, "_record_llm_call"):
                        agent._record_llm_call(
                            "resource_advisory",
                            time.time() - t0,
                            None,
                            "ok",
                            recovery=True,
                            turn_kind="resource_advisory",
                        )
                    return advisory
                except Exception as exc:
                    if hasattr(agent, "_record_llm_call"):
                        try:
                            agent._record_llm_call(
                                "resource_advisory",
                                time.time() - t0,
                                None,
                                "error",
                                recovery=True,
                                turn_kind="resource_advisory",
                            )
                        except Exception:
                            pass
                    return {
                        "preference": "unknown",
                        "confidence": "low",
                        "reason": f"main_agent_resource_advisory_error:{type(exc).__name__}",
                        "ttl_sec": 120,
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "advisory_status": "error",
                        "blocked_state": blocked_state,
                        "_audit": {
                            "event_version": 1,
                            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                            "status": "error",
                            "blocked_state": blocked_state,
                            "defer_reason": unsafe_reason if blocked_state else "cache_friendly_direct_advisory",
                            "proposal_id": proposal.get("proposal_id"),
                            "proposal_type": proposal.get("proposal_type"),
                            "request": prompt[:6000],
                            "error": type(exc).__name__,
                            "duration_sec": round(time.time() - t0, 3),
                        },
                    }
            if not safe:
                return {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": f"inline_resource_advisory_deferred:{unsafe_reason}",
                    "ttl_sec": 120,
                    "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                    "advisory_status": "deferred",
                    "_audit": {
                        "event_version": 1,
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "status": "deferred",
                        "defer_reason": unsafe_reason,
                        "proposal_id": proposal.get("proposal_id"),
                        "proposal_type": proposal.get("proposal_type"),
                    },
                }

            t0 = time.time()
            previous_callback = getattr(agent, "_lnr_text_only_callback", None)
            result_box: dict[str, Any] = {}

            async def _inline_resource_advisory_callback(
                *,
                agent: Any,
                assistant_text: str,
                round_idx: int,
                max_steps: int,
            ) -> str:
                _ = round_idx, max_steps
                tool_rejected = bool(getattr(agent, "_lnr_resource_advisory_tool_call_rejected", False))
                if tool_rejected:
                    advisory = normalize_inline_resource_advisory(
                        {
                            "preference": "unknown",
                            "confidence": "low",
                            "reason": "advisory_tool_call_rejected",
                        }
                    )
                    block_text = ""
                    parse_reason = "tool_call_rejected"
                else:
                    advisory, block_text, parse_reason = parse_inline_resource_advisory_response(assistant_text)
                    if parse_reason:
                        advisory = normalize_inline_resource_advisory(
                            {
                                "preference": "unknown",
                                "confidence": "low",
                                "reason": parse_reason,
                            }
                        )
                advisory.update(
                    {
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "memory_edit_applied": True,
                        "memory_edit_removed_tokens": 0,
                        "advisory_tool_call_rejected": tool_rejected,
                    }
                )
                advisory["_audit"] = {
                    "event_version": 1,
                    "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                    "status": "captured" if not parse_reason else "captured_with_parse_issue",
                    "proposal_id": proposal.get("proposal_id"),
                    "proposal_type": proposal.get("proposal_type"),
                    "reason_code": proposal.get("reason_code"),
                    "request": prompt[:6000],
                    "raw_response": str(assistant_text or "")[:6000],
                    "parsed_response": {k: v for k, v in advisory.items() if not str(k).startswith("_")},
                    "block_text": block_text[:2000],
                    "parse_reason": parse_reason,
                    "tool_call_rejected": tool_rejected,
                    "memory_edit_applied": True,
                    "memory_edit_removed_tokens": 0,
                    "final_feedback_emitted": False,
                }
                result_box.clear()
                result_box.update(advisory)
                setattr(agent, "_lnr_suppress_current_text_only_memory", "resource_advisory_memory_edit")
                setattr(agent, "_lnr_resource_advisory_text_handled", True)
                return "RESOURCE_ADVISORY_CAPTURED"

            try:
                setattr(agent, "_lnr_text_only_callback", _inline_resource_advisory_callback)
                setattr(agent, "_lnr_transient_user_prompt", prompt)
                setattr(agent, "_lnr_transient_tool_choice_none", False)
                setattr(agent, "_lnr_transient_turn_kind", "inline_resource_advisory")
                setattr(agent, "_lnr_resource_advisory_text_pending", True)
                setattr(agent, "_lnr_resource_advisory_text_handled", False)
                setattr(agent, "_lnr_resource_advisory_tool_call_rejected", False)
                await agent.run(None)
                if result_box:
                    return dict(result_box)
                return {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": "inline_resource_advisory_empty_response",
                    "ttl_sec": 300,
                    "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                    "_audit": {
                        "event_version": 1,
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "status": "empty_response",
                        "proposal_id": proposal.get("proposal_id"),
                        "proposal_type": proposal.get("proposal_type"),
                        "request": prompt[:6000],
                        "memory_edit_applied": True,
                    },
                }
            except Exception as exc:
                return {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": f"main_agent_advisory_error:{type(exc).__name__}",
                    "ttl_sec": 300,
                    "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                    "_audit": {
                        "event_version": 1,
                        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                        "status": "error",
                        "proposal_id": proposal.get("proposal_id"),
                        "proposal_type": proposal.get("proposal_type"),
                        "request": prompt[:6000],
                        "error": type(exc).__name__,
                        "duration_sec": round(time.time() - t0, 3),
                    },
                }
            finally:
                try:
                    if previous_callback is None:
                        if hasattr(agent, "_lnr_text_only_callback"):
                            delattr(agent, "_lnr_text_only_callback")
                    else:
                        setattr(agent, "_lnr_text_only_callback", previous_callback)
                except Exception:
                    pass
                for name, value in (
                    ("_lnr_transient_user_prompt", ""),
                    ("_lnr_transient_tool_choice_none", False),
                    ("_lnr_transient_user_prompt_active", False),
                    ("_lnr_transient_turn_kind", ""),
                    ("_lnr_resource_advisory_text_pending", False),
                    ("_lnr_resource_advisory_text_handled", False),
                    ("_lnr_resource_advisory_tool_call_rejected", False),
                    ("_lnr_suppress_current_text_only_memory", ""),
                ):
                    try:
                        setattr(agent, name, value)
                    except Exception:
                        pass

        return _decide

    def _make_resource_admission_decider(self):
        if not bool(getattr(self.lhr, "resource_admission_llm_enabled", False)):
            return None

        async def _decide(task_card: dict[str, Any], rule_result: dict[str, Any]) -> str:
            agent = self._resource_admission_agent
            if agent is None:
                hook = self.orchestrator.make_llm_call_tracer(
                    node_id="resource_admission_arbiter",
                    process_id=f"{getattr(self.cfg, 'exp_id', '') or 'lhr'}:resource_admission_arbiter",
                    detail_prefix="mode=lnr;role=resource_admission_arbiter;llm_role=feedback",
                )
                agent = self.orchestrator.create_science_agent(
                    task_description=None,
                    run_policy=_WallClockAutoContinuePolicy(deadline_monotonic=self.deadline, max_text_only_retries=0),
                    system_prompt=(
                        "You are ResourceAdmissionArbiter. You are isolated from the main science agent. "
                        "Return JSON only. Do not call tools. Decide startup admission for GPU tasks using only "
                        "the provided ResourceTaskCard and rule result. You may approve RUN_NOW or OBSERVE_THEN_RUN "
                        "only when lease_grantable_by_llm is true; the runtime will then atomically acquire the lease."
                    ),
                    append_repl_system_prompt=False,
                    pin_task_description=False,
                    max_steps_override=1,
                    on_llm_call=hook,
                    memory_dir_override=self.global_log_dir / "resource" / "admission_memory",
                    load_existing_memory=False,
                    memory_agent_name="ResourceAdmissionArbiter",
                    teleport_mode="off",
                    repl_bash_write_mode=False,
                    stable_system_prompt=True,
                    pin_environment_context=False,
                    workspace_git_enabled=False,
                    workspace_git_auto_checkpoint=False,
                    bash_observation_summary_override=True,
                    llm_stage_override=(
                        self._worker_llm_stage_override("feedback")
                        or self.cfg.agent.feedback
                    ),
                )
                self._resource_admission_agent = agent
            prompt = build_resource_admission_prompt(task_card, rule_result)
            return await agent.run_ephemeral_agentic_route_prompt(
                prompt,
                trigger="resource_admission_arbiter",
                base_messages=[],
            )

        return _decide

    def _make_resource_observer(self) -> LHRResourceObserver | None:
        if not bool(getattr(self.lhr, "resource_monitor_enabled", True)):
            return None
        arbiter_enabled_cfg = bool(getattr(self.lhr, "resource_arbiter_enabled", False))
        gpu_share_enabled_cfg = bool(getattr(self.lhr, "resource_gpu_share_enabled", False))
        gpu_share_phase_cfg = str(getattr(self.lhr, "resource_gpu_share_phase", "observe") or "observe")
        contention_enabled_cfg = bool(getattr(self.lhr, "resource_arbiter_contention_review_enabled", False))
        if arbiter_enabled_cfg and gpu_share_enabled_cfg and gpu_share_phase_cfg in {"tt_share", "feature_share", "light_train"}:
            contention_enabled_cfg = True
        contention_min_runtime_sec_cfg = float(getattr(self.lhr, "resource_arbiter_contention_min_runtime_sec", 900.0) or 0.0)
        contention_min_waiter_age_sec_cfg = float(getattr(self.lhr, "resource_arbiter_contention_min_waiter_age_sec", 300.0) or 0.0)
        contention_min_interval_sec_cfg = float(getattr(self.lhr, "resource_arbiter_contention_min_interval_sec", 600.0) or 0.0)
        if contention_enabled_cfg and gpu_share_enabled_cfg:
            contention_min_runtime_sec_cfg = min(contention_min_runtime_sec_cfg, 300.0)
            contention_min_waiter_age_sec_cfg = min(contention_min_waiter_age_sec_cfg, 30.0)
            contention_min_interval_sec_cfg = min(contention_min_interval_sec_cfg, 120.0)
        return LHRResourceObserver(
            state_machine=self.state_machine,
            worker_id=self.worker_id or "W00",
            resource_control_profile=str(getattr(self.lhr, "resource_control_profile", "normal") or "normal"),
            min_register_sec=float(getattr(self.lhr, "resource_monitor_min_register_sec", 600.0) or 0.0),
            check_interval_sec=float(getattr(self.lhr, "resource_monitor_check_interval_sec", 10.0) or 10.0),
            stalled_stdout_sec=float(getattr(self.lhr, "resource_monitor_stalled_stdout_sec", 900.0) or 0.0),
            kill_enabled=bool(getattr(self.lhr, "resource_monitor_kill_enabled", True)),
            kill_mode=str(getattr(self.lhr, "resource_monitor_kill_mode", "recommend") or "recommend"),
            monitor_agent_mode=str(getattr(self.lhr, "resource_monitor_agent_mode", "off") or "off"),
            monitor_agent_min_interval_sec=float(getattr(self.lhr, "resource_monitor_agent_min_interval_sec", 60.0) or 60.0),
            low_progress_enabled=bool(getattr(self.lhr, "resource_monitor_low_progress_enabled", True)),
            low_progress_warmup_sec=float(getattr(self.lhr, "resource_monitor_low_progress_warmup_sec", 1800.0) or 0.0),
            low_progress_no_heartbeat_sec=float(getattr(self.lhr, "resource_monitor_low_progress_no_heartbeat_sec", 1800.0) or 0.0),
            low_progress_no_artifact_sec=float(getattr(self.lhr, "resource_monitor_low_progress_no_artifact_sec", 1800.0) or 0.0),
            review_state_enabled=bool(getattr(self.lhr, "resource_review_state_enabled", True)),
            review_heartbeat_sec=float(getattr(self.lhr, "resource_review_heartbeat_sec", 60.0) or 60.0),
            review_warmup_windows=int(getattr(self.lhr, "resource_review_warmup_windows", 10) or 0),
            review_inactive_windows=int(getattr(self.lhr, "resource_review_inactive_windows", 3) or 1),
            review_value_windows=int(getattr(self.lhr, "resource_review_value_windows", 5) or 1),
            review_progress_event_min_windows=int(getattr(self.lhr, "resource_review_progress_event_min_windows", 5) or 1),
            review_timebox_windows=int(getattr(self.lhr, "resource_review_timebox_windows", 10) or 1),
            review_max_proof_windows=int(getattr(self.lhr, "resource_review_max_proof_windows", 2) or 2),
            review_min_timebox_sec=float(getattr(self.lhr, "resource_review_min_timebox_sec", 60.0) or 60.0),
            review_max_timebox_sec=float(getattr(self.lhr, "resource_review_max_timebox_sec", 1800.0) or 1800.0),
            review_timebox_budget_fraction=float(getattr(self.lhr, "resource_review_timebox_budget_fraction", 0.10) or 0.10),
            bash_monitor_all_enabled=bool(getattr(self.lhr, "resource_bash_monitor_all_enabled", True)),
            arbiter_min_progress_windows=int(getattr(self.lhr, "resource_arbiter_min_progress_windows", 2) or 2),
            arbiter_kill_requires_high_confidence=bool(getattr(self.lhr, "resource_arbiter_kill_requires_high_confidence", True)),
            task_resource_dir=self.global_log_dir / "resource",
            resource_runtime_enabled=bool(getattr(self.lhr, "resource_runtime_enabled", True)),
            gpu_queue_enabled=bool(getattr(self.lhr, "resource_gpu_queue_enabled", True)),
            gpu_pool=[str(x) for x in (getattr(self.lhr, "resource_gpu_pool", []) or []) if str(x).strip()],
            gpu_default_request=int(getattr(self.lhr, "resource_gpu_default_request", 1) or 1),
            gpu_max_request=int(getattr(self.lhr, "resource_gpu_max_request", 1) or 1),
            gpu_assignment=str(getattr(self.lhr, "resource_gpu_assignment", "lease") or "lease"),
            gpu_queue_max_wait_sec=float(getattr(self.lhr, "resource_gpu_queue_max_wait_sec", 1800.0) or 0.0),
            gpu_queue_heartbeat_sec=float(getattr(self.lhr, "resource_gpu_queue_heartbeat_sec", 15.0) or 15.0),
            gpu_max_heavy_per_gpu=int(getattr(self.lhr, "resource_gpu_max_heavy_per_gpu", 1) or 1),
            gpu_capacity_slots=float(getattr(self.lhr, "resource_gpu_capacity_slots", 1.0) or 1.0),
            gpu_tt_max_per_gpu=int(getattr(self.lhr, "resource_gpu_tt_max_per_gpu", 3) or 3),
            gpu_feature_max_per_gpu=int(getattr(self.lhr, "resource_gpu_feature_max_per_gpu", 2) or 2),
            gpu_share_tt_with_train=bool(getattr(self.lhr, "resource_gpu_share_tt_with_train", False)),
            gpu_share_enabled=gpu_share_enabled_cfg,
            gpu_share_phase=gpu_share_phase_cfg,
            gpu_share_policy_profile=str(getattr(self.lhr, "resource_gpu_share_policy_profile", "conservative") or "conservative"),
            gpu_share_memory_profile=str(getattr(self.lhr, "resource_gpu_share_memory_profile", "conservative") or "conservative"),
            gpu_share_cpu_policy=str(getattr(self.lhr, "resource_gpu_share_cpu_policy", "conservative") or "conservative"),
            gpu_trial_admission_policy=str(getattr(self.lhr, "resource_gpu_trial_admission_policy", "llm_grant") or "llm_grant"),
            resource_startup_policy=str(getattr(self.lhr, "resource_startup_policy", "trial_first") or "trial_first"),
            resource_trial_window_sec=float(getattr(self.lhr, "resource_trial_window_sec", 300.0) or 300.0),
            resource_trial_hard_review_sec=float(getattr(self.lhr, "resource_trial_hard_review_sec", 900.0) or 900.0),
            gpu_lease_ttl_sec=float(getattr(self.lhr, "resource_gpu_lease_ttl_sec", 7200.0) or 7200.0),
            gpu_duplicate_digest_cooldown_sec=float(getattr(self.lhr, "resource_gpu_duplicate_digest_cooldown_sec", 600.0) or 600.0),
            gpu_duplicate_digest_threshold=int(getattr(self.lhr, "resource_gpu_duplicate_digest_threshold", 2) or 2),
            gpu_admission_queue_enabled=bool(getattr(self.lhr, "resource_gpu_admission_queue_enabled", True)),
            gpu_admission_waiter_ttl_sec=float(getattr(self.lhr, "resource_gpu_admission_waiter_ttl_sec", 900.0) or 900.0),
            admission_llm_enabled=bool(getattr(self.lhr, "resource_admission_llm_enabled", False)),
            admission_llm_mode=str(getattr(self.lhr, "resource_admission_llm_mode", "low_confidence") or "low_confidence"),
            admission_llm_timeout_sec=float(getattr(self.lhr, "resource_admission_llm_timeout_sec", 60.0) or 60.0),
            admission_decider=self._make_resource_admission_decider(),
            stale_pressure_observe_first_enabled=bool(getattr(self.lhr, "resource_stale_pressure_observe_first_enabled", True)),
            stale_pressure_observe_window_sec=float(getattr(self.lhr, "resource_stale_pressure_observe_window_sec", 180.0) or 180.0),
            stale_pressure_observe_max_sec=float(getattr(self.lhr, "resource_stale_pressure_observe_max_sec", 300.0) or 300.0),
            stale_pressure_observe_min_free_mem_gb=float(getattr(self.lhr, "resource_stale_pressure_observe_min_free_mem_gb", 8.0) or 0.0),
            stale_pressure_healthy_skip_llm=bool(getattr(self.lhr, "resource_stale_pressure_healthy_skip_llm", True)),
            gpu_pressure_yellow_hold_sec=float(getattr(self.lhr, "resource_gpu_pressure_yellow_hold_sec", 120.0) or 120.0),
            gpu_pressure_red_to_yellow_sec=float(getattr(self.lhr, "resource_gpu_pressure_red_to_yellow_sec", 120.0) or 120.0),
            gpu_pressure_yellow_util_pct=float(getattr(self.lhr, "resource_gpu_pressure_yellow_util_pct", 85.0) or 85.0),
            gpu_pressure_min_free_mem_gb=float(getattr(self.lhr, "resource_gpu_pressure_min_free_mem_gb", 8.0) or 0.0),
            gpu_pressure_yellow_free_mem_buffer_gb=float(getattr(self.lhr, "resource_gpu_pressure_yellow_free_mem_buffer_gb", 8.0) or 8.0),
            observation_enabled=bool(getattr(self.lhr, "resource_observation_enabled", True)),
            observation_window_sec=float(getattr(self.lhr, "resource_observation_window_sec", 30.0) or 30.0),
            observation_shadow_workspace_enabled=bool(getattr(self.lhr, "resource_observation_shadow_workspace_enabled", True)),
            gpu_source_hint_enabled=bool(getattr(self.lhr, "resource_gpu_source_hint_enabled", True)),
            gpu_source_hint_mode=str(getattr(self.lhr, "resource_gpu_source_hint_mode", "observe") or "observe"),
            gpu_util_observer_enabled=bool(getattr(self.lhr, "resource_gpu_util_observer_enabled", True)),
            gpu_util_sample_interval_sec=float(getattr(self.lhr, "resource_gpu_util_sample_interval_sec", 30.0) or 30.0),
            gpu_idle_lease_guard_enabled=bool(getattr(self.lhr, "resource_gpu_idle_lease_guard_enabled", True)),
            gpu_idle_lease_warmup_sec=float(getattr(self.lhr, "resource_gpu_idle_lease_warmup_sec", 180.0) or 0.0),
            gpu_idle_lease_min_samples=int(getattr(self.lhr, "resource_gpu_idle_lease_min_samples", 3) or 0),
            gpu_idle_lease_util_pct=float(getattr(self.lhr, "resource_gpu_idle_lease_util_pct", 1.0) or 0.0),
            gpu_idle_lease_mem_gb=float(getattr(self.lhr, "resource_gpu_idle_lease_mem_gb", 1.0) or 0.0),
            gpu_idle_lease_require_pressure=bool(getattr(self.lhr, "resource_gpu_idle_lease_require_pressure", False)),
            gpu_idle_lease_action_mode=str(getattr(self.lhr, "resource_gpu_idle_lease_action_mode", "release") or "release"),
            resource_idle_release_admission_mode=str(getattr(self.lhr, "resource_idle_release_admission_mode", "strict_exclusive") or "strict_exclusive"),
            quick_probe_guard_enabled=bool(getattr(self.lhr, "resource_quick_probe_guard_enabled", True)),
            quick_probe_expected_runtime_sec=float(getattr(self.lhr, "resource_quick_probe_expected_runtime_sec", 300.0) or 300.0),
            quick_probe_hard_review_sec=float(getattr(self.lhr, "resource_quick_probe_hard_review_sec", 900.0) or 900.0),
            quick_probe_small_scope_threshold=int(getattr(self.lhr, "resource_quick_probe_small_scope_threshold", 1000) or 1000),
            gpu_dataloader_bottleneck_guard_enabled=bool(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_guard_enabled", True)),
            gpu_dataloader_bottleneck_warmup_sec=float(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_warmup_sec", 600.0) or 0.0),
            gpu_dataloader_bottleneck_min_samples=int(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_min_samples", 3) or 1),
            gpu_dataloader_bottleneck_util_pct=float(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_util_pct", 15.0) or 0.0),
            gpu_dataloader_bottleneck_min_mem_gb=float(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_min_mem_gb", 2.0) or 0.0),
            gpu_dataloader_bottleneck_child_cpu_pct=float(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_child_cpu_pct", 200.0) or 0.0),
            gpu_dataloader_bottleneck_busy_children=int(getattr(self.lhr, "resource_gpu_dataloader_bottleneck_busy_children", 2) or 1),
            deliverable_completion_guard_enabled=bool(getattr(self.lhr, "resource_deliverable_completion_guard_enabled", True)),
            deliverable_completion_warmup_sec=float(getattr(self.lhr, "resource_deliverable_completion_warmup_sec", 120.0) or 0.0),
            deliverable_completion_settle_sec=float(getattr(self.lhr, "resource_deliverable_completion_settle_sec", 120.0) or 0.0),
            deliverable_completion_scan_interval_sec=float(getattr(self.lhr, "resource_deliverable_completion_scan_interval_sec", 60.0) or 0.0),
            deliverable_completion_quiet_sec=float(getattr(self.lhr, "resource_deliverable_completion_quiet_sec", 60.0) or 0.0),
            metric_health_guard_enabled=bool(getattr(self.lhr, "resource_metric_health_guard_enabled", True)),
            metric_health_warmup_sec=float(getattr(self.lhr, "resource_metric_health_warmup_sec", 600.0) or 0.0),
            metric_health_invalid_min_events=int(getattr(self.lhr, "resource_metric_health_invalid_min_events", 2) or 1),
            metric_health_zero_score_min_events=int(getattr(self.lhr, "resource_metric_health_zero_score_min_events", 2) or 1),
            arbiter_enabled=arbiter_enabled_cfg,
            arbiter_mode=str(getattr(self.lhr, "resource_arbiter_mode", "policy") or "policy"),
            arbiter_timeout_sec=float(getattr(self.lhr, "resource_arbiter_timeout_sec", 90.0) or 90.0),
            arbiter_decider=self._make_resource_arbiter_decider(),
            main_agent_advisory_enabled=bool(getattr(self.lhr, "resource_main_agent_advisory_enabled", False)),
            main_agent_advisory_min_interval_sec=float(getattr(self.lhr, "resource_main_agent_advisory_min_interval_sec", 600.0) or 600.0),
            main_agent_advisory_timeout_sec=float(getattr(self.lhr, "resource_main_agent_advisory_timeout_sec", 60.0) or 60.0),
            main_agent_advisory_decider=self._make_resource_main_agent_advisory_decider(),
            arbiter_contention_review_enabled=contention_enabled_cfg,
            arbiter_contention_min_runtime_sec=contention_min_runtime_sec_cfg,
            arbiter_contention_min_waiter_age_sec=contention_min_waiter_age_sec_cfg,
            arbiter_contention_min_interval_sec=contention_min_interval_sec_cfg,
            research_cadence_enabled=bool(getattr(self.lhr, "resource_research_cadence_enabled", True)),
            research_cadence_observe_sec=float(getattr(self.lhr, "resource_research_cadence_observe_sec", 300.0) or 300.0),
            first_comparable_metric_budget_sec=float(getattr(self.lhr, "resource_first_comparable_metric_budget_sec", 900.0) or 900.0),
            proven_route_metric_budget_sec=float(getattr(self.lhr, "resource_proven_route_metric_budget_sec", 1800.0) or 1800.0),
            arbiter_periodic_review_enabled=bool(getattr(self.lhr, "resource_arbiter_periodic_review_enabled", False)),
            arbiter_periodic_min_runtime_sec=float(getattr(self.lhr, "resource_arbiter_periodic_min_runtime_sec", 1800.0) or 0.0),
            arbiter_periodic_min_interval_sec=float(getattr(self.lhr, "resource_arbiter_periodic_min_interval_sec", 900.0) or 900.0),
            arbiter_proposal_coalesce_window_sec=float(getattr(self.lhr, "resource_arbiter_proposal_coalesce_window_sec", 60.0) or 0.0),
            arbiter_job_llm_call_cap=int(getattr(self.lhr, "resource_arbiter_job_llm_call_cap", 6) or 0),
            arbiter_job_advisory_call_cap=int(getattr(self.lhr, "resource_arbiter_job_advisory_call_cap", 3) or 0),
            arbiter_job_token_cap=int(getattr(self.lhr, "resource_arbiter_job_token_cap", 24000) or 0),
            sidecar_enabled=bool(getattr(self.lhr, "resource_sidecar_enabled", False)),
            sidecar_min_parent_runtime_sec=float(getattr(self.lhr, "resource_sidecar_min_parent_runtime_sec", 900.0) or 900.0),
            estra_magent_enabled=bool(getattr(self.lhr, "estra_magent_enabled", False)),
            estra_magent_sidecar_enabled=bool(getattr(self.lhr, "estra_magent_sidecar_enabled", False)),
            estra_magent_min_parent_runtime_sec=float(getattr(self.lhr, "estra_magent_min_parent_runtime_sec", 300.0) or 300.0),
            estra_magent_sidecar_mode=str(getattr(self.lhr, "estra_magent_sidecar_mode", "cpu_only") or "cpu_only"),
            estra_magent_budget_sec=float(getattr(self.lhr, "estra_magent_budget_sec", 900.0) or 900.0),
            estra_magent_join_inject_parent=bool(getattr(self.lhr, "estra_magent_join_inject_parent", True)),
            estra_magent_join_inject_estra=bool(getattr(self.lhr, "estra_magent_join_inject_estra", True)),
            estra_magent_join_inject_resource_context=bool(getattr(self.lhr, "estra_magent_join_inject_resource_context", True)),
            checkpoint_submission_guard_enabled=bool(getattr(self.lhr, "resource_checkpoint_submission_guard_enabled", True)),
            timeout_hard_gate_enabled=bool(getattr(self.lhr, "resource_queue_timeout_hard_gate_enabled", True)),
            timeout_block_train_after=int(getattr(self.lhr, "resource_queue_timeout_block_train_after", 1) or 1),
            timeout_tt_only_after=int(getattr(self.lhr, "resource_queue_timeout_tt_only_after", 2) or 2),
        )

    @staticmethod
    def _safe_ledger_filename(raw: str) -> str:
        name = str(raw or ".run_results.md").strip().replace("\\", "/")
        if not name or "/" in name or name in {".", ".."}:
            return ".run_results.md"
        if name.startswith(".") and name != ".run_results.md":
            return ".run_results.md"
        return name

    def _jsonl_roots(self) -> tuple[Path, ...]:
        # LHR control-plane logs stay outside the agent workspace. The workspace
        # .logs/ tree is reserved for interaction/traj logs and run evidence.
        return (self.log_dir,)

    def _jsonl(self, name: str, record: dict[str, Any]) -> None:
        record = {"timestamp": time.time(), **record}
        if name not in LHR_UNIFIED_ONLY_JSONL:
            for root in self._jsonl_roots():
                try:
                    root.mkdir(parents=True, exist_ok=True)
                    with (root / name).open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                except OSError:
                    logger.debug("[lnr] could not append %s", name, exc_info=True)
        self._mirror_state_event(name=name, record=record)

    @staticmethod
    def _state_task_type_for_event(event: str) -> str:
        if event.startswith("stage_"):
            return "stage_commit"
        if event.endswith("_estra_check") or event in {
            "estra_decision",
            "estra_invalid",
            "estra_llm_error",
            "estra_main_context_invalid_fallback",
            "estra_fallback_llm_error",
            "estra_deterministic_fallback",
            "estra_no_valid_switch_candidate",
            "text_only_estra_noop",
        }:
            return "estra_decision"
        if event in {"estra_keep_current_compacted", "estra_stage_switched", "estra_failed", "state_packet_built"}:
            return "estra_restore"
        if event.startswith("worker_") or event.startswith("multi_worker"):
            return "coordinator"
        return "repl_search"

    def _relativize_control_payload(self, value: Any) -> Any:
        if isinstance(value, str):
            text = value
            replacements = [
                (str(self.workspace_dir), "workspace"),
                (str(self.root_dir), "."),
                (str(self.task_root_dir), "."),
            ]
            replacements.sort(key=lambda item: len(item[0]), reverse=True)
            for raw, repl in replacements:
                if raw and raw in text:
                    text = text.replace(raw, repl)
            return text
        if isinstance(value, dict):
            return {str(k): self._relativize_control_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._relativize_control_payload(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._relativize_control_payload(v) for v in value)
        return value

    def _mirror_state_event(self, *, name: str, record: dict[str, Any]) -> None:
        if name == LHR_EVENTS_JSONL:
            return
        event = str(record.get("event") or Path(name).stem or "event")
        task_type = self._state_task_type_for_event(event)
        if event.endswith("_estra_check") or event.startswith("estra_"):
            stage_id = str(record.get("latest_stage") or record.get("stage_id") or record.get("target_stage") or "")
        else:
            stage_id = str(record.get("stage_id") or record.get("target_stage") or "")
        task_id = f"{task_type}:{self.worker_id or 'W00'}"
        if stage_id:
            task_id = f"{task_id}:{stage_id}"
        status = ""
        if event == "context_compact_started":
            status = "running"
        elif event == "context_compact_estra_check":
            status = "running"
        elif event == "context_compact_deferred":
            status = "succeeded"
        elif event == "context_compact_finished":
            compact_status = str(record.get("status") or "")
            status = "failed" if compact_status == "failed" else "succeeded"
        elif event == "context_compact_failed":
            status = "failed"
        elif event.endswith("_failed") or event.endswith("_error"):
            status = "failed"
        elif event.endswith("_ok") or event.endswith("_captured") or event.endswith("_restored"):
            status = "succeeded"
        try:
            payload = self._relativize_control_payload({k: v for k, v in record.items() if k != "timestamp"})
            self.state_machine.append_event(
                event,
                task_type=task_type,
                task_id=task_id,
                status=status,
                payload=payload,
            )
        except OSError:
            logger.debug("[lnr] state-machine event mirror failed", exc_info=True)

    def _write_stage_map(self) -> None:
        archived = self._archived_stage_snapshot_index()
        payload = {
            "ledger_filename": self.ledger_filename,
            "stages": {
                sid: {
                    "stage_id": snap.stage_id,
                    "node_uid": self._snapshot_node_uid(snap),
                    "lineage_id": str(getattr(snap, "lineage_id", "") or (snap.source_event.get("lineage_id") if isinstance(snap.source_event, dict) else "") or ""),
                    "snapshot_id": snap.snapshot_id,
                    "snapshot_path": str(snap.snapshot_path),
                    "metric_value": snap.metric_value,
                    "metric_name": snap.metric_name,
                    "lower_is_better": snap.lower_is_better,
                    "memory_cut": snap.memory_cut,
                    "candidate_ready": (
                        snap.source_event.get("candidate_ready")
                        if isinstance(snap.source_event, dict)
                        else None
                    ),
                    "validation_ok": (
                        snap.source_event.get("validation_ok")
                        if isinstance(snap.source_event, dict)
                        else None
                    ),
                    "validation_issue": (
                        snap.source_event.get("validation_issue")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "reported_val_score": (
                        snap.source_event.get("reported_val_score")
                        if isinstance(snap.source_event, dict)
                        else None
                    ),
                    "val_score_type": (
                        snap.source_event.get("val_score_type")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "selection_eligible": (
                        snap.source_event.get("selection_eligible")
                        if isinstance(snap.source_event, dict)
                        else None
                    ),
                    "selection_score": (
                        snap.source_event.get("selection_score")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "selection_note": (
                        snap.source_event.get("selection_note")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "metric_source_note": (
                        snap.source_event.get("metric_source_note")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "metric_protocol": (
                        snap.source_event.get("metric_protocol")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "train_data_used": (
                        snap.source_event.get("train_data_used")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "metric_eval_data": (
                        snap.source_event.get("metric_eval_data")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "execution_mode": (
                        snap.source_event.get("execution_mode")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "submission_status": (
                        snap.source_event.get("submission_status")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    "duplicate_submission_of_stage": (
                        snap.source_event.get("duplicate_submission_of_stage")
                        if isinstance(snap.source_event, dict)
                        else ""
                    ),
                    **{
                        key: snap.source_event.get(key)
                        for key in (
                            "artifact_path",
                            "artifact_sha",
                            "metric_validity",
                            "metric_validity_note",
                            "metric_validity_reason_code",
                            "gate_metric_validity",
                            "gate_policy",
                            "gate_policy_version",
                            "gate_action",
                            "gate_accepted",
                            "gate_reason_code",
                            "route_id",
                            "solution_sha",
                            "submission_sha",
                            "submission_snapshot",
                        )
                        if isinstance(snap.source_event, dict)
                    },
                }
                for sid, snap in sorted(self.stage_snapshots.items())
            },
            "archive": {
                node_uid: {
                    "stage_id": snap.stage_id,
                    "node_uid": self._snapshot_node_uid(snap),
                    "lineage_id": str(getattr(snap, "lineage_id", "") or ""),
                    "snapshot_id": snap.snapshot_id,
                    "snapshot_path": str(snap.snapshot_path),
                    "metric_value": snap.metric_value,
                    "metric_name": snap.metric_name,
                    "lower_is_better": snap.lower_is_better,
                }
                for node_uid, snap in sorted(archived.items())
            },
        }
        for root in (self.log_dir,):
            try:
                root.mkdir(parents=True, exist_ok=True)
                atomic_write(
                    root / "lhr_stage_map.json",
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )
            except OSError:
                logger.debug("[lnr] stage map write failed", exc_info=True)

    def _load_existing_stage_snapshots(self) -> None:
        """Rehydrate committed stage snapshots for process-level resume."""

        # A Stage is durable only after both the ledger append and snapshot
        # capture complete. Repair a process crash between those writes before
        # deriving the active Stage set from the ledger.
        self._recover_stage_commit_transactions()
        discover_all = getattr(self.snapshot_store, "discover_all", None)
        if callable(discover_all):
            self.archived_stage_snapshots = dict(discover_all())
        else:
            self.archived_stage_snapshots = {
                self._archive_snapshot_key(snapshot): snapshot
                for snapshot in self.snapshot_store.discover().values()
            }
        cards = parse_stage_cards(read_ledger(self.ledger_path))
        active = {str(card.stage_id).strip().upper() for card in cards}
        if not active:
            return
        loaded: dict[str, StageSnapshot] = {}
        stage_map = self._read_json_file(self.log_dir / "lhr_stage_map.json")
        stages = stage_map.get("stages") if isinstance(stage_map.get("stages"), dict) else {}
        for stage_id, row in stages.items():
            sid = str(stage_id or "").strip().upper()
            if sid not in active or not isinstance(row, dict):
                continue
            snapshot_path = str(row.get("snapshot_path") or "").strip()
            if not snapshot_path:
                continue
            snapshot = self.snapshot_store.load(snapshot_path)
            if snapshot is not None and snapshot.stage_id == sid:
                loaded[sid] = snapshot
                self._index_archived_stage_snapshot(snapshot)
        for sid, snapshot in self.snapshot_store.discover().items():
            if sid in active and sid not in loaded:
                loaded[sid] = snapshot
                self._index_archived_stage_snapshot(snapshot)
        self.stage_snapshots = loaded
        if loaded:
            latest_sid = sorted(
                loaded,
                key=lambda value: int(value[1:]) if value.startswith("S") and value[1:].isdigit() else -1,
            )[-1]
            latest = loaded[latest_sid]
            source = latest.source_event if isinstance(latest.source_event, dict) else {}
            self.last_captured_solution_sha = str(source.get("solution_sha") or "")
            self.last_captured_run_signature = self._stage_run_signature(source)
            lineage_numbers = [
                int(text[1:])
                for snapshot in loaded.values()
                if (text := str(snapshot.lineage_id or "").strip().upper()).startswith("L")
                and text[1:].isdigit()
            ]
            if lineage_numbers:
                self.current_lineage_no = max(lineage_numbers)
                self.current_lineage_id = f"L{self.current_lineage_no:02d}"
        missing = sorted(active - set(loaded))
        self._jsonl(
            "lhr_resume_events.jsonl",
            {
                "event": "stage_snapshots_rehydrated",
                "active_stage_count": len(active),
                "loaded_stage_count": len(loaded),
                "loaded_stages": sorted(loaded),
                "missing_snapshot_stages": missing,
            },
        )

    def _prepare_workspace(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_split_logs()
        self._prune_workspace_control_artifacts()
        init_applied = self._prepare_init_workspace()
        desc = self.workspace_dir / "description.md"
        if self.task_desc.strip() and (init_applied or not desc.exists()):
            desc.write_text(self.task_desc.strip() + "\n", encoding="utf-8")
        self._prepare_dataset_symlink()
        self._prepare_workspace_git()

    def _init_workspace_path(self) -> str:
        init_cfg = getattr(self.lhr, "init_workspace", None)
        if init_cfg is None:
            return ""
        if isinstance(init_cfg, dict):
            return str(init_cfg.get("workspace_path") or "").strip()
        return str(getattr(init_cfg, "workspace_path", "") or "").strip()

    def _prepare_init_workspace(self) -> bool:
        raw_path = self._init_workspace_path()
        self.initial_workspace_state = ""
        if not raw_path:
            return False
        if (self.memory_dir / "ScienceAgent").is_dir():
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "init_workspace_skipped",
                    "reason": "existing_agent_memory",
                    "source_workspace": raw_path,
                },
            )
            return False
        result = initialize_workspace_from_path(
            source_workspace=raw_path,
            target_workspace=self.workspace_dir,
        )
        self.initial_workspace_state = result.initial_workspace_state or (
            "This run starts from files copied from a previous workspace. "
            "Inspect the existing files before changing the route."
        )
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "init_workspace_applied",
                "source_workspace": result.source_workspace,
                "applied": result.applied,
                "copied_file_count": result.copied_file_count,
                "skipped_entry_count": result.skipped_entry_count,
                "initial_workspace_state_path": result.initial_workspace_state_path,
                "initial_workspace_state_chars": len(result.initial_workspace_state or ""),
            },
        )
        return bool(result.applied)

    def _prepare_workspace_git(self) -> None:
        enabled = bool(getattr(self.lhr, "workspace_git_enabled", True))
        track_globs = normalize_workspace_git_track_globs(
            getattr(self.lhr, "workspace_git_track_globs", None),
        )
        try:
            self.lhr.workspace_git_track_globs = list(track_globs)
        except Exception:
            logger.debug("[lnr] could not normalize workspace git globs", exc_info=True)
        result = ensure_workspace_source_git(
            self.workspace_dir,
            enabled=enabled,
            track_globs=track_globs,
            initial_commit=bool(getattr(self.lhr, "workspace_git_initial_commit", True)),
            user_name="ScienceFlow NLR",
            user_email="scienceflow-lnr@local",
        )
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "workspace_git_init",
                "enabled": result.enabled,
                "ready": result.ready,
                "initialized": result.initialized,
                "committed": result.committed,
                "message": result.message,
                "track_globs": list(track_globs),
            },
        )
        if result.enabled and not result.ready:
            self.lhr.workspace_git_enabled = False
            if result.message:
                logger.warning("[lnr] workspace git unavailable: %s", result.message)

    def _interaction_log_node_dir(self) -> Path:
        # Kept as the stable logger identity root; actual LNR files live in self.log_dir.
        return self.root_dir

    def _split_logs_dir(self, name: str) -> Path:
        return self.log_dir / name

    def _close_lnr_interaction_loggers(self) -> None:
        color = bool(getattr(self.cfg, "scienceflow_interaction_log_color", True))
        seen: set[Path] = set()
        close_targets = (
            (self.workspace_dir, None),
            (self._interaction_log_node_dir(), self.log_dir),
        )
        for node_dir, direct_log_dir in close_targets:
            try:
                resolved = Path(node_dir).resolve()
            except OSError:
                resolved = Path(node_dir)
            key = resolved if direct_log_dir is None else Path(direct_log_dir).resolve(strict=False)
            if key in seen:
                continue
            seen.add(key)
            close_workspace_interaction_logger(
                resolved,
                color=color,
                layout="split",
                log_dir_override=direct_log_dir,
            )

    def _attach_lnr_interaction_logger(self, agent: Any) -> None:
        self._close_lnr_interaction_loggers()
        node_dir = self._interaction_log_node_dir()
        try:
            ensure_stage_log_dir(self.workspace_dir)
            agent._ws_interaction_log = attach_workspace_interaction_logger(
                node_dir,
                color=bool(getattr(self.cfg, "scienceflow_interaction_log_color", True)),
                layout="split",
                log_dir_override=self.log_dir,
            )
            attach_stage_interaction_handlers(
                agent._ws_interaction_log,
                self.workspace_dir,
                color=bool(getattr(self.cfg, "scienceflow_interaction_log_color", True)),
            )
        except Exception:
            logger.debug("[lnr] interaction logger reattach failed", exc_info=True)
        try:
            agent._tool_output_artifacts = agent._make_tool_output_artifact_store(
                node_dir,
                log_dir_override=self.log_dir,
            )
            stage_dir = stage_log_dir(self.workspace_dir)
            setter_stage_dir = getattr(agent._tool_output_artifacts, "set_stage_log_dir", None)
            if callable(setter_stage_dir):
                setter_stage_dir(stage_dir)
            setter = getattr(agent._tool_output_artifacts, "set_mirror_raw_id_prefix", None)
            if callable(setter):
                setter(self._next_stage_id_for_logging())
        except Exception:
            logger.debug("[lnr] tool output store reset failed", exc_info=True)

    def _worker_uid_prefix(self) -> str:
        raw = str(self.worker_id or "").strip()
        if raw:
            return raw
        return f"W{int(self.worker_index or 0):02d}"

    def _lineage_uid_prefix(self) -> str:
        raw = str(getattr(self, "current_lineage_id", "") or "").strip()
        if raw:
            return raw
        return f"L{int(getattr(self, 'current_lineage_no', 1) or 1):02d}"

    def _stage_node_uid(self, stage_id: str, *, lineage_id: str | None = None) -> str:
        lineage = str(lineage_id or self._lineage_uid_prefix()).strip()
        return f"{self._worker_uid_prefix()}:{lineage}:{str(stage_id or '').upper()}"

    def _start_new_lineage(self) -> str:
        self.current_lineage_no = int(getattr(self, "current_lineage_no", 1) or 1) + 1
        self.current_lineage_id = f"L{self.current_lineage_no:02d}"
        return self.current_lineage_id

    @staticmethod
    def _snapshot_node_uid(snap: StageSnapshot | None) -> str:
        if snap is None:
            return ""
        raw = str(getattr(snap, "node_uid", "") or "").strip()
        if raw:
            return raw
        raw_source = getattr(snap, "source_event", {})
        source = raw_source if isinstance(raw_source, dict) else {}
        return str(source.get("node_uid") or "")

    @classmethod
    def _archive_snapshot_key(cls, snap: StageSnapshot) -> str:
        return cls._snapshot_node_uid(snap) or f"legacy:{snap.snapshot_id}"

    def _archived_stage_snapshot_index(self) -> dict[str, StageSnapshot]:
        archived = getattr(self, "archived_stage_snapshots", None)
        if not isinstance(archived, dict):
            archived = {}
            self.archived_stage_snapshots = archived
        for snap in getattr(self, "stage_snapshots", {}).values():
            archived[self._archive_snapshot_key(snap)] = snap
        return archived

    def _index_archived_stage_snapshot(self, snap: StageSnapshot) -> None:
        self._archived_stage_snapshot_index()[self._archive_snapshot_key(snap)] = snap

    def _stage_snapshot_for_restore(self, *, stage_id: str, node_uid: str = "") -> StageSnapshot | None:
        sid = str(stage_id or "").strip().upper()
        expected_node_uid = str(node_uid or "").strip()
        active = getattr(self, "stage_snapshots", {}).get(sid)
        if not expected_node_uid:
            return active
        if active is not None and self._snapshot_node_uid(active) == expected_node_uid:
            return active
        archived = self._archived_stage_snapshot_index()
        snap = archived.get(expected_node_uid)
        if snap is None or str(snap.stage_id or "").strip().upper() != sid:
            return None
        return snap

    def _active_node_uid_for_stage(self, stage_id: str) -> str:
        return self._snapshot_node_uid(self.stage_snapshots.get(str(stage_id or "").upper()))

    def _next_active_stage_id(self, cards: list[Any] | None = None) -> str:
        if cards is None:
            cards = parse_stage_cards(read_ledger(self.ledger_path))
        return next_stage_id(cards)

    def _next_stage_id_for_logging(self) -> str:
        return self._next_active_stage_id()

    def _latest_stage_id_for_logging(self) -> str:
        cards = parse_stage_cards(read_ledger(self.ledger_path))
        if cards:
            return str(cards[-1].stage_id)
        seen: list[int] = []
        for sid in self.stage_snapshots:
            raw = str(sid or "").upper()
            if raw.startswith("S") and raw[1:].isdigit():
                seen.append(int(raw[1:]))
        if not seen:
            return ""
        return f"S{max(seen):02d}"

    def _record_context_compact_event(self, *, phase: str, **payload: Any) -> None:
        phase_name = str(phase or "event").strip().lower() or "event"
        event = f"context_compact_{phase_name}"
        latest_stage = self._latest_stage_id_for_logging()
        next_stage = self._next_stage_id_for_logging()
        record = {
            "event": event,
            "stage_id": latest_stage,
            "latest_stage": latest_stage,
            "next_stage": next_stage,
            **payload,
        }
        self._jsonl("lhr_context_events.jsonl", record)
        stage_label = latest_stage or next_stage or "-"
        mode = str(payload.get("mode") or "")
        round_no = payload.get("round")
        if phase_name == "started":
            logger.info(
                "[context-compact] started mode=%s reason=%s stage=%s round=%s omitted_before=%s recent_keep=%s",
                mode,
                payload.get("reason") or "",
                stage_label,
                round_no,
                payload.get("omitted_before"),
                payload.get("recent_keep"),
            )
        elif phase_name == "finished":
            tokens_in = int(payload.get("tokens_in") or 0)
            tokens_cached = int(payload.get("tokens_cached") or 0)
            cache_rate = tokens_cached / tokens_in if tokens_in > 0 else None
            cache_text = f"{cache_rate:.3f}" if cache_rate is not None else "-"
            logger.info(
                "[context-compact] finished mode=%s status=%s stage=%s round=%s omitted_after=%s summary_chars=%s tokens_in=%s tokens_out=%s cache_rate=%s",
                mode,
                payload.get("status") or "",
                stage_label,
                round_no,
                payload.get("omitted_after"),
                payload.get("summary_chars"),
                tokens_in,
                payload.get("tokens_out"),
                cache_text,
            )
        elif phase_name == "failed":
            logger.warning(
                "[context-compact] failed mode=%s stage=%s round=%s reason=%s fallback=%s",
                mode,
                stage_label,
                round_no,
                payload.get("reason_detail") or "",
                payload.get("fallback") or "",
            )
        elif phase_name == "estra_check":
            logger.info(
                "[context-compact] estra_check mode=%s stage=%s round=%s omitted_before=%s",
                mode,
                stage_label,
                round_no,
                payload.get("omitted_before"),
            )
        elif phase_name == "deferred":
            logger.info(
                "[context-compact] deferred mode=%s stage=%s round=%s reason=%s",
                mode,
                stage_label,
                round_no,
                payload.get("reason") or "",
            )

    def _ensure_split_logs(self) -> None:
        for rel in (
            ("interaction",),
            ("interaction", "tool_outputs"),
            ("traj_interaction",),
            ("traj_interaction", "tool_outputs"),
        ):
            self.log_dir.joinpath(*rel).mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "tmp").mkdir(parents=True, exist_ok=True)
        for rel in (("agentic_route",), ("submission_snapshots",)):
            self.log_dir.joinpath(*rel).mkdir(parents=True, exist_ok=True)
        ensure_stage_log_dir(self.workspace_dir)

    def _prune_workspace_control_artifacts(self) -> None:
        """Keep LHR control-plane files out of the agent-visible workspace log tree."""
        for control_dir in (self.workspace_dir / "logs",):
            try:
                if control_dir.is_dir() and not control_dir.is_symlink():
                    shutil.rmtree(control_dir)
                elif control_dir.exists() or control_dir.is_symlink():
                    control_dir.unlink()
            except OSError:
                logger.debug("[lnr] workspace control dir cleanup skipped: %s", control_dir, exc_info=True)
        log_dir = self.workspace_dir / ".logs"
        for name in (
            "agentic_route_decisions.jsonl",
            "agentic_route_response.md",
            "full_run_stamp.json",
            "fullrun_tail_snapshot.json",
            "lhr_stage_map.json",
            "lhr_snapshot_meta.json",
        ):
            try:
                (log_dir / name).unlink(missing_ok=True)
            except OSError:
                logger.debug("[lnr] workspace control artifact cleanup skipped", exc_info=True)
        try:
            for path in log_dir.glob("lhr_*.jsonl"):
                path.unlink(missing_ok=True)
        except OSError:
            logger.debug("[lnr] workspace jsonl cleanup skipped", exc_info=True)
        try:
            legacy_tmp = log_dir / "tmp"
            if legacy_tmp.is_dir():
                shutil.rmtree(legacy_tmp)
            elif legacy_tmp.exists() or legacy_tmp.is_symlink():
                legacy_tmp.unlink()
        except OSError:
            logger.debug("[lnr] workspace legacy tmp cleanup skipped", exc_info=True)
        for name in (
            ".memory",
            ".snapshots",
            "logs",
            "snapshots",
            "stage_memory",
            "submission_history",
            "submission_snapshots",
            "submissions",
        ):
            path = self.workspace_dir / name
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists() or path.is_symlink():
                    path.unlink()
            except OSError:
                logger.debug("[lnr] workspace artifact cleanup skipped: %s", path, exc_info=True)
        ensure_stage_log_dir(self.workspace_dir)

    def _reset_interaction_stage_files(self) -> None:
        inter_dir = self._split_logs_dir("interaction")
        try:
            if inter_dir.exists():
                shutil.rmtree(inter_dir)
            (inter_dir / "tool_outputs").mkdir(parents=True, exist_ok=True)
            (inter_dir / "interaction.log").touch(exist_ok=True)
        except OSError:
            logger.debug("[lnr] interaction stage reset skipped", exc_info=True)
        try:
            reset_stage_log_dir(self.workspace_dir)
        except OSError:
            logger.debug("[lnr] stage-local interaction reset skipped", exc_info=True)
        self._ensure_split_logs()

    def _reset_agent_interaction_stage(self, agent: Any) -> None:
        self._reset_interaction_stage_files()
        self._attach_lnr_interaction_logger(agent)

    @staticmethod
    def _agent_memory_messages(agent: Any) -> list[Any]:
        try:
            records = agent.memory.chat_history_memory.retrieve(window_size=None)
        except Exception:
            return []
        out: list[Any] = []
        for rec in records or []:
            msg = getattr(getattr(rec, "memory_record", None), "message", None)
            if msg is not None:
                out.append(msg)
        return out

    @staticmethod
    def _message_content_text(msg: Any) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _tool_calls_serialized_chars(msg: Any) -> int:
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            return 0
        try:
            return len(json.dumps(tcs, ensure_ascii=False))
        except (TypeError, ValueError):
            return len(str(tcs))

    @staticmethod
    def _clean_eda_fact_line(line: str) -> str:
        text = _ANSI_RE.sub("", str(line or ""))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        return text

    @staticmethod
    def _looks_like_eda_fact(line: str) -> bool:
        if not line or len(line) < 4:
            return False
        if line.startswith("[exit=") or line.startswith("[bash:") or line.startswith("[stream"):
            return False
        if _LHR_EXPERIMENT_NOISE_RE.search(line):
            return False
        return bool(_LHR_EDA_FACT_RE.search(line))

    def _build_protected_eda_facts_summary(self, agent: Any, *, end_index: int) -> str:
        max_chars = max(1200, int(getattr(self.lhr, "protected_eda_facts_max_chars", 6000) or 6000))
        messages = self._agent_memory_messages(agent)
        # The first user/task prompt is preserved separately by MemoryContextManager.
        candidates = messages[1:max(1, min(int(end_index), len(messages)))]
        facts: list[str] = []
        seen: set[str] = set()
        scratch_tool_call_chars = 0
        scratch_tool_call_count = 0
        for msg in candidates:
            role = str(getattr(msg, "role", "") or "")
            tc_chars = self._tool_calls_serialized_chars(msg)
            if tc_chars:
                scratch_tool_call_chars += tc_chars
                scratch_tool_call_count += 1
            if role != "tool":
                continue
            text = self._message_content_text(msg)
            for raw_line in text.splitlines():
                line = self._clean_eda_fact_line(raw_line)
                if not self._looks_like_eda_fact(line):
                    continue
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                facts.append(line)
                used = sum(len(x) + 3 for x in facts)
                if used >= max_chars - 700 or len(facts) >= 36:
                    break
            if sum(len(x) + 3 for x in facts) >= max_chars - 700 or len(facts) >= 36:
                break

        lines = [
            "Fixed EDA facts retained from the first successful stage.",
            "Scratch commands, temporary code, full heredocs, and model-probe tool-call payloads were removed from the fixed prefix; raw audit remains in `.logs/interaction/`, `.logs/traj_interaction/`, and stage snapshots.",
        ]
        if scratch_tool_call_count:
            lines.append(
                f"Removed raw pre-S01 scratch payloads: {scratch_tool_call_count} assistant tool-call message(s), about {scratch_tool_call_chars} serialized chars."
            )
        if facts:
            lines.append("EDA facts:")
            lines.extend(f"- {fact}" for fact in facts)
        else:
            lines.append("EDA facts: no deterministic fact lines were confidently extracted; inspect `dataset/` or `.logs/` if a specific early detail is needed.")
        summary = "\n".join(lines).strip()
        if len(summary) > max_chars:
            summary = summary[: max_chars - 38].rstrip() + "\n... [EDA facts truncated]"
        return summary

    def _mark_protected_eda_prefix(self, agent: Any, *, stage_id: str) -> dict[str, Any]:
        if str(stage_id or "").upper() != "S01":
            return {}
        if not bool(getattr(self.lhr, "preserve_prefix_and_eda", True)):
            return {}
        ctx = getattr(agent, "_memory_ctx", None)
        warn_chars = int(getattr(self.lhr, "protected_eda_warn_chars", 50_000) or 0)
        captured_end = getattr(self, "_s01_eda_prefix_end_index", None)
        end_index = (
            int(captured_end)
            if captured_end is not None
            else self._count_memory_records()
        )
        mode = str(getattr(self.lhr, "protected_eda_mode", "facts") or "facts").strip().lower()
        try:
            if mode in {"raw", "verbatim"}:
                setter = getattr(ctx, "set_protected_raw_prefix", None)
                if not callable(setter):
                    return {}
                info = setter(
                    end_index,
                    warn_chars=warn_chars,
                    label="LHR protected EDA fixed prefix",
                )
                info = {**dict(info), "mode": "raw"}
            else:
                setter = getattr(ctx, "replace_protected_raw_prefix_with_summary", None)
                if not callable(setter):
                    return {}
                summary = self._build_protected_eda_facts_summary(agent, end_index=end_index)
                info = setter(
                    end_index,
                    summary,
                    warn_chars=warn_chars,
                    label="LHR protected EDA facts",
                )
                info = {**dict(info), "mode": "facts", "summary_chars": len(summary)}
        except Exception:
            logger.debug("[lnr] protected EDA prefix marker failed", exc_info=True)
            return {}
        try:
            effective_end_index = int(info.get("end_index", end_index))
        except (TypeError, ValueError):
            effective_end_index = end_index
        if captured_end is not None:
            self._s01_eda_prefix_end_index = effective_end_index
        info = {
            **dict(info),
            "protected_end_index": effective_end_index,
            "boundary_source": (
                "pre_stage_commit" if captured_end is not None else "current_memory"
            ),
        }
        self._jsonl(
            "lhr_context_events.jsonl",
            {
                "event": "protected_eda_prefix_set",
                "stage_id": "S01",
                **info,
            },
        )
        chars = int(info.get("chars") or 0)
        original_chars = int(info.get("original_chars") or 0)
        if original_chars:
            logger.info(
                "[lnr] protected EDA mode=%s fixed_chars=%d original_chars=%d original_messages=%s",
                info.get("mode"),
                chars,
                original_chars,
                info.get("original_message_count"),
            )
        if warn_chars > 0 and chars > warn_chars:
            logger.warning(
                "[lnr] protected EDA fixed prefix is %d chars, exceeding warning threshold %d; keeping it verbatim",
                chars,
                warn_chars,
            )
        return dict(info)

    def _capture_s01_eda_prefix_end(self, *, stage_id: str) -> None:
        if str(stage_id or "").upper() != "S01":
            return
        if getattr(self, "_s01_eda_prefix_end_index", None) is None:
            self._s01_eda_prefix_end_index = self._count_memory_records()

    def _restore_protected_eda_prefix_marker(self, agent: Any) -> None:
        if not bool(getattr(self.lhr, "preserve_prefix_and_eda", True)):
            return
        snap = self.stage_snapshots.get("S01")
        if snap is None or not int(getattr(snap, "memory_cut", 0) or 0):
            return
        raw_source = getattr(snap, "source_event", {})
        source = raw_source if isinstance(raw_source, dict) else {}
        captured_end = source.get("protected_eda_end_index")
        try:
            end_index = int(captured_end)
        except (TypeError, ValueError):
            end_index = int(snap.memory_cut)
        self._s01_eda_prefix_end_index = end_index
        ctx = getattr(agent, "_memory_ctx", None)
        setter = getattr(ctx, "set_protected_raw_prefix", None)
        if not callable(setter):
            return
        warn_chars = int(getattr(self.lhr, "protected_eda_warn_chars", 50_000) or 0)
        try:
            setter(
                end_index,
                warn_chars=warn_chars,
                label="LHR protected EDA fixed prefix",
            )
        except Exception:
            logger.debug("[lnr] protected EDA prefix restore failed", exc_info=True)

    def _append_traj_summary(self, *, target_stage: str, summary: str) -> None:
        text = str(summary or "").strip()
        if not text:
            return
        path = self._split_logs_dir("traj_interaction") / "traj_interaction.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    "\n"
                    f"[estra-summary] target={str(target_stage or '').upper()}\n"
                    f"{text}\n"
                )
        except OSError:
            logger.debug("[lnr] traj summary append failed", exc_info=True)

    @staticmethod
    def _memory_user_record(content: str) -> dict[str, Any]:
        return {
            "uuid": str(uuid.uuid4()),
            "message": {"__class__": "Message", "role": "user", "content": str(content or "")},
            "role": "user",
            "extra_info": {},
            "timestamp": time.time(),
            "agent_id": "",
        }

    @staticmethod
    def _memory_record_role(record: dict[str, Any]) -> str:
        msg = record.get("message")
        if isinstance(msg, dict):
            return str(msg.get("role") or record.get("role") or "").strip()
        return str(record.get("role") or "").strip()

    def _minimal_task_prefix_memory_records(self, agent_dir: Path) -> list[dict[str, Any]]:
        max_messages = max(1, min(int(getattr(self.cfg, "max_messages", 100) or 100), 8))
        records = select_prefix_safe_agent_memory_records(Path(agent_dir), max_messages=max_messages)
        prefix: list[dict[str, Any]] = []
        for rec in records:
            if self._memory_record_role(rec) in {"system", "user"}:
                prefix.append(rec)
                continue
            break
        if prefix:
            return prefix
        for rec in records:
            if self._memory_record_role(rec) in {"system", "user"}:
                return [rec]
        return []

    def _base_stage_id(self) -> str:
        cards = parse_stage_cards(read_ledger(self.ledger_path))
        snapshots = getattr(self, "stage_snapshots", {}) or {}
        for card in cards:
            sid = normalize_stage_id(getattr(card, "stage_id", ""))
            if sid and sid in snapshots:
                return sid
        seen = [normalize_stage_id(sid) for sid in snapshots]
        seen = [sid for sid in seen if sid]
        if not seen:
            return ""
        return sorted(seen, key=lambda sid: int(sid[1:]))[0]

    def _base_stage_agent_memory_dir(self) -> Path | None:
        base_stage = self._base_stage_id()
        if not base_stage:
            return None
        base = self.stage_snapshots.get(base_stage)
        if base is None:
            return None
        snapshot_path = getattr(base, "snapshot_path", None)
        if snapshot_path is None:
            return None
        for candidate in (
            Path(snapshot_path) / "logs" / "memory" / "ScienceAgent",
            Path(snapshot_path) / ".memory" / "ScienceAgent",
        ):
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _metric_value_float(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out == out else None

    @staticmethod
    def _stage_card_override_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return f"{float(value):.12g}"
        return str(value or "").strip()

    def _stage_card_effective_overrides(self, cards: list[Any]) -> dict[str, dict[str, Any]]:
        overrides: dict[str, dict[str, Any]] = {}
        for card in cards:
            sid = normalize_stage_id(getattr(card, "stage_id", ""))
            if not sid:
                continue
            snap = self.stage_snapshots.get(sid)
            raw_source = getattr(snap, "source_event", {}) if snap is not None else {}
            source = raw_source if isinstance(raw_source, dict) else {}
            if not source:
                continue
            fields: dict[str, Any] = {}
            metric = source.get("metric_value")
            if metric not in (None, ""):
                fields["metric"] = self._stage_card_override_text(metric)
            lower = source.get("lower_is_better")
            if isinstance(lower, bool):
                fields["lower_is_better"] = "true" if lower else "false"
            metric_type = source.get("val_score_type") or source.get("metric_protocol")
            if metric_type not in (None, ""):
                fields["metric_type"] = self._stage_card_override_text(metric_type)
            note = source.get("selection_note") or source.get("metric_validity_note") or source.get("validation_issue")
            if note not in (None, ""):
                fields["metric_note"] = self._stage_card_override_text(note)
            validity = source.get("metric_validity")
            if validity not in (None, ""):
                fields["metric_validity"] = self._stage_card_override_text(validity).lower()
            eligible = source.get("selection_eligible")
            if eligible not in (None, ""):
                fields["selection_eligible"] = self._stage_card_override_text(eligible).lower()
            reason_code = source.get("metric_validity_reason_code") or source.get("metric_reason_code")
            if reason_code not in (None, ""):
                fields["metric_validity_reason_code"] = self._stage_card_override_text(reason_code)
            if fields:
                overrides[sid] = fields
        return overrides

    def _effective_stage_cards(self, cards: list[Any]) -> list[Any]:
        return stage_cards_with_overrides(cards, self._stage_card_effective_overrides(cards))

    def _effective_stage_cards_from_ledger(self) -> list[Any]:
        return self._effective_stage_cards(parse_stage_cards(read_ledger(self.ledger_path)))

    def _stage_cards_from_snapshots(self) -> list[StageCard]:
        cards: list[StageCard] = []
        for sid in sorted(self.stage_snapshots, key=lambda x: (normalize_stage_id(x) or str(x))):
            stage_id = normalize_stage_id(sid)
            if not stage_id:
                continue
            snap = self.stage_snapshots.get(stage_id)
            raw_source = getattr(snap, "source_event", {}) if snap is not None else {}
            source = raw_source if isinstance(raw_source, dict) else {}
            metric = source.get("metric_value", getattr(snap, "metric_value", ""))
            lower = source.get("lower_is_better", getattr(snap, "lower_is_better", ""))
            lower_text = self._stage_card_override_text(lower).lower() if lower not in (None, "") else ""
            cards.append(
                StageCard(
                    stage_id=stage_id,
                    body="",
                    metric=self._stage_card_override_text(metric) if metric not in (None, "") else "",
                    lower_is_better=lower_text,
                    metric_validity=self._stage_card_override_text(source.get("metric_validity")).lower(),
                    selection_eligible=self._stage_card_override_text(source.get("selection_eligible")).lower(),
                    metric_validity_reason_code=self._stage_card_override_text(
                        source.get("metric_validity_reason_code") or source.get("metric_reason_code")
                    ),
                    brief="restorable stage snapshot",
                    why="ledger card unavailable; using checkpoint metadata",
                )
            )
        return cards

    def _stage_memory_view_for_prompt(
        self,
        cards: list[Any],
        *,
        target_stage: str = "",
        latest_stage: str = "",
        best_stage: str = "",
    ) -> str:
        if not bool(getattr(self.lhr, "stage_memory_folding_enabled", True)):
            return render_stage_cards(cards)
        budget = int(getattr(self.lhr, "stage_memory_context_budget_chars", 24000) or 24000)
        view = build_stage_memory_view(
            self.workspace_dir,
            cards,
            context_budget_chars=budget,
            rebuild_on_stale=bool(getattr(self.lhr, "stage_memory_rebuild_on_stale", True)),
            target_stage=target_stage,
            latest_stage=latest_stage,
            best_stage=best_stage,
        )
        if view.folded_stage_count or view.summary_ids:
            self._jsonl(
                "lhr_stage_memory_events.jsonl",
                {
                    "event": "stage_memory_view_built",
                    "raw_chars": view.raw_chars,
                    "view_chars": view.view_chars,
                    "folded_stage_count": view.folded_stage_count,
                    "summary_ids": list(view.summary_ids),
                    "verification_stage_ids": list(view.verification_stage_ids),
                    "reused_summary_count": view.reused_summary_count,
                    "created_summary_count": view.created_summary_count,
                },
            )
        return view.text

    @staticmethod
    def _stage_card_one_line(text: str, *, max_chars: int) -> str:
        raw = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(raw) <= max_chars:
            return raw
        return raw[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"

    def _best_candidate_stage_id(self, cards: list[Any], candidates: list[str]) -> str:
        allowed = {str(sid).upper() for sid in candidates}
        best_stage = ""
        best_metric: float | None = None
        best_lower = True
        for card in cards:
            sid = str(getattr(card, "stage_id", "") or "").upper()
            if sid not in allowed:
                continue
            metric = self._metric_value_float(getattr(card, "metric", ""))
            if metric is None:
                continue
            lower_raw = str(getattr(card, "lower_is_better", "") or "").strip().lower()
            lower = lower_raw not in {"false", "0", "no"}
            if best_metric is None:
                best_stage = sid
                best_metric = metric
                best_lower = lower
                continue
            if (lower and metric < best_metric) or ((not lower) and metric > best_metric):
                best_stage = sid
                best_metric = metric
                best_lower = lower
        _ = best_lower
        return best_stage

    def _deterministic_estra_decision(
        self,
        *,
        cards: list[Any],
        candidates: list[str],
        latest: str,
        trigger_source: str,
        reason: str,
    ) -> dict[str, Any]:
        latest = str(latest or "").upper()
        candidates = [str(sid or "").upper() for sid in candidates]
        switch_candidates = self._estra_switch_candidates(candidates, latest)
        best = self._best_candidate_stage_id(cards, candidates)
        if best and str(best).upper() in switch_candidates:
            action = "switch_stage"
            target = str(best).upper()
        else:
            action = "keep_current"
            target = latest if latest in candidates else (candidates[-1] if candidates else "")
        startpoint, intent = self._estra_axes_from_action(action)
        return {
            "action": action,
            "startpoint": startpoint,
            "intent": intent,
            "target_stage": target,
            "compact": self._derive_estra_compact(action=action, trigger_source=trigger_source),
            "reason": reason,
        }

    def _stage_packet_line(self, card: Any, *, max_chars: int) -> str:
        sid = str(getattr(card, "stage_id", "") or "").upper()
        snap = self.stage_snapshots.get(sid)
        source_raw = getattr(snap, "source_event", {}) if snap is not None else {}
        source = source_raw if isinstance(source_raw, dict) else {}
        node_uid = self._snapshot_node_uid(snap) if snap is not None else ""
        lineage_id = str(
            getattr(snap, "lineage_id", "")
            or source.get("lineage_id")
            or ""
        )
        commit = str(source.get("source_commit_sha") or "")[:12]
        validation_ok = source.get("validation_ok")
        parts = [
            f"- {sid}",
            f"metric={getattr(card, 'metric', '') or 'unknown'}",
        ]
        lower = str(getattr(card, "lower_is_better", "") or "").strip()
        if lower:
            parts.append(f"lower_is_better={lower}")
        if node_uid:
            parts.append(f"node_uid={node_uid}")
        if lineage_id:
            parts.append(f"lineage={lineage_id}")
        if commit:
            parts.append(f"source_commit={commit}")
        if validation_ok is not None:
            parts.append(f"validation_ok={validation_ok}")
        metric_validity = str(getattr(card, "metric_validity", "") or "").strip()
        if metric_validity:
            parts.append(f"metric_validity={metric_validity}")
        selection_eligible = str(getattr(card, "selection_eligible", "") or "").strip()
        if selection_eligible:
            parts.append(f"selection_eligible={selection_eligible}")
        reason_code = str(getattr(card, "metric_validity_reason_code", "") or "").strip()
        if reason_code:
            parts.append(f"metric_validity_reason={reason_code}")
        brief = self._stage_card_one_line(getattr(card, "brief", ""), max_chars=180)
        why = self._stage_card_one_line(getattr(card, "why", ""), max_chars=180)
        route_evidence = self._stage_card_one_line(getattr(card, "route_evidence", ""), max_chars=220)
        if route_evidence and route_evidence not in why:
            why = self._stage_card_one_line(f"{why}; {route_evidence}" if why else route_evidence, max_chars=260)
        line = " | ".join(parts)
        if brief:
            line += f"\n  BRIEF: {brief}"
        if why:
            line += f"\n  WHY: {why}"
        stage_events = tuple(getattr(card, "stage_events", ()) or ())
        if stage_events:
            event_text = self._stage_card_one_line(" | ".join(str(x) for x in stage_events[:2]), max_chars=220)
            line += f"\n  EVENT: {event_text}"
        if len(line) > max_chars:
            line = line[: max(0, max_chars - 24)].rstrip() + "\n  ... [truncated]"
        return line

    def _build_lhr_state_packet(
        self,
        *,
        action: str,
        target_stage: str,
        terminal_stage: str,
        reason: str,
        tail_summary: str = "",
        estra_fields: dict[str, Any] | None = None,
    ) -> str:
        cards = self._effective_stage_cards_from_ledger()
        if not cards:
            cards = self._stage_cards_from_snapshots()
        candidates = [c.stage_id for c in cards if c.stage_id in self.stage_snapshots]
        latest = str(terminal_stage or (cards[-1].stage_id if cards else "")).upper()
        target = str(target_stage or "").upper()
        best = self._best_candidate_stage_id(cards, candidates)
        max_chars = int(getattr(self.lhr, "state_packet_max_chars", 12000) or 12000)
        max_branch_chars = int(
            getattr(self.lhr, "state_packet_archived_branch_max_chars", 1500) or 1500
        )
        summary = str(tail_summary or "").strip()
        estra_fields = estra_fields or {}
        startpoint = str(estra_fields.get("startpoint") or self._estra_axes_from_action(action)[0])
        intent = str(estra_fields.get("intent") or self._estra_axes_from_action(action)[1])
        if len(summary) > max_branch_chars:
            summary = summary[: max(0, max_branch_chars - 40)].rstrip() + "\n... [branch summary truncated]"

        def render(target_cards: list[Any], *, include_why: bool = True) -> str:
            lines = [
                "## LHR State Packet v2",
                f"ESTRA action: {str(action or '').strip() or 'keep_current'}",
                f"ESTRA startpoint: {startpoint or '(unknown)'}",
                f"ESTRA intent: {intent or '(unknown)'}",
                f"ESTRA target stage: {target or '(none)'}",
                f"Terminal stage before estra: {latest or '(none)'}",
                f"Best known restorable stage: {best or '(unknown)'}",
                f"ESTRA reason: {self._compact_event_text(reason, max_chars=360)}",
            ]
            for label, key in (
                ("Exploration summary", "exploration_summary"),
                ("ESTRA bottleneck", "bottleneck"),
                ("ESTRA evidence", "evidence"),
                ("Missing evidence", "missing_evidence"),
                ("Redirect focus", "redirect_focus"),
            ):
                value = self._compact_event_text(str(estra_fields.get(key) or ""), max_chars=240)
                if value:
                    lines.append(f"{label}: {value}")
            if include_why:
                checkpoint_context = self._estra_stage_checkpoint_context(candidates)
                if checkpoint_context:
                    lines.extend(["", "## Stage Checkpoint Metadata", checkpoint_context])
                lines.append(
                    self._stage_memory_view_for_prompt(
                        target_cards,
                        target_stage=target,
                        latest_stage=latest,
                        best_stage=best,
                    )
                )
            else:
                lines.extend(["", "## Active Stage Cards (shrunk)"])
                for card in target_cards:
                    sid = str(getattr(card, "stage_id", "") or "").upper()
                    brief = self._stage_card_one_line(getattr(card, "brief", ""), max_chars=160)
                    lines.append(
                        f"- {sid} | metric={getattr(card, 'metric', '') or 'unknown'} | BRIEF={brief}"
                    )
            if summary:
                branch_label = (
                    "## Historical Exploration Summary (abandoned branch evidence)"
                    if str(action or "") == "switch_stage"
                    else "## Current Branch Compact Summary"
                )
                lines.extend(["", branch_label, summary])
            lines.extend(
                [
                    "",
                    "## Workspace State",
                    "Workspace files are the source of truth. Inspect files before editing.",
                    "Tracked source globs: *.py, *.md",
                    "Reusable layout preference: train.py / predict.py / util.py when useful.",
                    self._workspace_state_deliverable_line(),
                ]
            )
            return "\n".join(lines).strip()

        packet = render(cards)
        if len(packet) <= max_chars:
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "state_packet_built",
                    "action": action,
                    "target_stage": target,
                    "terminal_stage": latest,
                    "best_stage": best,
                    "stage_count": len(cards),
                    "candidate_count": len(candidates),
                    "chars": len(packet),
                    "shrink_level": 0,
                },
            )
            return packet

        base_stage = self._base_stage_id()
        keep_ids = {sid for sid in (base_stage, target, latest, best) if sid}
        for card in cards[-3:]:
            keep_ids.add(str(getattr(card, "stage_id", "") or "").upper())
        shrunk_cards = [card for card in cards if str(getattr(card, "stage_id", "") or "").upper() in keep_ids]
        packet = render(shrunk_cards)
        shrink_level = 1
        if len(packet) > max_chars:
            packet = render(shrunk_cards, include_why=False)
            shrink_level = 2
        if len(packet) > max_chars:
            packet = packet[: max(0, max_chars - 44)].rstrip() + "\n... [state packet truncated to fit budget]"
            shrink_level = 3
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "state_packet_built",
                "action": action,
                "target_stage": target,
                "terminal_stage": latest,
                "best_stage": best,
                "stage_count": len(cards),
                "candidate_count": len(candidates),
                "chars": len(packet),
                "shrink_level": shrink_level,
            },
        )
        return packet

    def _rebuild_memory_after_estra(
        self,
        *,
        target_stage: str,
        summary: str,
        state_packet: str = "",
        strict_context_limit: bool = False,
    ) -> tuple[bool, int]:
        if not bool(getattr(self.lhr, "estra_compact_enabled", True)):
            return False, 0
        base_stage = self._base_stage_id()
        base_dir = self._base_stage_agent_memory_dir()
        if base_dir is None:
            self._jsonl(
                "lhr_estras.jsonl",
                {"event": "estra_memory_compact_skipped", "reason": f"missing_{(base_stage or 'base_stage').lower()}_memory", "base_stage": base_stage},
            )
            return False, 0
        max_messages = int(getattr(self.cfg, "max_messages", 100) or 100)
        records = (
            self._minimal_task_prefix_memory_records(base_dir)
            if strict_context_limit
            else select_prefix_safe_agent_memory_records(base_dir, max_messages=max_messages)
        )
        if not records:
            self._jsonl(
                "lhr_estras.jsonl",
                {"event": "estra_memory_compact_skipped", "reason": f"empty_{(base_stage or 'base_stage').lower()}_memory", "base_stage": base_stage},
            )
            return False, 0
        next_stage = self._next_stage_id_for_logging()
        prompt = build_estra_resume_prompt(
            target_stage=str(target_stage or "").upper(),
            next_stage=next_stage,
            ledger_filename=self.ledger_filename,
            base_stage=base_stage,
            compact_summary=summary,
            state_packet=state_packet,
            parallel_worker_snapshot=self._parallel_worker_snapshot_for_prompt(),
            resource_context=self._resource_context_for_prompt(),
        )
        records = [*records, self._memory_user_record(prompt)]
        dest = self.memory_dir / "ScienceAgent"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            write_agent_memory_record_files(dest, records, write_long_term=True)
        except OSError:
            logger.debug("[lnr] estra memory compact write failed", exc_info=True)
            return False, 0
        context_preview = {
            "event": "estra_context_prepared",
            "target_stage": str(target_stage or "").upper(),
            "next_stage": next_stage,
            "base_stage": base_stage,
            "memory_records": len(records),
            "compact_strength": "strict_context_limit" if strict_context_limit else "normal",
            "context_shape": [
                (
                    f"{base_stage or 'base-stage'} minimal task prefix records"
                    if strict_context_limit
                    else f"{base_stage or 'base-stage'} prefix-safe memory records"
                ),
                "deterministic LHR state packet",
                "one estra resume user prompt",
                "compact tail summary from skipped stages",
            ],
            "resume_prompt_excerpt": self._compact_event_text(prompt, max_chars=900),
            "tail_summary_excerpt": self._compact_event_text(summary, max_chars=900),
            "summary_chars": len(str(summary or "")),
            "state_packet_chars": len(str(state_packet or "")),
        }
        self._jsonl("lhr_estras.jsonl", context_preview)
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_memory_compacted",
                "target_stage": str(target_stage or "").upper(),
                "base_stage": base_stage,
                "records": len(records),
                "compact_strength": "strict_context_limit" if strict_context_limit else "normal",
                "summary_chars": len(str(summary or "")),
                "state_packet_chars": len(str(state_packet or "")),
                "next_stage": next_stage,
            },
        )
        return True, len(records)

    def _rebuild_memory_after_keep_current(
        self,
        *,
        terminal_stage: str,
        summary: str,
        state_packet: str = "",
        strict_context_limit: bool = False,
        continuation_action: str = "keep_current",
        estra_fields: dict[str, Any] | None = None,
    ) -> tuple[bool, int]:
        if not bool(getattr(self.lhr, "estra_compact_enabled", True)):
            return False, 0
        base_stage = self._base_stage_id()
        base_dir = self._base_stage_agent_memory_dir()
        if base_dir is None:
            self._jsonl(
                "lhr_estras.jsonl",
                {"event": "estra_keep_current_memory_compact_skipped", "reason": f"missing_{(base_stage or 'base_stage').lower()}_memory", "base_stage": base_stage},
            )
            return False, 0
        max_messages = int(getattr(self.cfg, "max_messages", 100) or 100)
        records = (
            self._minimal_task_prefix_memory_records(base_dir)
            if strict_context_limit
            else select_prefix_safe_agent_memory_records(base_dir, max_messages=max_messages)
        )
        if not records:
            self._jsonl(
                "lhr_estras.jsonl",
                {"event": "estra_keep_current_memory_compact_skipped", "reason": f"empty_{(base_stage or 'base_stage').lower()}_memory", "base_stage": base_stage},
            )
            return False, 0
        next_stage = self._next_stage_id_for_logging()
        fields = estra_fields or {}
        prompt = build_keep_current_compact_prompt(
            terminal_stage=str(terminal_stage or "").upper(),
            next_stage=next_stage,
            compact_summary=summary,
            base_stage=base_stage,
            state_packet=state_packet,
            parallel_worker_snapshot=self._parallel_worker_snapshot_for_prompt(),
            resource_context=self._resource_context_for_prompt(),
            continuation_action=continuation_action,
            exploration_summary=str(fields.get("exploration_summary") or ""),
            bottleneck=str(fields.get("bottleneck") or ""),
            evidence=str(fields.get("evidence") or ""),
            missing_evidence=str(fields.get("missing_evidence") or ""),
            decision_reason=str(fields.get("decision_reason") or ""),
            redirect_focus=str(fields.get("redirect_focus") or ""),
        )
        records = [*records, self._memory_user_record(prompt)]
        dest = self.memory_dir / "ScienceAgent"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            write_agent_memory_record_files(dest, records, write_long_term=True)
        except OSError:
            logger.debug("[lnr] terminal continue memory compact write failed", exc_info=True)
            return False, 0
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_keep_current_context_prepared",
                "action": continuation_action,
                "terminal_stage": str(terminal_stage or "").upper(),
                "next_stage": next_stage,
                "base_stage": base_stage,
                "memory_records": len(records),
                "compact_strength": "strict_context_limit" if strict_context_limit else "normal",
                "context_shape": [
                    (
                        f"{base_stage or 'base-stage'} minimal task prefix records"
                        if strict_context_limit
                        else f"{base_stage or 'base-stage'} prefix-safe memory records"
                    ),
                    "deterministic LHR state packet",
                    "one terminal-continue user prompt",
                    "compact current trajectory summary",
                ],
                "resume_prompt_excerpt": self._compact_event_text(prompt, max_chars=900),
                "tail_summary_excerpt": self._compact_event_text(summary, max_chars=900),
                "summary_chars": len(str(summary or "")),
                "state_packet_chars": len(str(state_packet or "")),
            },
        )
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_keep_current_memory_compacted",
                "action": continuation_action,
                "terminal_stage": str(terminal_stage or "").upper(),
                "base_stage": base_stage,
                "records": len(records),
                "compact_strength": "strict_context_limit" if strict_context_limit else "normal",
                "summary_chars": len(str(summary or "")),
                "state_packet_chars": len(str(state_packet or "")),
                "next_stage": next_stage,
            },
        )
        return True, len(records)

    def _prepare_dataset_symlink(self) -> None:
        inp = Path(self.cfg.input_data_dir).expanduser().resolve(strict=False)
        if not str(inp).strip() or not inp.exists():
            return
        ws_dataset = self.workspace_dir / "dataset"
        if ws_dataset.exists() or ws_dataset.is_symlink():
            return
        from scienceflow.solver.lnr.prep_fs import (
            prepare_workspace_dataset_flat,
            resolve_workspace_dataset_source,
        )

        source, _layout = resolve_workspace_dataset_source(inp)
        prepare_workspace_dataset_flat(source, ws_dataset)
        resolved = inp.resolve()
        roots = list(getattr(self.cfg, "path_guard_extra_roots", None) or [])
        if resolved not in {Path(p).expanduser().resolve(strict=False) for p in roots}:
            roots.append(resolved)
            self.cfg.path_guard_extra_roots = roots

    def _count_memory_records(self) -> int:
        p = self.memory_dir / "ScienceAgent" / "short_term.json"
        if not p.is_file():
            return 0
        try:
            return sum(1 for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
        except OSError:
            return 0

    def _workspace_metric_audit_source_text(self, *, source_rel: str, max_files: int = 40, max_total_chars: int = 120_000) -> str:
        """Collect bounded workspace Python source for metric-semantics audit."""
        parts: list[str] = []
        seen: set[Path] = set()

        def add(path: Path) -> None:
            nonlocal parts
            if len(seen) >= max_files:
                return
            try:
                resolved = path.resolve(strict=False)
            except OSError:
                resolved = path
            if resolved in seen or not path.is_file():
                return
            try:
                rel = path.relative_to(self.workspace_dir)
            except ValueError:
                rel = path.name
            rel_text = str(rel).replace("\\", "/")
            if rel_text.startswith((".venv/", "dataset/", "artifacts/", ".git/", ".logs/", "logs/", ".memory/")):
                return
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
            remaining = max_total_chars - sum(len(x) for x in parts)
            if remaining <= 0:
                return
            seen.add(resolved)
            parts.append(f"\n# FILE: {rel_text}\n{text[:remaining]}")

        add(self.workspace_dir / source_rel)
        for path in sorted(self.workspace_dir.glob("*.py")):
            add(path)
        return "\n".join(parts)

    @staticmethod
    def _compact_run_command(command: str, *, max_chars: int = 900) -> str:
        raw = str(command or "").strip()
        if not raw:
            return ""
        compact = normalize_shell_command(raw)
        compact = re.sub(r"\s+", " ", compact or raw).strip()
        if len(compact) > max_chars:
            return compact[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"
        return compact

    @classmethod
    def _stage_run_signature(cls, metric_event: dict[str, Any]) -> str:
        raw_command = re.sub(r"\s+", " ", str(metric_event.get("bash_cmd") or "")).strip()
        payload = {
            "solution_sha": str(metric_event.get("solution_sha") or ""),
            "submission_sha": str(metric_event.get("submission_sha") or ""),
            "artifact_sha": str(metric_event.get("artifact_sha") or ""),
            "metric_value": metric_event.get("metric_value"),
            "metric_name": str(metric_event.get("metric_name") or ""),
            "run_command": cls._compact_run_command(str(metric_event.get("bash_cmd") or "")),
            "raw_command": raw_command,
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()

    def _metric_event_from_workspace(self) -> dict[str, Any] | None:
        p = find_node_log_path(self.workspace_dir, "fullrun_tail_snapshot.json")
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            metric_value = float(data.get("metric_value"))
        except (TypeError, ValueError):
            if bool(self.lhr.stage_commit_require_metric):
                return None
            metric_value = None
        if metric_value is not None and (metric_value != metric_value):
            return None
        direction = self._metric_lower_is_better_decision(
            {
                "declared_lower_is_better": data.get("lower_is_better"),
                "lower_is_better": data.get("lower_is_better"),
                "metric_name": data.get("metric_name") or "Final Validation Score",
                "val_score_type": data.get("val_score_type"),
                "metric_protocol": data.get("metric_protocol"),
                "metric_source_note": data.get("metric_source_note"),
            },
            fallback=data.get("lower_is_better"),
        )
        lower_is_better = bool(direction["lower_is_better"])
        validation_ok = data.get("validation_ok")
        execution_mode_hint = str(data.get("execution_mode") or "").strip()
        source_rel = str(data.get("solution_path") or "").replace("\\", "/").strip().lstrip("/")
        if not source_rel:
            source_rel = "" if execution_mode_hint == "inline_bash" else "solution.py"
        if source_rel and ".." in source_rel.split("/"):
            source_rel = "solution.py"
        validation_issue = ""
        audit_source_text = self._workspace_metric_audit_source_text(source_rel=source_rel)
        if validation_ok is not False and bool(getattr(self.lhr, "metric_validation_leakage_guard_enabled", True)):
            if source_rel:
                try:
                    source_text = (self.workspace_dir / source_rel).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    source_text = ""
            else:
                source_text = audit_source_text
            reason = _validation_leakage_reason(source_text, str(data.get("stdout_tail") or ""))
            if reason:
                validation_issue = reason
                validation_ok = False
                self._jsonl(
                    "lhr_stage_events.jsonl",
                    {
                        "event": "stage_metric_rejected",
                        "reason": reason,
                        "metric_value": metric_value,
                        "metric_name": str(data.get("metric_name") or "Final Validation Score"),
                        "snapshot_path": str(p),
                    },
                )
        semantics = classify_metric_semantics(
            metric_value=metric_value,
            data=data,
            workspace_source_text=audit_source_text,
            source_changed=data.get("source_changed"),
        )
        if not semantics.get("route_id") and data.get("solution_sha"):
            semantics["route_id"] = f"source:{str(data.get('solution_sha'))[:16]}"
        wall_sec = data.get("wall_sec")
        duration_sec = data.get("duration_sec")
        run_time_sec = wall_sec if wall_sec not in (None, "") else duration_sec
        bash_cmd = str(data.get("bash_cmd") or "")
        event = {
            "metric_value": metric_value,
            "metric_name": str(data.get("metric_name") or "Final Validation Score"),
            "lower_is_better": lower_is_better,
            "declared_lower_is_better": direction.get("declared_lower_is_better"),
            "metric_direction_source": direction.get("metric_direction_source"),
            "metric_direction_conflict": direction.get("metric_direction_conflict"),
            "solution_sha": str(data.get("solution_sha") or ""),
            "solution_path": source_rel,
            "submission_sha": str(data.get("submission_sha") or ""),
            "bash_cmd": bash_cmd,
            "validation_ok": validation_ok,
            "validation_issue": validation_issue,
            "submission_validation_ok": data.get("submission_validation_ok"),
            "submission_status": str(data.get("submission_status") or ""),
            "execution_mode": execution_mode_hint,
            "wall_sec": wall_sec,
            "duration_sec": duration_sec,
            "run_time_sec": run_time_sec,
            "snapshot_path": str(p),
        }
        event.update(semantics)
        return event

    @staticmethod
    def _fmt_csv_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return f"{value:.6f}" if value == value else ""
        return str(value)

    @staticmethod
    def _csv_bool(value: Any, *, default: bool = True) -> bool:
        text = str(value if value is not None else "").strip().lower()
        if not text:
            return default
        if text in {"0", "false", "no", "n"}:
            return False
        if text in {"1", "true", "yes", "y"}:
            return True
        return default

    @staticmethod
    def _csv_metric(value: Any) -> float | None:
        try:
            metric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if metric != metric:
            return None
        return metric

    @staticmethod
    def _compact_peer_method(text: str, *, max_chars: int = 180) -> str:
        out = re.sub(r"\s+", " ", str(text or "")).strip()
        if not out:
            return "method not recorded"
        if len(out) > max_chars:
            out = out[: max_chars - 1].rstrip() + "…"
        return out

    def _magent_recommendations_for_resource_context(self) -> str:
        lhr_cfg = getattr(self, "lhr", None)
        if not bool(getattr(lhr_cfg, "estra_magent_enabled", False)):
            return ""
        if not bool(getattr(lhr_cfg, "estra_magent_join_inject_resource_context", True)):
            return ""
        runtime = getattr(getattr(self, "resource_observer", None), "resource_runtime", None)
        resource_dir = getattr(runtime, "resource_dir", None)
        if resource_dir is None:
            return ""
        try:
            packets = load_join_packets(resource_dir, parent_worker_id=self.worker_id or "W00", max_packets=3)
            return format_magent_recommendations(packets, max_items=3, max_chars=1200)
        except Exception:
            logger.debug("[lnr] estra magent recommendations failed", exc_info=True)
            return ""

    def _stage_payloads_for_score_summary(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for sid, snap in sorted(getattr(self, "stage_snapshots", {}).items()):
            source = snap.source_event if isinstance(getattr(snap, "source_event", None), dict) else {}
            payloads.append({"stage_id": sid, "metric_event": source})
        return payloads

    def _score_summary_for_prompt(self) -> dict[str, Any]:
        try:
            return build_stage_performance_score_summary(
                self.global_log_dir / LHR_STAGE_PERFORMANCE_CSV,
                current_worker_id=self._worker_uid_prefix(),
                fallback_stage_payloads=self._stage_payloads_for_score_summary(),
            )
        except Exception:
            logger.debug("[lnr] stage-performance score summary failed", exc_info=True)
            return {}

    def _allocated_compute_context_lines(self) -> list[str]:
        exec_cfg = getattr(getattr(self, "cfg", None), "exec", None)
        worker_env = self.worker_extra_env if isinstance(getattr(self, "worker_extra_env", None), dict) else {}
        worker_cpu = (
            str(worker_env.get("SCIENCEFLOW_WORKER_CPU_LIST") or "").strip()
            or str(worker_env.get("SCIENCEFLOW_CPU_LIST") or "").strip()
            or str(worker_env.get("_SCIENCEFLOW_CPU_SET") or "").strip()
            or os.environ.get("SCIENCEFLOW_WORKER_CPU_LIST", "").strip()
            or os.environ.get("_SCIENCEFLOW_CPU_SET", "").strip()
        )
        task_cpu = (
            str(worker_env.get("SCIENCEFLOW_TASK_CPU_LIST") or "").strip()
            or os.environ.get("SCIENCEFLOW_TASK_CPU_LIST", "").strip()
        )
        cfg_cpu = str(getattr(exec_cfg, "cpu_list", "") or "").strip()
        cpu_text = worker_cpu or task_cpu or cfg_cpu
        cpu_count = 0
        if cpu_text:
            try:
                cpu_count = len(parse_cpu_list(cpu_text))
            except Exception:
                logger.debug("[lnr] allocated compute CPU parse failed: %s", cpu_text, exc_info=True)
        omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()

        visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        task_gpu = os.environ.get("SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL", "").strip()
        cfg_gpu = str(getattr(exec_cfg, "gpu_list", "") or "").strip()
        gpu_text = visible_gpu or task_gpu or cfg_gpu

        if not any((cpu_text, task_cpu, cfg_cpu, omp_threads, gpu_text, task_gpu, cfg_gpu)):
            return []

        lines = ["allocated_compute:"]
        worker_id = self._worker_uid_prefix()
        if worker_id:
            lines.append(f"  - worker_id: {worker_id}")
        if cpu_text:
            label = "worker_cpu_list" if worker_cpu else "task_cpu_list"
            lines.append(f"  - {label}: {cpu_text}")
            if cpu_count > 0:
                lines.append(f"  - worker_cpu_count: {cpu_count}")
        if task_cpu and task_cpu != cpu_text:
            lines.append(f"  - task_cpu_list: {task_cpu}")
        if omp_threads:
            lines.append(f"  - thread_env: OMP_NUM_THREADS={omp_threads}")
        if gpu_text:
            label = "visible_gpu_list" if visible_gpu else "task_gpu_list"
            if gpu_text == "-1":
                lines.append(f"  - {label}: none")
            else:
                lines.append(f"  - {label}: {gpu_text}")
                if visible_gpu:
                    lines.append("  - cuda_index_note: use cuda:0 for the first visible assigned GPU")
        if cfg_gpu and cfg_gpu != gpu_text:
            lines.append(f"  - configured_gpu_list: {cfg_gpu}")
        lines.append("  - expectation: scale validated routes to use assigned CPU/GPU; if staying resource-light, state why")
        return lines

    def _resource_context_for_prompt(self) -> str:
        lhr_cfg = getattr(self, "lhr", None)
        if not bool(getattr(lhr_cfg, "resource_context_prompt_enabled", True)):
            return ""
        magent_block = self._magent_recommendations_for_resource_context()
        allocation_lines = self._allocated_compute_context_lines()
        state_machine = getattr(self, "state_machine", None)
        if state_machine is None:
            parts = [*allocation_lines]
            if magent_block:
                parts.append(magent_block)
            return "\n".join(parts).strip()
        try:
            state = state_machine._base_state()
        except Exception:
            logger.debug("[lnr] resource context snapshot failed", exc_info=True)
            parts = [*allocation_lines]
            if magent_block:
                parts.append(magent_block)
            return "\n".join(parts).strip()
        context_version = int(state.get("event_count") or 0)
        pressure_rows = [x for x in (state.get("resource_gpu_pressure_states") or []) if isinstance(x, dict)]
        pending = [x for x in (state.get("resource_pending_gpu_jobs") or []) if isinstance(x, dict)]
        active = [x for x in (state.get("resource_active_leases") or []) if isinstance(x, dict)]
        pressure_generation = 0
        runtime = getattr(getattr(self, "resource_observer", None), "resource_runtime", None)
        if runtime is not None:
            try:
                pressure_snapshot = runtime.pressure_store.snapshot()
                pressure_generation = int(pressure_snapshot.get("generation") or 0)
                global_pressure_rows = [
                    dict(row)
                    for row in (pressure_snapshot.get("gpus") or {}).values()
                    if isinstance(row, dict)
                ]
                if global_pressure_rows:
                    by_gpu = {str(row.get("gpu_id") or idx): row for idx, row in enumerate(pressure_rows)}
                    for row in global_pressure_rows:
                        by_gpu[str(row.get("gpu_id") or len(by_gpu))] = row
                    pressure_rows = list(by_gpu.values())
            except Exception:
                logger.debug("[lnr] global resource pressure snapshot failed", exc_info=True)
            try:
                active_snapshot = runtime.gpu_store.snapshot_active()
                leases = active_snapshot.get("leases") if isinstance(active_snapshot.get("leases"), dict) else {}
                waiters = active_snapshot.get("waiters") if isinstance(active_snapshot.get("waiters"), dict) else {}
                active_by_job = {str(row.get("job_id") or idx): row for idx, row in enumerate(active)}
                for job_id, raw in leases.items():
                    if not isinstance(raw, dict):
                        continue
                    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
                    active_by_job[str(job_id)] = {
                        "job_id": str(job_id),
                        "worker_id": str(raw.get("worker_id") or ""),
                        "resource_class": str(meta.get("policy_resource_class") or meta.get("resource_class") or ""),
                        "gpu_ids": [str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()],
                        "slot_weight": meta.get("slot_weight"),
                    }
                active = list(active_by_job.values())
                pending_by_job = {str(row.get("job_id") or idx): row for idx, row in enumerate(pending)}
                for job_id, raw in waiters.items():
                    if not isinstance(raw, dict):
                        continue
                    pending_by_job[str(job_id)] = {
                        "job_id": str(job_id),
                        "worker_id": str(raw.get("worker_id") or ""),
                        "resource_class": str(raw.get("resource_class") or ""),
                        "gpu_ids": [str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()],
                        "queue_position": raw.get("queue_position"),
                        "slot_weight": raw.get("slot_weight"),
                    }
                pending = list(pending_by_job.values())
            except Exception:
                logger.debug("[lnr] global resource lease snapshot failed", exc_info=True)
        try:
            if self.resource_observer is not None:
                self.resource_observer.set_planning_resource_context_version(
                    context_version,
                    pressure_generation=pressure_generation,
                )
        except Exception:
            logger.debug("[lnr] resource context version sync failed", exc_info=True)
        policy_gates = int(state.get("resource_policy_gates") or 0)
        queue_timeouts = int(state.get("resource_gpu_queue_timeouts") or 0)
        admission_deferred = int(state.get("resource_admission_deferred") or 0)
        resource_last_rows = [x for x in (state.get("resource_last_events") or []) if isinstance(x, dict)]
        score_summary = self._score_summary_for_prompt()
        has_score_context = bool((score_summary.get("best_score") or {}) or score_summary.get("cheap_signal_available"))
        last_feedback = None
        last_boundary_event = None
        for row in reversed(resource_last_rows):
            action = str(row.get("post_feedback_action") or "").lower()
            reason = str(row.get("reason") or "").lower()
            if last_boundary_event is None and (action == "stop_boundary_violation" or "boundary_violation" in reason):
                last_boundary_event = row
            if row.get("feedback_status") or row.get("resource_mode") or row.get("allowed_classes"):
                last_feedback = row
                if last_boundary_event is not None:
                    break
        if (
            not pressure_rows
            and not pending
            and not active
            and policy_gates <= 0
            and queue_timeouts <= 0
            and admission_deferred <= 0
            and last_feedback is None
            and last_boundary_event is None
            and not magent_block
            and not has_score_context
            and not allocation_lines
        ):
            return ""
        modes = {str(row.get("mode") or "").upper() for row in pressure_rows}
        feedback_mode = str((last_feedback or {}).get("resource_mode") or "").upper()
        feedback_status = str((last_feedback or {}).get("feedback_status") or "").upper()
        if "RED" in modes or feedback_mode == "RED" or policy_gates > 0 or queue_timeouts > 0:
            mode = "RED"
        elif pending or admission_deferred > 0 or feedback_mode == "YELLOW" or feedback_status in {"DEFERRED", "PENDING"}:
            mode = "YELLOW"
        else:
            mode = "GREEN"
        allowed = []
        if last_feedback and isinstance(last_feedback.get("allowed_classes"), list):
            allowed = [str(x) for x in last_feedback.get("allowed_classes") if str(x).strip()]
        if not allowed:
            if mode == "RED":
                allowed = ["pure_tt_cpu", "gpu_tt_light", "readonly_cpu", "light_cpu"]
            elif mode == "YELLOW":
                allowed = ["pure_tt_cpu", "gpu_tt_light", "readonly_cpu", "light_cpu", "one_deferred_train"]
            else:
                allowed = ["heavy_gpu_train", "gpu_feature_extract", "gpu_tt_light", "pure_tt_cpu"]
        blocked = []
        blocked_class = str((last_feedback or {}).get("blocked_class") or "").strip()
        if mode == "RED":
            blocked = ["heavy_gpu_train", "heavy_gpu_candidate", "unknown_gpu_exec"]
        elif mode == "YELLOW":
            blocked = ["duplicate_failed_digest", "unknown_gpu_exec", "low_value_repeat_train"]
            if feedback_status in {"DEFERRED", "PENDING", "REPLAN", "DENIED_REPLAN"} and blocked_class:
                blocked.insert(0, blocked_class)
                if blocked_class == "heavy_gpu_candidate":
                    blocked.insert(1, "heavy_gpu_train")
        cooldown = max([float(row.get("cooldown_remaining_sec") or row.get("cooldown_sec") or 0.0) for row in pressure_rows] or [0.0])
        eta = cooldown + 300.0 if mode == "RED" else 0.0
        if last_feedback and last_feedback.get("eta_next_train_sec") is not None:
            try:
                eta = max(0.0, float(last_feedback.get("eta_next_train_sec") or 0.0))
            except (TypeError, ValueError):
                pass
        lines = [
            *allocation_lines,
            f"resource_context_version: {context_version}",
            f"observed_at_utc: {state.get('generated_at_utc') or ''}",
            "pressure_scope: gpu",
            f"pressure_generation: {pressure_generation}",
            f"resource_mode: {mode}",
            f"policy_gates: {policy_gates}",
            f"queue_timeouts: {queue_timeouts}",
            f"admission_pending_or_replan: {admission_deferred}",
        ]
        if has_score_context:
            lines.extend(format_score_summary_context(score_summary))
        if pressure_rows:
            lines.append("gpu_pressure:")
            for row in pressure_rows[:4]:
                gpu_id = str(row.get("gpu_id") or "")
                row_mode = str(row.get("mode") or "")
                qtc = int(row.get("queue_timeout_count") or 0)
                cd = max(0.0, float(row.get("cooldown_remaining_sec") or 0.0))
                reason = str(row.get("last_reason") or row.get("last_event") or "")
                lines.append(f"  - gpu_id: {gpu_id}; mode: {row_mode}; queue_timeout_count: {qtc}; cooldown_sec: {cd:.0f}; reason: {reason}")
        if active:
            lines.append("active_leases:")
            for row in active[:4]:
                lines.append(
                    "  - worker_id: "
                    f"{row.get('worker_id') or ''}; class: {row.get('resource_class') or ''}; "
                    f"gpu_ids: {','.join(str(x) for x in (row.get('gpu_ids') or []))}; "
                    f"slot_weight: {row.get('slot_weight') or ''}"
                )
        if pending:
            lines.append("pending_gpu_jobs:")
            for row in pending[:4]:
                lines.append(
                    "  - worker_id: "
                    f"{row.get('worker_id') or ''}; class: {row.get('resource_class') or ''}; "
                    f"gpu_ids: {','.join(str(x) for x in (row.get('gpu_ids') or []))}"
                )
        lines.append("allowed_classes:")
        lines.extend(f"  - {x}" for x in allowed[:8])
        if blocked:
            lines.append("blocked_classes:")
            lines.extend(f"  - {x}" for x in blocked[:8])
        if last_boundary_event:
            actual = ",".join(str(x) for x in (last_boundary_event.get("actual_gpu_ids") or []) if str(x).strip()) or "unknown"
            allowed_gpu = ",".join(str(x) for x in (last_boundary_event.get("allowed_gpu_ids") or []) if str(x).strip()) or "unknown"
            lines.append("recent_boundary_event:")
            lines.append(
                "  - action: {action}; reason: {reason}; actual_gpu_ids: {actual}; "
                "allowed_gpu_ids: {allowed}; command_digest: {digest}".format(
                    action=str(last_boundary_event.get("post_feedback_action") or ""),
                    reason=str(last_boundary_event.get("reason") or ""),
                    actual=actual,
                    allowed=allowed_gpu,
                    digest=str(last_boundary_event.get("command_digest") or ""),
                )
            )
        if cooldown > 0:
            lines.append(f"cooldown_sec: {cooldown:.0f}")
        if eta > 0:
            lines.append(f"eta_next_train_sec: {eta:.0f}")
            lines.append(f"eta_confidence: {str((last_feedback or {}).get('eta_confidence') or 'low')}")
            eta_source = str((last_feedback or {}).get("eta_source") or "")
            if eta_source:
                lines.append(f"eta_source: {eta_source}")
            if (last_feedback or {}).get("runtime_history_count") is not None:
                lines.append(f"runtime_history_count: {last_feedback.get('runtime_history_count')}")
        if last_feedback and last_feedback.get("admission_priority_score") is not None:
            lines.append(f"admission_priority_score: {last_feedback.get('admission_priority_score')}")
            if last_feedback.get("expected_value_score") is not None:
                lines.append(f"expected_value_score: {last_feedback.get('expected_value_score')}")
            if last_feedback.get("near_submission_score") is not None:
                lines.append(f"near_submission_score: {last_feedback.get('near_submission_score')}")
        lines.append("resource_feedback_contract: facts_and_constraint_logic_only; main_agent_chooses_research_route")
        if magent_block:
            lines.append(magent_block)
        lines.append("long_command_heartbeat: for any command, script phase, loop, or optimization process expected to run >2 minutes, print one flushed SCIENCEFLOW_HB stdout line every 30-60s")
        lines.append("heartbeat_scope: cover data prep, search/optimization, evaluation, inference, aggregation, export, and artifact writing; use metric=na when no metric exists")
        lines.append("heartbeat_format: SCIENCEFLOW_HB v=1 phase=<phase> tick=<n> elapsed_s=<sec> progress=<done>/<total_or_unknown> unit=<unit> metric=<name:value_or_na> loss=<value_or_na> artifact=<path_or_none>")
        lines.append("heartbeat_artifact: when a deliverable is written, print artifact=<path> in the heartbeat")
        lines.append("gpu_start_decision_source: use ResourceContext/RESOURCE_FEEDBACK lease status; nvidia-smi and torch.cuda probes are advisory")
        lines.append("resource_intent_hint: prefix clear bash commands with SCIENCEFLOW_RESOURCE_INTENT=cpu_support|gpu_train|light_train|gpu_tt|readonly_cpu; GPU evidence overrides CPU hints")
        lines.append("stale_resource_context_behavior: if preflight returns new RESOURCE_FEEDBACK, treat it as truth and replan")
        return "\n".join(lines)

    def _parallel_worker_snapshot_for_prompt(self) -> str:
        if not bool(getattr(self.lhr, "worker_peer_summary_enabled", True)):
            return ""
        worker_count = max(1, int(getattr(self, "worker_count", 1) or 1))
        if worker_count <= 1:
            return ""
        max_chars = max(600, int(getattr(self.lhr, "worker_peer_summary_max_chars", 2200) or 2200))
        current_worker = self._worker_uid_prefix()
        path = Path(getattr(self, "global_log_dir", getattr(self, "log_dir", Path(".")))) / LHR_STAGE_PERFORMANCE_CSV
        raw_best_by_worker: dict[str, dict[str, Any]] = {}
        ok_best_by_worker: dict[str, dict[str, Any]] = {}

        def row_is_better(row: dict[str, Any], previous: dict[str, Any] | None, *, lower_is_better: bool) -> bool:
            metric = self._csv_metric(row.get("metric_value"))
            if metric is None:
                return False
            if previous is None:
                return True
            previous_metric = self._csv_metric(previous.get("metric_value"))
            if previous_metric is None:
                return True
            return metric < previous_metric if lower_is_better else metric > previous_metric

        def row_metric_text(row: dict[str, Any]) -> str:
            metric = self._csv_metric(row.get("metric_value"))
            return f"{metric:.6f}" if metric is not None else str(row.get("metric_value") or "unknown")

        def row_node_label(row: dict[str, Any]) -> str:
            stage = str(row.get("stage_id") or "").strip() or "unknown"
            lineage = str(row.get("lineage_id") or "").strip()
            return f"{lineage}:{stage}" if lineage else stage

        def row_method(row: dict[str, Any]) -> str:
            return self._compact_peer_method(str(row.get("brief") or row.get("why") or ""))

        def row_elapsed_text(row: dict[str, Any]) -> str:
            try:
                elapsed = float(str(row.get("elapsed_min") or "").strip())
            except (TypeError, ValueError):
                elapsed = -1.0
            if elapsed >= 0:
                return f"{elapsed:.1f}m"
            try:
                sec = float(str(row.get("solution_run_sec") or "").strip())
            except (TypeError, ValueError):
                sec = -1.0
            if sec >= 0:
                return f"{sec / 60.0:.1f}m"
            return "unknown"

        def same_candidate(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
            if a is None or b is None:
                return False
            aid = str(a.get("candidate_id") or "").strip()
            bid = str(b.get("candidate_id") or "").strip()
            if aid or bid:
                return aid == bid
            return (
                str(a.get("worker_id") or "") == str(b.get("worker_id") or "")
                and str(a.get("lineage_id") or "") == str(b.get("lineage_id") or "")
                and str(a.get("stage_id") or "") == str(b.get("stage_id") or "")
            )

        if path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    rows = list(csv.DictReader(f))
                task_hint = getattr(self, "_task_metric_lower_is_better", None)
                lower_for_rows = bool(task_hint) if task_hint is not None else infer_stage_rows_lower_is_better(rows)
                for row in rows:
                    worker_id = str(row.get("worker_id") or "").strip() or "W00"
                    if self._csv_metric(row.get("metric_value")) is None:
                        continue
                    if row_is_better(row, raw_best_by_worker.get(worker_id), lower_is_better=lower_for_rows):
                        raw_best_by_worker[worker_id] = row
                    if self._csv_bool(row.get("validation_ok"), default=True) is False:
                        continue
                    if row_is_better(row, ok_best_by_worker.get(worker_id), lower_is_better=lower_for_rows):
                        ok_best_by_worker[worker_id] = row
            except OSError:
                logger.debug("[lnr] could not read worker peer summary csv", exc_info=True)
        lines = [
            "Parallel worker snapshot:",
            f"- There are {worker_count} workers exploring this task independently; use peer progress as high-level search context.",
        ]
        for idx in range(worker_count):
            worker_id = f"W{idx:02d}"
            label = f"{worker_id} (this worker)" if worker_id == current_worker else worker_id
            raw = raw_best_by_worker.get(worker_id)
            ok = ok_best_by_worker.get(worker_id)
            if raw is None:
                lines.append(f"- {label}: no metric-backed stage recorded yet.")
                continue
            if same_candidate(raw, ok):
                lines.append(
                    f"- {label}: best_metric={row_metric_text(raw)} (ok) at {row_node_label(raw)}; "
                    f"elapsed={row_elapsed_text(raw)}; method={row_method(raw)}"
                )
                continue
            issue = self._compact_peer_method(str(raw.get("validation_issue") or "validation_ok=false"), max_chars=100)
            line = (
                f"- {label}: best_metric={row_metric_text(raw)} at {row_node_label(raw)}; "
                f"elapsed={row_elapsed_text(raw)}; status=suspicious; issue={issue}; method={row_method(raw)}"
            )
            if ok is None:
                line += "; ok_best_metric=none"
            else:
                line += (
                    f"; ok_best_metric={row_metric_text(ok)} at {row_node_label(ok)}; "
                    f"ok_elapsed={row_elapsed_text(ok)}; ok_method={row_method(ok)}"
                )
            lines.append(line)
        summary = "\n".join(lines).strip()
        if len(summary) > max_chars:
            summary = summary[: max_chars - 38].rstrip() + "\n... [worker snapshot truncated]"
        return summary

    def _estra_peer_route_evidence(self) -> PeerRouteEvidence:
        if not bool(getattr(self.lhr, "estra_reflection_prompt_enabled", True)):
            return PeerRouteEvidence(text="")
        if not bool(getattr(self.lhr, "estra_peer_evidence_enabled", True)):
            return PeerRouteEvidence(text="")
        worker_count = max(1, int(getattr(self, "worker_count", 1) or 1))
        if worker_count <= 1:
            return PeerRouteEvidence(text="")
        path = Path(getattr(self, "global_log_dir", getattr(self, "log_dir", Path(".")))) / LHR_STAGE_PERFORMANCE_CSV
        return build_peer_route_evidence_from_csv(
            path,
            current_worker_id=self._worker_uid_prefix(),
            max_chars=max(400, int(getattr(self.lhr, "estra_peer_evidence_max_chars", 1400) or 1400)),
            min_delta_ratio=float(getattr(self.lhr, "estra_peer_evidence_min_delta_ratio", 0.0) or 0.0),
        )

    def _estra_backtrack_reflection(
        self,
        switch_candidates: list[str],
        *,
        latest_stage: str,
    ) -> tuple[str, dict[str, Any]]:
        if not bool(getattr(self.lhr, "estra_reflection_prompt_enabled", True)):
            return "", {"backtrack_reflection_present": False, "backtrack_candidate_count": 0}
        if not bool(getattr(self.lhr, "estra_backtrack_reflection_enabled", True)):
            return "", {"backtrack_reflection_present": False, "backtrack_candidate_count": 0}
        candidates = [str(sid or "").strip().upper() for sid in switch_candidates if str(sid or "").strip()]
        ranked: list[tuple[float, str, StageSnapshot, str]] = []
        latest_snap = self.stage_snapshots.get(str(latest_stage or "").strip().upper())
        lower = True
        latest_lower = getattr(latest_snap, "lower_is_better", None) if latest_snap is not None else None
        if latest_lower is not None:
            lower = bool(latest_lower)
        for sid in candidates:
            snap = self.stage_snapshots.get(sid)
            metric_value = getattr(snap, "metric_value", None) if snap is not None else None
            if snap is None or metric_value is None:
                continue
            raw_source = getattr(snap, "source_event", {})
            source = raw_source if isinstance(raw_source, dict) else {}
            if self._csv_bool(source.get("validation_ok"), default=True) is False:
                continue
            if str(source.get("metric_validity") or "").strip().lower() == "low":
                continue
            selection_eligible = source.get("selection_eligible")
            if selection_eligible not in (None, "") and self._csv_bool(selection_eligible, default=True) is False:
                continue
            snap_lower = getattr(snap, "lower_is_better", None)
            if snap_lower is not None:
                lower = bool(snap_lower)
            score = float(metric_value) if lower else -float(metric_value)
            status_bits = []
            metric_validity = str(source.get("metric_validity") or "").strip()
            if metric_validity:
                status_bits.append(f"metric_validity={metric_validity}")
            if selection_eligible not in (None, ""):
                status_bits.append(f"selection_eligible={self._stage_card_override_text(selection_eligible)}")
            if source.get("submission_sha"):
                status_bits.append("submission_ready=True")
            if source.get("source_commit_sha"):
                status_bits.append(f"commit={str(source.get('source_commit_sha'))[:12]}")
            ranked.append((score, sid, snap, ",".join(status_bits) or "metric-backed"))
        ranked.sort(key=lambda item: item[0])
        max_chars = max(400, int(getattr(self.lhr, "estra_backtrack_reflection_max_chars", 1000) or 1000))
        meta: dict[str, Any] = {
            "backtrack_reflection_present": False,
            "backtrack_candidate_count": len(ranked),
            "backtrack_best_stage": ranked[0][1] if ranked else "",
            "backtrack_best_metric": ranked[0][2].metric_value if ranked else None,
        }
        if not ranked:
            return "", meta
        lines = [
            "- Previous-stage restore is a normal research action when an older stage is a cleaner base.",
        ]
        if latest_snap is not None and latest_snap.metric_value is not None:
            lines.append(f"- Current/latest {latest_stage}: metric={latest_snap.metric_value}.")
        for _score, sid, snap, status in ranked[:3]:
            source = snap.source_event if isinstance(snap.source_event, dict) else {}
            method = self._compact_peer_method(str(source.get("brief") or source.get("why") or ""), max_chars=120)
            lines.append(f"- Candidate {sid}: metric={snap.metric_value}; status={status}; method={method}")
        lines.append("- Compare current continue, current redirect, and previous_stage redirect before choosing.")
        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 39].rstrip() + "\n... [backtrack reflection truncated]"
        meta["backtrack_reflection_present"] = bool(text)
        return text, meta

    def _next_global_stage_row_order(self, path: Path) -> int:
        if not path.is_file():
            return 1
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                return sum(1 for _ in csv.DictReader(f)) + 1
        except OSError:
            return 1

    def _best_stage_id_so_far(self) -> str:
        best_stage = ""
        best_metric: float | None = None
        lower = True
        for sid, snap in self.stage_snapshots.items():
            if snap.metric_value is None:
                continue
            if best_metric is None:
                best_metric = float(snap.metric_value)
                best_stage = sid
                lower = bool(snap.lower_is_better is not False)
                continue
            value = float(snap.metric_value)
            if lower and value < best_metric:
                best_metric = value
                best_stage = sid
            elif not lower and value > best_metric:
                best_metric = value
                best_stage = sid
        return best_stage

    @staticmethod
    def _stage_with_submission_sha(
        stage_snapshots: dict[str, StageSnapshot],
        submission_sha: str,
    ) -> StageSnapshot | None:
        needle = str(submission_sha or "").strip()
        if not needle:
            return None
        for _sid, snap in sorted(stage_snapshots.items()):
            source = snap.source_event if isinstance(snap.source_event, dict) else {}
            if str(source.get("submission_sha") or "").strip() == needle:
                return snap
        return None

    @staticmethod
    def _stage_with_artifact_sha(
        stage_snapshots: dict[str, StageSnapshot],
        artifact_sha: str,
    ) -> StageSnapshot | None:
        needle = str(artifact_sha or "").strip()
        if not needle:
            return None
        for _sid, snap in sorted(stage_snapshots.items()):
            source = snap.source_event if isinstance(snap.source_event, dict) else {}
            if str(source.get("artifact_sha") or source.get("submission_sha") or "").strip() == needle:
                return snap
        return None

    @staticmethod
    def _duplicate_stage_same_design(metric_event: dict[str, Any], duplicate: StageSnapshot) -> bool:
        source = duplicate.source_event if isinstance(duplicate.source_event, dict) else {}
        current_solution = str(metric_event.get("solution_sha") or "").strip()
        prior_solution = str(source.get("solution_sha") or "").strip()
        current_artifact = str(
            metric_event.get("artifact_sha")
            or metric_event.get("submission_sha")
            or ""
        ).strip()
        prior_artifact = str(
            source.get("artifact_sha")
            or source.get("submission_sha")
            or ""
        ).strip()
        if current_solution and prior_solution and current_solution != prior_solution:
            return False
        if current_artifact and prior_artifact and current_artifact != prior_artifact:
            return False
        return bool(current_solution and prior_solution and current_artifact and prior_artifact)


    def _metric_lower_is_better_decision(self, metric_event: dict[str, Any], *, fallback: Any = None) -> dict[str, Any]:
        declared_raw = metric_event.get("declared_lower_is_better")
        if declared_raw in (None, ""):
            declared_raw = metric_event.get("lower_is_better")
        declared = self._csv_bool(declared_raw, default=None)
        fallback_declared = self._csv_bool(fallback, default=None)
        if declared is None:
            declared = fallback_declared
        hint_text = " ".join(
            str(metric_event.get(key) or "")
            for key in ("metric_name", "val_score_type", "metric_protocol", "metric_source_note")
        )
        event_hint = metric_lower_is_better_hint(hint_text)
        task_hint = getattr(self, "_task_metric_lower_is_better", None)
        if metric_event.get("metric_authoritative") is True and declared is not None:
            lower = bool(declared)
            source = "authoritative_evaluator"
        elif task_hint is not None:
            lower = bool(task_hint)
            source = "task_description"
        elif event_hint is not None:
            lower = bool(event_hint)
            source = "metric_text_hint"
        elif declared is not None:
            lower = bool(declared)
            source = "declared"
        else:
            lower = True
            source = "default_lower"
        conflict = bool(declared is not None and bool(declared) != lower)
        if event_hint is not None and bool(event_hint) != lower:
            conflict = True
        return {
            "lower_is_better": lower,
            "metric_direction_source": source,
            "declared_lower_is_better": declared if declared is not None else "",
            "metric_direction_conflict": conflict,
        }

    def _apply_metric_direction_audit(self, metric_event: dict[str, Any], *, fallback: Any = None) -> dict[str, Any]:
        decision = self._metric_lower_is_better_decision(metric_event, fallback=fallback)
        metric_event.update(decision)
        return decision

    def _metric_lower_is_better_for_event(self, metric_event: dict[str, Any], *, fallback: Any = None) -> bool:
        return bool(self._metric_lower_is_better_decision(metric_event, fallback=fallback)["lower_is_better"])


    @staticmethod
    def _metric_source_note(metric_event: dict[str, Any], *, brief: str = "", why: str = "") -> str:
        parts: list[str] = []
        for key, label in (
            ("val_score_type", "type"),
            ("metric_protocol", "protocol"),
            ("metric_eval_data", "eval"),
            ("train_data_used", "train_data"),
            ("execution_mode", "execution"),
        ):
            value = str(metric_event.get(key) or "").strip()
            if value:
                parts.append(f"{label}={value}")
        for key, label in (("selection_note", "note"), ("validation_issue", "validation_issue")):
            value = str(metric_event.get(key) or "").strip()
            if value:
                parts.append(f"{label}={value[:160]}")
        risk_text = " ".join(
            str(x or "")
            for x in (
                metric_event.get("selection_note"),
                metric_event.get("validation_issue"),
                brief,
                why,
            )
        ).lower()
        risk_tags: list[str] = []
        if any(term in risk_text for term in ("target leak", "target-leaking", "leaky", "leakage risk", "data leakage")):
            risk_tags.append("leakage_suspected")
        if any(term in risk_text for term in ("overfit", "overfitting", "unreliable", "not generalizable", "inflated")):
            risk_tags.append("overfit_suspected")
        if any(term in risk_text for term in ("benign_malignant", "diagnosis", "pat_malig", "pat_mal", "mal_ratio", "pos_rate", "positive rate", "target-derived", "patient positive")):
            risk_tags.append("target_derived_feature_suspected")
        if any(term in risk_text for term in ("trained and evaluated on the same", "evaluated on the same", "same validation set", "meta-learner overfit")):
            risk_tags.append("same_validation_overfit_suspected")
        if any(term in risk_text for term in ("honest estimate is", "honest score would", "real honest score")):
            risk_tags.append("reported_metric_differs_from_honest_estimate")
        if risk_tags:
            parts.append("risk=" + ",".join(dict.fromkeys(risk_tags)))
        return "; ".join(parts)[:500]

    def _metric_validity_card_fields(self, cards_after: list[Any], stage_id: str) -> dict[str, Any]:
        for card in cards_after:
            if getattr(card, "stage_id", "") == stage_id:
                return {
                    "metric_validity": getattr(card, "metric_validity", ""),
                    "brief": getattr(card, "brief", ""),
                    "why": getattr(card, "why", ""),
                    "route_evidence": getattr(card, "route_evidence", ""),
                }
        return {}

    def _task_metric_context_excerpt(self, *, max_chars: int = 1200) -> str:
        task = str(getattr(self, "task_desc", "") or "")
        if not task:
            return ""
        lines = []
        for line in task.splitlines():
            lowered = line.lower()
            if any(token in lowered for token in ("metric", "evaluation", "score", "higher", "lower", "kendall", "auc", "rmse", "dice")):
                compact = re.sub(r"\s+", " ", line).strip()
                if compact:
                    lines.append(compact)
        excerpt = "\n".join(lines) if lines else task[:max_chars]
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"
        return excerpt

    @staticmethod
    def _metric_validity_snapshot_tail(metric_event: dict[str, Any]) -> dict[str, str]:
        path = Path(str(metric_event.get("snapshot_path") or ""))
        if not path.is_file():
            return {"stdout_tail": "", "stderr_tail": ""}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {"stdout_tail": "", "stderr_tail": ""}
        return {
            "stdout_tail": str(data.get("stdout_tail") or "")[-5000:],
            "stderr_tail": str(data.get("stderr_tail") or "")[-2000:],
        }

    def _metric_validity_fact_card(
        self,
        *,
        metric_event: dict[str, Any],
        card_fields: dict[str, Any],
    ) -> dict[str, Any]:
        keys = (
            "metric_value",
            "metric_name",
            "lower_is_better",
            "declared_lower_is_better",
            "declared_stage_commit_lower_is_better",
            "metric_direction_source",
            "metric_direction_conflict",
            "validation_ok",
            "validation_issue",
            "submission_validation_ok",
            "submission_status",
            "val_score_type",
            "selection_eligible",
            "selection_score",
            "selection_note",
            "metric_source_note",
            "metric_validity",
            "metric_validity_note",
            "task_profile",
            "metric_protocol",
            "train_data_used",
            "metric_eval_data",
            "execution_mode",
            "candidate_ready",
            "metric_authoritative",
        )
        facts = {key: metric_event.get(key) for key in keys if key in metric_event}
        facts["stage_card"] = {
            key: str(card_fields.get(key) or "")[:1200]
            for key in ("metric_validity", "brief", "why", "route_evidence")
        }
        facts.update(self._metric_validity_snapshot_tail(metric_event))
        task_hint = getattr(self, "_task_metric_lower_is_better", None)
        if task_hint is not None:
            facts["task_metric_lower_is_better"] = bool(task_hint)
            facts["task_metric_direction_source"] = "task_description"
        excerpt = self._task_metric_context_excerpt()
        if excerpt:
            facts["task_metric_context_excerpt"] = excerpt
        if facts.get("stdout_tail"):
            facts["stdout_tail"] = str(facts["stdout_tail"])[-5000:]
        if facts.get("stderr_tail"):
            facts["stderr_tail"] = str(facts["stderr_tail"])[-2000:]
        return facts

    def _metric_feedback_llm_client(self) -> Any | None:
        metric_llm = getattr(self, "_metric_validity_feedback_llm", None)
        if metric_llm is not None:
            return metric_llm
        cfg = getattr(self, "cfg", None)
        feedback_cfg = getattr(getattr(cfg, "agent", None), "feedback", None)
        if feedback_cfg is None:
            return None
        metric_llm = _build_llm(feedback_cfg)
        self._metric_validity_feedback_llm = metric_llm
        return metric_llm

    async def _metric_output_interpretation_callback(
        self,
        *,
        agent: Any,
        stdout: str,
        script_label: str,
    ) -> dict[str, Any] | None:
        if not bool(getattr(self.lhr, "metric_validity_adjudicator_enabled", True)):
            return None
        metric_llm = self._metric_feedback_llm_client()
        if metric_llm is None or not hasattr(metric_llm, "ask_tool_stream"):
            return None

        lines = str(stdout or "").splitlines()
        stdout_tail = "\n".join(lines[-160:])[-12000:]
        candidate_lines = [
            line
            for line in lines
            if re.search(
                r"(?<![A-Za-z0-9])"
                r"(?:best|final|val(?:idation)?|hold[ _-]?out|cv|oof|metric|score)"
                r"(?![A-Za-z0-9])",
                line,
                re.IGNORECASE,
            )
            and re.search(r"\d", line)
        ]
        metric_candidates = "\n".join(candidate_lines[-80:])[-8000:]
        facts = {
            "script_label": str(script_label or "solution.py")[:300],
            "task_metric_context_excerpt": self._task_metric_context_excerpt(),
            "metric_candidate_lines": metric_candidates,
            "stdout_tail": stdout_tail,
        }
        prompt = build_metric_output_interpreter_prompt(facts)
        t0 = time.time()
        try:
            assistant_msg = await self._ask_agent_tool_stream_guarded(
                agent,
                llm=metric_llm,
                messages=[Message.user_message(prompt)],
                system_msgs=[Message.system_message(build_metric_output_interpreter_system_prompt())],
                timeout=max(
                    1.0,
                    float(getattr(self.lhr, "metric_validity_adjudicator_timeout_sec", 60.0) or 60.0),
                ),
                tools=[],
                tool_choice="none",
                parallel_tool_calls=False,
                collect_all_tool_calls=True,
            )
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_metric_output_interpreter",
                    time.time() - t0,
                    None,
                    "ok",
                    recovery=True,
                    turn_kind="metric_interpretation",
                    llm_override=metric_llm,
                    llm_role="feedback",
                )
            self._accumulate_ephemeral_tokens(agent, "stage", llm=metric_llm)
            text = (getattr(assistant_msg, "content", None) or "").strip()
            if not text:
                text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
            interpretation = parse_metric_output_interpretation_text(text)
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_output_interpreter_response",
                    "parsed": interpretation is not None,
                    "script_label": facts["script_label"],
                    "response": text[:1600],
                },
            )
            if interpretation is None:
                return None
            return {
                "metric_found": interpretation.metric_found,
                "metric_name": interpretation.metric_name,
                "metric_value": interpretation.metric_value,
                "split": interpretation.split,
                "is_final": interpretation.is_final,
                "evidence_line": interpretation.evidence_line,
                "confidence": interpretation.confidence,
                "reason": interpretation.reason,
            }
        except Exception as exc:
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_metric_output_interpreter",
                    time.time() - t0,
                    None,
                    "error",
                    recovery=True,
                    turn_kind="metric_interpretation",
                    llm_override=metric_llm,
                    llm_role="feedback",
                )
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_output_interpreter_error",
                    "script_label": facts["script_label"],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return None

    async def _metric_validity_feedback_judgment(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        card_fields: dict[str, Any],
        scope: str,
    ) -> Any | None:
        if (
            metric_event.get("metric_authoritative") is True
            and str(metric_event.get("evaluator_status") or "").strip().lower() == "ok"
            and metric_event.get("validation_ok") is True
        ):
            return None
        llm_enabled = bool(getattr(self.lhr, "metric_validity_adjudicator_enabled", True))
        if not llm_enabled:
            return None
        metric_llm = self._metric_feedback_llm_client()
        if metric_llm is None or not hasattr(metric_llm, "ask_tool_stream"):
            return None
        prompt = build_metric_validity_adjudicator_prompt(
            self._metric_validity_fact_card(metric_event=metric_event, card_fields=card_fields)
        )
        t0 = time.time()
        try:
            assistant_msg = await self._ask_agent_tool_stream_guarded(
                agent,
                llm=metric_llm,
                messages=[Message.user_message(prompt)],
                system_msgs=[Message.system_message(build_metric_validity_adjudicator_system_prompt())],
                timeout=max(1.0, float(getattr(self.lhr, "metric_validity_adjudicator_timeout_sec", 60.0) or 60.0)),
                tools=[],
                tool_choice="none",
                parallel_tool_calls=False,
                collect_all_tool_calls=True,
            )
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_metric_validity_adjudicator",
                    time.time() - t0,
                    None,
                    "ok",
                    recovery=True,
                    turn_kind="metric_validity",
                    llm_override=metric_llm,
                    llm_role="feedback",
                )
            self._accumulate_ephemeral_tokens(agent, "stage", llm=metric_llm)
            text = (getattr(assistant_msg, "content", None) or "").strip()
            if not text:
                text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
            judgment = parse_metric_validity_judgment_text(text)
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_validity_adjudicator_response",
                    "stage_id": stage_id,
                    "scope": scope,
                    "parsed": bool(judgment),
                    "response": text[:1200],
                },
            )
            return judgment
        except BaseException as exc:
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_metric_validity_adjudicator",
                    time.time() - t0,
                    None,
                    "error",
                    recovery=True,
                    turn_kind="metric_validity",
                    llm_override=metric_llm,
                    llm_role="feedback",
                )
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_validity_adjudicator_error",
                    "stage_id": stage_id,
                    "scope": scope,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return None

    @staticmethod
    def _stage_commit_card_fields_from_judgment(judgment: dict[str, Any] | None) -> dict[str, Any]:
        data = judgment if isinstance(judgment, dict) else {}
        return {
            "metric_validity": data.get("metric_validity") or "",
            "brief": data.get("brief") or data.get("BRIEF") or "",
            "why": data.get("why") or data.get("WHY") or "",
            "route_evidence": data.get("route_evidence") or data.get("routeEvidence") or data.get("route") or "",
            "metric_note": data.get("metric_source") or data.get("metric_note") or "",
        }

    async def _audit_stage_result_before_commit(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        judgment: dict[str, Any] | None,
        block_text: str,
        source: str,
    ) -> None:
        before = {
            "lower_is_better": metric_event.get("lower_is_better"),
            "metric_validity": metric_event.get("metric_validity"),
            "selection_eligible": metric_event.get("selection_eligible"),
        }
        data = judgment if isinstance(judgment, dict) else {}
        if data.get("lower_is_better") not in (None, ""):
            metric_event["declared_stage_commit_lower_is_better"] = data.get("lower_is_better")
        self._apply_metric_direction_audit(metric_event)
        card_fields = self._stage_commit_card_fields_from_judgment(data)
        fields = build_metric_validity_fields(metric_event=metric_event, card_fields=card_fields)
        feedback = await self._metric_validity_feedback_judgment(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            card_fields=card_fields,
            scope="pre_commit",
        )
        if feedback is not None and feedback.expected_lower_is_better is not None:
            expected = bool(feedback.expected_lower_is_better)
            current = bool(metric_event.get("lower_is_better") is not False)
            source_name = str(metric_event.get("metric_direction_source") or "")
            if current != expected:
                metric_event["metric_direction_conflict"] = True
                if source_name in {"", "declared", "default_lower"} and str(feedback.confidence or "").lower() == "high":
                    metric_event["lower_is_better"] = expected
                    metric_event["metric_direction_source"] = "feedback_llm"
            if feedback.metric_direction_reason:
                metric_event["metric_direction_feedback_reason"] = feedback.metric_direction_reason
        adjudicated = adjudicate_metric_validity(fields, llm_judgment=feedback)
        metric_event.update(adjudicated)
        if not bool(adjudicated.get("selection_eligible")):
            metric_event["selection_eligible"] = False
        metric_event["stage_result_audited_before_commit"] = True
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_result_audited_before_commit",
                "stage_id": stage_id,
                "source": source,
                "before": before,
                "after": {
                    "lower_is_better": metric_event.get("lower_is_better"),
                    "metric_validity": metric_event.get("metric_validity"),
                    "selection_eligible": metric_event.get("selection_eligible"),
                    "metric_direction_source": metric_event.get("metric_direction_source"),
                    "metric_direction_conflict": metric_event.get("metric_direction_conflict"),
                    "metric_validity_source": metric_event.get("metric_validity_source"),
                    "metric_validity_reason_code": metric_event.get("metric_validity_reason_code"),
                },
                "declared_stage_commit_lower_is_better": metric_event.get("declared_stage_commit_lower_is_better"),
                "block_text": self._compact_event_text(block_text),
            },
        )

    async def _adjudicate_metric_validity_for_stage(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        cards_after: list[Any],
    ) -> None:
        if metric_event.get("stage_result_audited_before_commit"):
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_validity_adjudicated",
                    "stage_id": stage_id,
                    "metric_value": metric_event.get("metric_value"),
                    "scope": "pre_commit_reused",
                    "metric_validity": metric_event.get("metric_validity"),
                    "metric_validity_note": metric_event.get("metric_validity_note"),
                    "metric_validity_reason_code": metric_event.get("metric_validity_reason_code"),
                    "metric_validity_confidence": metric_event.get("metric_validity_confidence"),
                    "metric_validity_source": metric_event.get("metric_validity_source"),
                    "selection_eligible": metric_event.get("selection_eligible"),
                },
            )
            return
        card_fields = self._metric_validity_card_fields(cards_after, stage_id)
        fields = build_metric_validity_fields(metric_event=metric_event, card_fields=card_fields)
        judgment = await self._metric_validity_feedback_judgment(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            card_fields=card_fields,
            scope="post_commit",
        )
        adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)
        metric_event.update(adjudicated)
        if not bool(adjudicated.get("selection_eligible")):
            metric_event["selection_eligible"] = False
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "metric_validity_adjudicated",
                "stage_id": stage_id,
                "metric_value": metric_event.get("metric_value"),
                **adjudicated,
            },
        )

    def _append_stage_performance_row(
        self,
        *,
        stage_id: str,
        snap: StageSnapshot,
        metric_event: dict[str, Any],
        cards_after: list[Any],
    ) -> None:
        worker_id = self._worker_uid_prefix()
        node_uid = self._snapshot_node_uid(snap) or self._stage_node_uid(stage_id)
        lineage_id = str(getattr(snap, "lineage_id", "") or (snap.source_event.get("lineage_id") if isinstance(snap.source_event, dict) else "") or self._lineage_uid_prefix())
        candidate_id = node_uid
        parent_stage = ""
        for idx, card in enumerate(cards_after):
            if card.stage_id == stage_id and idx > 0:
                parent_stage = cards_after[idx - 1].stage_id
                break
        card_map = {c.stage_id: c for c in cards_after}
        card = card_map.get(stage_id)
        restored = self.current_restored_from_stage
        restored_node_uid = self.current_restored_from_node_uid
        parent_node_uid = self._active_node_uid_for_stage(parent_stage) if parent_stage else ""
        out = self.global_log_dir / LHR_STAGE_PERFORMANCE_CSV
        brief = getattr(card, "brief", "") if card is not None else ""
        why = getattr(card, "why", "") if card is not None else ""
        route_evidence = getattr(card, "route_evidence", "") if card is not None else ""
        card_metric_validity = getattr(card, "metric_validity", "") if card is not None else ""
        metric_source_note = str(metric_event.get("metric_source_note") or "").strip()
        if not metric_source_note:
            metric_source_note = self._metric_source_note(metric_event, brief=brief, why=why)
        lower_is_better = self._metric_lower_is_better_for_event(metric_event, fallback=snap.lower_is_better)
        if metric_event.get("metric_validity_source"):
            metric_validity = str(metric_event.get("metric_validity") or "medium")
            metric_validity_note = str(metric_event.get("metric_validity_note") or "metric_validity_adjudicated")
        else:
            metric_validity, metric_validity_note = infer_metric_validity(
                {
                    **metric_event,
                    "metric_source_note": metric_source_note,
                    "metric_validity": card_metric_validity or metric_event.get("metric_validity"),
                    "brief": brief,
                    "why": why,
                    "route_evidence": route_evidence,
                }
            )
        estra_counts = self._estra_decision_counts()
        row = {
            "row_order": self._next_global_stage_row_order(out),
            "candidate_id": candidate_id,
            "worker_id": worker_id,
            "worker_index": self.worker_index,
            "worker_stage_order": len(self.stage_snapshots),
            "stage_id": stage_id,
            "lineage_id": lineage_id,
            "node_uid": node_uid,
            "parent_candidate_id": parent_node_uid,
            "parent_stage_id": parent_stage,
            "restored_from_candidate_id": restored_node_uid,
            "restored_from_stage": restored,
            "restored_from_node_uid": restored_node_uid,
            "metric_value": metric_event.get("metric_value"),
            "metric_name": metric_event.get("metric_name") or snap.metric_name,
            "lower_is_better": lower_is_better,
            "validation_ok": metric_event.get("validation_ok"),
            "validation_issue": metric_event.get("validation_issue"),
            "reported_val_score": metric_event.get("reported_val_score"),
            "val_score_type": metric_event.get("val_score_type"),
            "selection_eligible": metric_event.get("selection_eligible"),
            "selection_score": metric_event.get("selection_score"),
            "selection_note": metric_event.get("selection_note"),
            "metric_source_note": metric_source_note,
            "metric_validity": metric_validity,
            "metric_validity_note": metric_validity_note,
            "metric_validity_reason_code": metric_event.get("metric_validity_reason_code"),
            "metric_validity_confidence": metric_event.get("metric_validity_confidence"),
            "metric_validity_source": metric_event.get("metric_validity_source"),
            "task_profile": metric_event.get("task_profile") or self._evaluator_task_profile(),
            "metric_protocol": metric_event.get("metric_protocol"),
            "train_data_used": metric_event.get("train_data_used"),
            "metric_eval_data": metric_event.get("metric_eval_data"),
            "artifacts_reused_from": metric_event.get("artifacts_reused_from"),
            "training_rows": metric_event.get("training_rows"),
            "validation_rows": metric_event.get("validation_rows"),
            "execution_mode": metric_event.get("execution_mode"),
            "route_id": metric_event.get("route_id"),
            "execution_scale": metric_event.get("execution_scale"),
            "is_best_so_far": self._best_stage_id_so_far() == stage_id,
            "brief": brief,
            "why": why,
            "route_evidence": route_evidence,
            "solution_sha": metric_event.get("solution_sha"),
            "submission_sha": metric_event.get("submission_sha"),
            "source_commit_sha": metric_event.get("source_commit_sha"),
            "source_changed": metric_event.get("source_changed"),
            "semantic_source_changed": metric_event.get("semantic_source_changed"),
            "capture_type": metric_event.get("capture_type"),
            "workspace_git_stage_id": metric_event.get("workspace_git_stage_id"),
            "submission_snapshot": metric_event.get("submission_snapshot"),
            "artifact_path": metric_event.get("artifact_path"),
            "artifact_sha": metric_event.get("artifact_sha"),
            "artifact_kind": metric_event.get("artifact_kind"),
            "evaluator_backend": metric_event.get("evaluator_backend"),
            "evaluator_status": metric_event.get("evaluator_status"),
            "gate_metric_validity": metric_event.get("gate_metric_validity"),
            "gate_policy": metric_event.get("gate_policy"),
            "gate_policy_version": metric_event.get("gate_policy_version"),
            "gate_action": metric_event.get("gate_action"),
            "gate_accepted": metric_event.get("gate_accepted"),
            "gate_reason_code": metric_event.get("gate_reason_code"),
            "submission_changed": metric_event.get("submission_changed"),
            "candidate_ready": metric_event.get("candidate_ready"),
            "submission_status": metric_event.get("submission_status"),
            "duplicate_submission_of_stage": metric_event.get("duplicate_submission_of_stage"),
            "duplicate_submission_of_snapshot_id": metric_event.get("duplicate_submission_of_snapshot_id"),
            "workspace_git_ready": metric_event.get("workspace_git_ready"),
            "workspace_git_message": metric_event.get("workspace_git_message"),
            "solution_run_sec": metric_event.get("wall_sec") or metric_event.get("duration_sec"),
            "elapsed_min": (time.time() - self.started_at) / 60.0,
            "main_llm_calls": self.main_llm_calls,
            "main_tokens_input": self.main_tokens_in,
            "main_tokens_output": self.main_tokens_out,
            "main_tokens_cached": self.main_tokens_cached,
            "main_cache_rate": self.main_tokens_cached / self.main_tokens_in if self.main_tokens_in else "",
            "stage_commit_llm_calls": self.stage_llm_calls,
            "estra_llm_calls": self.estra_llm_calls,
            "estra_decision_count_before": estra_counts["decisions"],
            "estra_continue_count_before": estra_counts["continue"],
            "estra_redirect_count_before": estra_counts["redirect"],
            "estra_switch_count_before": estra_counts["switch"],
            "estra_current_continue_count_before": estra_counts["current_continue"],
            "estra_current_redirect_count_before": estra_counts["current_redirect"],
            "estra_stage_continue_count_before": estra_counts["stage_continue"],
            "estra_stage_redirect_count_before": estra_counts["stage_redirect"],
            "estra_switch_stage_count_before": self._count_jsonl_events("lhr_estras.jsonl", "estra_stage_switched"),
            "snapshot_id": snap.snapshot_id,
            "snapshot_path": str(snap.snapshot_path),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            self.global_log_dir.mkdir(parents=True, exist_ok=True)
            file_exists = out.exists() and out.stat().st_size > 0
            with out.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LHR_STAGE_PERFORMANCE_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({k: self._fmt_csv_value(row.get(k)) for k in LHR_STAGE_PERFORMANCE_COLUMNS})
        except OSError:
            logger.debug("[lnr] stage performance csv append failed", exc_info=True)
        self.current_restored_from_stage = ""
        self.current_restored_from_node_uid = ""

    @staticmethod
    def _tool_targets_ledger(tool_name: str, args: dict[str, Any], ledger: str) -> bool:
        if tool_name in {"write", "edit"}:
            raw_path = str(args.get("path") or "").replace("\\", "/").strip().lstrip("./")
            return raw_path == ledger or raw_path.endswith("/" + ledger)
        if tool_name != "bash":
            return False
        cmd = str(args.get("command") or "")
        probe = "\n".join(cmd.splitlines()[:12]) if cmd else ""
        if ledger not in probe:
            return False
        escaped = re.escape(ledger)
        rel = rf"(?:['\"]?\./)?{escaped}(?:['\"])?\b"
        return bool(
            re.search(rf"(?:>>|>)\s*{rel}", probe, re.MULTILINE)
            or re.search(rf"\btee\s+(?:-a\s+)?{rel}", probe, re.MULTILINE)
            or re.search(rf"\b(?:cp|mv)\b.*\s{rel}(?:\s|$)", probe, re.MULTILINE)
            or re.search(rf"\bsed\s+-i\b.*\s{rel}(?:\s|$)", probe, re.MULTILINE)
        )

    def _accumulate_ephemeral_tokens(self, agent: Any, kind: str, *, llm: Any | None = None) -> None:
        source_llm = llm or agent.llm
        try:
            ti = int(getattr(source_llm, "_last_call_input_tokens", 0) or 0)
            to = int(getattr(source_llm, "_last_call_output_tokens", 0) or 0)
            tc = int(getattr(source_llm, "_last_call_input_cached_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        if kind == "stage":
            self.stage_tokens_in += ti
            self.stage_tokens_out += to
            self.stage_tokens_cached += tc
            self.stage_llm_calls += 1
        else:
            self.estra_tokens_in += ti
            self.estra_tokens_out += to
            self.estra_tokens_cached += tc
            self.estra_llm_calls += 1

    def _sanitize_agent_visible_payload(self, agent: Any, value: Any) -> Any:
        if isinstance(value, str):
            return agent._sanitize_agent_visible_paths(value)
        if isinstance(value, dict):
            return {str(k): self._sanitize_agent_visible_payload(agent, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_agent_visible_payload(agent, v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_agent_visible_payload(agent, v) for v in value)
        return value

    def _sanitize_agent_prompt_surfaces(self, agent: Any) -> None:
        for attr in ("systemPrompt", "_system_prompt_core"):
            value = getattr(agent, attr, None)
            if isinstance(value, str) and value:
                try:
                    setattr(agent, attr, agent._sanitize_agent_visible_paths(value))
                except Exception:
                    logger.debug("[lnr] prompt path sanitize skipped for %s", attr, exc_info=True)
        tools = getattr(agent, "_tools_with_thought", None)
        if tools:
            try:
                setattr(agent, "_tools_with_thought", self._sanitize_agent_visible_payload(agent, tools))
            except Exception:
                logger.debug("[lnr] tool schema path sanitize skipped", exc_info=True)
        sanitizer = getattr(agent, "sanitize_existing_memory_agent_visible_paths", None)
        if callable(sanitizer):
            try:
                sanitizer()
            except Exception:
                logger.debug("[lnr] loaded memory path sanitize skipped", exc_info=True)

    def _prune_active_stage_snapshots_to_ledger(self) -> None:
        cards = parse_stage_cards(read_ledger(self.ledger_path))
        active = {str(card.stage_id).upper() for card in cards}
        if not active:
            return
        removed = sorted(sid for sid in self.stage_snapshots if str(sid).upper() not in active)
        removed_node_uids = {sid: self._snapshot_node_uid(self.stage_snapshots.get(sid)) for sid in removed}
        if not removed:
            return
        self.stage_snapshots = {
            sid: snap
            for sid, snap in self.stage_snapshots.items()
            if str(sid).upper() in active
        }
        self._write_stage_map()
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_active_stage_pruned",
                "active_stages": sorted(active),
                "removed_stages": removed,
                "removed_node_uids": removed_node_uids,
            },
        )

    def _sanitize_metric_event_for_agent_prompt(
        self,
        agent: Any,
        metric_event: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._sanitize_agent_visible_payload(agent, dict(metric_event or {}))
        if isinstance(out, dict) and out.get("snapshot_path"):
            out["snapshot_path"] = ".logs/fullrun_tail_snapshot.json"
        if isinstance(out, dict):
            out.pop("bash_cmd", None)
        return out if isinstance(out, dict) else {}

    @staticmethod
    def _stage_commit_one_line(value: Any, *, max_chars: int | None = 220) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        text = text.replace("###", "stage")
        if max_chars and max_chars > 0 and len(text) > max_chars:
            text = text[: max(0, max_chars - 3)].rstrip() + "..."
        return text

    @staticmethod
    def _stage_commit_bool_text(value: Any, *, default: bool = False) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value or "").strip().lower()
        if text in {"true", "1", "yes", "y", "lower"}:
            return "true"
        if text in {"false", "0", "no", "n", "higher"}:
            return "false"
        return "true" if default else "false"

    @staticmethod
    def _stage_commit_explicit_no_files(value: Any) -> bool:
        return str(value or "").strip().lower() in {"none", "no files", "no core files", "n/a", "na"}

    @classmethod
    def _stage_commit_default_judgment(cls, metric_event: dict[str, Any]) -> dict[str, str]:
        solution = cls._stage_commit_one_line(metric_event.get("solution_path") or metric_event.get("bash_kind") or "experiment", max_chars=80)
        status = cls._stage_commit_one_line(metric_event.get("submission_status") or "metric captured", max_chars=80)
        validity = cls._stage_commit_one_line(metric_event.get("metric_validity") or "unknown", max_chars=40)
        return {
            "brief": f"Recorded metric-backed {solution} stage with {status}.",
            "why": f"Preserves {validity} validation evidence for estra and merge review; compare against current best.",
        }

    def _stage_commit_fallback_judgment(self, metric_event: dict[str, Any]) -> dict[str, str]:
        """Build a deterministic, ledger-valid judgment for an accepted Gate result."""
        judgment = self._stage_commit_default_judgment(metric_event)
        inferred_files = format_stage_files(
            None,
            metric_event=metric_event,
            workspace_dir=self.workspace_dir,
        )
        # The snapshot and metric event retain the evaluated artifact path/SHA.
        # FILES is limited to reusable source/weight files, so an artifact-only
        # task can legitimately have no eligible FILES entry.
        judgment["files"] = inferred_files or "none"
        return judgment

    @classmethod
    def _stage_commit_metric_note(cls, metric_event: dict[str, Any]) -> str:
        for key in ("selection_note", "metric_validity_note", "validation_issue"):
            text = cls._stage_commit_one_line(metric_event.get(key), max_chars=180)
            if text:
                return text
        metric_type = cls._stage_commit_one_line(metric_event.get("val_score_type") or metric_event.get("metric_protocol"), max_chars=60)
        validity = cls._stage_commit_one_line(metric_event.get("metric_validity"), max_chars=40)
        if metric_type and metric_type != "unknown":
            return f"{metric_type} metric evidence; validity={validity or 'unknown'}."
        return f"Metric semantics unknown; validity={validity or 'unknown'}."

    @classmethod
    def _stage_commit_entry_from_metric_event(
        cls,
        *,
        stage_id: str,
        metric_event: dict[str, Any],
        judgment: dict[str, Any] | None,
        workspace_dir: Path | str | None = None,
    ) -> str:
        data = judgment if isinstance(judgment, dict) else {}
        fallback = cls._stage_commit_default_judgment(metric_event)
        metric_raw = metric_event.get("metric_value", metric_event.get("reported_val_score", "unknown"))
        if isinstance(metric_raw, (int, float)):
            metric = f"{float(metric_raw):.12g}"
        else:
            metric = cls._stage_commit_one_line(metric_raw or "unknown", max_chars=80)
        run_time = metric_event.get("run_time_sec")
        if run_time in (None, ""):
            run_time = metric_event.get("wall_sec")
        if run_time in (None, ""):
            run_time = metric_event.get("duration_sec")
        if isinstance(run_time, (int, float)):
            run_time_text = f"{float(run_time):.3f}".rstrip("0").rstrip(".")
        else:
            run_time_text = cls._stage_commit_one_line(run_time or "unknown", max_chars=50)
        metric_type = cls._stage_commit_one_line(metric_event.get("val_score_type") or "unknown", max_chars=80)
        metric_validity = cls._stage_commit_one_line(metric_event.get("metric_validity") or "medium", max_chars=40).lower()
        if metric_validity not in {"high", "medium", "low"}:
            metric_validity = "medium"
        brief = cls._stage_commit_one_line(data.get("brief") or data.get("BRIEF") or fallback["brief"], max_chars=None)
        why = cls._stage_commit_one_line(data.get("why") or data.get("WHY") or fallback["why"], max_chars=None)
        route_raw = data.get("route_evidence") or data.get("routeEvidence") or data.get("route") or ""
        if isinstance(route_raw, dict):
            route_raw = "; ".join(f"{k}={v}" for k, v in route_raw.items() if v not in (None, ""))
        route_evidence = cls._stage_commit_one_line(route_raw or fallback.get("route_evidence", ""), max_chars=None)
        if route_evidence and route_evidence not in why:
            why = cls._stage_commit_one_line(f"{why}; {route_evidence}", max_chars=None)
        files_raw = (
            data.get("files")
            or data.get("FILES")
            or data.get("stage_files")
            or data.get("stageFiles")
            or data.get("artifacts")
            or ""
        )
        files = format_stage_files(
            files_raw,
            metric_event=metric_event,
            workspace_dir=workspace_dir,
        )
        lines = [
            f"### {normalize_stage_id(stage_id) or stage_id}",
            f"metric: {metric}",
            f"lower_is_better: {cls._stage_commit_bool_text(metric_event.get('lower_is_better'))}",
            f"run_time_sec: {run_time_text}",
            f"metric_type: {metric_type or 'unknown'}",
            f"metric_note: {cls._stage_commit_metric_note(metric_event)}",
            f"metric_validity: {metric_validity}",
            f"BRIEF: {brief or fallback['brief']}",
            f"WHY: {why or fallback['why']}",
        ]
        if files:
            lines.append(f"FILES: {files}")
        elif cls._stage_commit_explicit_no_files(files_raw):
            lines.append("FILES: none")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _stage_commit_parse_judgment(text: str) -> dict[str, Any]:
        data = parse_arbiter_decision_text(text)
        return data if isinstance(data, dict) else {}

    async def _fork_stage_commit_judgment(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        prompt: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        if getattr(agent, "llm", None) is None:
            return {}
        stage_system = (
            "You are a forked bookkeeping reviewer. Return JSON only. "
            "Do not call tools, inspect files, write files, continue experiments, or use markdown."
        )
        t0 = time.time()
        try:
            text = ""
            if hasattr(getattr(agent, "llm", None), "ask_tool_stream") and hasattr(agent, "_memory_ctx"):
                messages = self._drop_dangling_tool_call_tail(agent._memory_ctx.build_messages_for_llm())
                messages.append(Message.user_message(prompt))
                system_msgs = list(agent._build_system_messages())
                system_msgs.append(Message.system_message(stage_system))
                assistant_msg = await self._ask_agent_tool_stream_guarded(
                    agent,
                    messages=messages,
                    system_msgs=system_msgs,
                    timeout=timeout_sec,
                    tools=[],
                    tool_choice="none",
                    parallel_tool_calls=False,
                    collect_all_tool_calls=True,
                )
                text = (getattr(assistant_msg, "content", None) or "").strip()
                if not text:
                    text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
            elif hasattr(getattr(agent, "llm", None), "ask"):
                text = await agent.llm.ask(
                    messages=[Message.user_message(prompt)],
                    system_msgs=[Message.system_message(stage_system)],
                    stream=False,
                    timeout=timeout_sec,
                )
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_stage_commit",
                    time.time() - t0,
                    None,
                    "ok",
                    recovery=True,
                    turn_kind="stage_commit",
                )
            self._accumulate_ephemeral_tokens(agent, "stage")
            parsed = self._stage_commit_parse_judgment(text)
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_fork_judgment_ok",
                    "stage_id": stage_id,
                    "response_chars": len(str(text or "")),
                    "has_brief": bool(parsed.get("brief") or parsed.get("BRIEF")),
                    "has_why": bool(parsed.get("why") or parsed.get("WHY")),
                },
            )
            return parsed
        except BaseException as exc:
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_stage_commit",
                    time.time() - t0,
                    None,
                    "error",
                    recovery=True,
                    turn_kind="stage_commit",
                )
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_fork_judgment_error",
                    "stage_id": stage_id,
                    "error": str(exc),
                },
            )
            return {}

    async def _ephemeral_stage_commit(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
    ) -> tuple[bool, str]:
        prompt = build_stage_commit_judgment_prompt(
            stage_id=stage_id,
            ledger_filename=self.ledger_filename,
            metric_event=self._sanitize_metric_event_for_agent_prompt(agent, metric_event),
            existing_ledger=agent._sanitize_agent_visible_paths(read_ledger(self.ledger_path)),
        )
        legacy_persist = bool(getattr(self.lhr, "stage_commit_persist_to_memory", False))
        persist_agent_write = bool(getattr(self.lhr, "stage_commit_persist_agent_write_to_memory", True)) or legacy_persist
        persist_stage_prompt = bool(getattr(self.lhr, "stage_commit_persist_prompt_to_memory", False)) or legacy_persist
        stage_user_msg = Message.user_message(prompt)
        stage_timeout = max(
            1.0,
            float(
                getattr(self.lhr, "stage_commit_llm_timeout_sec", 180.0)
                or 180.0
            ),
        )
        judgment = await self._fork_stage_commit_judgment(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            prompt=prompt,
            timeout_sec=stage_timeout,
        )
        await self._audit_stage_result_before_commit(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            judgment=judgment,
            block_text="",
            source="fork_context" if judgment else "fallback",
        )
        ledger_before = read_ledger(self.ledger_path)
        entry = self._stage_commit_entry_from_metric_event(
            stage_id=stage_id,
            metric_event=metric_event,
            judgment=judgment,
            workspace_dir=self.workspace_dir,
        )
        sep = ""
        if ledger_before and not ledger_before.endswith("\n\n"):
            sep = "\n" if ledger_before.endswith("\n") else "\n\n"
        ledger_after = ledger_before + sep + entry
        ok, reason = validate_append_only_stage_commit(ledger_before, ledger_after, stage_id)
        if not ok:
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_deterministic_append_rejected",
                    "stage_id": stage_id,
                    "reason": reason,
                },
            )
            return False, reason
        try:
            self._prepare_stage_commit_transaction(
                stage_id=stage_id,
                ledger_before=ledger_before,
                ledger_after=ledger_after,
                metric_event=metric_event,
            )
        except OSError as exc:
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_transaction_prepare_error",
                    "stage_id": stage_id,
                    "error": str(exc),
                },
            )
            return False, f"tool_error:{exc}"
        try:
            atomic_write(self.ledger_path, ledger_after)
        except OSError as exc:
            self._rollback_stage_commit_transaction(stage_id=stage_id, reason=f"ledger_write:{exc}")
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_deterministic_append_error",
                    "stage_id": stage_id,
                    "error": str(exc),
                },
            )
            return False, f"tool_error:{exc}"
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_appended_deterministic",
                "stage_id": stage_id,
                "judgment_source": "fork_context" if judgment else "fallback",
            },
        )
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {"event": "stage_commit_ok", "stage_id": stage_id, "turn": 1},
        )
        self._capture_s01_eda_prefix_end(stage_id=stage_id)
        if persist_agent_write:
            if persist_stage_prompt:
                agent.memory.add_message(stage_user_msg)
            stage_msg = Message.assistant_message(agent._sanitize_agent_visible_paths(entry))
            agent.memory.add_message(stage_msg)
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_persisted_to_memory",
                    "stage_id": stage_id,
                    "turn": 1,
                    "agent_write_persisted": True,
                    "prompt_persisted": persist_stage_prompt,
                    "deterministic_append": True,
                },
            )
        return True, ""

    def _estra_stage_checkpoint_context(self, candidate_stages: list[str]) -> str:
        lines: list[str] = []
        for sid in candidate_stages:
            snap = self.stage_snapshots.get(sid)
            if snap is None:
                continue
            raw_source = getattr(snap, "source_event", {})
            source = raw_source if isinstance(raw_source, dict) else {}
            metric = source.get("metric_value", getattr(snap, "metric_value", ""))
            commit = str(source.get("source_commit_sha") or "")[:12]
            validation_ok = source.get("validation_ok")
            metric_validity = source.get("metric_validity")
            selection_eligible = source.get("selection_eligible")
            reason_code = source.get("metric_validity_reason_code") or source.get("metric_reason_code")
            parts = [f"- {sid}: metric={metric}"]
            if commit:
                parts.append(f"source_commit={commit}")
            if validation_ok is not None:
                parts.append(f"validation_ok={validation_ok}")
            if metric_validity not in (None, ""):
                parts.append(f"metric_validity={metric_validity}")
            if selection_eligible not in (None, ""):
                parts.append(f"selection_eligible={self._stage_card_override_text(selection_eligible)}")
            if reason_code not in (None, ""):
                parts.append(f"metric_validity_reason={reason_code}")
            lines.append("; ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _estra_switch_candidates(candidates: list[str], latest: str) -> list[str]:
        latest = str(latest or "").upper()
        return [str(sid or "").upper() for sid in candidates if str(sid or "").upper() and str(sid or "").upper() != latest]

    @staticmethod
    def _is_keep_like_estra_action(action: str) -> bool:
        return str(action or "").strip() in {"keep_current", "keep_but_redirect"}

    @staticmethod
    def _estra_axes_from_action(action: str) -> tuple[str, str]:
        normalized = str(action or "").strip()
        if normalized == "switch_stage":
            return "previous_stage", "continue"
        if normalized == "keep_but_redirect":
            return "current_workspace", "redirect"
        return "current_workspace", "continue"

    @staticmethod
    def _estra_action_from_axes(startpoint: str, intent: str) -> str:
        start = str(startpoint or "").strip()
        goal = str(intent or "").strip()
        if start == "previous_stage":
            return "switch_stage"
        if goal == "redirect":
            return "keep_but_redirect"
        return "keep_current"

    @staticmethod
    def _estra_decision_kind(*, action: str, startpoint: str, intent: str) -> str:
        if action == "switch_stage" or startpoint == "previous_stage":
            return "switch"
        if action == "keep_but_redirect" or intent == "redirect":
            return "redirect"
        if action == "keep_current":
            return "continue"
        return "invalid"

    @staticmethod
    def _normalize_estra_startpoint(value: Any, *, action: str = "") -> str:
        raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in {"previous_stage", "historical_stage", "switch_stage", "restore_stage", "stage"}:
            return "previous_stage"
        if raw in {"current_workspace", "current", "latest", "keep_current", "workspace"}:
            return "current_workspace"
        return LnrSolver._estra_axes_from_action(action)[0]

    @staticmethod
    def _normalize_estra_intent(value: Any, *, action: str = "") -> str:
        raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in {"redirect", "refocus", "change_tactic", "change_focus"}:
            return "redirect"
        if raw in {"continue", "keep", "deepen", "resume"}:
            return "continue"
        return LnrSolver._estra_axes_from_action(action)[1]

    @staticmethod
    def _derive_estra_compact(*, action: str, trigger_source: str) -> bool:
        action = str(action or "").strip()
        if action in {"keep_current", "keep_but_redirect", "switch_stage"}:
            return True
        return str(trigger_source or "").strip() in {"context_limit", "text_only", "context_hygiene"}

    def _estra_decision_fields(self, parsed: dict[str, Any]) -> dict[str, str]:
        field_names = (
            "startpoint",
            "intent",
            "exploration_summary",
            "bottleneck",
            "evidence",
            "missing_evidence",
            "decision_reason",
            "redirect_focus",
        )
        fields: dict[str, str] = {}
        for name in field_names:
            value = parsed.get(name)
            if value is None:
                continue
            compact = self._compact_event_text(str(value), max_chars=240)
            if compact:
                fields[name] = compact
        return fields

    def _estra_decision_reason(self, parsed: dict[str, Any]) -> str:
        fields = self._estra_decision_fields(parsed)
        reason = fields.get("decision_reason") or str(parsed.get("reason") or "").strip()
        bottleneck = fields.get("bottleneck") or ""
        evidence = fields.get("evidence") or ""
        parts = []
        if reason:
            parts.append(reason)
        if bottleneck:
            parts.append(f"bottleneck: {bottleneck}")
        if evidence:
            parts.append(f"evidence: {evidence}")
        return self._compact_event_text(" | ".join(parts), max_chars=360)

    @staticmethod
    def _estra_observation_key(*, cards: list[Any], latest: str, candidates: list[str]) -> str:
        ids = [str(getattr(card, "stage_id", "") or "").upper() for card in cards]
        candidate_text = ",".join(str(sid or "").upper() for sid in candidates)
        digest = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()[:12]
        return f"stages={len(ids)};latest={str(latest or '').upper()};candidates={digest}"

    def _emit_estra_decision(
        self,
        *,
        action: str,
        target_stage: str,
        latest_stage: str,
        trigger_source: str,
        compact: bool,
        reason: str,
        decision_mode: str,
        candidates: list[str],
        switch_candidates: list[str],
        candidate_node_uids: dict[str, str],
        raw: str = "",
        estra_fields: dict[str, Any] | None = None,
        estra_context: dict[str, Any] | None = None,
    ) -> None:
        target = str(target_stage or "").upper()
        latest = str(latest_stage or "").upper()
        fields = estra_fields or {}
        context = dict(estra_context or {})
        startpoint = str(fields.get("startpoint") or self._estra_axes_from_action(action)[0])
        intent = str(fields.get("intent") or self._estra_axes_from_action(action)[1])
        decision_kind = self._estra_decision_kind(action=action, startpoint=startpoint, intent=intent)
        decision_text = " ".join(str(fields.get(name) or "") for name in ("decision_reason", "bottleneck", "evidence", "redirect_focus")).lower()
        if context.get("peer_route_evidence_present"):
            peer_tokens = [
                str(context.get("peer_best_worker") or "").lower(),
                str(context.get("peer_best_stage") or "").lower(),
                "peer",
                "other worker",
            ]
            context["estra_addressed_peer_best"] = any(token and token in decision_text for token in peer_tokens)
        if context.get("backtrack_reflection_present"):
            backtrack_stage = str(context.get("backtrack_best_stage") or "").lower()
            context["estra_addressed_backtrack"] = (
                startpoint == "previous_stage"
                or "previous" in decision_text
                or "restore" in decision_text
                or "backtrack" in decision_text
                or bool(backtrack_stage and backtrack_stage in decision_text)
            )
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "estra_decision",
                "action": action,
                "target_stage": target,
                "target_node_uid": candidate_node_uids.get(target, ""),
                "latest_stage": latest,
                "latest_node_uid": candidate_node_uids.get(latest, ""),
                "trigger_source": trigger_source,
                "compact": bool(compact),
                "decision_mode": decision_mode,
                "candidates": candidates,
                "switch_candidates": switch_candidates,
                "candidate_node_uids": candidate_node_uids,
                "reason": reason,
                "startpoint": startpoint,
                "intent": intent,
                "decision_kind": decision_kind,
                "exploration_summary": str(fields.get("exploration_summary") or ""),
                "bottleneck": str(fields.get("bottleneck") or ""),
                "evidence": str(fields.get("evidence") or ""),
                "missing_evidence": str(fields.get("missing_evidence") or ""),
                "decision_reason": str(fields.get("decision_reason") or ""),
                "redirect_focus": str(fields.get("redirect_focus") or ""),
                **context,
                "raw": raw,
                "response_chars": len(str(raw or "")),
            },
        )

    def _normalize_estra_decision(
        self,
        parsed: dict[str, Any],
        *,
        latest: str,
        switch_candidates: list[str],
        trigger_source: str,
    ) -> dict[str, Any]:
        latest = str(latest or "").upper()
        legacy_action = str(parsed.get("action") or "").strip()
        has_axis = "startpoint" in parsed or "intent" in parsed
        if not legacy_action and not has_axis:
            return {}
        startpoint = self._normalize_estra_startpoint(parsed.get("startpoint"), action=legacy_action)
        intent = self._normalize_estra_intent(parsed.get("intent"), action=legacy_action)
        action = self._estra_action_from_axes(startpoint, intent)
        target = str(parsed.get("target_stage") or parsed.get("target_step") or "").strip().upper()

        if action == "switch_stage":
            if target not in switch_candidates:
                return {}
        elif self._is_keep_like_estra_action(action):
            target = latest
        else:
            return {}

        fields = self._estra_decision_fields({**parsed, "startpoint": startpoint, "intent": intent})
        reason = self._estra_decision_reason(parsed)
        return {
            "action": action,
            "startpoint": startpoint,
            "intent": intent,
            "target_stage": target,
            "compact": self._derive_estra_compact(action=action, trigger_source=trigger_source),
            "reason": reason,
            **fields,
        }

    async def _ask_estra(self, agent: Any, *, trigger_source: str = "manual") -> dict[str, Any]:
        cards = self._effective_stage_cards_from_ledger()
        latest = cards[-1].stage_id if cards else ""
        candidates = [c.stage_id for c in cards if c.stage_id in self.stage_snapshots]
        candidate_node_uids = {sid: self._active_node_uid_for_stage(sid) for sid in candidates}
        switch_candidates = self._estra_switch_candidates(candidates, latest)
        if len(cards) < 2 or not candidates:
            return {
                "action": "keep_current",
                "target_stage": latest,
                "compact": self._derive_estra_compact(action="keep_current", trigger_source=trigger_source),
                "reason": "not enough completed stages for switch_stage",
            }
        forced_target = ""
        if str(trigger_source or "") == "force_stage_capture":
            forced_target = str(getattr(self.lhr, "force_estra_target_stage", "") or "").strip().upper()
        if forced_target:
            if forced_target == str(latest or "").upper():
                decision = {
                    "action": "keep_current",
                    "target_stage": latest,
                    "compact": self._derive_estra_compact(action="keep_current", trigger_source=trigger_source),
                    "reason": f"forced estra target {forced_target} is current stage",
                }
            elif forced_target in switch_candidates:
                decision = {
                    "action": "switch_stage",
                    "target_stage": forced_target,
                    "compact": True,
                    "reason": f"forced estra target {forced_target}",
                }
            else:
                decision = {
                    "action": "keep_current",
                    "target_stage": latest,
                    "compact": self._derive_estra_compact(action="keep_current", trigger_source=trigger_source),
                    "reason": f"forced estra target {forced_target} is not a valid switch candidate",
                }
            self._emit_estra_decision(
                action=decision["action"],
                target_stage=decision["target_stage"],
                latest_stage=latest,
                trigger_source=trigger_source,
                compact=bool(decision["compact"]),
                reason=str(decision.get("reason") or ""),
                decision_mode="forced_config",
                candidates=candidates,
                switch_candidates=switch_candidates,
                candidate_node_uids=candidate_node_uids,
            )
            return decision

        if not switch_candidates:
            decision = {
                "action": "keep_current",
                "target_stage": latest,
                "compact": self._derive_estra_compact(action="keep_current", trigger_source=trigger_source),
                "reason": "no valid historical switch candidates",
            }
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_no_valid_switch_candidate",
                    "latest_stage": latest,
                    "latest_node_uid": self._active_node_uid_for_stage(latest),
                    "trigger_source": trigger_source,
                    "candidates": candidates,
                    "switch_candidates": switch_candidates,
                    "candidate_node_uids": candidate_node_uids,
                    "estra_action": decision["action"],
                    "target_stage": decision["target_stage"],
                    "reason": decision["reason"],
                },
            )
            return decision

        best = self._best_candidate_stage_id(cards, candidates)
        stage_memory_text = self._stage_memory_view_for_prompt(
            cards,
            latest_stage=latest,
            best_stage=best,
        )
        peer_evidence = self._estra_peer_route_evidence()
        backtrack_text, backtrack_meta = self._estra_backtrack_reflection(
            switch_candidates,
            latest_stage=latest,
        )
        estra_context_audit = {**peer_evidence.audit_fields(), **backtrack_meta}
        prompt = build_estra_prompt(
            ledger_filename="Stage Memory View",
            ledger_text=agent._sanitize_agent_visible_paths(stage_memory_text),
            latest_stage=latest,
            switch_candidate_stages=switch_candidates,
            stage_checkpoint_context=self._estra_stage_checkpoint_context(candidates),
            peer_route_evidence=peer_evidence.text,
            backtrack_reflection=backtrack_text,
            resource_context=self._resource_context_for_prompt(),
        )
        if str(trigger_source or "") == "context_limit":
            prompt = (
                "ESTRA trigger: the main agent context is about to overflow before the next LLM turn. "
                "Choose a compacted next stage by setting startpoint and intent. Use previous_stage only if a historical completed stage is a better restart point.\n\n"
                + prompt
            )
        estra_system = (
            "You are a estra decision controller for one ML search run. "
            "Return exactly one JSON object and nothing else. "
            "Do not call tools, emit DSML/tool markup, write files, or use markdown."
        )
        decision_mode = "isolated"
        t0 = time.time()
        try:
            use_main_context = bool(getattr(self.lhr, "estra_use_main_agent_context", True))
            if use_main_context and hasattr(getattr(agent, "llm", None), "ask_tool_stream") and hasattr(agent, "_memory_ctx"):
                decision_mode = "main_agent_context"
                messages = agent._memory_ctx.build_messages_for_llm()
                messages.append(Message.user_message(prompt))
                assistant_msg = await self._ask_agent_tool_stream_guarded(
                    agent,
                    messages=messages,
                    system_msgs=agent._build_system_messages(),
                    timeout=agent._llm_stream_timeout_sec,
                    tools=[],
                    tool_choice="none",
                    parallel_tool_calls=False,
                    collect_all_tool_calls=True,
                )
                text = (getattr(assistant_msg, "content", None) or "").strip()
                if not text:
                    text = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
            else:
                text = await agent.llm.ask(
                    messages=[Message.user_message(prompt)],
                    system_msgs=[Message.system_message(estra_system)],
                    stream=False,
                    timeout=agent._llm_stream_timeout_sec,
                )
            agent._record_llm_call(
                "lnr_estra",
                time.time() - t0,
                None,
                "ok",
                recovery=True,
                turn_kind="estra",
            )
            self._accumulate_ephemeral_tokens(agent, "estra")
        except BaseException as exc:
            agent._record_llm_call(
                "lnr_estra",
                time.time() - t0,
                None,
                "error",
                recovery=True,
                turn_kind="estra",
            )
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_llm_error",
                    "error": str(exc),
                    "candidates": candidates,
                    "switch_candidates": switch_candidates,
                    "candidate_node_uids": candidate_node_uids,
                    "latest_stage": latest,
                    "trigger_source": trigger_source,
                    "decision_mode": decision_mode,
                },
            )
            return {
                "action": "keep_current",
                "target_stage": latest,
                "compact": self._derive_estra_compact(action="keep_current", trigger_source=trigger_source),
                "reason": f"estra llm error: {exc}",
            }

        parsed = self._parse_estra_decision(text, switch_candidates=switch_candidates)
        normalized = self._normalize_estra_decision(
            parsed,
            latest=latest,
            switch_candidates=switch_candidates,
            trigger_source=trigger_source,
        )
        if decision_mode == "main_agent_context" and not normalized:
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_main_context_invalid_fallback",
                    "latest_stage": latest,
                    "trigger_source": trigger_source,
                    "candidates": candidates,
                    "switch_candidates": switch_candidates,
                    "raw": text,
                    "response_chars": len(str(text or "")),
                },
            )
            fallback_t0 = time.time()
            try:
                fallback_text = await agent.llm.ask(
                    messages=[Message.user_message(prompt)],
                    system_msgs=[Message.system_message(estra_system)],
                    stream=False,
                    timeout=agent._llm_stream_timeout_sec,
                )
                agent._record_llm_call(
                    "lnr_estra",
                    time.time() - fallback_t0,
                    None,
                    "ok",
                    recovery=True,
                    turn_kind="estra",
                )
                self._accumulate_ephemeral_tokens(agent, "estra")
                fallback_parsed = self._parse_estra_decision(fallback_text, switch_candidates=switch_candidates)
                fallback_normalized = self._normalize_estra_decision(
                    fallback_parsed,
                    latest=latest,
                    switch_candidates=switch_candidates,
                    trigger_source=trigger_source,
                )
                if fallback_normalized:
                    text = fallback_text
                    normalized = fallback_normalized
                    decision_mode = "isolated_fallback"
            except BaseException as exc:
                agent._record_llm_call(
                    "lnr_estra",
                    time.time() - fallback_t0,
                    None,
                    "error",
                    recovery=True,
                    turn_kind="estra",
                )
                self._jsonl(
                    "lhr_estra_events.jsonl",
                    {
                        "event": "estra_fallback_llm_error",
                        "error": str(exc),
                        "latest_stage": latest,
                        "trigger_source": trigger_source,
                    },
                )
        if not normalized:
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_invalid",
                    "latest_stage": latest,
                    "trigger_source": trigger_source,
                    "decision_mode": decision_mode,
                    "candidates": candidates,
                    "switch_candidates": switch_candidates,
                    "candidate_node_uids": candidate_node_uids,
                    "raw": text,
                    "response_chars": len(str(text or "")),
                },
            )
            normalized = {
                "action": "keep_current",
                "target_stage": latest,
                "compact": self._derive_estra_compact(action="keep_current", trigger_source=trigger_source),
                "reason": "invalid estra response",
            }
        reason = str(normalized.get("reason") or parsed.get("reason") or "")
        normalized["reason"] = reason
        self._emit_estra_decision(
            action=str(normalized.get("action") or "keep_current"),
            target_stage=str(normalized.get("target_stage") or latest),
            latest_stage=latest,
            trigger_source=trigger_source,
            compact=bool(normalized.get("compact")),
            reason=reason,
            decision_mode=decision_mode,
            candidates=candidates,
            switch_candidates=switch_candidates,
            candidate_node_uids=candidate_node_uids,
            raw=text,
            estra_fields=normalized,
            estra_context=estra_context_audit,
        )
        return normalized

    @staticmethod
    def _normalize_estra_archive_summary_text(text: str, *, max_chars: int = 1000) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        raw = re.sub(r"^```(?:text|markdown)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        lines: list[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            cleaned = re.sub(r"^[-*•]\s*", "", cleaned).strip()
            if cleaned:
                lines.append(f"- {cleaned}")
            if len(lines) >= 5:
                break
        summary = "\n".join(lines) if lines else re.sub(r"\s+", " ", raw).strip()
        if len(summary) > max_chars:
            summary = summary[: max(0, max_chars - 32)].rstrip() + "\n... [archive summary truncated]"
        return summary

    async def _synthesize_estra_archive_summary(
        self,
        agent: Any,
        *,
        target_stage: str,
        terminal_stage: str,
        estra_reason: str,
        deterministic_summary: str,
    ) -> str:
        fallback = str(deterministic_summary or "").strip()
        if not fallback:
            return ""
        if not bool(getattr(self.lhr, "estra_archive_summary_llm_enabled", True)):
            return fallback
        llm = getattr(agent, "llm", None)
        if llm is None or not callable(getattr(llm, "ask", None)):
            return fallback
        prompt = build_estra_archive_summary_prompt(
            target_stage=target_stage,
            terminal_stage=terminal_stage,
            estra_reason=estra_reason,
            deterministic_summary=fallback,
        )
        system = (
            "You summarize estra-abandoned ML research tails. Return compact plain text only. "
            "Do not call tools, write files, or invent details."
        )
        t0 = time.time()
        try:
            text = await llm.ask(
                messages=[Message.user_message(prompt)],
                system_msgs=[Message.system_message(system)],
                stream=False,
                timeout=float(getattr(agent, "_llm_stream_timeout_sec", 60) or 60),
            )
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_estra_archive_summary",
                    time.time() - t0,
                    None,
                    "ok",
                    recovery=True,
                    turn_kind="estra",
                )
            self._accumulate_ephemeral_tokens(agent, "estra")
        except BaseException as exc:
            if hasattr(agent, "_record_llm_call"):
                agent._record_llm_call(
                    "lnr_estra_archive_summary",
                    time.time() - t0,
                    None,
                    "error",
                    recovery=True,
                    turn_kind="estra",
                )
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "estra_archive_summary_llm_error",
                    "target_stage": str(target_stage or "").upper(),
                    "terminal_stage": str(terminal_stage or "").upper(),
                    "error": str(exc),
                    "fallback_chars": len(fallback),
                },
            )
            return fallback

        summary = self._normalize_estra_archive_summary_text(str(text or ""), max_chars=1000)
        if not summary:
            return fallback
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_archive_summary_synthesized",
                "target_stage": str(target_stage or "").upper(),
                "terminal_stage": str(terminal_stage or "").upper(),
                "deterministic_summary_chars": len(fallback),
                "summary_chars": len(summary),
                "summary_excerpt": self._compact_event_text(summary, max_chars=600),
            },
        )
        return summary

    async def _set_pending_estra_from_decision(self, *, agent: Any, target: str, decision: dict[str, Any]) -> None:
        if not bool(decision.get("compact")):
            return
        current_ledger = read_ledger(self.ledger_path)
        cards = self._effective_stage_cards(parse_stage_cards(current_ledger))
        latest = cards[-1].stage_id if cards else ""
        action = str(decision.get("action") or "").strip()
        if self._is_keep_like_estra_action(action):
            target = latest
        elif action == "switch_stage":
            target = str(target or decision.get("target_stage") or "").upper()
        else:
            return
        if not target:
            return
        summary_max_chars = int(getattr(self.lhr, "estra_compact_max_chars", 0) or self.lhr.tail_summary_max_chars or 1200)
        if action == "switch_stage":
            deterministic_summary = tail_summary_after_cards(cards, target_stage=target, max_chars=summary_max_chars)
            summary = await self._synthesize_estra_archive_summary(
                agent,
                target_stage=target,
                terminal_stage=latest,
                estra_reason=str(decision.get("reason") or ""),
                deterministic_summary=deterministic_summary,
            )
        else:
            summary = tail_summary_from_cards(
                cards,
                start_stage="S02",
                target_stage=target,
                max_chars=summary_max_chars,
            )
            summary = summary.replace("Abandoned compact trajectory", "Compact current trajectory")
            summary = summary.replace(f" before estra to {target}", "")
            summary = summary.replace(" before estra", "")
        reason = str(decision.get("reason") or "")
        startpoint, intent = self._estra_axes_from_action(action)
        estra_fields = {
            "startpoint": str(decision.get("startpoint") or startpoint),
            "intent": str(decision.get("intent") or intent),
            **self._estra_decision_fields(decision),
        }
        state_packet = self._build_lhr_state_packet(
            action=action,
            target_stage=target,
            terminal_stage=latest,
            reason=reason,
            tail_summary=summary,
            estra_fields=estra_fields,
        )
        self.pending_estra = {
            "action": action,
            "target_stage": target,
            "target_node_uid": self._active_node_uid_for_stage(target),
            "terminal_stage": latest,
            "terminal_node_uid": self._active_node_uid_for_stage(latest),
            "tail_summary": summary,
            "terminal_ledger": current_ledger,
            "reason": reason,
            "state_packet": state_packet,
            "estra_fields": estra_fields,
            "compact_strength": str(decision.get("compact_strength") or ""),
            "trigger_source": str(decision.get("trigger_source") or ""),
            "context_generation": str(decision.get("context_generation") or ""),
            "restore_key": str(decision.get("restore_key") or ""),
        }

    def _estra_candidates_for_current_ledger(self) -> tuple[list[Any], str, list[str]]:
        cards = self._effective_stage_cards_from_ledger()
        latest = cards[-1].stage_id if cards else ""
        candidates = [c.stage_id for c in cards if c.stage_id in self.stage_snapshots]
        return cards, latest, candidates

    def _estra_trigger_allowed(self, *, cards: list[Any], candidates: list[str]) -> bool:
        if not bool(self.lhr.estra_enabled):
            return False
        trigger_after = int(self.lhr.estra_trigger_stage_count or 0)
        min_stages = trigger_after if trigger_after > 0 else 2
        if len(cards) < min_stages or not candidates:
            return False
        latest = cards[-1].stage_id if cards else ""
        if not self._estra_switch_candidates(candidates, latest):
            return False
        observation_key = self._estra_observation_key(cards=cards, latest=latest, candidates=candidates)
        if observation_key == str(getattr(self, "last_estra_observation_key", "") or ""):
            return False
        if len(cards) <= self.last_estra_stage_count:
            return False
        return True

    def _text_only_estra_trigger_allowed(self, *, cards: list[Any], candidates: list[str]) -> bool:
        if not bool(getattr(self.lhr, "estra_enabled", True)):
            return False
        if not cards or not candidates:
            return False
        if getattr(self, "pending_estra", None):
            return False
        return time.monotonic() < float(getattr(self, "deadline", 0.0) or 0.0)

    def _force_estra_trigger_allowed(
        self,
        *,
        cards: list[Any],
        candidates: list[str],
        observed_stage_count: int | None = None,
    ) -> bool:
        force_after = int(getattr(self.lhr, "force_estra_after_stage_count", 0) or 0)
        if force_after <= 0:
            return False
        if not bool(getattr(self.lhr, "estra_enabled", True)):
            return False
        observed_count = int(observed_stage_count if observed_stage_count is not None else len(cards))
        if observed_count < force_after or not candidates:
            return False
        latest = cards[-1].stage_id if cards else ""
        if not self._estra_switch_candidates(candidates, latest):
            return False
        last_observed = max(
            int(getattr(self, "last_force_estra_observation_count", 0) or 0),
            int(getattr(self, "last_estra_stage_count", 0) or 0),
        )
        if observed_count <= last_observed:
            return False
        observation_key = self._estra_observation_key(cards=cards, latest=latest, candidates=candidates)
        if observation_key == str(getattr(self, "last_force_estra_observation_key", "") or ""):
            return False
        return True


    def _current_main_token_totals(self, agent: Any) -> tuple[int, int]:
        run_in = int(getattr(agent, "_run_tokens_in", 0) or getattr(agent, "last_run_tokens_in", 0) or 0)
        run_cached = int(getattr(agent, "_run_tokens_cached", 0) or getattr(agent, "last_run_tokens_cached", 0) or 0)
        return int(getattr(self, "main_tokens_in", 0) or 0) + run_in, int(getattr(self, "main_tokens_cached", 0) or 0) + run_cached

    def _note_context_hygiene_stage_delta(self, agent: Any) -> dict[str, Any]:
        tokens_in, tokens_cached = self._current_main_token_totals(agent)
        delta_in = max(0, tokens_in - int(getattr(self, "context_hygiene_last_stage_tokens_in", 0) or 0))
        delta_cached = max(0, tokens_cached - int(getattr(self, "context_hygiene_last_stage_tokens_cached", 0) or 0))
        rate = None
        if delta_in > 0:
            rate = max(0.0, min(1.0, delta_cached / delta_in))
            rates = list(getattr(self, "context_hygiene_cache_rates", []) or [])
            rates.append(rate)
            self.context_hygiene_cache_rates = rates[-100:]
        self.context_hygiene_last_stage_tokens_in = tokens_in
        self.context_hygiene_last_stage_tokens_cached = tokens_cached
        return {
            "tokens_in": tokens_in,
            "tokens_cached": tokens_cached,
            "delta_input_tokens": delta_in,
            "delta_cached_tokens": delta_cached,
            "incremental_cache_rate": rate,
        }

    def _context_hygiene_active_high_risk_bash(self) -> bool:
        observer = getattr(self, "resource_observer", None)
        runtime = getattr(observer, "resource_runtime", None)
        jobs = getattr(observer, "_jobs", {}) if observer is not None else {}
        if not isinstance(jobs, dict) or runtime is None:
            return False
        for job in jobs.values():
            try:
                if not bool(getattr(job, "visible", False)):
                    continue
                if getattr(job, "terminal_signal_seen", False) or getattr(job, "exit_code_seen", False):
                    continue
                if runtime.has_active_lease(job_id=getattr(job, "job_id", "")):
                    return True
            except Exception:
                continue
        return False

    def _context_hygiene_snapshot_can_preserve_current_state(self, *, cards_after: list[Any]) -> tuple[bool, dict[str, Any]]:
        latest = str(cards_after[-1].stage_id if cards_after else "").upper()
        snap = self.stage_snapshots.get(latest)
        event = dict(getattr(snap, "source_event", {}) or {}) if snap is not None else {}
        fields = {
            "latest_stage": latest,
            "has_snapshot": snap is not None,
            "has_metric": event.get("metric_value") is not None,
            "has_submission_path": bool(event.get("submission_snapshot") or event.get("submission_sha") or event.get("submission_status") == "missing_submission"),
            "has_code_ref": bool(event.get("source_commit_sha") or event.get("solution_sha") or getattr(snap, "snapshot_path", None)),
            "has_constraints": True,
            "has_pending_high_value_marker": True,
        }
        ok = bool(fields["has_snapshot"] and fields["has_code_ref"])
        return ok, fields

    async def _context_hygiene_compact_after_stage_capture(self, *, agent: Any, cards_after: list[Any]) -> str | None:
        if not bool(getattr(self.lhr, "context_hygiene_compact_enabled", False)):
            return None
        if getattr(self, "pending_estra", None):
            return None
        token_delta = self._note_context_hygiene_stage_delta(agent)
        effective_cards_after = self._effective_stage_cards(cards_after)
        latest = str(effective_cards_after[-1].stage_id if effective_cards_after else "").upper()
        stage_count = len(effective_cards_after)
        stage_count_since = max(0, stage_count - int(getattr(self, "context_hygiene_last_compact_stage_count", 0) or 0))
        tokens_in, _tokens_cached = self._current_main_token_totals(agent)
        tokens_since = max(0, tokens_in - int(getattr(self, "context_hygiene_last_compact_tokens_in", 0) or 0))
        messages = self._agent_memory_messages(agent)
        file_counts = large_code_file_touch_counts(messages)
        tool_outputs = large_tool_output_count(
            messages,
            min_chars=int(getattr(self.lhr, "context_hygiene_large_tool_output_chars", 12000) or 12000),
        )
        snapshot_ok, snapshot_fields = self._context_hygiene_snapshot_can_preserve_current_state(cards_after=effective_cards_after)
        active_high_risk = self._context_hygiene_active_high_risk_bash()
        decision = evaluate_context_hygiene_compact(
            stage_count_since_last_compact=stage_count_since,
            total_input_tokens_since_last_compact=tokens_since,
            cache_rates=list(getattr(self, "context_hygiene_cache_rates", []) or []),
            large_file_touch_counts=file_counts,
            large_tool_output_count=tool_outputs,
            seconds_since_last_compact=time.time() - float(getattr(self, "context_hygiene_last_compact_ts", 0.0) or getattr(self, "started_at", time.time())),
            active_high_risk_bash=active_high_risk,
            snapshot_can_preserve_current_state=snapshot_ok,
            max_stages_without_compact=int(getattr(self.lhr, "context_hygiene_max_stages_without_compact", 25) or 25),
            code_churn_stage_threshold=int(getattr(self.lhr, "context_hygiene_code_churn_stage_threshold", 10) or 10),
            large_file_repeat_threshold=int(getattr(self.lhr, "context_hygiene_large_file_repeat_threshold", 10) or 10),
            low_incremental_cache_rate=float(getattr(self.lhr, "context_hygiene_low_incremental_cache_rate", 0.80) or 0.80),
            low_cache_window=int(getattr(self.lhr, "context_hygiene_low_cache_window", 5) or 5),
            min_tokens_since_compact=int(getattr(self.lhr, "context_hygiene_min_tokens_since_compact", 5_000_000) or 5_000_000),
            tool_output_min_interval_sec=float(getattr(self.lhr, "context_hygiene_tool_output_min_interval_sec", 1800.0) or 1800.0),
        )
        facts = {**decision.facts, "token_delta": token_delta, "snapshot_preservation": snapshot_fields, "latest_stage": latest}
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "cache_hygiene_compact_check",
                "latest_stage": latest,
                "stage_count": stage_count,
                "reason": decision.reason,
                "should_compact": decision.should_compact,
                "facts": facts,
            },
        )
        if not decision.should_compact:
            if active_high_risk:
                self._jsonl(
                    "lhr_estra_events.jsonl",
                    {
                        "event": "cache_hygiene_compact_delayed_active_bash",
                        "latest_stage": latest,
                        "stage_count": stage_count,
                        "facts": facts,
                    },
                )
            return None

        candidates = [c.stage_id for c in effective_cards_after if c.stage_id in self.stage_snapshots]
        candidate_node_uids = {sid: self._active_node_uid_for_stage(sid) for sid in candidates}
        switch_candidates = self._estra_switch_candidates(candidates, latest)
        can_call_estra = bool(
            getattr(self.lhr, "estra_enabled", True)
            and len(effective_cards_after) >= 2
            and candidates
            and switch_candidates
        )
        if can_call_estra:
            self.estra_decisions += 1
            compact_decision = await self._ask_estra(agent, trigger_source="context_hygiene")
            if "compact" not in compact_decision:
                compact_decision = {
                    **compact_decision,
                    "compact": self._derive_estra_compact(
                        action=str(compact_decision.get("action") or ""),
                        trigger_source="context_hygiene",
                    ),
                }
        else:
            compact_decision = self._deterministic_estra_decision(
                cards=effective_cards_after,
                candidates=candidates,
                latest=latest,
                trigger_source="context_hygiene",
                reason="context hygiene compact; deterministic fallback",
            )
            self._emit_estra_decision(
                action=str(compact_decision.get("action") or "keep_current"),
                target_stage=str(compact_decision.get("target_stage") or latest),
                latest_stage=latest,
                trigger_source="context_hygiene",
                compact=True,
                reason=str(compact_decision.get("reason") or ""),
                decision_mode="deterministic_fallback",
                candidates=candidates,
                switch_candidates=switch_candidates,
                candidate_node_uids=candidate_node_uids,
            )
        if compact_decision.get("action") not in {"keep_current", "keep_but_redirect", "switch_stage"}:
            compact_decision = self._deterministic_estra_decision(
                cards=effective_cards_after,
                candidates=candidates,
                latest=latest,
                trigger_source="context_hygiene",
                reason="context hygiene compact; invalid estra action fallback",
            )
        compact_decision = {
            **compact_decision,
            "compact": True,
            "trigger_source": "context_hygiene",
            "context_generation": f"stage_count={stage_count};tokens_since={tokens_since}",
        }
        target = str(compact_decision.get("target_stage") or latest).upper()
        await self._set_pending_estra_from_decision(agent=agent, target=target, decision=compact_decision)
        if not self.pending_estra:
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "cache_hygiene_compact_failed",
                    "latest_stage": latest,
                    "stage_count": stage_count,
                    "reason": "pending_estra_not_created",
                    "facts": facts,
                },
            )
            return None
        self.context_hygiene_last_compact_stage_count = stage_count
        self.context_hygiene_last_compact_tokens_in = tokens_in
        self.context_hygiene_last_compact_ts = time.time()
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "cache_hygiene_compact_triggered",
                "latest_stage": latest,
                "stage_count": stage_count,
                "target_stage": target,
                "action": str(compact_decision.get("action") or "keep_current"),
                "compact": True,
                "reason": str(compact_decision.get("reason") or "context_hygiene_low_cache_or_churn"),
                "facts": facts,
            },
        )
        if compact_decision.get("action") == "switch_stage":
            return f"[lnr] estra switch_stage chosen from context hygiene: {target}"
        action = str(compact_decision.get("action") or "keep_current")
        return f"[lnr] estra {action} chosen from context hygiene; compact current stage: {target}"

    async def _force_estra_after_stage_capture(self, *, agent: Any, cards_after: list[Any]) -> str | None:
        latest = cards_after[-1].stage_id if cards_after else ""
        candidates = [c.stage_id for c in cards_after if c.stage_id in self.stage_snapshots]
        candidate_node_uids = {sid: self._active_node_uid_for_stage(sid) for sid in candidates}
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "force_stage_estra_check",
                "latest_stage": latest,
                "latest_node_uid": self._active_node_uid_for_stage(latest),
                "stage_count": len(cards_after),
                "candidate_count": len(candidates),
                "candidate_node_uids": candidate_node_uids,
                "last_estra_stage_count": self.last_estra_stage_count,
                "last_force_estra_observation_count": int(getattr(self, "last_force_estra_observation_count", 0) or 0),
                "trigger_source": "force_stage_capture",
                "force_after_stage_count": int(getattr(self.lhr, "force_estra_after_stage_count", 0) or 0),
                "observed_stage_count": len(cards_after),
                "estra_observation_key": self._estra_observation_key(cards=cards_after, latest=latest, candidates=candidates),
            },
        )
        if not self._force_estra_trigger_allowed(
            cards=cards_after,
            candidates=candidates,
            observed_stage_count=len(cards_after),
        ):
            return None
        observation_key = self._estra_observation_key(cards=cards_after, latest=latest, candidates=candidates)
        self.last_estra_stage_count = max(self.last_estra_stage_count, len(cards_after))
        self.last_estra_observation_key = observation_key
        self.last_force_estra_observation_key = observation_key
        self.last_force_estra_observation_count = max(
            int(getattr(self, "last_force_estra_observation_count", 0) or 0),
            len(cards_after),
        )
        self.estra_decisions += 1
        decision = await self._ask_estra(agent, trigger_source="force_stage_capture")
        if "compact" not in decision:
            decision = {**decision, "compact": self._derive_estra_compact(action=str(decision.get("action") or ""), trigger_source="force_stage_capture")}
        if not bool(decision.get("compact")):
            return None
        target = str(decision.get("target_stage") or latest).upper()
        await self._set_pending_estra_from_decision(agent=agent, target=target, decision=decision)
        if decision.get("action") == "switch_stage":
            return f"[lnr] estra switch_stage chosen from forced stage estra: {target}"
        action = str(decision.get("action") or "keep_current")
        return f"[lnr] estra {action} chosen from forced stage estra; compact current stage: {target}"

    async def _text_only_estra_callback(
        self,
        *,
        agent: Any,
        assistant_text: str,
        round_idx: int,
        max_steps: int,
    ) -> str | None:
        if isinstance(getattr(self, "pending_text_stage_commit", None), dict):
            return await self._handle_pending_stage_commit_text(agent=agent, assistant_text=assistant_text)

        repeated_terminal_out = self._suppress_repeated_text_only_completion(
            agent=agent,
            assistant_text=assistant_text,
            round_idx=round_idx,
            max_steps=max_steps,
        )
        if repeated_terminal_out:
            return repeated_terminal_out
        cards, latest, candidates = self._estra_candidates_for_current_ledger()
        has_estra_context = bool(cards or candidates)
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "text_only_estra_check" if has_estra_context else "text_only_estra_noop",
                "noop_reason": "" if has_estra_context else "no_stage_candidates",
                "latest_stage": latest,
                "latest_node_uid": self._active_node_uid_for_stage(latest),
                "stage_count": len(cards),
                "candidate_count": len(candidates),
                "candidate_node_uids": {sid: self._active_node_uid_for_stage(sid) for sid in candidates},
                "last_estra_stage_count": self.last_estra_stage_count,
                "trigger_source": "text_only",
                "round": int(round_idx),
                "max_steps": int(max_steps),
                "assistant_text": self._compact_event_text(assistant_text),
                "assistant_text_chars": len(str(assistant_text or "")),
                "estra_observation_key": self._estra_observation_key(cards=cards, latest=latest, candidates=candidates),
            },
        )
        if not self._text_only_estra_trigger_allowed(cards=cards, candidates=candidates):
            return None
        self.last_estra_stage_count = len(cards)
        self.last_estra_observation_key = self._estra_observation_key(cards=cards, latest=latest, candidates=candidates)
        self.estra_decisions += 1
        decision = await self._ask_estra(agent, trigger_source="text_only")
        if "compact" not in decision:
            decision = {**decision, "compact": self._derive_estra_compact(action=str(decision.get("action") or ""), trigger_source="text_only")}
        decision = {**decision, "trigger_source": "text_only"}
        if not bool(decision.get("compact")):
            return None
        target = str(decision.get("target_stage") or latest).upper()
        await self._set_pending_estra_from_decision(agent=agent, target=target, decision=decision)
        if decision.get("action") == "switch_stage":
            return f"[lnr] estra switch_stage chosen from text-only: {target}"
        action = str(decision.get("action") or "keep_current")
        return f"[lnr] estra {action} chosen from text-only; compact current stage: {target}"

    async def _context_limit_estra_callback(
        self,
        *,
        agent: Any,
        round_idx: int,
        max_steps: int,
        omitted: int,
    ) -> str | None:
        _ = max_steps
        cards, latest, candidates = self._estra_candidates_for_current_ledger()
        try:
            memory_records = self._count_memory_records()
        except Exception:
            memory_records = 0
        context_generation = f"round={int(round_idx) + 1};memory_records={memory_records};omitted={int(omitted or 0)}"
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "context_limit_estra_check",
                "latest_stage": latest,
                "latest_node_uid": self._active_node_uid_for_stage(latest),
                "stage_count": len(cards),
                "candidate_count": len(candidates),
                "candidate_node_uids": {sid: self._active_node_uid_for_stage(sid) for sid in candidates},
                "last_estra_stage_count": self.last_estra_stage_count,
                "trigger_source": "context_limit",
                "omitted_before": int(omitted or 0),
                "context_generation": context_generation,
                "estra_observation_key": self._estra_observation_key(cards=cards, latest=latest, candidates=candidates),
            },
        )
        if not bool(getattr(self.lhr, "estra_enabled", True)) or len(cards) < 2 or not candidates:
            return None
        allowed = self._estra_trigger_allowed(cards=cards, candidates=candidates)
        if allowed:
            self.last_estra_stage_count = len(cards)
            self.last_estra_observation_key = self._estra_observation_key(cards=cards, latest=latest, candidates=candidates)
            self.last_context_limit_estra_generation = context_generation
            self.estra_decisions += 1
            decision = await self._ask_estra(agent, trigger_source="context_limit")
            if "compact" not in decision:
                decision = {**decision, "compact": self._derive_estra_compact(action=str(decision.get("action") or ""), trigger_source="context_limit")}
        else:
            decision = self._deterministic_estra_decision(
                cards=cards,
                candidates=candidates,
                latest=latest,
                trigger_source="context_limit",
                reason="context limit reached; deterministic estra fallback",
            )
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_deterministic_fallback",
                    "latest_stage": latest,
                    "latest_node_uid": self._active_node_uid_for_stage(latest),
                    "trigger_source": "context_limit",
                    "candidates": candidates,
                    "candidate_node_uids": {sid: self._active_node_uid_for_stage(sid) for sid in candidates},
                    "reason": decision.get("reason"),
                    "estra_action": decision.get("action"),
                    "target_stage": decision.get("target_stage"),
                    "estra_trigger_allowed": False,
                    "context_generation": context_generation,
                },
            )
        if decision.get("action") not in {"keep_current", "keep_but_redirect", "switch_stage"}:
            decision = self._deterministic_estra_decision(
                cards=cards,
                candidates=candidates,
                latest=latest,
                trigger_source="context_limit",
                reason=f"context limit estra returned {decision.get('action') or 'invalid'}; deterministic estra fallback",
            )
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_deterministic_fallback",
                    "latest_stage": latest,
                    "latest_node_uid": self._active_node_uid_for_stage(latest),
                    "trigger_source": "context_limit",
                    "candidates": candidates,
                    "candidate_node_uids": {sid: self._active_node_uid_for_stage(sid) for sid in candidates},
                    "reason": decision.get("reason"),
                    "estra_action": decision.get("action"),
                    "target_stage": decision.get("target_stage"),
                    "estra_trigger_allowed": allowed,
                    "context_generation": context_generation,
                },
            )
        target = str(decision.get("target_stage") or latest).upper()
        if not target:
            return None
        action = str(decision.get("action") or "keep_current").strip() or "keep_current"
        observation_key = self._estra_observation_key(cards=cards, latest=latest, candidates=candidates)
        restore_key = ";".join(
            [
                f"latest={latest}",
                f"target={target}",
                f"action={action}",
                f"observation={observation_key}",
                f"omitted={int(omitted or 0)}",
            ]
        )
        restore_keys = getattr(self, "context_limit_estra_restore_keys", None)
        if not isinstance(restore_keys, set):
            restore_keys = set()
            self.context_limit_estra_restore_keys = restore_keys
        if restore_key in restore_keys:
            self.last_context_limit_estra_generation = context_generation
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "context_limit_estra_restore_suppressed",
                    "latest_stage": latest,
                    "latest_node_uid": self._active_node_uid_for_stage(latest),
                    "target_stage": target,
                    "action": action,
                    "trigger_source": "context_limit",
                    "context_generation": context_generation,
                    "restore_key": restore_key,
                    "reason": "same context-limit estra restore already attempted; falling back to in-band compact",
                    "omitted_before": int(omitted or 0),
                },
            )
            return None
        restore_keys.add(restore_key)
        self.last_context_limit_estra_generation = context_generation
        decision = {
            **decision,
            "compact": True,
            "compact_strength": "strict_context_limit",
            "trigger_source": "context_limit",
            "context_generation": context_generation,
            "restore_key": restore_key,
        }
        await self._set_pending_estra_from_decision(agent=agent, target=target, decision=decision)
        if (self.pending_estra or {}).get("action") == "switch_stage":
            return f"[lnr] estra switch_stage chosen from context-limit: {target}"
        action = str((self.pending_estra or {}).get("action") or "keep_current")
        return f"[lnr] estra {action} chosen from context-limit; compact current stage: {target}"

    @classmethod
    def _parse_estra_decision(
        cls,
        text: str,
        *,
        switch_candidates: list[str],
    ) -> dict[str, Any]:
        parsed = cls._parse_estra_json(text)
        allowed = {str(c).upper() for c in switch_candidates}
        if parsed:
            action = str(parsed.get("action") or "").strip()
            target = str(parsed.get("target_stage") or parsed.get("target_step") or "").strip().upper()
            if action in {"keep_current", "keep_but_redirect"}:
                return {**parsed, "action": action}
            if action == "switch_stage" and target in allowed:
                return {**parsed, "action": "switch_stage", "target_stage": target}
            return parsed

        raw = (text or "").strip()
        if not raw:
            return {}
        reason = cls._compact_estra_reason(raw)
        switch_patterns = [
            r"(?:switch|estra|return|restore|rollback|go\s+back|back)\s+(?:to\s+)?(S\d{1,4})",
            r"(?:切到|切换到|回到|恢复到|转向)\s*(S\d{1,4})",
        ]
        for pattern in switch_patterns:
            for match in re.finditer(pattern, raw, flags=re.I):
                target = match.group(1).upper()
                if target in allowed:
                    return {"action": "switch_stage", "target_stage": target, "reason": reason}
        if re.search(r"\b(redirect|refocus)\b", raw, flags=re.I) or re.search(r"重新聚焦|调整方向|换个打法", raw):
            return {"action": "keep_but_redirect", "reason": reason, "bottleneck": reason}
        if re.search(r"\b(keep|continue|stay)\b.*\b(current|latest|trajectory|stage)\b", raw, flags=re.I) or re.search(r"保持|继续当前|不切换", raw):
            return {"action": "keep_current", "reason": reason}
        return {}

    @staticmethod
    def _compact_estra_reason(text: str, *, max_chars: int = 240) -> str:
        raw = re.sub(r"<[^>]+>", " ", text or "")
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:max_chars]

    @staticmethod
    def _compact_event_text(text: str, *, max_chars: int = 900) -> str:
        raw = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(raw) <= max_chars:
            return raw
        return raw[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _text_only_completion_signature(text: str) -> str | None:
        raw = re.sub(r"\s+", " ", str(text or "")).strip()
        if not raw:
            return None
        lower = raw.lower()
        terminal_markers = (
            "conversation has definitively concluded",
            "definitively concluded",
            "stop responding",
            "not respond further",
            "no further action",
            "no further actions",
            "all experiments concluded",
            "final status confirmed",
            "goodbye",
        )
        completion_markers = (
            "the task is complete",
            "task is complete",
            "final validation score",
            "final test score",
            "final score",
            *terminal_markers,
        )
        if not any(marker in lower for marker in completion_markers):
            return None
        parts: list[str] = []
        if "task is complete" in lower:
            parts.append("task_complete")
        if any(marker in lower for marker in terminal_markers):
            parts.append("terminal_text")
        score_re = re.compile(
            r"\b(final\s+(?:validation\s+|test\s+)?score)\s*[:=]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
            flags=re.I,
        )
        for label, value in score_re.findall(raw):
            normalized_label = re.sub(r"\s+", "_", label.strip().lower())
            parts.append(f"{normalized_label}:{value.strip().lower()}")
        return "|".join(parts) if parts else "completion_status"

    def _text_only_continue_search_prompt(self, *, signature: str, repeat_count: int) -> str:
        remaining_sec = max(0, int(float(getattr(self, "deadline", 0.0) or 0.0) - time.monotonic()))
        remaining_min = max(0, int(round(remaining_sec / 60.0)))
        return (
            "[LNR_CONTINUE_SEARCH]\n"
            f"Wall-clock budget is still active (~{remaining_min} min remaining). "
            "A text-only final/goodbye message is not a valid stop condition for this worker; "
            "workers stop only when the time budget expires. Continue the ML search now with a concrete tool call in the current workspace. "
            "Do not reply with another final/goodbye/status-only message.\n"
            "Exploration state:\n"
            "This is a continued search state after a text-only terminal response. Prior summaries are completed evidence, not a stop signal and not something to repeat. "
            "Continue from the current workspace by testing a materially different idea, verifying the current route, or preserving/restoring the best known artifact. "
            "Do not overwrite the best submission without a validated candidate.\n"
            f"Suppressed repeated terminal text signature={signature}; repeat_count={int(repeat_count)}."
        )

    def _suppress_repeated_text_only_completion(
        self,
        *,
        agent: Any,
        assistant_text: str,
        round_idx: int,
        max_steps: int,
    ) -> str | None:
        signature = self._text_only_completion_signature(assistant_text)
        if not signature:
            self._last_text_only_completion_signature = ""
            self._last_text_only_completion_repeat_count = 0
            return None
        previous = str(getattr(self, "_last_text_only_completion_signature", "") or "")
        if signature != previous:
            self._last_text_only_completion_signature = signature
            self._last_text_only_completion_repeat_count = 1
            return None
        repeat_count = int(getattr(self, "_last_text_only_completion_repeat_count", 1) or 1) + 1
        self._last_text_only_completion_repeat_count = repeat_count
        reason = f"repeated_text_only_completion:{repeat_count}"
        continue_prompt = self._text_only_continue_search_prompt(signature=signature, repeat_count=repeat_count)
        prompt_injected = False
        try:
            setattr(agent, "_lnr_suppress_current_text_only_memory", reason)
        except Exception:
            logger.debug("[lnr] could not mark repeated text-only completion for suppression", exc_info=True)
        try:
            memory = getattr(agent, "memory", None)
            if memory is not None and hasattr(memory, "add_message"):
                memory.add_message(Message.user_message(continue_prompt))
                prompt_injected = True
        except Exception:
            logger.debug("[lnr] could not inject repeated text-only continuation prompt", exc_info=True)
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "text_only_completion_duplicate_suppressed",
                "trigger_source": "text_only",
                "round": int(round_idx),
                "max_steps": int(max_steps),
                "signature": signature,
                "repeat_count": repeat_count,
                "assistant_text": self._compact_event_text(assistant_text),
                "assistant_text_chars": len(str(assistant_text or "")),
                "continue_prompt_injected": prompt_injected,
                "continue_prompt": self._compact_event_text(continue_prompt),
            },
        )
        return "[lnr] repeated text-only terminal response suppressed; continue-search prompt injected"

    @staticmethod
    def _parse_estra_json(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return {}

    @classmethod
    def _parse_stage_commit_text_block(cls, text: str) -> tuple[dict[str, Any], str, str]:
        raw = str(text or "")
        match = re.search(r"STAGE_COMMIT_BEGIN\s*(.*?)\s*STAGE_COMMIT_END", raw, flags=re.I | re.S)
        if not match:
            return {}, "", "missing_block"
        block_body = match.group(1).strip()
        block_text = "STAGE_COMMIT_BEGIN\n" + block_body + "\nSTAGE_COMMIT_END"
        data: dict[str, Any] = {}
        current_key = ""
        for line in block_body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            key_part, sep, value_part = stripped.partition(":")
            if sep:
                key = re.sub(r"[^a-z0-9_]+", "_", key_part.strip().lower()).strip("_")
                if key:
                    data[key] = value_part.strip()
                    current_key = key
                continue
            if current_key:
                data[current_key] = (str(data.get(current_key) or "") + " " + stripped).strip()
        required = ("stage_id", "metric", "metric_validity", "brief", "why", "files")
        missing = [k for k in required if not str(data.get(k) or "").strip()]
        if missing:
            return data, block_text, "missing=" + ",".join(missing)
        validity = str(data.get("metric_validity") or "").strip().lower()
        if validity not in {"high", "medium", "low"}:
            return data, block_text, "invalid_metric_validity"
        try:
            float(str(data.get("metric") or "").strip())
        except (TypeError, ValueError):
            return data, block_text, "invalid_metric"
        return data, block_text, ""

    @classmethod
    def _stage_commit_judgment_from_text_block(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "brief": parsed.get("brief") or "",
            "why": parsed.get("why") or "",
            "route_evidence": parsed.get("route_evidence") or "",
            "metric_validity": parsed.get("metric_validity") or "",
            "metric_source": parsed.get("metric_source") or "",
            "lower_is_better": parsed.get("lower_is_better") or "",
            "files": parsed.get("files") or parsed.get("stage_files") or parsed.get("artifacts") or "",
        }

    def _build_stage_commit_text_prompt(self, *, stage_id: str, metric_event: dict[str, Any], correction: str = "") -> str:
        def one(key: str, default: str = "") -> str:
            return self._stage_commit_one_line(metric_event.get(key) if isinstance(metric_event, dict) else default, max_chars=180)

        lines = [
            "[LNR_STAGE_COMMIT_REQUEST]",
            "Emit exactly one text-only STAGE_COMMIT block now. Do not call tools in this turn.",
            "Use only the observed metric/artifact facts below. Keep brief/why as compact judgments, not a log summary.",
            "The request is transient; only your STAGE_COMMIT block and the append confirmation will remain in memory.",
        ]
        if correction:
            lines.append(f"Previous block parse issue: {self._stage_commit_one_line(correction, max_chars=220)}")
        lines.extend(
            [
                "",
                "FACTS:",
                f"stage_id: {normalize_stage_id(stage_id) or stage_id}",
                f"metric: {metric_event.get('metric_value', metric_event.get('reported_val_score', 'unknown'))}",
                f"metric_name: {one('metric_name', 'Final Validation Score')}",
                f"metric_validity: {one('metric_validity', 'medium') or 'medium'}",
                f"metric_source: {one('metric_source_note') or one('val_score_type') or one('metric_protocol') or 'metric source not specified'}",
                f"lower_is_better: {self._stage_commit_bool_text(metric_event.get('lower_is_better'))}",
                f"run_time_sec: {metric_event.get('run_time_sec') or metric_event.get('wall_sec') or metric_event.get('duration_sec') or 'unknown'}",
                f"solution_path: {one('solution_path') or 'unknown'}",
                f"submission_status: {one('submission_status') or 'unknown'}",
                f"candidate_ready: {metric_event.get('candidate_ready')}",
                f"validation_issue: {one('validation_issue') or 'none'}",
                "",
                "Required output shape:",
                "STAGE_COMMIT_BEGIN",
                f"stage_id: {normalize_stage_id(stage_id) or stage_id}",
                "metric: <numeric metric>",
                "metric_validity: high|medium|low",
                "metric_source: <one short source phrase>",
                "lower_is_better: true|false",
                "run_time_sec: <seconds|unknown>",
                "brief: <one compact judgment sentence>",
                "why: <one compact reason this stage matters, including any route lesson or avoid-repeat evidence>",
                "files: <code=core.py,helper.py weights=model.ckpt; write none only when no core code/weights exist>",
                "STAGE_COMMIT_END",
            ]
        )
        return "\n".join(lines).strip()

    def _extend_stage_commit_text_policy_deadline(self, agent: Any) -> None:
        try:
            timeout = float(getattr(getattr(self, "lhr", None), "stage_commit_llm_timeout_sec", 180.0) or 180.0)
        except (TypeError, ValueError):
            timeout = 180.0
        grace_deadline = time.monotonic() + max(30.0, min(300.0, timeout + 30.0))
        policy = getattr(agent, "_run_policy", None)
        if policy is not None and hasattr(policy, "deadline_monotonic"):
            try:
                policy.deadline_monotonic = max(float(policy.deadline_monotonic), grace_deadline)
            except (TypeError, ValueError):
                policy.deadline_monotonic = grace_deadline

    def _set_stage_commit_transient_prompt(self, agent: Any, *, stage_id: str, metric_event: dict[str, Any], correction: str = "") -> None:
        prompt = self._build_stage_commit_text_prompt(stage_id=stage_id, metric_event=metric_event, correction=correction)
        setattr(agent, "_lnr_transient_user_prompt", prompt)
        # Stage commits are ledger records, not work turns. Enforce text-only at
        # the API layer and keep a run-loop guard for providers that still return
        # tool calls.
        setattr(agent, "_lnr_transient_tool_choice_none", True)
        setattr(agent, "_lnr_transient_context_mode", "stage_commit_compact")
        setattr(agent, "_lnr_stage_commit_text_pending", True)
        setattr(agent, "_lnr_stage_commit_text_handled", False)
        self._extend_stage_commit_text_policy_deadline(agent)

    def _clear_stage_commit_transient_prompt(self, agent: Any) -> None:
        for name in (
            "_lnr_transient_user_prompt",
            "_lnr_transient_tool_choice_none",
            "_lnr_transient_context_mode",
            "_lnr_transient_user_prompt_active",
            "_lnr_stage_commit_text_pending",
            "_lnr_suppress_current_text_only_memory",
        ):
            try:
                if name in {"_lnr_transient_tool_choice_none", "_lnr_transient_user_prompt_active", "_lnr_stage_commit_text_pending"}:
                    setattr(agent, name, False)
                else:
                    setattr(agent, name, "")
            except Exception:
                pass

    def _pending_stage_commit_text_active(self) -> bool:
        return isinstance(getattr(self, "pending_text_stage_commit", None), dict)

    def _stage_commit_text_memory_message(self, *, block_text: str, stage_id: str, metric_event: dict[str, Any], entry: str) -> str:
        metric = self._stage_commit_one_line(metric_event.get("metric_value"), max_chars=80)
        validity = self._stage_commit_one_line(metric_event.get("metric_validity") or "medium", max_chars=40).lower()
        checked_text = entry.strip() or block_text.strip()
        return (
            f"{checked_text}\n\n"
            "bash/edit output:\n"
            "[stage append-only write]\n"
            "path: .memory/stage_ledger.md\n"
            "status: ok\n"
            f"stage_id: {normalize_stage_id(stage_id) or stage_id}\n"
            f"metric: {metric}\n"
            f"metric_validity: {validity}\n"
            "appended: true"
        )

    def _append_stage_commit_from_judgment(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        judgment: dict[str, Any],
        block_text: str,
        source: str,
    ) -> tuple[bool, str]:
        ledger_before = read_ledger(self.ledger_path)
        entry = self._stage_commit_entry_from_metric_event(
            stage_id=stage_id,
            metric_event=metric_event,
            judgment=judgment,
            workspace_dir=self.workspace_dir,
        )
        sep = ""
        if ledger_before and not ledger_before.endswith("\n\n"):
            sep = "\n" if ledger_before.endswith("\n") else "\n\n"
        ledger_after = ledger_before + sep + entry
        ok, reason = validate_append_only_stage_commit(ledger_before, ledger_after, stage_id)
        if not ok:
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {"event": "stage_commit_append_rejected", "stage_id": stage_id, "reason": reason, "source": source},
            )
            return False, reason
        try:
            self._prepare_stage_commit_transaction(
                stage_id=stage_id,
                ledger_before=ledger_before,
                ledger_after=ledger_after,
                metric_event=metric_event,
            )
        except OSError as exc:
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_transaction_prepare_error",
                    "stage_id": stage_id,
                    "error": str(exc),
                    "source": source,
                },
            )
            return False, f"tool_error:{exc}"
        try:
            atomic_write(self.ledger_path, ledger_after)
        except OSError as exc:
            self._rollback_stage_commit_transaction(stage_id=stage_id, reason=f"ledger_write:{exc}")
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {"event": "stage_commit_append_error", "stage_id": stage_id, "error": str(exc), "source": source},
            )
            return False, f"tool_error:{exc}"
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {"event": "stage_commit_appended_text_block", "stage_id": stage_id, "judgment_source": source},
        )
        self._jsonl("lhr_stage_commit_events.jsonl", {"event": "stage_commit_ok", "stage_id": stage_id, "turn": 1})
        persist_agent_write = bool(getattr(self.lhr, "stage_commit_persist_agent_write_to_memory", True)) or bool(
            getattr(self.lhr, "stage_commit_persist_to_memory", False)
        )
        self._capture_s01_eda_prefix_end(stage_id=stage_id)
        if persist_agent_write:
            mem_text = self._stage_commit_text_memory_message(
                block_text=block_text,
                stage_id=stage_id,
                metric_event=metric_event,
                entry=entry,
            )
            try:
                mem_text = agent._sanitize_agent_visible_paths(mem_text)
            except Exception:
                pass
            memory = getattr(agent, "memory", None)
            if memory is not None and hasattr(memory, "add_message"):
                memory.add_message(Message.assistant_message(mem_text))
                self._jsonl(
                    "lhr_stage_commit_events.jsonl",
                    {
                        "event": "stage_commit_persisted_to_memory",
                        "stage_id": stage_id,
                        "turn": 1,
                        "agent_write_persisted": True,
                        "prompt_persisted": False,
                        "text_block_append": True,
                    },
                )
        return True, ""

    def _stage_transaction_root(self) -> Path:
        return self.log_dir / "stage_transactions"

    def _prepare_stage_commit_transaction(
        self,
        *,
        stage_id: str,
        ledger_before: str,
        ledger_after: str,
        metric_event: dict[str, Any],
    ) -> None:
        transaction_id = f"{self._lineage_uid_prefix()}-{stage_id}".replace(":", "-")
        # Persist the exact transaction identity in the eventual snapshot. On
        # restart this prevents an older snapshot with the same visible Stage ID
        # from completing a newer, interrupted transaction.
        metric_event["stage_transaction_id"] = transaction_id
        root = self._stage_transaction_root()
        backup = root / f"{transaction_id}.ledger_before.md"
        manifest = root / f"{transaction_id}.json"
        atomic_write(backup, ledger_before)
        payload = {
            "version": 2,
            "transaction_id": transaction_id,
            "stage_id": stage_id,
            "lineage_id": self._lineage_uid_prefix(),
            "status": "prepared",
            "prepared_at": time.time(),
            "ledger_before_path": backup.name,
            "ledger_before_sha256": hashlib.sha256(ledger_before.encode("utf-8")).hexdigest(),
            "ledger_after_sha256": hashlib.sha256(ledger_after.encode("utf-8")).hexdigest(),
            "artifact_sha": str(metric_event.get("artifact_sha") or metric_event.get("submission_sha") or ""),
            "gate_accepted": metric_event.get("gate_accepted"),
            "gate_reason_code": str(metric_event.get("gate_reason_code") or ""),
        }
        atomic_write(manifest, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.pending_stage_commit_transaction = {"manifest": manifest, "payload": payload}

    def _complete_stage_commit_transaction(self, *, stage_id: str, snapshot: StageSnapshot) -> None:
        pending = getattr(self, "pending_stage_commit_transaction", None)
        if not isinstance(pending, dict):
            return
        payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
        manifest = pending.get("manifest")
        if not isinstance(manifest, Path) or str(payload.get("stage_id") or "") != stage_id:
            return
        payload.update(
            {
                "status": "committed",
                "committed_at": time.time(),
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_path": str(snapshot.snapshot_path),
            }
        )
        atomic_write(manifest, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.pending_stage_commit_transaction = None

    def _rollback_stage_commit_transaction(self, *, stage_id: str, reason: str) -> bool:
        pending = getattr(self, "pending_stage_commit_transaction", None)
        if not isinstance(pending, dict):
            return False
        payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
        manifest = pending.get("manifest")
        if (
            not isinstance(manifest, Path)
            or str(payload.get("stage_id") or "") != stage_id
            or str(payload.get("status") or "") != "prepared"
        ):
            return False
        backup = manifest.parent / str(payload.get("ledger_before_path") or "")
        try:
            ledger_before = backup.read_text(encoding="utf-8")
            atomic_write(self.ledger_path, ledger_before)
            payload.update(
                {
                    "status": "rolled_back",
                    "rolled_back_at": time.time(),
                    "rollback_reason": reason,
                }
            )
            atomic_write(manifest, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            self.pending_stage_commit_transaction = None
            return True
        except OSError:
            logger.exception("[lnr] failed to roll back stage transaction %s", stage_id)
            return False

    def _recover_stage_commit_transactions(self) -> None:
        root = self._stage_transaction_root()
        if not root.is_dir():
            return
        discover_all = getattr(self.snapshot_store, "discover_all", None)
        snapshots = (
            list(discover_all().values())
            if callable(discover_all)
            else list(self.snapshot_store.discover().values())
        )
        for manifest in sorted(root.glob("*.json")):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("status") != "prepared":
                continue
            stage_id = str(payload.get("stage_id") or "").strip().upper()
            expected_lineage = str(payload.get("lineage_id") or "").strip()
            expected_artifact = str(payload.get("artifact_sha") or "").strip()
            expected_transaction = str(payload.get("transaction_id") or "").strip()
            snapshot = None
            for candidate in snapshots:
                if str(candidate.stage_id or "").strip().upper() != stage_id:
                    continue
                source = candidate.source_event if isinstance(candidate.source_event, dict) else {}
                candidate_artifact = str(source.get("artifact_sha") or source.get("submission_sha") or "").strip()
                candidate_transaction = str(source.get("stage_transaction_id") or "").strip()
                if expected_transaction and candidate_transaction != expected_transaction:
                    continue
                if expected_lineage and str(candidate.lineage_id or "").strip() != expected_lineage:
                    continue
                if expected_artifact and candidate_artifact != expected_artifact:
                    continue
                snapshot = candidate
                break
            if snapshot is not None:
                payload.update(
                    {
                        "status": "committed",
                        "committed_at": time.time(),
                        "recovered_after_restart": True,
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_path": str(snapshot.snapshot_path),
                    }
                )
                atomic_write(manifest, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
                continue
            backup = manifest.parent / str(payload.get("ledger_before_path") or "")
            try:
                ledger_before = backup.read_text(encoding="utf-8")
                current = read_ledger(self.ledger_path)
            except OSError:
                continue
            current_sha = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if current_sha != str(payload.get("ledger_after_sha256") or ""):
                continue
            atomic_write(self.ledger_path, ledger_before)
            payload.update(
                {
                    "status": "rolled_back",
                    "rolled_back_at": time.time(),
                    "rollback_reason": "resume_missing_snapshot",
                }
            )
            atomic_write(manifest, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            self._jsonl(
                "lhr_resume_events.jsonl",
                {
                    "event": "incomplete_stage_transaction_rolled_back",
                    "stage_id": stage_id,
                    "transaction_id": payload.get("transaction_id"),
                },
            )

    async def _finalize_stage_capture_after_commit(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        now: float,
        solution_sha: str,
        run_signature: str,
    ) -> str | None:
        try:
            return await self._finalize_stage_capture_after_commit_impl(
                agent=agent,
                stage_id=stage_id,
                metric_event=metric_event,
                now=now,
                solution_sha=solution_sha,
                run_signature=run_signature,
            )
        except BaseException as exc:
            rolled_back = self._rollback_stage_commit_transaction(
                stage_id=stage_id,
                reason=f"finalize:{type(exc).__name__}:{exc}",
            )
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_finalize_failed",
                    "stage_id": stage_id,
                    "ledger_rolled_back": rolled_back,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    async def _finalize_stage_capture_after_commit_impl(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        now: float,
        solution_sha: str,
        run_signature: str,
    ) -> str | None:
        self.last_stage_commit_ts = now
        self.last_captured_solution_sha = solution_sha
        self.last_captured_run_signature = run_signature
        metric_value_raw = metric_event.get("metric_value")
        metric_value_override = float(metric_value_raw) if isinstance(metric_value_raw, (int, float)) else None
        checkpoint = auto_checkpoint_workspace_source(
            self.workspace_dir,
            enabled=bool(
                getattr(self.lhr, "workspace_git_enabled", True)
                and getattr(self.lhr, "workspace_git_auto_checkpoint", True)
            ),
            track_globs=getattr(self.lhr, "workspace_git_track_globs", None),
            tool_name=f"lhr_stage_{stage_id.lower()}",
            metric_value_override=metric_value_override,
            stage_id=stage_id,
            submission_snapshot_dir=self.log_dir / "submission_snapshots",
            checkpoint_dir=self.log_dir / "checkpoints",
        )
        metric_event.update(
            {
                "source_commit_sha": checkpoint.commit_sha,
                "source_changed": checkpoint.source_changed,
                "workspace_git_stage_id": checkpoint.stage_id,
                "submission_snapshot": checkpoint.submission_snapshot,
                "submission_changed": checkpoint.submission_changed,
                "workspace_git_enabled": checkpoint.enabled,
                "workspace_git_ready": checkpoint.ready,
                "workspace_git_message": checkpoint.message,
                "workspace_git_ledger_path": checkpoint.ledger_path,
            }
        )
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "stage_workspace_git_checkpoint",
                "stage_id": stage_id,
                "enabled": checkpoint.enabled,
                "ready": checkpoint.ready,
                "committed": checkpoint.committed,
                "commit_sha": checkpoint.commit_sha,
                "workspace_git_stage_id": checkpoint.stage_id,
                "source_changed": checkpoint.source_changed,
                "submission_snapshot": checkpoint.submission_snapshot,
                "submission_changed": checkpoint.submission_changed,
                "ledger_path": checkpoint.ledger_path,
                "message": checkpoint.message,
            },
        )
        self._prune_workspace_control_artifacts()
        self._mark_protected_eda_prefix(agent, stage_id=stage_id)
        cards_after = parse_stage_cards(read_ledger(self.ledger_path))
        await self._adjudicate_metric_validity_for_stage(agent=agent, stage_id=stage_id, metric_event=metric_event, cards_after=cards_after)
        lineage_id = self._lineage_uid_prefix()
        node_uid = self._stage_node_uid(stage_id, lineage_id=lineage_id)
        metric_event_for_snapshot = {
            **dict(metric_event),
            "node_uid": node_uid,
            "lineage_id": lineage_id,
            "visible_stage_id": stage_id,
            "worker_id": self._worker_uid_prefix(),
        }
        if stage_id == "S01" and self._s01_eda_prefix_end_index is not None:
            metric_event_for_snapshot["protected_eda_end_index"] = int(
                self._s01_eda_prefix_end_index
            )
        append_stage_event(
            stage_log_dir(self.workspace_dir),
            "stage_capture_begin",
            stage_id=stage_id,
            node_uid=node_uid,
            metric_value=metric_event.get("metric_value"),
            metric_name=str(metric_event.get("metric_name") or ""),
        )
        snap = self.snapshot_store.capture(
            stage_id=stage_id,
            metric_value=metric_event.get("metric_value"),
            metric_name=str(metric_event.get("metric_name") or ""),
            lower_is_better=metric_event.get("lower_is_better") if isinstance(metric_event.get("lower_is_better"), bool) else None,
            memory_cut=self._count_memory_records(),
            source_event=metric_event_for_snapshot,
            node_uid=node_uid,
            lineage_id=lineage_id,
        )
        self.stage_snapshots[stage_id] = snap
        self._index_archived_stage_snapshot(snap)
        self._write_stage_map()
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "stage_captured",
                "stage_id": stage_id,
                "node_uid": node_uid,
                "lineage_id": lineage_id,
                "snapshot_id": snap.snapshot_id,
                "snapshot_path": str(snap.snapshot_path),
                "snapshot_mode": getattr(snap, "snapshot_mode", "full"),
                "snapshot_warnings": list(getattr(snap, "snapshot_warnings", ()) or ()),
                "metric_event": metric_event_for_snapshot,
                "preserved_logs": (snap.snapshot_path / "logs").is_dir(),
                "preserved_stage_logs": (snap.snapshot_path / ".logs").is_dir(),
                "preserved_memory": (snap.snapshot_path / "logs" / "memory").is_dir(),
            },
        )
        self._append_stage_performance_row(stage_id=stage_id, snap=snap, metric_event=metric_event_for_snapshot, cards_after=cards_after)
        self._complete_stage_commit_transaction(stage_id=stage_id, snapshot=snap)
        if bool(getattr(self.lhr, "stage_memory_folding_enabled", True)):
            try:
                sync_current_segment(self.workspace_dir, self._effective_stage_cards(cards_after))
            except OSError:
                logger.debug("[lnr] stage memory current segment sync failed", exc_info=True)
        forced_estra_out = await self._force_estra_after_stage_capture(agent=agent, cards_after=cards_after)
        hygiene_estra_out = None
        if not forced_estra_out:
            hygiene_estra_out = await self._context_hygiene_compact_after_stage_capture(agent=agent, cards_after=cards_after)
        self._reset_agent_interaction_stage(agent)
        if forced_estra_out:
            return forced_estra_out
        if hygiene_estra_out:
            return hygiene_estra_out
        return None

    async def _handle_pending_stage_commit_text(
        self,
        *,
        agent: Any,
        assistant_text: str,
    ) -> str | None:
        pending = getattr(self, "pending_text_stage_commit", None)
        if not isinstance(pending, dict):
            return None
        stage_id = str(pending.get("stage_id") or "").strip().upper()
        metric_event = pending.get("metric_event") if isinstance(pending.get("metric_event"), dict) else {}
        parsed, block_text, reason = self._parse_stage_commit_text_block(assistant_text)
        parsed_stage = normalize_stage_id(parsed.get("stage_id")) if parsed else ""
        if not reason and parsed_stage and parsed_stage != stage_id:
            reason = f"stage_id_mismatch:{parsed_stage}!={stage_id}"
        if reason:
            attempts = int(pending.get("attempts") or 0) + 1
            pending["attempts"] = attempts
            self.pending_text_stage_commit = pending
            setattr(agent, "_lnr_suppress_current_text_only_memory", f"stage_commit_parse_failed:{reason}")
            if attempts <= 1:
                self._set_stage_commit_transient_prompt(agent, stage_id=stage_id, metric_event=metric_event, correction=reason)
                setattr(agent, "_lnr_stage_commit_text_handled", True)
                self._jsonl(
                    "lhr_stage_commit_events.jsonl",
                    {"event": "stage_commit_text_parse_retry", "stage_id": stage_id, "reason": reason, "attempts": attempts},
                )
                return None
            judgment = self._stage_commit_fallback_judgment(metric_event)
            block_text = (
                "STAGE_COMMIT_BEGIN\n"
                f"stage_id: {stage_id}\n"
                f"metric: {metric_event.get('metric_value', 'unknown')}\n"
                f"metric_validity: {metric_event.get('metric_validity') or 'medium'}\n"
                "brief: fallback stage commit after malformed text block.\n"
                f"why: parser issue {self._stage_commit_one_line(reason, max_chars=120)}; metric event preserved; continue from preserved evidence.\n"
                f"files: {judgment['files']}\n"
                "STAGE_COMMIT_END"
            )
            source = "fallback_after_text_parse_failed"
        else:
            judgment = self._stage_commit_judgment_from_text_block(parsed)
            if parsed.get("metric_validity"):
                metric_event["metric_validity"] = str(parsed.get("metric_validity") or "").strip().lower()
            source = "main_agent_text_block"
        await self._audit_stage_result_before_commit(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            judgment=judgment,
            block_text=block_text,
            source=source,
        )
        ok, append_reason = self._append_stage_commit_from_judgment(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            judgment=judgment,
            block_text=block_text,
            source=source,
        )
        if not ok:
            if "FILES" in str(append_reason or ""):
                attempts = int(pending.get("attempts") or 0) + 1
                pending["attempts"] = attempts
                self.pending_text_stage_commit = pending
                setattr(agent, "_lnr_suppress_current_text_only_memory", f"stage_commit_append_failed:{append_reason}")
                if attempts <= 2:
                    self._set_stage_commit_transient_prompt(
                        agent,
                        stage_id=stage_id,
                        metric_event=metric_event,
                        correction=append_reason,
                    )
                    setattr(agent, "_lnr_stage_commit_text_handled", True)
                    self._jsonl(
                        "lhr_stage_commit_events.jsonl",
                        {
                            "event": "stage_commit_text_parse_retry",
                            "stage_id": stage_id,
                            "reason": append_reason,
                            "attempts": attempts,
                        },
                    )
                    return None
                fallback_judgment = self._stage_commit_fallback_judgment(metric_event)
                fallback_block = (
                    "STAGE_COMMIT_BEGIN\n"
                    f"stage_id: {stage_id}\n"
                    f"metric: {metric_event.get('metric_value', 'unknown')}\n"
                    f"metric_validity: {metric_event.get('metric_validity') or 'medium'}\n"
                    "brief: deterministic fallback after invalid FILES metadata.\n"
                    f"why: Gate evidence was accepted; bookkeeping fallback preserves it after {self._stage_commit_one_line(append_reason, max_chars=120)}.\n"
                    f"files: {fallback_judgment['files']}\n"
                    "STAGE_COMMIT_END"
                )
                fallback_source = "fallback_after_files_retry_exhausted"
                self._jsonl(
                    "lhr_stage_commit_events.jsonl",
                    {
                        "event": "stage_commit_text_fallback_applied",
                        "stage_id": stage_id,
                        "reason": append_reason,
                        "attempts": attempts,
                        "source": fallback_source,
                    },
                )
                ok, append_reason = self._append_stage_commit_from_judgment(
                    agent=agent,
                    stage_id=stage_id,
                    metric_event=metric_event,
                    judgment=fallback_judgment,
                    block_text=fallback_block,
                    source=fallback_source,
                )
                if ok:
                    block_text = fallback_block
            if not ok:
                self.pending_text_stage_commit = None
                self._clear_stage_commit_transient_prompt(agent)
                setattr(agent, "_lnr_stage_commit_text_handled", True)
                self._jsonl(
                    "lhr_stage_events.jsonl",
                    {"event": "stage_capture_failed", "stage_id": stage_id, "reason": append_reason, "metric_event": metric_event},
                )
                return None
        self.pending_text_stage_commit = None
        self._clear_stage_commit_transient_prompt(agent)
        setattr(agent, "_lnr_stage_commit_text_handled", True)
        return await self._finalize_stage_capture_after_commit(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            now=float(pending.get("now") or time.monotonic()),
            solution_sha=str(pending.get("solution_sha") or ""),
            run_signature=str(pending.get("run_signature") or ""),
        )

    def _evaluator_stage_source_mode(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        mode = str(getattr(evaluator, "stage_source_mode", "primary") or "primary").strip().lower()
        return mode if mode in {"shadow", "adjudicate", "primary"} else "primary"

    def _evaluator_event_log_name(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        raw = str(getattr(evaluator, "event_log", "evaluator_events.jsonl") or "evaluator_events.jsonl").strip()
        if not raw or "/" in raw or "\\" in raw or raw.startswith("."):
            return "evaluator_events.jsonl"
        return raw

    def _evaluator_task_profile(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        return str(
            getattr(evaluator, "task_profile", "")
            or getattr(cfg, "task_profile", "auto")
            or "auto"
        )

    def _code_organization_hint(self) -> str:
        hint = str(getattr(self.lhr, "code_organization_hint", "") or "").strip()
        profile = self._evaluator_task_profile().strip().lower().replace("-", "_")
        if profile in {"opt_solver", "optimization_solver", "artifact_solver"}:
            default_hints = {"", "beyond_mfiles", "beyond_multifile"}
            if hint.lower().replace("-", "_").replace(" ", "_") in default_hints:
                return "opt_solver"
        return hint

    def _evaluator_candidate_artifact(self) -> str:
        cfg = getattr(self, "cfg", None)
        evaluator = getattr(cfg, "evaluator", None)
        candidate = getattr(evaluator, "candidate", None)
        artifact = str(getattr(candidate, "artifact", "") or "").strip()
        spec = self._task_package_spec()
        if spec is not None and self._evaluator_backend_name() == "task_package" and artifact in {"", "submission.csv"}:
            return spec.artifact_path
        return artifact

    def _evaluator_candidate_artifact_kind(self) -> str:
        evaluator = getattr(self.cfg, "evaluator", None)
        candidate = getattr(evaluator, "candidate", None)
        kind = str(getattr(candidate, "artifact_kind", "") or "").strip()
        spec = self._task_package_spec()
        if spec is not None and self._evaluator_backend_name() == "task_package" and kind in {"", "submission_csv"}:
            return spec.artifact_kind
        return kind

    def _task_package_spec(self) -> Any | None:
        try:
            return find_task_package(str(getattr(self.cfg, "exp_id", "") or ""))
        except Exception:
            logger.debug("[lnr] task package lookup failed", exc_info=True)
            return None

    def _evaluator_backend_name(self) -> str:
        evaluator = getattr(self.cfg, "evaluator", None)
        configured = str(getattr(evaluator, "backend", "") or "").strip()
        if configured.lower() not in {"", "auto"}:
            return configured
        manager = getattr(self, "evaluator_manager", None)
        if manager is None:
            return ""
        try:
            ctx = EvalContext(
                task_profile=self._evaluator_task_profile(),
                task_id=str(getattr(self.cfg, "exp_id", "") or ""),
                task_root=self.task_root_dir,
                workspace=self.workspace_dir,
                worker_id=self._worker_uid_prefix(),
                cfg=self.cfg,
            )
            return str(manager.backend_name_for_context(ctx) or "").strip()
        except Exception:
            logger.debug("[lnr] evaluator backend resolution failed", exc_info=True)
            return ""

    def _task_runtime_context(self) -> Any:
        return SimpleNamespace(cfg=self.cfg, task_python_executable="")

    def _task_python_executable(self) -> Path | None:
        try:
            return task_python_executable(self._task_runtime_context())
        except Exception:
            logger.debug("[lnr] task python resolution failed", exc_info=True)
            return None

    def _task_runtime_extra_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "SCIENCEFLOW_TASK_PROFILE": self._evaluator_task_profile(),
            "SCIENCEFLOW_EVALUATOR_BACKEND": self._evaluator_backend_name(),
            "SCIENCEFLOW_CANDIDATE_ARTIFACT": self._evaluator_candidate_artifact(),
        }
        artifact_kind = self._evaluator_candidate_artifact_kind()
        if artifact_kind:
            env["SCIENCEFLOW_CANDIDATE_ARTIFACT_KIND"] = artifact_kind
        python_path = self._task_python_executable()
        if not python_path:
            return env
        env["SCIENCEFLOW_TASK_PYTHON"] = str(python_path)
        try:
            selected_env = task_command_env(self._task_runtime_context(), python=python_path)
        except Exception:
            logger.debug("[lnr] task command environment failed", exc_info=True)
            selected_env = {}
        for key in ("PATH", "CONDA_PREFIX", "VIRTUAL_ENV"):
            value = str(selected_env.get(key) or "").strip()
            if value:
                env[key] = value
        return env

    def _task_runtime_prompt_contract(self) -> str:
        lines: list[str] = []
        artifact = self._evaluator_candidate_artifact()
        if artifact:
            lines.append(f"- Candidate artifact path: `{artifact}`.")
            lines.append("- Stage capture and resume use this artifact path as the deliverable signal.")
        python_path = self._task_python_executable()
        if python_path:
            lines.extend(
                [
                    f"- Task Python executable: `{python_path}`.",
                    "- In bash commands, use `$SCIENCEFLOW_TASK_PYTHON` for task dependencies; `python` is also pointed at this environment.",
                    "- Do not install task dependencies into the controller `.venv` or a new ad-hoc workspace environment unless this configured Python fails.",
                ]
            )
        return "\n".join(lines)

    def _workspace_state_deliverable_line(self) -> str:
        artifact = self._evaluator_candidate_artifact()
        profile = self._evaluator_task_profile().strip().lower()
        if profile in {"", "default", "mlebench"} and artifact == "submission.csv":
            return "A valid run must create root-level submission.csv and print Final Validation Score: <float>."
        return f"A valid run must create `{artifact}` and let the configured evaluator report the authoritative metric."

    def _evaluator_prompt_contract(self) -> str:
        service = getattr(self, "gate_service", None) or getattr(self, "evaluation_service", None)
        if service is None:
            return ""
        evaluator_cfg = getattr(self.cfg, "evaluator", None)
        if getattr(evaluator_cfg, "enabled", True) is False:
            return ""
        try:
            ctx = EvalContext(
                task_profile=self._evaluator_task_profile(),
                task_id=str(getattr(self.cfg, "exp_id", "") or ""),
                task_root=self.task_root_dir,
                workspace=self.workspace_dir,
                worker_id=self._worker_uid_prefix(),
                cfg=self.cfg,
            )
            request = EvaluationRequest(context=ctx, trigger="prompt")
            return service.build_prompt_contract(request)
        except Exception:
            logger.debug("[lnr] evaluator prompt contract failed", exc_info=True)
            return ""

    def _record_evaluator_stage_events(self, *, stage_id: str, metric_event: dict[str, Any]) -> dict[str, Any]:
        service = getattr(self, "gate_service", None) or getattr(self, "evaluation_service", None)
        if service is None:
            return metric_event
        mode = self._evaluator_stage_source_mode()

        def merge_facts(facts: dict[str, Any]) -> dict[str, Any]:
            if mode == "shadow":
                return metric_event
            if mode == "adjudicate":
                return merge_adjudicated_stage_facts(metric_event, facts)
            return merge_primary_stage_facts(metric_event, facts)

        def failed_facts(reason_code: str, message: str) -> dict[str, Any]:
            return {
                "validation_ok": False,
                "candidate_ready": False,
                "selection_eligible": False,
                "metric_validity": "low",
                "metric_validity_reason_code": reason_code,
                "metric_source_note": message,
                "evaluator_backend": self._evaluator_backend_name(),
                "evaluator_status": reason_code,
                "gate_action": "retry",
                "gate_accepted": False,
                "gate_reason_code": reason_code,
                "gate_message": message,
                "_gate_evaluated": True,
            }

        try:
            ctx = EvalContext(
                task_profile=self._evaluator_task_profile(),
                task_id=str(getattr(self.cfg, "exp_id", "") or ""),
                task_root=self.task_root_dir,
                workspace=self.workspace_dir,
                worker_id=self._worker_uid_prefix(),
                stage_id=str(stage_id or ""),
                cfg=self.cfg,
                wall_clock_remaining_sec=max(0.0, float(self.deadline - time.monotonic())),
                metadata={"metric_event": dict(metric_event or {})},
            )
            request = EvaluationRequest(context=ctx, trigger="stage_end")
            outcomes = list(service.evaluate(request))
            selected_facts: dict[str, Any] = {}
            for outcome in outcomes:
                event = outcome.event
                facts = metric_event_to_stage_facts(event)
                gate_trace = dict((event.extra or {}).get("gate") or {})
                facts.update(
                    {
                        # Preserve the evidence level seen by Gate separately
                        # from later Stage-result adjudication, which may lower
                        # metric_validity for selection without rewriting the
                        # historical Gate decision.
                        "gate_metric_validity": event.metric_validity,
                        "gate_policy": str(gate_trace.get("policy") or ""),
                        "gate_policy_version": str(gate_trace.get("version") or ""),
                        "gate_action": outcome.decision.action,
                        "gate_accepted": outcome.decision.accepted,
                        "gate_reason_code": outcome.decision.reason_code,
                        "gate_message": outcome.decision.message,
                        "_gate_evaluated": True,
                    }
                )
                self._jsonl(
                    self._evaluator_event_log_name(),
                    {
                        "event": "evaluator_metric_event",
                        "gate_decision": outcome.decision.to_dict(),
                        "stage_source_mode": mode,
                        "stage_facts": facts,
                        **event.to_dict(),
                    },
                )
                if not selected_facts:
                    selected_facts = facts
            if len(outcomes) != 1:
                reason = "evaluator_no_outcome" if not outcomes else "evaluator_multiple_outcomes"
                message = (
                    "evaluator did not produce a candidate outcome"
                    if not outcomes
                    else f"evaluator produced {len(outcomes)} outcomes for one stage candidate"
                )
                self._jsonl(
                    self._evaluator_event_log_name(),
                    {
                        "event": "stage_gate_failed_closed",
                        "stage_id": stage_id,
                        "stage_source_mode": mode,
                        "reason_code": reason,
                        "message": message,
                    },
                )
                return merge_facts(failed_facts(reason, message))
            return merge_facts(selected_facts)
        except Exception as exc:
            logger.debug("[lnr] evaluator stage event failed", exc_info=True)
            reason = "evaluator_service_exception"
            message = f"{type(exc).__name__}: {exc}"
            self._jsonl(
                self._evaluator_event_log_name(),
                {
                    "event": "stage_gate_failed_closed",
                    "stage_id": stage_id,
                    "stage_source_mode": mode,
                    "reason_code": reason,
                    "message": message,
                },
            )
            return merge_facts(failed_facts(reason, message))

    def _candidate_artifact_sha_from_workspace(self, metric_event: dict[str, Any]) -> str:
        candidates: list[str] = []
        explicit = str(metric_event.get("artifact_path") or "").strip()
        if explicit:
            candidates.append(explicit)
        configured = self._evaluator_candidate_artifact()
        if configured and configured not in candidates:
            candidates.append(configured)
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = self.workspace_dir / path
            if not path.is_file():
                continue
            try:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()
            except OSError:
                logger.debug("[lnr] candidate artifact sha failed: %s", path, exc_info=True)
        return ""

    def _evaluator_feedback_for_agent(self, metric_event: dict[str, Any]) -> str:
        return format_invalid_evaluator_feedback(
            metric_event,
            candidate_artifact=self._evaluator_candidate_artifact(),
        )

    def _archive_candidate_artifact_after_tool(
        self,
        *,
        agent: Any,
        tool_name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> None:
        _ = agent, args
        artifact_path = self._evaluator_candidate_artifact()
        is_submission = artifact_path.replace("\\", "/").strip() == "submission.csv"
        snapshot_dir = self.log_dir / ("submission_snapshots" if is_submission else "artifact_snapshots")
        result = archive_workspace_candidate_artifact(
            self.workspace_dir,
            artifact_path=artifact_path,
            artifact_kind=self._evaluator_candidate_artifact_kind(),
            snapshot_dir=snapshot_dir,
            ledger_path=self.log_dir / "checkpoints" / "artifact_archive.jsonl",
            trigger=tool_name,
            tool_error=bool(tool_result.error),
        )
        if not result.archived and result.ready:
            return
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "candidate_artifact_archived" if result.archived else "candidate_artifact_archive_error",
                "capture_type": "candidate_artifact_persisted",
                "trigger": tool_name,
                "tool_error": bool(tool_result.error),
                "artifact_path": result.artifact_path,
                "artifact_kind": result.artifact_kind,
                "artifact_sha": result.artifact_sha256,
                "artifact_snapshot": result.snapshot_path,
                "archive_ledger_path": result.ledger_path,
                "size_bytes": result.size_bytes,
                "selection_eligible": None,
                "metric_value": None,
                "message": result.message,
            },
        )

    def _materialized_existing_stage_target(
        self,
        *,
        cards_before: list[Any],
        metric_event: dict[str, Any],
        semantic_source_changed: bool,
    ) -> str:
        if semantic_source_changed:
            return ""
        if not self._csv_bool(metric_event.get("candidate_ready"), default=False):
            return ""
        if not str(metric_event.get("submission_sha") or "").strip():
            return ""
        if not cards_before:
            return ""
        latest = cards_before[-1]
        target_stage = normalize_stage_id(getattr(latest, "stage_id", ""))
        if not target_stage:
            return ""
        snap = self.stage_snapshots.get(target_stage)
        source_event = snap.source_event if snap is not None and isinstance(snap.source_event, dict) else {}
        if self._csv_bool(source_event.get("candidate_ready"), default=False):
            return ""
        previous_metric = self._metric_value_float(getattr(latest, "metric", ""))
        current_metric = self._metric_value_float(metric_event.get("metric_value"))
        if previous_metric is None or current_metric is None:
            return ""
        if abs(previous_metric - current_metric) > max(1e-9, abs(previous_metric) * 1e-6):
            return ""
        previous_solution = str(source_event.get("solution_sha") or "").strip()
        current_solution = str(metric_event.get("solution_sha") or "").strip()
        if previous_solution and current_solution and previous_solution != current_solution:
            return ""
        return target_stage

    def _record_stage_materialization_event(
        self,
        *,
        target_stage: str,
        metric_event: dict[str, Any],
        now: float,
        solution_sha: str,
        run_signature: str,
    ) -> None:
        self.last_stage_commit_ts = now
        self.last_captured_solution_sha = solution_sha
        self.last_captured_run_signature = run_signature
        stage_id = normalize_stage_id(target_stage)
        summary = "Ready submission generated from the same route; no new research stage."
        append_stage_event_summary(self.ledger_path, target_stage=stage_id, summary=summary)
        metric_value_raw = metric_event.get("metric_value")
        metric_value_override = float(metric_value_raw) if isinstance(metric_value_raw, (int, float)) else None
        checkpoint = auto_checkpoint_workspace_source(
            self.workspace_dir,
            enabled=bool(
                getattr(self.lhr, "workspace_git_enabled", True)
                and getattr(self.lhr, "workspace_git_auto_checkpoint", True)
            ),
            track_globs=getattr(self.lhr, "workspace_git_track_globs", None),
            tool_name=f"lhr_stage_event_{stage_id.lower()}",
            metric_value_override=metric_value_override,
            stage_id=f"{stage_id}-event",
            submission_snapshot_dir=self.log_dir / "submission_snapshots",
            checkpoint_dir=self.log_dir / "checkpoints",
        )
        snap = self.stage_snapshots.get(stage_id)
        if snap is not None and isinstance(snap.source_event, dict):
            snap.source_event.update(
                {
                    "candidate_ready": True,
                    "submission_status": "ready",
                    "materialized_ready_submission": True,
                    "submission_sha": metric_event.get("submission_sha"),
                    "submission_snapshot": checkpoint.submission_snapshot,
                    "capture_type": "stage_event_materialization",
                }
            )
            self._write_stage_map()
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "stage_materialized_ready_submission",
                "target_stage": stage_id,
                "metric_value": metric_event.get("metric_value"),
                "metric_name": metric_event.get("metric_name"),
                "solution_sha": solution_sha,
                "submission_sha": metric_event.get("submission_sha"),
                "semantic_source_changed": metric_event.get("semantic_source_changed"),
                "candidate_ready": True,
                "stage_policy": "materialized_existing_stage_no_new_research_stage",
                "workspace_git_enabled": checkpoint.enabled,
                "workspace_git_ready": checkpoint.ready,
                "workspace_git_stage_id": checkpoint.stage_id,
                "source_commit_sha": checkpoint.commit_sha,
                "source_changed": checkpoint.source_changed,
                "submission_snapshot": checkpoint.submission_snapshot,
                "submission_changed": checkpoint.submission_changed,
                "workspace_git_message": checkpoint.message,
                "ledger_summary": summary,
            },
        )

    async def _stage_capture_callback(self, *, agent: Any, args: dict[str, Any], tool_result: ToolResult) -> str | None:
        _ = args, tool_result
        if not bool(self.lhr.stage_capture_enabled):
            return None
        now = time.monotonic()
        min_gap = max(0.0, float(self.lhr.stage_commit_min_seconds_between or 0.0))
        if self.last_stage_commit_ts and now - self.last_stage_commit_ts < min_gap:
            return None
        metric_event = self._metric_event_from_workspace()
        if not metric_event:
            if self._evaluator_stage_source_mode() != "primary":
                return None
            cards_before_for_eval = parse_stage_cards(read_ledger(self.ledger_path))
            stage_id_for_eval = self._next_active_stage_id(cards_before_for_eval)
            # Tool callbacks are frequent. If the configured artifact is byte-
            # identical to a committed Stage, this is not a new candidate and
            # must not produce another Gate accept observation. This is an
            # idempotency check, not a stricter scientific policy.
            try:
                observed_artifact_sha = self._candidate_artifact_sha_from_workspace({})
            except Exception:
                observed_artifact_sha = ""
            prior_snapshot = self._stage_with_artifact_sha(
                self.stage_snapshots,
                observed_artifact_sha,
            )
            if prior_snapshot is not None:
                self._jsonl(
                    "lhr_stage_events.jsonl",
                    {
                        "event": "duplicate_candidate_pre_gate_skipped",
                        "stage_id": stage_id_for_eval,
                        "artifact_sha": observed_artifact_sha,
                        "duplicate_of_stage": prior_snapshot.stage_id,
                        "duplicate_of_snapshot_id": prior_snapshot.snapshot_id,
                        "stage_policy": "deduplicate_before_gate",
                    },
                )
                return None
            if observed_artifact_sha and observed_artifact_sha == str(
                getattr(self, "_last_metric_missing_gate_artifact_sha", "") or ""
            ):
                self._jsonl(
                    "lhr_stage_events.jsonl",
                    {
                        "event": "duplicate_pending_metric_pre_gate_skipped",
                        "stage_id": stage_id_for_eval,
                        "artifact_sha": observed_artifact_sha,
                        "stage_policy": "deduplicate_metric_missing_before_gate",
                    },
                )
                return None
            metric_event = self._record_evaluator_stage_events(stage_id=stage_id_for_eval, metric_event={})
            if metric_event.get("_gate_evaluated") and metric_event.get("gate_accepted") is not True:
                if (
                    observed_artifact_sha
                    and metric_event.get("gate_reason_code") == "metric_missing"
                ):
                    self._last_metric_missing_gate_artifact_sha = observed_artifact_sha
                feedback = self._evaluator_feedback_for_agent(metric_event)
                return feedback or None
            if not metric_event or metric_event.get("metric_value") is None:
                feedback = self._evaluator_feedback_for_agent(metric_event)
                if feedback:
                    return feedback
                return None
            metric_event.setdefault("capture_type", "evaluator_primary")
            metric_event.setdefault("candidate_ready", True)
            metric_event.setdefault("submission_status", metric_event.get("evaluator_status") or "ready")
            metric_event.setdefault("semantic_source_changed", True)
            metric_event.setdefault("artifact_sha", metric_event.get("submission_sha") or "")
            metric_event["_evaluator_stage_facts_applied"] = True
        else:
            self._last_metric_missing_gate_artifact_sha = ""
        if not str(metric_event.get("artifact_sha") or "").strip():
            artifact_sha_from_file = self._candidate_artifact_sha_from_workspace(metric_event)
            if artifact_sha_from_file:
                metric_event["artifact_sha"] = artifact_sha_from_file
                metric_event.setdefault("artifact_path", self._evaluator_candidate_artifact())
        solution_sha = str(metric_event.get("solution_sha") or "")
        run_signature = self._stage_run_signature(metric_event)
        if not run_signature and metric_event.get("artifact_sha"):
            run_signature = "evaluator:" + str(metric_event.get("artifact_sha") or "")
        if run_signature and run_signature == str(getattr(self, "last_captured_run_signature", "") or ""):
            return None
        generic_evaluator_capture = self._evaluator_stage_source_mode() == "primary" and bool(metric_event.get("evaluator_backend"))
        submission_sha = str(metric_event.get("submission_sha") or "").strip()
        artifact_sha = str(metric_event.get("artifact_sha") or "").strip()
        submission_status_hint = str(metric_event.get("submission_status") or "").strip()
        submission_invalid = (
            metric_event.get("submission_validation_ok") is False
            or submission_status_hint == "invalid_submission"
        )
        duplicate = (
            self._stage_with_submission_sha(self.stage_snapshots, submission_sha)
            or self._stage_with_artifact_sha(self.stage_snapshots, artifact_sha)
        )
        semantic_source_changed: bool | None = None
        if bool(getattr(self.lhr, "workspace_git_enabled", True)):
            try:
                semantic_source_changed = workspace_source_changed(
                    self.workspace_dir,
                    getattr(self.lhr, "workspace_git_track_globs", None),
                )
            except Exception as exc:
                metric_event["semantic_source_change_error"] = str(exc)
        if semantic_source_changed is None:
            raw_source_changed = metric_event.get("semantic_source_changed", metric_event.get("source_changed"))
            semantic_source_changed = raw_source_changed if isinstance(raw_source_changed, bool) else True
        metric_event["semantic_source_changed"] = bool(semantic_source_changed)
        metric_event.setdefault("capture_type", "full_stage")
        if submission_invalid:
            metric_event.update(
                {
                    "candidate_ready": False,
                    "submission_status": "invalid_submission",
                    "capture_type": "invalid_submission_skipped",
                }
            )
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_invalid_submission_skipped",
                    "submission_sha": submission_sha,
                    "solution_sha": solution_sha,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "submission_status": "invalid_submission",
                    "candidate_ready": False,
                    "stage_policy": "skip_invalid_submission_not_a_stage",
                },
            )
            return None
        elif duplicate is not None:
            same_design_duplicate = self._duplicate_stage_same_design(metric_event, duplicate)
            force_capture_duplicate = bool(
                getattr(self.lhr, "force_estra_capture_duplicate_submissions", False)
                and int(getattr(self.lhr, "force_estra_after_stage_count", 0) or 0) > 0
            )
            metric_event.update(
                {
                    "candidate_ready": False,
                    "submission_status": "duplicate_or_stale_submission",
                    "duplicate_submission_of_stage": duplicate.stage_id,
                    "duplicate_submission_of_snapshot_id": duplicate.snapshot_id,
                    "capture_type": "duplicate_capture",
                }
            )
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            if same_design_duplicate:
                stage_policy = "duplicate_candidate_same_design"
            elif bool(semantic_source_changed):
                stage_policy = "duplicate_candidate_same_artifact_different_source"
            else:
                stage_policy = "duplicate_candidate_no_semantic_workspace_change"
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_duplicate_submission_skipped",
                    "duplicate_of_stage": duplicate.stage_id,
                    "duplicate_of_snapshot_id": duplicate.snapshot_id,
                    "artifact_sha": artifact_sha,
                    "force_capture_requested": bool(force_capture_duplicate),
                    "submission_sha": submission_sha,
                    "solution_sha": solution_sha,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "semantic_source_changed": bool(semantic_source_changed),
                    "same_design_duplicate": bool(same_design_duplicate),
                    "stage_policy": stage_policy,
                },
            )
            return None
        elif generic_evaluator_capture:
            metric_event.setdefault("candidate_ready", True)
            metric_event.setdefault("submission_status", metric_event.get("evaluator_status") or "ready")
        elif not submission_sha and not artifact_sha:
            metric_event.update(
                {
                    "candidate_ready": False,
                    "submission_status": "missing_submission",
                }
            )
        else:
            metric_event.update(
                {
                    "candidate_ready": True,
                    "submission_status": "ready",
                }
            )
        cards_before = parse_stage_cards(read_ledger(self.ledger_path))
        stage_id = self._next_active_stage_id(cards_before)
        if not metric_event.pop("_evaluator_stage_facts_applied", False):
            metric_event = self._record_evaluator_stage_events(stage_id=stage_id, metric_event=metric_event)
        gate_evaluated = bool(metric_event.get("_gate_evaluated"))
        if (
            self._evaluator_stage_source_mode() != "shadow"
            and gate_evaluated
            and metric_event.get("gate_accepted") is not True
        ):
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_gate_rejected",
                    "stage_id": stage_id,
                    "gate_action": metric_event.get("gate_action"),
                    "gate_reason_code": metric_event.get("gate_reason_code"),
                    "gate_message": metric_event.get("gate_message"),
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "artifact_path": metric_event.get("artifact_path"),
                    "artifact_sha": metric_event.get("artifact_sha"),
                    "candidate_ready": metric_event.get("candidate_ready"),
                    "selection_eligible": metric_event.get("selection_eligible"),
                    "stage_policy": "gate_accept_required_for_stage",
                },
            )
            feedback = self._evaluator_feedback_for_agent(metric_event)
            return feedback or None
        mark_valid = getattr(agent, "_lnr_mark_valid_bare_run", None)
        if callable(mark_valid) and str(metric_event.get("solution_sha") or ""):
            mark_valid(
                bash_cmd=str(metric_event.get("bash_cmd") or ""),
                solution_rel_path=str(metric_event.get("solution_path") or "solution.py"),
            )
        materialized_stage = self._materialized_existing_stage_target(
            cards_before=cards_before,
            metric_event=metric_event,
            semantic_source_changed=bool(semantic_source_changed),
        )
        if materialized_stage:
            self._record_stage_materialization_event(
                target_stage=materialized_stage,
                metric_event=metric_event,
                now=now,
                solution_sha=solution_sha,
                run_signature=run_signature,
            )
            return None
        stage_cap = int(getattr(self.lhr, "stage_capture_max_count", 0) or 0)
        if stage_cap > 0 and len(cards_before) >= stage_cap:
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_capture_cap_reached",
                    "observed_stage_count": len(cards_before),
                    "stage_capture_max_count": stage_cap,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "candidate_ready": metric_event.get("candidate_ready"),
                    "worker_id": self._worker_uid_prefix(),
                },
            )
            return None
        primary_evaluator_capture = self._evaluator_stage_source_mode() == "primary" and bool(metric_event.get("evaluator_backend"))
        if primary_evaluator_capture and (
            metric_event.get("validation_ok") is False
            or not self._csv_bool(metric_event.get("candidate_ready"), default=False)
        ):
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            metric_event["candidate_ready"] = False
            metric_event.setdefault("submission_status", metric_event.get("evaluator_status") or "invalid_artifact")
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_invalid_evaluator_candidate_skipped",
                    "stage_id": stage_id,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "artifact_path": metric_event.get("artifact_path"),
                    "evaluator_backend": metric_event.get("evaluator_backend"),
                    "evaluator_status": metric_event.get("evaluator_status"),
                    "submission_status": metric_event.get("submission_status"),
                    "candidate_ready": False,
                    "stage_policy": "skip_invalid_evaluator_candidate_not_a_stage",
                },
            )
            feedback = self._evaluator_feedback_for_agent(metric_event)
            return feedback or None
        if primary_evaluator_capture and metric_event.get("metric_value") is None:
            feedback = self._evaluator_feedback_for_agent(metric_event)
            if feedback:
                return feedback
            return None
        if bool(getattr(self.lhr, "stage_commit_text_mode", True)):
            self.pending_text_stage_commit = {
                "stage_id": stage_id,
                "metric_event": copy.deepcopy(metric_event),
                "now": now,
                "solution_sha": solution_sha,
                "run_signature": run_signature,
                "attempts": 0,
            }
            self._set_stage_commit_transient_prompt(agent, stage_id=stage_id, metric_event=metric_event)
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_text_requested",
                    "stage_id": stage_id,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "candidate_ready": metric_event.get("candidate_ready"),
                },
            )
            return None

        ok, reason = await self._ephemeral_stage_commit(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
        )
        if not ok:
            self._jsonl(
                "lhr_stage_events.jsonl",
                {"event": "stage_capture_failed", "stage_id": stage_id, "reason": reason, "metric_event": metric_event},
            )
            return None
        return await self._finalize_stage_capture_after_commit(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            now=now,
            solution_sha=solution_sha,
            run_signature=run_signature,
        )


    def _make_agent(self, *, load_existing_memory: bool) -> Any:
        policy = _WallClockAutoContinuePolicy(deadline_monotonic=self.deadline, max_text_only_retries=0)
        node_id = "lnr" if not self.worker_id else f"lnr:{self.worker_id}"
        process_id = getattr(self.cfg, "exp_id", "") or "lhr"
        if self.worker_id:
            process_id = f"{process_id}:{self.worker_id}"
        llm_stage_override = self._worker_llm_stage_override()
        hook = self.orchestrator.make_llm_call_tracer(
            node_id=node_id,
            process_id=process_id,
            detail_prefix="mode=lnr;role=main_agent"
            + (f";worker={self.worker_id}" if self.worker_id else ""),
        )
        agent_extra_env = dict(self.worker_extra_env)
        agent_extra_env.update(self._task_runtime_extra_env())
        agent = self.orchestrator.create_science_agent(
            task_description=None,
            run_policy=policy,
            max_steps_override=max(1, int(self.lhr.max_steps or 1)),
            append_repl_system_prompt=True,
            pin_task_description=False,
            on_llm_call=hook,
            memory_dir_override=self.memory_dir,
            load_existing_memory=load_existing_memory,
            memory_agent_name="ScienceAgent",
            teleport_mode="off",
            repl_bash_write_mode=True,
            stable_system_prompt=True,
            pin_environment_context=False,
            code_organization_hint=self._code_organization_hint(),
            workspace_git_enabled=bool(getattr(self.lhr, "workspace_git_enabled", True)),
            workspace_git_track_globs=list(
                normalize_workspace_git_track_globs(getattr(self.lhr, "workspace_git_track_globs", None)),
            ),
            workspace_git_auto_review=bool(getattr(self.lhr, "workspace_git_auto_review", False)),
            # LNR owns source checkpoints at metric-backed stage boundaries, with
            # ledgers under worker logs/checkpoints. Disabling the agent-level
            # per-tool checkpoint avoids a second workspace/.scienceflow_checkpoints ledger.
            workspace_git_auto_checkpoint=False,
            bash_max_output_chars_override=int(getattr(self.cfg, "repl_bash_max_output_chars", 8000) or 8000),
            bash_max_stream_line_chars_override=int(getattr(self.cfg, "repl_bash_max_stream_line_chars", 2400) or 2400),
            bash_observation_summary_override=True,
            interaction_log_layout="split",
            extra_env_override=agent_extra_env,
            skill_registry=self.skill_registry,
            task_type=self.skill_task_category or None,
            skill_allow_names=self.skill_allow_names,
            skill_tool_mode=self.skill_tool_mode,
            skill_allow_generic_wildcard=self.skill_allow_generic_wildcard,
            skill_visible_max=self.skill_visible_max,
            llm_stage_override=llm_stage_override,
        )
        self._resource_main_agent_ref = agent
        setattr(agent, "_workspace_relative_path_mode", True)
        setattr(agent, "_agent_hidden_workspace_filenames", (self.ledger_filename,))
        setattr(
            agent,
            "_agent_hidden_workspace_path_prefixes",
            (
                self.ledger_filename,
                "logs",
                "submission_snapshots",
                ".logs",
                ".agent_memory",
                ".git",
                ".memory",
                ".scienceflow_checkpoints",
            ),
        )
        hidden_denied_prefixes = (
            "logs",
            ".logs",
            ".agent_memory",
            ".git",
            ".memory",
            ".scienceflow_checkpoints",
            "stage_memory",
            "submission_snapshots",
            "submission_history",
            "submissions",
        )
        apply_denials = getattr(agent, "_apply_agent_hidden_path_denials", None)
        if callable(apply_denials):
            apply_denials(hidden_denied_prefixes)
        setattr(agent, "_agentic_route_log_dir_override", self.log_dir / "agentic_route")
        self._attach_lnr_interaction_logger(agent)
        bash_tool = (getattr(getattr(agent, "availableTools", None), "tool_map", {}) or {}).get("bash")
        if bash_tool is not None:
            try:
                bash_tool.forbid_host_absolute_paths = True
                bash_tool.resource_observer = self.resource_observer
                bash_tool.resource_progress_heartbeat_min_interval_sec = float(
                    getattr(self.lhr, "resource_progress_heartbeat_min_interval_sec", 30.0) or 0.0,
                )
                bash_tool.resource_artifact_heartbeat_scan_interval_sec = float(
                    getattr(self.lhr, "resource_artifact_heartbeat_scan_interval_sec", 30.0) or 30.0,
                )
                bash_tool.resource_artifact_recoverable_settle_sec = float(
                    getattr(self.lhr, "resource_artifact_recoverable_settle_sec", 5.0) or 0.0,
                )
                bash_tool.resource_recoverable_stop_enabled = bool(
                    getattr(self.lhr, "resource_recoverable_stop_enabled", True),
                )
                bash_tool.resource_recoverable_stop_sigusr1_grace_sec = float(
                    getattr(self.lhr, "resource_recoverable_stop_sigusr1_grace_sec", 60.0) or 0.0,
                )
                bash_tool.resource_recoverable_stop_marker_exit_grace_sec = float(
                    getattr(self.lhr, "resource_recoverable_stop_marker_exit_grace_sec", 10.0) or 0.0,
                )
                bash_tool.resource_recoverable_stop_sigterm_grace_sec = float(
                    getattr(self.lhr, "resource_recoverable_stop_sigterm_grace_sec", 5.0) or 0.0,
                )
                finalization_reserve = max(0.0, float(
                    getattr(self.lhr, "resource_bash_hard_fuse_finalization_reserve_sec", 900.0) or 0.0,
                ))
                remaining_fuse = max(1.0, float(self.deadline - time.monotonic()))
                bash_tool.bash_hard_fuse_deadline_monotonic = float(self.deadline)
                bash_tool.bash_hard_fuse_finalization_reserve_sec = finalization_reserve
                bash_tool.bash_timeout_sec = _effective_lnr_bash_timeout_sec(
                    float(getattr(bash_tool, "bash_timeout_sec", 1.0) or 1.0),
                    remaining_fuse,
                )
                bash_tool.bash_timeout_slow_sec = _effective_lnr_bash_timeout_sec(
                    float(getattr(bash_tool, "bash_timeout_slow_sec", 1.0) or 1.0),
                    remaining_fuse,
                )
                setattr(agent, "_bash_timeout_sec", float(bash_tool.bash_timeout_sec))
                setattr(agent, "_bash_timeout_slow_sec", float(bash_tool.bash_timeout_slow_sec))
            except Exception:
                logger.debug("[lnr] could not configure bash guards", exc_info=True)
        try:
            core = str(getattr(agent, "_system_prompt_core", "") or "")
            if core:
                setattr(agent, "_system_prompt_core", _append_lnr_main_agent_protocols(core))
            else:
                base = str(getattr(agent, "systemPrompt", "") or "")
                setattr(agent, "systemPrompt", _append_lnr_main_agent_protocols(base))
        except Exception:
            logger.debug("[lnr] could not attach LNR main-agent protocols", exc_info=True)
        self._sanitize_agent_prompt_surfaces(agent)
        if bool(getattr(self.lhr, "clean_repl_mode", True)):
            # LHR is a continuous REPL loop. Keep legacy agent coaching out of
            # the main transcript so text-only replies remain a natural stop/estra signal.
            setattr(agent, "_lnr_phase_header", "")
            setattr(agent, "_guard_manager", None)
            setattr(agent, "_lhr_clean_repl_mode", True)
            setattr(agent, "_lnr_fresh_hint_injected", True)
            setattr(agent, "_lnr_stage_commit_enabled", False)
            setattr(agent, "_lnr_force_stage_journal_after_run", False)
            setattr(agent, "_lnr_stage_journal_pending", False)
        setattr(agent, "_submission_history_archive_enabled", False)
        store = getattr(agent, "_tool_output_artifacts", None)
        setter = getattr(store, "set_mirror_raw_id_prefix", None)
        if callable(setter):
            setter(self._next_stage_id_for_logging())
        setattr(agent, "_lnr_allow_any_stage_script", True)
        setattr(agent, "_lnr_candidate_archive_callback", self._archive_candidate_artifact_after_tool)
        setattr(agent, "_lnr_stage_capture_callback", self._stage_capture_callback)
        setattr(
            agent,
            "_lnr_metric_output_interpretation_callback",
            self._metric_output_interpretation_callback,
        )
        setattr(
            agent,
            "_lnr_stage_capture_on_candidate_artifact",
            self._evaluator_stage_source_mode() == "primary",
        )
        task_profile = self._evaluator_task_profile()
        evaluator_backend = self._evaluator_backend_name()
        candidate_artifact = self._evaluator_candidate_artifact()
        setattr(agent, "_scienceflow_task_profile", task_profile)
        setattr(agent, "_scienceflow_evaluator_backend", evaluator_backend)
        setattr(agent, "_scienceflow_candidate_artifact_rel", candidate_artifact)
        setattr(agent, "_lnr_candidate_artifact_rel", candidate_artifact)
        setattr(agent, "_scienceflow_evaluation_service", self.evaluation_service)
        setattr(agent, "_scienceflow_evaluation_cfg", self.cfg)
        setattr(agent, "_scienceflow_task_root", self.task_root_dir)
        setattr(agent, "_scienceflow_worker_id", self._worker_uid_prefix())
        setattr(agent, "_lnr_text_only_callback", self._text_only_estra_callback)
        setattr(agent, "_context_limit_estra_callback", self._context_limit_estra_callback)
        setattr(agent, "_context_compact_event_callback", self._record_context_compact_event)
        self._restore_protected_eda_prefix_marker(agent)
        setattr(agent, "_lnr_compact_on_context_threshold", bool(self.lhr.compact_on_context_limit))
        setattr(agent, "_mlebench_data_dir", str(getattr(self.cfg, "mlebench_data_root_dir", "") or "") or None)
        setattr(agent, "_mlebench_exp_id", str(getattr(self.cfg, "exp_id", "") or "") or None)
        setattr(
            agent,
            "_lnr_mlebench_validate_enabled",
            legacy_score_contract_enabled(
                task_profile=task_profile,
                evaluator_backend=evaluator_backend,
                candidate_artifact=candidate_artifact,
            ),
        )
        return agent

    def _worker_llm_stage_override(self, stage_name: str = "code") -> Any | None:
        """Spread LHR workers across one LLM stage's endpoint pool.

        Regular REPL uses the configured routing unchanged. In LHR multi-worker
        mode, sticky routing can pin each worker to a different primary key.
        When ``agent.<stage>.models`` is configured, it is treated as an indexed
        worker endpoint pool: worker i uses models[i], api_keys[i], base_urls[i]
        with single-value key/url lists broadcast and shorter lists cycled.
        """
        if int(getattr(self, "worker_count", 1) or 1) <= 1:
            return None
        stage_name = str(stage_name or "code").strip() or "code"
        stage = getattr(getattr(self.cfg, "agent", None), stage_name, None)
        if stage is None:
            return None

        def _clean_list(value: Any) -> list[str]:
            return [str(x).strip() for x in (value or []) if str(x).strip()]

        models = _clean_list(getattr(stage, "models", []))
        api_keys = _clean_list(getattr(stage, "api_keys", []))
        base_urls = [x.rstrip("/") for x in _clean_list(getattr(stage, "base_urls", []))]
        worker_index = int(getattr(self, "worker_index", 0) or 0)
        worker_tag = getattr(self, "worker_id", "") or self._worker_uid_prefix()
        stage_copy = copy.deepcopy(stage)

        if models:
            scalar_key = str(getattr(stage, "api_key", "") or "").strip()
            if not api_keys and scalar_key:
                api_keys = [scalar_key]
            scalar_url = str(getattr(stage, "base_url", "") or "").strip().rstrip("/")
            if not base_urls and scalar_url:
                base_urls = [scalar_url]

            periods = [len(models)]
            if api_keys:
                periods.append(len(api_keys))
            if base_urls and len(base_urls) > 1:
                periods.append(len(base_urls))
            pool_size = max(1, math.lcm(*periods))
            aligned_models = [models[i % len(models)] for i in range(pool_size)]
            aligned_keys = [api_keys[i % len(api_keys)] for i in range(pool_size)] if api_keys else []
            if base_urls:
                aligned_urls = [base_urls[0] if len(base_urls) == 1 else base_urls[i % len(base_urls)] for i in range(pool_size)]
            else:
                aligned_urls = []

            primary_index = worker_index % pool_size

            def _rotate(values: list[str]) -> list[str]:
                return values[primary_index:] + values[:primary_index]

            rotated_models = _rotate(aligned_models)
            rotated_keys = _rotate(aligned_keys) if aligned_keys else []
            rotated_urls = _rotate(aligned_urls) if aligned_urls else []
            stage_copy.model = rotated_models[0]
            stage_copy.models = rotated_models
            selected_key_index: int | None = None
            selected_url_index: int | None = None
            if rotated_keys:
                selected_key_index = primary_index % len(api_keys)
                stage_copy.api_key = rotated_keys[0]
                stage_copy.api_keys = rotated_keys
            if rotated_urls:
                selected_url_index = 0 if len(base_urls) == 1 else primary_index % len(base_urls)
                stage_copy.base_url = rotated_urls[0]
                stage_copy.base_urls = rotated_urls
            if rotated_keys:
                routing_mode = str(getattr(stage, "api_routing_mode", "") or "").strip().lower()
                if routing_mode not in {"sticky", "task_sticky", "sticky_failover", "task_sticky_failover"}:
                    stage_copy.api_routing_mode = "sticky_failover"
                base_sticky_id = str(
                    getattr(stage_copy, "api_sticky_id", "") or getattr(self.cfg, "exp_id", "") or "lhr"
                )
                stage_copy.api_sticky_id = f"{base_sticky_id}:{worker_tag}"
                stage_copy.api_sticky_primary_index = 0
            else:
                stage_copy.api_sticky_id = ""
                stage_copy.api_sticky_primary_index = None
            self._jsonl(
                "lhr_llm_routing_events.jsonl",
                {
                    "event": "worker_model_endpoint_assigned",
                    "stage_role": stage_name,
                    "worker_id": worker_tag,
                    "worker_index": self.worker_index,
                    "worker_count": self.worker_count,
                    "model_index": primary_index % len(models),
                    "model_pool_size": len(models),
                    "api_key_index": selected_key_index,
                    "api_key_pool_size": len(api_keys),
                    "base_url_index": selected_url_index,
                    "base_url_pool_size": len(base_urls),
                    "failover_pool_size": pool_size if rotated_keys else 0,
                    "sticky_primary_index": stage_copy.api_sticky_primary_index,
                    "pinned_primary_endpoint": bool(rotated_keys or rotated_urls),
                },
            )
            return stage_copy

        if len(api_keys) <= 1:
            return None
        routing_mode = str(getattr(stage, "api_routing_mode", "") or "").strip().lower()
        if routing_mode not in {"sticky", "task_sticky", "sticky_failover", "task_sticky_failover"}:
            return None
        primary_index = worker_index % len(api_keys)
        base_sticky_id = str(
            getattr(stage_copy, "api_sticky_id", "") or getattr(self.cfg, "exp_id", "") or "lhr"
        )
        stage_copy.api_sticky_id = f"{base_sticky_id}:{worker_tag}"
        stage_copy.api_sticky_primary_index = primary_index
        self._jsonl(
            "lhr_llm_routing_events.jsonl",
            {
                "event": "worker_sticky_primary_assigned",
                "stage_role": stage_name,
                "worker_id": worker_tag,
                "worker_index": self.worker_index,
                "worker_count": self.worker_count,
                "routing_mode": routing_mode,
                "pool_size": len(api_keys),
                "sticky_primary_index": primary_index,
            },
        )
        return stage_copy

    async def _restore_pending_estra(self) -> None:
        if not self.pending_estra:
            return
        target = str(self.pending_estra.get("target_stage") or "").upper()
        action = str(self.pending_estra.get("action") or "switch_stage")
        requested_node_uid = str(self.pending_estra.get("target_node_uid") or "").strip()
        snap = self._stage_snapshot_for_restore(stage_id=target, node_uid=requested_node_uid)
        if snap is None:
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "estra_failed",
                    "target_stage": target,
                    "target_node_uid": requested_node_uid,
                    "reason": "missing_node_snapshot" if requested_node_uid else "missing_snapshot",
                },
            )
            self.pending_estra = None
            return
        self._close_lnr_interaction_loggers()
        summary = str(self.pending_estra.get("tail_summary") or "")
        state_packet = str(self.pending_estra.get("state_packet") or "")
        target_node_uid = requested_node_uid or self._snapshot_node_uid(snap)
        strict_context_limit = str(self.pending_estra.get("compact_strength") or "") == "strict_context_limit"
        previous_lineage = self._lineage_uid_prefix()
        if self._is_keep_like_estra_action(action):
            self._prune_workspace_control_artifacts()
            estra_fields = self.pending_estra.get("estra_fields") if isinstance(self.pending_estra.get("estra_fields"), dict) else {}
            memory_compacted, memory_records = self._rebuild_memory_after_keep_current(
                terminal_stage=target,
                summary=summary,
                state_packet=state_packet,
                strict_context_limit=strict_context_limit,
                continuation_action=action,
                estra_fields=estra_fields,
            )
            self._append_traj_summary(target_stage=target, summary=summary)
            self._reset_interaction_stage_files()
            new_lineage = self._start_new_lineage()
            self.current_restored_from_stage = target
            self.current_restored_from_node_uid = target_node_uid
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "estra_keep_current_compacted",
                    "action": action,
                    "target_stage": target,
                    "target_node_uid": target_node_uid,
                    "previous_lineage_id": previous_lineage,
                    "new_lineage_id": new_lineage,
                    "snapshot_id": snap.snapshot_id,
                    "snapshot_path": str(snap.snapshot_path),
                    "tail_summary_chars": len(summary),
                    "state_packet_chars": len(state_packet),
                    "memory_compacted": memory_compacted,
                    "memory_records": memory_records,
                    "compact_strength": "strict_context_limit" if strict_context_limit else "normal",
                    "context_generation": str(self.pending_estra.get("context_generation") or ""),
                    "restore_key": str(self.pending_estra.get("restore_key") or ""),
                    "preserved_logs": self.log_dir.is_dir(),
                    "preserved_memory": (self.memory_dir / "ScienceAgent").is_dir(),
                },
            )
            self.pending_estra = None
            return

        previous_stage_snapshots = dict(self.stage_snapshots)
        previous_restored_stage = str(getattr(self, "current_restored_from_stage", "") or "")
        previous_restored_node_uid = str(getattr(self, "current_restored_from_node_uid", "") or "")
        archive: Path | None = None
        try:
            archive = self.snapshot_store.restore(snap)
            # The visible Stage ID may also exist in the lineage being abandoned.
            # Keep the active view aligned with the exact archived node restored
            # above; the full node-keyed archive remains available independently.
            self.stage_snapshots[target] = snap
            self._prepare_dataset_symlink()
            self._prune_workspace_control_artifacts()
            append_archived_trajectory_summary(
                self.workspace_dir / self.ledger_filename,
                target_stage=target,
                summary=summary,
            )
            self._prune_active_stage_snapshots_to_ledger()
            self._write_stage_map()
            memory_compacted, memory_records = self._rebuild_memory_after_estra(
                target_stage=target,
                summary=summary,
                state_packet=state_packet,
                strict_context_limit=strict_context_limit,
            )
            self._append_traj_summary(target_stage=target, summary=summary)
            self._reset_interaction_stage_files()
        except Exception as exc:
            rollback_attempted = archive is not None
            rollback_succeeded = archive is None
            rollback_error = ""
            if archive is not None:
                try:
                    self.snapshot_store.restore_terminal_archive(archive)
                    rollback_succeeded = True
                except Exception as rollback_exc:
                    rollback_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
                    logger.exception("[lnr] estra terminal workspace rollback failed")
            self.stage_snapshots = previous_stage_snapshots
            self.current_restored_from_stage = previous_restored_stage
            self.current_restored_from_node_uid = previous_restored_node_uid
            try:
                self._write_stage_map()
            except Exception:
                logger.debug("[lnr] stage map rollback write failed", exc_info=True)
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "estra_failed",
                    "action": action,
                    "target_stage": target,
                    "target_node_uid": target_node_uid,
                    "reason": "unfold_transaction_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "rollback_attempted": rollback_attempted,
                    "rollback_succeeded": rollback_succeeded,
                    "rollback_error": rollback_error,
                    "lineage_id": previous_lineage,
                },
            )
            self.pending_estra = None
            return

        new_lineage = self._start_new_lineage()
        self.current_restored_from_stage = target
        self.current_restored_from_node_uid = target_node_uid
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_stage_switched",
                "target_stage": target,
                "target_node_uid": target_node_uid,
                "previous_lineage_id": previous_lineage,
                "new_lineage_id": new_lineage,
                "snapshot_id": snap.snapshot_id,
                "snapshot_path": str(snap.snapshot_path),
                "terminal_archive": str(archive),
                "tail_summary_chars": len(summary),
                "state_packet_chars": len(state_packet),
                "memory_compacted": memory_compacted,
                "memory_records": memory_records,
                "compact_strength": "strict_context_limit" if strict_context_limit else "normal",
                "context_generation": str(self.pending_estra.get("context_generation") or ""),
                "restore_key": str(self.pending_estra.get("restore_key") or ""),
                "preserved_logs": self.log_dir.is_dir(),
                "preserved_memory": (self.memory_dir / "ScienceAgent").is_dir(),
            },
        )
        self.pending_estra = None

    def _accumulate_main_run_tokens(self, agent: Any) -> None:
        self.main_tokens_in += int(getattr(agent, "last_run_tokens_in", 0) or 0)
        self.main_tokens_out += int(getattr(agent, "last_run_tokens_out", 0) or 0)
        self.main_tokens_cached += int(getattr(agent, "last_run_tokens_cached", 0) or 0)
        self.main_llm_calls += int(getattr(agent, "last_run_llm_calls", 0) or 0)

    def _result(self, *, stop_reason: str) -> dict[str, Any]:
        main_cache_rate = self.main_tokens_cached / self.main_tokens_in if self.main_tokens_in else 0.0
        estra_counts = self._estra_decision_counts()
        best_stage = None
        best_metric = None
        lower = True
        for sid, snap in self.stage_snapshots.items():
            if snap.metric_value is None:
                continue
            if best_metric is None:
                best_metric = snap.metric_value
                best_stage = sid
                lower = bool(snap.lower_is_better is not False)
                continue
            if lower and snap.metric_value < best_metric:
                best_metric = snap.metric_value
                best_stage = sid
            elif not lower and snap.metric_value > best_metric:
                best_metric = snap.metric_value
                best_stage = sid
        return {
            "solver": self.solver_name,
            "status": "success",
            "stop_reason": stop_reason,
            "stage_count": len(self.stage_snapshots),
            "estra_switch_stage_count": self._count_jsonl_events("lhr_estras.jsonl", "estra_stage_switched"),
            "estra_compact_count": self._count_jsonl_events("lhr_estras.jsonl", "estra_keep_current_compacted"),
            "estra_decisions": self.estra_decisions,
            "estra_continue_count": estra_counts["continue"],
            "estra_redirect_count": estra_counts["redirect"],
            "estra_switch_count": estra_counts["switch"],
            "estra_current_continue_count": estra_counts["current_continue"],
            "estra_current_redirect_count": estra_counts["current_redirect"],
            "estra_stage_continue_count": estra_counts["stage_continue"],
            "estra_stage_redirect_count": estra_counts["stage_redirect"],
            "best_stage": best_stage,
            "best_metric": best_metric,
            "score_summary": self._score_summary_for_prompt(),
            "main_cache_rate": main_cache_rate,
            "main_tokens_input": self.main_tokens_in,
            "main_tokens_cached": self.main_tokens_cached,
            "main_tokens_output": self.main_tokens_out,
            "main_llm_calls": self.main_llm_calls,
            "stage_tokens_input": self.stage_tokens_in,
            "stage_tokens_cached": self.stage_tokens_cached,
            "stage_llm_calls": self.stage_llm_calls,
            "estra_tokens_input": self.estra_tokens_in,
            "estra_tokens_cached": self.estra_tokens_cached,
            "estra_llm_calls": self.estra_llm_calls,
            "workspace_dir": str(self.workspace_dir),
            "ledger": str(self.ledger_path),
        }

    def _count_jsonl_events(self, name: str, event: str) -> int:
        paths = [self.log_dir / name]
        if name in LHR_UNIFIED_ONLY_JSONL or name == LHR_EVENTS_JSONL:
            paths = [self.log_dir / LHR_EVENTS_JSONL]
        n = 0
        for path in paths:
            if not path.is_file():
                continue
            try:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("event") == event:
                        n += 1
            except (OSError, json.JSONDecodeError):
                return n
        return n

    def _estra_decision_counts(self) -> dict[str, int]:
        counts = {
            "decisions": 0,
            "continue": 0,
            "redirect": 0,
            "switch": 0,
            "current_continue": 0,
            "current_redirect": 0,
            "stage_continue": 0,
            "stage_redirect": 0,
        }
        path = self.log_dir / LHR_EVENTS_JSONL
        if not path.is_file():
            return counts
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return counts
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("event") != "estra_decision":
                continue
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
            action = str(payload.get("action") or obj.get("action") or "")
            startpoint = str(
                payload.get("startpoint")
                or obj.get("startpoint")
                or ("previous_stage" if action == "switch_stage" else "current_workspace")
            )
            intent = str(
                payload.get("intent")
                or obj.get("intent")
                or ("redirect" if action == "keep_but_redirect" else "continue")
            )
            kind = self._estra_decision_kind(action=action, startpoint=startpoint, intent=intent)
            counts["decisions"] += 1
            if kind in {"continue", "redirect", "switch"}:
                counts[kind] += 1
            if action in {"keep_current", "keep_but_redirect", "switch_stage"}:
                if startpoint == "previous_stage" and intent == "redirect":
                    counts["stage_redirect"] += 1
                elif startpoint == "previous_stage":
                    counts["stage_continue"] += 1
                elif intent == "redirect":
                    counts["current_redirect"] += 1
                else:
                    counts["current_continue"] += 1
        return counts

    @staticmethod
    def _format_cpu_ids(cpu_ids: list[int]) -> str:
        if not cpu_ids:
            return ""
        ids = sorted(set(int(x) for x in cpu_ids))
        ranges: list[str] = []
        start = prev = ids[0]
        for cur in ids[1:]:
            if cur == prev + 1:
                prev = cur
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = cur
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ",".join(ranges)

    @staticmethod
    def _slice_cpu_ids(cpu_ids: list[int], *, worker_index: int, worker_count: int) -> list[int]:
        if not cpu_ids:
            return []
        n = max(1, int(worker_count or 1))
        idx = max(0, min(n - 1, int(worker_index or 0)))
        base = len(cpu_ids) // n
        rem = len(cpu_ids) % n
        start = idx * base + min(idx, rem)
        size = base + (1 if idx < rem else 0)
        return cpu_ids[start:start + size]

    def _worker_extra_env(self, *, worker_index: int, worker_count: int) -> dict[str, str]:
        raw = os.environ.get("SCIENCEFLOW_TASK_CPU_LIST", "").strip()
        if not raw:
            raw = os.environ.get("SCIENCEFLOW_CPU_LIST", "").strip()
        if not raw:
            raw = str(getattr(getattr(self.cfg, "exec", None), "cpu_list", "") or "").strip()
        cpu_ids = parse_cpu_list(raw) if raw else []
        worker_cpu_ids = self._slice_cpu_ids(
            cpu_ids,
            worker_index=worker_index,
            worker_count=worker_count,
        )
        env: dict[str, str] = {}
        task_cpu_set = self._format_cpu_ids(cpu_ids)
        if task_cpu_set:
            env["SCIENCEFLOW_TASK_CPU_LIST"] = task_cpu_set
        cpu_set = self._format_cpu_ids(worker_cpu_ids)
        if cpu_set:
            env["_SCIENCEFLOW_CPU_SET"] = cpu_set
            env["SCIENCEFLOW_WORKER_CPU_LIST"] = cpu_set
            env["SCIENCEFLOW_CPU_LIST"] = cpu_set
            thread_cap = int(getattr(self.lhr, "omp_threads_cap", 8) or 0)
            threads = max(1, len(worker_cpu_ids))
            if thread_cap > 0:
                threads = min(thread_cap, threads)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                env[key] = str(threads)
        try:
            base_seed = int(getattr(self.lhr, "seed", 0) or 0)
        except (TypeError, ValueError):
            base_seed = 0
        if base_seed > 0:
            worker_seed = base_seed + max(0, int(worker_index or 0))
            env["SCIENCEFLOW_RANDOM_SEED"] = str(worker_seed)
            env["SEED"] = str(worker_seed)
            env["PYTHONHASHSEED"] = str(worker_seed)
        return env

    def _worker_root(self, worker_index: int) -> Path:
        return lnr_worker_layout(self.root_dir, self.lhr, worker_index).root

    def _worker_workspace_dir(self, worker_index: int) -> Path:
        return lnr_worker_layout(self.root_dir, self.lhr, worker_index).workspace

    def _worker_log_dir(self, worker_index: int) -> Path:
        return lnr_worker_layout(self.root_dir, self.lhr, worker_index).logs

    def _merge_dir(self) -> Path:
        dirname = str(getattr(self.lhr, "merge_dirname", "merge") or "merge").strip() or "merge"
        return self.root_dir / dirname

    def _global_merge_reserve_sec(self) -> float:
        if not bool(getattr(self.lhr, "merge_enabled", True)):
            return 0.0
        budget = float(getattr(self.lhr, "wall_clock_budget_sec", 0) or 0)
        configured = float(getattr(self.lhr, "global_merge_wall_clock_sec", 0) or 0)
        if budget <= 120.0 or configured <= 0.0:
            return 0.0
        reserve = min(configured, max(300.0, budget * 0.15))
        return min(reserve, max(0.0, budget - 60.0))

    def _worker_wall_clock_budget_sec(self) -> int:
        budget = int(float(getattr(self.lhr, "wall_clock_budget_sec", 0) or 0))
        reserve = int(self._global_merge_reserve_sec())
        if budget <= 0 or reserve <= 0:
            return budget
        return max(60, budget - reserve)

    def _refresh_submission_links(self, *, n_workers: int, include_merge: bool) -> None:
        raw_submission_dir = getattr(self.cfg, "submission_dir", None)
        if raw_submission_dir is None or not str(raw_submission_dir).strip():
            return
        try:
            links = refresh_submission_links(
                submission_dir=Path(raw_submission_dir),
                artifact_path=self._evaluator_candidate_artifact(),
                merge_dir=self._merge_dir() if include_merge else None,
                worker_roots=[self._worker_root(i) for i in range(max(1, int(n_workers or 1)))],
            )
            self._jsonl(
                "lhr_coordinator_events.jsonl",
                {
                    "event": "submission_links_refreshed",
                    "link_count": len(links),
                    "include_merge": include_merge,
                    "submission_dir": str(raw_submission_dir),
                },
            )
        except Exception as exc:  # noqa: BLE001 - submission links are convenience outputs.
            logger.warning("failed to refresh LNR submission links: %s", exc)

    def _cleanup_coordinator_workspace_shell(self) -> None:
        if self.worker_id or max(1, int(self.lhr.num_workers or 1)) <= 1:
            return
        ws = self.workspace_dir
        if not ws.exists() or not ws.is_dir():
            return
        # Older layouts created a minimal coordinator workspace. In worker-indexed
        # mode actual agents live under workers/wXX, so remove the empty shell
        # only when it is clearly not the task root.
        if ws.resolve(strict=False) == self.root_dir.resolve(strict=False):
            return
        allowed = {"logs", "submissions"}
        try:
            names = {p.name for p in ws.iterdir()}
        except OSError:
            return
        if not names.issubset(allowed):
            return
        for child in sorted(ws.iterdir(), key=lambda p: len(p.parts), reverse=True):
            if child.is_dir():
                try:
                    shutil.rmtree(child)
                except OSError:
                    logger.debug("[lnr] coordinator workspace cleanup skipped", exc_info=True)
                    return
        try:
            ws.rmdir()
        except OSError:
            logger.debug("[lnr] coordinator workspace shell not empty", exc_info=True)

    def _cleanup_worker_root_artifacts(self, worker_root: Path) -> None:
        try:
            (worker_root / "resolved_config.yaml").unlink(missing_ok=True)
        except OSError:
            logger.debug("[lnr] worker resolved_config cleanup skipped", exc_info=True)

    def _aggregate_worker_state(self, *, n_workers: int, run_status: str) -> dict[str, Any]:
        log_dirs = [self.global_log_dir]
        for idx in range(max(1, int(n_workers or 1))):
            log_dirs.append(self._worker_log_dir(idx))
        try:
            return LHRStateMachineStore.aggregate_logs(
                output_log_dir=self.global_log_dir,
                input_log_dirs=log_dirs,
                worker_count=n_workers,
                run_status=run_status,
                ledger_filename=self.ledger_filename,
            )
        except OSError:
            logger.debug("[lnr] worker state aggregation failed", exc_info=True)
            return {}

    async def _aggregate_worker_state_periodically(
        self,
        *,
        n_workers: int,
        run_status: str = "running",
        interval_sec: float = 60.0,
    ) -> None:
        """Refresh coordinator LHR state while worker-indexed runs are active."""

        interval = max(5.0, float(interval_sec or 60.0))
        while True:
            try:
                self._aggregate_worker_state(n_workers=n_workers, run_status=run_status)
            except Exception:  # noqa: BLE001 - monitor state refresh must not affect workers.
                logger.debug("[lnr] live worker state aggregation failed", exc_info=True)
            await asyncio.sleep(interval)

    def _write_global_time_trace(self, worker_results: list[dict[str, Any]]) -> None:
        from scienceflow.utils.time_trace import TRACE_COLUMNS, TRACE_FILENAME

        out = self.global_log_dir / TRACE_FILENAME
        rows: list[dict[str, str]] = []
        for result in worker_results:
            worker_id = str(result.get("worker_id") or "")
            worker_index = str(result.get("worker_index") or "")
            try:
                worker_index_int = int(result.get("worker_index") or 0)
            except (TypeError, ValueError):
                continue
            src = self._worker_log_dir(worker_index_int) / TRACE_FILENAME
            if not src.is_file():
                continue
            try:
                with src.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    for row in csv.DictReader(f):
                        merged = {"worker_id": worker_id, "worker_index": worker_index}
                        for col in TRACE_COLUMNS:
                            merged[col] = str(row.get(col) or "")
                        rows.append(merged)
            except OSError:
                logger.debug("[lnr] worker time trace read failed: %s", src, exc_info=True)
        if not rows:
            return
        try:
            self.global_log_dir.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["worker_id", "worker_index", *TRACE_COLUMNS])
                writer.writeheader()
                writer.writerows(rows)
            for result in worker_results:
                try:
                    (self._worker_log_dir(int(result.get("worker_index") or 0)) / TRACE_FILENAME).unlink(missing_ok=True)
                except (OSError, TypeError, ValueError):
                    pass
        except OSError:
            logger.debug("[lnr] global time trace write failed", exc_info=True)

    @staticmethod
    def _worker_error_kind(error: str) -> str:
        text = str(error or "")
        lower = text.lower()
        if not text.strip():
            return ""
        if "llm_quota_error" in lower or "insufficient balance" in lower or "error code: 402" in lower:
            return "llm_quota_error"
        if (
            "context_compact_failed" in lower
            or "compact did not fit context" in lower
            or "omitted-history main-agent request" in lower
        ):
            return "context_compact_failed"
        if "llm_api_error" in lower or "apistatuserror" in lower:
            return "llm_api_error"
        if (
            "separator is found, but chunk is longer than limit" in lower
            or "limitoverrunerror" in lower
            or "search output line exceeded" in lower
        ):
            return "tool_output_limit"
        if (
            "readerror" in lower
            or "apiconnectionerror" in lower
            or "remoteprotocolerror" in lower
            or "connection reset" in lower
            or "server disconnected" in lower
        ):
            return "llm_transport_error"
        if "timeout" in lower:
            return "worker_timeout"
        return "worker_error"

    @classmethod
    def _multi_worker_failure_kind(cls, worker_results: list[dict[str, Any]]) -> tuple[str, list[str]]:
        if not worker_results:
            return "no_workers", []

        kinds = sorted(
            {
                kind
                for result in worker_results
                for kind in [cls._worker_error_kind(str(result.get("error") or ""))]
                if kind
            }
        )
        if not kinds:
            statuses = {str(result.get("status") or "").strip() for result in worker_results}
            if statuses and statuses <= {"failed"}:
                return "all_workers_failed", []
            return "no_viable_worker_result", []
        if len(kinds) == 1:
            return kinds[0], kinds
        return "worker_failed_mixed", kinds

    @classmethod
    def _multi_worker_stop_reason(
        cls, *, run_succeeded: bool, worker_results: list[dict[str, Any]]
    ) -> tuple[str, list[str]]:
        if run_succeeded:
            return "budget_expired", []
        return "", cls._multi_worker_failure_kind(worker_results)[1]

    def _make_worker_cfg(self, worker_index: int, worker_count: int) -> Config:
        cfg = copy.deepcopy(self.cfg)
        layout = ensure_lnr_worker_layout(self.root_dir, self.lhr, worker_index)
        cfg.task_workspace_root_dir = layout.root
        setattr(cfg, "_workspace_dir_override", layout.workspace)
        setattr(cfg, "_log_dir_override", layout.logs)
        cfg.lnr.num_workers = 1
        cfg.lnr.merge_enabled = False
        cfg.lnr.wall_clock_budget_sec = self._worker_wall_clock_budget_sec()
        return cfg

    async def _run_one_worker(
        self, worker_index: int, worker_count: int
    ) -> dict[str, Any]:
        worker_id = f"W{worker_index:02d}"
        from scienceflow.core.orchestrator import Orchestrator

        worker_cfg = self._make_worker_cfg(worker_index, worker_count)
        worker_env = self._worker_extra_env(
            worker_index=worker_index,
            worker_count=worker_count,
        )
        self._jsonl(
            "lhr_coordinator_events.jsonl",
            {
                "event": "worker_start",
                "worker_id": worker_id,
                "worker_index": worker_index,
                "worker_root": str(
                    Path(worker_cfg.task_workspace_root_dir).relative_to(self.root_dir)
                ),
                "cpu_set": worker_env.get("_SCIENCEFLOW_CPU_SET", ""),
                "omp_threads": worker_env.get("OMP_NUM_THREADS", ""),
                "worker_wall_clock_budget_sec": int(
                    worker_cfg.lnr.wall_clock_budget_sec or 0
                ),
                "global_merge_reserve_sec": int(self._global_merge_reserve_sec()),
            },
        )
        worker_orchestrator = Orchestrator(worker_cfg)
        worker_solver = LnrSolver(
            task_desc=self.task_desc,
            cfg=worker_orchestrator.cfg,
            orchestrator=worker_orchestrator,
            worker_id=worker_id,
            worker_index=worker_index,
            worker_count=worker_count,
            worker_extra_env=worker_env,
            task_root_dir=self.root_dir,
        )
        merge_owner = str(getattr(self.lhr, "merge_owner_worker", "W00") or "W00")
        keep_agent_open = (
            bool(getattr(self.lhr, "merge_enabled", True))
            and str(getattr(self.lhr, "merge_mode", "worker_reduce") or "worker_reduce")
            == "worker_reduce"
            and worker_id == merge_owner
        )
        try:
            result = await worker_solver._run_single(keep_agent_open=keep_agent_open)
            if keep_agent_open and worker_solver._live_agent is not None:
                self._merge_owner_solver = worker_solver
            return {"worker_id": worker_id, "worker_index": worker_index, **result}
        except BaseException as exc:
            logger.exception("[lnr] worker %s failed", worker_id)
            return {
                "worker_id": worker_id,
                "worker_index": worker_index,
                "solver": self.solver_name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "workspace_dir": str(
                    getattr(
                        worker_cfg,
                        "_workspace_dir_override",
                        worker_cfg.task_workspace_root_dir,
                    )
                ),
            }
        finally:
            self._cleanup_worker_root_artifacts(
                Path(worker_cfg.task_workspace_root_dir)
            )

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    def _candidate_from_stage(
        self,
        *,
        worker_id: str,
        worker_root: Path,
        stage_id: str,
        stage: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot_path = Path(str(stage.get("snapshot_path") or ""))
        meta = self._read_json_file(snapshot_path / "logs" / "lhr_snapshot_meta.json") if snapshot_path else {}
        source = meta.get("source_event") if isinstance(meta.get("source_event"), dict) else {}
        candidate_id = f"{worker_id}:{stage_id}"
        return {
            "candidate_id": candidate_id,
            "worker_id": worker_id,
            "stage_id": stage_id,
            "worker_root": str(worker_root),
            "workspace_path": str(worker_root / "workspace"),
            "snapshot_id": str(stage.get("snapshot_id") or ""),
            "snapshot_path": str(snapshot_path) if str(snapshot_path) else "",
            "metric_name": str(stage.get("metric_name") or source.get("metric_name") or "Final Validation Score"),
            "metric_value": stage.get("metric_value"),
            "lower_is_better": stage.get("lower_is_better"),
            "memory_cut": stage.get("memory_cut"),
            "validation_ok": source.get("validation_ok"),
            "validation_issue": source.get("validation_issue") or stage.get("validation_issue") or "",
            "reported_val_score": source.get("reported_val_score") or stage.get("reported_val_score"),
            "val_score_type": source.get("val_score_type") or stage.get("val_score_type") or "",
            "selection_eligible": (
                source.get("selection_eligible")
                if "selection_eligible" in source
                else stage.get("selection_eligible")
            ),
            "selection_score": source.get("selection_score") or stage.get("selection_score") or "",
            "selection_note": source.get("selection_note") or stage.get("selection_note") or "",
            "metric_source_note": source.get("metric_source_note") or stage.get("metric_source_note") or "",
            "metric_protocol": source.get("metric_protocol") or stage.get("metric_protocol") or "",
            "train_data_used": source.get("train_data_used") or stage.get("train_data_used") or "",
            "metric_eval_data": source.get("metric_eval_data") or stage.get("metric_eval_data") or "",
            "execution_mode": source.get("execution_mode") or stage.get("execution_mode") or "",
            "metric_validity": source.get("metric_validity") or stage.get("metric_validity") or "",
            "metric_validity_note": source.get("metric_validity_note") or stage.get("metric_validity_note") or "",
            "metric_validity_reason_code": source.get("metric_validity_reason_code") or stage.get("metric_validity_reason_code") or "",
            "lineage_id": str(stage.get("lineage_id") or source.get("lineage_id") or ""),
            "node_uid": str(stage.get("node_uid") or source.get("node_uid") or ""),
            "route_id": str(source.get("route_id") or stage.get("route_id") or ""),
            "solution_sha": str(source.get("solution_sha") or stage.get("solution_sha") or ""),
            "submission_sha": str(source.get("submission_sha") or stage.get("submission_sha") or ""),
            "submission_snapshot": str(source.get("submission_snapshot") or stage.get("submission_snapshot") or ""),
            "artifact_path": str(source.get("artifact_path") or stage.get("artifact_path") or ""),
            "artifact_sha": str(source.get("artifact_sha") or stage.get("artifact_sha") or ""),
            "candidate_ready": (
                source.get("candidate_ready")
                if "candidate_ready" in source
                else stage.get("candidate_ready")
                if "candidate_ready" in stage
                else bool(source.get("submission_sha") or stage.get("submission_sha"))
            ),
            "submission_status": str(source.get("submission_status") or ""),
            "duplicate_submission_of_stage": str(source.get("duplicate_submission_of_stage") or ""),
        }

    def _load_worker_candidates(self, worker_result: dict[str, Any]) -> list[dict[str, Any]]:
        worker_id = str(worker_result.get("worker_id") or "")
        worker_index = int(worker_result.get("worker_index") or 0)
        worker_root = self._worker_root(worker_index)
        stage_map = self._read_json_file(self._worker_log_dir(worker_index) / "lhr_stage_map.json")
        stages = stage_map.get("stages") if isinstance(stage_map.get("stages"), dict) else {}
        peer_evidence = load_peer_candidate_evidence(
            self.log_dir / "lhr_stage_performance.csv",
            worker_id=worker_id,
        )
        candidates: list[dict[str, Any]] = []
        for stage_id, stage in sorted(stages.items()):
            if isinstance(stage, dict):
                candidate = self._candidate_from_stage(
                    worker_id=worker_id,
                    worker_root=worker_root,
                    stage_id=str(stage_id),
                    stage=stage,
                )
                candidate = apply_candidate_evidence(
                    candidate,
                    peer_evidence.get(str(stage_id).upper()),
                )
                candidates.append(
                    recover_candidate_artifact(
                        candidate,
                        artifact_path=self._evaluator_candidate_artifact(),
                    )
                )
        return candidates

    @staticmethod
    def _metric_float(candidate: dict[str, Any]) -> float | None:
        return metric_float(candidate)

    def _write_stage_collection_outputs(
        self,
        *,
        worker_results: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> Path:
        collection_dir = self._merge_dir()
        collection_dir.mkdir(parents=True, exist_ok=True)
        (collection_dir / "global_candidates.jsonl").write_text(
            "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in candidates),
            encoding="utf-8",
        )
        (collection_dir / "worker_results.json").write_text(
            json.dumps(worker_results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload = {
            "candidates": candidates,
            "merge": {
                "merge_mode": "not_run",
                "reason": "global merge has not run",
            },
        }
        (collection_dir / "global_stage_map.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = [
            "# Long Horizon REPL Stage Collection",
            "",
            f"candidate_count: {len(candidates)}",
            f"worker_count: {len(worker_results)}",
            "global_merge: not_run",
            "",
            "## Workers",
        ]
        for result in worker_results:
            report.append(
                f"- {result.get('worker_id')}: status={result.get('status')} "
                f"best_stage={result.get('best_stage')} best_metric={result.get('best_metric')}"
            )
        report.extend(["", "## Stage Candidates"])
        for candidate in candidates:
            report.append(
                f"- {candidate.get('candidate_id')}: metric={candidate.get('metric_value')} "
                f"snapshot={candidate.get('snapshot_path')} submission_sha={candidate.get('submission_sha')}"
            )
        (collection_dir / "stage_collection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return collection_dir

    async def _run_live_merge_agent(
        self,
        workspace: Path,
        prompt: str,
        budget_sec: float,
        read_roots: list[Path],
    ) -> None:
        owner = getattr(self, "_merge_owner_solver", None)
        agent = owner._live_agent if owner is not None else None
        if owner is None or agent is None:
            raise RuntimeError("configured merge owner has no live agent")

        reduction_deadline = min(
            self.deadline,
            time.monotonic() + max(30.0, float(budget_sec or 0.0)),
        )
        owner.deadline = reduction_deadline
        hook = owner.orchestrator.make_llm_call_tracer(
            node_id=f"lnr:{owner.worker_id}:global_merge",
            process_id=(
                f"{getattr(owner.cfg, 'exp_id', '') or 'lhr'}:"
                f"{owner.worker_id}:global_merge"
            ),
            detail_prefix=f"mode=lnr;role=global_merge;worker={owner.worker_id}",
        )
        policy = _WallClockAutoContinuePolicy(
            deadline_monotonic=reduction_deadline,
            max_text_only_retries=0,
        )
        agent.swap_workspace(
            workspace_dir=workspace,
            path_guard_extra_roots=read_roots,
            readonly_dirs=["candidates", "dataset"],
            run_policy=policy,
            max_steps_override=40,
            bash_timeout_sec=max(30.0, reduction_deadline - time.monotonic()),
            bash_timeout_slow_sec=max(30.0, reduction_deadline - time.monotonic()),
            extra_env={
                **owner.worker_extra_env,
                **owner._task_runtime_extra_env(),
            },
            skill_registry=owner.skill_registry,
            task_type=owner.skill_task_category or None,
            skill_allow_names=owner.skill_allow_names,
            skill_tool_mode=owner.skill_tool_mode,
            skill_allow_generic_wildcard=owner.skill_allow_generic_wildcard,
            skill_visible_max=owner.skill_visible_max,
            resource_observer=owner.resource_observer,
            interaction_log_session="global_merge",
            interaction_log_phase="reduce",
            copy_memory_storage_to_new_workspace=False,
            on_llm_call=hook,
        )
        bash_tool = (
            getattr(getattr(agent, "availableTools", None), "tool_map", {}) or {}
        ).get("bash")
        if bash_tool is not None:
            bash_tool.forbid_host_absolute_paths = True
            bash_tool.bash_hard_fuse_deadline_monotonic = reduction_deadline
            bash_tool.bash_hard_fuse_finalization_reserve_sec = 0.0

        owner.state_machine.mark_run_status(
            "reducing",
            payload={"worker_id": owner.worker_id, "workspace": str(workspace)},
        )
        request: str | None = prompt
        while time.monotonic() < reduction_deadline:
            await agent.run(request)
            owner._accumulate_main_run_tokens(agent)
            request = None
            artifact = owner._evaluator_candidate_artifact()
            if any(
                (path / artifact).is_file()
                for path in (workspace / "finals").glob("final_*")
                if path.is_dir()
            ):
                break
            await asyncio.sleep(0.1)

    async def _close_merge_owner_agent(self) -> None:
        owner = getattr(self, "_merge_owner_solver", None)
        if owner is None or owner._live_agent is None:
            return
        agent = owner._live_agent
        owner._live_agent = None
        self._merge_owner_solver = None
        try:
            owner.state_machine.mark_run_status(
                "finished",
                payload={"worker_id": owner.worker_id, "phase": "global_merge"},
            )
        except OSError:
            logger.debug("[lnr] merge-owner final status write skipped", exc_info=True)
        try:
            await aclose_llm_clients(agent.llm)
        except Exception:
            logger.debug("[lnr] merge-owner llm close skipped", exc_info=True)

    async def _write_merge_outputs(
        self,
        *,
        worker_results: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        merge_dir = self._merge_dir()
        configured_wall_clock_sec = float(
            getattr(self.lhr, "global_merge_wall_clock_sec", 900.0) or 900.0
        )
        remaining_sec = max(0.0, float(self.deadline - time.monotonic()))
        wall_clock_sec = min(configured_wall_clock_sec, max(0.0, remaining_sec - 60.0))
        artifact_path = self._evaluator_candidate_artifact()
        owner = getattr(self, "_merge_owner_solver", None)
        merge_workspace = (
            owner.workspace_dir / ".scienceflow_global_merge"
            if owner is not None
            else merge_dir / "global_merge_workspace"
        )

        dataset_source = (
            Path(self.cfg.input_data_dir).expanduser().resolve(strict=False)
        )
        manifest = await run_global_merge(
            merge_dir=merge_dir,
            candidates=candidates,
            worker_results=worker_results,
            task_desc=self.task_desc,
            artifact_path=artifact_path,
            ledger_filename=self.ledger_filename,
            wall_clock_sec=wall_clock_sec,
            evaluator_manager=self.evaluator_manager,
            cfg=self.cfg,
            task_profile=self._evaluator_task_profile(),
            task_id=str(getattr(self.cfg, "exp_id", "") or ""),
            task_root=self.task_root_dir,
            dataset_source=dataset_source,
            merge_executor=self._run_live_merge_agent if owner is not None else None,
            workspace_override=merge_workspace,
            max_prediction_file_bytes=int(
                getattr(self.lhr, "merge_prediction_file_max_bytes", 536_870_912) or 0
            ),
            max_prediction_total_bytes=int(
                getattr(self.lhr, "merge_prediction_total_max_bytes", 2_147_483_648)
                or 0
            ),
            required_finals=max(
                1,
                int(getattr(self.lhr, "merge_required_finals", 3) or 3),
            ),
            max_finals=max(
                1,
                int(getattr(self.lhr, "merge_max_finals", 3) or 3),
                int(getattr(self.lhr, "merge_required_finals", 3) or 3),
            ),
        )
        payload = {"candidates": candidates, "merge": manifest}
        (merge_dir / "global_stage_map.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        report = [
            "# Long Horizon REPL Merge Report",
            "",
            f"status: {manifest.get('status')}",
            f"merge_mode: {manifest.get('merge_mode')}",
            f"global_merge_workspace: {manifest.get('workspace')}",
            "",
            "## Workers",
        ]
        for result in worker_results:
            report.append(
                f"- {result.get('worker_id')}: status={result.get('status')} "
                f"best_stage={result.get('best_stage')} best_metric={result.get('best_metric')} "
                f"main_cache_rate={result.get('main_cache_rate')}"
            )
        report.extend(
            [
                "",
                "## Global Merge",
                f"agent_status: {manifest.get('agent_status')}",
                f"packed_candidate_count: {manifest.get('packed_candidate_count')}",
                f"required_final_count: {manifest.get('required_final_count')}",
                f"final_count: {manifest.get('final_count')}",
                f"valid_final_count: {manifest.get('valid_final_count')}",
                f"requirement_met: {manifest.get('requirement_met')}",
                f"merge_budget_sec: {wall_clock_sec:.1f}",
                "manifest: global_merge_manifest.json",
            ]
        )
        (merge_dir / "merge_report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        return manifest

    async def _run_multi_worker(self) -> dict[str, Any]:
        n_workers = max(1, int(self.lhr.num_workers or 1))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_machine.mark_run_status(
            "running", payload={"num_workers": n_workers, "mode": "multi_worker"}
        )
        self._jsonl(
            "lhr_coordinator_events.jsonl",
            {"event": "multi_worker_start", "num_workers": n_workers},
        )
        tasks = [
            asyncio.create_task(self._run_one_worker(i, n_workers))
            for i in range(n_workers)
        ]
        live_aggregation_task = asyncio.create_task(
            self._aggregate_worker_state_periodically(
                n_workers=n_workers, run_status="running"
            )
        )
        try:
            worker_results = await asyncio.gather(*tasks)
        except BaseException:
            await self._close_merge_owner_agent()
            raise
        finally:
            live_aggregation_task.cancel()
            try:
                await live_aggregation_task
            except asyncio.CancelledError:
                pass
        self._aggregate_worker_state(n_workers=n_workers, run_status="reducing")
        try:
            candidates: list[dict[str, Any]] = []
            for result in worker_results:
                candidates.extend(self._load_worker_candidates(result))
            collection_dir = self._write_stage_collection_outputs(
                worker_results=worker_results,
                candidates=candidates,
            )
        except BaseException:
            await self._close_merge_owner_agent()
            raise
        merge_enabled = bool(getattr(self.lhr, "merge_enabled", True))
        merge_manifest: dict[str, Any] = {}
        try:
            if merge_enabled:
                merge_manifest = await self._write_merge_outputs(
                    worker_results=worker_results,
                    candidates=candidates,
                )
            else:
                self._jsonl(
                    "lhr_coordinator_events.jsonl",
                    {
                        "event": "global_merge_skipped",
                        "reason": "merge_enabled=false",
                        "candidate_count": len(candidates),
                        "stage_collection_dir": str(collection_dir),
                    },
                )
        finally:
            try:
                owner = getattr(self, "_merge_owner_solver", None)
                if owner is not None:
                    refreshed = owner._result(stop_reason="budget_expired")
                    for worker_result in worker_results:
                        if worker_result.get("worker_id") == owner.worker_id:
                            worker_result.update(refreshed)
                            break
            finally:
                await self._close_merge_owner_agent()
        canonicalized = canonicalize_worker_workspace_artifacts(
            worker_roots=[self._worker_root(i) for i in range(n_workers)],
            candidates=candidates,
            artifact_path=self._evaluator_candidate_artifact(),
        )
        self._jsonl(
            "lhr_coordinator_events.jsonl",
            {
                "event": "worker_workspace_artifacts_canonicalized",
                "artifact_count": len(canonicalized),
                "artifacts": canonicalized,
            },
        )
        self._refresh_submission_links(n_workers=n_workers, include_merge=merge_enabled)
        self._write_global_time_trace(worker_results)
        self._cleanup_coordinator_workspace_shell()

        main_in = sum(int(r.get("main_tokens_input", 0) or 0) for r in worker_results)
        main_cached = sum(
            int(r.get("main_tokens_cached", 0) or 0) for r in worker_results
        )
        main_out = sum(int(r.get("main_tokens_output", 0) or 0) for r in worker_results)
        main_calls = sum(int(r.get("main_llm_calls", 0) or 0) for r in worker_results)
        stage_in = sum(int(r.get("stage_tokens_input", 0) or 0) for r in worker_results)
        stage_cached = sum(
            int(r.get("stage_tokens_cached", 0) or 0) for r in worker_results
        )
        stage_calls = sum(int(r.get("stage_llm_calls", 0) or 0) for r in worker_results)
        estra_in = sum(int(r.get("estra_tokens_input", 0) or 0) for r in worker_results)
        estra_cached = sum(
            int(r.get("estra_tokens_cached", 0) or 0) for r in worker_results
        )
        estra_calls = sum(int(r.get("estra_llm_calls", 0) or 0) for r in worker_results)
        estra_switches = sum(
            int(r.get("estra_switch_stage_count", 0) or 0) for r in worker_results
        )
        estra_compacts = sum(
            int(r.get("estra_compact_count", 0) or 0) for r in worker_results
        )
        estra_decisions = sum(
            int(r.get("estra_decisions", 0) or 0) for r in worker_results
        )
        estra_continue = sum(
            int(r.get("estra_continue_count", 0) or 0) for r in worker_results
        )
        estra_redirect = sum(
            int(r.get("estra_redirect_count", 0) or 0) for r in worker_results
        )
        estra_switch_decisions = sum(
            int(r.get("estra_switch_count", 0) or 0) for r in worker_results
        )
        estra_current_continue = sum(
            int(r.get("estra_current_continue_count", 0) or 0) for r in worker_results
        )
        estra_current_redirect = sum(
            int(r.get("estra_current_redirect_count", 0) or 0) for r in worker_results
        )
        estra_stage_continue = sum(
            int(r.get("estra_stage_continue_count", 0) or 0) for r in worker_results
        )
        estra_stage_redirect = sum(
            int(r.get("estra_stage_redirect_count", 0) or 0) for r in worker_results
        )
        worker_run_succeeded = any(
            str(r.get("status") or "") == "success" for r in worker_results
        )
        partial_merge_available = bool(candidates) or bool(
            merge_manifest.get("final_count")
        )
        run_succeeded = worker_run_succeeded
        failure_kind, worker_error_kinds = (
            ("", [])
            if run_succeeded
            else self._multi_worker_failure_kind(worker_results)
        )
        stop_reason = "budget_expired" if run_succeeded else ""
        result = {
            "solver": self.solver_name,
            "status": "success" if run_succeeded else "failed",
            "stop_reason": stop_reason,
            "failure_kind": failure_kind,
            "worker_error_kinds": worker_error_kinds,
            "worker_run_succeeded": worker_run_succeeded,
            "partial_merge_available": partial_merge_available,
            "num_workers": n_workers,
            "merge_enabled": merge_enabled,
            "merge_status": str(merge_manifest.get("status") or ""),
            "merge_final_count": int(merge_manifest.get("final_count") or 0),
            "stage_count": len(candidates),
            "global_candidate_count": len(candidates),
            "estra_switch_stage_count": estra_switches,
            "estra_compact_count": estra_compacts,
            "estra_decisions": estra_decisions,
            "estra_continue_count": estra_continue,
            "estra_redirect_count": estra_redirect,
            "estra_switch_count": estra_switch_decisions,
            "estra_current_continue_count": estra_current_continue,
            "estra_current_redirect_count": estra_current_redirect,
            "estra_stage_continue_count": estra_stage_continue,
            "estra_stage_redirect_count": estra_stage_redirect,
            "main_cache_rate": main_cached / main_in if main_in else 0.0,
            "main_tokens_input": main_in,
            "main_tokens_cached": main_cached,
            "main_tokens_output": main_out,
            "main_llm_calls": main_calls,
            "stage_tokens_input": stage_in,
            "stage_tokens_cached": stage_cached,
            "stage_llm_calls": stage_calls,
            "estra_tokens_input": estra_in,
            "estra_tokens_cached": estra_cached,
            "estra_llm_calls": estra_calls,
            "worker_results": worker_results,
            "stage_collection_dir": str(collection_dir),
            "merge_dir": str(self._merge_dir()),
            "workspace_dir": str(self._merge_dir()),
            "ledger": str(collection_dir / "stage_collection_report.md"),
        }
        self._jsonl(
            "lhr_coordinator_events.jsonl", {"event": "multi_worker_done", **result}
        )
        terminal_status = "finished" if run_succeeded else "failed"
        self.state_machine.mark_run_status(
            terminal_status,
            payload={
                "num_workers": n_workers,
                "best_metric": result.get("best_metric"),
                "stop_reason": stop_reason,
                "failure_kind": failure_kind,
                "worker_error_kinds": worker_error_kinds,
                "partial_merge_available": partial_merge_available,
            },
        )
        self._aggregate_worker_state(n_workers=n_workers, run_status=terminal_status)
        return result

    async def _run_single(self, *, keep_agent_open: bool = False) -> dict[str, Any]:
        self._prepare_workspace()
        self._load_existing_stage_snapshots()
        if not self.snapshot_store.initialize_baseline():
            logger.warning(
                "[lnr] snapshot baseline initialization failed; falling back to allowlist snapshots"
            )
        self.state_machine.mark_run_status(
            "running",
            payload={"mode": "single_worker", "worker_id": self.worker_id or "W00"},
        )
        first_request = build_first_user_prompt(
            self.task_desc,
            wall_clock_budget_sec=int(self.lhr.wall_clock_budget_sec or 0),
            seed=int(getattr(self.lhr, "seed", 0) or 0),
            worker_id=self.worker_id or "W00",
            parallel_worker_snapshot=self._parallel_worker_snapshot_for_prompt(),
            resource_context=self._resource_context_for_prompt(),
            skill_hint=self._lnr_skill_hint(),
            evaluator_contract=self._evaluator_prompt_contract(),
            task_profile=self._evaluator_task_profile(),
            task_runtime_contract=self._task_runtime_prompt_contract(),
            initial_workspace_state=self.initial_workspace_state,
        )
        load_existing = (self.memory_dir / "ScienceAgent").is_dir()
        agent = self._make_agent(load_existing_memory=load_existing)
        stop_reason = "budget_expired"
        retain_agent = False
        try:
            request: str | None = None if load_existing else first_request
            resume_early_out: str | None = None
            post_budget_stage_commit_runs = 0
            if load_existing:
                resume_result = await resume_loaded_agent_from_memory(
                    agent,
                    effective_max=max(1, int(self.lhr.max_steps or 1)),
                    initial_max=max(1, int(self.lhr.max_steps or 1)),
                )
                self._jsonl(
                    "lhr_resume_events.jsonl",
                    {"event": "agent_memory_resume", **resume_result.to_event()},
                )
                resume_early_out = resume_result.early_out
            while (
                time.monotonic() < self.deadline
                or self._pending_stage_commit_text_active()
            ):
                if (
                    time.monotonic() >= self.deadline
                    and self._pending_stage_commit_text_active()
                ):
                    post_budget_stage_commit_runs += 1
                    if post_budget_stage_commit_runs > 3:
                        pending = getattr(self, "pending_text_stage_commit", {})
                        pending_stage_id = (
                            pending.get("stage_id") if isinstance(pending, dict) else ""
                        )
                        pending_attempts = (
                            pending.get("attempts") if isinstance(pending, dict) else ""
                        )
                        self._jsonl(
                            "lhr_stage_commit_events.jsonl",
                            {
                                "event": "stage_commit_text_post_budget_cap_reached",
                                "stage_id": pending_stage_id,
                                "attempts": pending_attempts,
                            },
                        )
                        break
                    self._extend_stage_commit_text_policy_deadline(agent)
                if resume_early_out is not None:
                    out = resume_early_out
                    resume_early_out = None
                else:
                    out = await agent.run(request)
                    self._accumulate_main_run_tokens(agent)
                _ = out
                request = None
                if self.pending_estra:
                    await aclose_llm_clients(agent.llm)
                    await self._restore_pending_estra()
                    agent = self._make_agent(load_existing_memory=True)
                    continue
                if (
                    time.monotonic() >= self.deadline
                    and not self._pending_stage_commit_text_active()
                ):
                    break
                # Continue the same REPL memory after text-only/round-limit returns.
                await asyncio.sleep(0.1)
            result = self._result(stop_reason=stop_reason)
            self.state_machine.mark_run_status(
                "finished", payload={"best_metric": result.get("best_metric")}
            )
            if keep_agent_open:
                self._live_agent = agent
                retain_agent = True
            return result
        except BaseException as exc:
            try:
                self.state_machine.mark_run_status(
                    "failed", payload={"error": f"{type(exc).__name__}: {exc}"}
                )
            except OSError:
                logger.debug(
                    "[lnr] state-machine failure status write failed", exc_info=True
                )
            raise
        finally:
            if not retain_agent:
                try:
                    await aclose_llm_clients(agent.llm)
                except Exception:
                    logger.debug("[lnr] llm close skipped", exc_info=True)
            metric_llm = getattr(self, "_metric_validity_feedback_llm", None)
            if metric_llm is not None:
                try:
                    await aclose_llm_clients(metric_llm)
                except Exception:
                    logger.debug(
                        "[lnr] metric-validity feedback llm close skipped",
                        exc_info=True,
                    )

    async def run(self) -> dict[str, Any]:
        if str(getattr(self, "worker_id", "") or ""):
            return await self._run_single()
        return await self._run_multi_worker()

    @staticmethod
    async def _ask_agent_tool_stream_guarded(
        agent: Any,
        *,
        llm: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        guarded = getattr(agent, "_ask_tool_stream_guarded", None)
        if callable(guarded):
            return await guarded(llm=llm, **kwargs)
        # Compatibility for focused test doubles and custom legacy agents. The
        # production ScienceAgent always provides the guarded path above.
        target = llm or agent.llm
        handle = StreamHandle()
        return await target.ask_tool_stream(handle=handle, **kwargs)
