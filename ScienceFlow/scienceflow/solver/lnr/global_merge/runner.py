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
import json
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from scienceflow.gates.evaluator import (
    EvalContext,
    EvaluationRequest,
    EvaluatorManager,
)
from scienceflow.gates import GateService
from scienceflow.solver.lnr.global_merge.candidate_pack import (
    pack_candidates,
)
from scienceflow.solver.lnr.global_merge.fallback import metric_float
from scienceflow.solver.lnr.global_merge.final_artifacts import (
    _canonicalize_workspace_finals,
    _cleanup_legacy_single_winner_outputs,
    _expose_dataset,
    _write_fallback_finals,
)
from scienceflow.solver.lnr.global_merge.prompt import build_global_merge_prompt
from scienceflow.solver.lnr.submission_links import refresh_submission_links


MergeExecutor = Callable[[Path, str, float, list[Path]], Awaitable[None]]


def _metric_event_dict(event: Any) -> dict[str, Any]:
    if event is None:
        return {}
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(event) if isinstance(event, dict) else {}


def _evaluate_finals(
    *,
    finals_dir: Path,
    artifact_path: str,
    evaluator_manager: EvaluatorManager,
    cfg: Any,
    task_profile: str,
    task_id: str,
    task_root: Path,
    dataset_source: Path | None,
) -> list[dict[str, Any]]:
    finals: list[dict[str, Any]] = []
    artifact = artifact_path or "submission.csv"
    service = GateService(evaluator_manager)
    for final_dir in sorted(p for p in finals_dir.glob("final_*") if p.is_dir()):
        artifact_file = final_dir / artifact
        record: dict[str, Any] = {
            "candidate_id": final_dir.name,
            "final_dir": str(final_dir),
            "artifact_path": artifact,
            "artifact_exists": artifact_file.is_file(),
        }
        if artifact_file.is_file():
            _expose_dataset(final_dir, dataset_source)
            ctx = EvalContext(
                task_profile=task_profile,
                task_id=task_id,
                task_root=task_root,
                workspace=final_dir,
                worker_id="MERGE",
                stage_id=final_dir.name,
                cfg=cfg,
                metadata={"metric_event": {}},
            )
            try:
                outcomes = list(
                    service.evaluate(
                        EvaluationRequest(context=ctx, trigger="global_merge")
                    )
                )
                evaluation_error = ""
            except Exception as exc:
                outcomes = []
                evaluation_error = f"{type(exc).__name__}: {exc}"
            outcome = outcomes[0] if len(outcomes) == 1 else None
            event = outcome.event if outcome is not None else None
            record["metric_event"] = _metric_event_dict(event)
            if event is not None:
                record.update(
                    {
                        "metric_value": event.metric_value,
                        "metric_name": event.metric_name,
                        "lower_is_better": event.lower_is_better,
                        "validation_ok": event.validation_ok,
                        "candidate_ready": event.candidate_ready,
                        "selection_eligible": event.selection_eligible,
                        "artifact_sha": event.artifact_sha,
                        "evaluator_status": event.evaluator_status,
                        "gate_decision": outcome.decision.to_dict(),
                    }
                )
            else:
                evaluator_cfg = (
                    cfg.get("evaluator")
                    if isinstance(cfg, dict)
                    else getattr(cfg, "evaluator", None)
                )
                evaluator_enabled = (
                    evaluator_cfg.get("enabled", True)
                    if isinstance(evaluator_cfg, dict)
                    else getattr(evaluator_cfg, "enabled", True)
                )
                # Disabled evaluation historically means "not evaluated", not
                # "invalid". Keep that compatibility for final artifacts.
                if not outcomes and not evaluation_error and evaluator_enabled is False:
                    reason_code = ""
                    message = ""
                elif evaluation_error:
                    reason_code = "evaluator_service_exception"
                    message = evaluation_error
                elif outcomes:
                    reason_code = "evaluator_multiple_outcomes"
                    message = f"evaluator produced {len(outcomes)} outcomes for one final candidate"
                else:
                    reason_code = "evaluator_no_outcome"
                    message = "evaluator did not produce a final candidate outcome"
                if reason_code:
                    record.update(
                        {
                            "validation_ok": False,
                            "candidate_ready": False,
                            "selection_eligible": False,
                            "evaluator_status": reason_code,
                            "gate_decision": {
                                "action": "reject",
                                "accepted": False,
                                "candidate_ready": False,
                                "selection_eligible": False,
                                "reason_code": reason_code,
                                "message": message,
                            },
                        }
                    )
        source_meta = final_dir / "source_candidate_metadata.json"
        if source_meta.is_file():
            try:
                source_candidate = json.loads(source_meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                source_candidate = {}
            if isinstance(source_candidate, dict):
                record["source_candidate"] = source_candidate
                if metric_float(record) is None:
                    for key in (
                        "metric_value",
                        "metric_name",
                        "lower_is_better",
                        "metric_validity",
                        "selection_eligible",
                        "candidate_ready",
                        "submission_sha",
                    ):
                        if key in source_candidate:
                            record[key] = source_candidate[key]
        (final_dir / "eval_result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        finals.append(record)
    return finals


async def run_global_merge(
    *,
    merge_dir: Path,
    candidates: list[dict[str, Any]],
    worker_results: list[dict[str, Any]],
    task_desc: str,
    artifact_path: str,
    ledger_filename: str,
    wall_clock_sec: float,
    evaluator_manager: EvaluatorManager,
    cfg: Any,
    task_profile: str,
    task_id: str,
    task_root: Path,
    dataset_source: Path | None,
    merge_executor: MergeExecutor | None = None,
    workspace_override: Path | None = None,
    max_prediction_file_bytes: int = 536_870_912,
    max_prediction_total_bytes: int = 2_147_483_648,
    required_finals: int = 3,
    max_finals: int = 3,
) -> dict[str, Any]:
    required_finals = max(1, int(required_finals))
    max_finals = max(required_finals, int(max_finals))
    merge_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_single_winner_outputs(merge_dir)
    workspace = workspace_override or (merge_dir / "global_merge_workspace")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    _expose_dataset(workspace, dataset_source)
    (merge_dir / "global_candidates.jsonl").write_text(
        "".join(
            json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in candidates
        ),
        encoding="utf-8",
    )
    (merge_dir / "worker_results.json").write_text(
        json.dumps(worker_results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_dir = workspace / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "global_candidates.jsonl").write_text(
        "".join(
            json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in candidates
        ),
        encoding="utf-8",
    )
    (evidence_dir / "worker_results.json").write_text(
        json.dumps(worker_results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "index.json").write_text(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "candidate_evidence": "global_candidates.jsonl",
                "worker_results": "worker_results.json",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    packed = pack_candidates(
        merge_dir=merge_dir,
        merge_workspace=workspace,
        candidates=candidates,
        artifact_path=artifact_path,
        ledger_filename=ledger_filename,
        max_prediction_file_bytes=max_prediction_file_bytes,
        max_prediction_total_bytes=max_prediction_total_bytes,
    )
    prompt = build_global_merge_prompt(
        task_desc=task_desc,
        artifact_path=artifact_path,
        candidate_count=len(packed),
        wall_clock_sec=wall_clock_sec,
        required_finals=required_finals,
    )
    (workspace / "merge_task.md").write_text(prompt, encoding="utf-8")

    agent_status = "skipped_no_candidates"
    agent_error = ""
    if packed and float(wall_clock_sec or 0.0) >= 30.0:
        if merge_executor is None:
            agent_status = "skipped_no_live_owner"
        else:
            try:
                read_roots = sorted(
                    {
                        path.resolve(strict=True)
                        for path in (workspace / "candidates").rglob("*")
                        if path.is_file() and path.is_symlink()
                    },
                    key=str,
                )
                if dataset_source is not None and dataset_source.exists():
                    read_roots.append(dataset_source.resolve(strict=True))
                await asyncio.wait_for(
                    merge_executor(workspace, prompt, wall_clock_sec, read_roots),
                    timeout=max(5.0, float(wall_clock_sec or 0)),
                )
                agent_status = "completed"
            except Exception as exc:  # noqa: BLE001 - merge has deterministic fallback.
                agent_status = "failed"
                agent_error = f"{type(exc).__name__}: {exc}"
    elif packed:
        agent_status = "skipped_no_merge_budget"

    workspace_finals = workspace / "finals"
    finals_dir = merge_dir / "finals"
    promoted_finals = _canonicalize_workspace_finals(
        workspace_finals=workspace_finals,
        finals_dir=finals_dir,
        merge_dir=merge_dir,
        candidates=candidates,
        artifact_path=artifact_path,
        ledger_filename=ledger_filename,
        max_finals=max_finals,
    )
    fallback_final_sources = _write_fallback_finals(
        finals_dir=finals_dir,
        candidates=candidates,
        artifact_path=artifact_path,
        ledger_filename=ledger_filename,
        max_finals=max_finals,
    )
    finals = _evaluate_finals(
        finals_dir=finals_dir,
        artifact_path=artifact_path,
        evaluator_manager=evaluator_manager,
        cfg=cfg,
        task_profile=task_profile,
        task_id=task_id,
        task_root=task_root,
        dataset_source=dataset_source,
    )
    shutil.rmtree(workspace / "tmp", ignore_errors=True)

    # A final artifact is not a research Stage. MLEbench finals commonly have
    # authoritative schema validation but no recomputed training metric, so
    # preserve the historical validation-based final contract. Gate acceptance
    # remains authoritative only for Stage admission; its final trace is still
    # recorded for audit and task types whose evaluator can recompute a metric.
    valid_final_count = sum(
        1
        for final in finals
        if final.get("artifact_exists") is True
        and final.get("validation_ok") is not False
    )
    requirement_met = valid_final_count >= required_finals
    status = (
        "success"
        if requirement_met
        else "insufficient_finals" if finals else "no_final_artifacts"
    )
    manifest = {
        "status": status,
        "agent_status": agent_status,
        "agent_error": agent_error,
        "artifact_path": artifact_path,
        "created_at": time.time(),
        "candidate_count": len(candidates),
        "packed_candidate_count": len(packed),
        "required_final_count": required_finals,
        "requirement_met": requirement_met,
        "final_count": len(finals),
        "valid_final_count": valid_final_count,
        "finals": finals,
        "merge_mode": "worker_reduce",
        "promoted_finals": promoted_finals,
        "fallback_final_sources": fallback_final_sources,
        "workspace": str(workspace),
    }
    raw_submission_dir = getattr(cfg, "submission_dir", None)
    manifest["submission_links"] = (
        refresh_submission_links(
            submission_dir=Path(raw_submission_dir),
            artifact_path=artifact_path,
            merge_dir=merge_dir,
        )
        if raw_submission_dir is not None and str(raw_submission_dir).strip()
        else []
    )
    (merge_dir / "global_merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
