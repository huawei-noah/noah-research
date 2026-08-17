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

"""Trusted shared evaluator for SciModelingBench ordered-candidate Tasks."""

from __future__ import annotations

import fcntl
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    canonical_candidates_sha,
    evaluator_config,
    expected_package_version,
    file_sha256,
    load_official_task,
    mapping,
    metric_config,
    source_config,
    stable_json_sha,
    submission_config,
    task_config,
    write_json,
)
from query_budget import (  # noqa: E402
    evaluate_with_query_budget,
    query_budget_identity,
    query_state_path,
)


TASK_CACHE_SCHEMA_VERSION = 1
OFFICIAL_TASK_LOAD_ATTEMPTS = 3


def evaluate(
    *,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = workspace_dir
    try:
        manifest = _validate_public_dataset(dataset_dir, config=config)
        candidates = _load_candidates(artifact_path)
        query_sha = canonical_candidates_sha(candidates)
        artifact_sha = file_sha256(artifact_path)
        selected = task_config(config)
        local_task_id = str(selected.get("id") or "").strip()
        scope, owner = query_budget_identity(config)
        state_path = query_state_path(
            task_dir=task_dir,
            task_id=local_task_id,
            scope=scope,
            owner=owner,
        )

        def evaluate_official() -> Mapping[str, Any]:
            task, task_cache_hit = _load_or_create_official_task(
                config=config,
                manifest=manifest,
                state_path=state_path,
                dataset_dir=dataset_dir,
            )
            _validate_loaded_task(task, config=config, manifest=manifest)
            summary = _evaluation_summary(task.evaluate(candidates))
            summary["official_task_cache_hit"] = task_cache_hit
            return summary

        budget = evaluate_with_query_budget(
            task_dir=task_dir,
            config=config,
            task_id=local_task_id,
            query=candidates,
            query_limit=_max_queries(config),
            evaluate=evaluate_official,
        )
        if budget.exhausted or budget.payload is None:
            return _invalid_payload(
                config,
                "scientific_design_query_budget_exhausted",
                "official evaluator query budget exhausted; "
                f"queries_remaining=0/{budget.query_limit}",
                extra={
                    "query_limit": budget.query_limit,
                    "queries_used": budget.queries_used,
                    "queries_remaining": 0,
                    "queries_terminal": True,
                },
            )
        summary = dict(budget.payload)
        used = budget.queries_used
        remaining = budget.queries_remaining
        cached = budget.cache_hit
        task_cache_hit = summary.pop("official_task_cache_hit", None)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _invalid_payload(config, "scientific_design_input_invalid", str(exc))
    except (ImportError, ModuleNotFoundError) as exc:
        return _invalid_payload(
            config,
            "scientific_design_dependency_unavailable",
            f"install sci-modeling-bench=={expected_package_version(config)}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - trusted evaluator process boundary.
        return _invalid_payload(
            config,
            "scientific_design_official_evaluator_failed",
            f"{type(exc).__name__}: {exc}",
        )

    metrics = summary["metrics"]
    primary_metric = str(summary["primary_metric"])
    score = metrics.get(primary_metric)
    candidate_ready = bool(summary["evaluation_eligible"] and score is not None)
    submission_valid = bool(summary["submission_valid"])
    if not submission_valid:
        status = "invalid_submission"
        reason_code = "scientific_design_submission_invalid"
    elif not candidate_ready:
        status = "invalid_candidate_batch"
        reason_code = "scientific_design_candidate_batch_ineligible"
    else:
        status = "ok"
        reason_code = "scientific_design_official_task_scored"

    feedback = _format_feedback(
        summary,
        config=config,
        remaining=remaining,
        limit=_max_queries(config),
    )
    extra = {
        "official_task_id": summary["task_id"],
        "metrics": metrics,
        "metric_directions": summary["metric_directions"],
        "primary_metric": primary_metric,
        "expected_candidates": summary["expected_candidates"],
        "submitted_candidates": summary["submitted_candidates"],
        "valid_candidates": summary["valid_candidates"],
        "invalid_candidates": summary["invalid_candidates"],
        "all_candidates_valid": summary["all_candidates_valid"],
        "evaluation_eligible": summary["evaluation_eligible"],
        "summary_size": summary["summary_size"],
        "reference_scope": summary["reference_scope"],
        "reference_size": summary["reference_size"],
        "dataset_revision": manifest.resolved_revision,
        "dataset_repo_id": manifest.repo_id,
        "protocol_id": manifest.protocol_id,
        "artifact_sha256": artifact_sha,
        "query_sha256": query_sha,
        "query_limit": _max_queries(config),
        "queries_used": used,
        "queries_remaining": remaining,
        "queries_terminal": remaining == 0,
        "query_cache_hit": cached,
        "official_task_cache_enabled": _task_cache_enabled(config),
        "official_task_cache_hit": task_cache_hit,
        "evaluator": "sci_modeling_bench_official_task",
        "package_version": expected_package_version(config),
        "candidate_outputs_exposed": False,
    }
    for name in ("candidate_pool_size", "evaluated_candidates", "ignored_candidates"):
        if summary.get(name) is not None:
            extra[name] = summary[name]
    return {
        "valid": candidate_ready,
        "candidate_ready": candidate_ready,
        "selection_eligible": candidate_ready,
        "status": status,
        "metric": {
            "name": primary_metric,
            "value": score if candidate_ready else None,
            "lower_is_better": summary["metric_direction"] == "minimize",
        },
        "metric_validity": "high" if candidate_ready else "low",
        "reason_code": reason_code,
        "feedback": feedback,
        "extra": extra,
    }

def _load_or_create_official_task(
    *,
    config: Mapping[str, Any],
    manifest: Any,
    state_path: Path,
    dataset_dir: Path,
) -> tuple[Any, bool]:
    if not _task_cache_enabled(config):
        return _load_official_task_with_retry(config, dataset_dir=dataset_dir), False

    cache_path, metadata_path = _task_cache_paths(state_path)
    with _exclusive_file_lock(_task_cache_lock_path(state_path)):
        identity = _task_cache_identity(config, manifest)
        identity_sha = stable_json_sha(identity)
        try:
            metadata = _read_task_cache_metadata(metadata_path)
            if (
                metadata.get("schema_version") != TASK_CACHE_SCHEMA_VERSION
                or metadata.get("identity_sha256") != identity_sha
                or metadata.get("size_bytes") != cache_path.stat().st_size
                or metadata.get("sha256") != file_sha256(cache_path)
            ):
                raise ValueError("official Task cache identity or size mismatch")
            from cloudpickle import load

            # Cloudpickle is executable serialization. This path must remain in the
            # trusted evaluator state and must never accept an Agent-provided file.
            with cache_path.open("rb") as handle:
                task = load(handle)
            _validate_loaded_task(task, config=config, manifest=manifest)
            return task, True
        except Exception:  # noqa: BLE001 - a bad cache is rebuilt from the pinned Task.
            cache_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            task = _load_official_task_with_retry(config, dataset_dir=dataset_dir)
            _validate_loaded_task(task, config=config, manifest=manifest)
            _write_official_task_cache_unlocked(
                task,
                config=config,
                manifest=manifest,
                cache_path=cache_path,
                metadata_path=metadata_path,
            )
            return task, False


def _write_official_task_cache_unlocked(
    task: Any,
    *,
    config: Mapping[str, Any],
    manifest: Any,
    cache_path: Path,
    metadata_path: Path,
) -> bool:
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from cloudpickle import __version__ as cloudpickle_version
        from cloudpickle import dump

        identity = _task_cache_identity(config, manifest)
        with temporary.open("wb") as handle:
            dump(task, handle, protocol=5)
        temporary.chmod(0o600)
        temporary.replace(cache_path)
        write_json(
            metadata_path,
            {
                "schema_version": TASK_CACHE_SCHEMA_VERSION,
                "identity": identity,
                "identity_sha256": stable_json_sha(identity),
                "size_bytes": cache_path.stat().st_size,
                "sha256": file_sha256(cache_path),
                "cloudpickle_version": cloudpickle_version,
            },
        )
        metadata_path.chmod(0o600)
        return True
    except Exception:  # noqa: BLE001 - caching must never invalidate a score.
        cache_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _task_cache_paths(state_path: Path) -> tuple[Path, Path]:
    directory = state_path.parent
    return (
        directory / "official_task-v1.cloudpickle",
        directory / "official_task-v1.json",
    )


def _task_cache_lock_path(state_path: Path) -> Path:
    return state_path.parent / "official_task-v1.lock"


def _official_task_build_lock_path(dataset_dir: Path) -> Path:
    directory = dataset_dir.resolve().parent.parent / ".evaluator_locks"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory / "official-task-build.lock"


def _load_official_task_with_retry(
    config: Mapping[str, Any], *, dataset_dir: Path
) -> Any:
    # Different Tasks may share one cold Hugging Face Dataset cache. Serialize
    # their first load so concurrent workers do not race the Hub/cache builder.
    with _exclusive_file_lock(_official_task_build_lock_path(dataset_dir)):
        for attempt in range(1, OFFICIAL_TASK_LOAD_ATTEMPTS + 1):
            try:
                return load_official_task(config)
            except Exception as exc:  # noqa: BLE001 - retry only Dataset loading.
                if (
                    type(exc).__name__ != "DatasetLoadError"
                    or attempt == OFFICIAL_TASK_LOAD_ATTEMPTS
                ):
                    raise
                time.sleep(float(attempt * 2))
    raise AssertionError("unreachable official Task retry loop")


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _read_task_cache_metadata(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return mapping(data, name="official Task cache metadata")


def _task_cache_identity(config: Mapping[str, Any], manifest: Any) -> dict[str, Any]:
    from cloudpickle import __version__ as cloudpickle_version

    source = source_config(config)
    return {
        "schema_version": TASK_CACHE_SCHEMA_VERSION,
        "evaluation": _evaluation_identity(config, manifest),
        "task_factory": source.get("task_factory"),
        "task_kwargs": source.get("task_kwargs"),
        "python_implementation": platform.python_implementation(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "cloudpickle_version": cloudpickle_version,
    }


def _task_cache_enabled(config: Mapping[str, Any]) -> bool:
    return bool(evaluator_config(config).get("cache_official_task", True))


def _validate_loaded_task(
    task: Any,
    *,
    config: Mapping[str, Any],
    manifest: Any,
) -> None:
    source = source_config(config)
    expected_task_id = str(source.get("benchmark_task_id") or "")
    if task.task_id != expected_task_id or manifest.task_id != task.task_id:
        raise ValueError("loaded official Task identity differs from public input")
    expected_size = int(submission_config(config).get("size") or 0)
    if task.submission_size != expected_size:
        raise ValueError("loaded official Task submission size differs from task package")
    expected_summary_size = int(
        submission_config(config).get("summary_size") or 0
    )
    if expected_summary_size and task.summary_size != expected_summary_size:
        raise ValueError("loaded official Task summary size differs from task package")
    if task.primary_metric != _primary_metric(config):
        raise ValueError("loaded official Task primary metric differs from task package")
    if task.dataset.resolved_revision != manifest.resolved_revision:
        raise ValueError("loaded official Task Dataset revision differs from public input")


def _evaluation_summary(result: Any) -> dict[str, Any]:
    official = result.model_dump(mode="json")
    metrics = official.get("metrics")
    metric_directions = official.get("metric_directions")
    if not isinstance(metrics, dict) or not isinstance(metric_directions, dict):
        raise ValueError("official evaluation has no metric mapping")
    primary_metric = str(official.get("primary_metric") or "")
    if primary_metric not in metric_directions:
        raise ValueError(f"official evaluation has no direction for {primary_metric!r}")
    summary = {
        "task_id": str(official.get("task_id") or ""),
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "metric_directions": {
            str(key): str(value) for key, value in metric_directions.items()
        },
        "primary_metric": primary_metric,
        "metric_direction": str(
            official.get("metric_direction") or metric_directions[primary_metric]
        ),
        "submission_valid": bool(official.get("submission_valid")),
        "all_candidates_valid": bool(official.get("all_candidates_valid")),
        "evaluation_eligible": bool(official.get("evaluation_eligible")),
        "expected_candidates": int(official.get("expected_candidates") or 0),
        "submitted_candidates": int(official.get("submitted_candidates") or 0),
        "valid_candidates": int(official.get("valid_candidates") or 0),
        "invalid_candidates": int(official.get("invalid_candidates") or 0),
        "summary_size": int(official.get("summary_size") or 1),
        "reference_scope": official.get("reference_scope"),
        "reference_size": official.get("reference_size"),
        "first_violation": _first_violation(official),
    }
    for name in ("candidate_pool_size", "evaluated_candidates", "ignored_candidates"):
        value = official.get(name)
        summary[name] = int(value) if value is not None else None
    return summary


def _first_violation(official: Mapping[str, Any]) -> dict[str, str] | None:
    submission = official.get("submission_validation")
    if isinstance(submission, Mapping):
        violations = submission.get("violations")
        if isinstance(violations, list) and violations:
            item = violations[0]
            if isinstance(item, Mapping):
                return {
                    "code": str(item.get("code") or "invalid_submission"),
                    "message": str(item.get("message") or "submission validation failed"),
                }
    candidates = official.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            validation = candidate.get("validation")
            violations = validation.get("violations") if isinstance(validation, Mapping) else None
            if isinstance(violations, list) and violations:
                item = violations[0]
                if isinstance(item, Mapping):
                    return {
                        "code": str(item.get("code") or "invalid_candidate"),
                        "message": str(item.get("message") or "candidate validation failed"),
                    }
    return None


def _format_feedback(
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    remaining: int,
    limit: int,
) -> str:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    primary = str(summary.get("primary_metric") or _primary_metric(config))
    score = metrics.get(primary)
    if not bool(summary.get("evaluation_eligible")) or score is None:
        violation = summary.get("first_violation")
        detail = "candidate batch is not eligible for official metrics"
        if isinstance(violation, Mapping):
            detail = f"{violation.get('code')}: {violation.get('message')}"
        return (
            f"official evaluation invalid: {detail}; "
            f"queries_remaining={remaining}/{limit}"
        )
    pieces = [f"{primary}={_format_metric(score)}"]
    for name in _feedback_metrics(config):
        if name == primary or name not in metrics:
            continue
        pieces.append(f"{name}={_format_metric(metrics[name])}")
    pieces.append(f"queries_remaining={remaining}/{limit}")
    return "official SciModelingBench: " + "; ".join(pieces)


def _format_metric(value: Any) -> str:
    return format(float(value), ".6g")


def _max_queries(config: Mapping[str, Any]) -> int:
    raw = evaluator_config(config).get("max_queries", 10)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 10
    return value if value > 0 else 10


def _feedback_metrics(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = evaluator_config(config).get("feedback_metrics", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())[:3]


def _primary_metric(config: Mapping[str, Any]) -> str:
    value = str(metric_config(config).get("name") or "").strip()
    if not value:
        raise ValueError("task metric.name must be non-empty")
    return value


def _primary_lower_is_better(config: Mapping[str, Any]) -> bool:
    return bool(metric_config(config).get("lower_is_better", False))


def _load_candidates(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        raise ValueError("submission must be a JSON array or an object with a candidates array")
    return candidates


def _validate_public_dataset(dataset_dir: Path, *, config: Mapping[str, Any]) -> Any:
    from sci_modeling_bench import AgentInputManifest

    manifest_path = dataset_dir / "dataset_manifest.json"
    files_path = dataset_dir / "dataset_files.json"
    contract_path = dataset_dir / "task_contract.json"
    manifest = AgentInputManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = json.loads(files_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(files, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("public dataset indexes must contain JSON objects")

    selected = task_config(config)
    source = source_config(config)
    expected = {
        "task_id": str(source.get("benchmark_task_id") or ""),
        "repo_id": str(source.get("repo_id") or ""),
        "config_name": str(source.get("config_name") or ""),
        "split": str(source.get("split") or ""),
        "resolved_revision": str(source.get("revision") or ""),
        "protocol_id": str(source.get("protocol_id") or ""),
    }
    for field, value in expected.items():
        if value and getattr(manifest, field) != value:
            raise ValueError(
                f"dataset manifest {field} mismatch: expected {value!r}, "
                f"got {getattr(manifest, field)!r}"
            )
    if files.get("task_id") != selected.get("id"):
        raise ValueError("dataset_files.json task_id mismatch")
    if files.get("benchmark_task_id") != manifest.task_id:
        raise ValueError("dataset_files.json benchmark_task_id mismatch")
    if contract.get("task_id") != selected.get("id"):
        raise ValueError("task_contract.json task_id mismatch")
    if contract.get("benchmark_task_id") != manifest.task_id:
        raise ValueError("task_contract.json benchmark_task_id mismatch")
    contract_submission = mapping(
        contract.get("submission"), name="task contract submission"
    )
    if dict(contract_submission) != dict(submission_config(config)):
        raise ValueError("task_contract.json submission contract mismatch")

    raw_files = files.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("dataset_files.json files must be a list")
    declared = {view.name: view for view in manifest.views}
    indexed: set[str] = set()
    verify_hashes = bool(evaluator_config(config).get("verify_input_hashes", False))
    for item in raw_files:
        entry = mapping(item, name="dataset file entry")
        name = str(entry.get("view") or "")
        if name not in declared or name in indexed:
            raise ValueError(f"invalid or duplicate dataset view index: {name!r}")
        indexed.add(name)
        relative = Path(str(entry.get("path") or ""))
        path = (dataset_dir / relative).resolve()
        try:
            path.relative_to(dataset_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"dataset view path escapes dataset directory: {relative}") from exc
        if not path.is_file():
            raise ValueError(f"dataset view file is missing: {relative}")
        if verify_hashes and file_sha256(path) != str(entry.get("sha256") or ""):
            raise ValueError(f"dataset view checksum mismatch: {relative}")
    if indexed != set(declared):
        raise ValueError("dataset file index does not cover every manifest view")
    return manifest


def _evaluation_identity(config: Mapping[str, Any], manifest: Any) -> dict[str, Any]:
    selected = task_config(config)
    return {
        "local_task_id": selected.get("id"),
        "benchmark_task_id": manifest.task_id,
        "package_version": expected_package_version(config),
        "dataset_repo_id": manifest.repo_id,
        "dataset_revision": manifest.resolved_revision,
        "protocol_id": manifest.protocol_id,
        "primary_metric": _primary_metric(config),
        "submission_size": int(submission_config(config).get("size") or 0),
        "summary_size": int(
            submission_config(config).get("summary_size") or 0
        ),
    }


def _invalid_payload(
    config: Mapping[str, Any],
    reason_code: str,
    feedback: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "valid": False,
        "candidate_ready": False,
        "selection_eligible": False,
        "status": "invalid",
        "metric": {
            "name": _primary_metric(config),
            "value": None,
            "lower_is_better": _primary_lower_is_better(config),
        },
        "metric_validity": "low",
        "reason_code": reason_code,
        "feedback": feedback,
        "extra": dict(extra or {}),
    }
