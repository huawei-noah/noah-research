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

#!/usr/bin/env python3
"""Materialize one SciModelingBench Task's disclosure-scoped Agent input."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    canonical_candidates_sha,
    evaluator_config,
    file_sha256,
    load_official_task,
    mapping,
    metric_config,
    resolve_bundle_views,
    source_config,
    submission_config,
    stable_json_sha,
    task_config,
    write_json,
)

PUBLIC_INPUT_CACHE_SCHEMA_VERSION = 3
PUBLIC_INPUT_PRODUCER_VERSION = 3


def prepare(task_id: str, output_dir: Path) -> dict[str, Any]:
    from scienceflow.core.task_package import find_task_package

    spec = find_task_package(task_id)
    if spec is None:
        raise ValueError(f"SciModelingBench task package not found: {task_id}")
    config = dict(spec.config)
    task = load_official_task(config)
    reports = tuple(task.prepare())
    identity = _preparation_identity(task_id, config, task)
    lock_path = output_dir / ".prepare.lock"
    with _exclusive_file_lock(lock_path):
        reused = _reuse_prepared_output(output_dir, identity)
        if reused is not None:
            reused["public_input_cache_hit"] = True
            reused["derived_artifacts"] = _serialize_reports(reports)
            return reused

        result = _materialize(task_id, output_dir, config=config, task=task)
        result["public_input_cache_hit"] = False
        result["derived_artifacts"] = _serialize_reports(reports)
        _write_preparation_state(output_dir, identity=identity, result=result)
        return result


def _materialize(
    task_id: str,
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    task: Any,
) -> dict[str, Any]:
    bundle = task.build_input()
    _validate_contract(task_id, config, task, bundle)

    public_dir = output_dir / "public"
    validation_dir = output_dir / "validation"
    views_dir = public_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    _guard_existing_output(public_dir, task_id)

    tables = resolve_bundle_views(bundle)
    manifest_views = {view.name: view for view in bundle.manifest.views}
    files: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for name, table in tables.items():
        view = manifest_views[name]
        path = views_dir / f"{name}.parquet"
        expected_paths.add(path.resolve())
        temporary = path.with_suffix(".parquet.tmp")
        table.to_parquet(str(temporary))
        temporary.replace(path)
        columns = list(table.column_names)
        declared_columns = [field.name for field in view.fields]
        if columns != declared_columns:
            raise ValueError(
                f"view {name!r} columns differ from AgentInputManifest: "
                f"{columns!r} != {declared_columns!r}"
            )
        files.append(
            {
                "view": name,
                "role": view.role,
                "path": f"views/{name}.parquet",
                "sha256": file_sha256(path),
                "num_rows": len(table),
                "columns": columns,
            }
        )

    for old_path in views_dir.glob("*.parquet"):
        if old_path.resolve() not in expected_paths:
            old_path.unlink()

    manifest_data = bundle.manifest.model_dump(mode="json")
    write_json(public_dir / "dataset_manifest.json", manifest_data)
    expose_knowledge = bool(task_config(config).get("knowledge", False))
    knowledge_files = (
        _materialize_knowledge(task.dataset, public_dir)
        if expose_knowledge
        else []
    )
    if not expose_knowledge:
        _remove_materialized_knowledge(public_dir)
    file_index = {
        "schema_version": 1,
        "task_id": task_id,
        "benchmark_task_id": task.task_id,
        "files": files,
    }
    if expose_knowledge:
        file_index["knowledge"] = knowledge_files
    write_json(public_dir / "dataset_files.json", file_index)

    contract = _task_contract(task_id, config, task)
    write_json(public_dir / "task_contract.json", contract)
    validation_submission = _write_validation_submission(
        validation_dir,
        config=config,
        task=task,
        tables=tables,
    )
    _remove_legacy_adapter_artifacts(public_dir, validation_dir)
    result = {
        "task_id": task_id,
        "benchmark_task_id": task.task_id,
        "dataset_id": bundle.manifest.dataset_id,
        "dataset_version": bundle.manifest.dataset_version,
        "resolved_revision": bundle.manifest.resolved_revision,
        "protocol_id": bundle.manifest.protocol_id,
        "submission_size": task.submission_size,
        "primary_metric": task.primary_metric,
        "views": files,
        "public_dir": str(public_dir.resolve()),
        "validation_submission": (
            str(validation_submission.resolve()) if validation_submission else None
        ),
    }
    if expose_knowledge:
        result["knowledge"] = knowledge_files
    return result


def _preparation_identity(
    task_id: str,
    config: Mapping[str, Any],
    task: Any,
) -> dict[str, Any]:
    source = dict(source_config(config))
    source.pop("package", None)
    return {
        "schema_version": PUBLIC_INPUT_CACHE_SCHEMA_VERSION,
        "producer_version": PUBLIC_INPUT_PRODUCER_VERSION,
        "task_id": task_id,
        "benchmark_task_id": task.task_id,
        "resolved_revision": task.dataset.resolved_revision,
        "source": source,
        "artifact": dict(
            mapping(task_config(config).get("artifact"), name="task artifact")
        ),
        "submission": dict(submission_config(config)),
        "metric": dict(metric_config(config)),
        "knowledge": bool(task_config(config).get("knowledge", False)),
        "evaluator": {
            key: evaluator_config(config).get(key)
            for key in ("max_queries", "feedback_metrics")
        },
    }


def _reuse_prepared_output(
    output_dir: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    state_path = output_dir / "preparation_state.json"
    try:
        state = mapping(
            json.loads(state_path.read_text(encoding="utf-8")),
            name="public input preparation state",
        )
        if state.get("schema_version") != PUBLIC_INPUT_CACHE_SCHEMA_VERSION:
            return None
        if state.get("identity_sha256") != stable_json_sha(identity):
            return None
        files = state.get("files")
        result = state.get("result")
        if not isinstance(files, list) or not isinstance(result, Mapping):
            return None
        for item in files:
            if not isinstance(item, Mapping):
                return None
            relative = Path(str(item.get("path") or ""))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                return None
            path = output_dir / relative
            if not path.is_file():
                return None
            stat = path.stat()
            if (
                stat.st_size != item.get("size_bytes")
                or stat.st_mtime_ns != item.get("mtime_ns")
            ):
                return None
        return dict(result)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_preparation_state(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    paths = [
        output_dir / "public" / "dataset_manifest.json",
        output_dir / "public" / "dataset_files.json",
        output_dir / "public" / "task_contract.json",
    ]
    file_index = json.loads(paths[1].read_text(encoding="utf-8"))
    if "knowledge" in file_index:
        paths.append(output_dir / "public" / "knowledge_manifest.json")
    for item in file_index["files"]:
        paths.append(output_dir / "public" / str(item["path"]))
    for item in file_index.get("knowledge", []):
        paths.append(output_dir / "public" / str(item["path"]))
    validation = result.get("validation_submission")
    if validation:
        paths.append(Path(str(validation)))
    files = [
        {
            "path": str(path.resolve().relative_to(output_dir.resolve())),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    write_json(
        output_dir / "preparation_state.json",
        {
            "schema_version": PUBLIC_INPUT_CACHE_SCHEMA_VERSION,
            "identity": identity,
            "identity_sha256": stable_json_sha(identity),
            "files": files,
            "result": result,
        },
    )


def _serialize_reports(reports: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": report.artifact_id,
            "cache_enabled": report.cache_enabled,
            "cache_hit": report.cache_hit,
            "rebuilt": report.rebuilt,
            "path": str(report.path) if report.path is not None else None,
            "elapsed_sec": report.elapsed_sec,
        }
        for report in reports
    ]


def _materialize_knowledge(dataset: Any, public_dir: Path) -> list[dict[str, Any]]:
    knowledge_dir = public_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for name, resource in dataset.knowledge.items():
        path = knowledge_dir / f"{name}.md"
        expected_paths.add(path.resolve())
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(resource.read_text(), encoding="utf-8")
        temporary.replace(path)
        files.append(
            {
                "name": name,
                "title": resource.title,
                "description": resource.description,
                "path": f"knowledge/{name}.md",
                "media_type": resource.media_type,
                "sha256": file_sha256(path),
            }
        )
    for old_path in knowledge_dir.glob("*.md"):
        if old_path.resolve() not in expected_paths:
            old_path.unlink()
    write_json(
        public_dir / "knowledge_manifest.json",
        {"schema_version": 1, "resources": files},
    )
    return files


def _remove_materialized_knowledge(public_dir: Path) -> None:
    manifest_path = public_dir / "knowledge_manifest.json"
    if manifest_path.is_file():
        manifest_path.unlink()
    knowledge_dir = public_dir / "knowledge"
    if knowledge_dir.is_dir():
        for path in knowledge_dir.glob("*.md"):
            path.unlink()
        knowledge_dir.rmdir()


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _validate_contract(
    task_id: str,
    config: Mapping[str, Any],
    task: Any,
    bundle: Any,
) -> None:
    source = source_config(config)
    benchmark_task_id = str(source.get("benchmark_task_id") or "").strip()
    if task.task_id != benchmark_task_id:
        raise ValueError(
            f"official task_id mismatch for {task_id}: "
            f"{task.task_id!r} != {benchmark_task_id!r}"
        )
    if bundle.manifest.task_id != task.task_id:
        raise ValueError("Task.build_input() did not bind its task_id to the manifest")
    expected_revision = str(source.get("revision") or "").strip()
    if expected_revision and bundle.manifest.resolved_revision != expected_revision:
        raise ValueError(
            f"resolved Dataset revision mismatch: "
            f"{bundle.manifest.resolved_revision!r} != {expected_revision!r}"
        )
    expected_size = int(submission_config(config).get("size") or 0)
    if task.submission_size != expected_size:
        raise ValueError(
            f"submission size mismatch: {task.submission_size} != {expected_size}"
        )
    expected_summary_size = int(
        submission_config(config).get("summary_size") or 0
    )
    if expected_summary_size and task.summary_size != expected_summary_size:
        raise ValueError(
            f"summary size mismatch: {task.summary_size} != {expected_summary_size}"
        )
    expected_primary = str(metric_config(config).get("name") or "").strip()
    if task.primary_metric != expected_primary:
        raise ValueError(
            f"primary metric mismatch: {task.primary_metric!r} != {expected_primary!r}"
        )


def _guard_existing_output(public_dir: Path, task_id: str) -> None:
    contract_path = public_dir / "task_contract.json"
    if not contract_path.is_file():
        return
    existing = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(existing, Mapping) or existing.get("task_id") != task_id:
        raise ValueError(
            f"refusing to overwrite public input owned by another task: {public_dir}"
        )


def _task_contract(
    task_id: str,
    config: Mapping[str, Any],
    task: Any,
) -> dict[str, Any]:
    selected = task_config(config)
    artifact = dict(mapping(selected.get("artifact"), name="task artifact"))
    metric = dict(metric_config(config))
    evaluator = dict(mapping(selected.get("evaluator"), name="task evaluator"))
    submission = dict(submission_config(config))
    return {
        "schema_version": 1,
        "task_id": task_id,
        "benchmark_task_id": task.task_id,
        "artifact": artifact,
        "submission": submission,
        "metric": metric,
        "evaluator_feedback": {
            "max_queries": evaluator.get("max_queries"),
            "feedback_metrics": evaluator.get("feedback_metrics", []),
            "candidate_outputs_exposed": False,
        },
    }


def _write_validation_submission(
    validation_dir: Path,
    *,
    config: Mapping[str, Any],
    task: Any,
    tables: Mapping[str, Any],
) -> Path | None:
    submission = submission_config(config)
    view_name = str(submission.get("validation_view") or "").strip()
    if not view_name:
        return None
    table = tables.get(view_name)
    if table is None:
        raise ValueError(f"submission validation_view is not disclosed: {view_name!r}")
    raw_fields = submission.get("candidate_fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise ValueError("submission.candidate_fields must be a non-empty list")
    fields = tuple(str(field) for field in raw_fields)
    sort_field = str(submission.get("validation_sort_field") or "").strip()
    descending = bool(submission.get("validation_sort_descending", False))
    rows = list(table)
    if sort_field:
        rows.sort(key=lambda row: row[sort_field], reverse=descending)

    candidates: list[dict[str, Any]] = []
    identities: set[str] = set()
    for row in rows:
        candidate = {field: row[field] for field in fields}
        identity = canonical_candidates_sha([candidate])
        if identity in identities:
            continue
        identities.add(identity)
        candidates.append(candidate)
        if len(candidates) == task.submission_size:
            break
    if len(candidates) != task.submission_size:
        raise ValueError(
            f"validation view {view_name!r} yielded only {len(candidates)} "
            f"unique candidates; expected {task.submission_size}"
        )
    path = validation_dir / "validation_submission.json"
    write_json(path, {"candidates": candidates})
    return path


def _remove_legacy_adapter_artifacts(
    public_dir: Path,
    validation_dir: Path,
) -> None:
    """Remove files owned by the pre-0.6 TFBind8-only adapter."""

    for path in (
        public_dir / "offline_data.csv",
        validation_dir / "visible_top128_submission.json",
    ):
        if path.is_file():
            path.unlink()


def _default_output_dir(task_id: str) -> Path:
    slug = task_id.removeprefix("sci-modeling-bench-")
    return Path("cache") / "sci_modeling_bench" / slug


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir(args.task_id)
    print(json.dumps(prepare(args.task_id, output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
