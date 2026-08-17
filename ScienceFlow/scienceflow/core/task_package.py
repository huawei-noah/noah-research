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

"""Task package discovery and runtime registration.

This module is intentionally task-domain agnostic. It knows how to find a
``task.yaml``, copy the package into a system-only runtime directory, and
resolve evaluator entrypoints. It does not know about MLE-bench, opt-solver, or
any task-specific metric rules.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TaskPackageSpec:
    task_id: str
    category: str
    provider: str
    profile: str
    source_dir: Path
    config: Mapping[str, Any]

    @property
    def description_relpath(self) -> str:
        return _clean_relpath(_cfg(self.config, "description", "description_lite.md"), "description_lite.md")

    @property
    def artifact_path(self) -> str:
        artifact = _mapping(self.config.get("artifact"))
        return _clean_relpath(artifact.get("path"), "submission.csv")

    @property
    def artifact_kind(self) -> str:
        artifact = _mapping(self.config.get("artifact"))
        return str(artifact.get("kind") or "artifact").strip()

    @property
    def metric_name(self) -> str:
        metric = _mapping(self.config.get("metric"))
        return str(metric.get("name") or "metric").strip()

    @property
    def metric_type(self) -> str:
        metric = _mapping(self.config.get("metric"))
        return str(metric.get("type") or "benchmark").strip()

    @property
    def metric_authoritative(self) -> bool:
        metric = _mapping(self.config.get("metric"))
        return bool(_as_bool(metric.get("authoritative")))

    @property
    def lower_is_better(self) -> bool | None:
        metric = _mapping(self.config.get("metric"))
        return _as_bool(metric.get("lower_is_better"))

    @property
    def evaluator_entrypoint(self) -> str:
        evaluator = _mapping(self.config.get("evaluator"))
        return str(evaluator.get("entrypoint") or "").strip()

    @property
    def evaluator_timeout_sec(self) -> float:
        evaluator = _mapping(self.config.get("evaluator"))
        try:
            return max(1.0, float(evaluator.get("timeout_sec") or 300.0))
        except (TypeError, ValueError):
            return 300.0


@dataclass(frozen=True)
class TaskRuntimeRegistration:
    spec: TaskPackageSpec
    runtime_root: Path
    runtime_task_dir: Path
    entrypoint_path: Path
    entrypoint_function: str
    entrypoint_sha256: str
    package_sha256: str
    manifest_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tasks_root() -> Path:
    return repo_root() / "tasks"


def find_task_package(task_id: str, *, tasks_root: Path | None = None) -> TaskPackageSpec | None:
    task_id = str(task_id or "").strip()
    if not task_id:
        return None
    matches = [spec for spec in iter_task_packages(tasks_root=tasks_root) if spec.task_id == task_id]
    if len(matches) > 1:
        paths = ", ".join(str(spec.source_dir) for spec in matches)
        raise ValueError(f"ambiguous task package for {task_id!r}: {paths}")
    return matches[0] if matches else None


def iter_task_packages(*, tasks_root: Path | None = None) -> list[TaskPackageSpec]:
    root = (tasks_root or default_tasks_root()).resolve()
    if not root.is_dir():
        return []
    specs: list[TaskPackageSpec] = []
    for path in sorted(root.rglob("task.yaml")):
        if any(part.startswith("_") for part in path.relative_to(root).parts):
            continue
        data = _load_task_package_config(path)
        if not isinstance(data, Mapping):
            continue
        task_id = str(data.get("id") or path.parent.name).strip()
        if not task_id:
            continue
        rel_parts = path.parent.relative_to(root).parts
        category = str(data.get("category") or (rel_parts[0] if rel_parts else "")).strip()
        provider = str(data.get("provider") or "").strip()
        profile = str(data.get("profile") or provider or category or "default").strip()
        specs.append(
            TaskPackageSpec(
                task_id=task_id,
                category=category,
                provider=provider,
                profile=profile,
                source_dir=path.parent.resolve(),
                config=dict(data),
            )
        )
    return specs


def _load_task_package_config(path: Path) -> dict[str, Any]:
    """Merge optional provider-family defaults with a task manifest."""
    task_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(task_data, Mapping):
        return {}
    defaults_path = path.parent.parent / "_shared" / "task_defaults.yaml"
    defaults_data: Mapping[str, Any] = {}
    if defaults_path.is_file():
        loaded = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, Mapping):
            defaults_data = loaded
    return _merge_nested_mappings(defaults_data, task_data)


def _merge_nested_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_nested_mappings(current, value)
        else:
            merged[key] = value
    return merged


def description_path_for_task(task_id: str, *, tasks_root: Path | None = None) -> Path | None:
    spec = find_task_package(task_id, tasks_root=tasks_root)
    if spec is None:
        return None
    path = spec.source_dir / spec.description_relpath
    return path if path.is_file() else None


def prepare_task_runtime(
    task_id: str,
    *,
    task_root: Path,
    tasks_root: Path | None = None,
) -> TaskRuntimeRegistration:
    spec = find_task_package(task_id, tasks_root=tasks_root)
    if spec is None:
        raise FileNotFoundError(f"task package not found for {task_id!r}")
    if not spec.evaluator_entrypoint:
        raise ValueError(f"task package {task_id!r} has no evaluator.entrypoint")
    source_root = _runtime_copy_source(spec)
    root = (tasks_root or default_tasks_root()).resolve()
    runtime_root = Path(task_root).resolve() / "task_runtime"
    dest_root = runtime_root / source_root.relative_to(root)
    _copy_clean(source_root, dest_root)
    runtime_task_dir = runtime_root / spec.source_dir.relative_to(root)
    entrypoint_path, entrypoint_fn = _resolve_entrypoint(runtime_task_dir, runtime_root, spec.evaluator_entrypoint)
    if not entrypoint_path.is_file():
        raise FileNotFoundError(f"evaluator entrypoint not found: {entrypoint_path}")
    manifest = {
        "task_id": spec.task_id,
        "category": spec.category,
        "provider": spec.provider,
        "profile": spec.profile,
        "source_dir": str(spec.source_dir),
        "runtime_task_dir": str(runtime_task_dir),
        "entrypoint": str(entrypoint_path),
        "entrypoint_function": entrypoint_fn,
        "entrypoint_sha256": file_sha256(entrypoint_path),
        "package_sha256": tree_sha256(dest_root),
    }
    manifest_path = runtime_root / "registry.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TaskRuntimeRegistration(
        spec=spec,
        runtime_root=runtime_root,
        runtime_task_dir=runtime_task_dir,
        entrypoint_path=entrypoint_path,
        entrypoint_function=entrypoint_fn,
        entrypoint_sha256=str(manifest["entrypoint_sha256"]),
        package_sha256=str(manifest["package_sha256"]),
        manifest_path=manifest_path,
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in child.parts:
            continue
        rel = child.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(file_sha256(child).encode("ascii"))
    return h.hexdigest()


def _runtime_copy_source(spec: TaskPackageSpec) -> Path:
    parent = spec.source_dir.parent
    if (parent / "_shared").is_dir():
        return parent
    return spec.source_dir


def _copy_clean(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _resolve_entrypoint(task_dir: Path, runtime_root: Path, entrypoint: str) -> tuple[Path, str]:
    if ":" not in entrypoint:
        raise ValueError(f"evaluator entrypoint must be '<path>:<function>', got {entrypoint!r}")
    raw_path, raw_fn = entrypoint.rsplit(":", 1)
    fn = raw_fn.strip()
    if not fn:
        raise ValueError(f"evaluator entrypoint function is empty: {entrypoint!r}")
    text = raw_path.replace("\\", "/").strip().lstrip("/")
    if not text:
        raise ValueError(f"evaluator entrypoint path is empty: {entrypoint!r}")
    resolved = (task_dir / text).resolve()
    try:
        resolved.relative_to(runtime_root.resolve())
    except ValueError as exc:
        raise ValueError(f"evaluator entrypoint escapes task runtime: {entrypoint!r}") from exc
    return resolved, fn


def _clean_relpath(value: Any, default: str) -> str:
    text = str(value or default).replace("\\", "/").strip().lstrip("/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return default
    return "/".join(parts)


def _cfg(data: Mapping[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "lower"}:
        return True
    if text in {"0", "false", "no", "n", "off", "higher"}:
        return False
    return None
