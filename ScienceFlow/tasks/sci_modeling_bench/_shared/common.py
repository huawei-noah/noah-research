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

"""Shared configuration and serialization helpers for SciModelingBench tasks."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


PACKAGE_NAME = "sci-modeling-bench"


def task_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the task-package mapping passed by ScienceFlow or loaded from YAML."""

    nested = config.get("task")
    return nested if isinstance(nested, Mapping) else config


def evaluation_context(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return trusted per-evaluation context supplied by ScienceFlow."""

    nested = config.get("evaluation_context")
    return nested if isinstance(nested, Mapping) else {}


def mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def source_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return mapping(task_config(config).get("source"), name="task source")


def submission_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return mapping(task_config(config).get("submission"), name="task submission")


def evaluator_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return mapping(task_config(config).get("evaluator"), name="task evaluator")


def metric_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return mapping(task_config(config).get("metric"), name="task metric")


def expected_package_version(config: Mapping[str, Any]) -> str:
    requirement = str(source_config(config).get("package") or "").strip()
    match = re.fullmatch(r"sci-modeling-bench==([^\s]+)", requirement)
    if match is None:
        raise ValueError(
            "task source.package must pin sci-modeling-bench with =="
        )
    return match.group(1)


def installed_package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(f"{PACKAGE_NAME} is not installed") from exc


def verify_package_version(config: Mapping[str, Any]) -> str:
    expected = expected_package_version(config)
    installed = installed_package_version()
    if installed != expected:
        raise ImportError(
            f"expected {PACKAGE_NAME}=={expected}, found {installed}"
        )
    return installed


def load_official_task(config: Mapping[str, Any]) -> Any:
    """Instantiate a pinned public Task through its declared ``from_hub`` factory."""

    verify_package_version(config)
    source = source_config(config)
    factory_path = str(source.get("task_factory") or "").strip()
    if ":" not in factory_path:
        raise ValueError(
            "task source.task_factory must be '<module>:<TaskClass>'"
        )
    module_name, attribute = factory_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    from_hub = getattr(factory, "from_hub", None)
    if not callable(from_hub):
        raise ValueError(f"official Task factory has no from_hub(): {factory_path}")
    raw_kwargs = source.get("task_kwargs", {})
    kwargs = dict(mapping(raw_kwargs, name="task source.task_kwargs"))
    return from_hub(**kwargs)


def resolve_bundle_views(bundle: Any) -> dict[str, Any]:
    """Map manifest view names to the concrete Hugging Face Dataset tables."""

    manifest = bundle.manifest
    views = tuple(manifest.views)
    data = bundle.data
    if len(views) == 1 and callable(getattr(data, "to_parquet", None)):
        return {views[0].name: data}

    resolved: dict[str, Any] = {}
    for view in views:
        table = getattr(data, view.name, None)
        if table is None or not callable(getattr(table, "to_parquet", None)):
            raise ValueError(
                f"AgentInputBundle data has no serializable view {view.name!r}"
            )
        resolved[view.name] = table
    return resolved


def canonical_candidates_sha(candidates: list[Any]) -> str:
    payload = json.dumps(
        candidates,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_task_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    if not cleaned:
        raise ValueError("task id has no filesystem-safe characters")
    return cleaned


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
