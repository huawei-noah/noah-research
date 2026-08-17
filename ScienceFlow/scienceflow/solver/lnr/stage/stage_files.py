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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_CODE_SUFFIXES = {".py"}
_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".joblib",
    ".keras",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
_EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".logs",
    ".memory",
    ".venv",
    "__pycache__",
    "artifact",
    "artifacts",
    "cache",
    "data",
    "dataset",
    "dist",
    "input",
    "inputs",
    "log",
    "logs",
    "output",
    "outputs",
    "prediction",
    "predictions",
    "result",
    "results",
    "submission",
    "submissions",
    "temp",
    "tmp",
    "venv",
    "wandb",
}
_OUTPUT_STEMS = {
    "best_solution",
    "final_solution",
    "predictions",
    "result",
    "results",
    "score",
    "scores",
    "submission",
}
_NONE_VALUES = {"", "-", "none", "n/a", "na", "null", "unknown"}


@dataclass(frozen=True)
class StageFiles:
    code: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()

    def format(self) -> str:
        parts: list[str] = []
        if self.code:
            parts.append("code=" + ",".join(self.code))
        if self.weights:
            parts.append("weights=" + ",".join(self.weights))
        return " ".join(parts)


def format_stage_files(
    value: Any = None,
    *,
    metric_event: dict[str, Any] | None = None,
    workspace_dir: Path | str | None = None,
    max_code: int = 5,
    max_weights: int = 3,
) -> str:
    return normalize_stage_files(
        value,
        metric_event=metric_event,
        workspace_dir=workspace_dir,
        max_code=max_code,
        max_weights=max_weights,
    ).format()


def normalize_stage_files(
    value: Any = None,
    *,
    metric_event: dict[str, Any] | None = None,
    workspace_dir: Path | str | None = None,
    max_code: int = 5,
    max_weights: int = 3,
) -> StageFiles:
    root = Path(workspace_dir).resolve() if workspace_dir else None
    code_raw: list[str] = []
    weights_raw: list[str] = []
    unknown_raw: list[str] = []

    _collect_value(value, code_raw=code_raw, weights_raw=weights_raw, unknown_raw=unknown_raw)
    event = metric_event if isinstance(metric_event, dict) else {}
    for key in (
        "solution_path",
        "source_path",
        "script_path",
        "entrypoint",
        "train_script",
        "predict_script",
        "inference_script",
    ):
        if event.get(key):
            unknown_raw.extend(_split_paths(event.get(key)))
    for key in (
        "model_path",
        "checkpoint_path",
        "checkpoints",
        "weights_path",
        "weight_path",
        "best_model_path",
    ):
        if event.get(key):
            weights_raw.extend(_split_paths(event.get(key)))

    code = _dedupe(
        rel
        for raw in [*code_raw, *unknown_raw]
        for rel in [_normalize_path(raw, root=root)]
        if rel and _classify_path(rel) == "code"
    )[:max_code]
    weights = _dedupe(
        rel
        for raw in [*weights_raw, *unknown_raw]
        for rel in [_normalize_path(raw, root=root)]
        if rel and _classify_path(rel) == "weights"
    )[:max_weights]
    return StageFiles(code=tuple(code), weights=tuple(weights))


def _collect_value(value: Any, *, code_raw: list[str], weights_raw: list[str], unknown_raw: list[str]) -> None:
    if value in (None, ""):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key or "").strip().lower()
            if normalized in {"code", "code_files", "scripts", "source", "source_files"}:
                code_raw.extend(_split_paths(item))
            elif normalized in {"weight", "weights", "weight_files", "model", "models", "checkpoints"}:
                weights_raw.extend(_split_paths(item))
            elif normalized in {"files", "stage_files"}:
                unknown_raw.extend(_split_paths(item))
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_value(item, code_raw=code_raw, weights_raw=weights_raw, unknown_raw=unknown_raw)
        return

    text = str(value or "").strip()
    if text.lower() in _NONE_VALUES:
        return
    sectioned = False
    for key, bucket in (("code", code_raw), ("weights", weights_raw), ("weight", weights_raw)):
        pattern = rf"(?i)\b{key}\s*=\s*([^;]+?)(?=\s+\w+\s*=|$)"
        for match in re.finditer(pattern, text):
            bucket.extend(_split_paths(match.group(1)))
            sectioned = True
    if not sectioned:
        unknown_raw.extend(_split_paths(text))


def _split_paths(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_split_paths(item))
        return out
    text = str(value or "")
    text = text.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")
    return [
        part.strip().strip("'\"`")
        for part in re.split(r"[,;\s]+", text)
        if part.strip().strip("'\"`").lower() not in _NONE_VALUES
    ]


def _normalize_path(raw: str, *, root: Path | None) -> str:
    text = str(raw or "").strip().strip("'\"`")
    if not text or "://" in text or text.startswith("~"):
        return ""
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = Path(text)
    if path.is_absolute():
        if root is None:
            return ""
        try:
            path = path.resolve().relative_to(root)
        except (OSError, ValueError):
            return ""
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    lowered = [part.lower() for part in parts]
    if any(part in _EXCLUDED_PARTS for part in lowered):
        return ""
    rel = Path(*parts).as_posix()
    if root is not None and not (root / rel).exists():
        return ""
    return rel


def _classify_path(rel: str) -> str:
    path = Path(rel)
    suffix = path.suffix.lower()
    stem = path.stem.lower()
    if suffix in _CODE_SUFFIXES:
        return "code"
    if suffix in _WEIGHT_SUFFIXES and stem not in _OUTPUT_STEMS:
        return "weights"
    return ""


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
