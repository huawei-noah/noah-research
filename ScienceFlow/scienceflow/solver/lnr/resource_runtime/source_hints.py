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
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from scienceflow.core.tools.resource_classifier import (
    RESOURCE_READONLY_CPU,
    classify_bash_command,
    command_has_cpu_only_cuda,
    normalize_shell_command,
)


_NPROC_RE = re.compile(r"--nproc[-_]per[-_]node(?:=|\s+)(\d+)", re.IGNORECASE)
_ACCELERATE_PROC_RE = re.compile(r"--num[-_]processes(?:=|\s+)(\d+)", re.IGNORECASE)
_DEEPSPEED_GPU_RE = re.compile(r"--num[-_]gpus(?:=|\s+)(\d+)", re.IGNORECASE)
_CUDA_ENV_RE = re.compile(r"(?:^|\s)CUDA_VISIBLE_DEVICES=([^\s;&|]+)")
_TRAIN_HINT_RE = re.compile(
    r"\b(train|training|trainer|fit|epoch|epochs|fold|folds|backward|optimizer)\b|"
    r"loss\.backward|model\.train\s*\(",
    re.IGNORECASE,
)
_TT_HINT_RE = re.compile(
    r"\b(predict|prediction|predicting|infer|inference|tta|ensemble|submission|evaluate|eval)\b|"
    r"model\.eval\s*\(|torch\.no_grad|inference_mode",
    re.IGNORECASE,
)
_FEATURE_HINT_RE = re.compile(
    r"\b(extract(?:_features?)?|feature(?:s)?|embedding(?:s)?|embed(?:s|ding)?|encode(?:_images?)?)\b",
    re.IGNORECASE,
)
_GPU_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("torch_cuda", re.compile(r"torch\.cuda|torch\.device\(\s*['\"]cuda|\.cuda\(|to\(\s*['\"]cuda|device\s*=\s*['\"]cuda", re.IGNORECASE)),
    ("ddp", re.compile(r"DistributedDataParallel|torch\.distributed|init_process_group", re.IGNORECASE)),
    ("data_parallel", re.compile(r"\bDataParallel\s*\(", re.IGNORECASE)),
    ("xgboost_gpu", re.compile(r"tree_method\s*=\s*['\"]gpu_hist|device\s*=\s*['\"]cuda", re.IGNORECASE)),
    ("lightgbm_gpu", re.compile(r"device_type\s*=\s*['\"]gpu|device\s*=\s*['\"]gpu", re.IGNORECASE)),
)
_GPU_COMMAND_COMPUTE_RE = re.compile(
    r"\.cuda\s*\(|\.to\(\s*['\"]cuda|device\s*=\s*['\"]cuda(?::\d+)?|"
    r"cuda(?::\d+)?\s*['\"]|\.to\(\s*device\s*\)",
    re.IGNORECASE,
)
_SKIP_PARTS = {".git", ".venv", "dataset", "input", "artifacts", ".logs", "logs", "__pycache__"}
_WRITE_HEREDOC_RE = re.compile(r"^\s*cat\s*(?:>|>>|<<)", re.IGNORECASE)


@dataclass(frozen=True)
class ResourceSourceHint:
    command_gpu_evidence: bool = False
    command_gpu_compute_evidence: bool = False
    command_cpu_only: bool = False
    command_gpu_request: int | None = None
    entrypoints: list[str] = field(default_factory=list)
    source_gpu_evidence: bool = False
    source_gpu_request: int | None = None
    source_files_inspected: int = 0
    hint_labels: list[str] = field(default_factory=list)
    command_train_evidence: bool = False
    command_tt_evidence: bool = False
    command_feature_evidence: bool = False
    source_train_evidence: bool = False
    source_tt_evidence: bool = False
    source_feature_evidence: bool = False

    @property
    def has_gpu_evidence(self) -> bool:
        if self.command_cpu_only:
            return False
        return bool(self.command_gpu_evidence or self.source_gpu_evidence)

    @property
    def has_train_evidence(self) -> bool:
        return bool(self.command_train_evidence or self.source_train_evidence)

    @property
    def has_tt_evidence(self) -> bool:
        return bool(self.command_tt_evidence or self.source_tt_evidence)

    @property
    def has_feature_evidence(self) -> bool:
        return bool(self.command_feature_evidence or self.source_feature_evidence)

    @property
    def requested_gpu_count(self) -> int | None:
        values = [v for v in [self.command_gpu_request, self.source_gpu_request] if v and v > 0]
        return max(values) if values else None

    def to_event_payload(self) -> dict[str, object]:
        return {
            "command_gpu_evidence": self.command_gpu_evidence,
            "command_gpu_compute_evidence": self.command_gpu_compute_evidence,
            "command_cpu_only": self.command_cpu_only,
            "command_gpu_request": self.command_gpu_request,
            "entrypoints": list(self.entrypoints),
            "source_gpu_evidence": self.source_gpu_evidence,
            "source_gpu_request": self.source_gpu_request,
            "source_files_inspected": self.source_files_inspected,
            "hint_labels": list(self.hint_labels),
            "requested_gpu_count": self.requested_gpu_count,
            "command_train_evidence": self.command_train_evidence,
            "command_tt_evidence": self.command_tt_evidence,
            "command_feature_evidence": self.command_feature_evidence,
            "source_train_evidence": self.source_train_evidence,
            "source_tt_evidence": self.source_tt_evidence,
            "source_feature_evidence": self.source_feature_evidence,
        }


def detect_resource_source_hint(
    *,
    command: str,
    workspace_dir: Path | None = None,
    max_source_bytes: int = 64_000,
) -> ResourceSourceHint:
    raw_command = str(command or "")
    normalized = normalize_shell_command(raw_command)
    if _WRITE_HEREDOC_RE.search(normalized):
        return ResourceSourceHint()
    labels: list[str] = []
    command_cpu_only = command_has_cpu_only_cuda(raw_command)
    command_gpu_evidence = False
    request: int | None = None
    if command_cpu_only:
        labels.append("command_cpu_only_cuda")
    for label, pattern in [
        ("torchrun_nproc", _NPROC_RE),
        ("accelerate_num_processes", _ACCELERATE_PROC_RE),
        ("deepspeed_num_gpus", _DEEPSPEED_GPU_RE),
    ]:
        match = pattern.search(normalized)
        if match:
            command_gpu_evidence = True
            labels.append(label)
            request = max(request or 0, _safe_int(match.group(1)) or 1)
    if re.search(r"\b(torchrun|accelerate|deepspeed)\b", normalized, flags=re.IGNORECASE):
        command_gpu_evidence = True
        labels.append("gpu_launcher")
        request = max(request or 0, 1)
    cuda_env = _CUDA_ENV_RE.search(raw_command) or _CUDA_ENV_RE.search(normalized)
    if cuda_env:
        ids = _split_cuda_visible(cuda_env.group(1))
        if ids:
            command_gpu_evidence = True
            labels.append("cuda_visible_devices_inline")
            request = max(request or 0, len(ids))
    command_gpu_compute = bool(_GPU_COMMAND_COMPUTE_RE.search(normalized))
    if command_gpu_compute:
        if not command_cpu_only:
            command_gpu_evidence = True
        labels.append("command_cuda_compute")

    command_train = bool(_TRAIN_HINT_RE.search(normalized))
    command_tt = bool(_TT_HINT_RE.search(normalized))
    command_feature = bool(_FEATURE_HINT_RE.search(normalized))
    if command_train:
        labels.append("command_train")
    if command_tt:
        labels.append("command_tt")
    if command_feature:
        labels.append("command_feature_extract")

    command_class = classify_bash_command(normalized).resource_class
    entrypoints = (
        [] if command_class == RESOURCE_READONLY_CPU else _entrypoints_from_command(normalized)
    )
    source_gpu_evidence = False
    source_request: int | None = None
    source_train = False
    source_tt = False
    source_feature = False
    source_files_inspected = 0
    if workspace_dir is not None:
        root = Path(workspace_dir).resolve()
        for rel in entrypoints[:5]:
            path = (root / rel).resolve()
            if not _is_allowed_workspace_file(path, root):
                continue
            text = _read_prefix(path, max_source_bytes)
            if not text:
                continue
            source_files_inspected += 1
            for label, pattern in _GPU_SOURCE_PATTERNS:
                if pattern.search(text):
                    source_gpu_evidence = True
                    labels.append(label)
            if re.search(r"DistributedDataParallel|torch\.distributed|DataParallel\s*\(", text, re.IGNORECASE):
                source_request = max(source_request or 0, 2)
            if _TRAIN_HINT_RE.search(text):
                source_train = True
                labels.append("source_train")
            if _TT_HINT_RE.search(text):
                source_tt = True
                labels.append("source_tt")
            if _FEATURE_HINT_RE.search(text):
                source_feature = True
                labels.append("source_feature_extract")

    clean_labels = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            clean_labels.append(label)
    return ResourceSourceHint(
        command_gpu_evidence=command_gpu_evidence,
        command_gpu_compute_evidence=command_gpu_compute,
        command_cpu_only=command_cpu_only,
        command_gpu_request=request,
        entrypoints=entrypoints,
        source_gpu_evidence=source_gpu_evidence,
        source_gpu_request=source_request,
        source_files_inspected=source_files_inspected,
        hint_labels=clean_labels,
        command_train_evidence=command_train,
        command_tt_evidence=command_tt,
        command_feature_evidence=command_feature,
        source_train_evidence=source_train,
        source_tt_evidence=source_tt,
        source_feature_evidence=source_feature,
    )


def _safe_int(raw: str) -> int | None:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _split_cuda_visible(raw: str) -> list[str]:
    value = str(raw or "").strip().strip("'\"")
    if not value or value.lower() in {"-1", "none", "cpu", "nodevfile"}:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _entrypoints_from_command(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    out: list[str] = []

    def add(raw: str) -> None:
        rel = str(raw or "").replace("\\", "/").lstrip("./")
        if rel and rel.endswith(".py") and not rel.startswith("/") and ".." not in Path(rel).parts:
            out.append(rel)

    for token in tokens:
        if token.endswith(".py") and not token.startswith("-"):
            add(token)
    for match in re.finditer(r"(?:open|Path)\(\s*['\"]?([^'\"\)\s]+\.py)['\"]?", str(command or "")):
        add(match.group(1))
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def _is_allowed_workspace_file(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if any(part in _SKIP_PARTS for part in rel.parts):
        return False
    return path.is_file() and path.suffix == ".py"


def _read_prefix(path: Path, max_bytes: int) -> str:
    try:
        data = path.read_bytes()[: max(0, int(max_bytes or 0))]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")
