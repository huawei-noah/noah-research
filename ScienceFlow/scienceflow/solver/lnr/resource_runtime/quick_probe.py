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
from typing import Any

from scienceflow.core.tools.resource_classifier import normalize_shell_command


_QUICK_TEXT_RE = re.compile(
    r"\b(quick\s+test|smoke\s+test|smoke\s+run|sanity\s+check|dry\s+run|probe|trial\s+run)\b",
    re.IGNORECASE,
)
_SMALL_SCOPE_RE = re.compile(
    r"\b(first|head|subset|sample|small\s+sample|limit|benchmark\s+first)\b",
    re.IGNORECASE,
)
_FEATURE_SMALL_RE = re.compile(
    r"\bextract(?:_features?|\s+features?)\b.{0,80}\b(?:for\s+)?(\d{1,6})\s+(?:products?|rows?|items?|images?|samples?)\b",
    re.IGNORECASE | re.DOTALL,
)
_FLAG_LIMIT_RE = re.compile(
    r"(?:--(?:limit|sample(?:[-_]size)?|num[-_]samples|n[-_]samples|rows|max[-_]items|head|first|subset)\s+|--(?:limit|sample(?:[-_]size)?|num[-_]samples|n[-_]samples|rows|max[-_]items|head|first|subset)=)(\d{1,6})",
    re.IGNORECASE,
)
_HEAD_RE = re.compile(r"\bhead\s+-n\s+(\d{1,6})\b", re.IGNORECASE)
_TRAIN_LONG_RE = re.compile(r"\b(?:epochs?|folds?|n_estimators|num_boost_round)\s*(?:=|\s+)\s*(\d{1,5})", re.IGNORECASE)
_PY_ENTRYPOINT_RE = re.compile(r"(?:^|[\s;&|])(?:python(?:3)?(?:\s+-u)?\s+)?([A-Za-z0-9_./-]+\.py)\b", re.IGNORECASE)
_SMALL_SCRIPT_HINT_RE = re.compile(
    r"\b(?:extract|decode|precache|cache|build).{0,40}\b(?:small|sample|subset|first|quick|probe|smoke|head|limit)\b|"
    r"\b(?:small|sample|subset|first|quick|probe|smoke|head|limit)\b.{0,40}\b(?:extract|decode|precache|cache|build)\b",
    re.IGNORECASE | re.DOTALL,
)
_COUNT_ASSIGN_RE = re.compile(
    r"\b(?:n|num|limit|max|sample_size|n_train|n_val|n_test|rows|products|samples|images)\w*\s*=\s*(\d{1,7})\b",
    re.IGNORECASE,
)
_EXTRACT_SCRIPT_RE = re.compile(r"\b(?:extract|decode|precache|cache|build)[A-Za-z0-9_-]*\.py\b", re.IGNORECASE)


@dataclass(frozen=True)
class QuickProbeFacts:
    candidate: bool
    reason: str = ""
    expected_runtime_sec: float = 300.0
    hard_review_sec: float = 600.0
    scope_count: int | None = None
    output_pattern: str = "unknown"
    confidence: str = "low"

    def to_json(self) -> dict[str, Any]:
        return {
            "quick_probe_candidate": self.candidate,
            "quick_probe_reason": self.reason,
            "quick_probe_expected_runtime_sec": self.expected_runtime_sec,
            "quick_probe_hard_review_sec": self.hard_review_sec,
            "quick_probe_scope_count": self.scope_count,
            "output_pattern": self.output_pattern,
            "confidence": self.confidence,
        }


def classify_quick_probe_command(
    command: str,
    *,
    expected_runtime_sec: float = 300.0,
    hard_review_sec: float = 600.0,
    small_scope_threshold: int = 1000,
    workspace_dir: str | Path | None = None,
) -> QuickProbeFacts:
    """Conservatively detect commands that describe themselves as short probes.

    Detection is mostly command-based, but when a workspace is available it also
    reads the small Python entrypoint. This catches commands like
    ``python3 extract_features.py`` where the short scope is declared inside the
    script instead of on the shell command line. Large bounded extraction scripts
    are intentionally not marked quick; they should be handled by the generic
    stalled-output review window.
    """

    raw = str(command or "")
    normalized = normalize_shell_command(raw)
    source_text = _read_entrypoint_source(raw, workspace_dir)
    text = f"{raw}\n{normalized}\n{source_text}"
    expected = max(30.0, float(expected_runtime_sec or 300.0))
    hard = max(expected, float(hard_review_sec or 600.0))
    small_limit = max(1, int(small_scope_threshold or 1000))

    quick_match = _QUICK_TEXT_RE.search(text)
    feature_match = _FEATURE_SMALL_RE.search(text)
    limit_count = _first_small_count(text, small_limit)
    source_count = _first_source_count(source_text, small_limit)
    scope_language = bool(_SMALL_SCOPE_RE.search(text))
    small_script_hint = bool(source_text and _SMALL_SCRIPT_HINT_RE.search(source_text))
    extract_script = bool(_EXTRACT_SCRIPT_RE.search(raw))
    long_train = _looks_like_full_train(text)

    if feature_match:
        count = _safe_int(feature_match.group(1))
        if count is not None and count <= small_limit:
            return QuickProbeFacts(
                True,
                reason="small_feature_extract",
                expected_runtime_sec=expected,
                hard_review_sec=hard,
                scope_count=count,
                output_pattern="terminal_write_expected",
                confidence="high" if quick_match else "medium",
            )

    if source_count is not None and (small_script_hint or extract_script or quick_match) and not long_train:
        return QuickProbeFacts(
            True,
            reason="small_scope_entrypoint",
            expected_runtime_sec=expected,
            hard_review_sec=hard,
            scope_count=source_count,
            output_pattern="terminal_write_expected" if extract_script else "heartbeat_expected",
            confidence="medium" if not quick_match else "high",
        )

    if quick_match and limit_count is not None and limit_count <= small_limit:
        return QuickProbeFacts(
            True,
            reason="explicit_quick_small_scope",
            expected_runtime_sec=expected,
            hard_review_sec=hard,
            scope_count=limit_count,
            output_pattern="heartbeat_expected" if not long_train else "unknown",
            confidence="high",
        )

    if quick_match and not long_train:
        return QuickProbeFacts(
            True,
            reason="explicit_quick_probe",
            expected_runtime_sec=expected,
            hard_review_sec=hard,
            scope_count=limit_count,
            output_pattern="heartbeat_expected",
            confidence="medium",
        )

    if scope_language and limit_count is not None and limit_count <= small_limit and not long_train:
        return QuickProbeFacts(
            True,
            reason="small_scope_probe_language",
            expected_runtime_sec=expected,
            hard_review_sec=hard,
            scope_count=limit_count,
            output_pattern="unknown",
            confidence="medium",
        )

    return QuickProbeFacts(False, expected_runtime_sec=expected, hard_review_sec=hard)


def _read_entrypoint_source(command: str, workspace_dir: str | Path | None) -> str:
    if workspace_dir is None:
        return ""
    workspace = Path(workspace_dir)
    for match in _PY_ENTRYPOINT_RE.finditer(str(command or "")):
        rel = match.group(1)
        if rel.startswith("-"):
            continue
        path = (workspace / rel).resolve(strict=False)
        try:
            path.relative_to(workspace.resolve(strict=False))
        except ValueError:
            continue
        if path.is_file() and path.suffix == ".py":
            try:
                return path.read_text(encoding="utf-8", errors="replace")[:12000]
            except OSError:
                return ""
    return ""


def _first_small_count(text: str, threshold: int) -> int | None:
    counts: list[int] = []
    for pattern in (_FLAG_LIMIT_RE, _HEAD_RE):
        for match in pattern.finditer(text):
            value = _safe_int(match.group(1))
            if value is not None:
                counts.append(value)
    for match in re.finditer(r"\b(?:first|head|subset|sample)\s+(\d{1,6})\b", text, flags=re.IGNORECASE):
        value = _safe_int(match.group(1))
        if value is not None:
            counts.append(value)
    small = [value for value in counts if value <= threshold]
    if small:
        return min(small)
    return min(counts) if counts else None


def _first_source_count(text: str, threshold: int) -> int | None:
    counts: list[int] = []
    for match in _COUNT_ASSIGN_RE.finditer(text or ""):
        value = _safe_int(match.group(1))
        if value is not None:
            counts.append(value)
    small = [value for value in counts if value <= threshold]
    return min(small) if small else None


def _looks_like_full_train(text: str) -> bool:
    lowered = text.lower()
    if "quick" in lowered or "smoke" in lowered or "probe" in lowered:
        return False
    for match in _TRAIN_LONG_RE.finditer(text):
        value = _safe_int(match.group(1))
        if value is not None and value >= 3:
            return True
    return bool(re.search(r"\b(torchrun|deepspeed|accelerate)\b", text, flags=re.IGNORECASE))


def _safe_int(raw: Any) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None
