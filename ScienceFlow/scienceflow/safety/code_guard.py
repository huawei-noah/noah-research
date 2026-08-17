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

"""Pre-execution code guard — catches issues BEFORE expensive training runs.

Consolidates checks that may also run post-execution in ``_evaluate_node``, plus
static-only checks (hardcoded paths, device hints, submission save heuristics).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scienceflow.safety.leakage_detector import validate_no_leakage
from scienceflow.safety.stdout_checker import check_stdout_hostile, strip_tqdm

# ---------------------------------------------------------------------------
# Hardcoded / environment-specific paths
# ---------------------------------------------------------------------------

_BAD_PATH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/kaggle/(?:input|working)", re.IGNORECASE), "Kaggle-specific path"),
    (re.compile(r"/mnt/(?:data|pfs)", re.IGNORECASE), "mount path from another environment"),
    (re.compile(r"/root/workspace", re.IGNORECASE), "absolute /root/workspace"),
    (re.compile(r"/home/[^/\s\"']+/", re.IGNORECASE), "hard-coded home directory path"),
]


def _check_hardcoded_paths(code: str) -> list[str]:
    issues: list[str] = []
    for pat, label in _BAD_PATH_PATTERNS:
        if pat.search(code):
            issues.append(f"[PATH] {label} detected in source")
    return issues


def _check_import_issues(code: str) -> list[str]:
    issues: list[str] = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"[SYNTAX] {e.msg} at line {e.lineno}")
    return issues


def _has_submission_save(code: str) -> bool:
    return bool(
        re.search(
            r"\.to_csv\s*\(|\.savetxt\s*\(|submission|\.to_parquet\s*\(",
            code,
            re.IGNORECASE,
        )
    )


_RE_OPEN_RESULT_MD_WRITE = re.compile(
    r"open\s*\(\s*['\"]result\.md['\"]\s*,\s*['\"][wax+]",
    re.IGNORECASE,
)
_RE_PATH_RESULT_MD_WRITE = re.compile(
    r"(?:Path|pathlib\.Path)\s*\(\s*['\"]result\.md['\"]\s*\)\s*\.\s*write",
    re.IGNORECASE,
)


def _check_writes_result_md(code: str) -> list[str]:
    """Detect writing ``result.md`` from solution code (system-generated, not agent-written)."""
    if not (code or "").strip() or "result.md" not in code:
        return []
    issues: list[str] = []
    _msg = (
        "result.md is generated automatically by the system — do not write it from solution.py"
    )
    if _RE_OPEN_RESULT_MD_WRITE.search(code):
        issues.append(
            f"[RESULT_MD_IN_CODE] open(result.md) with write/append mode — {_msg}",
        )
    if _RE_PATH_RESULT_MD_WRITE.search(code):
        issues.append(
            f"[RESULT_MD_IN_CODE] Path('result.md').write_* — {_msg}",
        )
    # Broad fallback: open(result.md) with non-default mode via variable or kwargs
    if re.search(
        r"open\s*\(\s*['\"]result\.md['\"]\s*,\s*mode\s*=\s*['\"][^'\"r]",
        code,
        re.IGNORECASE,
    ):
        issues.append(
            f"[RESULT_MD_IN_CODE] open(result.md, mode=...) not read-only — {_msg}",
        )
    return issues


def _check_device_mismatch(code: str, gpu_available: bool) -> list[str]:
    if gpu_available:
        return []
    has_cuda = bool(re.search(r"\.cuda\(\)|\.to\(\s*['\"]cuda", code))
    has_fallback = bool(
        re.search(
            r"torch\.cuda\.is_available\s*\(\)|device\s*=.*cpu|['\"]cpu['\"]",
            code,
            re.IGNORECASE,
        )
    )
    if has_cuda and not has_fallback:
        return [
            "[DEVICE] Code uses CUDA APIs but GPU unavailable and no CPU fallback detected",
        ]
    return []


_RE_TEST_LABEL_FILES = re.compile(
    r"(?:test_labels|test_label|answers\.csv|solution\.csv|sample_submission)",
    re.IGNORECASE,
)
_RE_READ_TEST_AS_TARGET = re.compile(
    r"(?:read_csv|pd\.read_csv)\s*\([^)]*(?:test.*label|label.*test|ground.*truth.*test)",
    re.IGNORECASE | re.DOTALL,
)


def deep_leakage_check(code: str) -> list[str]:
    """Stricter leakage heuristics for extreme-metric re-check."""
    base = list(validate_no_leakage(code))
    if not (code or "").strip():
        return base
    # Extra patterns: loading test-side labels as supervision
    if _RE_READ_TEST_AS_TARGET.search(code):
        base.append(
            "Possible read of test-set labels / answers for training (extreme-check heuristic)"
        )
    low = code.lower()
    if "test_labels" in low and (".fit(" in low or "fit(" in low):
        base.append("test_labels referenced near training — possible label leakage")
    if _RE_TEST_LABEL_FILES.search(code) and "train" in low and ".fit(" in low:
        base.append("test/answer files and training in same script — review for leakage")
    return base


def check_submission_content_quality(
    submission_path: Path,
    *,
    constant_threshold: float = 1.0,
) -> list[str]:
    """Detect fully constant prediction columns (placeholders).

    Default ``constant_threshold`` is 1.0 so only columns where *every* non-null
    value is identical are flagged — imbalanced classification with ~95%+ one
    class is not treated as a placeholder.
    """
    issues: list[str] = []
    try:
        df = pd.read_csv(submission_path)
    except Exception as exc:
        return [f"[CONTENT] Cannot read submission: {exc}"]

    common = {"id", "_id", "image_id", "patient_id"}
    id_cols = {c for c in df.columns if c.lower() in common or c in common}
    if not id_cols and len(df.columns):
        id_cols = {df.columns[0]}
    pred_cols = [
        c
        for c in df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in pred_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            issues.append(f"[CONTENT] Column '{col}' is entirely empty")
            continue
        vc = vals.value_counts()
        most_common_ratio = float(vc.iloc[0]) / len(vals)
        if most_common_ratio >= constant_threshold:
            issues.append(
                f"[CONTENT] Column '{col}' is {most_common_ratio:.0%} constant "
                f"(value={vc.index[0]}) — likely placeholder"
            )
    return issues


@dataclass
class GuardResult:
    patched_code: str
    issues: list[str]
    is_fatal: bool
    auto_patched: list[str]


def pre_execution_guard(
    code: str,
    *,
    gpu_available: bool = False,
    dedup_checker: Any = None,
    solution_path: Path | None = None,
    leakage_checks_enabled: bool = True,
) -> GuardResult:
    """Run static checks on *code*; may auto-patch tqdm; may mark fatal issues."""
    issues: list[str] = []
    auto_patched: list[str] = []
    patched = code or ""
    is_fatal = False

    # 1. Leakage (warning — AST heuristics have false positives; orchestrator may run repair session)
    if leakage_checks_enabled:
        leakage = validate_no_leakage(patched)
        if leakage:
            issues.extend(f"[LEAKAGE] {i}" for i in leakage)

    # 2. stdout / tqdm
    stdout = check_stdout_hostile(patched)
    if stdout:
        fixed = strip_tqdm(patched)
        if fixed != patched:
            patched = fixed
            auto_patched.append("Auto-stripped tqdm from code")
            stdout = check_stdout_hostile(patched)
        if stdout:
            issues.extend(f"[STDOUT] {i}" for i in stdout)

    # 3. Dedup (code string vs already registered hashes)
    if dedup_checker is not None and solution_path is not None:
        try:
            ws_root = solution_path.parent
            if hasattr(dedup_checker, "is_duplicate_workspace") and ws_root.is_dir():
                if dedup_checker.is_duplicate_workspace(ws_root):
                    issues.append("[DEDUP] Workspace Python sources duplicate a previous node")
                    is_fatal = True
            elif dedup_checker.is_duplicate_code(patched):
                issues.append("[DEDUP] Solution code duplicates a previous node")
                is_fatal = True
        except Exception as exc:
            issues.append(f"[DEDUP] Check failed: {exc}")

    # 4. Hardcoded paths
    issues.extend(_check_hardcoded_paths(patched))

    # 5. Syntax / parse
    syn = _check_import_issues(patched)
    if syn:
        issues.extend(syn)
        is_fatal = True

    # 6. Submission heuristic
    if not _has_submission_save(patched):
        issues.append(
            "[WARN] No submission.csv save logic detected — "
            "code may finish without producing predictions",
        )

    # 7. Device
    issues.extend(_check_device_mismatch(patched, gpu_available))

    # 8. result.md is system-generated; solution.py must not write it (LNR contract)
    issues.extend(_check_writes_result_md(patched))

    return GuardResult(
        patched_code=patched,
        issues=issues,
        is_fatal=is_fatal,
        auto_patched=auto_patched,
    )
