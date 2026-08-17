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

"""Post-agent full execution: run ``python3 solution.py`` without quick-test env.

Agent-side dependency installs go through the bash tool; bare ``pip`` / ``pip3``
commands are normalized to ``uv pip`` (see
``scienceflow.core.tools.bash_tool.normalize_bash_command_for_agent``), logged as
``[bash-normalize]``. This safety policy only runs ``solution.py``, not ``pip``.

Ensures a full training/inference pass after the draft agent finishes, so the
search does not depend on the LLM remembering to unset ``QUICK_TEST_ROWS``.
Valid ``.logs/full_run_stamp.json`` (from a prior safety-policy run or agent full pass)
skips duplicate execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from scienceflow.core.subprocess_utils import _LIVE_PGIDS, spawn_exec, terminate_tree
from scienceflow.utils.node_paths import (
    find_node_log_path,
    legacy_node_logs_dir,
    node_log_path,
    node_logs_dir,
)
from scienceflow.utils.quick_test_perf import parse_epoch_line
from scienceflow.utils.workspace_interaction_log import ConsecutiveLineDeduper

logger = logging.getLogger("scienceflow")

# StreamReader buffer (bytes); long tqdm lines need headroom like BashTool.
_DEFAULT_SUBPROCESS_STREAM_LIMIT = 50 * 1024 * 1024
_FINAL_VALIDATION_SCORE_LABEL_RE = re.compile(r"Final\s+Validation\s+Score", re.IGNORECASE)
_FINAL_VALIDATION_SCORE_VALUE_RE = re.compile(
    r"Final\s+Validation\s+Score(?:\s*\([^\n)]*\))?\s*:\s*([^\s]+)",
    re.IGNORECASE,
)
_SOURCE_SCORE_TEMPLATE_RE = re.compile(
    r"\bprint\s*\(|\blogger\.\w+\s*\(|\blogging\.\w+\s*\(|\{[^}\n]*\}",
    re.IGNORECASE,
)
_NUMBERED_SOURCE_LINE_RE = re.compile(r"^\s*\d+\s*[|:]\s*")
_TAGGED_TIMEOUT_MARKER_RE = re.compile(
    r"(?m)^\[[^\]\r\n]+\][ \t]+TIMEOUT after(?:[ \t]|$)"
)


def _has_timeout_marker(stdout: str, stderr: str) -> bool:
    """Recognize component-tagged timeout output without depending on its producer name."""

    return _TAGGED_TIMEOUT_MARKER_RE.search(f"{stdout}\n{stderr}") is not None


class FullRunEpochWatchdog:
    """Kill full-run subprocess if projected remaining epoch time exceeds wall budget."""

    def __init__(self, budget_sec: float, enabled: bool) -> None:
        self.budget_sec = max(1.0, float(budget_sec))
        self.enabled = bool(enabled)
        self._t0 = time.monotonic()

    def should_kill_on_line(self, line: str) -> bool:
        if not self.enabled:
            return False
        parsed = parse_epoch_line(line)
        if not parsed:
            return False
        cur, total, t_ep = parsed
        rem = max(0, total - cur)
        if rem <= 0:
            return False
        elapsed = time.monotonic() - self._t0
        remaining_budget = max(0.0, self.budget_sec - elapsed)
        projected = t_ep * rem
        if projected > remaining_budget:
            logger.warning(
                "[full-run-watchdog] epoch %d/%d time=%.1fs; projected remaining %.1fs > "
                "remaining budget %.1fs — terminating process",
                cur,
                total,
                t_ep,
                projected,
                remaining_budget,
            )
            return True
        return False

FULL_RUN_STAMP_VERSION = 1
FULL_RUN_STAMP_NAME = "full_run_stamp.json"
FULL_RUN_SOURCE_SAFETY = "safety"
_VALID_FULL_RUN_SOURCES = (
    FULL_RUN_SOURCE_SAFETY,
    "agent",
)
# Written by ScienceAgent after embedded full run; consumed by the LNR safety policy to avoid duplicate subprocess/log.
EMBEDDED_FULL_RUN_RESULT_NAME = "embedded_full_run_result.json"
# Lightweight tail snapshot persisted alongside the above; NOT consumed/deleted; survives clone.
FULLRUN_TAIL_SNAPSHOT_NAME = "fullrun_tail_snapshot.json"

# Library / Python warnings noise in full-run stdout/stderr (LightGBM bracket warnings, warnings.warn).
_WARNING_NOISE_RE = re.compile(
    r"(?:\[Warning\])"
    r"|\b\w+Warning:\s"
    r"|^\s*See https?://",
)


def drop_warning_noise_lines(text: str) -> str:
    """Remove warning-style lines and any indented continuation that follows.

    No whitelist: lines that do not match are kept. Used before tail/collapse for clone prompts.
    """
    if not text:
        return text
    out: list[str] = []
    drop_continuation = False
    for ln in text.splitlines():
        if _WARNING_NOISE_RE.search(ln):
            drop_continuation = True
            continue
        if drop_continuation:
            if ln and ln[0] in (" ", "\t"):
                continue
            drop_continuation = False
        out.append(ln)
    return "\n".join(out)


def compute_fullrun_text_tail(text: str, *, line_limit: int, max_chars: int) -> str:
    """Last ``line_limit`` lines of stdout/stderr for full-run snapshot and ``result.md``.

    Applies :func:`drop_warning_noise_lines`, collapses consecutive duplicate lines, then
    truncates to ``max_chars`` from the end (same rules as :func:`write_fullrun_tail_snapshot`).
    """
    if line_limit <= 0 or not text:
        return ""
    text = drop_warning_noise_lines(text)
    lines = text.splitlines()
    # Collapse consecutive identical lines (e.g. LightGBM repeated warnings).
    collapsed: list[str] = []
    prev = None
    repeat = 0
    for ln in lines:
        if ln == prev:
            repeat += 1
        else:
            if repeat > 0:
                collapsed.append(f"  [previous line repeated {repeat} more time(s)]")
            collapsed.append(ln)
            prev = ln
            repeat = 0
    if repeat > 0:
        collapsed.append(f"  [previous line repeated {repeat} more time(s)]")
    tail = "\n".join(collapsed[-line_limit:])
    if len(tail) > max_chars:
        trunc = len(tail) - max_chars
        tail = f"# ...truncated {trunc} chars...\n" + tail[-max_chars:]
    return tail


def _full_run_stamp_path(node_dir: Path) -> Path:
    return find_node_log_path(node_dir, FULL_RUN_STAMP_NAME)


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_full_run_stamp(node_dir: Path) -> dict | None:
    p = _full_run_stamp_path(node_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _stamp_hashes_match(node_dir: Path, data: dict) -> bool:
    """Return False when a hash-bound stamp no longer matches current artifacts."""
    expected_solution = str(data.get("solution_sha") or "").strip()
    expected_submission = str(data.get("submission_sha") or "").strip()
    if expected_solution and _sha256_file(node_dir / "solution.py") != expected_solution:
        return False
    if expected_submission and _sha256_file(node_dir / "submission.csv") != expected_submission:
        return False
    return True


def _is_valid_full_run_stamp(data: dict, node_dir: Path | None = None) -> bool:
    if data.get("version") != FULL_RUN_STAMP_VERSION:
        return False
    if data.get("source") not in _VALID_FULL_RUN_SOURCES:
        return False
    if node_dir is not None and not _stamp_hashes_match(node_dir, data):
        return False
    raw = data.get("metric_value")
    if raw is None:
        return False
    try:
        return math.isfinite(float(raw))
    except (TypeError, ValueError):
        return False


def _write_full_run_stamp(
    node_dir: Path,
    source: str,
    *,
    metric_value: float | None = None,
    metric_name: str = "unknown",
    lower_is_better: bool = True,
) -> None:
    if source not in _VALID_FULL_RUN_SOURCES:
        raise ValueError(source)
    node_logs_dir(node_dir, create=True)
    body: dict[str, object] = {
        "version": FULL_RUN_STAMP_VERSION,
        "source": source,
        "unix_ts": time.time(),
    }
    if metric_value is not None:
        mv = float(metric_value)
        if not math.isfinite(mv):
            raise ValueError(f"metric_value must be finite, got {metric_value!r}")
        body["metric_value"] = mv
    body["metric_name"] = metric_name
    body["lower_is_better"] = lower_is_better
    solution_sha = _sha256_file(node_dir / "solution.py")
    submission_sha = _sha256_file(node_dir / "submission.csv")
    if solution_sha:
        body["solution_sha"] = solution_sha
    if submission_sha:
        body["submission_sha"] = submission_sha
    node_log_path(node_dir, FULL_RUN_STAMP_NAME, create_parent=True).write_text(
        json.dumps(body, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric_from_stamp(stamp: dict) -> tuple[float | None, str, bool]:
    """Read metric fields from stamp; missing keys are backward-compatible."""
    mv: float | None = None
    raw = stamp.get("metric_value")
    if raw is not None:
        try:
            mv_f = float(raw)
            mv = mv_f if math.isfinite(mv_f) else None
        except (TypeError, ValueError):
            mv = None
    mname = stamp.get("metric_name")
    if not isinstance(mname, str) or not mname:
        mname = "unknown"
    lib = stamp.get("lower_is_better")
    lower = True if lib is None else bool(lib)
    return mv, mname, lower


@dataclass
class EnsureFullRunResult:
    """Outcome of :func:`ensure_full_execution`."""

    executed: bool
    skipped: bool
    reason: str
    exit_code: int | None = None
    wall_sec: float = 0.0
    metric_value: float | None = None
    metric_name: str = "unknown"
    lower_is_better: bool = True
    stdout: str = ""
    stderr: str = ""
    score_contract_error: str = ""


def format_full_run_log_body(stdout: str, stderr: str) -> str:
    """Return the legacy full-run log layout used by ``interaction.log``."""
    from scienceflow.utils.workspace_interaction_log import (
        collapse_consecutive_lightgbm_warnings_in_text,
    )

    out = collapse_consecutive_lightgbm_warnings_in_text(stdout or "")
    err = collapse_consecutive_lightgbm_warnings_in_text(stderr or "")
    return f"=== stdout ===\n{out}\n\n=== stderr ===\n{err}\n"


def _should_skip_full_run(node_dir: Path) -> tuple[bool, str]:
    """Skip when there is nothing to run, or a valid full-run stamp is present.

    Quick-test runs can leave ``result.md`` / ``submission.csv`` with a metric;
    those alone never skip. A ``logs/full_run_stamp.json`` written by the safety policy
    (after exit 0) or by the agent after a real full run (no ``QUICK_TEST_ROWS``)
    skips a duplicate full execution.
    """
    sol = node_dir / "solution.py"
    if not sol.is_file():
        return True, "no solution.py"
    stamp = _load_full_run_stamp(node_dir)
    if stamp is not None and _is_valid_full_run_stamp(stamp, node_dir):
        return True, "full_run_stamp_present"
    return False, ""


def _stdout_suggests_score_maximize(stdout: str) -> bool:
    """True when logs indicate a higher-is-better validation objective.

    Used to avoid treating ``Final Validation Score`` as a minimization metric when
    the same stdout clearly reports a classification/ranking score.  MLEBench
    agents commonly print variants such as ``Validation AUC:`` rather than a
    machine-like ``val_auc=`` token, so keep these patterns semantic instead of
    tied to one naming convention.
    """
    text = stdout or ""
    if re.search(r"(?i)(?:\b(?:roc[_\s-]?auc|auroc)\b|(?:^|[^a-z0-9])auc\b)", text):
        return True
    if re.search(
        r"(?i)\b(?:accuracy|acc|f1|fbeta|jaccard|dice|iou|mcc|matthews(?:_|\s|-)?corr(?:coef|elation)?)\b\s*[:=]",
        text,
    ):
        return True
    if re.search(
        r"(?i)Final\s+Validation\s+Score\s*\([^)]*\b"
        r"(?:accuracy|acc|f1|fbeta|jaccard|dice|iou|mcc|matthews(?:_|\s|-)?corr(?:coef|elation)?)\b",
        text,
    ):
        return True
    if re.search(r"(?i)\b(?:average\s+precision|map@?\d*|mAP|ndcg@?\d*)\b", text):
        return True
    if re.search(r"(?i)\b(?:r2|r\^2|pearson|spearman|kappa)\b\s*[:=]", text):
        return True
    return False


def _stdout_suggests_auc_maximize(stdout: str) -> bool:
    """Backward-compatible wrapper for older tests/imports."""
    return _stdout_suggests_score_maximize(stdout)


def _bounded_score_range_from_stdout(stdout: str) -> tuple[str, float, float] | None:
    """Return an expected score range when stdout names a bounded metric.

    This is a contract guard, not a scorer. It catches invalid validation metrics
    produced by candidate code, e.g. MCC implementations that overflow integer
    products on large validation sets and print values outside ``[-1, 1]``.
    """
    text = stdout or ""
    if re.search(
        r"(?i)\b(?:mcc|matthews(?:_|\s|-)?corr(?:coef|elation)?)\b(?:\s*[:=]|(?=[^)\n]*\)))",
        text,
    ):
        return "MCC", -1.0, 1.0
    if re.search(
        r"(?i)\b(?:cohen(?:'s)?\s+kappa|kappa)\b\s*[:=]",
        text,
    ):
        return "kappa", -1.0, 1.0
    if re.search(r"(?i)(?:\b(?:roc[_\s-]?auc|auroc)\b|(?:^|[^a-z0-9])auc\b)", text):
        return "AUC", 0.0, 1.0
    if re.search(
        r"(?i)\b(?:accuracy|acc|f1|fbeta|jaccard|dice|iou|average\s+precision|map@?\d*|mAP|ndcg@?\d*)\b\s*[:=]?",
        text,
    ):
        return "bounded classification/ranking score", 0.0, 1.0
    return None


def _bounded_score_contract_error(stdout: str, value: float) -> str | None:
    bounds = _bounded_score_range_from_stdout(stdout)
    if bounds is None:
        return None
    label, lo, hi = bounds
    eps = 1e-9
    if (lo - eps) <= float(value) <= (hi + eps):
        return None
    return (
        f"stdout suggests {label}, whose expected validation range is "
        f"[{lo:g}, {hi:g}], but Final Validation Score is {float(value)!r}"
    )


def validation_metric_value_error(stdout: str, value: float) -> str | None:
    """Return a range/finite-value error for a non-canonical validation metric."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return f"validation metric value is not parseable as float: {value!r}"
    if not math.isfinite(parsed):
        return f"validation metric value is not finite: {value!r}"
    bounded = _bounded_score_contract_error(stdout, parsed)
    if bounded:
        return bounded.replace("Final Validation Score", "interpreted validation metric")
    return None


def _looks_like_score_source_template(line: str) -> bool:
    """True for source/listing lines that mention the score contract text.

    Tool outputs from ``cat``, ``grep``, or numbered file readers often contain
    code such as ``print(f"FINAL VALIDATION SCORE (MCC): {score:.5f}")``.
    Those lines are useful diagnostics, but they are not a completed validation
    metric and should not trigger repeated stage-gate repair loops.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _NUMBERED_SOURCE_LINE_RE.search(stripped):
        return True
    return _SOURCE_SCORE_TEMPLATE_RE.search(stripped) is not None


def looks_like_score_source_template(line: str) -> bool:
    """Public guard for rejecting source-code lines as runtime metric evidence."""
    return _looks_like_score_source_template(line)


def _line_for_match(text: str, match: re.Match[str]) -> str:
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    if end < 0:
        end = len(text)
    return text[start:end]


def _usable_final_validation_score_matches(text: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in _FINAL_VALIDATION_SCORE_VALUE_RE.finditer(text):
        if _looks_like_score_source_template(_line_for_match(text, match)):
            continue
        matches.append(match)
    return matches


def _has_runtime_final_validation_score_label(text: str) -> bool:
    for match in _FINAL_VALIDATION_SCORE_LABEL_RE.finditer(text):
        if _looks_like_score_source_template(_line_for_match(text, match)):
            continue
        return True
    return False


def _line_has_runtime_final_validation_score_value(line: str) -> bool:
    if _looks_like_score_source_template(line):
        return False
    return _FINAL_VALIDATION_SCORE_VALUE_RE.search(line) is not None


def _parse_final_validation_score_value(stdout: str) -> tuple[bool, float | None, str | None]:
    """Return ``(seen_contract, value, error)`` for the final-score stdout contract."""
    text = stdout or ""
    matches = _usable_final_validation_score_matches(text)
    if not matches:
        if _has_runtime_final_validation_score_label(text):
            return True, None, "Final Validation Score label is present but no `: <value>` token was found"
        return False, None, None
    raw = matches[-1].group(1).strip().strip("`").rstrip(",;")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return True, None, f"Final Validation Score value is not parseable as float: {raw!r}"
    if not math.isfinite(value):
        return True, None, f"Final Validation Score value is not finite: {raw!r}"
    return True, value, None


def final_validation_score_contract_error(
    stdout: str,
    *,
    require_present: bool = False,
    require_final_line: bool = False,
) -> str | None:
    """Human-readable reason when stdout cannot provide a finite final score.

    ``require_present=False`` keeps the legacy generic fallback behavior for
    logs that simply omit the LNR contract. If a ``Final Validation Score`` label
    is present, however, the value must be parseable and finite; otherwise we must
    not fall back to an intermediate metric such as LightGBM's per-target RMSE.
    """
    seen, _value, error = _parse_final_validation_score_value(stdout)
    if error:
        return error
    if seen and _value is not None:
        bounded_error = _bounded_score_contract_error(stdout, _value)
        if bounded_error:
            return bounded_error
    if require_present and not seen:
        return "missing required `Final Validation Score: <finite_float>` line"
    if require_final_line and seen:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        if lines and not _line_has_runtime_final_validation_score_value(lines[-1]):
            return "`Final Validation Score: <finite_float>` is not the last non-empty stdout line"
    return None


def format_score_contract_repair_feedback(
    detail: str,
    *,
    script_label: str = "solution.py",
) -> str:
    """Actionable user-facing repair guidance for score-contract failures.

    The most expensive failure mode is rerunning training only to repair a missing
    final metric line. Keep the feedback short, but explicitly bias the agent
    toward reusing existing artifacts and writing a predict/score finalizer first.
    """
    label = str(script_label or "solution.py").strip() or "solution.py"
    return (
        f"[SCORE-CONTRACT-INVALID] The full `python {label}` run exited, but "
        "the safety policy could not use it as the stage-performance metric for this node.\n"
        f"Reason: {detail}\n\n"
        "If this run already produced useful checkpoints, validation predictions, "
        "fold metrics, OOF files, logs with a comparable metric, or a draft "
        "submission, do not retrain from scratch by default. Preserve those "
        "artifacts and first write a lightweight `predict.py` / "
        "`score_existing.py` finalization path that loads the existing artifacts, "
        "computes a comparable validation metric, writes root `submission.csv`, "
        "and prints exactly one final stdout line in this form:\n"
        "print(f\"Final Validation Score: {float(score)}\")\n\n"
        "`score` must be a computed finite validation metric, not `inf`, `nan`, "
        "a placeholder, a single-fold progress metric, or an intermediate "
        "per-target training metric. Only retrain if the existing artifacts cannot "
        "be loaded or scored. Then re-run the cheapest finalization command, not "
        "the expensive training command, unless retraining is truly required."
    )


def _extract_metric_from_stdout(stdout: str) -> tuple[float | None, str, bool]:
    """Best-effort metric from solution stdout. Returns (value, name, lower_is_better)."""
    text = stdout or ""
    # Highest priority: agent contract line (see agent_roles _DRAFT_OUTPUT_CONTRACT).
    seen_fvs, val, fvs_error = _parse_final_validation_score_value(text)
    if seen_fvs:
        lb = not _stdout_suggests_score_maximize(text)
        if fvs_error is not None or val is None:
            return None, "Final Validation Score", lb
        if _bounded_score_contract_error(text, val):
            return None, "Final Validation Score", lb
        return val, "Final Validation Score", lb
    # Multi-output / regression style (common in tabular competitions)
    m = re.search(r"Average\s+R²\s*[:=]\s*([0-9eE+\-.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), "Average R²", False
        except ValueError:
            pass
    m = re.search(r"Overall:\s*R²\s*=\s*([0-9eE+\-.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), "Overall R²", False
        except ValueError:
            pass
    m = re.search(r"R²\s*=\s*([0-9eE+\-.]+)", text)
    if m:
        try:
            return float(m.group(1)), "R²", False
        except ValueError:
            pass
    # RMSE / MAE — lower is better
    m = re.search(r"RMSE\s*[:=]\s*([0-9eE+\-.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), "RMSE", True
        except ValueError:
            pass
    m = re.search(r"MAE\s*[:=]\s*([0-9eE+\-.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), "MAE", True
        except ValueError:
            pass
    # Accuracy / score
    m = re.search(r"Accuracy\s*[:=]\s*([0-9eE+\-.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), "Accuracy", False
        except ValueError:
            pass
    # Last METRIC: line
    found = re.findall(r"^METRIC:\s*([0-9eE+\-.]+)", text, re.MULTILINE)
    if found:
        try:
            return float(found[-1]), "METRIC", True
        except ValueError:
            pass
    return None, "unknown", True


def parse_lower_is_better_from_result_md(result_md_path: Path) -> bool | None:
    """Return explicit ``lower_is_better`` from ``result.md`` when present.

    Matches plain ``lower_is_better:`` lines and embedded-full-run markdown list style.
    Returns ``None`` if the file is missing or the key is absent.
    """
    if not result_md_path.is_file():
        return None
    try:
        text = result_md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw: str | None = None
    m = re.search(r"^lower_is_better:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    else:
        m2 = re.search(
            r"^-\s*\*\*lower_is_better\*\*:\s*(.+)$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if m2:
            raw = m2.group(1).strip()
    if raw is None:
        return None
    low = raw.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    return None


def _parse_custom_is_lower_better_from_env() -> bool | None:
    """Mirror ``apply_environment_overrides_to_config`` for ``CUSTOM_IS_LOWER_BETTER``."""
    cilb = os.environ.get("CUSTOM_IS_LOWER_BETTER", "").strip().lower()
    if cilb in ("1", "true", "yes", "on"):
        return True
    if cilb in ("0", "false", "no", "off"):
        return False
    return None


def resolve_lower_is_better_for_bare_snapshot(
    *,
    workspace: Path,
    context_json: dict[str, Any] | None,
    stdout: str,
) -> bool:
    """Pick metric direction for bare-run tail snapshots.

    Order: explicit ``result.md`` → ``CUSTOM_IS_LOWER_BETTER`` env →
    high-confidence stdout maximize hints → ``context.json`` →
    stdout heuristics from :func:`_extract_metric_from_stdout` (e.g. RMSE vs R²).

    The context can be a generic task default, so a concrete stdout line such as
    ``Val MCC:`` or ``Validation AUC:`` must win over that default.
    """
    rd = parse_lower_is_better_from_result_md(workspace / "result.md")
    if rd is not None:
        return rd
    env_lb = _parse_custom_is_lower_better_from_env()
    if env_lb is not None:
        return env_lb
    if _stdout_suggests_score_maximize(stdout):
        return False
    ctx = context_json if isinstance(context_json, dict) else {}
    lb_ctx = ctx.get("lower_is_better")
    if isinstance(lb_ctx, bool):
        return lb_ctx
    _, _, heuristic = _extract_metric_from_stdout(stdout)
    return heuristic


async def _stream_subprocess_output(
    proc: asyncio.subprocess.Process,
    *,
    timeout_sec: float,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    epoch_watchdog: FullRunEpochWatchdog | None = None,
) -> tuple[str, str, bool]:
    """Drain stdout/stderr concurrently; return (stdout, stderr, killed_by_watchdog)."""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    killed = False

    async def _drain_stream(
        stream: asyncio.StreamReader | None,
        parts: list[str],
        cb: Callable[[str], None] | None,
        *,
        is_stdout: bool,
    ) -> None:
        nonlocal killed
        if stream is None:
            return
        _READ_CHUNK = 1024 * 1024
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                while True:
                    chunk_bytes = await stream.read(_READ_CHUNK)
                    if not chunk_bytes:
                        break
                    text = chunk_bytes.decode("utf-8", errors="replace")
                    parts.append(text)
                    if cb:
                        cb(text)
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            parts.append(text)
            if cb:
                cb(text)
            if epoch_watchdog is not None and is_stdout and not killed:
                if epoch_watchdog.should_kill_on_line(text):
                    killed = True
                    try:
                        # schedule tree kill without awaiting (we're inside a drain coroutine)
                        asyncio.ensure_future(terminate_tree(proc))
                    except Exception:
                        pass

    async def _run_drains() -> None:
        await asyncio.gather(
            _drain_stream(
                proc.stdout,
                stdout_parts,
                on_stdout_line,
                is_stdout=True,
            ),
            _drain_stream(
                proc.stderr,
                stderr_parts,
                on_stderr_line,
                is_stdout=False,
            ),
        )

    try:
        await asyncio.wait_for(
            _run_drains(),
            timeout=max(1.0, float(timeout_sec)),
        )
    except asyncio.TimeoutError:
        await terminate_tree(proc)
        return (
            "".join(stdout_parts),
            "".join(stderr_parts) + f"\n[safety] TIMEOUT after {timeout_sec}s\n",
            killed,
        )
    finally:
        _LIVE_PGIDS.discard(proc.pid)

    await proc.wait()
    return "".join(stdout_parts), "".join(stderr_parts), killed


async def ensure_full_execution(
    node_dir: Path | str,
    *,
    timeout_sec: float,
    extra_env: dict[str, str] | None = None,
    on_stream_line: Callable[[str, str], None] | None = None,
    fullrun_watchdog_enabled: bool = False,
    fullrun_watchdog_budget_sec: float | None = None,
    subprocess_stream_limit: int | None = None,
    bash_output_dedup_enabled: bool = True,
    bash_output_dedup_min_repeat: int = 3,
    bash_output_dedup_summary_prefix: str = "[log-dedup]",
) -> EnsureFullRunResult:
    """Run full ``python3 solution.py`` in *node_dir* when ``solution.py`` exists.

    Called by the solver after ``pre_execution_guard`` passes (no fatal issues).
    Skips if ``logs/full_run_stamp.json`` is already valid (agent or prior safety policy).
    Clears ``QUICK_TEST_ROWS`` from the subprocess environment so this pass is
    always a full run.     Does **not** write ``result.md`` (orchestrator runs a follow-up
    feedback-LLM or ScienceAgent summary after success).
    On subprocess exit 0, writes ``logs/full_run_stamp.json`` with ``source: safety``
    and optional metric fields. Stdout/stderr are returned on
    :class:`EnsureFullRunResult` (no separate safety-policy log file).
    """
    nd = Path(node_dir).resolve()
    skip, reason = _should_skip_full_run(nd)
    if skip:
        if reason == "full_run_stamp_present":
            stamp = _load_full_run_stamp(nd) or {}
            mv, mname, lower = _metric_from_stamp(stamp)
            return EnsureFullRunResult(
                executed=False,
                skipped=True,
                reason=reason,
                exit_code=0,
                wall_sec=0.0,
                metric_value=mv,
                metric_name=mname,
                lower_is_better=lower,
            )
        return EnsureFullRunResult(
            executed=False,
            skipped=True,
            reason=reason,
        )

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.pop("QUICK_TEST_ROWS", None)

    node_logs_dir(nd, create=True)

    cmd = ("python3", "solution.py")
    t0 = time.monotonic()
    stream_limit = int(subprocess_stream_limit or _DEFAULT_SUBPROCESS_STREAM_LIMIT)
    try:
        proc = await spawn_exec(
            *cmd,
            cwd=str(nd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=stream_limit,
        )
    except OSError as exc:
        logger.warning("[safety] Failed to spawn solution.py in %s: %s", nd, exc)
        wall = time.monotonic() - t0
        err_txt = f"[safety spawn error] {exc!r}\n"
        return EnsureFullRunResult(
            executed=True,
            skipped=False,
            reason="spawn_failed",
            exit_code=-1,
            wall_sec=wall,
            metric_value=None,
            stdout=err_txt,
            stderr="",
        )

    wd: FullRunEpochWatchdog | None = None
    if fullrun_watchdog_enabled:
        wd_budget = (
            float(fullrun_watchdog_budget_sec)
            if fullrun_watchdog_budget_sec is not None
            else float(timeout_sec)
        )
        wd = FullRunEpochWatchdog(wd_budget, True)

    _out_dedup: ConsecutiveLineDeduper | None = None
    _err_dedup: ConsecutiveLineDeduper | None = None
    if bash_output_dedup_enabled:
        _out_dedup = ConsecutiveLineDeduper(
            min_repeat=int(bash_output_dedup_min_repeat),
            summary_prefix=bash_output_dedup_summary_prefix,
        )
        _err_dedup = ConsecutiveLineDeduper(
            min_repeat=int(bash_output_dedup_min_repeat),
            summary_prefix=bash_output_dedup_summary_prefix,
        )

    def _stdout_cb(line: str) -> None:
        if on_stream_line:
            on_stream_line("stdout", line)
        raw = line.rstrip("\n\r")
        if _out_dedup is not None:
            for piece in _out_dedup.feed_line(raw):
                logger.info("[full-run-stream] %s", piece)
        else:
            logger.info("[full-run-stream] %s", raw)

    def _stderr_cb(line: str) -> None:
        if on_stream_line:
            on_stream_line("stderr", line)
        raw = line.rstrip("\n\r")
        if _err_dedup is not None:
            for piece in _err_dedup.feed_line(raw):
                logger.info("[full-run-stream] %s", piece)
        else:
            logger.info("[full-run-stream] %s", raw)

    out_dec, err_dec, killed_watchdog = await _stream_subprocess_output(
        proc,
        timeout_sec=max(1.0, float(timeout_sec)),
        on_stdout_line=_stdout_cb,
        on_stderr_line=_stderr_cb,
        epoch_watchdog=wd,
    )
    if _out_dedup is not None:
        for piece in _out_dedup.flush():
            logger.info("[full-run-stream] %s", piece)
    if _err_dedup is not None:
        for piece in _err_dedup.flush():
            logger.info("[full-run-stream] %s", piece)

    wall = time.monotonic() - t0
    exit_code = proc.returncode if proc.returncode is not None else -1
    if killed_watchdog and exit_code == 0:
        exit_code = -9

    if killed_watchdog:
        err_dec = (err_dec or "") + "\n[safety] Process terminated by full-run epoch watchdog\n"
        reason = "watchdog"
    elif _has_timeout_marker(out_dec, err_dec):
        reason = "timeout"
        if exit_code == 0:
            exit_code = -9
    else:
        reason = "ok" if exit_code == 0 else f"exit_{exit_code}"

    score_contract_error = final_validation_score_contract_error(out_dec)
    mv, mname, lower = _extract_metric_from_stdout(out_dec)
    if exit_code != 0:
        mv = None
        mname = "unknown"
        score_contract_error = ""
    elif score_contract_error:
        mv = None
        mname = "Final Validation Score"
        reason = "score_contract_failed"
        err_dec = (
            (err_dec or "")
            + "\n[safety] Final Validation Score contract failed.\n"
            + format_score_contract_repair_feedback(
                score_contract_error,
                script_label="solution.py",
            )
            + "\n"
        )

    if exit_code == 0 and mv is not None and not score_contract_error:
        _write_full_run_stamp(
            nd,
            FULL_RUN_SOURCE_SAFETY,
            metric_value=mv,
            metric_name=mname,
            lower_is_better=lower,
        )

    logger.info(
        "[safety] full run %s exit=%s wall=%.1fs metric=%s reason=%s",
        nd.name,
        exit_code,
        wall,
        mv,
        reason,
    )
    return EnsureFullRunResult(
        executed=True,
        skipped=False,
        reason=reason,
        exit_code=exit_code,
        wall_sec=wall,
        metric_value=mv,
        metric_name=mname,
        lower_is_better=lower,
        stdout=out_dec,
        stderr=err_dec,
        score_contract_error=score_contract_error,
    )


def _embedded_full_run_result_path(node_dir: Path) -> Path:
    return find_node_log_path(node_dir, EMBEDDED_FULL_RUN_RESULT_NAME)


def ensure_full_run_result_to_dict(r: EnsureFullRunResult) -> dict[str, object]:
    return {
        "executed": r.executed,
        "skipped": r.skipped,
        "reason": r.reason,
        "exit_code": r.exit_code,
        "wall_sec": r.wall_sec,
        "metric_value": r.metric_value,
        "metric_name": r.metric_name,
        "lower_is_better": r.lower_is_better,
        "stdout": r.stdout or "",
        "stderr": r.stderr or "",
        "score_contract_error": r.score_contract_error or "",
    }


def ensure_full_run_result_from_dict(data: dict[str, Any]) -> EnsureFullRunResult | None:
    try:
        ex = data.get("executed")
        sk = data.get("skipped")
        if not isinstance(ex, bool) or not isinstance(sk, bool):
            return None
        reason = data.get("reason")
        if not isinstance(reason, str):
            reason = str(reason or "")
        ec_raw = data.get("exit_code")
        ec: int | None
        if ec_raw is None:
            ec = None
        else:
            ec = int(ec_raw)
        ws = float(data.get("wall_sec") or 0.0)
        mv = data.get("metric_value")
        if mv is not None:
            try:
                mv = float(mv)
            except (TypeError, ValueError):
                mv = None
        mname = data.get("metric_name")
        if not isinstance(mname, str) or not mname:
            mname = "unknown"
        lib = data.get("lower_is_better")
        lower = True if lib is None else bool(lib)
        out = data.get("stdout")
        err = data.get("stderr")
        score_err = data.get("score_contract_error")
        return EnsureFullRunResult(
            executed=ex,
            skipped=sk,
            reason=reason,
            exit_code=ec,
            wall_sec=ws,
            metric_value=mv,
            metric_name=mname,
            lower_is_better=lower,
            stdout=out if isinstance(out, str) else "",
            stderr=err if isinstance(err, str) else "",
            score_contract_error=score_err if isinstance(score_err, str) else "",
        )
    except (TypeError, ValueError):
        return None


def write_embedded_full_run_result(node_dir: Path | str, r: EnsureFullRunResult) -> None:
    """Persist embedded full-run outcome for :func:`try_consume_embedded_full_run_result`."""
    nd = Path(node_dir).resolve()
    node_logs_dir(nd, create=True)
    body = json.dumps(
        ensure_full_run_result_to_dict(r),
        ensure_ascii=False,
        indent=2,
    )
    node_log_path(nd, EMBEDDED_FULL_RUN_RESULT_NAME, create_parent=True).write_text(
        body + "\n",
        encoding="utf-8",
    )


def write_fullrun_tail_snapshot(
    node_dir: Path | str,
    r: EnsureFullRunResult,
    *,
    stdout_tail_lines: int = 50,
    stderr_tail_lines: int = 20,
    max_chars: int = 4000,
    bash_cmd: str = "",
    validation_ok: bool | None = None,
    solution_path: str | None = None,
    submission_validation_ok: bool | None = None,
    submission_status: str = "",
    execution_mode: str = "",
) -> None:
    """Persist a lightweight tail of stdout/stderr alongside the full-run result.

    Unlike ``embedded_full_run_result.json`` this file is NOT consumed/deleted by the
    safety policy, so it survives clone and is available to the child node for its first-turn
    user prompt (Track A tail injection).
    """
    nd = Path(node_dir).resolve()
    logs_dir = node_logs_dir(nd, create=True)

    data = {
        "exit_code": r.exit_code,
        "wall_sec": r.wall_sec,
        "metric_value": r.metric_value,
        "metric_name": r.metric_name,
        "lower_is_better": r.lower_is_better,
        "stdout_tail": compute_fullrun_text_tail(
            r.stdout or "", line_limit=stdout_tail_lines, max_chars=max_chars
        ),
        "stderr_tail": compute_fullrun_text_tail(
            r.stderr or "", line_limit=stderr_tail_lines, max_chars=max_chars
        ),
        "bash_cmd": bash_cmd,
    }
    if solution_path == "":
        source_rel = ""
    else:
        source_rel = str(solution_path or "solution.py").replace("\\", "/").strip().lstrip("/")
        source_parts = tuple(part for part in source_rel.split("/") if part)
        if not source_parts or ".." in source_parts:
            source_rel = "solution.py"
    solution_sha = _sha256_file(nd / source_rel) if source_rel else ""
    submission_sha = _sha256_file(nd / "submission.csv")
    if solution_sha:
        data["solution_sha"] = solution_sha
        data["solution_path"] = source_rel
    if source_rel == "":
        data["solution_path"] = ""
    if submission_sha:
        data["submission_sha"] = submission_sha
    if validation_ok is not None:
        data["validation_ok"] = bool(validation_ok)
    if submission_validation_ok is not None:
        data["submission_validation_ok"] = bool(submission_validation_ok)
    if str(submission_status or "").strip():
        data["submission_status"] = str(submission_status).strip()
    if str(execution_mode or "").strip():
        data["execution_mode"] = str(execution_mode).strip()
    if r.score_contract_error:
        data["score_contract_error"] = r.score_contract_error
    body = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path = logs_dir / FULLRUN_TAIL_SNAPSHOT_NAME
    try:
        path.write_text(body, encoding="utf-8")
    except OSError:
        pass
    legacy_logs = legacy_node_logs_dir(nd)
    if legacy_logs.exists() and legacy_logs != logs_dir:
        try:
            legacy_logs.mkdir(parents=True, exist_ok=True)
            (legacy_logs / FULLRUN_TAIL_SNAPSHOT_NAME).write_text(
                body,
                encoding="utf-8",
            )
        except OSError:
            pass


def clear_embedded_full_run_result(node_dir: Path | str) -> None:
    """Remove snapshot (e.g. at start of a new ScienceAgent ``run()``)."""
    p = _embedded_full_run_result_path(Path(node_dir).resolve())
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def try_consume_embedded_full_run_result(node_dir: Path | str) -> EnsureFullRunResult | None:
    """Load embedded snapshot from a prior in-agent full run and delete the file.

    Returns ``None`` if missing or invalid. Caller should not append duplicate
    ``[full-run]`` lines to ``interaction.log`` (embedded path already mirrored).
    """
    p = _embedded_full_run_result_path(Path(node_dir).resolve())
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    if not isinstance(raw, dict):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    r = ensure_full_run_result_from_dict(raw)
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
    return r
