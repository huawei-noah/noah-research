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

"""Embedded full-run after quick-test.

Default LNR is agent-only full runs; quick-test + embedded handoff is opt-in
(``agent_embedded_full_run_enabled`` in ``lnr``). For tail-only
clone context with embedded off, see :meth:`EmbeddedFullRunMixin._maybe_write_bare_run_tail_snapshot`.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from deepcraft_core import Message
from deepcraft_core.tool import ToolResult

from scienceflow.core.agent.tools.bash_utils import (
    _embedded_failure_user_message,
    _looks_like_bare_solution_run,
    _python_script_run_rel_path,
    _quick_test_output_has_traceback,
)
from scienceflow.core.agent.io.interaction_log import truncate_for_interaction_log
from scienceflow.gates import legacy_score_contract_enabled
from scienceflow.gates.evaluator import EvalContext, EvaluationRequest
from scienceflow.safety.execution_policy import (
    EnsureFullRunResult,
    _extract_metric_from_stdout,
    _write_full_run_stamp,
    compute_fullrun_text_tail,
    ensure_full_execution,
    format_full_run_log_body,
    final_validation_score_contract_error,
    format_score_contract_repair_feedback,
    looks_like_score_source_template,
    resolve_lower_is_better_for_bare_snapshot,
    validation_metric_value_error,
    write_embedded_full_run_result,
    write_fullrun_tail_snapshot,
)
from scienceflow.utils.cost_tracker import format_fullrun_cost_report, parse_fullrun_cost_summary
from scienceflow.utils.quick_test_perf import (
    estimate_full_run_seconds,
    estimate_full_run_seconds_from_wall,
    format_extrapolation_block,
    load_dataset_total_rows_from_context,
    parse_bash_tool_elapsed_sec,
    parse_quick_test_stdout,
    strip_bash_tool_prefix,
)
from scienceflow.utils.workspace_interaction_log import (
    FullRunLightGBMStreamDeduper,
    write_raw_to_interaction_log,
)

logger = logging.getLogger("scienceflow")


def _agent_gate_policy_context(agent: Any) -> tuple[str, str, str]:
    profile = str(getattr(agent, "_scienceflow_task_profile", "auto") or "auto")
    backend = str(getattr(agent, "_scienceflow_evaluator_backend", "") or "")
    artifact = str(getattr(agent, "_lnr_candidate_artifact_rel", "") or "")
    return profile, backend, artifact


def _legacy_score_gate_enabled_for_agent(agent: Any) -> bool:
    profile, backend, artifact = _agent_gate_policy_context(agent)
    return legacy_score_contract_enabled(
        task_profile=profile,
        evaluator_backend=backend,
        candidate_artifact=artifact,
    )


def _evaluate_embedded_candidate(
    agent: Any, safety_result: EnsureFullRunResult
) -> Any | None:
    """Evaluate an embedded full-run through the framework service when attached."""

    service = getattr(agent, "_scienceflow_evaluation_service", None)
    if service is None:
        return None
    cfg = getattr(agent, "_scienceflow_evaluation_cfg", None)
    task_id = str(
        getattr(cfg, "exp_id", "")
        or getattr(agent, "_mlebench_exp_id", "")
        or ""
    )
    task_root = Path(
        getattr(agent, "_scienceflow_task_root", "") or agent._workspace_dir
    )
    ctx = EvalContext(
        task_profile=str(getattr(agent, "_scienceflow_task_profile", "") or ""),
        task_id=task_id,
        task_root=task_root,
        workspace=agent._workspace_dir,
        worker_id=str(getattr(agent, "_scienceflow_worker_id", "") or "AGENT"),
        stage_id="embedded_final",
        cfg=cfg,
        metadata={
            "metric_event": {
                "metric_value": safety_result.metric_value,
                "metric_name": safety_result.metric_name,
                "lower_is_better": safety_result.lower_is_better,
                "validation_ok": safety_result.exit_code == 0,
            }
        },
    )
    try:
        outcomes = service.evaluate(
            EvaluationRequest(context=ctx, trigger="finalize")
        )
    except Exception:
        logger.debug(
            "[embedded-full-run] unified candidate evaluation failed",
            exc_info=True,
        )
        return None
    return outcomes[0] if outcomes else None


def _score_contract_user_message(detail: str, *, script_label: str = "solution.py") -> str:
    return format_score_contract_repair_feedback(detail, script_label=script_label)


def _score_contract_errors(stdout: str) -> tuple[str | None, str | None]:
    """Return ``(hard_error, soft_warning)`` for bare-run score parsing.

    Missing, unparseable, or non-finite ``Final Validation Score`` is a hard gate:
    the node has no reliable metric. A parseable score that is not the last
    stdout line is only a formatting issue; forcing another full training run for
    that case is too expensive and does not improve metric trust once submission
    validation has already passed.
    """
    hard_error = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=False,
    )
    if hard_error:
        return hard_error, None
    soft_warning = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=True,
    )
    return None, soft_warning


_MISSING_FINAL_SCORE_ERROR = "missing required `Final Validation Score: <finite_float>` line"
_INTERPRETED_METRIC_SPLITS = {"validation", "val", "holdout", "heldout", "cv", "oof"}
_INTERPRETED_METRIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_NON_VALIDATION_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:private|public[ _-]+leaderboard|leaderboard|test(?:ing)?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_VALIDATION_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:val(?:idation)?|hold[ _-]?out|cv|oof)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _validate_llm_metric_interpretation(
    stdout: str,
    interpretation: Any,
) -> tuple[float | None, str, str | None]:
    """Verify that an LLM interpretation is directly grounded in stdout."""
    if not isinstance(interpretation, dict):
        return None, "", "metric interpreter returned no structured result"
    if interpretation.get("metric_found") is not True:
        return None, "", "metric interpreter did not find a final validation metric"
    if interpretation.get("is_final") is not True:
        return None, "", "interpreted metric is not explicitly final or best"
    if str(interpretation.get("confidence") or "").strip().lower() != "high":
        return None, "", "metric interpreter confidence is not high"

    split = str(interpretation.get("split") or "").strip().lower().replace("-", "")
    if split not in _INTERPRETED_METRIC_SPLITS:
        return None, "", f"interpreted metric split is not validation-like: {split or 'missing'}"

    evidence = str(interpretation.get("evidence_line") or "")
    if not evidence or "\n" in evidence or "\r" in evidence:
        return None, "", "metric evidence must be one non-empty stdout line"
    if evidence not in (stdout or ""):
        return None, "", "metric evidence line is not present verbatim in stdout"
    if looks_like_score_source_template(evidence):
        return None, "", "metric evidence is a source-code template, not runtime output"
    if _NON_VALIDATION_EVIDENCE_RE.search(evidence):
        return None, "", "metric evidence refers to test or leaderboard data"
    if re.search(
        r"(?<![A-Za-z0-9])train(?:ing)?(?![A-Za-z0-9])",
        evidence,
        re.IGNORECASE,
    ) and not _VALIDATION_EVIDENCE_RE.search(evidence):
        return None, "", "metric evidence is training-only"

    metric_name = str(interpretation.get("metric_name") or "").strip()
    if not metric_name:
        return None, "", "interpreted metric name is missing"
    try:
        value = float(interpretation.get("metric_value"))
    except (TypeError, ValueError):
        return None, "", "interpreted metric value is not numeric"
    value_error = validation_metric_value_error(stdout, value)
    if value_error:
        return None, "", value_error

    evidence_values: list[float] = []
    for match in _INTERPRETED_METRIC_NUMBER_RE.finditer(evidence):
        try:
            candidate = float(match.group(0))
        except ValueError:
            continue
        if math.isfinite(candidate):
            evidence_values.append(candidate)
    if not any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=1e-12) for candidate in evidence_values):
        return None, "", "interpreted metric value does not occur in the evidence line"
    return value, metric_name[:160], None


_METRIC_STAGE_PYTHON_RUN_RE = re.compile(
    r"(?is)(?:^|[;&|\n(]\s*)"
    r"(?:(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;&|()]+\s+)*)"
    r"(?:timeout\s+(?:--[^\s]+\s+)*\d+(?:\.\d+)?[smhd]?\s+)?"
    r"(?:env\s+(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;&|()]+\s+)*)?"
    r"(?:uv\s+run\s+)?"
    r"(?:python(?:\d+(?:\.\d+)?)?|/[^ \t\n;&|()]*python(?:\d+(?:\.\d+)?)?)"
    r"(?:\s+-(?:B|E|I|O|OO|P|q|s|S|u))*"
    r"\s+(?:-c\b|-?\s*<<|[^\s;&|()]+\.py\b)"
)


def _looks_like_metric_stage_python_run(cmd: str) -> bool:
    """True when bash appears to execute Python code that can emit a fresh metric."""
    s = (cmd or "").strip()
    if not s:
        return False
    if _python_script_run_rel_path(s):
        return True
    return bool(_METRIC_STAGE_PYTHON_RUN_RE.search(s))


def _count_nonempty_data_lines_csv(path: Path) -> int:
    """Return body row count for a CSV (exclude one header line)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)


def _expected_test_rows_from_dataset(workspace_dir: Path) -> int | None:
    """Prefer ``dataset/test.csv``, else ``dataset/sample_submission.csv`` row counts."""
    ds = Path(workspace_dir) / "dataset"
    for name in ("test.csv", "sample_submission.csv"):
        p = ds / name
        if p.is_file():
            n = _count_nonempty_data_lines_csv(p)
            if n > 0:
                return n
    return None


def submission_rows_plausible_for_progressive_bare(workspace_dir: Path) -> tuple[bool, str]:
    """If submission is much shorter than test/sample_submission, do not stamp progressive bare run."""
    ws = Path(workspace_dir)
    sub = ws / "submission.csv"
    if not sub.is_file():
        return False, "missing submission.csv"
    n_sub = _count_nonempty_data_lines_csv(sub)
    if n_sub <= 0:
        return False, "submission has no data rows"
    expected = _expected_test_rows_from_dataset(ws)
    if expected is None or expected < 1:
        return True, "no dataset test rows to compare"
    ratio = float(n_sub) / float(expected)
    if ratio < 0.95:
        return (
            False,
            f"submission rows={n_sub} expected≈{expected} (ratio={ratio:.3f})",
        )
    return True, ""


def _safe_metric_label(metric: Any) -> str:
    try:
        value = float(metric)
    except (TypeError, ValueError):
        return "metric_null"
    if value != value or value in (float("inf"), float("-inf")):
        return "metric_null"
    return f"metric_{value:.6g}".replace("-", "neg_").replace(".", "p")


def _sha256_file(path: Path) -> str | None:
    try:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def archive_submission_history_snapshot(
    workspace_dir: Path,
    result: EnsureFullRunResult,
    *,
    bash_cmd: str,
    validation_ok: bool,
) -> Path | None:
    """Persist run-control-ready REPL artifacts without adding them to LLM context."""
    ws = Path(workspace_dir)
    submission = ws / "submission.csv"
    if not submission.is_file():
        return None

    history_dir = ws / "submission_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in history_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    seq = len(existing) + 1
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    metric_label = _safe_metric_label(result.metric_value)
    stem = f"run_{seq:04d}_{timestamp}_{metric_label}"
    snap_dir = history_dir / stem
    while snap_dir.exists():
        seq += 1
        stem = f"run_{seq:04d}_{timestamp}_{metric_label}"
        snap_dir = history_dir / stem
    snap_dir.mkdir(parents=False, exist_ok=False)

    copied: dict[str, dict[str, Any]] = {}
    for rel in ("solution.py", "submission.csv"):
        src = ws / rel
        if not src.is_file():
            continue
        dst = snap_dir / rel
        shutil.copy2(src, dst)
        copied[rel] = {
            "bytes": dst.stat().st_size,
            "sha256": _sha256_file(dst),
        }

    metadata = {
        "created_at_utc": timestamp,
        "metric_value": result.metric_value,
        "metric_name": result.metric_name,
        "lower_is_better": result.lower_is_better,
        "exit_code": result.exit_code,
        "wall_sec": result.wall_sec,
        "validation_ok": validation_ok,
        "bash_cmd": bash_cmd,
        "stdout_tail": compute_fullrun_text_tail(result.stdout, line_limit=50, max_chars=4000),
        "stderr_tail": compute_fullrun_text_tail(result.stderr, line_limit=20, max_chars=2000),
        "copied_files": copied,
    }
    (snap_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snap_dir


def _node_exec_budget_sec_for_cost_report(workspace_dir: Path) -> int | None:
    """Resolve per-node wall budget for cost reports (L2 ``node_exec_budget_sec``).

    Order: ``SCIENCEFLOW_NODE_EXEC_BUDGET_SEC`` env, then hidden/root context from the LNR solver.
    """
    env = os.environ.get("SCIENCEFLOW_NODE_EXEC_BUDGET_SEC", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    from scienceflow.utils.node_paths import find_node_context_path

    ctx_path = find_node_context_path(Path(workspace_dir))
    if not ctx_path.is_file():
        return None
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    v = ctx.get("node_exec_budget_sec")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Module-level helpers for JSONL last-line rewriting
# ---------------------------------------------------------------------------

def _read_last_jsonl_tool_content(path: Path | None) -> str | None:
    """Return the ``message.content`` of the last JSONL line in *path* if its role is 'tool'.

    Returns None if the file is absent, the last line is not a valid tool record,
    or any I/O / parse error occurs.
    """
    if path is None or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        rec = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    msg = rec.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return None
    return str(msg.get("content") or "")


def _rewrite_last_jsonl_tool_content(path: Path, new_content: str) -> bool:
    """Overwrite the ``message.content`` of the last JSONL line in *path*.

    Only rewrites when the last record's role is 'tool'.  Returns True on success.
    Silently returns False on any error or role mismatch.
    """
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    try:
        rec = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False
    msg = rec.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return False
    msg["content"] = new_content
    lines[-1] = json.dumps(rec, ensure_ascii=False)
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


class EmbeddedFullRunMixin:
    """In-agent full ``python3 solution.py`` after successful quick-test."""

    async def _maybe_write_bare_run_tail_snapshot(
        self,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> None:
        """When embedded full-run is off, write a candidate metric snapshot."""
        self._lnr_snapshot_ok = False
        self._lnr_snapshot_reason = ""
        if self._embedded_full_run_enabled:
            self._lnr_snapshot_reason = "embedded_path"
            return
        cmd = (args.get("command") or "").strip()
        script_rel_path = _python_script_run_rel_path(cmd)
        lhr_any_script = bool(getattr(self, "_lnr_allow_any_stage_script", False))
        gate_candidate = _looks_like_bare_solution_run(cmd) or bool(lhr_any_script and script_rel_path)
        metric_stage_candidate = gate_candidate or _looks_like_metric_stage_python_run(cmd)
        if not metric_stage_candidate or tool_result.error:
            self._lnr_snapshot_reason = "not_bare_or_error"
            return
        script_rel_path = script_rel_path or ("solution.py" if gate_candidate else "")
        raw_out = str(tool_result.output or "")
        stripped = strip_bash_tool_prefix(raw_out)
        if _quick_test_output_has_traceback(stripped):
            self._lnr_snapshot_reason = "traceback_in_output"
            return
        if not _legacy_score_gate_enabled_for_agent(self):
            profile, backend, artifact = _agent_gate_policy_context(self)
            self._lnr_snapshot_reason = "artifact_evaluator_primary"
            self._log_info(
                "[run-control] skip legacy score contract profile=%s backend=%s artifact=%s",
                profile,
                backend or "default",
                artifact,
            )
            return
        ws = self._workspace_dir
        ctx_prog: dict[str, Any] = {}
        from scienceflow.utils.node_paths import find_node_context_path

        _ctx_pp = find_node_context_path(ws)
        if _ctx_pp.is_file():
            try:
                _loaded = json.loads(_ctx_pp.read_text(encoding="utf-8"))
                if isinstance(_loaded, dict):
                    ctx_prog = _loaded
            except (OSError, json.JSONDecodeError):
                ctx_prog = {}
        score_err, score_warning = _score_contract_errors(stripped)
        interpreted_metric: tuple[float, str] | None = None
        if score_err == _MISSING_FINAL_SCORE_ERROR:
            interpretation_cb = getattr(self, "_lnr_metric_output_interpretation_callback", None)
            if callable(interpretation_cb):
                try:
                    interpretation = interpretation_cb(
                        agent=self,
                        stdout=stripped,
                        script_label=script_rel_path or "solution.py",
                    )
                    if inspect.isawaitable(interpretation):
                        interpretation = await interpretation
                    interpreted_value, interpreted_name, interpretation_error = (
                        _validate_llm_metric_interpretation(stripped, interpretation)
                    )
                    if interpretation_error is None and interpreted_value is not None:
                        interpreted_metric = (interpreted_value, interpreted_name)
                        score_err = None
                        score_warning = (
                            "accepted a high-confidence non-canonical validation metric "
                            "grounded in stdout"
                        )
                    else:
                        self._log_info(
                            "[run-control] metric interpretation rejected: %s",
                            interpretation_error or "unknown reason",
                        )
                except Exception as exc:
                    self._log_warning(
                        "[run-control] metric interpretation failed: %s",
                        type(exc).__name__,
                    )
        if score_err:
            if gate_candidate:
                max_fix = int(getattr(self, "_lnr_run_control_max_fix_rounds", 5) or 0)
                if max_fix <= 0:
                    max_fix = 1_000_000
                inj = int(getattr(self, "_run_control_user_injections", 0) or 0)
                if inj < max_fix:
                    self._inject_run_control_user_message(
                        _score_contract_user_message(score_err, script_label=script_rel_path or "solution.py")
                    )
                    self._log_info("[run-control] score contract invalid: %s", score_err)
                    self._lnr_snapshot_reason = "score_contract_invalid"
                    return
                self._log_warning(
                    "[run-control] score contract invalid and max fix rounds exceeded; no snapshot",
                )
                self._lnr_snapshot_reason = "score_contract_exhausted"
                return
            self._lnr_snapshot_reason = "score_contract_invalid"
            return
        if score_warning:
            self._log_info("[run-control] score contract soft warning: %s", score_warning)
        if interpreted_metric is not None:
            mv_b, mname_b = interpreted_metric
        else:
            mv_b, mname_b, _ = _extract_metric_from_stdout(stripped)
        lower_b = resolve_lower_is_better_for_bare_snapshot(
            workspace=ws,
            context_json=ctx_prog,
            stdout=stripped,
        )
        wall_b = 0.0
        try:
            parsed_elapsed = parse_bash_tool_elapsed_sec(raw_out)
            if parsed_elapsed is not None:
                wall_b = float(parsed_elapsed)
        except (TypeError, ValueError):
            wall_b = 0.0
        bare_result = EnsureFullRunResult(
            executed=True,
            skipped=False,
            reason="gate_passing_bare_agent_run",
            exit_code=0,
            wall_sec=wall_b,
            metric_value=mv_b,
            metric_name=mname_b,
            lower_is_better=lower_b,
            stdout=stripped,
            stderr="",
        )
        source_for_snapshot = script_rel_path if script_rel_path else ""
        execution_mode = "python_script" if script_rel_path else "inline_bash"
        sub = ws / "submission.csv"
        if not sub.is_file():
            write_fullrun_tail_snapshot(
                ws,
                bare_result,
                stdout_tail_lines=int(getattr(self, "_fullrun_output_tail_stdout_lines", 50)),
                stderr_tail_lines=int(getattr(self, "_fullrun_output_tail_stderr_lines", 0)),
                max_chars=int(getattr(self, "_fullrun_output_tail_max_chars", 4000)),
                bash_cmd=cmd,
                validation_ok=True,
                solution_path=source_for_snapshot,
                submission_status="missing_submission",
                execution_mode=execution_mode,
            )
            self._lnr_snapshot_ok = True
            self._lnr_snapshot_reason = "metric_only_tail_snapshot"
            self._log_info(
                "[run-control] metric-only tail snapshot written: no submission.csv metric=%s",
                "null" if mv_b is None else str(mv_b),
            )
            return
        write_fullrun_tail_snapshot(
            ws,
            bare_result,
            stdout_tail_lines=int(getattr(self, "_fullrun_output_tail_stdout_lines", 50)),
            stderr_tail_lines=int(getattr(self, "_fullrun_output_tail_stderr_lines", 0)),
            max_chars=int(getattr(self, "_fullrun_output_tail_max_chars", 4000)),
            bash_cmd=cmd,
            validation_ok=True,
            solution_path=source_for_snapshot,
            submission_status="pending_evaluator",
            execution_mode=execution_mode,
        )
        if bool(getattr(self, "_submission_history_archive_enabled", True)):
            try:
                history_path = archive_submission_history_snapshot(
                    ws,
                    bare_result,
                    bash_cmd=cmd,
                    validation_ok=True,
                )
                if history_path is not None:
                    self._log_info(
                        "[run-control] submission_history archived path=%s metric=%s",
                        str(history_path.relative_to(ws)),
                        "null" if mv_b is None else str(mv_b),
                    )
            except Exception as exc:
                self._log_warning("[run-control] submission_history archive failed: %s", exc)
        if mv_b is not None:
            _write_full_run_stamp(
                ws,
                "agent",
                metric_value=mv_b,
                metric_name=mname_b,
                lower_is_better=lower_b,
            )
        self._lnr_snapshot_ok = True
        self._lnr_snapshot_reason = "candidate_tail_snapshot"
        try:
            import hashlib

            ch = hashlib.sha256(cmd.encode()).hexdigest()[:8]
        except Exception:
            ch = "?"
        self._log_info(
            "[candidate-trigger] tail snapshot written bash_cmd_hash=%s metric=%s",
            ch,
            "null" if mv_b is None else str(mv_b),
        )

    def _inject_run_control_user_message(self, msg: str) -> None:
        add_after_bundle = getattr(self, "_add_message_after_current_tool_bundle", None)
        if callable(add_after_bundle):
            add_after_bundle(Message.user_message(msg))
        else:
            self.memory.add_message(Message.user_message(msg))
        self._run_control_user_injections = int(
            getattr(self, "_run_control_user_injections", 0)
        ) + 1

    def _append_to_last_tool_message(self, text: str) -> None:
        """Persist *text* to the last tool record in both short_term.json and long_term.jsonl.

        Memory.messages is a property that re-reads from disk on every access, so
        mutating the returned Message object has no effect.  We instead rewrite the
        last line of each JSONL backing file directly.
        """
        snippet = (text or "").strip()
        if not snippet:
            return
        # Resolve short_term.json path via chat_history_memory.storage.json_path
        storage = getattr(getattr(self.memory, "chat_history_memory", None), "storage", None)
        sp = getattr(storage, "json_path", None) if storage is not None else None
        short_path = Path(sp) if sp else None
        # Resolve long_term.jsonl path via memory.long_term_log
        lp = getattr(self.memory, "long_term_log", None)
        long_path = Path(lp) if lp else None
        if short_path is None and long_path is None:
            return
        # Read current content from whichever file exists (prefer long_term)
        src = long_path if (long_path and long_path.is_file()) else short_path
        prev_content = _read_last_jsonl_tool_content(src)
        if prev_content is None:
            return
        new_content = prev_content.rstrip() + "\n\n" + snippet
        if short_path is not None:
            _rewrite_last_jsonl_tool_content(short_path, new_content)
        if long_path is not None:
            _rewrite_last_jsonl_tool_content(long_path, new_content)

    def _inject_run_control_ok_message(
        self,
        *,
        validator: str,
        submission_csv: Path,
        wall_sec: float,
        context_json: dict,
    ) -> None:
        """Deprecated no-op: run-control success must not mutate LLM-visible tool memory."""
        _ = validator, submission_csv, wall_sec, context_json
        return

    def _mirror_embedded_full_run_to_interaction_log(self, safety_result: EnsureFullRunResult) -> None:
        """Append safety-policy full-run lines to interaction.log (same shape as orchestrator)."""
        ws_log = self._ws_interaction_log
        if ws_log is None:
            return
        ws_log.info("[full-run] python3 solution.py (no QUICK_TEST_ROWS)")
        content = format_full_run_log_body(
            safety_result.stdout or "",
            safety_result.stderr or "",
        )
        ec = safety_result.exit_code if safety_result.exit_code is not None else -1
        ws_log.info(
            "[full-run] exit=%s [%.1fs]\n%s",
            ec,
            safety_result.wall_sec,
            content,
        )

    def _append_embedded_full_run_to_result_md(self, safety_result: EnsureFullRunResult) -> None:
        """Append a ``## Full run (system)`` section after embedded full-run."""
        from scienceflow.utils.formatting import _fmt_num
        path = self._workspace_dir / "result.md"
        mv = safety_result.metric_value
        mv_s = "null" if mv is None else _fmt_num(mv)
        ec = safety_result.exit_code
        wall_s = _fmt_num(safety_result.wall_sec)
        block = (
            "\n\n## Full run (system)\n\n"
            "Appended automatically after a successful **quick-test** bash run "
            "(``python3 solution.py`` under the configured quick-test mode). Full stdout/stderr: "
            "see ``[full-run]`` lines in ``logs/interaction.log``.\n\n"
            f"- **exit_code**: {ec}\n"
            f"- **wall_sec**: {wall_s}\n"
            f"- **metric_name**: {safety_result.metric_name}\n"
            f"- **metric_value**: {mv_s}\n"
            f"- **lower_is_better**: {safety_result.lower_is_better}\n"
            "\n"
            f"metric_value: {mv_s}\n"
            f"exec_time: {wall_s}\n"
            f"lower_is_better: {str(safety_result.lower_is_better).lower()}\n"
        )
        if mv is not None:
            block += f"METRIC: {mv}\n"

        # Append stdout/stderr tail for downstream clone nodes to see sub-metric details.
        _tail_stdout = int(getattr(self, "_fullrun_output_tail_stdout_lines", 50))
        _tail_stderr = int(getattr(self, "_fullrun_output_tail_stderr_lines", 0))
        _tail_max = int(getattr(self, "_fullrun_output_tail_max_chars", 4000) or 4000)
        if (_tail_stdout > 0 or _tail_stderr > 0) and (_tail_stdout + _tail_stderr > 0):
            stdout_tail = compute_fullrun_text_tail(
                safety_result.stdout or "",
                line_limit=_tail_stdout,
                max_chars=_tail_max,
            )
            stderr_tail = compute_fullrun_text_tail(
                safety_result.stderr or "",
                line_limit=_tail_stderr,
                max_chars=_tail_max,
            )
            if stdout_tail or stderr_tail:
                tail_section = "\n\n### Full run — output tail (system)\n\n"
                if stdout_tail:
                    tail_section += f"<details><summary>stdout tail (last {_tail_stdout} lines)</summary>\n\n```text\n{stdout_tail}\n```\n\n</details>\n"
                if stderr_tail:
                    tail_section += f"<details><summary>stderr tail (last {_tail_stderr} lines)</summary>\n\n```text\n{stderr_tail}\n```\n\n</details>\n"
                block += tail_section

        try:
            if path.is_file():
                prev = path.read_text(encoding="utf-8", errors="replace")
                if "## Full run (system)" in prev:
                    return
                path.write_text(prev + block, encoding="utf-8")
            else:
                path.write_text(
                    "# Results\n" + block,
                    encoding="utf-8",
                )
        except OSError as exc:
            self._log_warning("[embedded-full-run] could not update result.md: %s", exc)

    def _append_mlebench_validation_to_result_md(
        self,
        exp_id: str,
        ok_call: bool,
        is_valid: bool,
        result_text: str,
    ) -> None:
        """Append ``## MLE-bench submission validation`` after embedded full-run (if enabled)."""
        path = self._workspace_dir / "result.md"
        block = (
            "\n\n## MLE-bench submission validation (system)\n\n"
            f"- **exp_id**: `{exp_id}`\n"
            f"- **sdk_call_ok**: {ok_call}\n"
            f"- **is_valid**: {is_valid}\n\n"
            f"{result_text}\n"
        )
        try:
            if path.is_file():
                prev = path.read_text(encoding="utf-8", errors="replace")
                if "## MLE-bench submission validation (system)" in prev:
                    return
                path.write_text(prev + block, encoding="utf-8")
            else:
                path.write_text("# Results\n" + block, encoding="utf-8")
        except OSError as exc:
            self._log_warning("[mlebench-validate] could not update result.md: %s", exc)

    def _run_mlebench_validation_after_embedded_full_run(
        self,
    ) -> tuple[bool | None, str, tuple[str, bool, str] | None]:
        """After full-run success: local mlebench ``validate_submission``; log only (result.md in caller).

        Returns ``(status, text, append_payload)``:
        - ``(None, "", None)`` — validation disabled or skipped
        - ``(False, error_detail, (exp_id, ok_call, result_text))`` — invalid; caller writes
          full-run + failed validation blocks to ``result.md``
        - ``(True, assistant_suffix, (exp_id, ok_call, result_text))`` — valid; caller appends result.md
        """
        if not getattr(self, "_mlebench_validate_after_embedded_full_run", False):
            return None, "", None
        mdir = getattr(self, "_mlebench_data_dir", None) or ""
        if not str(mdir).strip():
            notice = (
                "\n\n[MLE-bench] validation skipped: `mlebench_data_root_dir` is not set "
                "(set `mlebench_data_root_dir` in YAML or `MLEBENCH_DATA_ROOT_DIR`).\n"
            )
            self._log_info(
                "[mlebench-validate] skipped: mlebench_data_root_dir empty — no local validation run",
            )
            return None, notice, None
        from scienceflow.gates.evaluator.providers.mlebench import (
            resolve_mlebench_exp_id,
            validate_submission_local,
        )

        exp_id = resolve_mlebench_exp_id(
            self._workspace_dir,
            cfg_exp_id=getattr(self, "_mlebench_exp_id", None) or "",
        )
        sub = self._workspace_dir / "submission.csv"
        if not exp_id:
            notice = (
                "\n\n[MLE-bench] validation skipped: could not resolve competition `exp_id` "
                "(set `exp_id` in config or use LNR layout `…/<slug>/wsp/<node>/`).\n"
            )
            self._log_info(
                "[mlebench-validate] skipped: could not resolve exp_id — no local validation run",
            )
            return None, notice, None
        if not sub.is_file():
            notice = (
                "\n\n[MLE-bench] validation skipped: `submission.csv` not found in workspace.\n"
            )
            self._log_info(
                "[mlebench-validate] skipped: submission.csv missing — no local validation run",
            )
            return None, notice, None
        ok_call, payload = validate_submission_local(exp_id, sub, mdir)
        is_valid = bool(payload.get("is_valid"))
        result_text = str(payload.get("result", ""))
        ws_log = self._ws_interaction_log
        if ws_log is not None:
            ws_log.info(
                "[mlebench-validate] exp_id=%s call_ok=%s is_valid=%s\n%s",
                exp_id,
                ok_call,
                is_valid,
                result_text[:8000],
            )
        self._log_info(
            "[mlebench-validate] ran exp_id=%s sdk_call_ok=%s is_valid=%s",
            exp_id,
            ok_call,
            is_valid,
        )
        if not is_valid:
            err_detail = (
                f"exp_id={exp_id!r} sdk_call_ok={ok_call}\n{result_text}"
            )
            return False, err_detail, (exp_id, ok_call, result_text)
        suffix = (
            f"\n\n[MLE-bench submission validation] exp_id={exp_id!r} "
            f"is_valid={is_valid} sdk_call_ok={ok_call}\n{result_text}"
        )
        return True, suffix, (exp_id, ok_call, result_text)

    async def _maybe_embedded_full_run_after_quick_test(
        self,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> str | None:
        """If quick-test passed, run full ``python3 solution.py``.

        On **success** (exit 0): optionally run MLE-bench ``validate_submission``; if invalid,
        still append full-run + validation (``is_valid: false``) to ``result.md`` when enabled,
        then inject a user message, reset the embedded flag, and return ``None`` so the agent can fix.
        If valid or validation skipped, append ``result.md`` and end the session (return message).

        On **failure** or **exception**, write a snapshot for the outer safety policy, inject a user
        message with stderr, reset the embedded flag, and return ``None`` so the agent can fix.
        """
        if not self._embedded_full_run_enabled or self._embedded_full_run_done:
            self._log_info(
                "[embedded-full-run] skip: embedded disabled=%s or already done=%s",
                not self._embedded_full_run_enabled,
                self._embedded_full_run_done,
            )
            return None
        if not _legacy_score_gate_enabled_for_agent(self):
            profile, backend, artifact = _agent_gate_policy_context(self)
            self._log_info(
                "[embedded-full-run] skip legacy score contract profile=%s backend=%s artifact=%s",
                profile,
                backend or "default",
                artifact,
            )
            return None
        cmd = (args.get("command") or "").strip()
        cmd_triggers_embedded = _looks_like_bare_solution_run(cmd)
        if tool_result.error or not cmd_triggers_embedded:
            self._log_info(
                "[embedded-full-run] skip: tool_error=%s trigger=%s cmd_prefix=%r",
                bool(tool_result.error),
                cmd_triggers_embedded,
                cmd[:160],
            )
            return None

        raw_out = str(tool_result.output or "")
        quick_test_wall_sec = parse_bash_tool_elapsed_sec(raw_out)
        stripped = strip_bash_tool_prefix(raw_out)

        # Piped commands (e.g. `python ... | head -N`) yield bash exit 0 from `head`
        # while Python actually failed — detect traceback before submission.csv check
        # (crashes often leave no submission.csv) and inject a corrective user message.
        if _quick_test_output_has_traceback(stripped):
            self._log_info(
                "[embedded-full-run] skip: quick-test output contains traceback "
                "(pipe may have masked non-zero exit code)",
            )
            tail = stripped[-2000:] if len(stripped) > 2000 else stripped
            self.memory.add_message(
                Message.user_message(
                    "[Guard] Your quick-test appears to have crashed — the output contains "
                    "a Python traceback, but the bash exit code was 0 (likely masked by a "
                    "pipe like `| head`). Fix `solution.py` and re-run the quick-test.\n\n"
                    "=== output (tail) ===\n"
                    + tail
                )
            )
            return None

        if not (self._workspace_dir / "submission.csv").is_file():
            self._log_info(
                "[embedded-full-run] skip: submission.csv missing under %s",
                self._workspace_dir,
            )
            return None

        parsed = parse_quick_test_stdout(stripped)
        total_rows = load_dataset_total_rows_from_context(self._workspace_dir)
        if total_rows is None:
            total_rows = parsed.total_train_rows

        extrap_budget = self._quick_test_extrapolation_budget_sec
        if extrap_budget is None:
            extrap_budget = float(
                self._embedded_full_run_timeout_sec
                if self._embedded_full_run_timeout_sec is not None
                else self._bash_timeout_sec,
            )
        qt_rows = int(self._quick_test_extrapolation_rows)
        if parsed.quick_test_rows is not None and parsed.quick_test_rows > 0:
            qt_rows = min(qt_rows, parsed.quick_test_rows)
        if parsed.data_rows is not None and parsed.data_rows > 0:
            qt_rows = min(qt_rows, parsed.data_rows)

        if self._quick_test_extrapolation_enabled:
            est = estimate_full_run_seconds(
                parsed,
                quick_test_rows=qt_rows,
                total_rows=total_rows,
            )
            if est is None and quick_test_wall_sec is not None:
                est = estimate_full_run_seconds_from_wall(
                    quick_test_wall_sec=quick_test_wall_sec,
                    quick_test_rows=qt_rows,
                    total_rows=total_rows,
                )
            safety_factor = float(
                getattr(self, "_quick_test_extrapolation_safety_factor", 5.0)
            )
            if est is not None and est > float(extrap_budget) * safety_factor:
                um = format_extrapolation_block(
                    est_sec=est,
                    budget_sec=float(extrap_budget),
                    total_rows=int(total_rows or 0),
                    quick_test_rows=qt_rows,
                    parsed=parsed,
                )
                self.memory.add_message(Message.user_message(um))
                self._log_info(
                    "[quick-test-extrapolation] blocked full-run est=%.0fs budget=%.0fs safety=%.1fx",
                    est,
                    extrap_budget,
                    safety_factor,
                )
                return None

        safety_result: EnsureFullRunResult | None = None
        skip_dup = bool(
            getattr(self, "_skip_embedded_duplicate_when_progressive_bare_success", True),
        )
        if skip_dup and not tool_result.error:
            ctx_prog: dict[str, Any] = {}
            from scienceflow.utils.node_paths import find_node_context_path

            _ctx_pp = find_node_context_path(self._workspace_dir)
            if _ctx_pp.is_file():
                try:
                    _loaded = json.loads(_ctx_pp.read_text(encoding="utf-8"))
                    if isinstance(_loaded, dict):
                        ctx_prog = _loaded
                except (OSError, json.JSONDecodeError):
                    ctx_prog = {}
            score_err_b = final_validation_score_contract_error(
                stripped,
                require_present=True,
                require_final_line=True,
            )
            mv_b, mname_b, _ = _extract_metric_from_stdout(stripped)
            if score_err_b:
                self._log_info(
                    "[embedded-full-run] progressive bare run ignored: %s",
                    score_err_b,
                )
                mv_b = None
            lower_b = resolve_lower_is_better_for_bare_snapshot(
                workspace=self._workspace_dir,
                context_json=ctx_prog,
                stdout=stripped,
            )
            wall_b = (
                float(quick_test_wall_sec)
                if quick_test_wall_sec is not None
                else 0.0
            )
            if mv_b is not None:
                ok_rows, row_reason = submission_rows_plausible_for_progressive_bare(
                    self._workspace_dir,
                )
                if not ok_rows:
                    self._log_info(
                        "[embedded-full-run] skip progressive bare stamp: submission row check: %s",
                        row_reason,
                    )
                    mv_b = None
            if mv_b is not None:
                safety_result = EnsureFullRunResult(
                    executed=True,
                    skipped=False,
                    reason="progressive_bare_agent_run",
                    exit_code=0,
                    wall_sec=wall_b,
                    metric_value=mv_b,
                    metric_name=mname_b,
                    lower_is_better=lower_b,
                    stdout=stripped,
                    stderr="",
                )
                self._embedded_full_run_done = True
                _write_full_run_stamp(
                    self._workspace_dir,
                    "agent",
                    metric_value=mv_b,
                    metric_name=mname_b,
                    lower_is_better=lower_b,
                )
                self._log_info(
                    "[embedded-full-run] progressive bare run complete; skipping duplicate "
                    "ensure_full_execution (metric=%s wall=%.1fs)",
                    mv_b,
                    wall_b,
                )

        if safety_result is None:
            self._embedded_full_run_done = True
            self._log_info(
                "[handoff] quick-test passed; starting in-agent full-run (no QUICK_TEST_ROWS)",
            )
            timeout = (
                float(self._embedded_full_run_timeout_sec)
                if self._embedded_full_run_timeout_sec is not None
                else self._bash_timeout_sec
            )
            wd_budget = self._fullrun_epoch_watchdog_budget_sec
            if wd_budget is None:
                wd_budget = float(timeout)

            deduper: FullRunLightGBMStreamDeduper | None = None
            if self._ws_interaction_log is not None:
                lg = self._ws_interaction_log

                def _emit_stream_line(msg: str) -> None:
                    write_raw_to_interaction_log(
                        lg,
                        msg if msg.endswith("\n") else msg + "\n",
                    )

                deduper = FullRunLightGBMStreamDeduper(
                    emit=_emit_stream_line,
                    format_line=lambda s, ln: f"[full-run-stream] {s}: {ln}",
                )

            def _on_stream(stream: str, line: str) -> None:
                if deduper is not None:
                    deduper(stream, line)

            try:
                safety_result = await ensure_full_execution(
                    self._workspace_dir,
                    timeout_sec=timeout,
                    extra_env=self._extra_env,
                    on_stream_line=_on_stream if deduper is not None else None,
                    fullrun_watchdog_enabled=self._fullrun_epoch_watchdog_enabled,
                    fullrun_watchdog_budget_sec=wd_budget,
                    bash_output_dedup_enabled=self._bash_output_dedup_enabled,
                    bash_output_dedup_min_repeat=self._bash_output_dedup_min_repeat,
                    bash_output_dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
                )
            except Exception as exc:
                logger.warning(
                    "[embedded-full-run] ensure_full_execution failed: %s",
                    exc,
                    exc_info=True,
                )
                self._embedded_full_run_done = False
                er = EnsureFullRunResult(
                    executed=False,
                    skipped=False,
                    reason=f"embedded_exception:{type(exc).__name__}",
                    exit_code=-1,
                    wall_sec=0.0,
                    metric_value=None,
                    metric_name="unknown",
                    lower_is_better=True,
                    stdout="",
                    stderr=str(exc),
                )
                write_embedded_full_run_result(self._workspace_dir, er)
                _exc_cost_note = (
                    "[Full-run cost report] status=failed (exception — process did not start)\n"
                    "No epoch data available."
                )
                um = _embedded_failure_user_message(str(exc), -1) + f"\n\n{_exc_cost_note}"
                self.memory.add_message(Message.user_message(um))
                self._log_info(
                    "[embedded-full-run] exception; continuing session for fixes: %s",
                    type(exc).__name__,
                )
                return None
            finally:
                if deduper is not None:
                    deduper.flush()

        self._mirror_embedded_full_run_to_interaction_log(safety_result)
        write_embedded_full_run_result(self._workspace_dir, safety_result)
        embedded_validation_ok = (
            safety_result.exit_code == 0
            and safety_result.metric_value is not None
            and not safety_result.score_contract_error
        )
        write_fullrun_tail_snapshot(
            self._workspace_dir,
            safety_result,
            stdout_tail_lines=int(getattr(self, "_fullrun_output_tail_stdout_lines", 50)),
            stderr_tail_lines=int(getattr(self, "_fullrun_output_tail_stderr_lines", 20)),
            max_chars=int(getattr(self, "_fullrun_output_tail_max_chars", 4000)),
            bash_cmd="python3 solution.py",
            validation_ok=embedded_validation_ok,
        )
        if embedded_validation_ok and bool(getattr(self, "_submission_history_archive_enabled", True)):
            try:
                history_path = archive_submission_history_snapshot(
                    self._workspace_dir,
                    safety_result,
                    bash_cmd="python3 solution.py",
                    validation_ok=True,
                )
                if history_path is not None:
                    self._log_info(
                        "[embedded-full-run] submission_history archived path=%s metric=%s",
                        str(history_path.relative_to(self._workspace_dir)),
                        str(safety_result.metric_value),
                    )
            except Exception as exc:
                self._log_warning(
                    "[embedded-full-run] submission_history archive failed: %s",
                    exc,
                )

        # Build cost report from [EPOCH] lines in full-run stdout (Phase 2)
        _node_budget = _node_exec_budget_sec_for_cost_report(self._workspace_dir)
        _cost_summary = parse_fullrun_cost_summary(
            safety_result.stdout or "",
            wall_sec=safety_result.wall_sec,
            exit_code=safety_result.exit_code,
            metric_value=safety_result.metric_value,
            metric_name=safety_result.metric_name,
            lower_is_better=safety_result.lower_is_better,
            node_exec_budget_sec=_node_budget,
        )
        _cost_report = format_fullrun_cost_report(_cost_summary)

        ec = safety_result.exit_code
        if ec is not None and ec == 0:
            service_outcome = _evaluate_embedded_candidate(self, safety_result)
            if service_outcome is None:
                score_err = (
                    safety_result.score_contract_error
                    or final_validation_score_contract_error(
                        safety_result.stdout or "",
                        require_present=True,
                        require_final_line=True,
                    )
                )
                if score_err or safety_result.metric_value is None:
                    detail = score_err or "missing finite Final Validation Score metric"
                    self._embedded_full_run_done = False
                    um = _score_contract_user_message(detail) + f"\n\n{_cost_report}"
                    self.memory.add_message(Message.user_message(um))
                    self._log_info(
                        "[embedded-full-run] score contract invalid; continuing session for fix: %s",
                        detail,
                    )
                    return None
                val_status, val_text, val_append = (
                    self._run_mlebench_validation_after_embedded_full_run()
                )
            else:
                decision = service_outcome.decision
                event = service_outcome.event
                if not decision.accepted:
                    detail = decision.message or event.metric_note or decision.reason_code
                    self._embedded_full_run_done = False
                    um = (
                        f"[Guard] Candidate evaluation rejected ({decision.reason_code}).\n"
                        f"{detail}\n\n{_cost_report}"
                    )
                    self.memory.add_message(Message.user_message(um))
                    self._log_info(
                        "[embedded-full-run] unified evaluation rejected candidate: %s",
                        decision.reason_code,
                    )
                    return None
                val_status = True
                val_text = (
                    f"\n\n[Unified evaluator] backend={event.evaluator_backend} "
                    f"status={event.evaluator_status} gate={decision.reason_code}"
                )
                val_append = None
            if val_status is False:
                if self._embedded_full_run_update_result_md:
                    self._append_embedded_full_run_to_result_md(safety_result)
                if val_append is not None:
                    eid, ocall, rtxt = val_append
                    self._append_mlebench_validation_to_result_md(
                        eid, ocall, False, rtxt,
                    )
                self._embedded_full_run_done = False
                um = (
                    "[Guard] MLE-bench submission validation FAILED after full-run.\n"
                    "Fix `solution.py` so `submission.csv` passes format checks, "
                    "then run quick-test again.\n\n"
                    "=== Validation error ===\n"
                    f"{val_text}\n\n"
                    f"{_cost_report}"
                )
                self.memory.add_message(Message.user_message(um))
                self._log_info(
                    "[mlebench-validate] invalid — continuing session for fix",
                )
                return None

            if self._embedded_full_run_update_result_md:
                self._append_embedded_full_run_to_result_md(safety_result)
            if val_status is True and val_append is not None:
                eid, ocall, rtxt = val_append
                self._append_mlebench_validation_to_result_md(eid, ocall, True, rtxt)

            val_suffix = val_text if val_text else ""
            mv = safety_result.metric_value
            msg = (
                f"[ScienceAgent] Embedded full-run finished: exit={ec} "
                f"wall={safety_result.wall_sec:.1f}s metric={mv} "
                f"({safety_result.metric_name}). See [full-run] in logs/interaction.log."
                f"\n\n{_cost_report}"
            )
            full_msg = msg + val_suffix
            self.memory.add_message(Message.assistant_message(full_msg))
            self._log_info("[assistant] %s", truncate_for_interaction_log(full_msg))
            if self._ui:
                self._ui.render_agent_reply(full_msg)
            else:
                print(full_msg, file=sys.stdout)
            return full_msg

        self._embedded_full_run_done = False
        stderr = safety_result.stderr or ""
        um = _embedded_failure_user_message(stderr, ec) + f"\n\n{_cost_report}"
        self.memory.add_message(Message.user_message(um))
        self._log_info(
            "[embedded-full-run] exit=%s; continuing session so the model can fix solution.py",
            ec,
        )
        return None
