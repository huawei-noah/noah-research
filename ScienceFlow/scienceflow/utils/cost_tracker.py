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

"""Full-run cost summary: parse [EPOCH] lines from training stdout and produce a compact report.

Used by the embedded full-run path to inject a cost-awareness message into agent memory
after the full training pass completes (success or failure), so subsequent improve sessions
can use accurate wall-time data when planning iterations.

Phase 2 of the LLM-Autonomous Experiment Control plan.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scienceflow.utils.quick_test_perf import parse_epoch_line

if TYPE_CHECKING:
    pass

# Match quick_test_perf conventions; applied per-line (last line wins).
_RE_DATASET = re.compile(
    r"\[DATASET\].*?total_train_rows\s*=\s*(\d+).*?quick_test_rows\s*=\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_RE_DATASET_SIMPLE = re.compile(
    r"\[DATASET\].*?total_train_rows\s*=\s*(\d+)",
    re.IGNORECASE,
)
_RE_DATA = re.compile(
    r"\[DATA\].*?loaded in ([\d.]+)s.*?rows=(\d+)",
    re.IGNORECASE,
)
_RE_SCALE_KV = re.compile(r"\b([a-zA-Z_][\w]*)\s*=\s*(\d+)\b")


@dataclass
class FullRunCostSummary:
    """Parsed cost data from one full ``python3 solution.py`` run."""

    wall_sec: float = 0.0
    exit_code: int | None = None
    metric_value: float | None = None
    metric_name: str = "unknown"
    lower_is_better: bool | None = None
    node_exec_budget_sec: int | None = None

    # [EPOCH] line stats
    epoch_times_sec: list[float] = field(default_factory=list)
    num_epochs_declared: int | None = None  # total from last [EPOCH] N/M line
    num_epochs_seen: int = 0              # how many [EPOCH] lines were parsed

    # Scale hints from stdout ([DATASET] / [DATA] / [SCALE]); optional
    total_train_rows: int | None = None
    trained_rows: int | None = None
    batch_size: int | None = None
    epochs_from_scale: int | None = None  # from ``[SCALE] epochs=`` only

    @property
    def avg_epoch_sec(self) -> float | None:
        if not self.epoch_times_sec:
            return None
        return sum(self.epoch_times_sec) / len(self.epoch_times_sec)

    @property
    def median_epoch_sec(self) -> float | None:
        if not self.epoch_times_sec:
            return None
        return statistics.median(self.epoch_times_sec)

    @property
    def total_epoch_sec(self) -> float:
        return sum(self.epoch_times_sec)


def _parse_scale_line_body(body: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in _RE_SCALE_KV.finditer(body or ""):
        key = m.group(1).lower()
        try:
            out[key] = int(m.group(2))
        except ValueError:
            continue
    return out


def _apply_stdout_scale_signals(stdout: str, summary: FullRunCostSummary) -> None:
    """Fill scale fields from [DATASET], [DATA], [SCALE] lines (last occurrence wins per family)."""
    last_dataset_total: int | None = None
    last_data_rows: int | None = None
    scale_merged: dict[str, int] = {}

    for line in (stdout or "").splitlines():
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("[DATASET]"):
            m = _RE_DATASET.search(s)
            if m:
                try:
                    last_dataset_total = int(m.group(1))
                except (ValueError, IndexError):
                    pass
            else:
                m2 = _RE_DATASET_SIMPLE.search(s)
                if m2:
                    try:
                        last_dataset_total = int(m2.group(1))
                    except (ValueError, IndexError):
                        pass
        elif u.startswith("[DATA]"):
            m = _RE_DATA.match(s)
            if m:
                try:
                    last_data_rows = int(m.group(2))
                except (ValueError, IndexError):
                    pass
        elif u.startswith("[SCALE]"):
            m_sc = re.match(r"^\[SCALE\]\s*(.*)$", s, re.IGNORECASE | re.DOTALL)
            rest = m_sc.group(1) if m_sc else ""
            scale_merged.update(_parse_scale_line_body(rest))

    summary.total_train_rows = scale_merged.get("total_train_rows", last_dataset_total)
    summary.trained_rows = scale_merged.get("trained_rows", last_data_rows)
    summary.batch_size = scale_merged.get("batch_size")
    if "epochs" in scale_merged:
        summary.epochs_from_scale = scale_merged["epochs"]


def parse_fullrun_cost_summary(
    stdout: str,
    *,
    wall_sec: float = 0.0,
    exit_code: int | None = None,
    metric_value: float | None = None,
    metric_name: str = "unknown",
    lower_is_better: bool | None = None,
    node_exec_budget_sec: int | None = None,
) -> FullRunCostSummary:
    """Parse [EPOCH] lines from full-run *stdout* and combine with safety metadata."""
    summary = FullRunCostSummary(
        wall_sec=wall_sec,
        exit_code=exit_code,
        metric_value=metric_value,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        node_exec_budget_sec=node_exec_budget_sec,
    )
    last_total: int | None = None
    for line in (stdout or "").splitlines():
        parsed = parse_epoch_line(line)
        if parsed is None:
            continue
        _cur_ep, total_ep, t_ep = parsed
        summary.epoch_times_sec.append(t_ep)
        summary.num_epochs_seen += 1
        last_total = total_ep
    summary.num_epochs_declared = last_total
    _apply_stdout_scale_signals(stdout, summary)
    return summary


def _fmt_compact_int(n: int) -> str:
    if n >= 1_000_000:
        x = n / 1_000_000
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return f"{s}M"
    if n >= 1000:
        x = n / 1000
        s = f"{x:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return str(n)


def _format_scale_line(summary: FullRunCostSummary) -> str | None:
    """Emit ``Scale:`` only when stdout carried [DATASET]/[DATA]/[SCALE] hints — not from [EPOCH] alone."""
    parts: list[str] = []
    tr = summary.trained_rows
    tt = summary.total_train_rows
    bs = summary.batch_size
    has_row_or_batch = tr is not None or tt is not None or bs is not None
    has_scale_only_epochs = summary.epochs_from_scale is not None
    if not has_row_or_batch and not has_scale_only_epochs:
        return None

    if tr is not None and tt is not None and tt > 0:
        pct = 100.0 * tr / tt
        parts.append(
            f"trained_rows={_fmt_compact_int(tr)}/{_fmt_compact_int(tt)} ({pct:.1f}%)"
        )
    elif tr is not None:
        parts.append(f"trained_rows={_fmt_compact_int(tr)}")
    elif tt is not None:
        parts.append(f"total_train_rows={_fmt_compact_int(tt)}")

    if bs is not None:
        parts.append(f"batch_size={bs}")

    if summary.epochs_from_scale is not None:
        parts.append(f"epochs={summary.epochs_from_scale}")
    elif has_row_or_batch and summary.num_epochs_declared is not None:
        parts.append(f"epochs={summary.num_epochs_declared}")

    if not parts:
        return None
    return "  Scale: " + " | ".join(parts)


def format_fullrun_cost_report(summary: FullRunCostSummary) -> str:
    """One-paragraph cost report injected into agent memory after a full run."""
    status = "succeeded" if summary.exit_code == 0 else f"failed (exit_code={summary.exit_code})"
    lines = [
        f"[Full-run cost report] status={status} | wall_time={summary.wall_sec:.1f}s",
    ]

    if summary.metric_value is not None:
        lb = "↓" if summary.lower_is_better else "↑"
        lines[0] += f" | {summary.metric_name}={summary.metric_value} ({lb}better)"

    scale_line = _format_scale_line(summary)
    if scale_line:
        lines.append(scale_line)

    if summary.epoch_times_sec:
        avg = summary.avg_epoch_sec
        med = summary.median_epoch_sec
        total_ep = summary.total_epoch_sec
        n_seen = summary.num_epochs_seen
        n_decl = summary.num_epochs_declared
        ep_str = f"{n_seen}" if n_decl is None else f"{n_seen}/{n_decl}"
        lines.append(
            f"  Epoch stats: epochs_seen={ep_str} | "
            f"avg={avg:.1f}s | median={med:.1f}s | total_epoch_time={total_ep:.1f}s"
        )
        lines.append(
            f"  Wall vs epoch-sum gap: {max(0.0, summary.wall_sec - total_ep):.1f}s "
            f"(data load + overhead)"
        )
    else:
        lines.append("  No [EPOCH] lines found in stdout — cannot report per-epoch timing.")

    nb = summary.node_exec_budget_sec
    if nb is not None and nb > 0 and summary.wall_sec > nb * 0.8:
        ratio = summary.wall_sec / nb
        lines.append(
            f"  ACTION REQUIRED: wall_time={summary.wall_sec:.0f}s exceeds node budget "
            f"{nb}s (ratio={ratio:.1f}x). "
            "Reduce model complexity: fewer iterations/estimators, bounded thread count, "
            "or switch to a lighter model."
        )
    else:
        lines.append(
            "  Use this data to plan the next iteration: if wall_time approaches your remaining "
            "budget, consider fewer epochs, a lighter model, or a different architecture."
        )
    return "\n".join(lines)
