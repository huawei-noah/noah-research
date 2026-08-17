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

"""Parse quick-test stdout and estimate full-run wall time (linear extrapolation)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from scienceflow.utils.node_paths import find_node_context_path

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
_RE_EPOCH = re.compile(
    r"\[EPOCH\]\s*(\d+)/(\d+)\s+time=([\d.]+)s",
    re.IGNORECASE,
)
_RE_BASH_PREFIX_ELAPSED = re.compile(
    r"^\[exit=[^,\]]+,\s*([\d.]+)s\]",
    re.IGNORECASE,
)

# Values >= this are almost always Unix timestamps mistakenly printed as "time=…s" (not wall seconds).
_EPOCH_TIME_UNIX_TS_FLOOR = 1e9
# Quick-test rows are tiny; a per-epoch wall time beyond this is not credible for extrapolation.
_MAX_EPOCH_SEC_QUICK_TEST_PLAUSIBLE = 86400.0  # 24h


def is_plausible_epoch_duration_sec(
    seconds: float,
    *,
    strict_quick_test: bool = False,
) -> bool:
    """Rule-based filter for ``[EPOCH] … time=…s`` values.

    Rejects mistaken ``time.time()`` (Unix ts ~1e9+) printed as duration. Optionally
    rejects huge per-epoch times when parsing **quick-test** stdout for extrapolation.
    The Safety full-run watchdog uses ``strict_quick_test=False`` so only ts-like values
    are dropped.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return False
    if s < 0 or s != s:  # nan
        return False
    if s >= _EPOCH_TIME_UNIX_TS_FLOOR:
        return False
    if strict_quick_test and s > _MAX_EPOCH_SEC_QUICK_TEST_PLAUSIBLE:
        return False
    return True


@dataclass
class QuickTestParseResult:
    """Structured fields extracted from quick-test ``solution.py`` stdout."""

    total_train_rows: int | None = None
    quick_test_rows: int | None = None
    data_load_sec: float | None = None
    data_rows: int | None = None
    epoch_times_sec: list[float] | None = None
    num_epochs: int | None = None


def strip_bash_tool_prefix(block: str) -> str:
    """Remove ``[exit=..., X.Xs]`` header from :class:`BashTool` output if present."""
    text = (block or "").strip()
    if text.startswith("[") and "]" in text[:48]:
        idx = text.find("]")
        if idx != -1 and idx < 40:
            rest = text[idx + 1 :].lstrip()
            if rest.startswith("\n"):
                return rest[1:]
    return text


def parse_bash_tool_elapsed_sec(block: str) -> float | None:
    """Parse elapsed seconds from BashTool header ``[exit=..., Xs]``."""
    text = (block or "").strip()
    m = _RE_BASH_PREFIX_ELAPSED.match(text)
    if not m:
        return None
    try:
        sec = float(m.group(1))
    except (TypeError, ValueError, IndexError):
        return None
    if sec < 0 or sec != sec:
        return None
    return sec


def parse_quick_test_stdout(stdout: str) -> QuickTestParseResult:
    """Best-effort parse of ``[DATASET]``, ``[DATA]``, ``[EPOCH]`` lines."""
    out = QuickTestParseResult()
    m = _RE_DATASET.search(stdout)
    if m:
        try:
            out.total_train_rows = int(m.group(1))
            out.quick_test_rows = int(m.group(2))
        except (ValueError, IndexError):
            pass
    if out.total_train_rows is None:
        m2 = _RE_DATASET_SIMPLE.search(stdout)
        if m2:
            try:
                out.total_train_rows = int(m2.group(1))
            except (ValueError, IndexError):
                pass

    m = _RE_DATA.search(stdout)
    if m:
        try:
            out.data_load_sec = float(m.group(1))
            out.data_rows = int(m.group(2))
        except (ValueError, IndexError):
            pass

    epoch_times: list[float] = []
    last_total_epochs: int | None = None
    for line in stdout.splitlines():
        em = _RE_EPOCH.search(line)
        if em:
            try:
                _cur = int(em.group(1))
                last_total_epochs = int(em.group(2))
                t_ep = float(em.group(3))
                if is_plausible_epoch_duration_sec(t_ep, strict_quick_test=True):
                    epoch_times.append(t_ep)
            except (ValueError, IndexError):
                pass
    if epoch_times:
        out.epoch_times_sec = epoch_times
        out.num_epochs = last_total_epochs
    return out


def load_dataset_total_rows_from_context(workspace: Path) -> int | None:
    """Read ``dataset_total_rows`` or ``train_rows`` from hidden/root context if present."""
    p = find_node_context_path(Path(workspace))
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("dataset_total_rows", "train_rows", "total_train_rows"):
        v = data.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


def estimate_full_run_seconds(
    parsed: QuickTestParseResult,
    *,
    quick_test_rows: int,
    total_rows: int | None,
    separate_fixed_epoch_overhead: bool = True,
) -> float | None:
    """Estimate full-data run time: scale data load and per-epoch work by ``total/quick_test``.

    When ``separate_fixed_epoch_overhead`` is True and there are at least two ``[EPOCH]`` samples,
    the first epoch's excess over the mean of later epochs is treated as **unscaled** fixed overhead
    (CUDA init, first-batch compile). Remaining per-epoch time is scaled by ``total_rows/qt_rows``.
    This reduces gross over-estimation on CV workloads where the first epoch is much slower.

    Returns ``None`` if not enough signal (missing epochs or non-positive scale).
    """
    if not parsed.epoch_times_sec:
        return None
    n_ep = parsed.num_epochs
    if not n_ep or n_ep < 1:
        return None

    ets = list(parsed.epoch_times_sec)
    if separate_fixed_epoch_overhead and len(ets) >= 2:
        rest_avg = sum(ets[1:]) / float(len(ets) - 1)
        fixed_epoch = max(0.0, float(ets[0]) - rest_avg)
        avg_epoch = rest_avg
    else:
        fixed_epoch = 0.0
        avg_epoch = sum(ets) / float(len(ets))
    qt_rows = parsed.quick_test_rows or parsed.data_rows or quick_test_rows
    if qt_rows is not None and qt_rows < 1:
        qt_rows = quick_test_rows

    tr = total_rows
    if tr is None:
        tr = parsed.total_train_rows
    if tr is None or tr < 1:
        return None
    if qt_rows is None or qt_rows < 1:
        qt_rows = quick_test_rows

    scale = float(tr) / float(qt_rows)
    if scale < 1.0:
        scale = 1.0

    data_part = 0.0
    if parsed.data_load_sec is not None and parsed.data_load_sec >= 0:
        data_part = parsed.data_load_sec * scale

    # One copy of fixed first-epoch overhead in the full run; per-epoch train work scales with data size.
    epoch_part = fixed_epoch + avg_epoch * scale * float(n_ep)
    return data_part + epoch_part


def estimate_full_run_seconds_from_wall(
    *,
    quick_test_wall_sec: float,
    quick_test_rows: int,
    total_rows: int | None,
) -> float | None:
    """Fallback estimate from total quick-test wall time when [EPOCH] lines are absent."""
    try:
        wall = float(quick_test_wall_sec)
    except (TypeError, ValueError):
        return None
    if wall <= 0:
        return None
    tr = int(total_rows) if total_rows is not None else 0
    if tr < 1:
        return None
    qr = int(quick_test_rows)
    if qr < 1:
        return None
    scale = float(tr) / float(qr)
    if scale < 1.0:
        scale = 1.0
    return wall * scale


def parse_epoch_line(line: str) -> tuple[int, int, float] | None:
    """Return (current_epoch, total_epochs, time_sec) from one ``[EPOCH]`` line."""
    em = _RE_EPOCH.search(line)
    if not em:
        return None
    try:
        t_ep = float(em.group(3))
        if not is_plausible_epoch_duration_sec(t_ep, strict_quick_test=False):
            return None
        return int(em.group(1)), int(em.group(2)), t_ep
    except (ValueError, IndexError):
        return None


def format_extrapolation_block(
    *,
    est_sec: float,
    budget_sec: float,
    total_rows: int,
    quick_test_rows: int,
    parsed: QuickTestParseResult,
) -> str:
    """User message when estimated full run exceeds budget."""
    return (
        "[SYSTEM PERFORMANCE ALERT] Quick-test output suggests the **full-data** run may exceed "
        f"the time budget.\n\n"
        f"- Estimated full run (linear extrapolation): **{est_sec:.0f}s** (~{est_sec / 3600:.2f} h)\n"
        f"- Budget (max allowed estimate): **{budget_sec:.0f}s**\n"
        f"- Assumed scale: total_train_rows≈{total_rows} vs quick-test rows≈{quick_test_rows}\n"
        f"- Parsed: data_load={parsed.data_load_sec}, "
        f"epochs={parsed.num_epochs}, epoch_times={parsed.epoch_times_sec}\n\n"
        "You MUST optimize `solution.py` for **speed** before re-running quick-test:\n"
        "- Data: chunked reads, caching, memory-map, fewer repeated full scans\n"
        "- DataLoader: `num_workers`, `persistent_workers`, `pin_memory`\n"
        "- Training: fewer epochs for debugging, AMP, larger batch, lighter model\n\n"
        "Then run a quick-test again and re-check runtime.\n"
    )
