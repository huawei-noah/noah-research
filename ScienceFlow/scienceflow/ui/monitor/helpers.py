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

"""State loading, GPU display, and formatting helpers for the LNR monitor."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

from scienceflow.utils.llm_cost import format_usd
from typing import Any

import yaml
from rich.panel import Panel
from rich.text import Text

from scienceflow.core.parallel_runner import (
    _manifest_task_exp_id,
    _manifest_task_run_id,
    resolve_manifest_task_workspace,
)
from scienceflow.ui.monitor.lnr_state import load_lnr_task_state


def _load_state(state_file: Path) -> dict[str, Any]:
    """Load monitor state.

    Prefer legacy ``logs/monitor_state.json``. If that file is absent, adapt
    current LNR task artifacts under ``task_logs/`` into a monitor state.
    """
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _load_lnr_state_from_monitor_path(state_file)


def _load_lnr_state_from_monitor_path(state_file: Path) -> dict[str, Any]:
    """Resolve ``.../<task>/logs/monitor_state.json`` to ``.../<task>``."""
    path = Path(state_file)
    if path.name != "monitor_state.json" or path.parent.name != "logs":
        return {}
    return load_lnr_task_state(path.parent.parent)


def _load_parallel_state(logs_dir: Path) -> dict[str, Any]:
    """Load ParallelRunner ``logs/state.json`` next to ``monitor_state.json``."""
    try:
        return json.loads((logs_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _gpu_display_from_assignment(manifest_gpu: str, logs_dir: Path) -> str:
    """Prefer ``logs/gpu_assignment.json`` (physical IDs) over manifest YAML (e.g. ``auto:2``)."""
    p = logs_dir / "gpu_assignment.json"
    if not p.is_file():
        return manifest_gpu.strip() or "—"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return manifest_gpu.strip() or "—"
    resolved = data.get("resolved_gpu")
    rs = str(resolved).strip() if resolved is not None else ""
    if not rs:
        return manifest_gpu.strip() or "—"
    if data.get("auto"):
        return f"{rs} [dim](auto)[/dim]"
    return rs


def _parallel_status_label(pstate: dict[str, Any]) -> str:
    status = str(pstate.get("status") or "").strip().lower()
    error = str(pstate.get("error") or "").strip().lower()
    if status == "timeout" and error.startswith("exceeded "):
        return "budget_done"
    return status


def _parallel_done_for_row(run_id: str | None, pstate: dict[str, Any]) -> bool:
    """True if parallel layer finished, exhausted its budget, or was skipped."""
    if not pstate:
        return False
    pr = pstate.get("run_id")
    if run_id and pr is not None and str(pr).strip() != "" and str(pr) != str(run_id):
        return False
    return _parallel_status_label(pstate) in ("completed", "budget_done", "timeout", "skipped")


def _multi_status_and_progress(
    run_id: str,
    lnr_status: str,
    mon_state: dict[str, Any],
    pstate: dict[str, Any],
) -> tuple[str, bool]:
    """Return (Rich status cell, whether to force progress bar to 100%)."""
    term = {"early_stop", "budget_expired", "steps_completed"}
    if lnr_status in term:
        return (f"[green]{lnr_status}[/green]", True)
    if _parallel_done_for_row(run_id, pstate):
        ps = _parallel_status_label(pstate)
        return (f"[green]DONE[/green] [dim](parallel {ps})[/dim]", True)
    return (_status_cell_non_terminal(lnr_status, mon_state), False)


def parse_monitor_manifest(manifest_path: Path | str) -> list[tuple[str, str, Path]]:
    """Parse parallel manifest: (run_id, gpu_list, monitor_state.json path)."""
    path = Path(manifest_path).expanduser().resolve()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    defaults = data.get("defaults") or {}
    out: list[tuple[str, str, Path]] = []
    for idx, t in enumerate(data.get("tasks") or []):
        if not isinstance(t, dict):
            continue
        exp_id = _manifest_task_exp_id(t, idx)
        run_id = _manifest_task_run_id(t, idx, exp_id)
        try:
            ws = resolve_manifest_task_workspace(defaults, t, idx, exp_id)
        except ValueError:
            continue
        gpu = str(t.get("gpu_list", defaults.get("gpu_list", ""))).strip()
        out.append((run_id, gpu, Path(ws).expanduser() / "logs" / "monitor_state.json"))
    return out


def _gpu_footer_compact_lines(gpus: list[dict], per_row: int = 4) -> list[str]:
    if not gpus:
        return ["[dim]nvidia-smi n/a[/dim]"]
    lines = []
    for i in range(0, len(gpus), per_row):
        chunk = gpus[i : i + per_row]
        parts = []
        for g in chunk:
            try:
                mu = float(g["mem_used_mb"])
                mt = float(g["mem_total_mb"])
                mu_g = mu / 1024.0
                mt_g = mt / 1024.0
            except (TypeError, ValueError):
                mu_g = mt_g = 0.0
            parts.append(
                f"GPU{g['index']}: {g['util_pct']}% {mu_g:.0f}G/{mt_g:.0f}G",
            )
        lines.append("  ".join(parts))
    return lines


def _short_node_id(nid: str, prefix: int = 8) -> str:
    """Truncate hex node id for display."""
    if not nid:
        return ""
    return f"{nid[:prefix]}…" if len(nid) > prefix else nid


def _fmt_running_node_cell(state: dict[str, Any]) -> str:
    """Format C/From/D (child workspace / expansion anchor / deep progress)."""
    exec_ids = state.get("running_shallow_node_ids") or []
    parent_ids = state.get("running_shallow_parent_node_ids") or []
    deep = state.get("running_deep_node_ids") or []
    if not isinstance(exec_ids, list):
        exec_ids = []
    if not isinstance(parent_ids, list):
        parent_ids = []
    if not isinstance(deep, list):
        deep = []
    try:
        deep_done = int(state.get("deep_completed_count") or 0)
    except (TypeError, ValueError):
        deep_done = 0
    in_flight = len(deep)
    parts: list[str] = []
    if exec_ids:
        parts.append(
            "C:" + ",".join(_short_node_id(str(x)) for x in exec_ids),
        )
    if parent_ids:
        parts.append(
            "From:" + ",".join(_short_node_id(str(x)) for x in parent_ids),
        )
    if deep_done > 0 or in_flight > 0:
        d_cell = f"D:{deep_done}"
        if in_flight:
            d_cell += f"+{in_flight}"
        parts.append(d_cell)
    return "  ".join(parts) if parts else "[dim]—[/dim]"


def _fmt_tokens_row(state: dict) -> str:
    tin = state.get("total_tokens_in")
    tout = state.get("total_tokens_out")
    tcached = state.get("total_tokens_cached")
    cache_rate = state.get("llm_cache_rate")
    calls = state.get("total_llm_calls")
    cost_usd = state.get("total_llm_cost_usd")
    if tin is None and tout is None:
        return "—"
    try:
        ti = int(tin or 0)
        to = int(tout or 0)
        tcache = int(tcached or 0)
        cc = int(calls or 0)
    except (TypeError, ValueError):
        return "—"
    cache_suffix = ""
    try:
        rate = float(cache_rate) if cache_rate is not None else (
            float(tcache) / float(ti) if ti > 0 else None
        )
    except (TypeError, ValueError, ZeroDivisionError):
        rate = None
    cost_text = ""
    if cost_usd is not None:
        try:
            cost_text = f"cost={format_usd(float(cost_usd))}"
        except (TypeError, ValueError):
            cost_text = ""
    second_parts = []
    if rate is not None:
        second_parts.append(f"cache={rate:.2f}")
    if cost_text:
        second_parts.append(cost_text)
    first = f"{ti / 1_000_000.0:.2f}M/{to / 1_000_000.0:.2f}M c={cc}"
    if not second_parts:
        return first
    return first + "\n" + " ".join(second_parts)


def _build_gpu_resource_block(gpus: list[dict]) -> Panel:
    """Host telemetry below the main table (not mixed with time budget columns)."""
    content = "\n".join(_gpu_footer_compact_lines(gpus, per_row=4))
    return Panel(
        Text.from_markup(content),
        title="[dim]Host resources (nvidia-smi)[/dim]",
        border_style="dim",
        expand=False,
    )


def _gpu_info() -> list[dict]:
    """Query nvidia-smi for per-GPU utilisation; return [] if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "util_pct": parts[2],
                        "mem_used_mb": parts[3],
                        "mem_total_mb": parts[4],
                    }
                )
        return gpus
    except Exception:
        return []


def _state_stale_age_seconds(state: dict[str, Any]) -> float | None:
    """Seconds since *state*['timestamp'] if parseable, else None."""
    ts_str = state.get("timestamp", "")
    if not ts_str:
        return None
    try:
        ts_parsed = datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
        return (datetime.datetime.now() - ts_parsed).total_seconds()
    except ValueError:
        return None


def _status_cell_non_terminal(status: str, state: dict[str, Any]) -> str:
    """Rich markup for non-finished rows: STALE if timestamp is old, else RUNNING."""
    stale_age = _state_stale_age_seconds(state)
    if stale_age is not None and stale_age > 60:
        return (
            f"[bold red]STALE ({int(stale_age)}s)[/bold red]"
        )
    if status == "running":
        return "[blue]run[/blue]"
    return str(status)[:14]


def _fmt_sec(sec: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _make_bar(ratio: float, width: int = 20) -> str:
    """Return a Unicode progress bar string of *width* characters."""
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)
