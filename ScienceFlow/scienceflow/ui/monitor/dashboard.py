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

"""Rich Live dashboard for lnr monitor_state.json."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from scienceflow.ui.monitor.helpers import (
    _build_gpu_resource_block,
    _fmt_running_node_cell,
    _fmt_sec,
    _fmt_tokens_row,
    _gpu_display_from_assignment,
    _gpu_info,
    _load_parallel_state,
    _load_state,
    _make_bar,
    _multi_status_and_progress,
    _parallel_done_for_row,
    _state_stale_age_seconds,
)


class MonitorDashboard:
    """Rich Live dashboard that tails ``monitor_state.json`` (single or multi-task)."""

    def __init__(
        self,
        state_file: Path | None = None,
        refresh_interval: float = 2.0,
        task_entries: list[tuple[str, str, Path]] | None = None,
    ) -> None:
        self._multi = bool(task_entries)
        self.task_entries = task_entries or []
        self.state_file = Path(state_file) if state_file else Path(".")
        self.refresh_interval = max(0.5, float(refresh_interval))
        self._console = Console()

    def run(self) -> None:
        """Block until KeyboardInterrupt, refreshing the display every tick."""
        with Live(
            self._render({}) if not self._multi else self._render_multi([]),
            console=self._console,
            auto_refresh=False,
            screen=False,
        ) as live:
            try:
                while True:
                    if self._multi:
                        rows = [
                            (e, g, _load_state(sf), sf)
                            for e, g, sf in self.task_entries
                        ]
                        live.update(self._render_multi(rows), refresh=True)
                    else:
                        live.update(self._render(_load_state(self.state_file)), refresh=True)
                    time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                pass

    async def run_async(self) -> None:
        """Async-friendly variant for embedding inside an event loop."""
        with Live(
            self._render({}) if not self._multi else self._render_multi([]),
            console=self._console,
            auto_refresh=False,
            screen=False,
        ) as live:
            try:
                while True:
                    if self._multi:
                        rows = [
                            (e, g, _load_state(sf), sf)
                            for e, g, sf in self.task_entries
                        ]
                        live.update(self._render_multi(rows), refresh=True)
                    else:
                        live.update(self._render(_load_state(self.state_file)), refresh=True)
                    await asyncio.sleep(self.refresh_interval)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass

    def _render_multi(self, rows: list[tuple[str, str, dict[str, Any], Path]]) -> Group:
        """Parallel mode: one row per task; GPU telemetry in a panel below the table."""
        compact = len(rows) >= 6 or self._console.width < 220
        table = Table(
            title="[bold cyan]LNR Parallel Monitor[/bold cyan]",
            show_header=True,
            show_lines=True,
            header_style="bold magenta",
            border_style="dim",
            expand=True,
        )
        if compact:
            table.add_column("Task", style="cyan", max_width=18, overflow="fold")
            table.add_column("GPU", style="dim", max_width=4)
            table.add_column("Status", max_width=18)
            table.add_column("Progress", max_width=18)
            table.add_column("Best", max_width=24)
            table.add_column("Stage", max_width=13)
            table.add_column("ESTRA", max_width=13)
            table.add_column("proc/job", min_width=18, max_width=26, overflow="fold")
            table.add_column("Resource", max_width=16)
            table.add_column("Token", max_width=20, overflow="fold")
        else:
            table.add_column("Task", style="cyan", max_width=28, overflow="fold")
            table.add_column("GPU", style="dim", max_width=5)
            table.add_column("Status", max_width=20)
            table.add_column("Progress", max_width=20)
            table.add_column("Best", max_width=28)
            table.add_column("Stage", max_width=16)
            table.add_column("ESTRA", max_width=14)
            table.add_column("proc/job", min_width=22, max_width=30, overflow="fold")
            table.add_column("Resource", max_width=18)
            table.add_column("Token", max_width=24, overflow="fold")
            table.add_column("TTFT/TPOT", max_width=14)
        if not rows:
            table.add_row(*(["[dim]waiting...[/dim]"] + [""] * (9 if compact else 10)))
            return Group(table, Text(""), _build_gpu_resource_block(_gpu_info()))
        for run_id, gpu_list, state, sf in rows:
            logs_dir = sf.parent
            gpu_cell = _gpu_display_from_assignment(gpu_list, logs_dir)
            if not state:
                row = [
                    self._lnr_task_cell(run_id, state, compact=compact),
                    gpu_cell,
                    "[dim]no state[/dim]",
                    "", "", "", "", "", "", "",
                ]
                if not compact:
                    row.append("")
                table.add_row(*row)
                continue
            if state.get("monitor_kind") == "lnr_task":
                row = [
                    self._lnr_task_cell(run_id, state, compact=compact),
                    gpu_cell,
                    self._status_cell(state),
                    self._lnr_budget_cell(state),
                    self._lnr_best_cell(state),
                    self._lnr_stage_cell(state),
                    self._lnr_estra_cell(state),
                    self._lnr_active_jobs_cell(state),
                    self._lnr_resource_cell(state),
                    self._lnr_cache_cell(state),
                ]
                if not compact:
                    row.append(self._lnr_latency_cell(state))
                table.add_row(*row)
                continue

            status = state.get("status", "unknown")
            pstate = _load_parallel_state(sf.parent)
            elapsed = float(state.get("elapsed_sec", 0))
            total = float(state.get("total_sec", 0))
            remaining = max(0.0, total - elapsed)
            if total <= 0 and "remaining_sec" in state:
                remaining = max(0.0, float(state.get("remaining_sec", 0)))
            pr = float(state.get("progress_ratio", 0))
            if pr <= 0 and state.get("progress_pct") is not None:
                try:
                    pr = float(state["progress_pct"]) / 100.0
                except (TypeError, ValueError):
                    pass
            st, force_pr = _multi_status_and_progress(run_id, status, state, pstate)
            if force_pr:
                pr = 1.0
            bar = _make_bar(pr, width=12)
            bc = f"{_fmt_sec(elapsed)}/{_fmt_sec(remaining)} [{bar}] {pr * 100:.0f}%"
            sm = state.get("best_shallow_metric", state.get("best_metric"))
            sv = f"{sm:.6f}" if isinstance(sm, float) else str(sm or "-")
            dm = state.get("best_deep_metric")
            dv = f"{dm:.6f}" if isinstance(dm, float) else str(dm or "-")
            row = [
                self._lnr_task_cell(run_id, state, compact=compact),
                gpu_cell,
                st,
                bc,
                sv,
                f"N={state.get('total_nodes', 0)}",
                "",
                _fmt_running_node_cell(state),
                dv,
                _fmt_tokens_row(state),
            ]
            if not compact:
                row.append("")
            table.add_row(*row)
        return Group(table, Text(""), _build_gpu_resource_block(_gpu_info()))

    def _render(self, state: dict) -> Group:
        """Build main table plus host resource panel from *state*."""
        table = Table(
            title="[bold cyan]LNR Monitor[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Field", style="cyan", no_wrap=True, width=30)
        table.add_column("Value", style="white")

        if not state:
            table.add_row("[dim]waiting for monitor_state.json…[/dim]", "")
            return Group(table, Text(""), _build_gpu_resource_block(_gpu_info()))

        if state.get("monitor_kind") == "lnr_task":
            return self._render_lnr_task(state)

        task_name = (state.get("task_name") or state.get("exp_id") or "unknown").strip() or "unknown"
        lower = state.get("lower_is_better", True)
        direction = "↓ lower=better" if lower else "↑ higher=better"
        table.add_row("Task", f"[bold]{task_name}[/bold]  [dim]{direction}[/dim]")
        if (self.state_file.parent / "gpu_assignment.json").is_file():
            table.add_row(
                "GPU",
                _gpu_display_from_assignment("", self.state_file.parent),
            )

        elapsed = float(state.get("elapsed_sec", 0))
        total = float(state.get("total_sec", 0))
        remaining = max(0.0, total - elapsed)
        if total <= 0 and "remaining_sec" in state:
            remaining = max(0.0, float(state.get("remaining_sec", 0)))
        phase = state.get("phase", "explore")
        status = state.get("status", "unknown")
        _terminal_statuses = {"early_stop", "budget_expired", "steps_completed"}
        is_finished = status in _terminal_statuses
        pstate = _load_parallel_state(self.state_file.parent)
        parallel_done = _parallel_done_for_row(None, pstate)

        progress_ratio = float(state.get("progress_ratio", 0))
        if progress_ratio <= 0 and state.get("progress_pct") is not None:
            try:
                progress_ratio = float(state["progress_pct"]) / 100.0
            except (TypeError, ValueError):
                pass
        if is_finished or parallel_done:
            progress_ratio = 1.0

        bar = _make_bar(progress_ratio, width=30)
        table.add_row(
            "Budget",
            f"{_fmt_sec(elapsed)} elapsed / {_fmt_sec(remaining)} remaining  "
            f"[{bar}] {progress_ratio * 100:.0f}%",
        )
        table.add_row("Phase", f"[bold yellow]{phase}[/bold yellow]")
        if is_finished:
            status_label = {
                "budget_expired": "BUDGET EXPIRED",
                "early_stop": "EARLY STOPPED",
                "steps_completed": "ALL STEPS DONE",
            }.get(status, status.upper())
            table.add_row("Status", f"[bold green]FINISHED ({status_label})[/bold green]")
        elif parallel_done:
            ps = str(pstate.get("status", ""))
            table.add_row(
                "Status",
                f"[bold green]FINISHED (DONE)[/bold green]  [dim]parallel {ps}[/dim]",
            )
        else:
            stale_age = _state_stale_age_seconds(state)
            if stale_age is not None and stale_age > 60:
                table.add_row(
                    "Status",
                    f"[bold red]STALE ({int(stale_age)}s ago, process may have exited)[/bold red]",
                )
            else:
                table.add_row("Status", "[bold blue]RUNNING[/bold blue]")

        table.add_row("Running nodes", _fmt_running_node_cell(state))

        shallow_metric = state.get("best_shallow_metric", state.get("best_metric"))
        shallow_time = state.get("shallow_exec_time_avg")
        total_nodes = state.get("total_nodes", 0)
        shallow_val = (
            f"{shallow_metric:.6f}" if isinstance(shallow_metric, float) else str(shallow_metric or "—")
        )
        shallow_time_str = (
            f"  [dim](avg {shallow_time:.0f}s)[/dim]" if isinstance(shallow_time, (int, float)) else ""
        )
        table.add_row("Shallow best", f"{shallow_val}{shallow_time_str}  nodes={total_nodes}")

        deep_metric = state.get("best_deep_metric")
        deep_time = state.get("deep_exec_time_avg")
        active_deep = state.get("active_deep_tasks", 0)
        deep_val = (
            f"{deep_metric:.6f}" if isinstance(deep_metric, float) else str(deep_metric or "—")
        )
        deep_time_str = (
            f"  [dim](avg {deep_time:.0f}s)[/dim]" if isinstance(deep_time, (int, float)) else ""
        )
        table.add_row("Deep best", f"{deep_val}{deep_time_str}  active={active_deep}")

        workers = state.get("workers", [])
        if workers:
            worker_lines = "  ".join(
                f"[dim]w{w.get('id', i)}:[/dim]{w.get('status', '?')}"
                for i, w in enumerate(workers)
            )
            table.add_row("Workers", worker_lines or "—")

        ts = state.get("timestamp", "")
        table.add_row("[dim]State file[/dim]", f"[dim]{self.state_file}  {ts}[/dim]")

        return Group(table, Text(""), _build_gpu_resource_block(_gpu_info()))

    def _render_lnr_task(self, state: dict[str, Any]) -> Group:
        """Render a task state built from current LNR artifacts."""
        table = Table(
            title="[bold cyan]LNR Task Monitor[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Field", style="cyan", no_wrap=True, width=28)
        table.add_column("Value", style="white")

        task_name = (state.get("task_name") or state.get("exp_id") or "unknown").strip()
        lower = bool(state.get("lower_is_better", True))
        direction = "↓ lower=better" if lower else "↑ higher=better"
        table.add_row("Task", f"[bold]{task_name}[/bold]  [dim]{direction}[/dim]")
        table.add_row("Status", self._status_cell(state))
        table.add_row("Progress", self._lnr_budget_cell(state))
        table.add_row("Best", self._lnr_best_cell(state))
        table.add_row("Stages", self._lnr_stage_cell(state))
        table.add_row("ESTRA", self._lnr_estra_cell(state))
        table.add_row("proc/job", self._lnr_active_jobs_cell(state))
        table.add_row("Resource", self._lnr_resource_cell(state))
        table.add_row("LLM", self._lnr_llm_cell(state))
        table.add_row("Token", self._lnr_cache_cell(state))
        table.add_row("TTFT/TPOT", self._lnr_latency_cell(state))
        table.add_row("State source", f"[dim]{state.get('task_root', '')}  {state.get('timestamp', '')}[/dim]")
        return Group(table, Text(""), _build_gpu_resource_block(_gpu_info()))

    def _status_cell(self, state: dict[str, Any]) -> str:
        stale_age = _state_stale_age_seconds(state)
        worker_line = self._worker_status_cell(state) or self._latest_workers_cell(state)
        suffix = f"\n{worker_line}" if worker_line else ""
        status = str(state.get("status") or "unknown")
        if status in {"finished", "failed", "early_stop", "budget_expired", "budget_done", "steps_completed", "timeout", "skipped"}:
            return f"{status}{suffix}"
        if status == "run_resume":
            return f"[blue]run[/blue] [dim](resume)[/dim]{suffix}"
        if stale_age is not None and stale_age > 180:
            return f"[bold red]STALE ({int(stale_age)}s)[/bold red]{suffix}"
        if status in {"run", "running"}:
            return f"[blue]run[/blue]{suffix}"
        return f"{status}{suffix}"

    def _worker_status_cell(self, state: dict[str, Any]) -> str:
        states = state.get("worker_states")
        if not isinstance(states, list) or not states:
            return ""
        parts = []
        for item in sorted(states, key=self._worker_status_sort_key)[:4]:
            if not isinstance(item, dict):
                continue
            worker = str(item.get("worker_id") or "?").upper()
            label = str(item.get("label") or item.get("kind") or "RUN")
            runtime = item.get("runtime_sec")
            runtime_text = f"/{self._short_sec(runtime)}" if isinstance(runtime, (int, float)) and runtime > 0 else ""
            parts.append(f"{worker}:{label}{runtime_text}")
        suffix = " …" if len(states) > 4 else ""
        return " ".join(parts) + suffix if parts else ""

    def _worker_status_sort_key(self, item: dict[str, Any]) -> tuple[int, float, str]:
        worker = str(item.get("worker_id") or "?").upper()
        label = str(item.get("label") or item.get("kind") or "RUN").upper()
        runtime = item.get("runtime_sec")
        runtime_sec = float(runtime) if isinstance(runtime, (int, float)) else 0.0
        long_running = label in {"LRUN", "LTRN"} or runtime_sec >= 900.0
        group = 0 if long_running else (1 if label == "WAIT" else 2)
        return group, -runtime_sec, worker

    def _latest_workers_cell(self, state: dict[str, Any]) -> str:
        latest = state.get("latest_stage_by_worker")
        if not isinstance(latest, dict) or not latest:
            return ""
        parts = []
        for worker, stage in sorted(latest.items())[:4]:
            if not isinstance(stage, dict):
                continue
            cid = self._short_candidate(stage.get("candidate_id") or "")
            metric = self._metric_text(stage.get("metric_value"))
            evaluator = str(stage.get("evaluator_backend") or "").strip()
            status = str(stage.get("evaluator_status") or "").strip()
            eval_text = f"/{evaluator}:{status}" if evaluator and status else (f"/{evaluator}" if evaluator else "")
            parts.append(f"{worker}:{cid}/{metric}{eval_text}")
        suffix = " …" if len(latest) > 4 else ""
        return " ".join(parts) + suffix

    def _lnr_budget_cell(self, state: dict[str, Any]) -> str:
        elapsed = state.get("elapsed_sec")
        remaining = state.get("remaining_sec")
        total = state.get("total_sec")
        ratio = state.get("progress_ratio")
        if isinstance(elapsed, (int, float)) and isinstance(remaining, (int, float)):
            if isinstance(total, (int, float)) and float(total) > 0:
                total_sec = float(total)
            else:
                total_sec = max(0.0, float(elapsed)) + max(0.0, float(remaining))
            if isinstance(ratio, (int, float)):
                r = float(ratio)
            elif total_sec > 0:
                r = max(0.0, min(1.0, float(elapsed) / total_sec))
            else:
                r = 0.0
            bar = _make_bar(r, width=12)
            pct = f"{r * 100:.1f}%"
            return f"{self._short_sec(elapsed)}/{self._short_sec(total_sec)} {pct}\nrem {self._short_sec(remaining)} {bar}"
        return "[dim]from task logs[/dim]"

    def _lnr_best_cell(self, state: dict[str, Any]) -> str:
        raw = self._metric_text(state.get("best_raw_metric"))
        valid = self._metric_text(state.get("best_valid_metric"))
        raw_id = state.get("best_raw_candidate") or "—"
        valid_id = state.get("best_valid_candidate") or "—"
        raw_v = state.get("best_raw_validity") or "—"
        valid_v = state.get("best_valid_validity") or "—"
        raw_id = self._short_candidate(raw_id)
        valid_id = self._short_candidate(valid_id)
        raw_v = self._short_validity(raw_v)
        valid_v = self._short_validity(valid_v)
        return f"valid {valid} {valid_id}/{valid_v}\ncand {raw} {raw_id}/{raw_v}"

    def _lnr_stage_cell(self, state: dict[str, Any]) -> str:
        total = int(state.get("stage_count") or 0)
        candidates = int(state.get("candidate_stage_count") or total)
        ready = int(state.get("ready_stage_count") or 0)
        eligible = int(state.get("eligible_stage_count") or 0)
        if candidates != total:
            return f"stage={total}\ncand={ready}/{candidates}\nelig={eligible}"
        return f"stage={total}\nready={ready}\nelig={eligible}"

    def _lnr_estra_cell(self, state: dict[str, Any]) -> str:
        decisions = int(state.get("estra_decision_count") or 0)
        raw_decisions = int(state.get("estra_raw_decision_count") or decisions)
        repeat_decisions = int(state.get("estra_repeat_decision_count") or max(0, raw_decisions - decisions))
        keep = int(state.get("estra_continue_count") or 0)
        redirect = int(state.get("estra_redirect_count") or 0)
        switch = int(state.get("estra_switch_count") or 0)
        context = int(state.get("estra_context_trigger_count") or 0)
        invalid = int(state.get("estra_invalid_count") or 0)
        text_only = int(state.get("estra_text_only_trigger_count") or 0)
        if decisions <= 0 and not any((context, invalid, text_only, raw_decisions)):
            return "[dim]—[/dim]"
        first_parts = []
        if keep:
            first_parts.append(f"k={keep}")
        if redirect:
            first_parts.append(f"r={redirect}")
        if switch:
            first_parts.append(f"sw={switch}")
        if not first_parts and decisions:
            first_parts.append(f"dec={decisions}")
        detail_parts = []
        if raw_decisions > decisions:
            detail_parts.append(f"raw={raw_decisions}")
        if repeat_decisions > 0:
            detail_parts.append(f"d={repeat_decisions}")
        trigger_parts = []
        if text_only:
            trigger_parts.append(f"txt={text_only}")
        if context:
            trigger_parts.append(f"ctx={context}")
        if invalid:
            trigger_parts.append(f"bad={invalid}")
        lines = [" ".join(parts) for parts in (first_parts, detail_parts, trigger_parts) if parts]
        return "\n".join(lines) or "[dim]—[/dim]"

    def _lnr_task_cell(self, run_id: str, state: dict[str, Any], *, compact: bool = False) -> str:
        if compact:
            base = run_id.split("-w", 1)[0]
            label = base[:16] + ("…" if len(base) > 16 else "")
        else:
            label = run_id[:24] + ("…" if len(run_id) > 24 else "")
        llm = self._lnr_llm_cell(state, compact=compact)
        if llm == "[dim]—[/dim]":
            return label
        return f"{label}\n[dim]{llm}[/dim]"

    def _lnr_llm_cell(self, state: dict[str, Any], *, compact: bool = False) -> str:
        cfg = state.get("llm_config")
        if not isinstance(cfg, dict):
            return "[dim]—[/dim]"
        models_by_role: dict[str, str] = {}
        for role in ("code", "feedback"):
            item = cfg.get(role)
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            models = item.get("models")
            model_count = len(models) if isinstance(models, list) else 0
            if not model and model_count:
                model = str(models[0] or "").strip()
            if model:
                if model_count > 1:
                    model = f"{model}+{model_count}"
                models_by_role[role] = model
        if not models_by_role:
            return "[dim]—[/dim]"
        if compact:
            code_model = self._short_model_name(models_by_role.get("code", ""))
            feedback_model = self._short_model_name(models_by_role.get("feedback", ""))
            if code_model and code_model == feedback_model:
                return f"m={code_model}"
            parts = []
            if code_model:
                parts.append(f"c={code_model}")
            if feedback_model:
                parts.append(f"f={feedback_model}")
            return " ".join(parts) if parts else "[dim]—[/dim]"
        parts = [f"{role}={models_by_role[role]}" for role in ("code", "feedback") if role in models_by_role]
        return "\n".join(parts) if parts else "[dim]—[/dim]"

    @staticmethod
    def _short_model_name(model: str) -> str:
        for prefix in ("deepseek-v4-", "deepseek-", "glm-"):
            if model.startswith(prefix):
                model = model[len(prefix):]
                break
        return model[:12] + ("…" if len(model) > 12 else "")

    def _lnr_resource_cell(self, state: dict[str, Any]) -> str:
        counts = state.get("resource_counts") if isinstance(state.get("resource_counts"), dict) else {}
        current = state.get("resource_current") if isinstance(state.get("resource_current"), dict) else {}
        raw_outcomes = state.get("resource_outcomes") if isinstance(state.get("resource_outcomes"), dict) else {}
        outcomes = (
            state.get("resource_actionable_outcomes")
            if isinstance(state.get("resource_actionable_outcomes"), dict)
            else {key: value for key, value in raw_outcomes.items() if key != "NO_ACTION"}
        )
        outcome_parts = []
        for key, label in (("KILL", "kill_req"), ("TIMEBOX", "tb")):
            if outcomes.get(key):
                outcome_parts.append(f"{label}={outcomes[key]}")
        current_parts = []
        if current.get("active_waiter_count"):
            current_parts.append(f"wait_now={current['active_waiter_count']}")
        if current.get("active_lease_count"):
            current_parts.append(f"lease_now={current['active_lease_count']}")
        event_parts = []
        for key, label in (
            ("resource_kill_executed", "kill_exec"),
            ("managed_resource_wait_offered", "wait_total"),
            ("admission_pending", "pend_total"),
        ):
            if counts.get(key):
                event_parts.append(f"{label}={counts[key]}")
        lines = []
        if outcome_parts:
            lines.append(" ".join(outcome_parts[:3]))
        if current_parts:
            lines.append(" ".join(current_parts[:3]))
        if event_parts:
            lines.append(" ".join(event_parts[:3]))
        heartbeat_count = counts.get("resource_monitor_heartbeat") or counts.get("progress_heartbeat")
        if not lines and heartbeat_count:
            lines.append(f"hb={heartbeat_count}")
        return "\n".join(lines) if lines else "[dim]—[/dim]"

    def _lnr_active_jobs_cell(self, state: dict[str, Any]) -> str:
        tracked_raw = state.get("active_jobs") if isinstance(state.get("active_jobs"), list) else []
        tracked = [job for job in tracked_raw if isinstance(job, dict)]
        running = int(state.get("running_process_count") or 0)
        if not tracked and running <= 0:
            return "[dim]—[/dim]"
        lines = [f"proc={running} job={len(tracked)}"]
        display_jobs = self._active_jobs_by_worker(tracked)
        for job in sorted(display_jobs, key=self._active_job_sort_key)[:3]:
            line = self._active_job_line(job)
            if line:
                lines.append(line)
        if len(tracked) > 3:
            lines.append("[dim]...[/dim]")
        return "\n".join(lines)

    def _active_jobs_by_worker(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for job in jobs:
            worker = str(job.get("worker_id") or "?").upper()
            previous = latest.get(worker)
            if previous is None or self._active_job_seen_at(job) >= self._active_job_seen_at(previous):
                latest[worker] = job
        return list(latest.values())

    def _active_job_seen_at(self, job: dict[str, Any]) -> float:
        heartbeat_at = self._number(job.get("heartbeat_observed_at")) or 0.0
        observed_at = self._number(job.get("observed_at")) or 0.0
        return max(heartbeat_at, observed_at)

    def _active_job_sort_key(self, job: dict[str, Any]) -> tuple[str, float]:
        worker = str(job.get("worker_id") or "?").upper()
        runtime = self._number(job.get("runtime_sec")) or 0.0
        return worker, -runtime

    def _active_job_line(self, job: dict[str, Any]) -> str:
        worker = self._safe_token(str(job.get("worker_id") or "?").upper(), max_len=4)
        label = self._active_job_label(job)
        heartbeat_at = self._number(job.get("heartbeat_observed_at"))
        observed_at = self._number(job.get("observed_at"))
        age_at = heartbeat_at if heartbeat_at is not None else observed_at
        age = max(0.0, time.time() - age_at) if age_at else None
        stale = heartbeat_at is not None and age is not None and age > 180.0
        if stale:
            label_text = "[bold red]STALL[/bold red]"
        else:
            style = self._active_job_style(label)
            label_text = f"[{style}]{label}[/{style}]"
        parts = [worker, label_text]
        if stale and label not in {"RUN", "STALL"}:
            parts.append(f"[dim]{label}[/dim]")

        progress = self._active_job_progress(job)
        metric = self._active_job_metric(job)
        if progress:
            parts.append(progress)
        if metric:
            parts.append(metric)
        if not progress and not metric:
            stdout_lines = self._int(job.get("stdout_lines"))
            if stdout_lines:
                parts.append(f"out={stdout_lines}")
        if stale:
            age_prefix = "stale_hb" if heartbeat_at is not None else "stale_obs"
        else:
            age_prefix = "hb" if heartbeat_at is not None else "obs"
        age_token = self._active_job_age_token(age, prefix=age_prefix)
        if age_token:
            parts.append(age_token)
        return " ".join(parts)

    def _active_job_label(self, job: dict[str, Any]) -> str:
        text = " ".join(
            str(part or "")
            for part in (
                job.get("heartbeat_phase"),
                job.get("known_stage"),
                job.get("command_excerpt"),
                job.get("entrypoint"),
                job.get("resource_class"),
                job.get("progress_signal"),
            )
        ).lower()
        runtime = self._number(job.get("runtime_sec")) or 0.0
        if any(token in text for token in ("valid", "eval", "score", "oof")):
            return "VAL"
        if any(token in text for token in ("train", "fit", "trainer", "epoch", "heavy_gpu_train", "heavy_cpu_train")):
            return "LTRN" if runtime >= 900.0 else "TRN"
        if any(token in text for token in ("predict", "infer", "inference", "submission", "submit", "ensemble", "test.py")):
            return "INF"
        if self._number(job.get("metric_value")) is not None:
            return "VAL"
        return "RUN"

    def _active_job_style(self, label: str) -> str:
        return {
            "TRN": "bright_blue",
            "LTRN": "blue",
            "VAL": "cyan",
            "EVAL": "cyan",
            "INF": "magenta",
            "PRED": "magenta",
            "WAIT": "yellow",
            "IDLE": "dim",
            "STALL": "bold red",
            "ERR": "bold red",
            "KILL": "bold red",
        }.get(label, "white")

    def _active_job_progress(self, job: dict[str, Any]) -> str:
        current = self._number(job.get("progress_current"))
        total = self._number(job.get("progress_total"))
        prefix = "~" if job.get("progress_advanced") is False else ""
        if current is not None and total is not None and total > 0:
            pct = max(0.0, min(999.0, current / total * 100.0))
            return f"{prefix}{pct:.0f}%"
        if current is not None:
            unit = self._safe_token(str(job.get("progress_unit") or "n"), max_len=5)
            return f"{prefix}{unit}={self._compact_number(current)}"
        return ""

    def _active_job_metric(self, job: dict[str, Any]) -> str:
        value = self._number(job.get("metric_value"))
        if value is None:
            return ""
        name = self._metric_alias(str(job.get("metric_name") or "m"))
        return f"{name}={self._compact_number(value)}"

    def _metric_alias(self, name: str) -> str:
        raw = name.strip().lower().replace(" ", "_")
        if "kendall" in raw or raw in {"kt", "tau"}:
            return "kt"
        if "rmse" in raw:
            return "rmse"
        if "auc" in raw:
            return "auc"
        if "score" in raw:
            return "score"
        return self._safe_token(raw or "m", max_len=5)

    def _active_job_age_token(self, age: float | None, *, prefix: str) -> str:
        if age is None:
            return ""
        if age < 60.0:
            return f"{prefix}={int(age)}s"
        if age < 3600.0:
            return f"{prefix}={int(age // 60)}m"
        return f"{prefix}={int(age // 3600)}h"

    def _safe_token(self, value: str, *, max_len: int) -> str:
        safe = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", "."})
        return (safe[:max_len] or "?")

    def _compact_number(self, value: float) -> str:
        abs_value = abs(value)
        if abs_value < 1.0:
            text = f"{value:.3f}"
            if text.startswith("0."):
                return text[1:]
            if text.startswith("-0."):
                return "-" + text[2:]
            return text
        if abs_value < 10.0:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.3g}"

    def _number(self, value: object) -> float | None:
        try:
            if value in (None, ""):
                return None
            return float(str(value))
        except (TypeError, ValueError):
            return None

    def _int(self, value: object) -> int:
        number = self._number(value)
        return int(number) if number is not None else 0

    def _lnr_cache_cell(self, state: dict[str, Any]) -> str:
        return _fmt_tokens_row(state)

    def _lnr_latency_cell(self, state: dict[str, Any]) -> str:
        ttft = state.get("max_ttft_sec")
        tpot = state.get("max_tpot_ms")
        if not isinstance(ttft, (int, float)) and not isinstance(tpot, (int, float)):
            return "[dim]—[/dim]"
        ttft_text = f"{float(ttft):.1f}s" if isinstance(ttft, (int, float)) else "—"
        tpot_text = f"{float(tpot):.0f}ms" if isinstance(tpot, (int, float)) else "—"
        return f"ttft {ttft_text}\ntpot {tpot_text}"

    def _counter_cell(self, value: object) -> str:
        if not isinstance(value, dict) or not value:
            return "[dim]—[/dim]"
        return " ".join(f"{k}={v}" for k, v in sorted(value.items(), key=lambda item: str(item[0]))[:6])

    def _short_candidate(self, candidate_id: object) -> str:
        text = str(candidate_id or "").strip()
        if not text or text == "—":
            return "—"
        parts = text.split(":")
        if len(parts) >= 3:
            return ":".join(parts[-2:])
        return text[:12] + ("…" if len(text) > 12 else "")

    def _short_validity(self, value: object) -> str:
        text = str(value or "").strip().lower()
        return {"high": "h", "medium": "m", "low": "l"}.get(text, text[:1] or "—")

    def _short_sec(self, sec: object) -> str:
        try:
            value = max(0, int(float(sec)))
        except (TypeError, ValueError):
            return "—"
        h, rem = divmod(value, 3600)
        m, _ = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}"
        return f"{m}m"

    def _metric_text(self, value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return "—"
