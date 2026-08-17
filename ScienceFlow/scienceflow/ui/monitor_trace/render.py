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

"""Render monitor trace data as a self-contained HTML document."""

from __future__ import annotations

import datetime as dt
import html
import json
from typing import Any

from scienceflow.ui.monitor_trace.assets import MONITOR_TRACE_CSS, monitor_trace_script
from scienceflow.ui.monitor_trace.models import (
    MonitorTraceReport,
    TaskTrace,
    TraceEvent,
    TracePoint,
)


def render_monitor_trace_html(report: MonitorTraceReport, *, refresh_sec: float | None = None) -> str:
    refresh = ""
    if refresh_sec is not None and refresh_sec > 0:
        refresh = f'<meta http-equiv="refresh" content="{int(refresh_sec)}">\n'
    task_sections = "\n".join(_render_task(task, idx) for idx, task in enumerate(report.tasks))
    embedded = _json_for_script(report.to_dict())
    script = monitor_trace_script()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}<title>ScienceFlow Monitor Trace</title>
<style>
{MONITOR_TRACE_CSS}
</style>
</head>
<body>
<main>
<div class="top">
  <div>
    <h1>ScienceFlow Monitor Trace</h1>
    <div class="meta">Generated {html.escape(report.generated_at_utc)} · {len(report.tasks)} task(s)</div>
  </div>
  <div class="meta">{html.escape(report.manifest_path)}</div>
</div>
{task_sections or '<div class="task">No tasks found in manifest.</div>'}
<script type="application/json" id="monitor-trace-data">{embedded}</script>
<script>{script}</script>
</main>
</body>
</html>
"""


def _render_task(task: TaskTrace, index: int) -> str:
    summary = task.summary
    chart = _render_chart(task, index)
    leaderboard = _render_leaderboard(task)
    return f"""<section class="task" data-task-index="{index}">
  <div class="task-head">
    <div>
      <h2>{html.escape(task.exp_id)}</h2>
      <div class="meta">{html.escape(task.status)} · {html.escape(task.metric_name)} · {'lower' if task.lower_is_better else 'higher'} is better</div>
    </div>
    <div class="meta">{html.escape(task.gpu_list or 'gpu n/a')}</div>
  </div>
  <div class="summary">
    {_stat('Stages', summary.get('stage_count'))}
    {_stat('Best valid', summary.get('best_valid_metric'))}
    {_stat('Best raw', summary.get('best_raw_metric'))}
    {_stat('ESTRA', _estra_text(summary))}
    {_stat('Remaining', _fmt_sec(summary.get('remaining_sec')))}
    {_stat('LLM', summary.get('llm_display_text') or 'n/a')}
  </div>
  {leaderboard}
  <div class="chart">{chart}</div>
  <div class="legend"><span><i class="swatch"></i>stage metric points</span><span><i class="line-swatch"></i>{html.escape(str(summary.get('best_kind') or 'best'))}</span><span>vertical markers: estra / guard / resume</span></div>
</section>"""


def _render_leaderboard(task: TaskTrace) -> str:
    leaderboard = task.leaderboard
    if leaderboard is None:
        return ""
    entries = sorted(leaderboard.entries, key=lambda entry: entry.rank)
    visible = entries[: leaderboard.display_limit]
    rows = "\n".join(_leaderboard_row(entry) for entry in visible)
    direction = "lower" if (leaderboard.lower_is_better if leaderboard.lower_is_better is not None else task.lower_is_better) else "higher"
    source = _leaderboard_source_link(leaderboard.source_url)
    rank_text = _rank_text(leaderboard.estimated_rank, len(entries))
    return f"""<div class=\"leaderboard\">
  <div class=\"leaderboard-head\">
    <div>
      <div class=\"leaderboard-title\">{html.escape(leaderboard.title)}</div>
      <div class=\"meta\">{html.escape(leaderboard.metric_name)} · {direction} is better · showing top {len(visible)} of {len(entries)} public entries</div>
    </div>
    {source}
  </div>
  <div class=\"leaderboard-stats\">
    {_mini_stat('Our best', _score_with_source(leaderboard.our_score, leaderboard.our_score_source))}
    {_mini_stat('Est. rank', rank_text)}
    {_mini_stat('Gap to #1', _fmt_gap(leaderboard.gap_to_rank1))}
    {_mini_stat('Gap to next', _fmt_gap(leaderboard.gap_to_next))}
  </div>
  <table class=\"leaderboard-table\">
    <thead><tr><th>Rank</th><th>User</th><th>Score</th><th>Submitted</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _leaderboard_row(entry: Any) -> str:
    submitted = entry.submitted_at[:10] if entry.submitted_at else ""
    return (
        "<tr>"
        f"<td>#{entry.rank}</td>"
        f"<td>{html.escape(entry.user)}</td>"
        f"<td>{html.escape(_fmt_score(entry.score))}</td>"
        f"<td>{html.escape(submitted or 'n/a')}</td>"
        "</tr>"
    )


def _leaderboard_source_link(url: str) -> str:
    if not url:
        return '<span class="meta">source n/a</span>'
    safe = html.escape(url, quote=True)
    return f'<a class="source-link" href="{safe}">source</a>'


def _mini_stat(label: str, value: Any) -> str:
    return f'<div class="mini-stat"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(_fmt_value(value))}</div></div>'


def _score_with_source(score: Any, source: str) -> str:
    if score is None:
        return "n/a"
    suffix = f" ({source})" if source else ""
    return f"{_fmt_score(score)}{suffix}"


def _rank_text(rank: int | None, known_entries: int) -> str:
    if rank is None:
        return "n/a"
    return f"#{rank} / {known_entries + 1} incl. ours"


def _fmt_score(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _fmt_value(value)
    if abs(num) >= 100:
        return f"{num:,.3f}"
    return _fmt_value(num)


def _fmt_gap(value: Any) -> str:
    try:
        gap = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(gap) < 1e-9:
        return "0"
    if gap > 0:
        return f"behind {_fmt_score(gap)}"
    return f"ahead {_fmt_score(abs(gap))}"


def _stat(label: str, value: Any) -> str:
    return f'<div class="stat"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(_fmt_value(value))}</div></div>'


def _estra_text(summary: dict[str, Any]) -> str:
    decisions = summary.get("estra_decision_count")
    switches = summary.get("estra_switch_count")
    if decisions is None and switches is None:
        return "n/a"
    return f"{decisions or 0} decision / {switches or 0} switch"


def _render_chart(task: TaskTrace, index: int) -> str:
    if not task.points:
        return '<div class="meta">No stage metric points yet.</div>'
    scale = _Scale(task.points, task.best_points)
    grid = _render_grid(scale)
    points = "\n".join(_point_svg(p, scale) for p in task.points)
    best = _polyline_svg(task.best_points, scale)
    events = "\n".join(_event_svg(e, scale) for e in task.events if e.x_value is not None)
    labels = _axis_labels(scale, task)
    controls = _chart_controls(scale, index)
    clip_id = f"plot-clip-{index}"
    return f"""{controls}
<svg data-task-index="{index}" viewBox="0 0 {scale.width} {scale.height}" role="img" aria-label="{html.escape(task.exp_id)} metric trace">
<defs><clipPath id="{clip_id}"><rect x="58" y="18" width="894" height="232"/></clipPath></defs>
{grid}
<g clip-path="url(#{clip_id})">
{events}
{best}
{points}
</g>
{labels}
</svg>"""


def _chart_controls(scale: "_Scale", index: int) -> str:
    return f"""<div class="chart-controls" data-task-index="{index}">
  <span class="label">Y range</span>
  <label>min <input class="y-min" data-task-index="{index}" type="number" step="any" placeholder="{html.escape(_fmt_value(scale.y_min))}"></label>
  <label>max <input class="y-max" data-task-index="{index}" type="number" step="any" placeholder="{html.escape(_fmt_value(scale.y_max))}"></label>
  <button type="button" data-range-action="apply" data-task-index="{index}">Apply</button>
  <button type="button" data-range-action="valid" data-task-index="{index}">Valid range</button>
  <button type="button" data-range-action="all" data-task-index="{index}">All range</button>
</div>"""


def _render_grid(scale: "_Scale") -> str:
    x0, y0, x1, y1 = scale.plot_box
    lines = [
        f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>',
    ]
    for idx in range(1, 4):
        y = y0 + ((y1 - y0) * idx / 4)
        lines.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
    return "\n".join(lines)


def _point_svg(point: TracePoint, scale: "_Scale") -> str:
    cls = "point" if point.valid_comparable else "point invalid"
    title = html.escape(
        f"{point.x_label} · {point.worker_id} {point.stage_id} · "
        f"metric={point.metric:.6g} · {point.validity}/{point.metric_validity}"
    )
    return (
        f'<circle class="{cls}" cx="{scale.x(point.x_value):.1f}" '
        f'cy="{scale.y(point.metric):.1f}" r="3.4"><title>{title}</title></circle>'
    )


def _polyline_svg(points: list[Any], scale: "_Scale") -> str:
    if len(points) < 2:
        return ""
    coords = " ".join(f"{scale.x(p.x_value):.1f},{scale.y(p.metric):.1f}" for p in points)
    return f'<polyline class="best" points="{coords}"/>'


def _event_svg(event: TraceEvent, scale: "_Scale") -> str:
    x = scale.x(float(event.x_value))
    y0, y1 = scale.plot_box[1], scale.plot_box[3]
    cls = f"event-{html.escape(event.kind)}"
    title = html.escape(f"{event.x_label} · {event.label}")
    return (
        f'<line class="{cls}" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" '
        f'y2="{y1}" stroke-width="1.4" opacity=".45"><title>{title}</title></line>'
    )


def _axis_labels(scale: "_Scale", task: TaskTrace) -> str:
    x0, y0, _, y1 = scale.plot_box
    y_min = _fmt_value(scale.y_min)
    y_max = _fmt_value(scale.y_max)
    labels = [
        f'<text class="tiny y-max-label" x="{x0}" y="{y0 - 7}">{html.escape(y_max)}</text>',
        f'<text class="tiny y-min-label" x="{x0}" y="{y1 - 4}">{html.escape(y_min)}</text>',
        *(_x_tick_labels(scale, task)),
    ]
    return "\n".join(labels)


def _x_tick_labels(scale: "_Scale", task: TaskTrace, *, count: int = 5) -> list[str]:
    _, _, _, y1 = scale.plot_box
    if count <= 1 or scale.x_min == scale.x_max:
        values = [scale.x_min]
    else:
        values = [
            scale.x_min + ((scale.x_max - scale.x_min) * idx / (count - 1))
            for idx in range(count)
        ]
    out: list[str] = []
    for value in values:
        label = _fmt_x_tick(value, task.x_mode)
        out.append(
            f'<text class="tiny x-tick" text-anchor="middle" x="{scale.x(value):.1f}" '
            f'y="{y1 + 24}">{html.escape(label)}</text>'
        )
    axis_name = "time UTC" if task.x_mode == "wall_clock" else task.x_mode
    out.append(f'<text class="tiny axis-name" x="58" y="{scale.height - 8}">{html.escape(axis_name)}</text>')
    return out


def _fmt_x_tick(value: float, x_mode: str) -> str:
    if x_mode == "wall_clock":
        try:
            return dt.datetime.fromtimestamp(value, tz=dt.UTC).strftime("%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return _fmt_value(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return _fmt_value(value)


class _Scale:
    width = 980
    height = 300
    plot_box = (58.0, 18.0, 952.0, 250.0)

    def __init__(self, points: list[TracePoint], best_points: list[Any]) -> None:
        xs = [p.x_value for p in points] + [p.x_value for p in best_points]
        ys = [p.metric for p in points] + [p.metric for p in best_points]
        self.x_min, self.x_max = _bounds(xs)
        self.y_min, self.y_max = _padded_bounds(ys)

    def x(self, value: float) -> float:
        x0, _, x1, _ = self.plot_box
        if self.x_min == self.x_max:
            return (x0 + x1) / 2
        return x0 + ((value - self.x_min) / (self.x_max - self.x_min)) * (x1 - x0)

    def y(self, value: float) -> float:
        _, y0, _, y1 = self.plot_box
        if self.y_min == self.y_max:
            return (y0 + y1) / 2
        return y1 - ((value - self.y_min) / (self.y_max - self.y_min)) * (y1 - y0)


def _bounds(values: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return 0.0, 1.0
    return min(clean), max(clean)


def _padded_bounds(values: list[float]) -> tuple[float, float]:
    low, high = _bounds(values)
    if low == high:
        pad = abs(low) * 0.05 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _fmt_sec(value: Any) -> str:
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


def _fmt_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _json_for_script(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
