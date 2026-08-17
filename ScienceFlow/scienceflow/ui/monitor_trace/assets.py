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

"""Static CSS and JavaScript assets for monitor trace HTML."""

from __future__ import annotations

MONITOR_TRACE_CSS = r"""
:root { color-scheme: light; --ink:#172033; --muted:#64748b; --line:#d8dee8; --panel:#ffffff; --bg:#f6f8fb; --accent:#1f7a5b; --point:#4f6f9f; --guard:#c2410c; --estra:#7c3aed; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
main { max-width:1180px; margin:0 auto; padding:22px 18px 36px; }
h1 { font-size:22px; margin:0 0 4px; letter-spacing:0; }
h2 { font-size:18px; margin:0; letter-spacing:0; }
button { font:inherit; }
.top { display:flex; justify-content:space-between; gap:16px; align-items:flex-end; margin-bottom:18px; }
.meta { color:var(--muted); font-size:13px; }
.task { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; margin:14px 0; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.task-head { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; flex-wrap:wrap; margin-bottom:10px; }
.summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:8px; margin:10px 0 12px; }
.stat { border:1px solid var(--line); border-radius:6px; padding:8px 9px; min-height:48px; }
.label { color:var(--muted); font-size:12px; }
.value { font-size:15px; margin-top:3px; overflow-wrap:anywhere; white-space:pre-line; }
.leaderboard { border:1px solid var(--line); border-radius:6px; background:#f8fafc; padding:10px; margin:4px 0 12px; }
.leaderboard-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap; margin-bottom:8px; }
.leaderboard-title { font-size:14px; font-weight:650; }
.source-link { color:#1d4ed8; font-size:12px; text-decoration:none; }
.source-link:hover { text-decoration:underline; }
.leaderboard-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:8px; margin-bottom:8px; }
.mini-stat { border:1px solid #e2e8f0; border-radius:6px; background:#fff; padding:7px 8px; min-height:42px; }
.leaderboard-table { width:100%; border-collapse:collapse; font-size:12px; }
.leaderboard-table th, .leaderboard-table td { border-top:1px solid #e2e8f0; padding:6px 8px; text-align:left; }
.leaderboard-table th { color:var(--muted); font-weight:600; }
.chart { overflow-x:auto; border-top:1px solid var(--line); padding-top:10px; }
.chart-controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:0 0 8px; color:var(--muted); font-size:12px; }
.chart-controls input { width:104px; height:28px; border:1px solid var(--line); border-radius:5px; padding:2px 6px; color:var(--ink); background:#fff; }
.chart-controls button { height:30px; border:1px solid var(--line); border-radius:5px; padding:0 9px; background:#fff; color:var(--ink); cursor:pointer; }
.chart-controls button:hover { border-color:#9aa7b8; }
svg { width:100%; min-width:760px; height:auto; display:block; }
.axis { stroke:#94a3b8; stroke-width:1; }
.grid { stroke:#e5eaf1; stroke-width:1; }
.point { fill:var(--point); opacity:.6; }
.point.invalid { fill:#9ca3af; }
.best { fill:none; stroke:var(--accent); stroke-width:2.4; }
.event-estra { stroke:var(--estra); }
.event-guard { stroke:var(--guard); }
.event-resume { stroke:#2563eb; }
.tiny { font-size:11px; fill:#64748b; }
.legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:12px; margin-top:6px; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px; opacity:.6; background:var(--point); }
.line-swatch { display:inline-block; width:18px; height:2px; margin-right:5px; vertical-align:3px; background:var(--accent); }
"""


def monitor_trace_script() -> str:
    return r"""
(() => {
  const raw = document.getElementById('monitor-trace-data');
  if (!raw) return;
  let report;
  try {
    report = JSON.parse(raw.textContent || '{}');
  } catch (error) {
    return;
  }
  const PLOT = { y0: 18, y1: 250 };
  const finite = (value) => Number.isFinite(Number(value));
  const number = (value) => Number(value);
  const formatMetric = (value) => {
    if (!finite(value)) return 'n/a';
    const num = number(value);
    if (Math.abs(num) >= 1000 || Math.abs(num) < 0.001) return num.toExponential(3);
    return Number(num.toPrecision(7)).toString();
  };
  const clean = (values) => values.map(number).filter(Number.isFinite);
  const bounds = (values) => {
    const vals = clean(values);
    if (!vals.length) return [0, 1];
    return [Math.min(...vals), Math.max(...vals)];
  };
  const padded = (values) => {
    let [low, high] = bounds(values);
    if (low === high) {
      const pad = Math.abs(low) * 0.05 || 1;
      return [low - pad, high + pad];
    }
    const pad = (high - low) * 0.08;
    return [low - pad, high + pad];
  };
  const key = (task, suffix) => `scienceflow.monitor_trace.${task.run_id || task.exp_id}.${suffix}`;
  const taskByIndex = (index) => (report.tasks || [])[Number(index)] || null;
  const yValues = (task) => [
    ...(task.points || []).map((p) => p.metric),
    ...(task.best_points || []).map((p) => p.metric),
  ];
  const validYValues = (task) => {
    const values = (task.points || []).filter((p) => p.valid_comparable).map((p) => p.metric);
    return values.length ? values : yValues(task);
  };
  const selectedYRange = (index, task) => {
    const controls = document.querySelector(`.chart-controls[data-task-index="${index}"]`);
    const minInput = controls?.querySelector('.y-min');
    const maxInput = controls?.querySelector('.y-max');
    const auto = padded(yValues(task));
    let yMin = finite(minInput?.value) ? number(minInput.value) : auto[0];
    let yMax = finite(maxInput?.value) ? number(maxInput.value) : auto[1];
    if (!(yMax > yMin)) {
      const center = (yMin + yMax) / 2 || auto[0];
      const pad = Math.abs(center) * 0.05 || 1;
      yMin = center - pad;
      yMax = center + pad;
    }
    return [yMin, yMax];
  };
  const yScale = (yMin, yMax) => (value) => {
    if (yMin === yMax) return (PLOT.y0 + PLOT.y1) / 2;
    return PLOT.y1 - ((number(value) - yMin) / (yMax - yMin)) * (PLOT.y1 - PLOT.y0);
  };
  const parsePolylineXs = (polyline) => String(polyline?.getAttribute('points') || '')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((pair) => pair.split(',', 1)[0]);
  const updateChartY = (index) => {
    const task = taskByIndex(index);
    const svg = document.querySelector(`svg[data-task-index="${index}"]`);
    if (!task || !svg) return;
    const [yMin, yMax] = selectedYRange(index, task);
    const y = yScale(yMin, yMax);
    const circles = [...svg.querySelectorAll('circle.point')];
    (task.points || []).forEach((point, pointIndex) => {
      const circle = circles[pointIndex];
      if (circle && finite(point.metric)) circle.setAttribute('cy', y(point.metric).toFixed(1));
    });
    const polyline = svg.querySelector('polyline.best');
    if (polyline) {
      const xs = parsePolylineXs(polyline);
      const coords = (task.best_points || []).map((point, pointIndex) => {
        const x = xs[pointIndex] || xs[xs.length - 1] || '0';
        return `${x},${y(point.metric).toFixed(1)}`;
      });
      polyline.setAttribute('points', coords.join(' '));
    }
    const yMaxLabel = svg.querySelector('.y-max-label');
    const yMinLabel = svg.querySelector('.y-min-label');
    if (yMaxLabel) yMaxLabel.textContent = formatMetric(yMax);
    if (yMinLabel) yMinLabel.textContent = formatMetric(yMin);
  };
  const saveInputs = (index, task) => {
    const controls = document.querySelector(`.chart-controls[data-task-index="${index}"]`);
    const minInput = controls?.querySelector('.y-min');
    const maxInput = controls?.querySelector('.y-max');
    try {
      if (minInput?.value) localStorage.setItem(key(task, 'y_min'), minInput.value);
      else localStorage.removeItem(key(task, 'y_min'));
      if (maxInput?.value) localStorage.setItem(key(task, 'y_max'), maxInput.value);
      else localStorage.removeItem(key(task, 'y_max'));
    } catch (error) {
      return;
    }
  };
  const loadInputs = (index, task) => {
    const controls = document.querySelector(`.chart-controls[data-task-index="${index}"]`);
    const minInput = controls?.querySelector('.y-min');
    const maxInput = controls?.querySelector('.y-max');
    try {
      const storedMin = localStorage.getItem(key(task, 'y_min'));
      const storedMax = localStorage.getItem(key(task, 'y_max'));
      if (storedMin && minInput) minInput.value = storedMin;
      if (storedMax && maxInput) maxInput.value = storedMax;
    } catch (error) {
      return;
    }
  };
  const setRange = (index, mode) => {
    const task = taskByIndex(index);
    const controls = document.querySelector(`.chart-controls[data-task-index="${index}"]`);
    const minInput = controls?.querySelector('.y-min');
    const maxInput = controls?.querySelector('.y-max');
    if (!task || !minInput || !maxInput) return;
    if (mode === 'all') {
      minInput.value = '';
      maxInput.value = '';
    } else if (mode === 'valid') {
      const [low, high] = padded(validYValues(task));
      minInput.value = formatMetric(low);
      maxInput.value = formatMetric(high);
    }
    saveInputs(index, task);
    updateChartY(index);
  };
  (report.tasks || []).forEach((task, index) => {
    loadInputs(index, task);
    const controls = document.querySelector(`.chart-controls[data-task-index="${index}"]`);
    controls?.querySelectorAll('input').forEach((input) => {
      input.addEventListener('change', () => {
        saveInputs(index, task);
        updateChartY(index);
      });
    });
    controls?.querySelectorAll('button[data-range-action]').forEach((button) => {
      button.addEventListener('click', () => setRange(index, button.dataset.rangeAction || 'apply'));
    });
    updateChartY(index);
  });
})();
"""
