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

"""Runner for the monitor trace HTML exporter."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scienceflow.ui.monitor_trace.builder import build_monitor_trace_report
from scienceflow.ui.monitor_trace.render import render_monitor_trace_html


def run_monitor_trace(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    cache_path: str | Path | None = None,
    refresh_sec: float = 60.0,
    once: bool = False,
) -> None:
    """Render monitor trace HTML and optional cache from existing monitor artifacts."""
    output = Path(output_path).expanduser()
    cache = Path(cache_path).expanduser() if cache_path else output.parent / "monitor_trace_data.json"
    interval = max(1.0, float(refresh_sec))
    try:
        while True:
            report = build_monitor_trace_report(manifest_path)
            cache_text = json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
            _write_text_atomic(cache, cache_text)
            html = render_monitor_trace_html(report, refresh_sec=(None if once else interval))
            _write_text_atomic(output, html)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        return


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
