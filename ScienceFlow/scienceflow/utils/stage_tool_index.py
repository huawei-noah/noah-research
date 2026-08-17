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

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def append_tool_index(
    log_dir: str | Path,
    *,
    raw_id: str,
    tool_name: str,
    raw_chars: int,
    compressed_chars: int,
    reducer_name: str,
    full_output_path: str | Path,
    tool_error: bool,
    args: dict[str, Any] | None = None,
) -> None:
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = _tool_descriptor(tool_name, args or {})
    line = (
        f"ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"tool={_safe_token(tool_name)} raw_id={_safe_token(raw_id)} "
        f"status={'err' if tool_error else 'ok'} raw_chars={int(raw_chars)} "
        f"compressed_chars={int(compressed_chars)} reducer={_safe_token(reducer_name)} "
        f"full_output={_one_line(full_output_path)}"
    )
    if descriptor:
        line += f" {descriptor}"
    with (root / "tool.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _tool_descriptor(tool_name: str, args: dict[str, Any]) -> str:
    name = str(tool_name or "")
    if name == "bash":
        command = _one_line(args.get("command", ""), max_chars=600)
        return f"command={command}" if command else ""
    path = _one_line(args.get("path", ""), max_chars=300)
    return f"path={path}" if path else ""


def _one_line(value: Any, *, max_chars: int = 500) -> str:
    text = str(value or "").replace("\n", "\\n").replace("\r", "\\r").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 15].rstrip() + "...[truncated]"
    return text.replace(" ", "%20")


def _safe_token(value: Any) -> str:
    text = _one_line(value, max_chars=120)
    return text or "none"
