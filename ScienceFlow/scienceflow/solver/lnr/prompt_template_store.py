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

from functools import lru_cache
from importlib import resources
from pathlib import PurePosixPath
from string import Formatter
from typing import Any

_TEMPLATE_PACKAGE = "scienceflow.solver.lnr.prompt_templates"


def _safe_template_parts(name: str) -> tuple[str, ...]:
    raw = str(name or "")
    if not raw.endswith(".md") or "\\" in raw:
        raise ValueError(f"Invalid LNR prompt template name: {name!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Invalid LNR prompt template name: {name!r}")
    return tuple(path.parts)


@lru_cache(maxsize=None)
def load_prompt_template(name: str) -> str:
    parts = _safe_template_parts(name)
    try:
        text = resources.files(_TEMPLATE_PACKAGE).joinpath(*parts).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Missing LNR prompt template: {name}") from exc
    return text[:-1] if text.endswith("\n") else text


def render_prompt_template(name: str, **values: Any) -> str:
    template = load_prompt_template(name)
    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing = sorted(field_names - values.keys())
    if missing:
        raise ValueError(f"Missing values for LNR prompt template {name}: {missing}")
    return template.format(**values)
