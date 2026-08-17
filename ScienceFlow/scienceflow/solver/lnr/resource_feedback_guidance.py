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

from scienceflow.solver.lnr.prompt_template_store import load_prompt_template

_GUIDANCE_TEMPLATE = "resources/resource_feedback_guidance.md"


@lru_cache(maxsize=1)
def load_resource_feedback_guidance() -> dict[str, str]:
    text = load_prompt_template(_GUIDANCE_TEMPLATE)
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key and value:
            values[key] = value
    return values


def resource_feedback_guidance_value(key: str, default: str = "") -> str:
    return load_resource_feedback_guidance().get(str(key or "").strip(), default)


def resource_feedback_guidance_csv(key: str) -> set[str]:
    value = resource_feedback_guidance_value(key)
    return {item.strip() for item in value.split(",") if item.strip()}
