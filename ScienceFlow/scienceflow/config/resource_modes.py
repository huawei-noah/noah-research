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

"""Coarse resource-control mode presets for LNR config."""

from __future__ import annotations

from typing import Any

RESOURCE_CONTROL_MODE_ALIASES: dict[str, str] = {
    "off": "off",
    "none": "off",
    "disabled": "off",
    "resource_off": "off",
    "basic": "resource_smart_policy",
    "policy": "resource_smart_policy",
    "smart_policy": "resource_smart_policy",
    "resource_basic": "resource_smart_policy",
    "resource_smart_policy": "resource_smart_policy",
    "full": "resource_smart_llm",
    "llm": "resource_smart_llm",
    "smart_llm": "resource_smart_llm",
    "resource_smart_llm": "resource_smart_llm",
}

RESOURCE_CONTROL_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "off": {
        "resource_monitor_enabled": False,
        "resource_monitor_kill_enabled": False,
        "resource_review_state_enabled": False,
        "resource_context_prompt_enabled": False,
        "resource_runtime_enabled": False,
        "resource_gpu_queue_enabled": False,
        "resource_gpu_admission_queue_enabled": False,
        "resource_admission_llm_enabled": False,
        "resource_observation_enabled": False,
        "resource_arbiter_enabled": False,
        "resource_arbiter_mode": "policy",
        "resource_main_agent_advisory_enabled": False,
        "resource_gpu_share_enabled": False,
    },
    "resource_smart_policy": {
        "resource_monitor_enabled": True,
        "resource_monitor_kill_enabled": True,
        "resource_monitor_kill_mode": "arbiter",
        "resource_review_state_enabled": True,
        "resource_context_prompt_enabled": True,
        "resource_runtime_enabled": True,
        "resource_gpu_queue_enabled": True,
        "resource_gpu_admission_queue_enabled": True,
        "resource_admission_llm_enabled": False,
        "resource_observation_enabled": True,
        "resource_arbiter_enabled": True,
        "resource_arbiter_mode": "policy",
        "resource_main_agent_advisory_enabled": False,
        "resource_gpu_share_enabled": False,
    },
    "resource_smart_llm": {
        "resource_monitor_enabled": True,
        "resource_monitor_kill_enabled": True,
        "resource_monitor_kill_mode": "arbiter",
        "resource_review_state_enabled": True,
        "resource_context_prompt_enabled": True,
        "resource_runtime_enabled": True,
        "resource_gpu_queue_enabled": True,
        "resource_gpu_admission_queue_enabled": True,
        "resource_admission_llm_enabled": True,
        "resource_observation_enabled": True,
        "resource_arbiter_enabled": True,
        "resource_arbiter_mode": "llm",
        "resource_main_agent_advisory_enabled": True,
        "resource_gpu_share_enabled": True,
    },
}


def normalize_resource_control_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "resource_smart_llm" if value else "off"
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mode = RESOURCE_CONTROL_MODE_ALIASES.get(raw)
    if mode is None:
        valid = ", ".join(("off", "resource_smart_policy", "resource_smart_llm"))
        raise ValueError(f"Invalid lnr.resource_control_mode={value!r}; expected one of: {valid}")
    return mode


def apply_lnr_resource_control_mode_mapping(lnr: dict[str, Any]) -> None:
    """Expand a user-facing resource mode into low-level LNR resource switches.

    Existing keys in the same mapping win, so task-specific expert overrides can
    still tune one switch after selecting the coarse mode.
    """
    if "resource_control_mode" not in lnr:
        return
    mode = normalize_resource_control_mode(lnr.get("resource_control_mode"))
    lnr["resource_control_mode"] = mode
    for key, value in RESOURCE_CONTROL_MODE_PRESETS[mode].items():
        lnr.setdefault(key, value)


def expand_lnr_resource_control_mode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an LNR payload with resource_control_mode expanded."""
    out = dict(payload)
    apply_lnr_resource_control_mode_mapping(out)
    return out
