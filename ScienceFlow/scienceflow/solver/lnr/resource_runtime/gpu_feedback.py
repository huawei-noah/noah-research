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


def build_gpu_boundary_feedback(
    *,
    actual_gpu_ids: list[str],
    allowed_gpu_ids: list[str],
    command_scope: str = "command",
) -> str:
    actual = ",".join(_clean_ids(actual_gpu_ids)) or "unknown"
    allowed = ",".join(_clean_ids(allowed_gpu_ids)) or "none"
    scope = str(command_scope or "command").strip() or "command"
    return (
        "RESOURCE_FEEDBACK: STOP_BOUNDARY_VIOLATION\n"
        f"This {scope} used physical GPU {actual}, outside its assigned task GPU set {allowed}.\n"
        "The process tree was terminated to protect other tasks.\n\n"
        "Do not hard-code or override host physical GPU ids. Remove code such as:\n"
        '- os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        "- CUDA_VISIBLE_DEVICES=0 python train.py\n"
        '- os.environ["CUDA_VISIBLE_DEVICES"] = "3"; torch.device("cuda:0")\n\n'
        "Use the runner-provided CUDA_VISIBLE_DEVICES / SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL.\n"
        "Inside that assigned environment, torch.device(\"cuda:0\") is valid because it means logical device 0.\n"
        "Patch physical-id overrides in the training script or bash command and rerun."
    )


def _clean_ids(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return [str(x).strip() for x in values if str(x).strip()]
