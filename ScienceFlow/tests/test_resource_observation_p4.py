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

import pytest

from scienceflow.core.tools.bash_tool import BashTool
from tests.lnr_resource_test_utils import event_types, make_observer


def _observer(tmp_path, *, window_sec: float):
    observer, sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        observation_enabled=True,
        observation_window_sec=window_sec,
        observation_shadow_workspace_enabled=True,
    )
    return observer, sm


@pytest.mark.asyncio
async def test_unknown_gpu_command_observation_promotes_and_replays_real(tmp_path) -> None:
    observer, sm = _observer(tmp_path, window_sec=0.1)
    script = tmp_path / "run.sh"
    script.write_text("sleep 0.3\necho replayed > real.txt\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("bash run.sh")

    assert not result.error
    assert (tmp_path / "real.txt").read_text(encoding="utf-8").strip() == "replayed"
    events = event_types(sm)
    assert "resource_observation" in events
    assert "resource_observation_promoted" in events
    assert "resource_gpu_lease_acquired" in events
    assert "resource_gpu_lease_released" in events


@pytest.mark.asyncio
async def test_unknown_gpu_command_quick_trial_still_replays_real_workspace(tmp_path) -> None:
    observer, sm = _observer(tmp_path, window_sec=1.0)
    script = tmp_path / "run.sh"
    script.write_text("echo quick > real.txt\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("bash run.sh")

    assert not result.error
    assert (tmp_path / "real.txt").read_text(encoding="utf-8").strip() == "quick"
    events = event_types(sm)
    assert "resource_observation" in events
    assert "resource_observation_promoted" not in events


@pytest.mark.asyncio
async def test_observation_fails_closed_on_shadow_protected_write(tmp_path) -> None:
    observer, sm = _observer(tmp_path, window_sec=1.0)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.csv").write_text("x\n1\n", encoding="utf-8")
    script = tmp_path / "run.sh"
    script.write_text("echo leak > dataset/leak.csv\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=observer,
    )

    result = await tool.execute("bash run.sh")

    assert result.error
    assert "shadow_isolation_failed:shadow" in result.error
    assert "resource_observation" in event_types(sm)
    assert not (tmp_path / "dataset" / "leak.csv").exists()
