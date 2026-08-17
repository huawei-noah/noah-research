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

from scienceflow.safety.resource.gpu_sublease import plan_gpu_sublease
from tests.lnr_resource_test_utils import make_observer


def test_sublease_defaults_to_one_gpu_without_multi_gpu_evidence() -> None:
    plan = plan_gpu_sublease(
        candidate_gpu_ids=["0", "1"],
        default_request=2,
        max_request=2,
        command="python train.py --epochs 10",
    )

    assert plan.requested_gpu_count == 1
    assert plan.reason == "single_gpu_until_multi_gpu_evidence"


def test_sublease_respects_explicit_multi_gpu_request() -> None:
    plan = plan_gpu_sublease(
        candidate_gpu_ids=["0", "1"],
        requested_gpu_count=2,
        default_request=1,
        max_request=2,
        command="python train.py",
    )

    assert plan.requested_gpu_count == 2
    assert plan.explicit_request is True


def test_sublease_detects_torchrun_multi_gpu_launch() -> None:
    plan = plan_gpu_sublease(
        candidate_gpu_ids=["2", "3"],
        default_request=1,
        max_request=2,
        command="torchrun --nproc_per_node=2 train.py",
    )

    assert plan.requested_gpu_count == 2
    assert plan.multi_gpu_evidence


def test_sublease_detects_cuda_visible_devices_multi_gpu_launch() -> None:
    plan = plan_gpu_sublease(
        candidate_gpu_ids=["2", "3"],
        default_request=1,
        max_request=2,
        command="CUDA_VISIBLE_DEVICES=2,3 python train.py",
    )

    assert plan.requested_gpu_count == 2
    assert "cuda_visible_devices_multi" in plan.multi_gpu_evidence

def test_queue_sublease_uses_free_gpu_when_task_pool_has_two_gpus(tmp_path) -> None:
    first, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_request=2,
    )
    second, _ = make_observer(
        tmp_path,
        worker_id="W01",
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_request=2,
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")

    first_job = first.job_created(
        command="python train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0", "1"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert first_job is not None
    first_result = first.queue_try_acquire(first_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0", "1"])
    assert first_result["acquired"] is True
    assert first_result["assigned_physical_gpus"] == ["0"]
    assert first_result["requested_gpu_count"] == 1

    second_job = second.job_created(
        command="python train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0", "1"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert second_job is not None
    second_result = second.queue_try_acquire(second_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0", "1"])

    assert second_result["acquired"] is True
    assert second_result["assigned_physical_gpus"] == ["1"]
    assert second_result["requested_gpu_count"] == 1

