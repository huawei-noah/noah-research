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

from tests.lnr_resource_test_utils import FakeStateMachine, event_types, make_observer


def _job(observer, tmp_path, *, command: str, resource_class: str, gpu_id: str = "0") -> str:
    job_id = observer.job_created(
        command=command,
        inferred_class=resource_class,
        gpu_ids=[gpu_id],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id


def test_lnr_heavy_train_is_exclusive_on_same_gpu(tmp_path) -> None:
    sm = FakeStateMachine()
    train_a, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0", "1"])
    train_b, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0", "1"])
    train_c, _ = make_observer(tmp_path, worker_id="W02", state_machine=sm, gpu_pool=["0", "1"])

    job_a = _job(train_a, tmp_path / "a", command="python3 train.py --epochs 1", resource_class="heavy_gpu_candidate", gpu_id="0")
    assert train_a.queue_try_acquire(job_a, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    job_b = _job(train_b, tmp_path / "b", command="python3 train.py --epochs 1", resource_class="heavy_gpu_candidate", gpu_id="0")
    blocked = train_b.queue_try_acquire(job_b, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"

    job_c = _job(train_c, tmp_path / "c", command="python3 train.py --epochs 1", resource_class="heavy_gpu_candidate", gpu_id="1")
    assert train_c.queue_try_acquire(job_c, inferred_class="heavy_gpu_candidate", gpu_ids=["1"])["acquired"] is True

    assert "resource_gpu_queue_wait_started" in event_types(sm)
    assert "resource_gpu_lease_acquired" in event_types(sm)


def test_lnr_tt_does_not_share_with_train_by_default(tmp_path) -> None:
    sm = FakeStateMachine()
    train, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], gpu_share_tt_with_train=False)
    tt, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], gpu_share_tt_with_train=False)

    train_job = _job(train, tmp_path / "train", command="python3 train.py --epochs 1", resource_class="heavy_gpu_candidate")
    assert train.queue_try_acquire(train_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    tt_job = _job(tt, tmp_path / "tt", command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 2", resource_class="gpu_tt_light")
    blocked = tt.queue_try_acquire(tt_job, inferred_class="gpu_tt_light", gpu_ids=["0"])

    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"
    assert "gpu_tt_light" in blocked["feedback"]


def test_lnr_two_feature_jobs_fit_slot_capacity(tmp_path) -> None:
    sm = FakeStateMachine()
    feat_a, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    feat_b, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])
    feat_c, _ = make_observer(tmp_path, worker_id="W02", state_machine=sm, gpu_pool=["0"])

    job_a = _job(feat_a, tmp_path / "a", command="python3 extract_features.py", resource_class="gpu_feature_extract")
    job_b = _job(feat_b, tmp_path / "b", command="python3 extract_features.py", resource_class="gpu_feature_extract")
    job_c = _job(feat_c, tmp_path / "c", command="python3 extract_features.py", resource_class="gpu_feature_extract")

    assert feat_a.queue_try_acquire(job_a, inferred_class="gpu_feature_extract", gpu_ids=["0"])["acquired"] is True
    assert feat_b.queue_try_acquire(job_b, inferred_class="gpu_feature_extract", gpu_ids=["0"])["acquired"] is True
    blocked = feat_c.queue_try_acquire(job_c, inferred_class="gpu_feature_extract", gpu_ids=["0"])
    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"
