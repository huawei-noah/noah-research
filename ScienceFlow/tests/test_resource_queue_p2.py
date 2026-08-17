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

import asyncio
import time

import pytest

from scienceflow.core.tools.bash_tool import (
    BashTool,
    _resource_admission_tool_result,
    _resource_policy_preflight_result,
)
from scienceflow.core.tools.resource_classifier import RESOURCE_GPU_TT_LIGHT, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC
from scienceflow.solver.lnr.resource_runtime.admission import should_request_admission_llm
from scienceflow.solver.lnr.resource_runtime.startup_trial import classify_preflight_block
from tests.lnr_resource_test_utils import FakeStateMachine, event_types, make_observer, payloads


def _heavy_job(observer, tmp_path, *, gpu_id: str = "0", command: str = "python3 train.py --device cuda --epochs 1", cpu_set: str | None = None) -> str:
    job_id = observer.job_created(
        command=command,
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[gpu_id],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
        cpu_set=cpu_set,
    )
    assert job_id is not None
    return job_id


def _fake_low_util_sample(gpu_ids=None):
    ids = [str(x) for x in (gpu_ids or ["0"])]
    return {
        "available": True,
        "gpus": [
            {
                "gpu_id": gpu_id,
                "utilization_gpu_pct": 0.0,
                "memory_used_mb": 4096.0,
                "memory_total_mb": 49152.0,
            }
            for gpu_id in ids
        ],
    }


def _fake_free_gpu_sample(gpu_ids=None):
    ids = [str(x) for x in (gpu_ids or ["0"])]
    return {
        "available": True,
        "gpus": [
            {
                "gpu_id": gpu_id,
                "utilization_gpu_pct": 0.0,
                "memory_used_mb": 0.0,
                "memory_total_mb": 49152.0,
            }
            for gpu_id in ids
        ],
    }


def test_startup_block_classifier_keeps_hard_boundaries_hard() -> None:
    hard = classify_preflight_block(
        reason="post_feedback_blocked_class",
        source_reason="invalid_deliverable_schema_preflight",
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
    )
    soft = classify_preflight_block(
        reason="stale_resource_context",
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
    )

    assert hard.block_class == "hard_boundary"
    assert soft.block_class == "soft_trial"


def test_trial_first_post_feedback_gate_starts_monitored_trial(tmp_path) -> None:
    sm = FakeStateMachine()
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        resource_startup_policy="trial_first",
        resource_trial_window_sec=12.0,
        resource_trial_hard_review_sec=34.0,
    )
    job = _heavy_job(observer, tmp_path / "trial")
    observer._jobs[job].post_feedback_gate = {
        "violates_feedback": True,
        "source_reason": "stale_resource_context",
        "source_resource_mode": "YELLOW",
        "allowed_classes": [RESOURCE_GPU_TT_LIGHT],
        "source_gpu_ids": ["0"],
    }

    preflight = observer.resource_preflight_decision(
        job,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )
    acquired = observer.queue_try_acquire(job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])

    assert preflight["allowed"] is True
    assert preflight["admission_action"] == "OBSERVE_THEN_RUN"
    assert preflight["startup_block_class"] == "soft_trial"
    assert acquired["acquired"] is True
    assert acquired["admission_observe_then_run"] is True
    assert acquired["resource_trial"] is True
    assert acquired["resource_trial_window_sec"] == 12.0
    assert acquired["resource_trial_hard_review_sec"] == 34.0
    assert "resource_startup_trial_candidate" in event_types(sm)
    assert "resource_trial_started" in event_types(sm)


def test_trial_first_does_not_convert_hard_post_feedback_gate(tmp_path) -> None:
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="lease",
        resource_startup_policy="trial_first",
    )
    job = _heavy_job(observer, tmp_path / "trial")
    observer._jobs[job].post_feedback_gate = {
        "violates_feedback": True,
        "source_reason": "invalid_deliverable_schema_preflight",
        "source_resource_mode": "RED",
        "allowed_classes": [RESOURCE_GPU_TT_LIGHT],
        "source_gpu_ids": ["0"],
    }

    preflight = observer.resource_preflight_decision(
        job,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )

    assert preflight["allowed"] is False
    assert preflight["reason"] == "post_feedback_blocked_class"
    assert "trial" not in preflight


def test_trial_first_high_profile_derives_short_trial_windows(tmp_path) -> None:
    sm = FakeStateMachine()
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        resource_control_profile="high",
        resource_startup_policy="trial_first",
        resource_trial_window_sec=300.0,
        resource_trial_hard_review_sec=900.0,
        timeout_block_train_after=1,
        timeout_tt_only_after=3,
    )
    observer._gpu_queue_timeout_count = 1
    job = _heavy_job(observer, tmp_path / "trial-high")

    preflight = observer.resource_preflight_decision(
        job,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )

    assert preflight["admission_action"] == "OBSERVE_THEN_RUN"
    assert preflight["trial_window_sec"] == 180.0
    assert preflight["trial_hard_review_sec"] == 480.0


def test_trial_first_queue_timeout_train_block_starts_monitored_trial(tmp_path) -> None:
    sm = FakeStateMachine()
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        resource_startup_policy="trial_first",
        timeout_block_train_after=1,
        timeout_tt_only_after=3,
    )
    observer._gpu_queue_timeout_count = 1
    job = _heavy_job(observer, tmp_path / "timeout")

    preflight = observer.resource_preflight_decision(
        job,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )
    acquired = observer.queue_try_acquire(job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])

    assert preflight["allowed"] is True
    assert preflight["gate"] == "block_train_after_queue_timeout"
    assert preflight["admission_action"] == "OBSERVE_THEN_RUN"
    assert acquired["acquired"] is True
    assert acquired["resource_trial"] is True
    assert "resource_trial_started" in event_types(sm)


def test_lnr_gpu_queue_blocks_same_gpu_heavy_and_releases(tmp_path) -> None:
    sm = FakeStateMachine()
    obs1, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    obs2, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])

    job1 = _heavy_job(obs1, tmp_path / "a")
    assert obs1.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    job2 = _heavy_job(obs2, tmp_path / "b")
    blocked = obs2.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert blocked["enabled"] is True
    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"
    assert "RESOURCE_FEEDBACK: PENDING because" in blocked["feedback"]
    assert "resource_gpu_queue_wait_started" in event_types(sm)
    assert "resource_admission_deferred" in event_types(sm)

    obs1.job_finished(job1, status="success", returncode=0, elapsed_sec=0.1)
    acquired = obs2.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["acquired"] is True
    assert "resource_gpu_lease_released" in event_types(sm)


def test_stale_pressure_preflight_allows_when_gpu_is_observed_free(tmp_path, monkeypatch) -> None:
    sm = FakeStateMachine()
    observer, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        stale_pressure_observe_first_enabled=True,
        stale_pressure_observe_window_sec=30.0,
    )
    assert observer.resource_runtime is not None
    observer.resource_runtime.record_runtime_pressure(
        job_id="old-job",
        resource_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        elapsed_sec=120.0,
        reason="runtime_gpu_pressure",
        command_digest="old-digest",
    )
    monkeypatch.setattr(
        "scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi",
        _fake_free_gpu_sample,
    )

    job = _heavy_job(observer, tmp_path / "train", gpu_id="0")
    decision = observer.resource_preflight_decision(
        job,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )

    assert decision == {"allowed": True, "resource_class": RESOURCE_HEAVY_GPU_CANDIDATE}
    assert "resource_observe_first_candidate" not in event_types(sm)
    assert "resource_policy_gate" not in event_types(sm)

    acquired = observer.queue_try_acquire(job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])
    assert acquired["acquired"] is True
    assert not acquired.get("admission_observe_then_run")
    assert "resource_observe_first_started" not in event_types(sm)


def test_lnr_repeated_blocked_gpu_train_uses_cached_backoff(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], gpu_queue_heartbeat_sec=0.05)
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], gpu_queue_heartbeat_sec=0.05)

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    queued_job = _heavy_job(queued, tmp_path / "queued")
    first = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert first["status"] == "PENDING"
    assert first.get("admission_cached_backoff") is False

    time.sleep(0.3)
    second = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert second["status"] == "DENIED_REPLAN"
    assert second["admission_cached_backoff"] is True
    assert second["blocked_until_unlock"] is True
    assert second["post_feedback_action"] == "cpu_support"
    assert float(second["retry_after_sec"]) > 0
    assert "cached_backoff=true" in second["feedback"]
    assert "post_feedback_action=cpu_support" in second["feedback"]
    assert "retry_after_sec=" in second["feedback"]
    assert "cpu_support_hint" not in second["feedback"]

    deferred = payloads(sm, "resource_admission_deferred")
    assert deferred[-1]["status"] == "DENIED_REPLAN"
    assert deferred[-1]["admission_cached_backoff"] is True
    assert deferred[-1]["post_feedback_action"] == "cpu_support"

    holder.job_finished(holder_job, status="success", returncode=0, elapsed_sec=0.1)
    acquired = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["acquired"] is True


def test_lnr_pending_feedback_offers_resource_wait_option(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued")

    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert blocked["status"] == "PENDING"
    assert blocked.get("resource_wait_option", {}).get("offered") is True
    feedback = blocked["feedback"]
    assert "action_options=cpu_support,resource_wait" in feedback
    assert "cpu_support_preferred=true" in feedback
    assert "wait_tool=resource_wait" in feedback
    assert "wait_token=rw-W01" in feedback


@pytest.mark.asyncio
async def test_bash_sleep_after_wait_offer_requires_resource_wait_tool(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued")
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    token = blocked["resource_wait_option"]["wait_token"]

    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        resource_observer=queued,
    )

    result = await tool.execute("sleep 30; echo Checking GPU after wait")

    assert result.error == "Use resource_wait tool"
    assert "RESOURCE_FEEDBACK: RESOURCE_WAIT_REQUIRED" in (result.output or "")
    assert "wait_tool=resource_wait" in (result.output or "")
    assert f"wait_token={token}" in (result.output or "")
    assert "bash_sleep_allowed=false" in (result.output or "")


@pytest.mark.asyncio
async def test_resource_wait_wakes_on_holder_release_before_timeout(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued")
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    token = blocked["resource_wait_option"]["wait_token"]

    async def release_holder() -> None:
        await asyncio.sleep(0.05)
        holder.job_finished(holder_job, status="success", returncode=0, elapsed_sec=0.1)

    release_task = asyncio.create_task(release_holder())
    result = await queued.managed_resource_wait(wait_token=token, max_wait_sec=1.0)
    await release_task

    assert result["status"] == "RESOURCE_AVAILABLE"
    assert result["elapsed_sec"] < 0.8
    assert "retry_allowed=true" in result["feedback"]


@pytest.mark.asyncio
async def test_resource_wait_timeout_suppresses_same_state_reoffer(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued")
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    token = blocked["resource_wait_option"]["wait_token"]

    result = await queued.managed_resource_wait(wait_token=token, max_wait_sec=0.05)

    assert result["status"] == "RESOURCE_WAIT_TIMEOUT"
    assert "resource_wait_allowed=false" in result["feedback"]
    repeated = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert repeated.get("resource_wait_option", {}).get("offered") is not True
    assert "wait_token=" not in repeated["feedback"]


@pytest.mark.asyncio
async def test_resource_wait_invalid_token_returns_compact_feedback(tmp_path) -> None:
    from scienceflow.core.tools.resource_wait_tool import ResourceWaitTool

    queued, _ = make_observer(tmp_path, worker_id="W01", gpu_pool=["0"])
    tool = ResourceWaitTool(resource_observer=queued)

    result = await tool.execute(wait_token="rw-missing")

    assert result.error is None
    assert "RESOURCE_FEEDBACK: RESOURCE_WAIT_NOT_AVAILABLE" in (result.output or "")
    assert "retry_allowed=false" in (result.output or "")


@pytest.mark.asyncio
async def test_lnr_admission_llm_can_replan_blocked_gpu_task(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])

    async def decide(card, rule_result):
        assert card["resource_class"] == "heavy_gpu_candidate"
        assert rule_result["status"] == "PENDING"
        return {"action": "REPLAN", "reason": "duplicate low value train under contention", "confidence": "high"}

    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        admission_llm_enabled=True,
        admission_llm_mode="always",
        admission_decider=decide,
    )

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued")
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    reviewed = await queued.admission_decide(queued_job, admission_result=blocked)

    assert reviewed["enabled"] is True
    assert reviewed["result"]["status"] == "REPLAN"
    assert reviewed["result"]["admission_llm_reviewed"] is True
    assert "RESOURCE_FEEDBACK: REPLAN because duplicate low value train under contention" in reviewed["result"]["feedback"]
    assert "resource_admission_llm_decision" in event_types(sm)


@pytest.mark.asyncio
async def test_lnr_admission_llm_can_grant_partial_gpu_subset(tmp_path, monkeypatch) -> None:
    from scienceflow.solver.lnr.resource_runtime import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "sample_nvidia_smi", _fake_low_util_sample)
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0", "1"], assignment="lease")

    async def decide(card, rule_result):
        opportunity = card["admission_opportunity"]
        assert opportunity["lease_grantable_by_llm"] is True
        assert opportunity["full_request_grantable"] is False
        assert "1" in opportunity["grantable_gpu_ids"]
        return {"action": "OBSERVE_THEN_RUN", "reason": "gpu1 has enough observed headroom", "confidence": "medium", "gpu_ids": ["1"], "observe_sec": 120}

    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_default_request=2,
        gpu_max_request=2,
        admission_llm_enabled=True,
        admission_llm_mode="low_confidence",
        admission_decider=decide,
    )

    holder_job = _heavy_job(holder, tmp_path / "holder", gpu_id="0")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["assigned_physical_gpus"] == ["0"]

    queued_job = queued.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0", "1"],
        timeout_sec=10.0,
        workspace_dir=tmp_path / "queued",
    )
    assert queued_job is not None
    queued._jobs[queued_job].gpu_request_count = 2
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0", "1"])
    assert blocked["status"] == "PENDING"
    assert blocked["admission_opportunity"]["lease_grantable_by_llm"] is True

    reviewed = await queued.admission_decide(queued_job, admission_result=blocked)

    assert reviewed["enabled"] is True
    result = reviewed["result"]
    assert result["acquired"] is True
    assert result["status"] == "GRANTED"
    assert result["admission_action"] == "OBSERVE_THEN_RUN"
    assert result["assigned_physical_gpus"] == ["1"]
    assert result["env_updates"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert result["admission_observe_then_run"] is True
    assert result["admission_llm_lease_attempted"] is True


@pytest.mark.asyncio
async def test_lnr_admission_llm_run_now_falls_back_when_atomic_grant_fails(tmp_path, monkeypatch) -> None:
    from scienceflow.solver.lnr.resource_runtime import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "sample_nvidia_smi", _fake_low_util_sample)
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], assignment="lease")

    async def decide(card, rule_result):
        return {"action": "RUN_NOW", "reason": "try available gpu", "confidence": "medium", "gpu_ids": ["0"]}

    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        admission_llm_enabled=True,
        admission_llm_mode="low_confidence",
        admission_decider=decide,
    )

    holder_job = _heavy_job(holder, tmp_path / "holder", gpu_id="0")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued", gpu_id="0")
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    blocked["admission_opportunity"] = {
        "enabled": True,
        "lease_grantable_by_llm": True,
        "grantable_gpu_ids": ["0"],
        "partial_gpu_candidates": [{"gpu_id": "0", "free_mem_gb": 44.0, "utilization_gpu_pct": 0.0}],
    }
    blocked["lease_grantable_by_llm"] = True

    reviewed = await queued.admission_decide(queued_job, admission_result=blocked)

    assert reviewed["enabled"] is True
    result = reviewed["result"]
    assert result["acquired"] is False
    assert result["status"] == "PENDING"
    assert "atomic lease acquire failed" in result["reason"]
    assert "RESOURCE_FEEDBACK: PENDING because LLM approved GPU start" in result["feedback"]
    assert result["admission_llm_lease_attempt"]["acquired"] is False


def test_policy_preflight_suppressed_feedback_does_not_synthesize_resource_feedback() -> None:
    class Observer:
        def __init__(self) -> None:
            self.finished = None

        def resource_preflight_decision(self, *_args, **_kwargs):
            return {
                "allowed": False,
                "reason": "invalid_deliverable_schema_preflight",
                "feedback": "",
                "feedback_suppressed": True,
                "resource_feedback_repeated_count": 7,
                "error": "Resource policy blocked command because deliverable schema is invalid",
            }

        def job_finished(self, *args, **kwargs):
            self.finished = (args, kwargs)

    observer = Observer()

    result = asyncio.run(
        _resource_policy_preflight_result(
            observer,
            "job-1",
            inferred_class="heavy_gpu_candidate",
            gpu_ids=["0"],
            elapsed_sec=0.0,
        )
    )

    assert result is not None
    assert result.output == ""
    assert result.error is None
    assert "RESOURCE_FEEDBACK" not in (result.output or "")
    assert observer.finished is not None


def test_blocked_states_mode_reviews_soft_and_cached_blocked_results() -> None:
    soft = {
        "enabled": True,
        "acquired": False,
        "status": "DENIED_REPLAN",
        "requires_admission_review": True,
        "reason": "post_feedback_blocked_class",
    }
    cached = {
        "enabled": True,
        "acquired": False,
        "status": "PENDING",
        "admission_cached_backoff": True,
        "reason": "admission_cached_backoff",
    }

    assert should_request_admission_llm(soft, mode="low_confidence") is True
    assert should_request_admission_llm(cached, mode="blocked_states") is True
    assert should_request_admission_llm(cached, mode="low_confidence") is False


def test_post_feedback_guard_returns_admission_review_when_llm_enabled(tmp_path) -> None:
    sm = FakeStateMachine()
    observer, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        admission_llm_enabled=True,
        admission_llm_mode="blocked_states",
        admission_decider=lambda _card, _rule_result: {
            "action": "PENDING",
            "reason": "review",
        },
    )
    job = _heavy_job(observer, tmp_path, command="python3 train.py --device cuda --epochs 1")
    observer._jobs[job].post_feedback_gate = {
        "allowed_classes": ["pure_tt_cpu", "readonly_cpu", "light_cpu"],
        "source_resource_mode": "RED",
        "source_gpu_ids": ["0"],
        "source_reason": "block_train_after_gpu_pressure",
    }

    decision = observer.resource_preflight_decision(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert decision["allowed"] is False
    assert decision["requires_admission_review"] is True
    assert decision["reason"] == "post_feedback_blocked_class"
    assert decision["soft_gate_reason"] == "post_feedback_blocked_class"
    assert decision["lease_grantable_by_llm"] is True
    planner_events = payloads(sm, "resource_planner_guard")
    assert planner_events and planner_events[-1]["requires_admission_review"] is True


@pytest.mark.asyncio
async def test_policy_preflight_review_uses_admission_result_not_policy_block() -> None:
    class Observer:
        def __init__(self) -> None:
            self.finished = None
            self.reviewed = False

        def resource_preflight_decision(self, *_args, **_kwargs):
            return {
                "allowed": False,
                "enabled": True,
                "acquired": False,
                "requires_admission_review": True,
                "status": "DENIED_REPLAN",
                "reason": "post_feedback_blocked_class",
                "feedback": "RESOURCE_FEEDBACK: DENIED_REPLAN because post_feedback_blocked_class.\n",
            }

        async def admission_decide(self, *_args, **_kwargs):
            self.reviewed = True
            return {
                "enabled": True,
                "result": {
                    "status": "REPLAN",
                    "admission_action": "REPLAN",
                    "reason": "soft gate reviewed by admission llm",
                    "feedback": "RESOURCE_FEEDBACK: REPLAN because soft gate reviewed by admission llm.\n",
                    "admission_llm_reviewed": True,
                },
            }

        def admission_share_decide(self, *_args, **_kwargs):
            return {"enabled": False}

        def job_finished(self, *args, **kwargs):
            self.finished = (args, kwargs)

    observer = Observer()

    result = await _resource_policy_preflight_result(
        observer,
        "job-1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        elapsed_sec=0.0,
    )

    assert observer.reviewed is True
    assert result is not None
    assert result.error == "Resource admission pending"
    assert result.output == "RESOURCE_FEEDBACK: REPLAN because soft gate reviewed by admission llm."
    assert observer.finished[1]["status"] == "resource_admission_replan"


@pytest.mark.asyncio
async def test_bash_tool_applies_admission_llm_replan_before_launch(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], max_wait_sec=0.05)

    async def decide(_card, _rule_result):
        return {"action": "REPLAN", "reason": "same command should not wait", "confidence": "high"}

    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        max_wait_sec=0.05,
        admission_llm_enabled=True,
        admission_llm_mode="always",
        admission_decider=decide,
    )
    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=queued,
    )

    result = await tool.execute("python3 train.py --epochs 1")

    assert result.error == "Resource admission pending"
    assert (result.output or "").startswith(
        "RESOURCE_FEEDBACK: REPLAN because same command should not wait"
    )
    assert "resource_admission_llm_decision" in event_types(sm)


def test_resource_admission_result_returns_pending_resource_wait_option(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], max_wait_sec=2.0)
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], max_wait_sec=2.0)
    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    queued_job = _heavy_job(queued, tmp_path / "queued")
    blocked = queued.queue_try_acquire(queued_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert blocked["status"] == "PENDING"
    assert blocked.get("resource_wait_option", {}).get("offered") is True

    result = _resource_admission_tool_result(queued, queued_job, blocked, elapsed_sec=0.0)

    assert result is not None
    assert result.error == "Resource admission pending"
    assert "RESOURCE_FEEDBACK: PENDING" in (result.output or "")
    assert "action_options=cpu_support,resource_wait" in (result.output or "")
    assert "wait_tool=resource_wait" in (result.output or "")
    assert "wait_token=rw-W01" in (result.output or "")


@pytest.mark.asyncio
async def test_bash_tool_rule_pending_without_wait_option_keeps_queue_wait(tmp_path, monkeypatch) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], max_wait_sec=2.0)
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], max_wait_sec=2.0)
    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    def no_wait_option(**kwargs):
        return {"offered": False, "reason": "test_wait_option_unavailable"}

    monkeypatch.setattr(queued.resource_runtime, "create_resource_wait_option", no_wait_option)

    train_py = tmp_path / "train.py"
    train_py.write_text("import torch\nprint('ran-train')\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=queued,
    )

    async def release_holder() -> None:
        await asyncio.sleep(0.05)
        holder.job_finished(holder_job, status="success", returncode=0, elapsed_sec=0.1)

    release_task = asyncio.create_task(release_holder())
    result = await tool.execute("SCIENCEFLOW_RESOURCE_INTENT=heavy_gpu_candidate python3 train.py --epochs 1")
    await release_task

    assert not result.error
    assert "ran-train" in (result.output or "")
    assert "resource_gpu_queue_timeout" not in event_types(sm)


def test_lnr_gpu_queue_allows_same_class_on_different_gpu(tmp_path) -> None:
    sm = FakeStateMachine()
    obs1, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0", "1"])
    obs2, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0", "1"])

    job1 = _heavy_job(obs1, tmp_path / "a", gpu_id="0")
    job2 = _heavy_job(obs2, tmp_path / "b", gpu_id="1")

    assert obs1.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    assert obs2.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["1"])["acquired"] is True
    leases = obs1.resource_runtime.gpu_store.snapshot_active()["leases"]
    assert set(leases) == {job1, job2}


def test_lnr_slot_weight_blocks_feature_when_capacity_full(tmp_path) -> None:
    sm = FakeStateMachine()
    train, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    feature, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])

    train_job = _heavy_job(train, tmp_path / "train")
    assert train.queue_try_acquire(train_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    feature_job = feature.job_created(
        command="python3 extract_features.py",
        inferred_class="gpu_feature_extract",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert feature_job is not None
    blocked = feature.queue_try_acquire(feature_job, inferred_class="gpu_feature_extract", gpu_ids=["0"])
    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"


def test_lnr_light_waiter_blocked_by_heavy_marks_share_override_candidate(tmp_path) -> None:
    sm = FakeStateMachine()
    train, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], assignment="lease")
    tt, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], assignment="lease")

    train_job = _heavy_job(train, tmp_path / "train")
    assert train.queue_try_acquire(train_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    tt_job = tt.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert tt_job is not None
    blocked = tt.queue_try_acquire(tt_job, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])

    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"
    assert blocked["share_override_candidate"] is True
    assert blocked["share_override_record"]["updated"] is True
    override = blocked["share_override"]
    assert override["proposal_type"] == "task_gpu_share_review"
    assert override["primary_job_ids"] == [train_job]
    assert override["waiter_job_id"] == tt_job
    assert override["waiter_resource_class"] == RESOURCE_GPU_TT_LIGHT
    assert "incompatible_active_class" in override["block_reasons"]
    snapshot = tt.resource_runtime.gpu_store.snapshot_active()
    waiter_meta = snapshot["waiters"][tt_job]["metadata"]
    assert waiter_meta["share_override"]["waiter_job_id"] == tt_job
    assert waiter_meta["share_override"]["primary_job_ids"] == [train_job]
    deferred = payloads(sm, "resource_admission_deferred")[-1]
    assert deferred["share_override_candidate"] is True
    assert deferred["share_override"]["primary_job_ids"] == [train_job]


def test_lnr_heavy_waiter_blocked_by_heavy_marks_revocable_trial_share_candidate(tmp_path) -> None:
    sm = FakeStateMachine()
    first, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], assignment="lease")
    second, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], assignment="lease")

    first_job = _heavy_job(first, tmp_path / "first")
    assert first.queue_try_acquire(first_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    second_job = _heavy_job(second, tmp_path / "second", command="python3 train_alt.py --epochs 1")
    blocked = second.queue_try_acquire(second_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])

    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"
    assert blocked["share_override_candidate"] is True
    override = blocked["share_override"]
    assert override["reason"] == "resource_contention_trial_share_review_required"
    assert override["proposal_type"] == "task_gpu_share_review"
    assert override["trial_share"] is True
    assert override["trial_mode"] == "revocable_secondary_trial"
    assert override["trial_initial_observe_sec"] == 180.0
    assert override["waiter_effective_slot_weight"] == 0.5
    assert "arbiter_llm_grant_required" in override["required_gates"]
    assert "trial_secondary_stop_first" in override["required_gates"]
    assert blocked["share_override_record"]["updated"] is True
    waiter_meta = second.resource_runtime.gpu_store.snapshot_active()["waiters"][second_job]["metadata"]
    assert waiter_meta["share_override"]["trial_share"] is True





def test_lnr_unknown_gpu_waiter_enters_conservative_revocable_trial_candidate(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], assignment="lease")
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], assignment="lease")

    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    unknown_job = queued.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 maybe_gpu.py --device cuda",
        inferred_class=RESOURCE_UNKNOWN_GPU_EXEC,
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
        cpu_set="4-7",
    )
    assert unknown_job is not None

    blocked = queued.queue_try_acquire(unknown_job, inferred_class=RESOURCE_UNKNOWN_GPU_EXEC, gpu_ids=["0"])

    assert blocked["acquired"] is False
    assert blocked["status"] == "PENDING"
    assert blocked["share_override_candidate"] is True
    override = blocked["share_override"]
    assert override["trial_share"] is True
    assert override["trial_initial_observe_sec"] == 90.0
    assert override["waiter_resource_class"] == RESOURCE_UNKNOWN_GPU_EXEC
    assert "unknown_gpu_conservative_memory_estimate" in override["required_gates"]


def test_lnr_heavy_waiter_blocked_by_non_heavy_holder_marks_trial_candidate(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], assignment="lease")
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], assignment="lease")

    holder_job = holder.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4 --device cuda",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
        cpu_set="0-3",
    )
    assert holder_job is not None
    assert holder.queue_try_acquire(holder_job, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])["acquired"] is True
    waiter_job = _heavy_job(queued, tmp_path / "waiter", command="python3 train_alt.py --epochs 1", cpu_set="4-7")

    blocked = queued.queue_try_acquire(waiter_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])

    assert blocked["acquired"] is False
    assert blocked["share_override_candidate"] is True
    override = blocked["share_override"]
    assert override["primary_job_ids"] == [holder_job]
    assert override["primary_classes"][holder_job] == RESOURCE_GPU_TT_LIGHT
    assert "non_heavy_holder_low_util_trial" in override["required_gates"]


@pytest.mark.asyncio
async def test_lnr_deterministic_trial_admission_grants_without_llm(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_low_util_sample)
    sm = FakeStateMachine()
    holder, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
    )
    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
        gpu_trial_admission_policy="deterministic_grant_when_hard_gates_pass",
        arbiter_enabled=False,
    )

    holder_job = _heavy_job(holder, tmp_path / "holder", cpu_set="0-3")
    assert holder.queue_try_acquire(holder_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    unknown_job = queued.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 maybe_gpu.py --device cuda",
        inferred_class=RESOURCE_UNKNOWN_GPU_EXEC,
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
        cpu_set="4-7",
    )
    assert unknown_job is not None
    blocked = queued.queue_try_acquire(unknown_job, inferred_class=RESOURCE_UNKNOWN_GPU_EXEC, gpu_ids=["0"])
    assert blocked["share_override_candidate"] is True

    reviewed = await queued.admission_share_decide(unknown_job, admission_result=blocked)

    assert reviewed["enabled"] is True
    assert reviewed["action"] == "GRANT_SHARED_GPU_LEASE"
    assert reviewed["result"]["acquired"] is True
    assert reviewed["result"]["admission_action"] == "RUN_NOW"
    grant = reviewed["result"]["shared_lease_grant"]
    assert grant["acquired"] is True
    assert grant["lease"]["metadata"]["lease_mode"] == "shared_secondary"
    assert grant["lease"]["metadata"]["shared_primary_job_id"] == holder_job
    assert reviewed["arbiter"]["decision"]["source"] == "deterministic_trial_admission"


@pytest.mark.asyncio
async def test_lnr_admission_share_review_grants_waiter_shared_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_low_util_sample)
    sm = FakeStateMachine()
    decisions = []

    async def arbiter_decide(proposal):
        decisions.append(proposal)
        assert proposal["proposal_type"] == "task_gpu_share_review"
        assert proposal["reason_code"] == "task_gpu_share_review:admission_shared_trial"
        assert proposal["resource_snapshot"]["hard_gates"]["memory_headroom_for_secondary"] is True
        facts = proposal["share_decision_facts"]
        assert facts["decision_source"] == "admission_waiter"
        assert facts["share_eligible"] is True
        assert facts["hard_gates_pass"] is True
        assert facts["primary_gpu_util_p90_pct"] == 0.0
        assert facts["secondary_estimated_peak_gb"] > 0.0
        return {"action": "GRANT_SHARED_GPU_LEASE", "reason": "low util revocable trial", "confidence": "high"}

    holder, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
    )
    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter_decide,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    holder_job = _heavy_job(holder, tmp_path / "holder", cpu_set="0-3")
    assert holder.queue_try_acquire(holder_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued", command="python3 train_alt.py --epochs 1", cpu_set="4-7")
    blocked = queued.queue_try_acquire(queued_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])
    assert blocked["share_override_candidate"] is True

    reviewed = await queued.admission_share_decide(queued_job, admission_result=blocked)

    assert reviewed["enabled"] is True
    assert reviewed["action"] == "GRANT_SHARED_GPU_LEASE"
    assert reviewed["result"]["acquired"] is True
    assert reviewed["result"]["admission_action"] == "RUN_NOW"
    assert reviewed["result"]["admission_shared_trial"] is True
    assert len(decisions) == 1
    acquired = queued.queue_try_acquire(queued_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])
    assert acquired["acquired"] is True
    lease_meta = acquired["details"]["lease"]["metadata"]
    assert lease_meta["lease_mode"] == "shared_secondary"
    assert lease_meta["shared_primary_job_id"] == holder_job
    assert lease_meta["admission_immediate_review"] is True


@pytest.mark.asyncio
async def test_lnr_admission_share_review_denies_to_waiter_cpu_support(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_low_util_sample)
    sm = FakeStateMachine()

    async def arbiter_decide(_proposal):
        return {"action": "DENY_SHARE_USE_CPU_SUPPORT", "reason": "secondary trial not worth the risk", "confidence": "medium"}

    holder, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
    )
    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter_decide,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    holder_job = _heavy_job(holder, tmp_path / "holder", cpu_set="0-3")
    assert holder.queue_try_acquire(holder_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    queued_job = _heavy_job(queued, tmp_path / "queued", command="python3 train_alt.py --epochs 1", cpu_set="4-7")
    blocked = queued.queue_try_acquire(queued_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])

    reviewed = await queued.admission_share_decide(queued_job, admission_result=blocked)

    assert reviewed["enabled"] is True
    assert reviewed["action"] == "DENY_SHARE_USE_CPU_SUPPORT"
    result = reviewed["result"]
    assert result["status"] == "PENDING"
    assert result["admission_llm_reviewed"] is True
    assert result["post_feedback_action"] == "cpu_support"
    assert "RESOURCE_FEEDBACK: LOCAL_GPU_BUSY_USE_CPU_SUPPORT" in result["feedback"]
    assert "post_feedback_action=cpu_support" in result["feedback"]


@pytest.mark.asyncio
async def test_bash_tool_admission_share_grant_runs_waiter_without_queue_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scienceflow.solver.lnr.resource_runtime.runtime.sample_nvidia_smi", _fake_low_util_sample)
    sm = FakeStateMachine()

    async def arbiter_decide(_proposal):
        return {"action": "GRANT_SHARED_GPU_LEASE", "reason": "low util revocable trial", "confidence": "high"}

    holder, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        max_wait_sec=0.5,
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
    )
    queued, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0"],
        assignment="lease",
        max_wait_sec=0.5,
        gpu_share_enabled=True,
        gpu_share_phase="feature_share",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter_decide,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    holder_job = _heavy_job(holder, tmp_path / "holder", cpu_set="0-3")
    assert holder.queue_try_acquire(holder_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    train_py = tmp_path / "train.py"
    train_py.write_text("print('ran-shared-trial')\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0", "_SCIENCEFLOW_CPU_SET": "4-7"},
        resource_observer=queued,
    )

    result = await tool.execute("python3 train.py --epochs 1")

    assert not result.error
    assert "ran-shared-trial" in (result.output or "")
    assert "resource_gpu_queue_timeout" not in event_types(sm)


@pytest.mark.asyncio
async def test_bash_tool_queue_timeout_returns_compact_resource_feedback(tmp_path, monkeypatch) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"], max_wait_sec=0.05)
    queued, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"], max_wait_sec=0.05)
    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    def no_wait_option(**kwargs):
        return {"offered": False, "reason": "test_wait_option_unavailable"}

    monkeypatch.setattr(queued.resource_runtime, "create_resource_wait_option", no_wait_option)

    train_py = tmp_path / "train.py"
    train_py.write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=queued,
    )
    result = await tool.execute("SCIENCEFLOW_RESOURCE_INTENT=heavy_gpu_candidate python3 train.py --epochs 1")

    assert result.error == "GPU train queue wait exceeded"
    feedback = result.output or ""
    assert feedback.startswith("[queue_timeout")
    assert "RESOURCE_FEEDBACK: queue_wait_aborted" in feedback
    assert "recommended_actions" not in feedback
    assert "guidance:" not in feedback
    assert "resource_gpu_queue_timeout" in event_types(sm)


@pytest.mark.asyncio
async def test_bash_tool_light_command_bypasses_gpu_queue(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(tmp_path, worker_id="W00", state_machine=sm, gpu_pool=["0"])
    light, _ = make_observer(tmp_path, worker_id="W01", state_machine=sm, gpu_pool=["0"])
    holder_job = _heavy_job(holder, tmp_path / "holder")
    assert holder.queue_try_acquire(holder_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True

    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        resource_observer=light,
    )
    result = await tool.execute("python3 -c 'print(1)'")

    assert not result.error
    assert "1" in (result.output or "")
    acquired_payloads = payloads(sm, "resource_gpu_lease_acquired")
    assert len(acquired_payloads) == 1



def test_lnr_lease_assignment_gang_acquires_two_gpus(tmp_path) -> None:
    sm = FakeStateMachine()
    obs, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_request=2,
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")

    job = obs.job_created(
        command="torchrun --nproc_per_node=2 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job is not None
    result = obs.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])

    assert result["acquired"] is True
    assert result["admission_action"] == "RUN_NOW"
    assert set(result["gpu_ids"]) == {"0", "1"}




def test_lnr_lease_assignment_spreads_gpu_work_across_pool(tmp_path) -> None:
    sm = FakeStateMachine()
    first, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_heavy_per_gpu=3,
        gpu_capacity_slots=3.0,
    )
    second, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_heavy_per_gpu=3,
        gpu_capacity_slots=3.0,
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")

    job1 = first.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job1 is not None
    r1 = first.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert r1["acquired"] is True
    assert r1["gpu_ids"] == ["0"]

    job2 = second.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert job2 is not None
    r2 = second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=[])

    assert r2["acquired"] is True
    assert r2["gpu_ids"] == ["1"]
    assert r2["assigned_physical_gpus"] == ["1"]

def test_lnr_two_gpu_pending_protects_against_low_value_fragmentation(tmp_path) -> None:
    sm = FakeStateMachine()
    holder, _ = make_observer(
        tmp_path,
        worker_id="W00",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_request=2,
    )
    two_gpu, _ = make_observer(
        tmp_path,
        worker_id="W01",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_request=2,
    )
    low_one, _ = make_observer(
        tmp_path,
        worker_id="W02",
        state_machine=sm,
        gpu_pool=["0", "1"],
        assignment="lease",
        gpu_max_request=2,
    )
    (tmp_path / "train.py").write_text("import torch\nprint(torch.cuda.is_available())\n", encoding="utf-8")

    hold_job = holder.job_created(
        command="python3 train.py --epochs 1",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
    )
    assert hold_job is not None
    hold = holder.queue_try_acquire(hold_job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert hold["acquired"] is True

    two_job = two_gpu.job_created(
        command="torchrun --nproc_per_node=2 train.py --epochs 8",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
        value_hint_override={"expected_value_score": 1.0, "near_submission_score": 0.8},
    )
    assert two_job is not None
    pending = two_gpu.queue_try_acquire(two_job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert pending["acquired"] is False
    assert pending["status"] == "PENDING"

    holder.job_finished(hold_job, status="success", returncode=0, elapsed_sec=0.1)
    low_job = low_one.job_created(
        command="python3 train.py --epochs 8",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=10.0,
        workspace_dir=tmp_path,
        value_hint_override={"expected_value_score": 0.1, "long_runtime_penalty": 1.0},
    )
    assert low_job is not None
    low = low_one.queue_try_acquire(low_job, inferred_class="heavy_gpu_candidate", gpu_ids=[])

    assert low["acquired"] is False
    assert low["status"] == "PENDING"
    assert low["reason"] == "admission_waiter_ahead"
    assert low["top_waiter_job_id"] == two_job


@pytest.mark.asyncio
async def test_bash_tool_blocks_background_long_resource_command(tmp_path) -> None:
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )

    result = await tool.execute("python3 -u train.py > tmp/train.log 2>&1 & echo $!")

    assert result.error is not None
    assert "background long-running resource commands are not allowed" in result.error
    assert "Run training, inference, and feature extraction in the foreground" in result.error


@pytest.mark.asyncio
async def test_bash_tool_blocks_mixed_heredoc_write_and_resource_execution(tmp_path) -> None:
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )
    cmd = """cat > tmp/train_model.py <<'PYEOF'
print('train')
PYEOF
CUDA_VISIBLE_DEVICES=0 python3 tmp/train_model.py 2>&1"""

    result = await tool.execute(cmd)

    assert result.error is not None
    assert "do not combine a heredoc file write" in result.error
    assert "separate foreground bash command" in result.error
