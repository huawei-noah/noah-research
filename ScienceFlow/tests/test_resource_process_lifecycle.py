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

from scienceflow.safety.resource.process_lifecycle import classify_process_lifecycle


def test_lifecycle_busy_child_is_active_work() -> None:
    lifecycle = classify_process_lifecycle(
        process_liveness={
            "status": "alive",
            "root_pid_alive": False,
            "process_group_alive": True,
            "live_descendant_count": 1,
            "busy_child_count": 1,
            "owner_pid_start_time_match": True,
        },
        process_tree_cpu={"total_cpu_pct": 180.0},
    )

    assert lifecycle.status == "alive_with_busy_children"
    assert lifecycle.safe_to_mark_finished is False


def test_lifecycle_stale_pid_binding_can_close_record_but_not_kill_pid() -> None:
    lifecycle = classify_process_lifecycle(
        process_liveness={
            "status": "inconsistent",
            "root_pid_alive": True,
            "owner_pid_start_time_match": False,
            "reason": "root_pid_start_time_mismatch",
        }
    )

    assert lifecycle.status == "stale_pid_binding"
    assert lifecycle.safe_to_mark_finished is True
    assert lifecycle.safe_to_terminate_pid is False


def test_lifecycle_completed_but_waiting() -> None:
    lifecycle = classify_process_lifecycle(
        process_liveness={
            "status": "exited",
            "root_pid_alive": False,
            "process_group_alive": False,
            "live_descendant_count": 0,
            "descendant_count": 0,
            "exit_code_seen": True,
        },
        tool_waiting=True,
    )

    assert lifecycle.status == "completed_but_waiting"
    assert lifecycle.safe_to_mark_finished is True


def test_lifecycle_zombie_only_is_not_active_work() -> None:
    lifecycle = classify_process_lifecycle(
        process_liveness={
            "status": "unknown",
            "root_pid_alive": False,
            "process_group_alive": False,
            "descendant_count": 2,
            "live_descendant_count": 0,
            "zombie_descendant_count": 2,
        }
    )

    assert lifecycle.status == "zombie_only"
    assert lifecycle.active_work_detected is False
