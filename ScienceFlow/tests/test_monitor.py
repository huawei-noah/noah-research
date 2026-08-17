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

import pytest

from scienceflow.core.monitor.process_tracker import ProcessInfo, ProcessStatus, ProcessTracker
from scienceflow.core.monitor.cpu_monitor import CPUMetrics
from scienceflow.core.monitor.gpu_monitor import GPUMetrics


class TestProcessTracker:

    @pytest.fixture()
    def tracker(self):
        return ProcessTracker()

    def test_register(self, tracker):
        info = tracker.register(pid=100, node_id="n1")
        assert isinstance(info, ProcessInfo)
        assert info.pid == 100
        assert info.node_id == "n1"
        assert info.status == ProcessStatus.RUNNING

    def test_mark_complete(self, tracker):
        tracker.register(pid=200, node_id="n2")
        info = tracker.mark_complete(pid=200, exit_code=0)
        assert info.status == ProcessStatus.COMPLETED
        assert info.exit_code == 0
        assert info.end_time is not None

    def test_mark_complete_unknown_pid(self, tracker):
        assert tracker.mark_complete(pid=999, exit_code=0) is None

    def test_mark_killed(self, tracker):
        tracker.register(pid=300, node_id="n3")
        info = tracker.mark_killed(pid=300)
        assert info.status == ProcessStatus.KILLED

    def test_mark_timeout(self, tracker):
        tracker.register(pid=400, node_id="n4")
        info = tracker.mark_timeout(pid=400)
        assert info.status == ProcessStatus.TIMEOUT

    def test_get_active(self, tracker):
        tracker.register(pid=1, node_id="a")
        tracker.register(pid=2, node_id="b")
        tracker.mark_complete(pid=1, exit_code=0)
        active = tracker.get_active()
        assert len(active) == 1
        assert active[0].pid == 2

    def test_get_by_node(self, tracker):
        tracker.register(pid=10, node_id="same")
        tracker.register(pid=20, node_id="same")
        tracker.register(pid=30, node_id="other")
        assert len(tracker.get_by_node("same")) == 2

    def test_get(self, tracker):
        tracker.register(pid=42, node_id="x")
        assert tracker.get(42).node_id == "x"
        assert tracker.get(999) is None

    def test_all_processes(self, tracker):
        tracker.register(pid=1, node_id="a")
        tracker.register(pid=2, node_id="b")
        assert len(tracker.all_processes) == 2


class TestCPUMetrics:

    def test_defaults(self):
        m = CPUMetrics()
        assert m.cpu_percent == 0.0
        assert m.memory_percent == 0.0
        assert m.memory_used_gb == 0.0
        assert m.io_read_bytes == 0
        assert m.io_write_bytes == 0

    def test_custom_values(self):
        m = CPUMetrics(cpu_percent=75.5, memory_used_gb=8.2)
        assert m.cpu_percent == pytest.approx(75.5)
        assert m.memory_used_gb == pytest.approx(8.2)


class TestGPUMetrics:

    def test_defaults(self):
        m = GPUMetrics()
        assert m.gpu_id == 0
        assert m.utilization == 0.0
        assert m.memory_used == 0
        assert m.temperature == 0

    def test_custom_values(self):
        m = GPUMetrics(gpu_id=1, utilization=95.0, memory_used=8000, memory_total=16000)
        assert m.gpu_id == 1
        assert m.utilization == pytest.approx(95.0)
        assert m.memory_used == 8000
        assert m.memory_total == 16000
