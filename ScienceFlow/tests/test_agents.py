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

from scienceflow.core.agent.registry import (
    _AGENT_REGISTRY,
    get_agent_class,
    list_agents,
    register_agent,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_AGENT_REGISTRY)
    _AGENT_REGISTRY.clear()
    yield
    _AGENT_REGISTRY.clear()
    _AGENT_REGISTRY.update(saved)


class TestRegisterAgent:

    def test_decorator_registers(self):
        @register_agent("test_agent")
        class TestAgent:
            pass

        assert "test_agent" in _AGENT_REGISTRY
        assert _AGENT_REGISTRY["test_agent"] is TestAgent

    def test_decorator_returns_original_class(self):
        @register_agent("alpha")
        class Alpha:
            pass

        assert Alpha.__name__ == "Alpha"

    def test_multiple_registrations(self):
        @register_agent("a")
        class A:
            pass

        @register_agent("b")
        class B:
            pass

        assert set(_AGENT_REGISTRY.keys()) == {"a", "b"}


class TestGetAgentClass:

    def test_returns_correct_class(self):
        @register_agent("my_agent")
        class MyAgent:
            pass

        assert get_agent_class("my_agent") is MyAgent

    def test_missing_agent_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_agent_class("nonexistent")


class TestListAgents:

    def test_empty_initially(self):
        assert list_agents() == []

    def test_lists_registered(self):
        @register_agent("x")
        class X:
            pass

        @register_agent("y")
        class Y:
            pass

        assert sorted(list_agents()) == ["x", "y"]
