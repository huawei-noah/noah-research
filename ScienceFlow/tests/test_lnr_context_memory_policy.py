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

from types import SimpleNamespace

from scienceflow.solver.lnr.context_memory_policy import ensure_lnr_context_memory_budget


def test_lnr_context_limit_raises_message_floor() -> None:
    cfg = SimpleNamespace(max_messages=240)
    lhr = SimpleNamespace(compact_on_context_limit=True, context_limit_min_messages=2000)

    result = ensure_lnr_context_memory_budget(cfg, lhr)

    assert result.changed is True
    assert result.original_max_messages == 240
    assert result.effective_max_messages == 2000
    assert result.reason == "context_limit_memory_floor"
    assert cfg.max_messages == 2000


def test_lnr_context_limit_keeps_larger_message_budget() -> None:
    cfg = SimpleNamespace(max_messages=5000)
    lhr = SimpleNamespace(compact_on_context_limit=True, context_limit_min_messages=2000)

    result = ensure_lnr_context_memory_budget(cfg, lhr)

    assert result.changed is False
    assert result.effective_max_messages == 5000
    assert result.reason == "already_sufficient"
    assert cfg.max_messages == 5000


def test_lnr_context_limit_floor_disabled_when_compact_disabled() -> None:
    cfg = SimpleNamespace(max_messages=240)
    lhr = SimpleNamespace(compact_on_context_limit=False, context_limit_min_messages=2000)

    result = ensure_lnr_context_memory_budget(cfg, lhr)

    assert result.changed is False
    assert result.reason == "compact_on_context_limit_disabled"
    assert cfg.max_messages == 240


def test_prep_cfg_applies_lnr_message_floor_before_resolved_config(tmp_path) -> None:
    from scienceflow.config.settings import Config, prep_cfg

    cfg = Config()
    cfg.task_workspace_root_dir = tmp_path
    cfg.max_messages = 240
    cfg.lnr.compact_on_context_limit = True
    cfg.lnr.context_limit_min_messages = 5000

    prep_cfg(cfg)

    assert cfg.max_messages == 5000
    resolved = (tmp_path / "resolved_config.yaml").read_text()
    assert "max_messages: 5000" in resolved
