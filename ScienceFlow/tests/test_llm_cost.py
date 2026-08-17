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

from pathlib import Path

from scienceflow.config.settings import load_cfg
from scienceflow.ui.monitor.helpers import _fmt_tokens_row
from scienceflow.utils.llm_cost import (
    LLMPrice,
    estimate_llm_cost_usd,
    extract_model_from_trace_detail,
    load_price_table,
)
from scienceflow.utils.time_trace import TimeTracer, TRACE_FILENAME


def test_estimate_llm_cost_counts_cached_input_discount() -> None:
    cost = estimate_llm_cost_usd(
        model="demo",
        tokens_input=1000,
        tokens_output=100,
        tokens_cached=800,
        price_table={"demo": LLMPrice(10.0, 1.0, 20.0)},
    )

    assert cost == ((200 * 10.0) + (800 * 1.0) + (100 * 20.0)) / 1_000_000


def test_default_config_contains_deepseek_prices() -> None:
    cfg = load_cfg(Path("scienceflow/config/default.yaml"), cli_args=False)

    assert cfg.agent.llm_prices["deepseek-v4-flash"]["cached_input_usd_per_1m"] == 0.0028
    assert cfg.agent.llm_prices["deepseek-v4-flash"]["input_usd_per_1m"] == 0.14
    assert cfg.agent.llm_prices["deepseek-v4-pro"]["output_usd_per_1m"] == 0.87


def test_load_price_table_uses_config_prices() -> None:
    prices = load_price_table({"config-model": {"input": 3, "cached_input": 0.3, "output": 9}})

    assert prices["config-model"] == LLMPrice(3.0, 0.3, 9.0)


def test_extract_model_from_trace_detail() -> None:
    assert extract_model_from_trace_detail("role=main;model=deepseek-v4-flash;token_scope=llm_api") == "deepseek-v4-flash"


def test_time_trace_writes_llm_cost_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "SCIENCEFLOW_LLM_PRICE_TABLE_JSON",
        '{"demo-model":{"input":10,"cached_input":1,"output":20}}',
    )
    tracer = TimeTracer(tmp_path, enabled=True)

    tracer.record(
        "llm_api",
        "ask_tool_stream",
        1.0,
        tokens_input=1000,
        tokens_output=100,
        tokens_cached=800,
        detail="model=demo-model",
    )

    text = (tmp_path / TRACE_FILENAME).read_text(encoding="utf-8")
    assert "llm_cost_usd" in text
    assert "0.00480000" in text


def test_time_trace_writes_llm_cost_from_config(tmp_path: Path) -> None:
    tracer = TimeTracer(
        tmp_path,
        enabled=True,
        price_table_config={"config-model": {"input": 10, "cached_input": 1, "output": 20}},
    )

    tracer.record(
        "llm_api",
        "ask_tool_stream",
        1.0,
        tokens_input=1000,
        tokens_output=100,
        tokens_cached=800,
        detail="model=config-model",
    )

    text = (tmp_path / TRACE_FILENAME).read_text(encoding="utf-8")
    assert "0.00480000" in text


def test_monitor_token_row_includes_cost() -> None:
    row = _fmt_tokens_row(
        {
            "total_tokens_in": 1000,
            "total_tokens_out": 100,
            "total_tokens_cached": 900,
            "total_llm_calls": 2,
            "llm_cache_rate": 0.9,
            "total_llm_cost_usd": 0.0048,
        }
    )

    assert "0.00M/0.00M c=2" in row
    assert "$0.0048" in row
    assert "cache=0.90" in row
