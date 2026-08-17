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

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMPrice:
    """USD rates per 1M text tokens."""

    input_usd_per_1m: float
    cached_input_usd_per_1m: float
    output_usd_per_1m: float


# DeepSeek official API pricing, per 1M tokens, checked against
# https://api-docs.deepseek.com/quick_start/pricing on 2026-06-19.
DEFAULT_PRICE_TABLE: dict[str, LLMPrice] = {
    "deepseek-v4-flash": LLMPrice(0.14, 0.0028, 0.28),
    "deepseek-chat": LLMPrice(0.14, 0.0028, 0.28),
    "deepseek-reasoner": LLMPrice(0.14, 0.0028, 0.28),
    "deepseek-v4-pro": LLMPrice(0.435, 0.003625, 0.87),
}

_MODEL_RE = re.compile(r"(?:^|;)model=([^;]+)")


def extract_model_from_trace_detail(detail: str | None) -> str | None:
    match = _MODEL_RE.search(str(detail or ""))
    if not match:
        return None
    model = match.group(1).strip()
    return model or None


def _norm_model_name(model: str | None) -> str:
    return str(model or "").strip().lower()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _price_from_obj(obj: Any) -> LLMPrice | None:
    if isinstance(obj, LLMPrice):
        return obj
    if not isinstance(obj, dict):
        return None
    input_rate = _float_or_none(
        obj.get("input_usd_per_1m")
        or obj.get("input")
        or obj.get("cache_miss_usd_per_1m")
        or obj.get("uncached_input_usd_per_1m")
    )
    cached_rate = _float_or_none(
        obj.get("cached_input_usd_per_1m")
        or obj.get("cached_input")
        or obj.get("cache_hit_usd_per_1m")
        or obj.get("cache")
    )
    output_rate = _float_or_none(obj.get("output_usd_per_1m") or obj.get("output"))
    if input_rate is None or cached_rate is None or output_rate is None:
        return None
    return LLMPrice(input_rate, cached_rate, output_rate)


def _merge_price_table(table: dict[str, LLMPrice], raw: Any) -> None:
    if not isinstance(raw, dict):
        return
    for model, obj in raw.items():
        price = _price_from_obj(obj)
        if price is not None:
            table[_norm_model_name(model)] = price


def load_price_table(config_prices: Any = None) -> dict[str, LLMPrice]:
    """Return model price table, with config first and env overrides last."""
    table = dict(DEFAULT_PRICE_TABLE)
    _merge_price_table(table, config_prices)

    raw = os.environ.get("SCIENCEFLOW_LLM_PRICE_TABLE_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        _merge_price_table(table, parsed)

    fallback = _price_from_obj(
        {
            "input": os.environ.get("SCIENCEFLOW_LLM_FALLBACK_INPUT_USD_PER_1M"),
            "cached_input": os.environ.get("SCIENCEFLOW_LLM_FALLBACK_CACHED_INPUT_USD_PER_1M"),
            "output": os.environ.get("SCIENCEFLOW_LLM_FALLBACK_OUTPUT_USD_PER_1M"),
        }
    )
    if fallback is not None:
        table["*"] = fallback
    return table


def resolve_price(model: str | None, table: dict[str, LLMPrice] | None = None) -> LLMPrice | None:
    prices = table if table is not None else load_price_table()
    name = _norm_model_name(model)
    if name in prices:
        return prices[name]
    if name:
        for key, price in prices.items():
            if key != "*" and key in name:
                return price
    return prices.get("*")


def estimate_llm_cost_usd(
    *,
    model: str | None,
    tokens_input: int | float | str | None,
    tokens_output: int | float | str | None,
    tokens_cached: int | float | str | None = 0,
    price_table: dict[str, LLMPrice] | None = None,
) -> float | None:
    price = resolve_price(model, price_table)
    if price is None:
        return None
    try:
        input_tokens = max(0, int(float(tokens_input or 0)))
        output_tokens = max(0, int(float(tokens_output or 0)))
        cached_tokens = max(0, int(float(tokens_cached or 0)))
    except (TypeError, ValueError):
        return None
    cached_tokens = min(cached_tokens, input_tokens)
    uncached_tokens = max(0, input_tokens - cached_tokens)
    return (
        uncached_tokens * price.input_usd_per_1m
        + cached_tokens * price.cached_input_usd_per_1m
        + output_tokens * price.output_usd_per_1m
    ) / 1_000_000.0


def format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 100:
        return f"${value:.2f}"
    return f"${value:,.0f}"
