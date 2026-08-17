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

import csv
import time
import logging
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager

from scienceflow.utils.llm_cost import LLMPrice, estimate_llm_cost_usd, extract_model_from_trace_detail, load_price_table
from scienceflow.utils.time_trace_summary import (
    classify_time_trace_llm_cache_bucket,
    is_llm_api_token_row,
    summarize_time_trace_cache,
    summarize_time_trace_cache_splits,
)

logger = logging.getLogger("scienceflow")

TRACE_FILENAME = "scienceflow_time_trace.csv"

# CSV ``detail`` hint: phase wall time may include nested ``llm_api`` rows (do not sum durations).
SKILL_EXEC_SCOPE_HINT = "scope=phase_wall_includes_llm_children"
TOKEN_SCOPE_LLM_API = "token_scope=llm_api"
TOKEN_SCOPE_AGGREGATE = "token_scope=aggregate"


def _has_token_payload(
    tokens_input: int | None,
    tokens_output: int | None,
    tokens_cached: int | None,
) -> bool:
    return (
        tokens_input is not None
        or tokens_output is not None
        or tokens_cached is not None
    )


def _annotate_token_scope(
    category: str,
    detail: str,
    *,
    tokens_input: int | None,
    tokens_output: int | None,
    tokens_cached: int | None,
    max_len: int = 200,
) -> str:
    """Tag token-bearing rows as true LLM calls or aggregate phase accounting."""
    base = (detail or "").replace("\n", " ")
    if "token_scope=" in base or not _has_token_payload(
        tokens_input,
        tokens_output,
        tokens_cached,
    ):
        return base[:max_len]
    scope = TOKEN_SCOPE_LLM_API if category == "llm_api" else TOKEN_SCOPE_AGGREGATE
    return merge_trace_details(base, scope, max_len=max_len)


def trace_detail_with_skill(base: str, skill_name: str, *, max_len: int = 200) -> str:
    """Append ``skill=<name>`` to trace ``detail`` (truncated for CSV)."""
    name = (skill_name or "").strip()
    if not name:
        return (base or "").replace("\n", " ")[:max_len]
    extra = f"skill={name}"
    merged = f"{base};{extra}" if base else extra
    return merged.replace("\n", " ")[:max_len]


def merge_trace_details(*parts: str, max_len: int = 200) -> str:
    """Join non-empty trace detail fragments with ``;`` (truncated for CSV)."""
    s = ";".join(p.strip() for p in parts if p and str(p).strip())
    return s.replace("\n", " ")[:max_len]


def format_injected_skills_detail(names: list[str], max_len: int = 200) -> str:
    """``injected=a,b,c`` and ``injected_n=N`` for prompt-injected skills (truncated)."""
    clean = [str(x).strip() for x in names if str(x).strip()]
    n = len(clean)
    counter = f"injected_n={n}"
    if n == 0:
        return merge_trace_details("injected=", counter, max_len=max_len)
    joined = ",".join(clean)
    prefix = "injected="
    suffix = f";{counter}"
    budget = max_len - len(prefix) - len(suffix)
    if budget <= 0:
        return counter[:max_len]
    if len(joined) <= budget:
        return f"{prefix}{joined}{suffix}"[:max_len]
    truncated = joined[: max(0, budget - 3)] + "..."
    return f"{prefix}{truncated}{suffix}"[:max_len]


TRACE_COLUMNS = [
    "timestamp",
    "node_id",
    "process_id",
    "fork_class",
    "category",
    "operation",
    "duration_hrs",
    "tokens_input",
    "tokens_output",
    "tokens_cached",
    "tokens_uncached",
    "token_cached_rate",
    "llm_cost_usd",
    "ttft_sec",
    "tpot_ms",
    "pool_index",
    "failover_count",
    "status",
    "detail",
]

_EMPTY_CTX: dict[str, None] = {
    "tokens_input": None,
    "tokens_output": None,
    "tokens_cached": None,
    "ttft_sec": None,
    "tpot_ms": None,
    "pool_index": None,
    "failover_count": None,
}


def _append_row(log_dir: Path, row: dict) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / TRACE_FILENAME
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _build_row(
    category: str,
    operation: str,
    duration_sec: float,
    status: str = "ok",
    node_id: int | str | None = None,
    process_id: int | str | None = None,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    tokens_cached: int | None = None,
    ttft_sec: float | None = None,
    tpot_ms: float | None = None,
    pool_index: int | None = None,
    failover_count: int | None = None,
    detail: str = "",
    fork_class: str | None = None,
    price_table: dict[str, LLMPrice] | None = None,
) -> dict:
    token_cached_rate = ""
    tokens_uncached = ""
    llm_cost_usd = ""
    if category == "llm_api" and tokens_input is not None and tokens_cached is not None:
        try:
            tokens_input_i = int(tokens_input)
            tokens_cached_i = int(tokens_cached)
            if tokens_input_i > 0:
                token_cached_rate = f"{tokens_cached_i / tokens_input_i:.6f}"
                tokens_uncached = str(max(0, tokens_input_i - tokens_cached_i))
            cost = estimate_llm_cost_usd(
                model=extract_model_from_trace_detail(detail),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_cached=tokens_cached,
                price_table=price_table,
            )
            if cost is not None:
                llm_cost_usd = f"{cost:.8f}"
        except (TypeError, ValueError, ZeroDivisionError):
            token_cached_rate = ""
            tokens_uncached = ""
            llm_cost_usd = ""
    detail = _annotate_token_scope(
        category,
        detail,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_cached=tokens_cached,
    )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node_id": str(node_id) if node_id is not None else "",
        "process_id": str(process_id) if process_id is not None else "",
        "fork_class": str(fork_class) if fork_class else "",
        "category": category,
        "operation": operation,
        "duration_hrs": f"{duration_sec / 3600.0:.6f}",
        "tokens_input": str(int(tokens_input)) if tokens_input is not None else "",
        "tokens_output": str(int(tokens_output)) if tokens_output is not None else "",
        "tokens_cached": str(int(tokens_cached)) if tokens_cached is not None else "",
        "tokens_uncached": tokens_uncached,
        "token_cached_rate": token_cached_rate,
        "llm_cost_usd": llm_cost_usd,
        "ttft_sec": f"{ttft_sec:.3f}" if ttft_sec is not None else "",
        "tpot_ms": f"{tpot_ms:.2f}" if tpot_ms is not None else "",
        "pool_index": str(int(pool_index)) if pool_index is not None else "",
        "failover_count": str(int(failover_count)) if failover_count is not None else "",
        "status": status,
        "detail": detail.replace("\n", " ")[:200] if detail else "",
    }


class TimeTracer:
    def __init__(self, log_dir: Path, enabled: bool = True, price_table_config: object | None = None):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.price_table = load_price_table(price_table_config)

    def record(
        self,
        category: str,
        operation: str,
        duration_sec: float,
        status: str = "ok",
        node_id: int | str | None = None,
        process_id: int | str | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        tokens_cached: int | None = None,
        ttft_sec: float | None = None,
        tpot_ms: float | None = None,
        pool_index: int | None = None,
        failover_count: int | None = None,
        detail: str = "",
        fork_class: str | None = None,
    ):
        if not self.enabled:
            return
        row = _build_row(
            category, operation, duration_sec, status, node_id, process_id,
            tokens_input, tokens_output, tokens_cached, ttft_sec, tpot_ms,
            pool_index=pool_index,
            failover_count=failover_count,
            detail=detail,
            fork_class=fork_class,
            price_table=self.price_table,
        )
        try:
            _append_row(self.log_dir, row)
        except Exception as e:
            logger.debug(f"time_trace write failed: {e}")

    @contextmanager
    def track(
        self,
        category: str,
        operation: str,
        node_id: int | str | None = None,
        process_id: int | str | None = None,
        detail: str = "",
    ):
        if not self.enabled:
            yield _EMPTY_CTX
            return
        t0 = time.time()
        status = "ok"
        err_msg = ""
        ctx: dict = {
            "tokens_input": None,
            "tokens_output": None,
            "tokens_cached": None,
            "ttft_sec": None,
            "tpot_ms": None,
            "pool_index": None,
            "failover_count": None,
        }
        try:
            yield ctx
        except Exception as exc:
            status = "error"
            err_msg = str(exc)
            raise
        finally:
            row_detail = (
                detail if status == "ok" else (f"{detail} | {err_msg}" if detail else err_msg)
            )
            self.record(
                category, operation, time.time() - t0, status, node_id, process_id,
                ctx.get("tokens_input"), ctx.get("tokens_output"), ctx.get("tokens_cached"),
                ctx.get("ttft_sec"), ctx.get("tpot_ms"),
                pool_index=ctx.get("pool_index"),
                failover_count=ctx.get("failover_count"),
                detail=row_detail,
            )

    @asynccontextmanager
    async def atrack(
        self,
        category: str,
        operation: str,
        node_id: int | str | None = None,
        process_id: int | str | None = None,
        detail: str = "",
    ):
        if not self.enabled:
            yield _EMPTY_CTX
            return
        t0 = time.time()
        status = "ok"
        err_msg = ""
        ctx: dict = {
            "tokens_input": None,
            "tokens_output": None,
            "tokens_cached": None,
            "ttft_sec": None,
            "tpot_ms": None,
            "pool_index": None,
            "failover_count": None,
        }
        try:
            yield ctx
        except Exception as exc:
            status = "error"
            err_msg = str(exc)
            raise
        finally:
            row_detail = (
                detail if status == "ok" else (f"{detail} | {err_msg}" if detail else err_msg)
            )
            self.record(
                category, operation, time.time() - t0, status, node_id, process_id,
                ctx.get("tokens_input"), ctx.get("tokens_output"), ctx.get("tokens_cached"),
                ctx.get("ttft_sec"), ctx.get("tpot_ms"),
                pool_index=ctx.get("pool_index"),
                failover_count=ctx.get("failover_count"),
                detail=row_detail,
            )
