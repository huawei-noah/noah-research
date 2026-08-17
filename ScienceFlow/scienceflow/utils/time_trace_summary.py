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
from pathlib import Path


def is_llm_api_token_row(row: dict[str, object]) -> bool:
    """True only for real per-call LLM trace rows.

    Cache-rate summaries should use this by default. LNR phase rows such as
    ``shallow_step`` / ``deep_execute`` can carry nested token deltas, but they
    are aggregate accounting rows rather than independent provider API calls.
    """
    return str(row.get("category") or "") == "llm_api"


def summarize_time_trace_cache(
    path: Path | str,
    *,
    llm_api_only: bool = True,
) -> dict[str, int | float | None]:
    """Summarize token/cache usage from a ScienceFlow time-trace CSV.

    Defaults to true LLM API rows only, which is the right denominator for
    provider KV-cache hit rate. Set ``llm_api_only=False`` only for legacy
    aggregate token accounting where cache-rate is not meaningful.
    """
    p = Path(path)
    calls = 0
    tokens_input = 0
    tokens_output = 0
    tokens_cached = 0
    if not p.is_file():
        return {
            "calls": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "tokens_cached": 0,
            "cache_rate": None,
        }
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if llm_api_only and not is_llm_api_token_row(row):
                continue
            try:
                ti = int(float(row.get("tokens_input") or 0))
                to = int(float(row.get("tokens_output") or 0))
                tc = int(float(row.get("tokens_cached") or 0))
            except (TypeError, ValueError):
                continue
            if ti == 0 and to == 0 and tc == 0:
                continue
            calls += 1
            tokens_input += ti
            tokens_output += to
            tokens_cached += tc
    return {
        "calls": calls,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_cached": tokens_cached,
        "cache_rate": (tokens_cached / tokens_input if tokens_input else None),
    }


def _empty_cache_summary() -> dict[str, int | float | None]:
    return {
        "calls": 0,
        "tokens_input": 0,
        "tokens_output": 0,
        "tokens_cached": 0,
        "cache_rate": None,
    }


def _add_cache_summary(
    summary: dict[str, int | float | None],
    *,
    tokens_input: int,
    tokens_output: int,
    tokens_cached: int,
) -> None:
    summary["calls"] = int(summary.get("calls") or 0) + 1
    summary["tokens_input"] = int(summary.get("tokens_input") or 0) + tokens_input
    summary["tokens_output"] = int(summary.get("tokens_output") or 0) + tokens_output
    summary["tokens_cached"] = int(summary.get("tokens_cached") or 0) + tokens_cached
    total_input = int(summary.get("tokens_input") or 0)
    summary["cache_rate"] = (
        int(summary.get("tokens_cached") or 0) / total_input if total_input else None
    )


def classify_time_trace_llm_cache_bucket(row: dict[str, object]) -> str:
    """Classify true LLM rows for cache reporting.

    ``code_non_route`` is the main agent cache metric. Agentic route prompts,
    long-horizon stage-commit bookkeeping, and feedback/result summarization
    prompts are intentionally separate so controller side channels cannot drag
    down the main agent cache rate.
    """
    detail = str(row.get("detail") or "")
    operation = str(row.get("operation") or "")
    if "llm_role=feedback" in detail:
        return "feedback"
    if operation == "lnr_stage_commit" or "turn=stage_commit" in detail:
        return "stage_commit"
    if operation == "agentic_route" or operation == "lnr_route" or "turn=route" in detail:
        return "route"
    if "llm_role=code" in detail:
        return "code_non_route"
    return "other"


def _trace_detail_int(detail: str, key: str) -> int | None:
    marker = f"{key}="
    for part in str(detail or "").split(";"):
        part = part.strip()
        if not part.startswith(marker):
            continue
        raw = part[len(marker) :].strip()
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def summarize_time_trace_cache_splits(
    path: Path | str,
) -> dict[str, dict[str, int | float | None]]:
    """Summarize LLM cache by role bucket from a ScienceFlow time-trace CSV.

    Buckets:
    - ``all_llm_api``: all real provider calls.
    - ``code_non_route``: main agent calls, excluding route prompts.
    - ``stage_commit``: lnr stage ledger bookkeeping calls.
    - ``route``: agentic route decision prompts.
    - ``feedback``: feedback/result summary LLM calls.
    - ``other``: true LLM calls without a recognized role marker.
    """
    p = Path(path)
    buckets = {
        name: _empty_cache_summary()
        for name in (
            "all_llm_api",
            "code_non_route",
            "code_non_route_seq1",
            "code_non_route_seq2plus",
            "stage_commit",
            "route",
            "feedback",
            "other",
        )
    }
    if not p.is_file():
        return buckets
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if not is_llm_api_token_row(row):
                continue
            try:
                ti = int(float(row.get("tokens_input") or 0))
                to = int(float(row.get("tokens_output") or 0))
                tc = int(float(row.get("tokens_cached") or 0))
            except (TypeError, ValueError):
                continue
            if ti == 0 and to == 0 and tc == 0:
                continue
            _add_cache_summary(
                buckets["all_llm_api"],
                tokens_input=ti,
                tokens_output=to,
                tokens_cached=tc,
            )
            bucket = classify_time_trace_llm_cache_bucket(row)
            _add_cache_summary(
                buckets[bucket],
                tokens_input=ti,
                tokens_output=to,
                tokens_cached=tc,
            )
            if bucket == "code_non_route":
                seq = _trace_detail_int(str(row.get("detail") or ""), "call_seq")
                seq_bucket = (
                    "code_non_route_seq1" if seq == 1 else "code_non_route_seq2plus"
                )
                _add_cache_summary(
                    buckets[seq_bucket],
                    tokens_input=ti,
                    tokens_output=to,
                    tokens_cached=tc,
                )
    return buckets
