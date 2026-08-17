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

"""LLM call tracing and stream consumption."""

from __future__ import annotations

import logging
import sys
from typing import Any

from deepcraft_core.llm import StreamHandle

from scienceflow.utils.workspace_interaction_log import write_raw_to_interaction_log

logger = logging.getLogger("scienceflow")


class RepetitionDetector:
    """Lightweight sliding-window repetition detector for streamed text."""

    def __init__(self, *, window_chars: int, ngram_len: int, max_repeats: int) -> None:
        self._window_chars = max(256, int(window_chars))
        self._ngram_len = max(32, int(ngram_len))
        self._max_repeats = max(2, int(max_repeats))
        self._buf = ""

    def feed(self, chunk: str) -> bool:
        if not chunk:
            return False
        self._buf += chunk
        if len(self._buf) > self._window_chars:
            self._buf = self._buf[-self._window_chars :]
        if len(self._buf) < self._ngram_len * 2:
            return False
        tail = self._buf[-self._ngram_len :]
        return self._buf.count(tail) >= self._max_repeats


class LLMStreamMixin:
    """Record per-call stats and print streamed assistant text."""

    def _record_llm_call(
        self,
        operation: str,
        duration_sec: float,
        round_idx: int | None,
        status: str = "ok",
        *,
        recovery: bool = False,
        turn_kind: str | None = None,
        first_tool_name: str | None = None,
        first_bash_kind: str | None = None,
        llm_override: Any | None = None,
        llm_role: str | None = None,
    ) -> None:
        """Invoke ``on_llm_call`` with one row of per-call stats (must not raise)."""
        if self._on_llm_call is None:
            return
        self._call_seq += 1
        ti: int | None = None
        to: int | None = None
        ttft: float | None = None
        tpot: float | None = None
        tc: int | None = None
        pool_index: int | None = None
        failover_count: int | None = None
        source_llm = llm_override or self.llm
        if status == "ok":
            try:
                _ti = getattr(source_llm, "_last_call_input_tokens", None)
                _to = getattr(source_llm, "_last_call_output_tokens", None)
                _ttft = getattr(source_llm, "_last_call_ttft", None)
                _tpot = getattr(source_llm, "_last_call_tpot", None)
                _tc = getattr(source_llm, "_last_call_input_cached_tokens", None)
                _pool_index = getattr(source_llm, "_last_call_pool_index", None)
                _failover_count = getattr(source_llm, "_last_call_failover_count", None)
                if _ti is not None:
                    ti = int(_ti)
                if _to is not None:
                    to = int(_to)
                if _ttft is not None:
                    ttft = float(_ttft)
                if _tpot is not None:
                    tpot = float(_tpot)
                if _tc is not None:
                    tc = int(_tc)
                if _pool_index is not None:
                    pool_index = int(_pool_index)
                if _failover_count is not None:
                    failover_count = int(_failover_count)
            except (TypeError, ValueError):
                pass
        model = getattr(source_llm, "model", None)
        model_s = str(model) if model is not None else None
        payload: dict[str, Any] = {
            "operation": operation,
            "duration_sec": float(duration_sec),
            "call_seq": self._call_seq,
            "round_idx": round_idx,
            "recovery": recovery,
            "tokens_input": ti,
            "tokens_output": to,
            "tokens_cached": tc,
            "ttft_sec": ttft,
            "tpot_ms": tpot,
            "pool_index": pool_index,
            "failover_count": failover_count,
            "model": model_s,
            "status": status,
        }
        if turn_kind is not None:
            payload["turn_kind"] = turn_kind
        if llm_role is not None:
            payload["llm_role"] = llm_role
        if first_tool_name is not None:
            payload["first_tool_name"] = first_tool_name
        if first_bash_kind is not None:
            payload["first_bash_kind"] = first_bash_kind
        try:
            self._on_llm_call(payload)
        except Exception:
            logger.debug("on_llm_call hook failed", exc_info=True)

    async def _consume_stream(
        self,
        handle: StreamHandle,
        interaction_logger: logging.Logger | None = None,
    ) -> None:
        """Print streamed assistant text in dim on TTY; mirror chunks to interaction log.

        Raw chunks are appended to workspace log files (no timestamp) so ``tail -f`` /
        editors can show tokens as they arrive.
        """
        _tty = sys.stdout.isatty()
        if _tty:
            sys.stdout.write("\033[2m")
            sys.stdout.flush()
        wrote_any = False
        last_ends_nl = True
        total_chars = 0
        detector: RepetitionDetector | None = None
        if (
            bool(getattr(self, "_stream_repetition_detection", False))
            and int(getattr(self, "_stream_repetition_ngram_len", 0) or 0) > 0
            and int(getattr(self, "_stream_repetition_max_repeats", 0) or 0) > 1
        ):
            detector = RepetitionDetector(
                window_chars=int(getattr(self, "_stream_repetition_window_chars", 4000) or 4000),
                ngram_len=int(getattr(self, "_stream_repetition_ngram_len", 150) or 150),
                max_repeats=int(getattr(self, "_stream_repetition_max_repeats", 3) or 3),
            )
        try:
            while True:
                chunk = await handle.queue.get()
                if chunk is None:
                    break
                if not chunk:
                    continue
                total_chars += len(chunk)
                soft_limit = int(getattr(self, "_stream_max_output_chars_soft", 0) or 0)
                if soft_limit > 0 and total_chars > soft_limit:
                    setattr(self, "_last_stream_guard_reason", "output_soft_limit")
                    setattr(self, "_last_stream_guard_chars", total_chars)
                    setattr(self, "_last_stream_guard_detail", f"chars={total_chars}>{soft_limit}")
                    logger.warning(
                        "[llm-guard] interrupt stream by soft output limit: chars=%d limit=%d",
                        total_chars,
                        soft_limit,
                    )
                    if interaction_logger is not None:
                        write_raw_to_interaction_log(
                            interaction_logger,
                            f"\n[llm-guard] output_soft_limit chars={total_chars}>{soft_limit}\n",
                        )
                    handle.interrupt()
                    break
                if detector is not None and detector.feed(chunk):
                    setattr(self, "_last_stream_guard_reason", "repetition")
                    setattr(self, "_last_stream_guard_chars", total_chars)
                    setattr(self, "_last_stream_guard_detail", "repeated_tail_ngram")
                    logger.warning(
                        "[llm-guard] interrupt stream by repetition detector: chars=%d",
                        total_chars,
                    )
                    if interaction_logger is not None:
                        write_raw_to_interaction_log(
                            interaction_logger,
                            "\n[llm-guard] repetition_detected\n",
                        )
                    handle.interrupt()
                    break
                wrote_any = True
                sys.stdout.write(chunk)
                sys.stdout.flush()
                if interaction_logger is not None:
                    write_raw_to_interaction_log(interaction_logger, chunk)
                last_ends_nl = chunk.endswith("\n")
        finally:
            if wrote_any and not last_ends_nl:
                sys.stdout.write("\n")
                if interaction_logger is not None:
                    write_raw_to_interaction_log(interaction_logger, "\n")
            if _tty:
                sys.stdout.write("\033[0m")
            sys.stdout.flush()
