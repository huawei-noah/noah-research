"""LongMemEval benchmark matrix adapter."""

from __future__ import annotations

import argparse
from typing import Any

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext
from mindmemos_eval.memory.config import _merged_runner_config, _option, resolve_public_search_strategy
from .env import LongMemEvalEnv


class LongMemEvalAdapter:
    """LongMemEval adapter backed by :class:`LongMemEvalEnv`."""

    name = "longmemeval"

    async def run(
        self,
        *,
        memory: Any,
        answer_llm: LLMClient,
        judge_llm: LLMClient,
        ctx: RunContext,
        bench_config: BenchmarkSpec,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Run LongMemEval using the dedicated eval environment."""
        runner = getattr(args, "runner_config", None)
        if runner is None:
            runner = _merged_runner_config(args)

        data = LongMemEvalEnv.load_dataset(bench_config.dataset)
        limit = bench_config.limit if bench_config.limit is not None else _option(args, "limit")
        if limit is not None:
            data = data[: int(limit)]

        search_params = bench_config.search_params
        public_search_strategy = resolve_public_search_strategy(
            search_params.get("public_search_strategy")
            or ("agentic" if search_params.get("agentic") is True else None)
            or runner.search_strategy
            or "fast"
        )
        top_k = search_params["top_k"] if "top_k" in search_params else runner.top_k
        rerank = search_params["rerank"] if "rerank" in search_params else runner.rerank
        env = LongMemEvalEnv(
            memory,
            answer_llm=answer_llm,
            judge_llm=judge_llm,
            top_k=None if top_k is None else int(top_k),
            search_strategy=public_search_strategy,
            rerank=bool(rerank),
        )
        run = await env.run_dataset(
            data,
            max_sample_concurrency=runner.max_conv_concurrency,
            max_qa_concurrency=runner.max_qa_concurrency,
            max_search_concurrency=runner.max_search_concurrency,
            max_score_concurrency=runner.max_score_concurrency,
            session_limit=_option(args, "session_limit"),
            add=runner.add,
            score=runner.score,
            show_progress=runner.show_progress,
        )
        result = run.model_dump()
        result["metrics"] = run.official_metrics()
        return result
