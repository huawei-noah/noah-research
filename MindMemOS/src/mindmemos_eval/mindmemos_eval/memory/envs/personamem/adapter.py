"""PersonaMem benchmark matrix adapter."""

from __future__ import annotations

import argparse
from typing import Any

from mindmemos_eval.llm import LLMClient
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext
from mindmemos_eval.memory.config import _merged_runner_config, _option, resolve_public_search_strategy
from .env import PersonaMemContextStore, PersonaMemEnv


class PersonaMemAdapter:
    """Official PersonaMem v1 adapter backed by :class:`PersonaMemEnv`."""

    name = "personamem"

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
        """Run PersonaMem with official context boundaries and deterministic scoring."""
        # ✅ 提前验证配置，避免浪费初始化资源
        context_dataset = str(bench_config.raw.get("context_dataset") or "")
        if not context_dataset:
            raise ValueError("personamem benchmark is missing context_dataset in raw config")
        evaluation_mode = str(bench_config.raw.get("evaluation_mode") or "memory_rag")
        if evaluation_mode not in {"memory_rag", "official_full_context"}:
            raise ValueError(
                f"personamem evaluation_mode must be 'memory_rag' or 'official_full_context', "
                f"got: {evaluation_mode}"
            )

        runner = getattr(args, "runner_config", None)
        if runner is None:
            runner = _merged_runner_config(args)

        items = PersonaMemEnv.load_items(bench_config.dataset)
        limit = bench_config.limit if bench_config.limit is not None else _option(args, "limit")
        if limit is not None:
            items = items[: int(limit)]

        search_params = bench_config.search_params
        public_search_strategy = resolve_public_search_strategy(
            search_params.get("public_search_strategy")
            or ("agentic" if search_params.get("agentic") is True else None)
            or runner.search_strategy
            or "fast"
        )
        top_k = search_params["top_k"] if "top_k" in search_params else runner.top_k
        rerank = search_params["rerank"] if "rerank" in search_params else runner.rerank
        add_batch_size = int(bench_config.execution_params.get("add_batch_size", 50))
        env = PersonaMemEnv(
            memory,
            answer_llm=answer_llm,
            context_store=PersonaMemContextStore(context_dataset),
            evaluation_mode=evaluation_mode,
            context_size=str(bench_config.raw.get("context_size") or "32k"),
            top_k=50 if top_k is None else int(top_k),
            search_strategy=public_search_strategy,
            rerank=bool(rerank),
            add_batch_size=add_batch_size,
        )
        run = await env.run_dataset(
            items,
            max_build_concurrency=runner.max_conv_concurrency,
            max_qa_concurrency=runner.max_qa_concurrency,
            add=runner.add,
            score=runner.score,
            show_progress=runner.show_progress,
        )
        print(run.format_report())
        return run.model_dump()
