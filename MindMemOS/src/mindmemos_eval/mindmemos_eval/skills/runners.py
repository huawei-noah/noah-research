"""Skill benchmark CLI dispatcher and runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from tqdm.auto import tqdm

from ..llm import LLMClient, LLMConfig
from .agents import ReactAgentFactory, RunResult
from .args import add_common_skill_args, add_spreadsheetbench_args
from .evolve import FastAPISkillEvolutionClient, NoopSkillEvolutionClient, SkillEvolutionClient

SkillArgRegistrar = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class SkillBenchmark:
    """CLI binding for one skill benchmark dataset."""

    help: str
    add_args: SkillArgRegistrar
    env_builder: str


@dataclass(frozen=True)
class SkillRunConfig:
    """Generic skill benchmark execution settings."""

    split: str = "all"
    limit: int | None = None
    seed: int | None = None
    evolve: bool = False
    evolve_every: int = 1
    stop_on_evolution_error: bool = True
    show_progress: bool = False
    concurrency: int = 1


SKILL_BENCHMARKS = {
    "spreadsheetbench": SkillBenchmark(
        help="Run SpreadsheetBench skill self-evolution evaluation.",
        add_args=add_spreadsheetbench_args,
        env_builder="mindmemos_eval.skills.envs.spreadsheetbench.env:build_env",
    ),
}


def _load_attr(path: str) -> Any:
    module_name, attr_name = path.split(":", maxsplit=1)
    return getattr(import_module(module_name), attr_name)


def add_skill_args(parser: argparse.ArgumentParser) -> None:
    """Register shared skill benchmark arguments."""
    common_args = argparse.ArgumentParser(add_help=False)
    add_common_skill_args(common_args)
    subparsers = parser.add_subparsers(
        dest="benchmark",
        metavar="BENCHMARK",
        required=True,
        help="Skill benchmark dataset to run.",
    )
    for name, benchmark in sorted(SKILL_BENCHMARKS.items()):
        subparser = subparsers.add_parser(
            name,
            help=benchmark.help,
            description=benchmark.help,
            parents=[common_args],
        )
        benchmark.add_args(subparser)
        subparser.set_defaults(env_builder=benchmark.env_builder)


def run_skill_benchmark(args: argparse.Namespace) -> int:
    """Run the selected skill benchmark environment."""
    env = _load_attr(args.env_builder)(args)
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    agent_factory = ReactAgentFactory(
        LLMClient(llm_config),
        max_turns=args.max_turns,
        skill_sources=args.skill or [],
        python_path=args.python_path,
    )
    evolver = _build_evolver(args, env.name)
    config = SkillRunConfig(
        limit=args.limit,
        seed=args.seed,
        evolve=args.evolve,
        evolve_every=args.evolve_every,
        stop_on_evolution_error=not args.continue_on_evolution_error,
        show_progress=args.progress,
        concurrency=args.concurrency or getattr(env, "default_concurrency", 1),
    )
    result = asyncio.run(_run_and_close(env, agent_factory, evolver, config))
    _print_result(args.benchmark, env, result, seed=args.seed)
    return 0


async def _run_and_close(
    env: Any,
    agent_factory: ReactAgentFactory,
    evolver: SkillEvolutionClient,
    config: SkillRunConfig,
) -> Any:
    try:
        return await SkillEvalRunner(env, agent_factory, evolver, config).run()
    finally:
        await evolver.aclose()


def _build_evolver(args: argparse.Namespace, benchmark_name: str) -> SkillEvolutionClient:
    if args.evolve and args.evolution_base_url:
        return FastAPISkillEvolutionClient(
            args.evolution_base_url,
            api_key=args.evolution_api_key,
            transcript_metadata={"benchmark": benchmark_name},
        )
    if args.evolve:
        print(f"{benchmark_name}: --evolve set without --evolution-base-url; running with no-op evolution.")
    return NoopSkillEvolutionClient()


class SkillEvalRunner:
    """Run one benchmark env with one agent factory and one evolution algorithm."""

    def __init__(
        self,
        env: Any,
        agent_factory: ReactAgentFactory,
        evolver: SkillEvolutionClient,
        config: SkillRunConfig,
    ) -> None:
        self.env = env
        self.agent_factory = agent_factory
        self.evolver = evolver
        self.config = config

    async def run(self) -> Any:
        cases = self.env.load_cases(self.config.split)
        if self.config.seed is not None:
            random.Random(self.config.seed).shuffle(cases)
        if self.config.limit is not None:
            cases = cases[: self.config.limit]
        if self.config.evolve:
            results = await self._run_with_evolution(cases)
        elif self.config.concurrency > 1:
            results = await self._run_concurrent(cases)
        else:
            results = await self._run_online(cases)
        return self.env.summarize(self.config.split, results)

    async def _run_case(self, case: Any, *, rollout: int = 0, log: bool = True) -> Any:
        workdir = self.env.case_workdir(case, rollout)
        workdir.mkdir(parents=True, exist_ok=True)
        self.env.setup_case(case, workdir)

        agent, _tools = self.agent_factory.build(workdir, self.env.system_prompt())
        messages = self.env.build_messages(case)
        started_at = time.time()
        error: str | None = None
        score = 0.0
        score_message = ""
        try:
            agent_result = await agent.run(messages)
        except Exception as exc:  # noqa: BLE001 - keep one failed case from stopping the run
            agent_result = RunResult(messages=messages, turns=0, finished=False)
            error = f"{type(exc).__name__}: {exc}"

        if error is None:
            try:
                score, score_message = self.env.score(case, workdir)
            except Exception as exc:  # noqa: BLE001 - scoring failures are case failures
                error = f"score error: {type(exc).__name__}: {exc}"
                score = 0.0

        result = self.env.build_result(
            case,
            agent_result,
            workdir=workdir,
            score=score,
            score_message=score_message,
            error=error,
            started_at=started_at,
            ended_at=time.time(),
            rollout=rollout,
        )
        if log:
            self.env.append_trajectory(result)
        return result

    async def _run_online(self, cases: list[Any]) -> list[Any]:
        results: list[Any] = []
        progress = tqdm(cases, desc=self.env.name, unit="task", disable=not self.config.show_progress)
        correct = 0
        for case in progress:
            result = await self._run_case(case)
            results.append(result)
            if result.score >= 1.0:
                correct += 1
            progress.set_postfix(correct=correct, score=f"{correct / len(results):.4f}", last=result.case_id)
        return results

    async def _run_concurrent(self, cases: list[Any]) -> list[Any]:
        results: list[Any | None] = [None] * len(cases)
        progress = tqdm(total=len(cases), desc=self.env.name, unit="task", disable=not self.config.show_progress)
        correct = 0
        completed = 0

        def update_progress(result: Any) -> None:
            nonlocal completed, correct
            completed += 1
            if result.score >= 1.0:
                correct += 1
            progress.update(1)
            progress.set_postfix(correct=correct, score=f"{correct / completed:.4f}", last=result.case_id)

        try:
            ordered = await self._run_cases_concurrently(
                cases,
                concurrency=self.config.concurrency,
                on_result=update_progress,
            )
            for index, result in enumerate(ordered):
                results[index] = result
                self.env.append_trajectory(result)
        finally:
            progress.close()
        return [result for result in results if result is not None]

    async def _run_with_evolution(self, cases: list[Any]) -> list[Any]:
        live_skills = self.agent_factory.stage_live_skills(self.env.run_dir)
        await self.evolver.prepare(live_skills)

        batch_size = max(1, self.config.evolve_every)
        total_batches = (len(cases) + batch_size - 1) // batch_size
        results: list[Any] = []
        progress = tqdm(total=len(cases), desc=self.env.name, unit="task", disable=not self.config.show_progress)
        correct = 0
        completed_total = 0
        try:
            for batch_index, start in enumerate(range(0, len(cases), batch_size)):
                batch = cases[start : start + batch_size]
                batch_done = 0
                batch_label = f"{batch_index + 1}/{total_batches}"

                def update_progress(result: Any) -> None:
                    nonlocal batch_done, completed_total, correct
                    batch_done += 1
                    completed_total += 1
                    if result.score >= 1.0:
                        correct += 1
                    progress.update(1)
                    progress.set_postfix(
                        batch=batch_label,
                        batch_done=f"{batch_done}/{len(batch)}",
                        correct=correct,
                        score=f"{correct / completed_total:.4f}",
                        last=result.case_id,
                    )

                progress.set_postfix(
                    batch=batch_label,
                    batch_done=f"0/{len(batch)}",
                    correct=correct,
                    score=f"{correct / completed_total:.4f}" if completed_total else "0.0000",
                )
                progress.refresh()
                batch_results = await self._run_cases_concurrently(batch, concurrency=len(batch), on_result=update_progress)
                for result in batch_results:
                    results.append(result)
                    self.env.append_trajectory(result)
                await self._evolve_batch(batch, batch_index, start, batch_results)
        finally:
            progress.close()
        return results

    async def _evolve_batch(
        self,
        batch: list[Any],
        batch_index: int,
        batch_start: int,
        batch_results: list[Any],
    ) -> None:
        evolution_started_at = time.time()
        try:
            await asyncio.gather(*(self.evolver.record_case(result) for result in batch_results))
            outcomes = await self.evolver.evolve()
            self.env.append_evolution_event(
                split=batch[0].split if batch else self.config.split,
                batch_index=batch_index,
                batch_start=batch_start,
                batch_end=batch_start + len(batch),
                batch_results=batch_results,
                started_at=evolution_started_at,
                ended_at=time.time(),
                outcomes=outcomes,
            )
            self._log_evolution(outcomes)
        except Exception as exc:
            self.env.append_evolution_event(
                split=batch[0].split if batch else self.config.split,
                batch_index=batch_index,
                batch_start=batch_start,
                batch_end=batch_start + len(batch),
                batch_results=batch_results,
                started_at=evolution_started_at,
                ended_at=time.time(),
                outcomes=[],
                error=exc,
            )
            if self.config.stop_on_evolution_error:
                raise

    async def _run_cases_concurrently(
        self,
        cases: list[Any],
        *,
        concurrency: int,
        on_result: Callable[[Any], None] | None = None,
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(max(1, concurrency))
        results: list[Any | None] = [None] * len(cases)

        async def run_indexed(index: int, case: Any) -> tuple[int, Any]:
            async with semaphore:
                return index, await self._run_case(case, log=False)

        tasks = [asyncio.create_task(run_indexed(index, case)) for index, case in enumerate(cases)]
        for completed in asyncio.as_completed(tasks):
            index, result = await completed
            results[index] = result
            if on_result is not None:
                on_result(result)
        return [result for result in results if result is not None]

    def _log_evolution(self, outcomes: list[Any]) -> None:
        for outcome in outcomes:
            if outcome.evolved:
                message = (
                    f"[evolve] {outcome.skill_name}: minted {len(outcome.new_version_ids or [])} "
                    f"version(s) -> {outcome.new_version_id} "
                    f"(consumed {outcome.consumed_count}/{outcome.pending_count} pending)"
                )
            elif outcome.pending_count >= outcome.threshold:
                message = (
                    f"[evolve] {outcome.skill_name}: mint failed "
                    f"({outcome.pending_count} pending >= {outcome.threshold}, "
                    f"summarized {outcome.summarized_count}, 0 versions minted)"
                )
            else:
                message = f"[evolve] {outcome.skill_name}: below threshold ({outcome.pending_count}/{outcome.threshold})"
            if self.config.show_progress:
                tqdm.write(message)
            else:
                print(message)


def _print_result(benchmark: str, env: Any, result: Any, *, seed: int | None) -> None:
    print(
        f"{benchmark} result: "
        f"total={result.total} correct={result.correct} score={result.accuracy:.4f} "
        f"seed={seed}"
    )
    print(
        json.dumps(
            {
                "env": benchmark,
                "data_root": str(getattr(env, "data_root", "")),
                "run_dir": result.run_dir,
                "summary": f"{result.split}_summary.json",
                "trajectories": str(env.trajectory_path(result.split)),
                "evolution_events": str(env.evolution_events_path(result.split)),
                "score": result.accuracy,
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone skill benchmark parser."""
    parser = argparse.ArgumentParser(description="Run MindMemOS skill benchmark evaluation.")
    add_skill_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entry point."""
    return run_skill_benchmark(build_parser().parse_args(argv))
