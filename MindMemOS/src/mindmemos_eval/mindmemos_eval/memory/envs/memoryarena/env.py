"""MemoryArena evaluation helpers backed by MindMemOS.

This module keeps the official MemoryArena repository as the task/environment
backend while replacing its memory server with a MindMemOS-backed adapter.  The
adapter preserves the official memory client surface:

- ``add(chunk)`` stores one textual experience chunk;
- ``wrap_user_prompt(question)`` retrieves memories and formats the
  ``<memory_context>`` block expected by MemoryArena agents.

The first runner implemented here targets the Formal Reasoning environments
(``math`` / ``phys``), because their official loop is the smallest reliable
surface for validating the MindMemOS integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindmemos_sdk.memory import AddResult, AsyncMemoryClient, MemorySearchHit, SearchResult, SearchStrategy
from mindmemos_sdk.transport import AsyncHttpTransport
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from mindmemos_eval.memory.scorer import ScoreResult
from .projects import ProjectKey

logger = logging.getLogger("mindmemos_eval.memory.memoryarena")

MEMORYARENA_HF_DATASET = "ZexueHe/memoryarena"
MEMORYARENA_FORMAL_CONFIGS = {
    "math": "formal_reasoning_math",
    "phys": "formal_reasoning_phys",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _format_memory_hit(hit: MemorySearchHit) -> str:
    prefix_parts = []
    if hit.event_time:
        prefix_parts.append(f"event_time: {hit.event_time}")
    if hit.source_timestamp:
        prefix_parts.append(f"source_timestamp: {hit.source_timestamp}")
    prefix = f"[{'; '.join(prefix_parts)}] " if prefix_parts else ""
    return f"{prefix}{hit.memory}"


class MindMemOSArenaMemoryAdapter:
    """MindMemOS implementation of MemoryArena's minimal memory client contract."""

    def __init__(
        self,
        memory: AsyncMemoryClient,
        *,
        user_id: str,
        session_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 5,
        search_strategy: SearchStrategy = "fast",
        rerank: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._memory = memory
        self.user_id = user_id
        self.session_id = session_id or user_id
        self.app_id = app_id
        self.agent_id = agent_id
        self.top_k = top_k
        self.search_strategy: SearchStrategy = search_strategy
        self.rerank = rerank
        self.metadata = metadata or {}

    async def add(self, chunk: str, *, metadata: dict[str, Any] | None = None) -> AddResult:
        """Store one official MemoryArena experience chunk as a text message."""
        merged_metadata = {
            **self.metadata,
            "benchmark": "memoryarena",
            "memoryarena_user_id": self.user_id,
            **(metadata or {}),
        }
        return await self._memory.add(
            [{"role": "user", "content": chunk, "timestamp": _now_ms()}],
            user_id=self.user_id,
            mode="sync",
            app_id=self.app_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            metadata=merged_metadata,
        )

    async def search(self, question: str) -> SearchResult:
        """Search MindMemOS memories for one Arena task/question."""
        return await self._memory.search(
            question,
            user_id=self.user_id,
            top_k=self.top_k,
            search_strategy=self.search_strategy,
            rerank=self.rerank,
            app_id=self.app_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )

    async def wrap_user_prompt(self, question: str) -> str:
        """Return the official ``<memory_context>`` prompt wrapper."""
        search = await self.search(question)
        lines = ["<memory_context>"]
        if search.memories:
            for hit in search.memories:
                lines.append(f"<memory>{_format_memory_hit(hit)}</memory>")
        else:
            lines.append("None")
        lines.append("</memory_context>")
        lines.append(f"User: {question}")
        return "\n".join(lines)


class MemoryArenaFormalTask(BaseModel):
    """One formal-reasoning MemoryArena episode.

    Purpose: Represent one paper/problem group from MemoryArena formal reasoning.
    Used in: MemoryArenaFormalReasoningEnv to run ordered subtasks through an
    agent, environment, and MindMemOS-backed memory adapter.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    paper_name: str = ""
    questions: list[str]
    answers: list[Any]
    backgrounds: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def task_key(self) -> str:
        return self.paper_name or self.id or str(abs(hash(tuple(self.questions))))


class MemoryArenaFormalStepResult(BaseModel):
    """Result for one ordered formal-reasoning subtask."""

    model_config = ConfigDict(extra="ignore")

    query_id: int
    query: str
    background: str = ""
    ground_truth: Any = None
    prompt: str
    output: Any
    memory_entry: str = ""
    observation: dict[str, Any] = Field(default_factory=dict)
    reward: Any = None
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    score: ScoreResult | None = None
    memory_context: str | None = None
    elapsed_seconds: float = 0.0


class MemoryArenaFormalEpisodeResult(BaseModel):
    """Result for one MemoryArena formal-reasoning episode."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    user_id: str
    task_key: str
    env_name: str
    steps: list[MemoryArenaFormalStepResult] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class MemoryArenaFormalRunResult(BaseModel):
    """Whole-run result for MemoryArena formal reasoning."""

    model_config = ConfigDict(extra="ignore")

    env_name: str
    dataset_config: str
    episodes: list[MemoryArenaFormalEpisodeResult] = Field(default_factory=list)
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    averaged_metrics: dict[str, float] = Field(default_factory=dict)
    paper_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    paper_table_metrics: dict[str, Any] = Field(default_factory=dict)

    def format_report(self) -> str:
        lines = ["=" * 72, f"MemoryArena formal reasoning paper metrics ({self.env_name})", "=" * 72]
        if self.paper_metrics:
            for paper, metrics in self.paper_metrics.items():
                lines.append(
                    f"  {paper}: progress_score={metrics['progress_score']:.4f}, "
                    f"paper_pass={float(metrics['is_paper_correct']):.4f}, "
                    f"steps={metrics['num_steps']}"
                )
        lines.append("-" * 60)
        if self.paper_table_metrics:
            display_order = [
                "overall_average_passrate",
                "avg_progress_score",
                "average_memory_length",
                "memory_length",
                "average_session_time",
                "average_task_time",
                "min_k",
                "passrate_at_k",
                "cummulative_passrate_at_k",
                "passrate_at_min_k",
                "cummulative_passrate_at_min_k",
            ]
            for name in display_order:
                if name in self.paper_table_metrics:
                    value = self.paper_table_metrics[name]
                    if isinstance(value, float):
                        lines.append(f"  {name}: {value:.4f}")
                    else:
                        lines.append(f"  {name}: {value}")
        else:
            lines.append("No scored steps.")
        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass(frozen=True)
class MemoryArenaFormalBackend:
    """Official MemoryArena classes/functions required by the formal runner."""

    agent_factory: Callable[..., Any]
    env_client_factory: Callable[..., Any]


class MemoryArenaFormalReasoningEnv:
    """Run MemoryArena formal reasoning with MindMemOS as the memory backend."""

    def __init__(
        self,
        memory: AsyncMemoryClient | None = None,
        *,
        backend: MemoryArenaFormalBackend,
        env_name: str = "math",
        env_server_url: str = "http://0.0.0.0:8001",
        env_timeout: int = 300,
        env_config: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
        top_k: int = 5,
        search_strategy: SearchStrategy = "fast",
        rerank: bool = False,
        judge_result_in_memory: bool = False,
        app_id: str | None = None,
        agent_id: str | None = None,
        user_id_prefix: str = "memoryarena",
        episode_keys: Sequence[ProjectKey] | None = None,
        concurrency: int = 1,
        base_url: str = "http://localhost:8000",
        request_timeout: float = 1800.0,
    ) -> None:
        normalized_env = env_name.strip().lower()
        if normalized_env not in MEMORYARENA_FORMAL_CONFIGS:
            raise ValueError(f"Unsupported formal reasoning env_name: {env_name!r}. Use 'math' or 'phys'.")
        if memory is None and not episode_keys:
            raise ValueError("Provide either a shared `memory` client or per-episode `episode_keys`.")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self._memory = memory
        self._backend = backend
        self.env_name = normalized_env
        self.env_server_url = env_server_url
        self.env_timeout = env_timeout
        self.env_config = env_config or {"max_steps": 10}
        self.agent_config = agent_config or {}
        self.top_k = top_k
        self.search_strategy: SearchStrategy = search_strategy
        self.rerank = rerank
        self.judge_result_in_memory = judge_result_in_memory
        self.app_id = app_id
        self.agent_id = agent_id
        self.user_id_prefix = user_id_prefix
        self.episode_keys = list(episode_keys) if episode_keys else None
        self.concurrency = concurrency
        self.base_url = base_url
        self.request_timeout = request_timeout

    def episode_user_id(self, task: MemoryArenaFormalTask, episode_idx: int, run_id: str) -> str:
        safe_key = task.task_key.replace("/", "_").replace(" ", "_")[:120]
        return f"{self.user_id_prefix}::{self.env_name}::{episode_idx}::{safe_key}::{run_id}"

    def _build_episode_client(
        self, episode_idx: int
    ) -> tuple[AsyncMemoryClient, AsyncHttpTransport | None, str | None]:
        """Resolve the memory client for one episode.

        With per-episode ``episode_keys`` a fresh transport/client is built from
        that episode's api key (giving it an isolated ``project_id``); the
        transport must be closed by the caller. Otherwise the shared client is
        reused and no transport ownership is transferred.
        """
        if self.episode_keys is None:
            return self._memory, None, None
        key = self.episode_keys[episode_idx]
        transport = AsyncHttpTransport(
            base_url=self.base_url,
            api_key=key.api_key,
            timeout_seconds=self.request_timeout,
        )
        return AsyncMemoryClient(transport), transport, key.project_id

    async def run_episode(
        self,
        task: MemoryArenaFormalTask,
        *,
        episode_idx: int,
        run_id: str,
        add_initial_observation: bool = True,
    ) -> MemoryArenaFormalEpisodeResult:
        """Run one formal reasoning task group through the Arena loop."""
        task_id = str(uuid.uuid4())
        user_id = self.episode_user_id(task, episode_idx, run_id)
        client, transport, project_id = self._build_episode_client(episode_idx)
        adapter_metadata: dict[str, Any] = {
            "memoryarena_env": self.env_name,
            "memoryarena_task_key": task.task_key,
        }
        if project_id:
            adapter_metadata["memoryarena_project_id"] = project_id
        memory = MindMemOSArenaMemoryAdapter(
            client,
            user_id=user_id,
            session_id=user_id,
            app_id=self.app_id,
            agent_id=self.agent_id,
            top_k=self.top_k,
            search_strategy=self.search_strategy,
            rerank=self.rerank,
            metadata=adapter_metadata,
        )
        agent = self._backend.agent_factory(**self.agent_config)
        env_client = self._backend.env_client_factory(
            task_id=task_id,
            env_name=self.env_name,
            base_url=self.env_server_url,
            timeout=self.env_timeout,
            env_config=self.env_config,
        )
        start = time.time()
        steps: list[MemoryArenaFormalStepResult] = []
        # Short, log-friendly episode tag so interleaved concurrent episodes stay distinguishable.
        ep_tag = f"ep{episode_idx}[{task.task_key[:40]}]"
        total_steps = len(task.questions)
        logger.info("%s episode start: %d subtasks (env=%s)", ep_tag, total_steps, self.env_name)
        try:
            obs = env_client.reset()
            if add_initial_observation and obs is not None:
                await memory.add("Initial result: Empty\n", metadata={"memoryarena_stage": "initial"})

            for query_id, question in enumerate(task.questions):
                step_start = time.time()
                ground_truth = task.answers[query_id] if query_id < len(task.answers) else None
                background = task.backgrounds[query_id] if query_id < len(task.backgrounds) else ""
                query = agent.build_prompt(task=question, background=background)

                t0 = time.time()
                prompt = await memory.wrap_user_prompt(query)
                t_search = time.time() - t0

                t0 = time.time()
                action = agent.act(prompt)
                t_act = time.time() - t0

                t0 = time.time()
                result = env_client.step(action, ground_truth=ground_truth, need_judge=True)
                t_env = time.time() - t0

                observation = result.get("observation") or {}
                reward = result.get("reward")
                info = result.get("info") or {}
                if observation and "memory_context" in observation:
                    memory_context = _as_text(observation.get("memory_context"))
                else:
                    memory_context = prompt.split("</memory_context>", 1)[0] + "</memory_context>"

                # Mirror official run_math.py's judge_result_in_memory branch: when enabled and the
                # env produced a judge verdict, persist the reward so build_memory_entry emits the
                # "## Judge: CORRECT/INCORRECT" line. (The official code reads result.get("judge_result"),
                # which is always None because env_client.step only returns it nested under observation;
                # we read it from observation so the flag actually takes effect.)
                judge_result = observation.get("judge_result")
                memory_reward = reward if (self.judge_result_in_memory and judge_result is not None) else None
                memory_entry = agent.build_memory_entry(
                    task=question,
                    observation=observation,
                    action=action,
                    reward=memory_reward,
                )
                t0 = time.time()
                await memory.add(
                    memory_entry,
                    metadata={"memoryarena_stage": "step", "memoryarena_query_id": query_id},
                )
                t_add = time.time() - t0

                score = None
                if isinstance(reward, (int, float, bool)):
                    score_value = float(reward)
                    score = ScoreResult(score=score_value, passed=score_value >= 1.0, reason="env_reward")
                step_elapsed = time.time() - step_start
                logger.info(
                    "%s step %d/%d done in %.1fs (search=%.1fs act=%.1fs env=%.1fs add=%.1fs) reward=%s",
                    ep_tag,
                    query_id + 1,
                    total_steps,
                    step_elapsed,
                    t_search,
                    t_act,
                    t_env,
                    t_add,
                    reward,
                )
                steps.append(
                    MemoryArenaFormalStepResult(
                        query_id=query_id,
                        query=question,
                        background=background,
                        ground_truth=ground_truth,
                        prompt=prompt,
                        output=action,
                        memory_entry=memory_entry,
                        observation=observation,
                        reward=reward,
                        done=bool(result.get("done", False)),
                        info=info,
                        score=score,
                        memory_context=memory_context,
                        elapsed_seconds=step_elapsed,
                    )
                )
        finally:
            env_client.close()
            if transport is not None:
                await transport.aclose()

        episode_elapsed = time.time() - start
        logger.info(
            "%s episode done: %d/%d subtasks in %.1fs (%.1fs/step avg)",
            ep_tag,
            len(steps),
            total_steps,
            episode_elapsed,
            episode_elapsed / len(steps) if steps else 0.0,
        )

        return MemoryArenaFormalEpisodeResult(
            task_id=task_id,
            user_id=user_id,
            task_key=task.task_key,
            env_name=self.env_name,
            steps=steps,
            elapsed_seconds=episode_elapsed,
        )

    async def run_dataset(
        self,
        data: Sequence[MemoryArenaFormalTask],
        *,
        limit: int = 0,
        print_report: bool = True,
        show_progress: bool = True,
        run_id: str | None = None,
    ) -> MemoryArenaFormalRunResult:
        """Run formal-reasoning episodes with bounded concurrency.

        Each episode gets its own ``project_id`` (via ``episode_keys``) so their
        memories never cross-pollute; ``self.concurrency`` caps how many run at
        once to avoid overloading the env server / MindMemOS backend. Results are
        returned in the original episode order regardless of completion order.
        """
        selected = list(data[:limit] if limit > 0 else data)
        actual_run_id = run_id or uuid.uuid4().hex[:8]
        if self.episode_keys is not None and len(self.episode_keys) < len(selected):
            raise ValueError(
                f"episode_keys has {len(self.episode_keys)} entries but {len(selected)} episodes are scheduled."
            )

        semaphore = asyncio.Semaphore(self.concurrency)
        progress = (
            tqdm(total=len(selected), desc=f"MemoryArena {self.env_name}", unit="episode") if show_progress else None
        )

        async def _run_one(idx: int, task: MemoryArenaFormalTask) -> MemoryArenaFormalEpisodeResult:
            async with semaphore:
                try:
                    return await self.run_episode(task, episode_idx=idx, run_id=actual_run_id)
                finally:
                    if progress is not None:
                        progress.update(1)

        try:
            episodes = await asyncio.gather(*(_run_one(idx, task) for idx, task in enumerate(selected)))
        finally:
            if progress is not None:
                progress.close()
        episodes = list(episodes)

        metric_lists: dict[str, list[float]] = {}
        for episode in episodes:
            rewards = [float(step.reward) for step in episode.steps if isinstance(step.reward, (int, float, bool))]
            if rewards:
                metric_lists.setdefault("episode_avg_reward", []).append(sum(rewards) / len(rewards))
                metric_lists.setdefault("episode_success_rate", []).append(
                    sum(1.0 for value in rewards if value >= 1.0) / len(rewards)
                )
            for step in episode.steps:
                if isinstance(step.reward, (int, float, bool)):
                    metric_lists.setdefault("step_reward", []).append(float(step.reward))
                metric_lists.setdefault("step_elapsed_seconds", []).append(float(step.elapsed_seconds))
            metric_lists.setdefault("episode_elapsed_seconds", []).append(float(episode.elapsed_seconds))
        averaged = {name: sum(values) / len(values) for name, values in metric_lists.items() if values}
        paper_metrics, paper_table_metrics = calculate_memoryarena_formal_paper_metrics(episodes)
        run = MemoryArenaFormalRunResult(
            env_name=self.env_name,
            dataset_config=MEMORYARENA_FORMAL_CONFIGS[self.env_name],
            episodes=episodes,
            metrics=metric_lists,
            averaged_metrics=averaged,
            paper_metrics=paper_metrics,
            paper_table_metrics=paper_table_metrics,
        )
        if print_report:
            print(run.format_report(), flush=True)
        return run

    @staticmethod
    def load_dataset(
        path: str | Path | None = None,
        *,
        hf_dataset: str = MEMORYARENA_HF_DATASET,
        env_name: str = "math",
        split: str = "test",
        limit: int = 0,
    ) -> list[MemoryArenaFormalTask]:
        """Load formal reasoning tasks from JSON/JSONL or HuggingFace."""
        if path:
            raw_items = _read_memoryarena_local_dataset(Path(path))
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install `datasets` to load MemoryArena from HuggingFace.") from exc
            normalized_env = env_name.strip().lower()
            if normalized_env not in MEMORYARENA_FORMAL_CONFIGS:
                raise ValueError(f"Unsupported formal reasoning env_name: {env_name!r}. Use 'math' or 'phys'.")
            ds = load_dataset(hf_dataset, MEMORYARENA_FORMAL_CONFIGS[normalized_env], split=split)
            raw_items = list(ds)
        tasks = [_coerce_formal_task(item) for item in raw_items]
        return tasks[:limit] if limit > 0 else tasks


def _coerce_formal_task(raw: dict[str, Any]) -> MemoryArenaFormalTask:
    questions = raw.get("questions") or raw.get("question") or []
    answers = raw.get("answers") or raw.get("answer") or []
    backgrounds = raw.get("backgrounds") or raw.get("background") or []
    if not isinstance(questions, list):
        questions = [questions]
    if not isinstance(answers, list):
        answers = [answers]
    if not isinstance(backgrounds, list):
        backgrounds = [backgrounds]
    return MemoryArenaFormalTask(
        id=str(raw.get("id") or ""),
        paper_name=str(raw.get("paper_name") or raw.get("name") or ""),
        questions=[_as_text(question) for question in questions],
        answers=answers,
        backgrounds=[_as_text(background) for background in backgrounds],
        raw={k: v for k, v in raw.items() if k not in {"questions", "answers", "backgrounds"}},
    )


def _read_memoryarena_local_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("items", payload.get("examples")))
        if isinstance(data, list):
            return data
        if "questions" in payload:
            return [payload]
    raise ValueError(f"Unsupported MemoryArena dataset file format: {path}")


def _reward_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 1.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes", "correct"}
    return False


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _token_count(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(encoding.encode(text or "", disallowed_special=()))
    except Exception:  # pragma: no cover - token stats should not block metrics
        return len((text or "").split())


def calculate_memoryarena_formal_paper_metrics(
    episodes: Sequence[MemoryArenaFormalEpisodeResult],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Calculate official-style formal-reasoning paper/table metrics.

    The shape mirrors MemoryArena's ``formal_reasoning_env/eval.py``:
    progress score is the per-paper fraction of correct subtasks, while paper
    passrate is determined by the final subtask correctness.
    """
    paper_metrics: dict[str, dict[str, Any]] = {}
    passrate_at_k_counts = [0] * 20
    paper_count_at_k = [0] * 20
    min_length = min((len(episode.steps) for episode in episodes if episode.steps), default=0)

    progress_scores: list[float] = []
    paper_passes: list[float] = []
    avg_memory_lengths: list[float] = []
    max_memory_lengths: list[float] = []
    avg_step_times: list[float] = []
    episode_times: list[float] = []

    for episode in episodes:
        if not episode.steps:
            continue
        correctness = [_reward_to_bool(step.reward) for step in episode.steps]
        correct_count = sum(1 for passed in correctness if passed)
        memory_lengths = [_token_count(step.memory_context or "") for step in episode.steps]
        step_times = [float(step.elapsed_seconds) for step in episode.steps]

        for idx, passed in enumerate(correctness[:20]):
            if passed:
                passrate_at_k_counts[idx] += 1
            paper_count_at_k[idx] += 1

        progress_score = correct_count / len(episode.steps)
        is_paper_correct = correctness[-1]
        avg_memory_length = _mean([float(value) for value in memory_lengths])
        max_memory_length = float(max(memory_lengths)) if memory_lengths else 0.0
        avg_step_time = _mean(step_times)

        progress_scores.append(progress_score)
        paper_passes.append(float(is_paper_correct))
        avg_memory_lengths.append(avg_memory_length)
        max_memory_lengths.append(max_memory_length)
        avg_step_times.append(avg_step_time)
        episode_times.append(float(episode.elapsed_seconds))

        paper_metrics[episode.task_key] = {
            "progress_score": progress_score,
            "is_paper_correct": is_paper_correct,
            "num_steps": len(episode.steps),
            "correct_count": correct_count,
            "avg_length": avg_memory_length,
            "max_length": max_memory_length,
            "avg_time": avg_step_time,
            "data_time": float(episode.elapsed_seconds),
        }

    passrate_at_k = [
        passrate_at_k_counts[idx] / paper_count_at_k[idx]
        for idx in range(len(passrate_at_k_counts))
        if paper_count_at_k[idx] > 0
    ]
    cummulative_passrate_at_k = []
    for idx in range(len(passrate_at_k_counts)):
        if paper_count_at_k[idx] > 0:
            count_so_far = sum(paper_count_at_k[: idx + 1])
            cummulative_passrate_at_k.append(sum(passrate_at_k_counts[: idx + 1]) / count_so_far)

    table_metrics = {
        "overall_average_passrate": _mean(paper_passes),
        "avg_progress_score": _mean(progress_scores),
        "average_session_time": _mean(avg_step_times),
        "average_memory_length": _mean(avg_memory_lengths),
        "average_task_time": _mean(episode_times),
        "memory_length": max(max_memory_lengths) if max_memory_lengths else 0.0,
        "min_k": min_length,
        "passrate_at_k": [float(value) for value in passrate_at_k],
        "cummulative_passrate_at_k": [float(value) for value in cummulative_passrate_at_k],
        "passrate_at_min_k": [float(value) for value in passrate_at_k[:min_length]],
        "cummulative_passrate_at_min_k": [float(value) for value in cummulative_passrate_at_k[:min_length]],
    }
    return paper_metrics, table_metrics


def save_memoryarena_formal_results(path: str | Path, run: MemoryArenaFormalRunResult) -> None:
    """Save formal reasoning results as JSON for later official/post-hoc eval."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run.model_dump(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
