"""Tests for MemoryArena MindMemOS integration helpers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from mindmemos_sdk.memory import AsyncMemoryClient
from mindmemos_sdk.transport import AsyncHttpTransport

from mindmemos_eval import (
    MemoryArenaFormalBackend,
    MemoryArenaFormalEpisodeResult,
    MemoryArenaFormalReasoningEnv,
    MemoryArenaFormalStepResult,
    MemoryArenaFormalTask,
    MindMemOSArenaMemoryAdapter,
    calculate_memoryarena_formal_paper_metrics,
    generate_project_keys,
)


def _memory(memories: list[dict[str, Any]]):
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append({"path": request.url.path, "body": body})
        data = {"memories": memories} if request.url.path.endswith("/search") else {"memories": []}
        return httpx.Response(200, json={"code": "ok", "message": "", "request_id": "r", "data": data})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(base_url="https://api.test", api_key="dev", client=client)
    return AsyncMemoryClient(transport), captured


@pytest.mark.asyncio
async def test_arena_memory_adapter_adds_text_chunk_and_wraps_prompt():
    memory, captured = _memory([{"id": "m1", "memory": "Earlier proof used induction."}])
    adapter = MindMemOSArenaMemoryAdapter(memory, user_id="arena-user", top_k=3, search_strategy="fast")

    await adapter.add("## Task: prove lemma")
    prompt = await adapter.wrap_user_prompt("Solve the next lemma")

    add_body = captured[0]["body"]
    assert captured[0]["path"] == "/v1/memory/add"
    assert add_body["user_id"] == "arena-user"
    assert add_body["session_id"] == "arena-user"
    assert add_body["mode"] == "sync"
    assert add_body["messages"][0]["role"] == "user"
    assert add_body["messages"][0]["content"] == "## Task: prove lemma"
    assert add_body["metadata"]["benchmark"] == "memoryarena"

    search_body = captured[1]["body"]
    assert captured[1]["path"] == "/v1/memory/search"
    assert search_body["query"] == "Solve the next lemma"
    assert search_body["top_k"] == 3
    assert "<memory_context>" in prompt
    assert "<memory>Earlier proof used induction.</memory>" in prompt
    assert prompt.endswith("User: Solve the next lemma")


class _FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def build_prompt(self, *, task: str, background: str = "") -> str:
        return f"BG: {background}\nTASK: {task}"

    def act(self, prompt: str) -> dict[str, Any]:
        return {"type": "final", "answer": "42", "prompt": prompt}

    def build_memory_entry(
        self,
        *,
        task: str,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: Any = None,
    ) -> str:
        entry = f"## Task: {task}\n## solution: {action['answer']}\n## Obs: {observation.get('final')}"
        if reward is not None:
            entry += f"\n## Judge: {'CORRECT' if reward else 'INCORRECT'}"
        return entry


class _FakeEnvClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def reset(self):
        return {"initial": True}

    def step(self, action: Any, ground_truth: Any = None, need_judge: bool = False):
        return {
            "observation": {"final": action["answer"], "judge_result": "correct"},
            "reward": 1.0,
            "done": False,
            "info": {"ground_truth": ground_truth, "need_judge": need_judge},
        }

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_formal_runner_uses_mindmemos_memory_contract():
    memory, captured = _memory([{"id": "m1", "memory": "Prior answer was 41."}])
    backend = MemoryArenaFormalBackend(agent_factory=_FakeAgent, env_client_factory=_FakeEnvClient)
    env = MemoryArenaFormalReasoningEnv(
        memory,
        backend=backend,
        env_name="math",
        env_server_url="https://env.test",
        agent_config={"model_name": "fake"},
        top_k=2,
    )
    task = MemoryArenaFormalTask(
        id="p1",
        paper_name="paper-a",
        questions=["What is x?"],
        answers=["42"],
        backgrounds=["Numbers"],
    )

    run = await env.run_dataset([task], show_progress=False, print_report=False, run_id="run")

    assert run.paper_table_metrics["avg_progress_score"] == 1.0
    assert run.paper_table_metrics["overall_average_passrate"] == 1.0
    assert run.paper_table_metrics["passrate_at_k"] == [1.0]
    step = run.episodes[0].steps[0]
    assert step.score is not None and step.score.passed is True
    assert "Prior answer was 41." in step.prompt
    assert step.memory_entry.startswith("## Task: What is x?")
    paths = [call["path"] for call in captured]
    assert paths == ["/v1/memory/add", "/v1/memory/search", "/v1/memory/add"]
    assert captured[1]["body"]["top_k"] == 2
    assert captured[2]["body"]["metadata"]["memoryarena_query_id"] == 0


@pytest.mark.asyncio
async def test_judge_result_in_memory_controls_reward_in_memory_entry():
    backend = MemoryArenaFormalBackend(agent_factory=_FakeAgent, env_client_factory=_FakeEnvClient)
    task = MemoryArenaFormalTask(
        id="p1", paper_name="paper-a", questions=["What is x?"], answers=["42"], backgrounds=["Numbers"]
    )

    # Default: judge verdict is NOT persisted -> reward=None -> no "## Judge" line.
    memory_off, _ = _memory([])
    env_off = MemoryArenaFormalReasoningEnv(
        memory_off, backend=backend, env_name="math", env_server_url="https://env.test", agent_config={}
    )
    run_off = await env_off.run_dataset([task], show_progress=False, print_report=False, run_id="run")
    assert "## Judge:" not in run_off.episodes[0].steps[0].memory_entry

    # Enabled: env judge_result is present -> reward flows into the memory entry.
    memory_on, _ = _memory([])
    env_on = MemoryArenaFormalReasoningEnv(
        memory_on,
        backend=backend,
        env_name="math",
        env_server_url="https://env.test",
        agent_config={},
        judge_result_in_memory=True,
    )
    run_on = await env_on.run_dataset([task], show_progress=False, print_report=False, run_id="run")
    assert "## Judge: CORRECT" in run_on.episodes[0].steps[0].memory_entry


@dataclass
class _Envelope:
    code: str
    request_id: str
    data: dict[str, Any]


class _RecordingTransport:
    """Fake AsyncHttpTransport recording the api key and tracking concurrency."""

    api_keys: list[str] = []
    headers: list[dict[str, str]] = []
    bodies: list[dict[str, Any]] = []
    active = 0
    max_active = 0

    @classmethod
    def reset(cls) -> None:
        cls.api_keys = []
        cls.headers = []
        cls.bodies = []
        cls.active = 0
        cls.max_active = 0

    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 30.0) -> None:
        self.api_key = api_key
        type(self).api_keys.append(api_key)

    async def post_envelope(
        self,
        path: str,
        *,
        json: dict[str, Any],
        request_id: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> _Envelope:
        type(self).headers.append(headers or {})
        type(self).bodies.append(json)
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            await asyncio.sleep(0.02)
        finally:
            type(self).active -= 1
        return _Envelope(code="ok", request_id="r", data={"memories": []})

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_per_episode_keys_isolate_projects_and_bound_concurrency(monkeypatch):
    monkeypatch.setattr("mindmemos_eval.memory.memoryarena.env.AsyncHttpTransport", _RecordingTransport)
    _RecordingTransport.reset()

    backend = MemoryArenaFormalBackend(agent_factory=_FakeAgent, env_client_factory=_FakeEnvClient)
    keys = generate_project_keys("run", "math", 4)
    env = MemoryArenaFormalReasoningEnv(
        backend=backend,
        env_name="math",
        agent_config={"model_name": "fake"},
        episode_keys=keys,
        concurrency=2,
        base_url="https://api.test",
    )
    tasks = [
        MemoryArenaFormalTask(id=f"p{i}", paper_name=f"paper-{i}", questions=["q?"], answers=["42"]) for i in range(4)
    ]

    run = await env.run_dataset(tasks, show_progress=False, print_report=False, run_id="run")

    # Each episode used its own api key -> its own project_id (no cross-pollution).
    assert sorted(_RecordingTransport.api_keys) == sorted(k.api_key for k in keys)
    assert len(set(_RecordingTransport.api_keys)) == 4
    assert {body["user_id"] for body in _RecordingTransport.bodies if "user_id" in body} == {
        f"memoryarena::math::{idx}::paper-{idx}::run" for idx in range(4)
    }
    # The semaphore caps simultaneous episodes at the configured concurrency.
    assert _RecordingTransport.max_active <= 2
    assert _RecordingTransport.max_active == 2
    # Results preserve episode order.
    assert [ep.task_key for ep in run.episodes] == ["paper-0", "paper-1", "paper-2", "paper-3"]


def test_env_requires_memory_or_episode_keys():
    backend = MemoryArenaFormalBackend(agent_factory=_FakeAgent, env_client_factory=_FakeEnvClient)
    with pytest.raises(ValueError, match="memory.*episode_keys"):
        MemoryArenaFormalReasoningEnv(backend=backend, env_name="math")


def test_formal_paper_metrics_match_official_eval_semantics():
    episodes = [
        MemoryArenaFormalEpisodeResult(
            task_id="t1",
            user_id="u1",
            task_key="paper-1",
            env_name="math",
            steps=[
                MemoryArenaFormalStepResult(
                    query_id=0,
                    query="q0",
                    prompt="p0",
                    output="a0",
                    reward=1.0,
                    memory_context="<memory_context>one</memory_context>",
                    elapsed_seconds=2.0,
                ),
                MemoryArenaFormalStepResult(
                    query_id=1,
                    query="q1",
                    prompt="p1",
                    output="a1",
                    reward=0.0,
                    memory_context="<memory_context>two words</memory_context>",
                    elapsed_seconds=4.0,
                ),
            ],
            elapsed_seconds=6.0,
        )
    ]

    paper_metrics, table = calculate_memoryarena_formal_paper_metrics(episodes)

    assert paper_metrics["paper-1"]["progress_score"] == 0.5
    assert paper_metrics["paper-1"]["is_paper_correct"] is False
    assert table["avg_progress_score"] == 0.5
    assert table["overall_average_passrate"] == 0.0
    assert table["passrate_at_k"] == [1.0, 0.0]
    assert table["cummulative_passrate_at_k"] == [1.0, 0.5]
    assert table["min_k"] == 2
