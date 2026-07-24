"""Tests for MemoryAgentBenchEnv."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from mindmemos_sdk.memory import AsyncMemoryClient
from mindmemos_sdk.transport import AsyncHttpTransport

from mindmemos_eval import (
    LLMClient,
    LLMConfig,
    MemoryAgentBenchEnv,
    MemoryAgentBenchItem,
    calculate_memoryagentbench_metrics,
    primary_metric_for_sub_dataset,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, parent: _FakeOpenAI) -> None:
        self._parent = parent

    async def create(self, **params: Any) -> _FakeResponse:
        self._parent.calls.append(params)
        return _FakeResponse(self._parent.reply)


class _FakeChat:
    def __init__(self, parent: _FakeOpenAI) -> None:
        self.completions = _FakeCompletions(parent)


class _FakeOpenAI:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


def _llm(reply: str) -> LLMClient:
    return LLMClient(LLMConfig(model="test"), client=_FakeOpenAI(reply))


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


def _item() -> MemoryAgentBenchItem:
    context = " ".join(["Alice likes Paris."] * 80)
    return MemoryAgentBenchItem(
        context=context,
        questions=["Where does Alice like?"],
        answers=["Paris"],
        source="eventqa_32k",
        qa_pair_ids=["qa-1"],
    )


def _fact_item() -> MemoryAgentBenchItem:
    context = " ".join(["1. The capital is Alpha. 2. The capital is Beta."] * 80)
    return MemoryAgentBenchItem(
        context=context,
        questions=["what is the current capital?"],
        answers=["Beta"],
        source="factconsolidation_sh_6k",
        qa_pair_ids=["fact-1"],
    )


def test_metrics_follow_memoryagentbench_rules():
    metrics = calculate_memoryagentbench_metrics("Answer: Paris, France", ["Paris"], "eventqa_32k")
    assert metrics["substring_exact_match"] == 1.0
    assert metrics["exact_match"] == 0.0
    assert "rougeL_f1" in metrics
    assert primary_metric_for_sub_dataset("eventqa_32k") == "substring_exact_match"
    assert primary_metric_for_sub_dataset("icl_banking") == "exact_match"


def test_item_builds_official_agentic_query():
    question = _item().build_questions("eventqa_32k")[0]
    assert "Search Archival Memory" in question.query
    assert "Where does Alice like?" in question.query
    assert question.qa_pair_id == "qa-1"


@pytest.mark.asyncio
async def test_add_context_sends_sync_dialogue_chunks():
    memory, captured = _memory([])
    env = MemoryAgentBenchEnv(memory, answer_llm=_llm("x"), sub_dataset="eventqa_32k", chunk_size=20)

    summary = await env.add_context(_item(), context_id=0)

    assert summary.added_chunks > 1
    first = captured[0]["body"]
    assert captured[0]["path"] == "/v1/memory/add"
    assert first["mode"] == "sync"
    assert first["user_id"] == "context_0_eventqa_32k"
    assert first["session_id"] == "context_0_eventqa_32k"
    assert [message["role"] for message in first["messages"]] == ["system", "user", "assistant"]
    assert "book excerpt" in first["messages"][1]["content"]
    assert first["messages"][2]["content"] == "I'll make sure to add the content into the memory."


@pytest.mark.asyncio
async def test_answer_searches_and_scores_primary_metric():
    memory, captured = _memory([{"id": "m1", "memory": "Alice likes Paris."}])
    llm = _llm("Answer: Paris")
    env = MemoryAgentBenchEnv(memory, answer_llm=llm, sub_dataset="eventqa_32k", top_k=7, search_strategy="fast")
    question = _item().build_questions("eventqa_32k")[0]

    result = await env.evaluate_question("context_0_eventqa_32k", question, query_id=0, context_id=0)

    assert result.score is not None and result.score.passed is True
    assert result.metrics["substring_exact_match"] == 1.0
    search_call = next(call for call in captured if call["path"].endswith("/search"))
    assert search_call["body"]["top_k"] == 7
    assert search_call["body"]["search_strategy"] == "fast"
    prompt = llm._client.calls[0]["messages"][-1]["content"]
    system_prompt = llm._client.calls[0]["messages"][0]["content"]
    assert "Answer the question based on query and memories" in system_prompt
    assert "Alice likes Paris." in system_prompt
    assert "Current Time:" in prompt


@pytest.mark.asyncio
async def test_run_dataset_limits_queries():
    memory, _ = _memory([{"id": "m1", "memory": "Alice likes Paris."}])
    env = MemoryAgentBenchEnv(memory, answer_llm=_llm("Paris"), sub_dataset="eventqa_32k", chunk_size=1000)

    run = await env.run_dataset([_item(), _item()], max_queries=1, show_progress=False)

    assert sum(ctx.num_questions for ctx in run.contexts) == 1
    assert run.averaged_metrics["substring_exact_match"] == 1.0


@pytest.mark.asyncio
async def test_run_dataset_uses_each_item_source_when_unfiltered():
    memory, captured = _memory([{"id": "m1", "memory": "The capital is Beta."}])
    env = MemoryAgentBenchEnv(memory, answer_llm=_llm("Beta"), chunk_size=1000)

    run = await env.run_dataset([_item(), _fact_item()], show_progress=False)

    assert [ctx.user_id for ctx in run.contexts] == ["context_0_eventqa_32k", "context_1_factconsolidation_sh_6k"]
    assert "primary/eventqa_32k/substring_exact_match" in run.metrics
    assert "primary/factconsolidation_sh_6k/substring_exact_match" in run.metrics
    add_bodies = [call["body"] for call in captured if call["path"].endswith("/add")]
    assert any("book excerpt" in body["messages"][1]["content"] for body in add_bodies)
    assert any("facts I have learned" in body["messages"][1]["content"] for body in add_bodies)
