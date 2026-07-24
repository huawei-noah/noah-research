"""Offline tests for PersonaMem-Evo SDK helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mindmemos.typing.llm import ChatResponse
from mindmemos_sdk.memory import AsyncMemoryClient
from mindmemos_sdk.transport import AsyncHttpTransport

from mindmemos_eval import (
    CompletionResult,
    PersonaMemEvoAnswer,
    PersonaMemEvoEnv,
    PersonaMemEvoQAResult,
    calculate_personamem_evo_metrics,
    check_mcq_correctness,
    create_mcq_options,
    extract_final_answer,
    parse_incorrect_answers,
    parse_user_query,
)


class _FakeProjectLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **params: Any) -> ChatResponse:
        self.calls.append(params)
        return ChatResponse(finish_reason="stop", content=self.reply, model="fake-answer-model")

    async def complete(self, messages: list[dict[str, Any]], **overrides: Any) -> CompletionResult:
        self.calls.append({"messages": messages, **overrides})
        return CompletionResult(content=self.reply)


def _llm(reply: str) -> _FakeProjectLLM:
    return _FakeProjectLLM(reply)


def _memory(memories: list[dict[str, Any]]):
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append({"path": request.url.path, "body": body, "headers": dict(request.headers)})
        if request.url.path.endswith("/add"):
            data = {"memories": [{"operation": "ADD", "content": item["memory"], "memory_id": item["id"]} for item in memories]}
        else:
            data = {"memories": memories} if request.url.path.endswith("/search") else {"memories": []}
        return httpx.Response(200, json={"code": "ok", "message": "done", "request_id": "r", "data": data})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(base_url="https://api.test", api_key="dev", client=client)
    return AsyncMemoryClient(transport), captured


def _row(chat_history_path: str) -> dict[str, str]:
    return {
        "persona_id": "7",
        "user_query": json.dumps({"role": "user", "content": "What snack do I prefer?"}),
        "correct_answer": "almonds",
        "incorrect_answers": json.dumps(["cookies", "chips", "cake"]),
        "chat_history_32k_link": chat_history_path,
        "chain_id": "chain-a",
        "ood_type": "preference",
        "ood_difficulty": "easy",
    }


def test_personamem_evo_parsers_and_mcq_helpers():
    parsed = parse_user_query(json.dumps({"role": "user", "content": "What do I like?"}))
    assert parsed["role"] == "user"
    assert parsed["content"].startswith("Please recall")
    assert "What do I like?" in parsed["content"]
    assert parse_incorrect_answers('["b", "c"]') == ["b", "c"]

    options, correct_letter = create_mcq_options("a", ["b", "c", "d"], seed=1)
    assert options[correct_letter] == "a"
    assert extract_final_answer("Reasoning...\nFinal Answer: b") == "B"
    assert check_mcq_correctness(correct_letter, correct_letter, options) is True


def test_personamem_evo_format_context_marks_archived_lineage():
    env = PersonaMemEvoEnv(object(), answer_llm=_llm("Final Answer: A"))

    contexts = env.format_context(
        [
            SimpleNamespace(memory="Current preference.", lineage=SimpleNamespace(role="current")),
            SimpleNamespace(memory="Old preference.", lineage=SimpleNamespace(role="archived")),
        ]
    )

    assert contexts == ["Current preference.", "[历史版本] Old preference."]


@pytest.mark.asyncio
async def test_async_feedback_sends_actor_body_when_provided():
    memory, captured = _memory([])

    result = await memory.feedback(
        mode="sync",
        user_id="u1",
        app_id="app",
        agent_id="agent",
        session_id="session",
    )

    assert result.code == "ok"
    assert captured[0]["path"] == "/v1/memory/feedback"
    assert captured[0]["body"] == {
        "user_id": "u1",
        "app_id": "app",
        "agent_id": "agent",
        "session_id": "session",
        "mode": "sync",
    }


@pytest.mark.asyncio
async def test_run_dataset_add_feedback_uses_offline_memory_contract(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            [
                {"role": "system", "content": "background"},
                {"role": "user", "content": "I prefer almonds."},
                {"role": "assistant", "content": "Noted."},
            ]
        ),
        encoding="utf-8",
    )
    memory, captured = _memory([{"id": "m1", "memory": "The user prefers almonds."}])
    llm = _llm("Final Answer: A")
    env = PersonaMemEvoEnv(memory, answer_llm=llm, add_batch_size=2, top_k=5, user_id_prefix="pm")

    item = env.build_items([_row(str(history))], size="32k", persona_root=tmp_path)[0]
    # Make the fake LLM's answer correct regardless of deterministic shuffle.
    llm.reply = f"Final Answer: {item.correct_letter}"
    run = await env.run_dataset([_row(str(history))], size="32k", persona_root=tmp_path, mode="add_feedback", show_progress=False)

    assert run.metrics["step_accuracy"] == 1.0
    assert run.metrics["chain_accuracy"] == 1.0
    assert run.metrics["memory_added_messages"] == 3
    assert run.metrics["memory_returned_count"] == 2
    assert run.metrics["api_add_calls"] == 2
    assert run.metrics["api_feedback_calls"] == 1
    assert run.metrics["api_search_calls"] == 1
    assert run.metrics["api_total_calls"] == 4
    assert run.metrics["answer_llm_calls"] == 1
    assert run.metrics["feedback_elapsed_seconds"] >= 0
    assert run.metrics["total_elapsed_seconds"] >= 0
    assert [call["path"] for call in captured] == [
        "/v1/memory/add",
        "/v1/memory/add",
        "/v1/memory/feedback",
        "/v1/memory/search",
    ]
    add_body = captured[0]["body"]
    assert add_body["user_id"] == "pm-add_feedback-7"
    assert add_body["session_id"] == "personamem-evo-add_feedback-7"
    assert add_body["metadata"]["benchmark"] == "personamem-evo"
    feedback_call = captured[2]
    assert feedback_call["body"] == {"user_id": "pm-add_feedback-7", "mode": "sync"}
    search_body = captured[3]["body"]
    assert search_body["top_k"] == 5
    assert search_body["search_strategy"] == "fast"
    assert "What snack do I prefer?" in search_body["query"]
    assert "Memories:" in llm.calls[0]["messages"][-1]["content"]


def test_personamem_evo_chain_metrics_require_all_steps_correct():
    results = [
        PersonaMemEvoQAResult(
            index=0,
            persona_id="p",
            user_id="u",
            question="q1",
            correct_answer="a",
            chain_key=("chain", "1"),
            answer=PersonaMemEvoAnswer(
                response="Final Answer: A",
                predicted_answer="A",
                is_correct=True,
                correct_letter="A",
                option_mapping={"A": "a"},
            ),
        ),
        PersonaMemEvoQAResult(
            index=1,
            persona_id="p",
            user_id="u",
            question="q2",
            correct_answer="b",
            chain_key=("chain", "1"),
            answer=PersonaMemEvoAnswer(
                response="Final Answer: C",
                predicted_answer="C",
                is_correct=False,
                correct_letter="B",
                option_mapping={"B": "b"},
            ),
        ),
    ]

    metrics = calculate_personamem_evo_metrics(results)

    assert metrics["step_accuracy"] == 0.5
    assert metrics["chain_total"] == 1
    assert metrics["chain_accuracy"] == 0.0
