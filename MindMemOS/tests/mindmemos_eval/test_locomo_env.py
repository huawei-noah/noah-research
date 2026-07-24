"""Tests for LocomoEnv: session add (mode=sync), answering, scoring, concurrency."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from mindmemos_sdk.memory import AsyncMemoryClient
from mindmemos_sdk.transport import AsyncHttpTransport

from mindmemos_eval import LLMClient, LLMConfig, LocomoEnv, LocomoLLMJudgeScorer


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
    """Async memory client backed by a MockTransport; records request bodies."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append({"path": request.url.path, "body": body})
        if request.url.path.endswith("/search"):
            data = {"memories": memories}
        else:
            data = {"memories": []}
        return httpx.Response(200, json={"code": "ok", "message": "", "request_id": "r", "data": data})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(base_url="https://api.test", api_key="dev", client=client)
    return AsyncMemoryClient(transport), captured


def _conv_item() -> dict[str, Any]:
    return {
        "conversation": {
            "session_1": [
                {"speaker": "Caroline", "text": "Hey Mel!", "dia_id": "D1:1"},
                {"speaker": "Melanie", "text": "I moved to Paris.", "dia_id": "D1:2"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_2": [{"speaker": "Caroline", "text": "Nice!", "dia_id": "D2:1"}],
            "session_2_date_time": "1:14 pm on 25 May, 2023",
        },
        "qa": [
            {"question": "Where does Melanie live?", "answer": "Paris", "category": 2},
            {"question": "adversarial", "answer": "n/a", "category": 5},
        ],
    }


@pytest.mark.asyncio
async def test_add_session_sends_one_sync_add_with_speaker_roles():
    memory, captured = _memory([])
    env = LocomoEnv(memory, answer_llm=_llm("x"))

    messages = [
        {"speaker": "Caroline", "text": "Hi", "blip_caption": "a cat"},
        {"speaker": "Melanie", "text": "Hello"},
    ]
    await env.add_session("conv_0", messages, "1:56 pm on 8 May, 2023")

    assert len(captured) == 1
    body = captured[0]["body"]
    assert captured[0]["path"] == "/v1/memory/add"
    assert body["mode"] == "sync"
    assert body["user_id"] == "conv_0"
    assert body["session_id"] == "conv_0"
    # whole session added in one call, role == speaker, image caption appended
    assert [m["role"] for m in body["messages"]] == ["Caroline", "Melanie"]
    assert body["messages"][0]["content"] == "Hi [Shared image: a cat]"
    # all messages share the session timestamp (13-digit millis)
    ts = body["messages"][0]["timestamp"]
    assert ts == body["messages"][1]["timestamp"]
    assert ts == 1683554160000  # 2023-05-08 13:56:00 UTC


@pytest.mark.asyncio
async def test_add_conversation_serial_sessions():
    memory, captured = _memory([])
    env = LocomoEnv(memory, answer_llm=_llm("x"))

    summary = await env.add_conversation(_conv_item(), idx=0)

    assert summary.total_sessions == 2
    assert summary.added_sessions == 2
    assert summary.failed_sessions == []
    # two add calls (one per session), in session order
    add_paths = [c["path"] for c in captured]
    assert add_paths == ["/v1/memory/add", "/v1/memory/add"]
    assert captured[0]["body"]["messages"][0]["role"] == "Caroline"


@pytest.mark.asyncio
async def test_answer_builds_prompt_and_extracts_answer():
    memory, captured = _memory(
        [{"id": "m1", "memory": "Melanie lives in Paris", "event_time": "2023-05-08", "source_timestamp": "2023-05-08"}]
    )
    answer_llm = _llm("reasoning... <answer>Paris</answer>")
    env = LocomoEnv(memory, answer_llm=answer_llm, top_k=50, search_strategy="agentic")

    result = await env.answer("conv_0", "Where does Melanie live?")

    assert result.answer == "Paris"
    assert result.chain_of_thought.startswith("reasoning")
    assert result.memories == ["[event_time: 2023-05-08; source_timestamp: 2023-05-08] Melanie lives in Paris"]
    search_call = next(c for c in captured if c["path"].endswith("/search"))
    assert search_call["body"]["top_k"] == 50
    assert search_call["body"]["search_strategy"] == "agentic"
    answer_prompt = answer_llm._client.calls[0]["messages"][0]["content"]
    assert "Melanie lives in Paris" in answer_prompt


@pytest.mark.asyncio
async def test_locomo_judge_scorer_label_correct():
    scorer = LocomoLLMJudgeScorer(_llm('{"label": "CORRECT"}'))
    result = await scorer.score(question="q", answer="Paris", gold="Paris")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_evaluate_question_scores_and_skips_nothing():
    memory, _ = _memory([{"id": "m1", "memory": "Melanie lives in Paris"}])
    env = LocomoEnv(
        memory,
        answer_llm=_llm("<answer>Paris</answer>"),
        judge_llm=_llm('{"label": "CORRECT"}'),
    )
    q = {"question": "Where does Melanie live?", "answer": "Paris", "category": 2}

    result = await env.evaluate_question("conv_0", q)

    assert result.response == "Paris"
    assert result.score is not None and result.score.passed is True


@pytest.mark.asyncio
async def test_run_dataset_concurrency_and_accuracy():
    memory, captured = _memory([{"id": "m1", "memory": "Melanie lives in Paris"}])
    env = LocomoEnv(
        memory,
        answer_llm=_llm("<answer>Paris</answer>"),
        judge_llm=_llm('{"label": "CORRECT"}'),
    )

    data = [_conv_item(), _conv_item()]
    run = await env.run_dataset(data, max_conv_concurrency=2, max_qa_concurrency=4)

    # category-5 questions are skipped -> 1 scored question per conversation
    assert run.total_questions == 2
    assert run.correct == 2
    assert run.accuracy == 1.0
    assert len(run.conversations) == 2
    assert run.conversations[0].num_questions == 1
    assert run.conversations[0].add_summary.added_sessions == 2
    # each conversation issued 2 add calls (2 sessions) -> 4 total add calls
    assert sum(1 for c in captured if c["path"].endswith("/add")) == 4


@pytest.mark.asyncio
async def test_run_dataset_metrics_by_category_and_overall(capsys):
    item = {
        "conversation": {
            "session_1": [{"speaker": "Caroline", "text": "Hi"}],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
        },
        "qa": [
            {"question": "Where does Melanie live?", "answer": "Paris", "category": 2},
            {"question": "What city?", "answer": "Paris", "category": 3},
        ],
    }
    memory, _ = _memory([{"id": "m1", "memory": "Melanie lives in Paris"}])

    # answer LLM echoes a fixed answer; judge labels cat-2 CORRECT, cat-3 WRONG by
    # reading the question text out of the prompt.
    class _RoutingOpenAI(_FakeOpenAI):
        async def _route(self, params):
            content = params["messages"][0]["content"]
            if "Generated answer" in content:
                return '{"label": "CORRECT"}' if "Where does Melanie live?" in content else '{"label": "WRONG"}'
            return "<answer>Paris</answer>"

    routing = _RoutingOpenAI("")

    async def create(**params):
        routing.calls.append(params)
        return _FakeResponse(await routing._route(params))

    routing.chat.completions.create = create
    llm = LLMClient(LLMConfig(model="test"), client=routing)
    env = LocomoEnv(memory, answer_llm=llm, judge_llm=llm)

    run = await env.run_dataset([item], max_conv_concurrency=1, max_qa_concurrency=2)

    by_cat = run.by_category()
    assert by_cat["2"].count == 1 and by_cat["2"].correct == 1 and by_cat["2"].accuracy == 1.0
    assert by_cat["3"].count == 1 and by_cat["3"].correct == 0 and by_cat["3"].accuracy == 0.0
    overall = run.overall()
    assert overall.count == 2 and overall.correct == 1 and overall.accuracy == 0.5

    # the env printed the metric report (both dimensions with count + accuracy)
    out = capsys.readouterr().out
    assert "By category (count, accuracy):" in out
    assert "category 2: n=1 acc=1.0000" in out
    assert "category 3: n=1 acc=0.0000" in out
    assert "Overall (count, accuracy):" in out
    assert "n=2 acc=0.5000" in out


@pytest.mark.asyncio
async def test_run_dataset_no_score_prints_skip_note(capsys):
    memory, _ = _memory([{"id": "m1", "memory": "x"}])
    env = LocomoEnv(memory, answer_llm=_llm("<answer>Paris</answer>"))

    run = await env.run_dataset([_conv_item()], score=False)

    assert run.total_questions == 0
    assert run.is_scored() is False
    assert "scoring skipped" in capsys.readouterr().out
