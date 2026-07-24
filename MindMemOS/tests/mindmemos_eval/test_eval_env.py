"""Tests for evaluation LLM client and scorers."""

from __future__ import annotations

from typing import Any

import pytest

from mindmemos_eval import (
    ExactMatchScorer,
    LLMClient,
    LLMConfig,
    LLMJudgeScorer,
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
        self.model = "gpt-test"
        self.usage = type("Usage", (), {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})()


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
    """Minimal stand-in for the OpenAI client used by LLMClient."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


@pytest.mark.asyncio
async def test_llm_client_complete_passes_params_and_strips():
    fake = _FakeOpenAI(reply="  hello world  ")
    llm = LLMClient(LLMConfig(model="gpt-test", temperature=0.3, max_tokens=64), client=fake)

    out = await llm.complete([{"role": "user", "content": "hi"}])

    assert out.content == "hello world"
    assert out.model == "gpt-test"
    assert out.prompt_tokens == 3
    assert out.completion_tokens == 2
    assert out.total_tokens == 5
    call = fake.calls[0]
    assert call["model"] == "gpt-test"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 64


@pytest.mark.asyncio
async def test_llm_client_complete_can_return_message_with_tools():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "function": {"name": "shell", "arguments": "{}"}}],
    }

    class FakeCompletions:
        async def create(self, **params: Any) -> _FakeResponse:
            fake.calls.append(params)
            response = _FakeResponse("")
            response.choices[0].message = message
            return response

    fake = _FakeOpenAI(reply="")
    fake.chat.completions = FakeCompletions()
    llm = LLMClient(LLMConfig(model="gpt-test"), client=fake)
    tools = [{"type": "function", "function": {"name": "shell"}}]

    out = await llm.complete([{"role": "user", "content": "hi"}], return_format="message", tools=tools)
    raw = await llm([{"role": "user", "content": "hi"}], tools=tools)

    assert out.message == {key: value for key, value in message.items() if value is not None}
    assert out.content == ""
    assert raw == out.message
    assert fake.calls[0]["tools"] == tools


@pytest.mark.asyncio
async def test_exact_match_scorer_substring():
    scorer = ExactMatchScorer()
    result = await scorer.score(question="q", answer="The capital is Paris.", gold="paris")
    assert result.passed is True
    assert result.score == 1.0

    miss = await scorer.score(question="q", answer="London", gold="paris")
    assert miss.passed is False
    assert miss.score == 0.0


@pytest.mark.asyncio
async def test_llm_judge_scorer_parses_json_codeblock():
    fake = _FakeOpenAI(reply='```json\n{"score": 0.9, "correct": true, "reason": "match"}\n```')
    scorer = LLMJudgeScorer(LLMClient(LLMConfig(model="test"), client=fake))

    result = await scorer.score(question="q", answer="a", gold="g")

    assert result.score == 0.9
    assert result.passed is True
    assert result.reason == "match"


@pytest.mark.asyncio
async def test_llm_judge_scorer_handles_unparseable_output():
    fake = _FakeOpenAI(reply="totally not json")
    scorer = LLMJudgeScorer(LLMClient(LLMConfig(model="test"), client=fake))

    result = await scorer.score(question="q", answer="a", gold="g")

    assert result.score == 0.0
    assert result.passed is False
