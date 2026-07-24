"""Tests for asynchronous long-turn middle summarization."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from mindmemos.components.chunker.vanilla.summarizer import LongTurnSummarizer
from mindmemos.components.chunker.vanilla.turn_grouper import _estimate_tokens
from mindmemos.typing.algo import TurnCompactionSummary

from mindmemos.config import VanillaAddConfig


class _FakeLlmClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self._fail = fail

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("summary unavailable")
        number = len(self.calls)
        return SimpleNamespace(
            parsed=TurnCompactionSummary(
                general_summary=f"summary-{number}",
                key_entities=[f"entity-{number}"],
            )
        )


def _middle_text(call: dict) -> str:
    return json.loads(call["messages"][-1]["content"])["middle_text"]


@pytest.mark.asyncio
async def test_middle_under_context_budget_uses_one_call() -> None:
    client = _FakeLlmClient()
    config = VanillaAddConfig(
        compaction_summary_context_token_budget=20,
        compaction_summary_output_token_budget=8,
    )

    summary = await LongTurnSummarizer(config, client).summarize("one two three")

    assert summary.general_summary == "summary-1"
    assert len(client.calls) == 1
    assert client.calls[0]["task"] == "memory.add.long_turn_summary"
    assert client.calls[0]["max_tokens"] == 8
    assert _middle_text(client.calls[0]) == "one two three"


@pytest.mark.asyncio
async def test_middle_over_context_budget_is_segmented_then_reduced() -> None:
    client = _FakeLlmClient()
    config = VanillaAddConfig(
        compaction_summary_context_token_budget=100,
        compaction_summary_output_token_budget=8,
    )
    middle = " ".join(f"word{i}" for i in range(250))

    summary = await LongTurnSummarizer(config, client).summarize(middle)

    assert summary.general_summary == f"summary-{len(client.calls)}"
    assert len(client.calls) == 4
    assert all(call["max_tokens"] == 8 for call in client.calls)
    assert all(_estimate_tokens(_middle_text(call)) <= 100 for call in client.calls)
    segment_inputs = [_middle_text(call) for call in client.calls[:-1]]
    assert "".join(segment_inputs).replace(" ", "") == middle.replace(" ", "")
    assert '"key_entities":["entity-1"]' in _middle_text(client.calls[-1])


@pytest.mark.asyncio
async def test_summary_failure_returns_omitted_middle_marker() -> None:
    client = _FakeLlmClient(fail=True)
    config = VanillaAddConfig(
        compaction_summary_context_token_budget=20,
        compaction_summary_output_token_budget=8,
    )

    summary = await LongTurnSummarizer(config, client).summarize("one two three")

    assert "omitted" in summary.general_summary.lower()
    assert "3 tokens" in summary.general_summary


@pytest.mark.asyncio
async def test_missing_llm_client_returns_omitted_middle_marker() -> None:
    summary = await LongTurnSummarizer(VanillaAddConfig(), None).summarize("one two three")

    assert "omitted" in summary.general_summary.lower()
