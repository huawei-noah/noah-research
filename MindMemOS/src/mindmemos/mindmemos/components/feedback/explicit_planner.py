"""LLM action planning for explicit feedback."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel

from ...llm import LLMClient, get_llm_client
from ...prompts.EN.feedback import EXPLICIT_ACTION_PLANNING_PROMPT, EXPLICIT_SEARCH_DECISION_PROMPT
from ...typing import FeedbackActionResult, FeedbackPipelineInput


class ExplicitFeedbackPlanner(Protocol):
    """Plan actions for explicit feedback without executing them."""

    async def decide_memory_search(self, inp: FeedbackPipelineInput) -> "FeedbackMemorySearchDecision":
        """Decide whether explicit feedback needs one supplemental memory search."""

    async def plan(self, inp: FeedbackPipelineInput) -> list[FeedbackActionResult]:
        """Return feedback actions for the provided explicit feedback input."""


class FeedbackMemorySearchDecision(BaseModel):
    need_search: bool
    """Whether another memory search is needed before planning write actions."""

    query: str | None = None
    """Search query to use when need_search is true."""


class ExplicitFeedbackActionPlan(BaseModel):
    """Planned feedback actions for one explicit feedback request."""

    actions: list[FeedbackActionResult]


class DefaultExplicitFeedbackPlanner:
    """Plan explicit feedback actions with an LLM."""

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    async def decide_memory_search(self, inp: FeedbackPipelineInput) -> FeedbackMemorySearchDecision:
        """Use one LLM call to decide whether more memory should be searched."""

        payload = _feedback_payload(inp)
        response = await self._client.chat(
            task="feedback.explicit.search_decision",
            messages=[
                {"role": "system", "content": EXPLICIT_SEARCH_DECISION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format_parser=_parse_search_decision,
            temperature=0,
        )
        decision = response.parsed
        if not isinstance(decision, FeedbackMemorySearchDecision):
            msg = "explicit feedback planner expected parsed search decision"
            raise TypeError(msg)
        if decision.need_search and not decision.query:
            msg = "explicit feedback search decision requires query when need_search is true"
            raise ValueError(msg)
        return decision

    async def plan(self, inp: FeedbackPipelineInput) -> list[FeedbackActionResult]:
        """Return planned memory actions for explicit user feedback."""

        payload = _feedback_payload(inp)
        response = await self._client.chat(
            task="feedback.explicit.plan",
            messages=[
                {"role": "system", "content": EXPLICIT_ACTION_PLANNING_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format_parser=_parse_feedback_plan,
            temperature=0,
        )
        plan = response.parsed
        if not isinstance(plan, ExplicitFeedbackActionPlan):
            msg = "explicit feedback planner expected parsed feedback plan"
            raise TypeError(msg)
        return plan.actions

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client


def _feedback_payload(inp: FeedbackPipelineInput) -> dict:
    return {
        "feedback": inp.feedback,
        "messages": [message.model_dump(mode="json") for message in inp.messages],
        "recalled_memories": [memory.model_dump(mode="json") for memory in inp.recalled_memories],
    }


def _parse_search_decision(content: str) -> FeedbackMemorySearchDecision:
    text = _json_object_text(content)
    return FeedbackMemorySearchDecision.model_validate_json(text)


def _parse_feedback_plan(content: str) -> ExplicitFeedbackActionPlan:
    text = _json_object_text(content)
    return ExplicitFeedbackActionPlan.model_validate_json(text)


def _json_object_text(content: str) -> str:
    text = content.strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return text[start : end + 1]
