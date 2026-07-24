"""LLM action planning for implicit feedback signals."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ...llm import LLMClient, get_llm_client
from ...prompts.EN.feedback import IMPLICIT_ACTION_PLANNING_PROMPT
from ...typing import FeedbackActionResult, ImplicitFeedbackRound, ImplicitFeedbackSignal, MemorySearchItem


class ImplicitFeedbackActionPlan(BaseModel):
    """Planned feedback actions for implicit feedback signals in one round."""

    actions: list[FeedbackActionResult] = Field(default_factory=list)


class ImplicitFeedbackActionPlanner:
    """Plan memory actions for one round's implicit feedback signals and memory pool."""

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    async def plan(
        self,
        *,
        round_: ImplicitFeedbackRound,
        signals: list[ImplicitFeedbackSignal],
        memories: list[MemorySearchItem],
    ) -> list[FeedbackActionResult]:
        payload = {
            "signals": [signal.model_dump(mode="json") for signal in signals],
            "round": round_.model_dump(mode="json"),
            "memories": [memory.model_dump(mode="json") for memory in memories],
        }
        response = await self._client.chat(
            task="feedback.implicit.plan_actions",
            messages=[
                {"role": "system", "content": IMPLICIT_ACTION_PLANNING_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format_parser=_parse_action_plan,
            temperature=0,
        )
        plan = response.parsed
        if not isinstance(plan, ImplicitFeedbackActionPlan):
            msg = "implicit feedback action planner expected parsed action plan"
            raise TypeError(msg)
        return plan.actions

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client


def _parse_action_plan(content: str) -> ImplicitFeedbackActionPlan:
    return ImplicitFeedbackActionPlan.model_validate_json(_json_object_text(content))


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
