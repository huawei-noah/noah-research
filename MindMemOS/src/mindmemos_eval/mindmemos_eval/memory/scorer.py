"""Scoring module for memory evaluation.

A :class:`Scorer` turns a (question, predicted answer, gold answer) triple into a
:class:`ScoreResult`. Two implementations ship here:

- :class:`ExactMatchScorer`: a cheap, deterministic baseline (normalized string
  / substring match), no LLM required.
- :class:`LLMJudgeScorer`: uses an :class:`~mindmemos_eval.llm.LLMClient` to
  judge semantic correctness, returning a 0..1 score with a short reason.

Custom metrics subclass :class:`Scorer` and implement :meth:`Scorer.score`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..llm import LLMClient, LLMConfig


class ScoreResult(BaseModel):
    """Score result for one evaluated prediction."""

    model_config = ConfigDict(extra="ignore")

    score: float
    passed: bool
    reason: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class Scorer:
    """Base scorer interface."""

    async def score(
        self,
        *,
        question: str,
        answer: str,
        gold: str,
        contexts: list[str] | None = None,
    ) -> ScoreResult:
        """Score one predicted answer."""
        raise NotImplementedError


def _normalize(text: str) -> str:
    """Normalize text for exact or substring matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ExactMatchScorer(Scorer):
    """Normalized exact or substring scorer that does not call an LLM."""

    def __init__(self, *, substring: bool = True) -> None:
        """Handle init."""
        self._substring = substring

    async def score(
        self,
        *,
        question: str,
        answer: str,
        gold: str,
        contexts: list[str] | None = None,
    ) -> ScoreResult:
        norm_answer = _normalize(answer)
        norm_gold = _normalize(gold)
        if self._substring:
            passed = bool(norm_gold) and norm_gold in norm_answer
        else:
            passed = norm_answer == norm_gold
        return ScoreResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="exact match" if passed else "no match",
        )


DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for a question-answering system. "
    "Given a question, a reference (gold) answer, and a predicted answer, decide whether "
    "the predicted answer is correct. The predicted answer is correct if it conveys the same "
    "factual information as the gold answer, even with different wording. "
    "Respond ONLY with a compact JSON object of the form "
    '{"score": <float between 0 and 1>, "correct": <true|false>, "reason": "<short explanation>"}.'
)


class LLMJudgeScorer(Scorer):
    """Semantic LLM-as-a-judge scorer."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        config: LLMConfig | None = None,
        system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
        pass_threshold: float = 0.5,
    ) -> None:
        """Handle init."""
        self._llm = llm or LLMClient(config)
        self._system_prompt = system_prompt
        self._threshold = pass_threshold

    def _build_messages(self, question: str, answer: str, gold: str) -> list[dict[str, Any]]:
        user = (
            f"Question:\n{question}\n\n"
            f"Gold answer:\n{gold}\n\n"
            f"Predicted answer:\n{answer}\n\n"
            "Grade the predicted answer."
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    async def score(
        self,
        *,
        question: str,
        answer: str,
        gold: str,
        contexts: list[str] | None = None,
    ) -> ScoreResult:
        raw_text = (await self._llm.complete(self._build_messages(question, answer, gold))).content
        payload = _parse_judge_json(raw_text)

        score_val = _coerce_score(payload.get("score"), payload.get("correct"))
        reason = str(payload.get("reason", "")).strip()
        passed = bool(payload.get("correct")) if "correct" in payload else score_val >= self._threshold
        return ScoreResult(score=score_val, passed=passed, reason=reason, raw=payload)


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM judge output."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"score": 0.0, "correct": False, "reason": f"unparseable judge output: {text[:200]}"}


def _coerce_score(score: Any, correct: Any) -> float:
    """Normalize judge score or correctness fields to a float."""
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score)))
    if isinstance(correct, bool):
        return 1.0 if correct else 0.0
    return 0.0
