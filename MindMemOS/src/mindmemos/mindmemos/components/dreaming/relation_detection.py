"""DTOs and parser for candidate-first dreaming consolidation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DreamingIssueType = Literal[
    "conflict",
    "duplicate",
    "near_duplicate",
    "complementary",
    "low_value",
    "ambiguous",
    "other",
]


class DetectedMemoryRelation(BaseModel):
    """A backward-compatible candidate memory pair worth sending to action planning."""

    candidate_type: Literal["needs_consolidation", "ignore"]
    primary_memory_id: str
    neighbor_memory_id: str
    subject_hint: str | None = None
    predicate_hint: str | None = None
    primary_value_hint: str | None = None
    neighbor_value_hint: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str = ""


class DetectedMemoryIssueGroup(BaseModel):
    """A problem-specific memory group emitted by the first dreaming LLM.

    Purpose: Carries one issue category and the memory IDs involved in that
    issue, so the second LLM can focus on solving one problem type at a time.
    Used in: DefaultDreamingPipeline relation detection output and action
    planning input construction.
    """

    issue_type: DreamingIssueType
    memory_ids: list[str] = Field(default_factory=list)
    subject_hint: str | None = None
    predicate_hint: str | None = None
    value_hints: dict[str, str] = Field(default_factory=dict)
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str = ""

    @field_validator("memory_ids")
    @classmethod
    def _dedupe_memory_ids(cls, value: list[str]) -> list[str]:
        """去重并保留模型输出中的记忆顺序。"""

        return list(dict.fromkeys(str(item) for item in value if str(item)))


class DetectedRelationBatch(BaseModel):
    """Batch of issue groups detected for one seed-centric graph scope."""

    issue_groups: list[DetectedMemoryIssueGroup] = Field(default_factory=list)
    candidates: list[DetectedMemoryRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _backfill_issue_groups_from_candidates(self) -> "DetectedRelationBatch":
        """兼容旧版 pair-only 输出，将候选 pair 转为 ambiguous issue group。"""

        if self.issue_groups:
            return self
        groups: list[DetectedMemoryIssueGroup] = []
        for candidate in self.candidates:
            if candidate.candidate_type != "needs_consolidation":
                continue
            groups.append(
                DetectedMemoryIssueGroup(
                    issue_type="ambiguous",
                    memory_ids=[candidate.primary_memory_id, candidate.neighbor_memory_id],
                    subject_hint=candidate.subject_hint,
                    predicate_hint=candidate.predicate_hint,
                    value_hints={
                        candidate.primary_memory_id: candidate.primary_value_hint or "",
                        candidate.neighbor_memory_id: candidate.neighbor_value_hint or "",
                    },
                    confidence=candidate.confidence,
                    reason=candidate.reason,
                )
            )
        self.issue_groups = groups
        return self


def relation_detection_parser(content: str) -> DetectedRelationBatch:
    """Parse candidate detector JSON output."""

    text = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return DetectedRelationBatch.model_validate_json(text)


__all__ = [
    "DetectedMemoryIssueGroup",
    "DetectedMemoryRelation",
    "DetectedRelationBatch",
    "DreamingIssueType",
    "relation_detection_parser",
]
