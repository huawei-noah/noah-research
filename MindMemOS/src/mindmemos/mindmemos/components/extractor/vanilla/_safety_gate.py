"""Safety gate for the add pipeline — deterministic validation before DB write."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ....typing import MemoryType, PreprocessedText

AddActionType = Literal["ADD", "REINFORCE", "UPDATE", "MERGE", "SKIP"]


class PlannedAddAction(BaseModel):
    """A safety gate decision for one extracted or fallback memory candidate."""

    action: AddActionType
    content: str
    mem_type: MemoryType | None = None
    reason: str
    target_memory_id: str | None = None
    related_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    extractor_action_hint: str | None = None


class AddSafetyGate:
    """Apply deterministic validation before DB write.

    With recall-before-extract, the LLM extractor already makes action
    decisions with recall context.  This gate validates those decisions
    rather than re-deriving them from recall.
    """

    def __init__(
        self,
        *,
        default_mem_type: MemoryType = "fact",
        min_content_chars: int = 1,
        allowed_memory_types: set[MemoryType] | None = None,
        min_update_confidence: float = 0.7,
        min_merge_confidence: float = 0.8,
    ) -> None:
        self._default_mem_type = default_mem_type
        self._min_content_chars = min_content_chars
        self._allowed_memory_types = allowed_memory_types
        self._min_update_confidence = min_update_confidence
        self._min_merge_confidence = min_merge_confidence

    def gate_segment(
        self,
        preprocessed: PreprocessedText,
        *,
        mem_type: MemoryType | None = None,
        action_hint: str | None = None,
        confidence: float | None = None,
        target_memory_id: str | None = None,
        related_memory_ids: list[str] | None = None,
    ) -> PlannedAddAction:
        """Validate an extracted candidate and produce a planned action.

        The LLM extractor (with recall context) provides action_hint,
        target_memory_id, and related_memory_ids.  This method validates
        those decisions and applies deterministic guardrails.
        """
        content = preprocessed.normalized_text.strip()
        target_type = mem_type or self._default_mem_type
        related = related_memory_ids or []
        conf = confidence if confidence is not None else 0.0

        # Validation 1: Empty content
        if len(content) < self._min_content_chars:
            return PlannedAddAction(
                action="SKIP",
                content=content,
                reason="empty_or_too_short",
                extractor_action_hint=action_hint,
            )

        # Validation 2: Disallowed memory type
        if self._allowed_memory_types is not None and target_type not in self._allowed_memory_types:
            return PlannedAddAction(
                action="SKIP",
                content=content,
                mem_type=target_type,
                reason="memory_type_not_allowed",
                extractor_action_hint=action_hint,
            )

        # Validation 3: Honor explicit skip
        if action_hint == "skip":
            return PlannedAddAction(
                action="SKIP",
                content=content,
                mem_type=target_type,
                reason="extractor_skip_hint",
                extractor_action_hint=action_hint,
            )

        # Validation 4: Confidence threshold for update actions
        if action_hint == "update" and conf < self._min_update_confidence:
            return PlannedAddAction(
                action="ADD",
                content=content,
                mem_type=target_type,
                reason="update_low_confidence",
                related_memory_ids=related,
                metadata={"content_hash": preprocessed.content_hash},
                extractor_action_hint=action_hint,
            )

        # Validation 5: Confidence threshold for merge actions
        if action_hint == "merge" and conf < self._min_merge_confidence:
            return PlannedAddAction(
                action="ADD",
                content=content,
                mem_type=target_type,
                reason="merge_low_confidence",
                related_memory_ids=related,
                metadata={"content_hash": preprocessed.content_hash},
                extractor_action_hint=action_hint,
            )

        # Map action_hint to PlannedAddAction.action
        action_map = {"add": "ADD", "reinforce": "REINFORCE", "update": "UPDATE", "merge": "MERGE"}
        action = action_map.get(action_hint or "add", "ADD")

        # Validation 6: update/reinforce without target → downgrade to ADD
        if action in ("UPDATE", "REINFORCE") and target_memory_id is None:
            return PlannedAddAction(
                action="ADD",
                content=content,
                mem_type=target_type,
                reason=f"{action_hint}_no_target",
                related_memory_ids=related,
                metadata={"content_hash": preprocessed.content_hash},
                extractor_action_hint=action_hint,
            )

        # Validation 7: merge with insufficient targets → downgrade to ADD
        if action == "MERGE" and len(related) < 2:
            return PlannedAddAction(
                action="ADD",
                content=content,
                mem_type=target_type,
                reason="merge_insufficient_targets",
                related_memory_ids=related,
                metadata={"content_hash": preprocessed.content_hash},
                extractor_action_hint=action_hint,
            )

        return PlannedAddAction(
            action=action,
            content=content,
            mem_type=target_type,
            reason=f"extractor_{action_hint or 'add'}_hint",
            target_memory_id=target_memory_id,
            related_memory_ids=related,
            metadata={"content_hash": preprocessed.content_hash, "entity_count": len(preprocessed.entities)},
            extractor_action_hint=action_hint,
        )
