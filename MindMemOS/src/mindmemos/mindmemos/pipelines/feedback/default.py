"""Default feedback pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

from ...infra.kafka import get_producer
from ...typing import FeedbackPipelineInput, FeedbackPipelineResult, MemoryRequestContext
from ..registry import register
from .explicit import ExplicitFeedbackHandler
from .implicit import ImplicitFeedbackHandler

MEMORY_FEEDBACK_TOPIC = "memory.feedback"


@register(type="feedback", name="default_feedback")
class DefaultFeedbackPipeline:
    """Route feedback requests to explicit or implicit handlers."""

    def __init__(
        self,
        *,
        explicit_handler: ExplicitFeedbackHandler | None = None,
        implicit_handler: ImplicitFeedbackHandler | None = None,
    ) -> None:
        self._explicit = explicit_handler
        self._implicit = implicit_handler

    async def feedback(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Compatibility wrapper for callers that still use the old single entrypoint."""

        if inp.mode == "async":
            return await self.feedback_async(inp, context)
        return await self.feedback_sync(inp, context)

    async def feedback_sync(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Route explicit feedback by payload, otherwise run implicit feedback."""

        if inp.feedback:
            if self._explicit is None:
                self._explicit = ExplicitFeedbackHandler()
            return await self._explicit.run(inp, context)

        if self._implicit is None:
            self._implicit = ImplicitFeedbackHandler()
        return await self._implicit.run(inp, context)

    async def feedback_async(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Queue feedback work for the Kafka worker."""

        await get_producer().send(
            MEMORY_FEEDBACK_TOPIC,
            value={
                "context": context.model_dump(mode="json"),
                "input": inp.model_dump(mode="json"),
                "submitted_at": datetime.now(UTC).isoformat(),
            },
            dispatch_key=f"{context.project_id}:{context.user_id}",
        )
        return FeedbackPipelineResult(status="queued", message="feedback queued")
