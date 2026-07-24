from typing import Protocol

from ...typing import FeedbackPipelineInput, FeedbackPipelineResult, MemoryRequestContext


class FeedbackPipeline(Protocol):
    async def feedback_sync(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Apply feedback immediately.

        Args:
            inp: Feedback request payload.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The completed feedback result.
        """

    async def feedback_async(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Queue feedback processing.

        Args:
            inp: Feedback request payload.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The queued feedback status.
        """
