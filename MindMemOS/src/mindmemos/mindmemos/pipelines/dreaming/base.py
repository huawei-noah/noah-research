from typing import Protocol

from ...typing import DreamingPipelineInput, DreamingPipelineResult, MemoryRequestContext


class DreamingPipeline(Protocol):
    async def dream(self, inp: DreamingPipelineInput, context: MemoryRequestContext) -> DreamingPipelineResult:
        """Run or queue memory consolidation for the request context.

        Args:
            inp: Dreaming request options.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The dreaming operation status.
        """
