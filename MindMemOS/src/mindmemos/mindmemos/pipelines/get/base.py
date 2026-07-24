from typing import Protocol

from ...typing import GetPipelineInput, GetPipelineResult, MemoryRequestContext


class GetPipeline(Protocol):
    async def get(self, inp: GetPipelineInput, context: MemoryRequestContext) -> GetPipelineResult:
        """Fetch memories by id or request filters.

        Args:
            inp: Get request payload.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The hydrated memory results.
        """
