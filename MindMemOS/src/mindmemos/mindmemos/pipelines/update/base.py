from typing import Protocol

from ...typing import MemoryRequestContext, UpdatePipelineInput, UpdatePipelineResult


class UpdatePipeline(Protocol):
    async def update(self, inp: UpdatePipelineInput, context: MemoryRequestContext) -> UpdatePipelineResult:
        """Update a memory in the current project.

        Args:
            inp: Update request payload.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The update operation result.
        """
