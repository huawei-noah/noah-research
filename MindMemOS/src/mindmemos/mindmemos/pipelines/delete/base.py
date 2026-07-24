from typing import Protocol

from ...typing import DeletePipelineInput, DeletePipelineResult, MemoryRequestContext


class DeletePipeline(Protocol):
    async def delete(self, inp: DeletePipelineInput, context: MemoryRequestContext) -> DeletePipelineResult:
        """Delete or archive memories matching the request.

        Args:
            inp: Delete request payload and mode.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The delete operation result.
        """
