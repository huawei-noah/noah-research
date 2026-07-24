"""Default delete-memory pipeline implementation."""

from __future__ import annotations

from ...typing import (
    DeletePipelineInput,
    DeletePipelineResult,
    MemoryDbDeleteCommand,
    MemoryDbMutationPlan,
    MemoryRequestContext,
)
from ..base import MemoryDbPipelineMixin
from ..registry import register


@register(type="delete", name="default_delete")
class DefaultDeletePipeline(MemoryDbPipelineMixin):
    """Archive one memory through the project-scoped memory DB writer."""

    async def delete(self, inp: DeletePipelineInput, context: MemoryRequestContext) -> DeletePipelineResult:
        """Archive a memory by id in the current project.

        Args:
            inp: Delete request containing the memory id.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            An ok result when the memory was archived, otherwise an error result.
        """
        command = MemoryDbDeleteCommand(memory_id=inp.id)
        write_result = await self.db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan(memory_deletes=[command]),
        )
        result = write_result.mutations[0] if write_result.mutations else None
        if result is None:
            return DeletePipelineResult(status="error", message=f"memory delete not applied: {inp.id}")
        if not result.changed:
            return DeletePipelineResult(status="error", message=f"memory not found: {inp.id}")
        return DeletePipelineResult(status="ok", message=None)
