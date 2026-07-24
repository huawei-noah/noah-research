"""Default update-memory pipeline implementation."""

from __future__ import annotations

from ...typing import (
    MemoryDbMutationPlan,
    MemoryDbUpdateCommand,
    MemoryRequestContext,
    UpdatePipelineInput,
    UpdatePipelineResult,
)
from ..base import MemoryDbPipelineMixin
from ..registry import register


@register(type="update", name="default_update")
class DefaultUpdatePipeline(MemoryDbPipelineMixin):
    """Patch one memory through the project-scoped memory DB writer."""

    async def update(self, inp: UpdatePipelineInput, context: MemoryRequestContext) -> UpdatePipelineResult:
        """Patch the content of an active memory.

        Args:
            inp: Update request with target memory id and replacement content.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            An ok result when the update was applied, otherwise an error result.
        """
        memory = await self.db_reader.get_memory(context, inp.id)
        if memory is None:
            return UpdatePipelineResult(status="error", message=f"memory not found: {inp.id}")
        if memory.status != "active":
            return UpdatePipelineResult(
                status="error",
                message=f"memory is not active (status={memory.status}): {inp.id}",
            )
        if not inp.content.strip():
            return UpdatePipelineResult(status="error", message="content is empty")

        command = MemoryDbUpdateCommand(memory_id=inp.id, content=inp.content)
        await self.db_writer.apply_mutation_plan(
            context,
            MemoryDbMutationPlan(memory_updates=[command]),
        )
        return UpdatePipelineResult(status="ok", message=None)
