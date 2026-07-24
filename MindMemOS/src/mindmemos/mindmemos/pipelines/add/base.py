from typing import Any, Protocol

from ...typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    MemoryRequestContext,
)


class AddPipeline(Protocol):
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        """Add memory content and return the completed write result.

        Args:
            inp: User content and options for the add request.
            context: Tenant, project, and actor context for hard isolation.
            add_record_id: Optional add record id to write the output back onto.

        Returns:
            The synchronous add result with created or updated memory items.
        """

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> AddPipelineAsyncResult:
        """Queue memory content for asynchronous processing.

        Args:
            inp: User content and options for the add request.
            context: Tenant, project, and actor context for hard isolation.
            add_record_id: Optional durable add record id to reuse.
            record_metadata: Optional audit metadata to persist from the worker.

        Returns:
            The queued add operation status and tracking identifier.
        """

    async def has_pending(self, context: MemoryRequestContext) -> bool:
        """Check whether this pipeline has pending asynchronous work.

        Args:
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            True when pending add records remain for the context.
        """
