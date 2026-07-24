from dataclasses import dataclass
from typing import Protocol

from ...typing import (
    MemoryRequestContext,
    MemorySearchItem,
    SearchPipelineInput,
    SearchPipelineResult,
)


class SearchPipeline(Protocol):
    async def search(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> SearchPipelineResult:
        """Run the configured memory search workflow.

        Args:
            inp: Search query, filters, and strategy options.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            Ranked memory search results.
        """


@dataclass(frozen=True)
class SearchEngineOptions:
    """Internal engine overrides supplied by wrappers such as agentic search."""

    num_hops: int | None = None
    recall_top_k: int | None = None
    result_top_n: int | None = None
    use_reranker: bool | None = None


class SearchEngine(Protocol):
    """One single-pass retrieval strategy."""

    name: str

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Retrieve candidate memories before final pipeline post-processing.

        Args:
            inp: Search query, filters, and strategy options.
            context: Tenant, project, and actor context for hard isolation.
            options: Optional wrapper-provided engine overrides.

        Returns:
            Candidate memory items in engine ranking order.
        """
