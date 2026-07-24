"""Explicit feedback handling.

Explicit feedback relies on the caller to provide the feedback text, full
conversation context, and memories recalled in that context.
"""

from __future__ import annotations

from ...components.feedback import DefaultExplicitFeedbackPlanner, ExplicitFeedbackPlanner
from ...typing import (
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    MemoryRequestContext,
    MemorySearchItem,
    SearchPipelineInput,
)
from ..registry import create_pipeline
from ..search import SearchPipeline
from .executor import FeedbackActionExecutor


class ExplicitFeedbackHandler:
    """Handle user-provided feedback without scanning operation records."""

    def __init__(
        self,
        *,
        planner: ExplicitFeedbackPlanner | None = None,
        executor: FeedbackActionExecutor | None = None,
        search_pipeline: SearchPipeline | None = None,
    ) -> None:
        self._planner = planner or DefaultExplicitFeedbackPlanner()
        self._executor = executor or FeedbackActionExecutor()
        self._search = search_pipeline

    async def run(self, inp: FeedbackPipelineInput, context: MemoryRequestContext) -> FeedbackPipelineResult:
        """Validate explicit feedback inputs, plan actions, and execute them."""

        if not inp.feedback:
            return FeedbackPipelineResult(status="error", message="explicit feedback requires feedback text")
        if not inp.messages:
            return FeedbackPipelineResult(status="error", message="explicit feedback requires messages context")

        decision = await self._planner.decide_memory_search(inp)
        if decision.need_search and decision.query:
            search_result = await self._search_pipeline.search(
                SearchPipelineInput(query=decision.query, search_pipeline="vanilla"), context
            )
            inp = inp.model_copy(
                update={"recalled_memories": _merge_memories(inp.recalled_memories, search_result.memories)}
            )

        planned_actions = await self._planner.plan(inp)
        actions = await self._executor.execute(planned_actions, context)
        return FeedbackPipelineResult(status="ok", message=None, actions=actions)

    @property
    def _search_pipeline(self) -> SearchPipeline:
        if self._search is None:
            self._search = create_pipeline(type="search", name="search_pipeline")
        return self._search


def _merge_memories(existing: list[MemorySearchItem], supplemental: list[MemorySearchItem]) -> list[MemorySearchItem]:
    by_id = {memory.id: memory for memory in existing}
    for memory in supplemental:
        by_id.setdefault(memory.id, memory)
    return list(by_id.values())
