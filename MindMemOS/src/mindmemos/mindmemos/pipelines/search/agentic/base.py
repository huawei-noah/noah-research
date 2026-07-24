"""Shared primitives for the agentic search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ....components.extractor.schema import parse_json_object
from ....components.memory_modeling.schema import TemporalEntity
from ....llm import LLMClient
from ....typing import MemoryRequestContext, SearchFilter


async def ask_json(llm: LLMClient, task: str, prompt: str) -> Any:
    """Send a single-turn user prompt and parse the response as JSON."""

    resp = await llm.chat(
        task=task,
        messages=[{"role": "user", "content": prompt}],
        format_parser=parse_json_object,
    )
    return resp.parsed


@dataclass(slots=True)
class AgenticQuery:
    """One query scheduled for an agentic search round."""

    query: str
    time_window: tuple[str, str] | None
    num_hops: int
    tool_name: str | None = None
    allow_time_extraction: bool = True


@dataclass(slots=True)
class SearchToolRequest:
    """Input passed from the agentic loop to a configured search tool."""

    query: str
    original_query: str
    time_window: tuple[str, str] | None
    num_hops: int
    context: MemoryRequestContext
    allow_time_extraction: bool = True
    filters: dict[str, Any] | None = None
    search_filter: SearchFilter | None = None
    entity_search_filter: SearchFilter | None = None


@dataclass(slots=True)
class SearchToolResult:
    """Search tool output consumed by the agentic loop."""

    entities: list[TemporalEntity] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class SearchTool(Protocol):
    """Search tool interface consumed by the agentic loop."""

    name: str

    async def search(self, request: SearchToolRequest) -> SearchToolResult:
        """Execute one agentic search tool request.

        Args:
            request: Query, filters, context, and retrieval options.

        Returns:
            Candidate memories and entities returned by the tool.
        """


class SearchToolRouter(Protocol):
    """Select a search tool for one scheduled query."""

    def select(self, requested_tool: str | None = None) -> SearchTool:
        """Select the tool that should execute a scheduled query.

        Args:
            requested_tool: Optional planner-requested tool name.

        Returns:
            The selected search tool.
        """


class AgenticPlanner(Protocol):
    """Generate follow-up query plans for the agentic loop."""

    async def generate_next_queries(
        self,
        *,
        original_query: str,
        all_entities: list[TemporalEntity],
        missing_info: list[str],
        query_history: list[str],
    ) -> list[dict[str, Any]]:
        """Generate follow-up search queries for unresolved information needs.

        Args:
            original_query: User query that started the loop.
            all_entities: Entities retrieved so far.
            missing_info: Information gaps reported by the sufficiency evaluator.
            query_history: Queries already attempted.

        Returns:
            Serialized query plans for the next loop iteration.
        """


class SufficiencyEvaluator(Protocol):
    """Evaluate whether retrieved entities are enough and optionally filter them."""

    async def evaluate_sufficiency(
        self,
        *,
        user_query: str,
        retrieved_entities: list[TemporalEntity],
    ) -> tuple[bool, str, list[str]]:
        """Decide whether retrieved entities sufficiently answer the query.

        Args:
            user_query: Original user query.
            retrieved_entities: Entities retrieved so far.

        Returns:
            A tuple of sufficiency flag, reason, and missing information items.
        """

    async def filter_entities_by_relevance(
        self,
        entities: list[TemporalEntity],
        query: str,
    ) -> tuple[list[TemporalEntity], list[TemporalEntity]]:
        """Split entities into relevant and irrelevant groups for a query.

        Args:
            entities: Candidate entities to evaluate.
            query: Query used for relevance filtering.

        Returns:
            Relevant entities followed by filtered-out entities.
        """


AgenticToolRequest = SearchToolRequest
AgenticToolResult = SearchToolResult
AgenticTool = SearchTool
AgenticToolRouter = SearchToolRouter
