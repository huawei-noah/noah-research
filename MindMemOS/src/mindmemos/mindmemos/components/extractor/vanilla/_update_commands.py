"""Update-command builders for vanilla add actions.

Each function constructs a ``MemoryDbUpdateCommand`` DTO with the
appropriate metadata patch and reason.  These are pure factories —
no side effects, no DB calls.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ....typing import MemoryDbUpdateCommand, MemoryRequestContext


def build_reinforcement_command(
    memory_id: str,
    context: MemoryRequestContext,
    now: datetime,
    *,
    consistency: str = "fast",
) -> MemoryDbUpdateCommand:
    """Build a reinforcement metadata patch command for a duplicate memory."""
    return MemoryDbUpdateCommand(
        memory_id=memory_id,
        metadata_patch={
            "last_reinforced_request_id": context.request_id,
            "last_reinforced_at": now.isoformat(),
        },
        reinforcement_count_delta=1,
        reason="add_duplicate",
        consistency=consistency,
        dedup_metadata_key="last_reinforced_request_id",
    )


def build_update_command(
    memory_id: str,
    content: str,
    context: MemoryRequestContext,
    now: datetime,
    *,
    dense_vector: list[float] | None = None,
    sparse_vectors: dict[str, Any] | None = None,
    metadata_refresh: dict[str, Any] | None = None,
    consistency: str = "fast",
) -> MemoryDbUpdateCommand:
    """Build an update command that replaces memory content and vectors."""
    metadata_patch: dict[str, Any] = {
        "last_updated_request_id": context.request_id,
        "last_updated_at": now.isoformat(),
    }
    if metadata_refresh:
        metadata_patch.update(metadata_refresh)
    return MemoryDbUpdateCommand(
        memory_id=memory_id,
        content=content,
        metadata_patch=metadata_patch,
        reason="add_update",
        consistency=consistency,
        dense_vector=dense_vector,
        sparse_vectors=sparse_vectors,
        graph_content_sync=True,
    )


def build_merge_archive_commands(
    related_memory_ids: list[str],
    context: MemoryRequestContext,
    now: datetime,
    *,
    consistency: str = "fast",
) -> list[MemoryDbUpdateCommand]:
    """Archive all old memories involved in a merge."""
    return [
        MemoryDbUpdateCommand(
            memory_id=mid,
            status="archived",
            metadata_patch={
                "archived_reason": "merged",
                "merged_request_id": context.request_id,
                "merged_at": now.isoformat(),
            },
            reason="add_merge_archive",
            consistency=consistency,
        )
        for mid in related_memory_ids
    ]
