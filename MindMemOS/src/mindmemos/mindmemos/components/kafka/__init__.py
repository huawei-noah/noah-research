"""Kafka task dispatch helpers."""

from __future__ import annotations

from ...typing import MemoryRequestContext


def memory_add_dispatch_key(context: MemoryRequestContext) -> str:
    """Return the Kafka key used to distribute public async add tasks.

    Keyed by ``project_id:user_id`` so tasks of the same user stay ordered while
    different users in the same project can be processed in parallel.
    """

    return f"{context.project_id}:{context.user_id}"
