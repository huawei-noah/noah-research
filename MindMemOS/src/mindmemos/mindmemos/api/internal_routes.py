"""Gateway-only internal Memory Data Plane routes.

These endpoints are for the commercial Console BFF and must be exposed only on
private network paths. They authenticate Gateway-issued short-lived internal
tokens instead of standalone memory-system API keys.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from ..errors import PermissionDeniedError
from ..infra.db import QdrantRecord, build_filter, match_text
from ..pipelines.memory_db import MemoryDbReader
from .deps import ensure_scopes, get_internal_request_context
from .schemas import ApiResponse, AuthContext

router = APIRouter(prefix="/internal/v1", tags=["internal-memory"])


class InternalMemoryListData(BaseModel):
    """List payload for console memory visualization."""

    items: list[dict[str, Any]]
    next_cursor: str | None = None


InternalMemoryListResponse = ApiResponse[InternalMemoryListData]
InternalMemoryDetailResponse = ApiResponse[dict[str, Any]]


@router.get("/projects/{project_id}/memories", response_model=InternalMemoryListResponse)
async def list_project_memories(
    project_id: str = Path(min_length=1),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryListResponse:
    """List project memories for the Console BFF."""

    _ensure_internal_read(ctx, project_id)
    query_filter = _query_filter(q)
    records, next_cursor = await MemoryDbReader().list_memory_records(
        ctx, filters=query_filter, limit=limit, cursor=cursor
    )
    items = [_record_to_item(record) for record in records]
    data = InternalMemoryListData(items=items, next_cursor=str(next_cursor) if next_cursor is not None else None)
    return InternalMemoryListResponse(request_id=ctx.request_id, data=data)


@router.get("/projects/{project_id}/memories/{memory_id}", response_model=InternalMemoryDetailResponse)
async def get_project_memory(
    project_id: str = Path(min_length=1),
    memory_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryDetailResponse:
    """Return one project memory for the Console BFF."""

    _ensure_internal_read(ctx, project_id)
    record = await MemoryDbReader().get_memory_record(ctx, memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return InternalMemoryDetailResponse(request_id=ctx.request_id, data=_record_to_item(record))


def _ensure_internal_read(ctx: AuthContext, project_id: str) -> None:
    ensure_scopes(ctx, ("memory:read",))
    if ctx.project_id != project_id:
        raise PermissionDeniedError("project scope mismatch", code="auth.project_scope_mismatch")


def _record_to_item(record: QdrantRecord) -> dict[str, Any]:
    item = dict(record.payload)
    item.setdefault("id", str(record.point_id))
    item.setdefault("memory_id", str(item.get("memory_id") or record.point_id))
    return item


def _query_filter(query: str | None):
    if query is None:
        return None
    text = query.strip()
    if not text:
        return None
    return build_filter(must=[match_text("content", text)])
