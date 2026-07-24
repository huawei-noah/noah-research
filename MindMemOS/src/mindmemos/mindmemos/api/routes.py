"""Memory HTTP routes.

Each handler depends on :func:`get_request_context` (auth + context assembly)
and :func:`get_memory_service` (business logic). Handlers stay thin: validate
input via the typed request body, delegate to the service, wrap the result in
:class:`ApiResponse`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .deps import require_scopes
from .mappers import (
    to_add_api_response,
    to_memory_list_api_response,
    to_status_api_response,
)
from .schemas import (
    AddData,
    AddRequest,
    ApiResponse,
    AuthContext,
    DeleteRequest,
    DreamingRequest,
    FeedbackRequest,
    GetRequest,
    MemoryListData,
    SearchRequest,
    UpdateRequest,
)
from .services import MemoryService, get_memory_service

router = APIRouter(
    prefix="/v1/memory",
    tags=["memory"],
)


AddResponse = ApiResponse[AddData]
GetResponse = ApiResponse[MemoryListData]
SearchResponse = ApiResponse[MemoryListData]
DeleteResponse = ApiResponse[None]
UpdateResponse = ApiResponse[None]
FeedbackResponse = ApiResponse[None]
DreamingResponse = ApiResponse[None]
SCOPE_MEM_WRITE = "memory:write"
SCOPE_MEM_READ = "memory:read"


@router.post("/add", response_model=AddResponse)
async def add_memory(
    payload: AddRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> AddResponse:
    result = await service.add(auth, payload)
    return to_add_api_response(result, auth.request_id)


@router.post("/get", response_model=GetResponse)
async def get_memory(
    payload: GetRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: MemoryService = Depends(get_memory_service),
) -> GetResponse:
    result = await service.get(auth, payload)
    return to_memory_list_api_response(result, auth.request_id)


@router.post("/delete", response_model=DeleteResponse)
async def delete_memory(
    payload: DeleteRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> DeleteResponse:
    result = await service.delete(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.post("/update", response_model=UpdateResponse)
async def update_memory(
    payload: UpdateRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> UpdateResponse:
    result = await service.update(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.post("/search", response_model=SearchResponse)
async def search_memory(
    payload: SearchRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: MemoryService = Depends(get_memory_service),
) -> SearchResponse:
    result = await service.search(auth, payload)
    return to_memory_list_api_response(result, auth.request_id)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_memory(
    payload: FeedbackRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> FeedbackResponse:
    result = await service.feedback(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.post("/dreaming", response_model=DreamingResponse)
async def dreaming_memory(
    payload: DreamingRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> DreamingResponse:
    result = await service.dream(auth, payload)
    return to_status_api_response(result, auth.request_id)
