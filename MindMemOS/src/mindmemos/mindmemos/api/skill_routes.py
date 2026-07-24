"""Skill HTTP routes (``/v1/skills/*``).

Handlers stay thin: validate via the typed request body, delegate to
:class:`SkillService`, wrap the result in :class:`ApiResponse`.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Path, Query

from ..typing import SkillSummary
from .deps import require_scopes
from .schemas import ApiResponse, AuthContext
from .services import SkillService, get_skill_service
from .skill_schemas import (
    SkillContentData,
    SkillEvolveData,
    SkillEvolveRequest,
    SkillListData,
    SkillRegisterData,
    SkillRegisterRequest,
    SkillSyncData,
    SkillSyncRequest,
    SkillVersionsData,
)

router = APIRouter(
    prefix="/v1/skills",
    tags=["skills"],
)

SkillRegisterResponse = ApiResponse[SkillRegisterData]
SkillListResponse = ApiResponse[SkillListData]
SkillDetailResponse = ApiResponse[SkillSummary]
SkillVersionsResponse = ApiResponse[SkillVersionsData]
SkillContentResponse = ApiResponse[SkillContentData]
SkillDeleteResponse = ApiResponse[None]
SkillSyncResponse = ApiResponse[SkillSyncData]
SkillEvolveResponse = ApiResponse[SkillEvolveData]
SCOPE_MEM_WRITE = "memory:write"
SCOPE_MEM_READ = "memory:read"


@router.get("", response_model=SkillListResponse)
async def list_skills(
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillListResponse:
    data = await service.list_skills(auth)
    return SkillListResponse(code="ok", request_id=auth.request_id, data=data)


@router.post("/register", response_model=SkillRegisterResponse)
async def register_skill(
    payload: SkillRegisterRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillRegisterResponse:
    data = await service.register(auth, payload)
    return SkillRegisterResponse(code="ok", request_id=auth.request_id, data=data)


@router.post("/evolve", response_model=SkillEvolveResponse)
async def evolve_skill(
    payload: SkillEvolveRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillEvolveResponse:
    data = await service.evolve(auth, payload)
    return SkillEvolveResponse(code=data.status, request_id=auth.request_id, data=data)


@router.post("/sync", response_model=SkillSyncResponse)
async def sync_skills(
    payload: SkillSyncRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillSyncResponse:
    data = await service.sync(auth, payload)
    return SkillSyncResponse(code="ok", request_id=auth.request_id, data=data)


@router.post("/{cloud_skill_id}/get", response_model=SkillDetailResponse)
async def get_skill(
    cloud_skill_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillDetailResponse:
    data = await service.get_skill(auth, cloud_skill_id)
    return SkillDetailResponse(code="ok", request_id=auth.request_id, data=data)


@router.get("/{cloud_skill_id}/versions", response_model=SkillVersionsResponse)
async def list_versions(
    cloud_skill_id: str = Path(min_length=1),
    since: datetime | None = Query(default=None),
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillVersionsResponse:
    data = await service.versions(auth, cloud_skill_id, since=since)
    return SkillVersionsResponse(code="ok", request_id=auth.request_id, data=data)


@router.get("/{cloud_skill_id}/versions/{version_id}/content", response_model=SkillContentResponse)
async def get_version_content(
    cloud_skill_id: str = Path(min_length=1),
    version_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillContentResponse:
    data = await service.content(auth, cloud_skill_id, version_id)
    return SkillContentResponse(code="ok", request_id=auth.request_id, data=data)


@router.post("/{cloud_skill_id}/delete", response_model=SkillDeleteResponse)
async def delete_skill(
    cloud_skill_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillDeleteResponse:
    await service.delete(auth, cloud_skill_id)
    return SkillDeleteResponse(code="ok", request_id=auth.request_id, data=None)
