"""HTTP schemas for the ``/v1/skills/*`` endpoints (design §5.2–§5.4).

Kept separate from ``api.schemas`` so the memory and skill surfaces evolve
independently while sharing the same :class:`ApiResponse` envelope.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from ..typing import (
    SkillEvolveResult,
    SkillSummary,
    SkillSyncRequestItem,
    SkillSyncResult,
    SkillVersion,
    SkillVersionStatus,
)
from .schemas import NonEmptyStr


class SkillRegisterRequest(BaseModel):
    """HTTP body for ``POST /v1/skills/register`` (design §5.2).

    ``content`` is the canonical bundle text (see ``components/skill``); a bare
    ``SKILL.md`` body is also accepted and treated as the single whitelisted
    file. ``parent_version_id`` branches off an existing version (lineage);
    omitting it registers a root version.
    """

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    content: NonEmptyStr
    version_label: NonEmptyStr | None = None
    parent_version_id: NonEmptyStr | None = None


class SkillRegisterData(BaseModel):
    """``data`` payload returned by ``POST /v1/skills/register`` (design §5.2)."""

    cloud_skill_id: str
    version_id: str
    version_label: str | None = None
    content_hash: str
    status: SkillVersionStatus


class SkillListData(BaseModel):
    """``data`` payload returned by ``GET /v1/skills`` (design §5.4)."""

    skills: list[SkillSummary]


class SkillVersionsData(BaseModel):
    """``data`` payload returned by ``GET .../versions`` (design §5.4)."""

    versions: list[SkillVersion]


class SkillContentData(BaseModel):
    """``data`` payload returned by ``GET .../content`` (design §5.4)."""

    version: SkillVersion
    content: str


class SkillSyncRequest(RootModel[list[SkillSyncRequestItem]]):
    """HTTP body for ``POST /v1/skills/sync`` (design §5.3).

    The wire contract is a top-level JSON array, matching the SDK-facing design:
    ``[{ "cloud_skill_id": "...", "local_version_id": "..." }]``.
    """

    root: list[SkillSyncRequestItem] = Field(min_length=1)


class SkillSyncData(BaseModel):
    """``data`` payload returned by ``POST /v1/skills/sync`` (design §5.3)."""

    results: list[SkillSyncResult]


class SkillEvolveRequest(BaseModel):
    """HTTP body for ``POST /v1/skills/evolve``.

    ``cloud_skill_id`` selects the injected add traces that drive the
    self-evolution pipeline. ``mode`` mirrors memory add: sync runs inline,
    async queues Kafka work. ``project_id`` is resolved from the bearer
    ``api_key`` (same as the memory endpoints), never taken from the body.
    """

    model_config = ConfigDict(extra="forbid")

    cloud_skill_id: NonEmptyStr
    mode: Literal["sync", "async"] = "sync"


# The evolve endpoint returns the full result DTO as its ``data`` payload.
SkillEvolveData = SkillEvolveResult
