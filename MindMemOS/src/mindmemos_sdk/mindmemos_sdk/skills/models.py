"""SDK-side skill management models.

The models in this module mirror the public skill API and local state contract,
but they do not import server DTOs. Response-oriented models ignore extra fields
so older SDKs tolerate additive server changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SkillEvolveMode = Literal["sync", "async"]


class HashState(str, Enum):
    """Local upload/cache state for one managed skill content hash."""

    UNKNOWN = "unknown"
    PENDING_UPLOAD = "pending_upload"
    CONFIRMED = "confirmed"


class SkillVersionStatus(str, Enum):
    """Cloud lifecycle status for one skill version."""

    OBSERVED = "observed"
    DRAFT = "draft"
    EVALUATING = "evaluating"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    ROLLED_BACK = "rolled_back"


class SkillOrigin(str, Enum):
    """Origin of one skill version."""

    EDGE = "edge"
    CLOUD = "cloud"


class SkillUsage(str, Enum):
    """How a skill was used in one add trace."""

    INJECTED = "injected"
    MODIFIED = "modified"


class SkillContext(BaseModel):
    """Lightweight skill reference sent with one memory add request."""

    name: str
    content_hash: str
    base_version_id: str
    version_label: str | None = None
    usage: SkillUsage | str | None = None


class SkillRecord(BaseModel):
    """One local skill managed by the SDK registry."""

    model_config = ConfigDict(extra="ignore")

    skill_id: str = ""
    alias: str | None = None
    path: str
    skill_name: str
    cloud_skill_id: str | None = None
    base_version_id: str = ""
    content_hash: str = ""
    hash_state: HashState = HashState.UNKNOWN
    version_label: str | None = None
    registered_at: str | None = None
    updated_at: str | None = None


class SkillPendingUpload(BaseModel):
    """One local outbox entry for an unconfirmed skill content snapshot."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    skill_id: str
    path: str
    skill_name: str
    cloud_skill_id: str | None = None
    parent_version_id: str = ""
    content_hash: str
    version_label: str | None = None
    content_cache_key: str
    attempts: int = 0
    next_retry_at: str | None = None
    last_error: str | None = None
    created_at: str
    updated_at: str


class SkillPendingUploadsFile(BaseModel):
    """Top-level ``skill_pending_uploads.json`` schema."""

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    uploads: dict[str, SkillPendingUpload] = Field(default_factory=dict)


class SkillFlushResult(BaseModel):
    """Outcome of one pending upload retry."""

    skill_id: str
    content_hash: str
    parent_version_id: str
    uploaded: bool
    version_id: str | None = None
    registry_advanced: bool = False
    error: str | None = None


class SkillVersion(BaseModel):
    """Cloud version metadata returned by ``/v1/skills/*``."""

    model_config = ConfigDict(extra="ignore")

    version_id: str
    project_id: str | None = None
    cloud_skill_id: str
    skill_name: str
    content_hash: str
    parent_version_id: str | None = None
    version_label: str | None = None
    status: SkillVersionStatus
    origin: SkillOrigin
    created_at: str


class LocalSkillVersion(BaseModel):
    """Version metadata persisted in ``skill_history.json``."""

    model_config = ConfigDict(extra="ignore")

    version_id: str
    parent_version_id: str | None = None
    version_label: str | None = None
    status: SkillVersionStatus
    origin: SkillOrigin
    content_hash: str
    created_at: str

    @classmethod
    def from_cloud(cls, version: SkillVersion) -> LocalSkillVersion:
        """Create a local history entry from cloud metadata."""

        return cls(
            version_id=version.version_id,
            parent_version_id=version.parent_version_id,
            version_label=version.version_label,
            status=version.status,
            origin=version.origin,
            content_hash=version.content_hash,
            created_at=version.created_at,
        )


class SkillSummary(BaseModel):
    """Project-scoped cloud skill summary."""

    model_config = ConfigDict(extra="ignore")

    cloud_skill_id: str
    skill_name: str
    latest_version: SkillVersion
    published_head: SkillVersion | None = None


class SkillListData(BaseModel):
    """Response data returned by ``GET /v1/skills``."""

    model_config = ConfigDict(extra="ignore")

    skills: list[SkillSummary] = Field(default_factory=list)


class SkillRegisterData(BaseModel):
    """Response data returned by ``POST /v1/skills/register``."""

    model_config = ConfigDict(extra="ignore")

    cloud_skill_id: str
    version_id: str
    version_label: str | None = None
    content_hash: str
    status: SkillVersionStatus


class SkillVersionsData(BaseModel):
    """Response data returned by ``GET .../versions``."""

    model_config = ConfigDict(extra="ignore")

    versions: list[SkillVersion] = Field(default_factory=list)


class SkillContentData(BaseModel):
    """Response data returned by ``GET .../versions/{version_id}/content``."""

    model_config = ConfigDict(extra="ignore")

    version: SkillVersion
    content: str


class SkillEvolveData(BaseModel):
    """Response data returned by ``POST /v1/skills/evolve``.

    Mirrors :class:`mindmemos.typing.skill.SkillEvolveResult`. ``evolved`` is
    false when the pending trajectory count is below ``threshold``; otherwise
    ``new_version_id`` is the newest minted version and ``new_version_ids`` lists
    every version minted by the call (one per serial batch, oldest-first).
    """

    model_config = ConfigDict(extra="ignore")

    cloud_skill_id: str
    status: str = "ok"
    evolved: bool
    pending_count: int
    threshold: int
    new_version_id: str | None = None
    new_version_ids: list[str] = Field(default_factory=list)
    summarized_count: int = 0
    consumed_count: int = 0


class SkillSyncRequestItem(BaseModel):
    """One local skill state sent to ``POST /v1/skills/sync``."""

    cloud_skill_id: str
    local_version_id: str


class SkillSyncResult(BaseModel):
    """Published-head diff result returned by ``POST /v1/skills/sync``."""

    model_config = ConfigDict(extra="ignore")

    cloud_skill_id: str
    local_version_id: str
    has_update: bool
    published_head: SkillVersion | None = None
    gating_status: str


class SkillSyncData(BaseModel):
    """Response data returned by ``POST /v1/skills/sync``."""

    model_config = ConfigDict(extra="ignore")

    results: list[SkillSyncResult] = Field(default_factory=list)


class SkillHistoryEntry(BaseModel):
    """One skill bucket inside ``skill_history.json``."""

    model_config = ConfigDict(extra="ignore")

    skill_name: str
    versions: list[LocalSkillVersion] = Field(default_factory=list)
    last_pulled_at: str | None = None


class SkillHistoryFile(BaseModel):
    """Top-level ``skill_history.json`` schema."""

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    skills: dict[str, SkillHistoryEntry] = Field(default_factory=dict)


class SkillCheckoutPlan(BaseModel):
    """Planned local replacement for one managed skill."""

    skill_id: str
    path: str
    from_version_id: str
    to_version_id: str
    from_content_hash: str
    to_content_hash: str
    files: list[str] = Field(default_factory=list)
    backup_path: str | None = None


class SkillDiffResult(BaseModel):
    """Text diff between two cached skill versions."""

    skill_id: str
    from_version_id: str
    to_version_id: str
    diff: str


class SkillUpdateResult(BaseModel):
    """Outcome of checking or applying one skill update."""

    skill_id: str
    skill_name: str
    had_update: bool
    plan: SkillCheckoutPlan | None = None
    record: SkillRecord | None = None
    message: str = ""


SkillUpdatePlan = SkillCheckoutPlan
RollbackPlan = SkillCheckoutPlan
