"""Skill version-store DTOs (design `docs/skill/design-human.md`).

These are pure business contracts for the lightweight git-like skill version
store. They describe the identity, lineage and lifecycle of a skill version plus
the per-turn ``skill_context`` reference carried by ``/v1/memory/add``.

Identity recap (design §1):

- ``content_hash`` is the "tree": SHA-256 over the whitelisted, normalized bundle
  files. It only says "is the content the same", it does not carry name/version.
- ``version_id`` is the "commit": the authoritative primary key. It is derived
  deterministically from ``(project_id, content_hash, parent_version_id)``.
- ``cloud_skill_id`` is the "repo": it groups all versions of one skill by
  lineage (parent chain), not by name.
- ``version_label`` is the "tag": display-only, may repeat, never part of
  identity.

DTOs here never import ``mindmemos.infra.db`` and never touch Qdrant; the
business <-> DB mapping lives in ``mindmemos.mappers``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SkillVersionStatus(str, Enum):
    """Lifecycle status of a skill version (design §3 state machine).

    ``observed ─┐``
    ``draft     ┴─→ evaluating ─→ published ─→ superseded``
    ``                                 └──────→ rolled_back``
    """

    OBSERVED = "observed"
    DRAFT = "draft"
    EVALUATING = "evaluating"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    ROLLED_BACK = "rolled_back"


class SkillOrigin(str, Enum):
    """Where a skill version came from (design §3)."""

    EDGE = "edge"
    CLOUD = "cloud"


class SkillUsage(str, Enum):
    """How a skill was used within the recognized turn (design §4.3 / §5.1)."""

    INJECTED = "injected"
    MODIFIED = "modified"


class SkillVersion(BaseModel):
    """Purpose: Version metadata of one skill version (the "commit").

    Used in: skill version-store repository, ``/v1/skills/register`` and
    ``/v1/skills/*`` read endpoints. ``version_id`` is the authoritative key;
    ``parent_version_id`` is the lineage link (``None`` for a root version);
    ``cloud_skill_id`` groups the whole lineage. The actual bundle text lives in
    a separate ``SkillBlob`` keyed by ``content_hash``.
    """

    version_id: str
    project_id: str
    cloud_skill_id: str
    skill_name: str
    content_hash: str
    parent_version_id: str | None = None
    version_label: str | None = None
    status: SkillVersionStatus
    origin: SkillOrigin
    created_at: datetime


class SkillBlob(BaseModel):
    """Purpose: Deduplicated bundle content keyed by ``(project_id, content_hash)``.

    Used in: skill version-store repository and ``.../versions/{id}/content``.
    ``content`` is the canonical text representation of the whitelisted bundle
    files (see ``components/skill``); identical content is stored once per
    project.
    """

    project_id: str
    content_hash: str
    content: str
    created_at: datetime


class SkillContext(BaseModel):
    """Purpose: Per-turn reference to a hit skill, carried by ``/v1/memory/add``.

    Used in: ``/v1/memory/add`` request and trace binding (design §2.1 / §5.1).
    Carries no full bundle text — only the connecting keys. ``base_version_id``
    is the version the local skill derived from and is an empty string before the
    first registration.
    """

    name: NonEmptyStr
    content_hash: NonEmptyStr
    base_version_id: str = ""
    version_label: NonEmptyStr | None = None
    usage: SkillUsage | None = None


class SkillBinding(BaseModel):
    """Purpose: Resolved per-skill binding recorded on one add trace.

    Used in: ``/v1/memory/add`` trace binding (design §2.1 / §5.1). One add can
    carry several ``SkillContext`` entries, so the trace (the ``add_record_v1``
    point) stores a list of these bindings rather than a single ``version_id``.
    ``version_id`` is ``None`` while the skill content is not yet registered: the
    trace is parked in ``skill_trace_pending_v1`` and the binding is filled in by
    rebind once ensure/register uploads the content.
    """

    name: str
    content_hash: str
    base_version_id: str = ""
    version_id: str | None = None
    version_label: str | None = None
    usage: SkillUsage | None = None


class SkillTracePending(BaseModel):
    """Purpose: A trace whose skill content is not yet registered (design §2.1).

    Used in: ``skill_trace_pending_v1``. When add carries a ``content_hash`` that
    has no matching version under ``base_version_id`` yet, the trace is parked
    here; once ensure registers the content, all same-key pending traces are
    rebound in batch. ``add_record_id`` points back at the ``add_record_v1`` point
    whose ``skill_bindings`` entry must be filled in on rebind; ``trace_id`` is the
    pending point's own unique id (so several skills missed in the same add do not
    collide).
    """

    trace_id: str
    project_id: str
    content_hash: str
    base_version_id: str = ""
    add_record_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SkillTraceSummary(BaseModel):
    """Purpose: Analytical summary of one injected add trajectory (self-evolution).

    Used in: ``skill_trace_summary_v1`` and the ``SkillEvolver`` pipeline. Stored
    1:1 with the originating ``/v1/memory/add`` trace (``add_record_id`` is the
    point id), so re-summarizing the same trace overwrites rather than
    duplicates. ``consumed_version_id`` is set once the summary has been folded
    into an evolved skill version, so a later evolve call does not re-use it;
    ``created_at`` mirrors the add's completion time and drives the add-order
    batching of multiple versions. ``score`` and ``task_id`` are copied from the
    originating add trace: ``score`` is the trajectory evaluation grade and
    ``task_id`` groups multiple rollout trajectories of the same task; both stay
    ``None`` when the add did not carry them.
    """

    summary_id: str
    project_id: str
    cloud_skill_id: str
    add_record_id: str
    skill_name: str
    summary: str
    created_at: datetime
    consumed_version_id: str | None = None
    score: float | None = None
    task_id: str | None = None


class SkillEvolveResult(BaseModel):
    """Purpose: Outcome of one ``POST /v1/skills/evolve`` call.

    Used in: the skill evolution endpoint. ``evolved`` is false when the pending
    summary count did not meet the threshold; the caller then reads
    ``pending_count`` / ``threshold`` to know how many more trajectories are
    needed. When ``evolved`` is true, ``new_version_id`` is the newest minted
    version and ``new_version_ids`` lists every version minted this call (one per
    serial batch, oldest-first).
    """

    cloud_skill_id: str
    status: Literal["ok", "queued"] = "ok"
    evolved: bool
    pending_count: int
    threshold: int
    new_version_id: str | None = None
    new_version_ids: list[str] = Field(default_factory=list)
    summarized_count: int = 0
    consumed_count: int = 0


class SkillSummary(BaseModel):
    """Purpose: Project-scoped summary of one managed cloud skill.

    Used in: ``GET /v1/skills`` and ``POST /v1/skills/{cloud_skill_id}/get``.
    ``latest_version`` is the newest metadata row for display and history
    anchoring; ``published_head`` is the version SDKs may checkout by default and
    can be ``None`` while only observed/edge versions exist.
    """

    cloud_skill_id: str
    skill_name: str
    latest_version: SkillVersion
    published_head: SkillVersion | None = None


class SkillContent(BaseModel):
    """Purpose: Full canonical bundle text for one skill version.

    Used in: ``GET .../versions/{version_id}/content``. The metadata stays next
    to the content so callers can update local registries without an extra
    lookup.
    """

    version: SkillVersion
    content: str


class SkillSyncRequestItem(BaseModel):
    """Purpose: One local skill state reported by SDK sync.

    Used in: ``POST /v1/skills/sync``. ``local_version_id`` is the version the
    edge currently has checked out; the cloud compares it with the published
    head for the same ``cloud_skill_id``.
    """

    cloud_skill_id: NonEmptyStr
    local_version_id: NonEmptyStr


class SkillSyncResult(BaseModel):
    """Purpose: Published-head diff result for one skill during sync.

    Used in: ``POST /v1/skills/sync``. ``has_update`` is true only when a
    published head exists and differs from the local version. ``gating_status``
    remains explicit so SDKs can distinguish "no published candidate yet" from
    a clean up-to-date state.
    """

    cloud_skill_id: str
    local_version_id: str
    has_update: bool
    published_head: SkillVersion | None = None
    gating_status: str
