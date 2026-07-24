"""Skill version-store orchestration: register + trace binding + rebind.

Layering: this sits at the orchestration layer (like ``pipelines/records.py``)
and is the only place outside ``infra.db`` that drives the skill repository for
registration and trace binding. It carries the business rules of design §2.1 and
§5.2; the repository under it stays a dumb point store.

Idempotency leans entirely on deterministic ids (``version_id`` derived from
``(project_id, content_hash, parent_version_id)``): a repeated register upserts
the same point and returns the existing version instead of forking a new
``cloud_skill_id``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from ...components.skill import (
    bundle_files_from_content,
    compute_content_hash,
    serialize_bundle,
)
from ...errors import SkillContentNotFoundError, SkillNotFoundError, SkillVersionNotFoundError
from ...infra.db import AddRecordPoint, get_database_clients
from ...logging import get_logger
from ...mappers import (
    skill_blob_from_record,
    skill_trace_pending_from_record,
    skill_version_from_record,
    skill_version_id,
    to_skill_blob_point,
    to_skill_trace_pending_point,
    to_skill_version_point,
)
from ...typing import (
    SkillBinding,
    SkillBlob,
    SkillContent,
    SkillContext,
    SkillOrigin,
    SkillSummary,
    SkillSyncRequestItem,
    SkillSyncResult,
    SkillTracePending,
    SkillVersion,
    SkillVersionStatus,
)
from ..memory_db import utcnow

logger = get_logger(__name__)


class SkillVersionStore:
    """Orchestrates skill registration and ``/v1/memory/add`` trace binding.

    Repositories are resolved lazily from the event-loop-scoped database clients on
    each call so the store survives ``reset_database_clients`` (tests / config
    reloads); they can be injected for unit tests.
    """

    def __init__(self, *, skill_repo=None, add_record_repo=None) -> None:
        self._skill_repo = skill_repo
        self._add_record_repo = add_record_repo

    @property
    def _skill(self):
        return self._skill_repo if self._skill_repo is not None else get_database_clients().skill

    @property
    def _add_record(self):
        return self._add_record_repo if self._add_record_repo is not None else get_database_clients().qdrant.add_record

    async def register(
        self,
        *,
        project_id: str,
        name: str,
        content: str,
        version_label: str | None = None,
        parent_version_id: str | None = None,
    ) -> SkillVersion:
        """Register a skill version idempotently (design §5.2).

        The dedup key is ``(project_id, content_hash, parent_version_id)``: a hit
        returns the existing version unchanged; a miss inserts a new
        ``observed`` / ``edge`` version and rebinds any pending traces waiting on
        this content. Branching off a parent inherits the parent's
        ``cloud_skill_id`` (lineage, not name); a root version mints a new one.

        Raises:
            SkillBundleError: If ``content`` yields no whitelisted bundle file.
            SkillVersionNotFoundError: If ``parent_version_id`` is given but the
                parent version does not exist in this project.
        """

        files = bundle_files_from_content(content)
        content_hash = compute_content_hash(files)
        canonical = serialize_bundle(files)
        parent = parent_version_id or None
        version_id = skill_version_id(project_id, content_hash, parent)

        existing = await self._skill.get_version(project_id, version_id)
        if existing is not None:
            version = skill_version_from_record(existing)
            await self._rebind_pending(project_id, version)
            return version

        cloud_skill_id = await self._resolve_cloud_skill_id(project_id, parent)
        now = utcnow()
        await self._skill.upsert_blob(
            to_skill_blob_point(
                SkillBlob(project_id=project_id, content_hash=content_hash, content=canonical, created_at=now)
            )
        )
        version = SkillVersion(
            version_id=version_id,
            project_id=project_id,
            cloud_skill_id=cloud_skill_id,
            skill_name=name,
            content_hash=content_hash,
            parent_version_id=parent,
            version_label=version_label,
            status=SkillVersionStatus.OBSERVED,
            origin=SkillOrigin.EDGE,
            created_at=now,
        )
        await self._skill.upsert_version(to_skill_version_point(version))
        await self._rebind_pending(project_id, version)
        return version

    async def create_evolved_version(
        self,
        *,
        project_id: str,
        parent_version_id: str,
        name: str,
        content: str,
        status: SkillVersionStatus = SkillVersionStatus.DRAFT,
        version_label: str | None = None,
    ) -> SkillVersion:
        """Mint a cloud-evolved child version off ``parent_version_id``.

        Unlike :meth:`register` (which records edge-observed versions), this writes
        an ``origin=cloud`` version with the given lifecycle ``status`` (design §3:
        evolution produces ``draft``/``cloud`` candidates). Idempotent via the
        deterministic ``version_id``; the parent must exist and supplies the
        inherited ``cloud_skill_id``.

        Raises:
            SkillBundleError: If ``content`` yields no whitelisted bundle file.
            SkillVersionNotFoundError: If ``parent_version_id`` does not exist.
        """

        files = bundle_files_from_content(content)
        content_hash = compute_content_hash(files)
        canonical = serialize_bundle(files)
        version_id = skill_version_id(project_id, content_hash, parent_version_id)

        existing = await self._skill.get_version(project_id, version_id)
        if existing is not None:
            return skill_version_from_record(existing)

        cloud_skill_id = await self._resolve_cloud_skill_id(project_id, parent_version_id)
        now = utcnow()
        await self._skill.upsert_blob(
            to_skill_blob_point(
                SkillBlob(project_id=project_id, content_hash=content_hash, content=canonical, created_at=now)
            )
        )
        version = SkillVersion(
            version_id=version_id,
            project_id=project_id,
            cloud_skill_id=cloud_skill_id,
            skill_name=name,
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            version_label=version_label,
            status=status,
            origin=SkillOrigin.CLOUD,
            created_at=now,
        )
        await self._skill.upsert_version(to_skill_version_point(version))
        return version

    async def list_skills(self, *, project_id: str) -> list[SkillSummary]:
        """Return one summary per managed cloud skill in a project (design §5.4)."""

        latest_by_skill: dict[str, SkillVersion] = {}
        cursor = None
        while True:
            records, cursor = await self._skill.list_versions(project_id, cursor=cursor)
            for record in records:
                version = skill_version_from_record(record)
                current = latest_by_skill.get(version.cloud_skill_id)
                if current is None or version.created_at > current.created_at:
                    latest_by_skill[version.cloud_skill_id] = version
            if cursor is None:
                break

        summaries: list[SkillSummary] = []
        for version in sorted(latest_by_skill.values(), key=lambda item: item.created_at, reverse=True):
            summaries.append(await self._summary_from_latest(project_id, version))
        return summaries

    async def get_skill(self, *, project_id: str, cloud_skill_id: str) -> SkillSummary:
        """Return skill metadata plus its current published head (design §5.4)."""

        latest = await self._latest_or_raise(project_id, cloud_skill_id)
        return await self._summary_from_latest(project_id, latest)

    async def versions_since(
        self,
        *,
        project_id: str,
        cloud_skill_id: str,
        since: datetime | None = None,
    ) -> list[SkillVersion]:
        """Return incremental version metadata for one cloud skill (design §5.4)."""

        await self._latest_or_raise(project_id, cloud_skill_id)
        records = await self._skill.versions_since(project_id, cloud_skill_id, since=since)
        return [skill_version_from_record(record) for record in records]

    async def get_content(self, *, project_id: str, cloud_skill_id: str, version_id: str) -> SkillContent:
        """Return the canonical bundle text for one version (design §5.4)."""

        version = await self._version_or_raise(project_id, version_id)
        if version.cloud_skill_id != cloud_skill_id:
            raise SkillVersionNotFoundError(
                f"version {version_id} not found under cloud skill {cloud_skill_id} in project {project_id}"
            )
        blob_record = await self._skill.get_blob(project_id, version.content_hash)
        if blob_record is None:
            raise SkillContentNotFoundError(f"content for version {version_id} not found in project {project_id}")
        blob = skill_blob_from_record(blob_record)
        return SkillContent(version=version, content=blob.content)

    async def delete_skill(self, *, project_id: str, cloud_skill_id: str) -> None:
        """Unmanage a cloud skill by deleting its version metadata (design §5.4)."""

        versions = await self.versions_since(project_id=project_id, cloud_skill_id=cloud_skill_id)
        await self._skill.delete_versions([version.version_id for version in versions])

    async def sync(self, *, project_id: str, items: list[SkillSyncRequestItem]) -> list[SkillSyncResult]:
        """Compare local versions with cloud published heads (design §5.3)."""

        results: list[SkillSyncResult] = []
        for item in items:
            await self._latest_or_raise(project_id, item.cloud_skill_id)
            head_record = await self._skill.published_head(project_id, item.cloud_skill_id)
            published_head = skill_version_from_record(head_record) if head_record is not None else None
            results.append(
                SkillSyncResult(
                    cloud_skill_id=item.cloud_skill_id,
                    local_version_id=item.local_version_id,
                    has_update=published_head is not None and published_head.version_id != item.local_version_id,
                    published_head=published_head,
                    gating_status=published_head.status.value if published_head is not None else "no_published_head",
                )
            )
        return results

    async def _summary_from_latest(self, project_id: str, latest: SkillVersion) -> SkillSummary:
        head_record = await self._skill.published_head(project_id, latest.cloud_skill_id)
        return SkillSummary(
            cloud_skill_id=latest.cloud_skill_id,
            skill_name=latest.skill_name,
            latest_version=latest,
            published_head=skill_version_from_record(head_record) if head_record is not None else None,
        )

    async def _latest_or_raise(self, project_id: str, cloud_skill_id: str) -> SkillVersion:
        record = await self._skill.latest_version(project_id, cloud_skill_id)
        if record is None:
            raise SkillNotFoundError(f"cloud skill {cloud_skill_id} not found in project {project_id}")
        return skill_version_from_record(record)

    async def _version_or_raise(self, project_id: str, version_id: str) -> SkillVersion:
        record = await self._skill.get_version(project_id, version_id)
        if record is None:
            raise SkillVersionNotFoundError(f"version {version_id} not found in project {project_id}")
        return skill_version_from_record(record)

    async def _resolve_cloud_skill_id(self, project_id: str, parent_version_id: str | None) -> str:
        if not parent_version_id:
            return str(uuid.uuid4())
        parent = await self._skill.get_version(project_id, parent_version_id)
        if parent is None:
            raise SkillVersionNotFoundError(f"parent version {parent_version_id} not found in project {project_id}")
        return skill_version_from_record(parent).cloud_skill_id

    async def bind_skill_context(
        self,
        *,
        project_id: str,
        add_record_id: str,
        skill_context: list[SkillContext],
    ) -> list[SkillBinding]:
        """Resolve each ``skill_context`` to a version, parking misses as pending.

        Never raises on a per-skill failure path: a skill that cannot be bound
        becomes a ``version_id=None`` binding plus a pending trace, so add is
        never blocked (design §2.1). The returned bindings are stored on the add
        trace (``add_record_v1``); pending ones are filled in later by rebind.
        """

        bindings: list[SkillBinding] = []
        for sc in skill_context:
            version_id = await self._resolve_binding(project_id, sc)
            bindings.append(
                SkillBinding(
                    name=sc.name,
                    content_hash=sc.content_hash,
                    base_version_id=sc.base_version_id,
                    version_id=version_id,
                    version_label=sc.version_label,
                    usage=sc.usage,
                )
            )
            if version_id is None:
                await self._skill.add_pending_trace(
                    to_skill_trace_pending_point(
                        SkillTracePending(
                            trace_id=str(uuid.uuid4()),
                            project_id=project_id,
                            content_hash=sc.content_hash,
                            base_version_id=sc.base_version_id,
                            add_record_id=add_record_id,
                            created_at=utcnow(),
                        )
                    )
                )
        return bindings

    async def _resolve_binding(self, project_id: str, sc: SkillContext) -> str | None:
        """Apply the trace-binding rules of design §2.1, returning a version_id or None."""

        base = sc.base_version_id
        if base:
            base_rec = await self._skill.get_version(project_id, base)
            if base_rec is not None and skill_version_from_record(base_rec).content_hash == sc.content_hash:
                # Rule 1: content unchanged relative to base -> bind base.
                return base
        # Rule 2 / brand-new: a local change derived from base (parent=base, or root
        # when base is empty). Bind the derived version if it already exists.
        parent = base or None
        candidate_id = skill_version_id(project_id, sc.content_hash, parent)
        candidate = await self._skill.get_version(project_id, candidate_id)
        return candidate_id if candidate is not None else None

    async def _rebind_pending(self, project_id: str, version: SkillVersion) -> None:
        """Rebind pending traces of the same key to a freshly inserted version (design §2.1).

        Same key = same ``content_hash`` and ``base_version_id == parent_version_id``
        (treating ``""`` and ``None`` as equal). Each matched trace's
        ``add_record`` has its ``skill_bindings`` entry filled in, then the pending
        point is deleted.
        """

        cursor = None
        base_version_id = version.parent_version_id or ""
        while True:
            records, cursor = await self._skill.scroll_pending_traces_page(
                project_id,
                version.content_hash,
                base_version_id=base_version_id,
                cursor=cursor,
            )
            if not records:
                break
            await self._apply_bindings(
                project_id,
                [skill_trace_pending_from_record(record) for record in records],
                version,
            )
            if cursor is None:
                break

    async def _apply_bindings(
        self,
        project_id: str,
        traces: list[SkillTracePending],
        version: SkillVersion,
    ) -> None:
        trace_ids = [trace.trace_id for trace in traces]
        traces_by_add_record: dict[str, list[SkillTracePending]] = {}
        for trace in traces:
            if trace.add_record_id:
                traces_by_add_record.setdefault(trace.add_record_id, []).append(trace)
        if not traces_by_add_record:
            await self._skill.delete_pending_traces(trace_ids)
            return

        records = await self._add_record.retrieve(project_id, list(traces_by_add_record))
        records_by_id = {record.point_id: record for record in records}
        missing_ids = set(traces_by_add_record) - set(records_by_id)
        for add_record_id in sorted(missing_ids):
            logger.warning(
                "skill rebind target add_record missing",
                add_record_id=add_record_id,
                version_id=version.version_id,
            )

        changed_points: list[AddRecordPoint] = []
        rebound_trace_ids: list[str] = []
        for add_record_id in traces_by_add_record:
            record = records_by_id.get(add_record_id)
            if record is None:
                continue
            if self._patch_skill_bindings(record.payload, version):
                changed_points.append(AddRecordPoint(add_record_id=add_record_id, payload=record.payload))
            rebound_trace_ids.extend(trace.trace_id for trace in traces_by_add_record[add_record_id])

        if changed_points:
            await self._add_record.upsert(changed_points)
        await self._skill.delete_pending_traces(rebound_trace_ids)

    @staticmethod
    def _patch_skill_bindings(payload: dict, version: SkillVersion) -> bool:
        bindings = payload.get("skill_bindings") or []
        changed = False
        for binding in bindings:
            if (
                binding.get("version_id") is None
                and binding.get("content_hash") == version.content_hash
                and (binding.get("base_version_id") or None) == version.parent_version_id
            ):
                binding["version_id"] = version.version_id
                changed = True
        return changed


_store: SkillVersionStore | None = None


def get_skill_version_store() -> SkillVersionStore:
    """Process-global :class:`SkillVersionStore` singleton."""

    global _store
    if _store is None:
        _store = SkillVersionStore()
    return _store
