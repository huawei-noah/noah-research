"""Skill HTTP business logic (design §5.2–§5.4).

Thin adapter between the ``/v1/skills/*`` routes and the
:class:`~mindmemos.pipelines.skill.SkillVersionStore` orchestration. It assembles
nothing of its own beyond translating the security-only ``AuthContext`` into a
``project_id`` and mapping domain skill errors to HTTP errors; the version-store
owns dedup, lineage and rebind.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ...errors import (
    BadRequestError,
    ResourceNotFoundError,
    SkillBundleError,
    SkillContentNotFoundError,
    SkillNotFoundError,
    SkillVersionNotFoundError,
)
from ...infra.kafka import get_producer
from ...logging import get_logger, traced
from ...pipelines import create_pipeline
from ...pipelines.skill import (
    SKILL_EVOLVE_TOPIC,
    SkillEvolvePipeline,
    SkillVersionStore,
    get_skill_evolver,
    get_skill_version_store,
)
from ...typing import SkillEvolveResult
from ..deps import annotate_request_trace
from ..schemas import AuthContext
from ..skill_schemas import (
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

logger = get_logger(__name__)


class SkillService:
    """Stateless facade routing skill endpoints to the version store."""

    def __init__(self, *, store: SkillVersionStore | None = None, evolver: SkillEvolvePipeline | None = None) -> None:
        self._store = store or get_skill_version_store()
        # Resolved lazily from config on first evolve(), so building the service
        # for the read/write endpoints never requires the evolve pipeline config.
        self._evolver = evolver

    @property
    def evolver(self) -> SkillEvolvePipeline:
        if self._evolver is None:
            self._evolver = get_skill_evolver()
        return self._evolver

    @traced("skill_service.register")
    async def register(self, auth: AuthContext, request: SkillRegisterRequest) -> SkillRegisterData:
        """Register a skill version idempotently (design §5.2)."""

        annotate_request_trace(auth)
        try:
            version = await self._store.register(
                project_id=auth.project_id,
                name=request.name,
                content=request.content,
                version_label=request.version_label,
                parent_version_id=request.parent_version_id,
            )
        except SkillBundleError as exc:
            raise BadRequestError(str(exc), code="skill.invalid_bundle") from exc
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.parent_not_found") from exc
        return SkillRegisterData(
            cloud_skill_id=version.cloud_skill_id,
            version_id=version.version_id,
            version_label=version.version_label,
            content_hash=version.content_hash,
            status=version.status,
        )

    @traced("skill_service.list")
    async def list_skills(self, auth: AuthContext) -> SkillListData:
        """List project-managed skills (design §5.4)."""

        annotate_request_trace(auth)
        skills = await self._store.list_skills(project_id=auth.project_id)
        return SkillListData(skills=skills)

    @traced("skill_service.get")
    async def get_skill(self, auth: AuthContext, cloud_skill_id: str):
        """Return skill metadata plus published head (design §5.4)."""

        annotate_request_trace(auth)
        try:
            return await self._store.get_skill(project_id=auth.project_id, cloud_skill_id=cloud_skill_id)
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc

    @traced("skill_service.versions")
    async def versions(
        self,
        auth: AuthContext,
        cloud_skill_id: str,
        since: datetime | None = None,
    ) -> SkillVersionsData:
        """Return incremental version metadata without content (design §5.4)."""

        annotate_request_trace(auth)
        try:
            versions = await self._store.versions_since(
                project_id=auth.project_id,
                cloud_skill_id=cloud_skill_id,
                since=since,
            )
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc
        return SkillVersionsData(versions=versions)

    @traced("skill_service.content")
    async def content(self, auth: AuthContext, cloud_skill_id: str, version_id: str) -> SkillContentData:
        """Return canonical bundle content for one version (design §5.4)."""

        annotate_request_trace(auth)
        try:
            content = await self._store.get_content(
                project_id=auth.project_id,
                cloud_skill_id=cloud_skill_id,
                version_id=version_id,
            )
        except SkillVersionNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.version_not_found") from exc
        except SkillContentNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.content_not_found") from exc
        return SkillContentData(version=content.version, content=content.content)

    @traced("skill_service.delete")
    async def delete(self, auth: AuthContext, cloud_skill_id: str) -> None:
        """Unmanage a cloud skill (design §5.4)."""

        annotate_request_trace(auth)
        try:
            await self._store.delete_skill(project_id=auth.project_id, cloud_skill_id=cloud_skill_id)
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc

    @traced("skill_service.evolve")
    async def evolve(self, auth: AuthContext, request: SkillEvolveRequest) -> SkillEvolveData:
        """Run one skill self-evolution pass for a cloud skill.

        Returns the result DTO: when the pending trajectory count is below the
        threshold ``evolved`` is false and the caller learns how many more are
        needed; otherwise it carries the freshly minted version id(s).
        """
        annotate_request_trace(auth)

        if request.mode == "async":
            await get_producer().send(
                SKILL_EVOLVE_TOPIC,
                value={
                    "request_id": auth.request_id,
                    "account_id": auth.account_id,
                    "project_id": auth.project_id,
                    "cloud_skill_id": request.cloud_skill_id,
                    "submitted_at": datetime.now(UTC).isoformat(),
                },
                dispatch_key=f"{auth.project_id}:{request.cloud_skill_id}",
            )
            return SkillEvolveResult(
                cloud_skill_id=request.cloud_skill_id,
                status="queued",
                evolved=False,
                pending_count=0,
                threshold=0,
            )
        try:
            return await self.evolver.evolve(
                project_id=auth.project_id,
                cloud_skill_id=request.cloud_skill_id,
            )
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc

    @traced("skill_service.sync")
    async def sync(self, auth: AuthContext, request: SkillSyncRequest) -> SkillSyncData:
        """Compare local versions with published cloud heads (design §5.3)."""

        annotate_request_trace(auth)
        try:
            results = await self._store.sync(project_id=auth.project_id, items=request.root)
        except SkillNotFoundError as exc:
            raise ResourceNotFoundError(str(exc), code="skill.not_found") from exc
        return SkillSyncData(results=results)


_service: SkillService | None = None
_service_key: str | None = None


def get_skill_service() -> SkillService:
    """Process-global skill service singleton, used as a FastAPI dependency.

    The skill-evolve algorithm version is selected by
    ``get_config().pipelines["skill_evolve"]``; the service is rebuilt if that
    config value changes (mirrors :func:`get_memory_service`).
    """

    global _service, _service_key
    from ...config import get_config

    evolve_name = get_config().pipelines["skill_evolve"]
    if _service is None or _service_key != evolve_name:
        _service = SkillService(evolver=create_pipeline(type="skill_evolve", name=evolve_name))
        _service_key = evolve_name
    return _service
