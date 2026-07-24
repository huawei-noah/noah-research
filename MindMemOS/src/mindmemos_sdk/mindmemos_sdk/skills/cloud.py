"""Typed client for the ``/v1/skills/*`` cloud API."""

from __future__ import annotations

from urllib.parse import quote

from ..transport import HttpTransport
from .models import (
    SkillContentData,
    SkillEvolveData,
    SkillEvolveMode,
    SkillListData,
    SkillRegisterData,
    SkillSummary,
    SkillSyncData,
    SkillSyncRequestItem,
    SkillVersion,
    SkillVersionsData,
)


def _path_part(value: str) -> str:
    return quote(value, safe="")


class SkillCloudClient:
    """Skill API resource client over the shared SDK ``HttpTransport``."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def register(
        self,
        *,
        name: str,
        content: str,
        version_label: str | None = None,
        parent_version_id: str | None = None,
    ) -> SkillRegisterData:
        """Register a local skill bundle with the cloud version store."""

        body = {"name": name, "content": content}
        if version_label is not None:
            body["version_label"] = version_label
        if parent_version_id is not None:
            body["parent_version_id"] = parent_version_id
        envelope = self._transport.post_envelope("/v1/skills/register", json=body)
        return SkillRegisterData.model_validate(envelope.data or {})

    def list_skills(self) -> list[SkillSummary]:
        """List cloud-managed skills in the current project."""

        envelope = self._transport.get_envelope("/v1/skills")
        return SkillListData.model_validate(envelope.data or {}).skills

    def get_skill(self, cloud_skill_id: str) -> SkillSummary:
        """Return metadata for one cloud-managed skill."""

        envelope = self._transport.post_envelope(f"/v1/skills/{_path_part(cloud_skill_id)}/get", json=None)
        return SkillSummary.model_validate(envelope.data or {})

    def versions_since(
        self,
        cloud_skill_id: str,
        *,
        since: str | None = None,
    ) -> list[SkillVersion]:
        """Return incremental version metadata for one cloud skill."""

        params = {"since": since} if since else None
        envelope = self._transport.get_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/versions",
            params=params,
        )
        return SkillVersionsData.model_validate(envelope.data or {}).versions

    def get_content(
        self,
        cloud_skill_id: str,
        version_id: str,
    ) -> SkillContentData:
        """Download the canonical bundle text for one skill version."""

        envelope = self._transport.get_envelope(
            f"/v1/skills/{_path_part(cloud_skill_id)}/versions/{_path_part(version_id)}/content",
        )
        return SkillContentData.model_validate(envelope.data or {})

    def evolve(self, cloud_skill_id: str, *, mode: SkillEvolveMode = "sync") -> SkillEvolveData:
        """Trigger one skill self-evolution pass for ``cloud_skill_id``.

        The server aggregates the injected ``/v1/memory/add`` trajectories bound to
        this skill and, once enough accumulate, mints one or more evolved versions.
        ``evolved`` is false when the pending count is still below the threshold.
        """

        envelope = self._transport.post_envelope(
            "/v1/skills/evolve",
            json={"cloud_skill_id": cloud_skill_id, "mode": mode},
        )
        data = SkillEvolveData.model_validate(envelope.data or {})
        return data.model_copy(update={"status": envelope.code or data.status})

    def sync(
        self,
        items: list[SkillSyncRequestItem | dict[str, str]],
    ) -> SkillSyncData:
        """Check whether reported local skills have newer published heads."""

        body = [item.model_dump() if isinstance(item, SkillSyncRequestItem) else item for item in items]
        envelope = self._transport.post_envelope("/v1/skills/sync", json=body)
        return SkillSyncData.model_validate(envelope.data or {})

    def delete_skill(self, cloud_skill_id: str) -> None:
        """Remove the cloud management relation for one skill."""

        self._transport.post_envelope(f"/v1/skills/{_path_part(cloud_skill_id)}/delete", json=None)
