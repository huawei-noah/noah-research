from typing import Protocol

from ...typing import SkillEvolveResult

SKILL_EVOLVE_TOPIC = "skill.evolve"


class SkillEvolvePipeline(Protocol):
    """Contract for a skill self-evolution algorithm version.

    Implementations are registered under ``type="skill_evolve"`` (see
    ``pipelines/registry``) so the active algorithm can be selected by config,
    mirroring the ``add`` / ``search`` pipeline families.
    """

    async def evolve(self, *, project_id: str, cloud_skill_id: str) -> SkillEvolveResult:
        """Run one evolution pass for a cloud skill.

        Args:
            project_id: Project that owns the skill.
            cloud_skill_id: Cloud skill identifier to evolve.

        Returns:
            The skill evolution result.
        """
