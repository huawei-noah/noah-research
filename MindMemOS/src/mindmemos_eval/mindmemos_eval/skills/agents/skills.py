"""Skill staging and loading tool shared by skill benchmarks."""

from __future__ import annotations

from pathlib import Path

from .react import Tool

SKILL_FILE = "SKILL.md"


class SkillSet:
    """Expose staged skill directories through a single ``skill`` tool."""

    def __init__(self, skill_dirs: dict[str, Path]) -> None:
        self.skill_dirs = {name: Path(path) for name, path in skill_dirs.items()}

    @property
    def names(self) -> list[str]:
        return list(self.skill_dirs)

    def load(self, name: str) -> str:
        skill_dir = self.skill_dirs.get(name)
        if skill_dir is None:
            available = ", ".join(self.names) or "(none)"
            return f"Error: unknown skill '{name}'. Available skills: {available}"
        skill_md = skill_dir / SKILL_FILE
        if not skill_md.exists():
            return f"Error: skill '{name}' has no {SKILL_FILE}"
        instructions = skill_md.read_text(encoding="utf-8")
        return (
            f"Loaded skill '{name}'.\n"
            f"Skill directory (absolute path): {skill_dir}\n"
            f"Reference files live under that directory; read or run them with the read/shell tools as needed.\n\n"
            f"----- {name}/{SKILL_FILE} -----\n{instructions}"
        )

    def as_tool(self) -> Tool:
        available = ", ".join(self.names) or "(none)"
        return Tool(
            name="skill",
            description=(
                "Load an expert skill to get detailed instructions and the absolute path to reusable reference "
                f"scripts. Call this before starting the task. Available skills: {available}."
            ),
            func=self.load,
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": f"Skill to load. One of: {available}."}},
                "required": ["name"],
            },
            deliver_result_as_user=True,
        )
