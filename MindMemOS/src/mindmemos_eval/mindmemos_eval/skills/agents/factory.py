"""Agent factories for skill benchmark runners."""

from __future__ import annotations

import shutil
from pathlib import Path

from .react import LLMCallable, ReactAgent
from .skills import SkillSet
from .tools import ReactAgentTools


class ReactAgentFactory:
    """Build ReAct agents and stage initial skills for benchmark cases."""

    skills_dir_name = "skills"

    def __init__(
        self,
        llm: LLMCallable,
        *,
        max_turns: int,
        skill_sources: list[Path | str] | None = None,
        python_path: Path | str | None = None,
    ) -> None:
        self.llm = llm
        self.max_turns = max_turns
        self.python_path = python_path
        self.skill_sources = [Path(skill).resolve() for skill in (skill_sources or [])]
        for source in self.skill_sources:
            if not (source / "SKILL.md").exists():
                raise FileNotFoundError(f"Skill folder missing SKILL.md: {source}")

    def stage_live_skills(self, run_dir: Path) -> list[Path]:
        """Copy initial skills into the run directory so evolution can mutate them."""
        if not self.skill_sources:
            return []
        live_root = run_dir / "skills_live"
        staged: list[Path] = []
        for source in self.skill_sources:
            dest = live_root / source.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            staged.append(dest.resolve())
        self.skill_sources = staged
        return staged

    def build(self, workdir: Path, system_prompt: str) -> tuple[ReactAgent, ReactAgentTools]:
        """Build one agent bound to a case working directory."""
        tools = ReactAgentTools(workdir, python_path=self.python_path)
        agent_tools = tools.as_tools()
        prompt = system_prompt
        if self.skill_sources:
            skill_dirs = self._stage_case_skills(workdir)
            agent_tools.append(SkillSet(skill_dirs).as_tool())
            prompt = self._append_skill_prompt(system_prompt)
        return (
            ReactAgent(
                llm=self.llm,
                tools=agent_tools,
                max_turns=self.max_turns,
                system_prompt=prompt,
            ),
            tools,
        )

    def _stage_case_skills(self, workdir: Path) -> dict[str, Path]:
        skill_root = workdir / self.skills_dir_name
        skill_dirs: dict[str, Path] = {}
        for source in self.skill_sources:
            dest = skill_root / source.name
            shutil.copytree(source, dest, dirs_exist_ok=True)
            skill_dirs[source.name] = dest
        return skill_dirs

    def _append_skill_prompt(self, system_prompt: str) -> str:
        names = ", ".join(source.name for source in self.skill_sources)
        first = self.skill_sources[0].name
        return (
            system_prompt
            + f"\n\nA `skill` tool is available with expert skills: {names}. "
            + f'Call it first (e.g. skill(name="{first}")) to '
            + "load detailed guidance and the absolute path to reusable reference scripts before you start."
        )
