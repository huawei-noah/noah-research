# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

from pathlib import Path

from scienceflow.config.settings import Config
from scienceflow.core.agent.run_policy import AutoContinuePolicy
from scienceflow.core.llm_http import aclose_llm_clients
from scienceflow.core.orchestrator import Orchestrator
from scienceflow.core.skills.paths import default_skill_library_dir
from scienceflow.core.skills.registry import SkillRegistry


def load_repo_skill_registry(repo_root: Path) -> SkillRegistry:
    """Load the repo-level ``.scienceflow`` skill library."""
    registry = SkillRegistry()
    registry.load_all(default_skill_library_dir(repo_root))
    return registry


def build_data_prep_agent_request(cfg: Config, task: str) -> str:
    workspace_dir = Path(cfg.workspace_dir).resolve()
    input_dir = Path(cfg.input_data_dir).expanduser().resolve(strict=False)
    dataset_dir = workspace_dir / "dataset"
    task_text = str(task or "").strip()

    return (
        "You are the ScienceFlow data preparation agent for one task.\n\n"
        "Goal: prepare the workspace `dataset/` directory so downstream solver "
        "agents can train, validate, and create submissions from a flat data view.\n\n"
        "Required first steps:\n"
        "1. Call the `skill` tool with action=list, category=data_processing, "
        "tag=data-preparation.\n"
        "2. Read `data_prep` with the `skill` tool.\n"
        "3. If the inspected dataset layout matches a more specific listed "
        "data-preparation skill, read that skill and follow it.\n"
        "4. Inspect the read-only input data root directly; `dataset/` may start "
        "as an empty output directory.\n\n"
        "Scope and safety:\n"
        f"- Workspace root: `{workspace_dir}`.\n"
        f"- Dataset output root: `{dataset_dir}`.\n"
        f"- Read-only input data root: `{input_dir}`.\n"
        "- Write only inside the workspace, preferably under `dataset/`.\n"
        "- Do not modify the read-only input data root.\n"
        "- Use symlinks for large media or bulky unchanged folders.\n"
        "- If a target file under `dataset/` is a symlink and must be replaced, "
        "unlink the symlink first; never write through a symlink into the input root.\n"
        "- Preserve task schemas and ids unless a skill explicitly instructs a "
        "normalization needed for reliable validation.\n"
        "- If the currently exposed dataset is already valid and no safer "
        "validation split can be inferred, leave it in place and write a short "
        "`dataset/split_report.md` explaining that decision.\n\n"
        "Deliverables:\n"
        "- `dataset/` contains the prepared training/test/submission inputs.\n"
        "- `dataset/split_manifest.json` exists when a train/validation split is "
        "created or materially changed.\n"
        "- `dataset/split_report.md` summarizes the layout, counts, validation "
        "policy, and known limitations.\n"
        "- Finish with a concise summary of created or reused files.\n\n"
        "Task description:\n"
        "```text\n"
        f"{task_text}\n"
        "```"
    )


async def run_data_prep_agent(cfg: Config, task: str, *, repo_root: Path) -> str:
    """Run an agent-backed data preparation turn using the repo skill library."""
    registry = load_repo_skill_registry(repo_root)
    orchestrator = Orchestrator(cfg)
    trace_hook = orchestrator.make_llm_call_tracer(
        node_id="prep",
        process_id=str(getattr(cfg, "exp_id", "") or "prep"),
        detail_prefix="mode=prep;skill_mode=all",
    )
    max_steps = int(getattr(cfg, "qa_max_steps", 20) or 20)
    if max_steps <= 0:
        max_steps = 20
    max_steps = max(8, max_steps)
    agent = orchestrator.create_science_agent(
        task_description=task or None,
        run_policy=AutoContinuePolicy(max_text_only_retries=2),
        max_steps_override=max_steps,
        on_llm_call=trace_hook,
        memory_dir_override=Path(cfg.log_dir) / "prep_agent_memory",
        memory_agent_name="DataPrepAgent",
        repl_bash_write_mode=True,
        stable_system_prompt=True,
        pin_environment_context=True,
        skill_registry=registry,
        task_type="*",
        skill_tool_mode="all",
        skill_allow_generic_wildcard=True,
        skill_visible_max=12,
        bash_max_output_chars_override=12000,
        bash_max_stream_line_chars_override=2400,
        bash_observation_summary_override=True,
    )
    try:
        return await agent.run(build_data_prep_agent_request(cfg, task)) or ""
    finally:
        await aclose_llm_clients(agent.llm)
