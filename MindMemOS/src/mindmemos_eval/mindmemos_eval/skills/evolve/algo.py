"""Online skill-evolution algorithm clients.

Two implementations satisfy :class:`SkillEvolutionClient`:

- :class:`NoopSkillEvolutionClient` — the no-evolution baseline.
- :class:`FastAPISkillEvolutionClient` — drives the real MindMemOS server:
  register the staged skills once, record each finished rollout as an *injected*
  ``/v1/memory/add`` trace, then call ``POST /v1/skills/evolve``. When the server
  mints a new version, its ``SKILL.md`` is written back into the local skill
  directory so the next batch of tasks runs against the evolved skill.

The generic runner records every rollout of a batch and triggers one evolution
pass per managed skill.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mindmemos_sdk.memory import MemoryClient
from mindmemos_sdk.skills import (
    SkillCloudClient,
    SkillContext,
    SkillUsage,
    deserialize_bundle,
    read_local_bundle,
    serialize_bundle,
)
from mindmemos_sdk.transport import HttpTransport

_DIALOGUE_ROLES = {"user", "assistant", "system", "tool"}


class SkillCaseResult(Protocol):
    """Minimum result shape consumed by skill evolution algorithms."""

    case_id: str
    messages: list[dict[str, Any]]
    score: float


@dataclass
class EvolveOutcome:
    """Result of one skill's evolution attempt at a batch boundary."""

    skill_name: str
    cloud_skill_id: str
    evolved: bool
    pending_count: int
    threshold: int
    new_version_id: str | None = None
    new_version_ids: list[str] | None = None
    summarized_count: int = 0
    consumed_count: int = 0


class SkillEvolutionClient(Protocol):
    """Lifecycle a runner drives around batched online evolution."""

    async def prepare(self, skill_dirs: list[Path]) -> None:
        """Register the staged skills with the cloud version store (once)."""

    async def record_case(self, result: SkillCaseResult) -> None:
        """Record one finished rollout as an injected skill trace."""

    async def evolve(self) -> list[EvolveOutcome]:
        """Trigger one evolution pass per managed skill and re-stage updates."""

    async def aclose(self) -> None:
        """Release any underlying resources."""


class NoopSkillEvolutionClient:
    """Baseline evolution client that intentionally leaves skills unchanged."""

    async def prepare(self, skill_dirs: list[Path]) -> None:
        del skill_dirs

    async def record_case(self, result: SkillCaseResult) -> None:
        del result

    async def evolve(self) -> list[EvolveOutcome]:
        return []

    async def aclose(self) -> None:
        return None


@dataclass
class _ManagedSkill:
    """One staged skill tracked across evolution rounds."""

    name: str
    directory: Path
    cloud_skill_id: str
    version_id: str
    content_hash: str


class FastAPISkillEvolutionClient:
    """Drive online skill evolution against the MindMemOS FastAPI server.

    The SDK clients are synchronous; calls happen only at batch boundaries (once
    per rollout to record a trace, once per batch to evolve), so they run in a
    worker thread to avoid blocking the event loop while the batch's agents run.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        user_id: str = "spreadsheetbench-eval",
        timeout_seconds: float = 600.0,
        transcript_metadata: dict[str, Any] | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self._transport = transport or HttpTransport(
            base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds
        )
        self._skills = SkillCloudClient(self._transport)
        # skill_manager=None: pass the explicit injected skill_context straight through.
        self._memory = MemoryClient(self._transport, default_user_id=user_id, skill_manager=None)
        self._transcript_metadata = transcript_metadata or {}
        self._managed: list[_ManagedSkill] = []

    async def prepare(self, skill_dirs: list[Path]) -> None:
        self._managed = []
        for directory in skill_dirs:
            managed = await asyncio.to_thread(self._register_skill, Path(directory))
            if managed is not None:
                self._managed.append(managed)

    def _register_skill(self, directory: Path) -> _ManagedSkill | None:
        files = read_local_bundle(directory)
        if not files:
            return None
        data = self._skills.register(name=directory.name, content=serialize_bundle(files))
        return _ManagedSkill(
            name=directory.name,
            directory=directory,
            cloud_skill_id=data.cloud_skill_id,
            version_id=data.version_id,
            content_hash=data.content_hash,
        )

    async def record_case(self, result: SkillCaseResult) -> None:
        if not self._managed:
            return
        messages = _to_dialogue_messages(result.messages)
        if not messages:
            return
        skill_context = [
            SkillContext(
                name=managed.name,
                content_hash=managed.content_hash,
                base_version_id=managed.version_id,
                usage=SkillUsage.INJECTED,
            )
            for managed in self._managed
        ]
        metadata = {**self._transcript_metadata, "case_id": result.case_id}
        await asyncio.to_thread(self._add_trace, messages, skill_context, metadata, result.score, result.case_id)

    def _add_trace(
        self,
        messages: list[dict[str, Any]],
        skill_context: list[SkillContext],
        metadata: dict[str, Any],
        score: float | None,
        task_id: str,
    ) -> None:
        self._memory.add(
            messages,
            skill_context=skill_context,
            metadata=metadata,
            mode="async",
            score=score,
            task_id=task_id,
        )

    async def evolve(self) -> list[EvolveOutcome]:
        outcomes: list[EvolveOutcome] = []
        for managed in self._managed:
            outcomes.append(await asyncio.to_thread(self._evolve_one, managed))
        return outcomes

    def _evolve_one(self, managed: _ManagedSkill) -> EvolveOutcome:
        data = self._skills.evolve(managed.cloud_skill_id, mode="sync")
        outcome = EvolveOutcome(
            skill_name=managed.name,
            cloud_skill_id=managed.cloud_skill_id,
            evolved=data.evolved,
            pending_count=data.pending_count,
            threshold=data.threshold,
            new_version_id=data.new_version_id,
            new_version_ids=list(data.new_version_ids),
            summarized_count=data.summarized_count,
            consumed_count=data.consumed_count,
        )
        if data.evolved and data.new_version_id:
            content = self._skills.get_content(managed.cloud_skill_id, data.new_version_id)
            _write_bundle(managed.directory, deserialize_bundle(content.content))
            managed.version_id = data.new_version_id
            managed.content_hash = content.version.content_hash
        return outcome

    async def aclose(self) -> None:
        await asyncio.to_thread(self._transport.close)


def _to_dialogue_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render agent (OpenAI-style) messages as ``DialogueMessage`` dicts.

    Roles outside the server whitelist collapse to ``user``; assistant turns with
    only ``tool_calls`` are rendered into text so the call is not lost. Empty
    turns are dropped. Timestamps are synthesized monotonically (13-digit ms).
    """

    base_ms = int(time.time() * 1000)
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        role = role if role in _DIALOGUE_ROLES else "user"
        content = message.get("content")
        if content in (None, ""):
            content = _render_tool_calls(message.get("tool_calls"))
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if not text.strip():
            continue
        out.append({"role": role, "content": text, "timestamp": base_ms + index})
    return out


def _render_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    parts: list[str] = []
    for call in tool_calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name", "?")
        arguments = function.get("arguments", "")
        parts.append(f"[tool_call {name}] {arguments}")
    return "\n".join(parts)


def _write_bundle(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
