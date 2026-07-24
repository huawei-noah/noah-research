"""Skill self-evolution pipeline (design ``docs/skill``).

Flow for one ``cloud_skill_id`` (triggered by ``POST /v1/skills/evolve``):

1. Find every ``/v1/memory/add`` trace that ``injected`` this skill (a binding
   whose ``version_id`` belongs to the skill's lineage).
2. Count how many of those traces are *pending* — already-stored unconsumed
   summaries plus traces not yet summarized. If the count is below the evolution
   threshold, stop and report the shortfall (we summarize nothing).
3. Otherwise summarize the not-yet-summarized traces in parallel (bounded
   concurrency) with an LLM, storing each summary 1:1 with its add trace.
4. In add-time order, batch the pending summaries (``min_aggregate``..
   ``max_aggregate`` each) and, per batch, propose a patch against the current
   ``SKILL.md`` and apply it (optionally reformat), minting a new draft/cloud
   version chained on the previous head. Batches that would leave fewer than
   ``min_aggregate`` summaries are deferred to a later call.

Offline ``skill-rl`` aggregates many rollouts of ONE task; online we cannot
re-run a task, so summaries are aggregated ACROSS different tasks that injected
the same skill (see ``prompts/EN/skills``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

from ...components.skill import apply_patch_ops, deserialize_bundle
from ...config import SkillEvolutionConfig, get_config
from ...errors import SkillBundleError
from ...infra.db import get_database_clients
from ...llm import get_llm_client
from ...logging import get_logger, traced
from ...mappers import skill_trace_summary_from_record, to_skill_trace_summary_point
from ...prompts.EN.skills import (
    APPLY_PATCH_SYSTEM,
    PROPOSE_PATCH_SCORED_SYSTEM,
    PROPOSE_PATCH_SYSTEM,
    REWRITE_SKILL_SYSTEM,
    SUMMARY_SYSTEM,
    apply_patch_user,
    propose_patch_user,
    rewrite_skill_user,
    summarize_trajectory_user,
)
from ...typing import (
    SkillEvolveResult,
    SkillTraceSummary,
    SkillUsage,
    SkillVersion,
    SkillVersionStatus,
)
from ..memory_db import utcnow
from ..registry import create_pipeline, register
from .version_store import SkillVersionStore, get_skill_version_store

logger = get_logger(__name__)


class _Candidate:
    """One injected add trace eligible for summarization."""

    __slots__ = ("add_record_id", "created_at", "skill_name", "transcript", "score", "task_id")

    def __init__(
        self,
        add_record_id: str,
        created_at: datetime,
        skill_name: str,
        transcript: str,
        *,
        score: float | None = None,
        task_id: str | None = None,
    ) -> None:
        self.add_record_id = add_record_id
        self.created_at = created_at
        self.skill_name = skill_name
        self.transcript = transcript
        self.score = score
        self.task_id = task_id


@register(type="skill_evolve", name="trace_v2_summary")
class SkillEvolver:
    """Orchestrates summarize -> aggregate -> patch -> mint version for one skill.

    Registered as the ``trace_v2_summary`` evolve algorithm version (adapted from
    ``skill-rl``'s offline ``trace_v2_summary``); the active version is chosen via
    ``get_config().pipelines["skill_evolve"]``, like the ``add`` / ``search``
    pipeline families. Add new algorithm versions by registering another class
    under ``type="skill_evolve"``.

    Repositories and the LLM client are resolved lazily from the process globals
    so the evolver survives ``reset_*`` in tests/config reloads; they can be
    injected for unit tests.
    """

    def __init__(
        self,
        *,
        store: SkillVersionStore | None = None,
        skill_repo: Any = None,
        add_record_repo: Any = None,
        llm_client: Any = None,
    ) -> None:
        self._store = store
        self._skill_repo = skill_repo
        self._add_record_repo = add_record_repo
        self._llm = llm_client

    @property
    def store(self) -> SkillVersionStore:
        """Return the configured skill version store."""
        return self._store if self._store is not None else get_skill_version_store()

    @property
    def _skill(self):
        return self._skill_repo if self._skill_repo is not None else get_database_clients().skill

    @property
    def _add_record(self):
        return self._add_record_repo if self._add_record_repo is not None else get_database_clients().qdrant.add_record

    @property
    def llm(self):
        """Return the configured LLM client."""
        return self._llm if self._llm is not None else get_llm_client()

    @traced("skill_evolver.evolve")
    async def evolve(self, *, project_id: str, cloud_skill_id: str) -> SkillEvolveResult:
        """Run one evolution pass for ``cloud_skill_id`` (see module docstring).

        Raises:
            SkillNotFoundError: If the cloud skill does not exist in this project.
        """

        cfg = get_config().algo_config.skill_evolution

        # Head version + current SKILL.md text (published head wins, else latest).
        summary = await self.store.get_skill(project_id=project_id, cloud_skill_id=cloud_skill_id)
        head = summary.published_head or summary.latest_version
        head_md = await self._head_skill_md(project_id, cloud_skill_id, head)

        version_ids = await self._lineage_version_ids(project_id, cloud_skill_id)
        existing = await self._existing_summaries(project_id, cloud_skill_id)
        candidates = await self._injected_candidates(project_id, version_ids, existing.keys(), cfg)
        unconsumed = [s for s in existing.values() if s.consumed_version_id is None]

        pending_count = len(unconsumed) + len(candidates)
        if pending_count < cfg.min_aggregate:
            logger.info(
                "skill evolution below threshold",
                cloud_skill_id=cloud_skill_id,
                pending_count=pending_count,
                threshold=cfg.min_aggregate,
            )
            return SkillEvolveResult(
                cloud_skill_id=cloud_skill_id,
                evolved=False,
                pending_count=pending_count,
                threshold=cfg.min_aggregate,
            )

        new_summaries = await self._summarize_candidates(project_id, cloud_skill_id, candidates, cfg)
        pending = sorted([*unconsumed, *new_summaries], key=lambda s: s.created_at)
        if len(pending) < cfg.min_aggregate:
            # Summarization failures dropped us back under the threshold.
            return SkillEvolveResult(
                cloud_skill_id=cloud_skill_id,
                evolved=False,
                pending_count=len(pending),
                threshold=cfg.min_aggregate,
                summarized_count=len(new_summaries),
            )

        new_versions: list[SkillVersion] = []
        consumed = 0
        parent_id = head.version_id
        skill_md = head_md
        status = self._evolved_status(cfg)
        for batch in _batches(pending, cfg.min_aggregate, cfg.max_aggregate):
            version, skill_md = await self._mint_version(
                project_id=project_id,
                parent_version_id=parent_id,
                skill_name=head.skill_name,
                skill_md=skill_md,
                batch=batch,
                status=status,
                cfg=cfg,
            )
            if version is None:
                break
            for item in batch:
                await self._skill.mark_summary_consumed(item.summary_id, version.version_id)
            new_versions.append(version)
            consumed += len(batch)
            parent_id = version.version_id

        if not new_versions:
            return SkillEvolveResult(
                cloud_skill_id=cloud_skill_id,
                evolved=False,
                pending_count=len(pending),
                threshold=cfg.min_aggregate,
                summarized_count=len(new_summaries),
            )

        return SkillEvolveResult(
            cloud_skill_id=cloud_skill_id,
            evolved=True,
            pending_count=len(pending),
            threshold=cfg.min_aggregate,
            new_version_id=new_versions[-1].version_id,
            new_version_ids=[v.version_id for v in new_versions],
            summarized_count=len(new_summaries),
            consumed_count=consumed,
        )

    async def _lineage_version_ids(self, project_id: str, cloud_skill_id: str) -> set[str]:
        versions = await self.store.versions_since(project_id=project_id, cloud_skill_id=cloud_skill_id)
        return {v.version_id for v in versions}

    async def _existing_summaries(self, project_id: str, cloud_skill_id: str) -> dict[str, SkillTraceSummary]:
        out: dict[str, SkillTraceSummary] = {}
        cursor = None
        while True:
            records, cursor = await self._skill.scroll_summaries(project_id, cloud_skill_id, cursor=cursor)
            for record in records:
                item = skill_trace_summary_from_record(record)
                out[item.add_record_id] = item
            if cursor is None:
                break
        return out

    async def _injected_candidates(
        self,
        project_id: str,
        version_ids: set[str],
        summarized_ids: Any,
        cfg: SkillEvolutionConfig,
    ) -> list[_Candidate]:
        """Scroll add traces, keeping injected, not-yet-summarized ones (oldest-first).

        Note: Qdrant disables cursor pagination when ``order_by`` is set (it always
        returns ``next_page_offset=None``), which would silently cap the scan at one
        page. We therefore paginate with the point-id cursor and sort the surviving
        candidates by ``task_completed_at`` in Python instead.
        """

        summarized = set(summarized_ids)
        candidates: list[_Candidate] = []
        scanned = 0
        cursor = None
        while scanned < cfg.max_trace_scan:
            page_limit = min(200, cfg.max_trace_scan - scanned)
            records, cursor = await self._add_record.scroll(project_id, limit=page_limit, cursor=cursor)
            for record in records:
                scanned += 1
                add_record_id = record.point_id
                if add_record_id in summarized:
                    continue
                skill_name = self._injected_skill_name(record.payload, version_ids)
                if skill_name is None:
                    continue
                transcript = _render_transcript(record.payload.get("messages") or [], cfg.transcript_max_chars)
                candidates.append(
                    _Candidate(
                        add_record_id=add_record_id,
                        created_at=_parse_dt(record.payload.get("task_completed_at")),
                        skill_name=skill_name,
                        transcript=transcript,
                        score=record.payload.get("score"),
                        task_id=record.payload.get("task_id"),
                    )
                )
            if cursor is None:
                break
        candidates.sort(key=lambda c: c.created_at)
        return candidates

    @staticmethod
    def _injected_skill_name(payload: dict[str, Any], version_ids: set[str]) -> str | None:
        for binding in payload.get("skill_bindings") or []:
            if binding.get("usage") == SkillUsage.INJECTED.value and binding.get("version_id") in version_ids:
                return binding.get("name") or ""
        return None

    async def _summarize_candidates(
        self,
        project_id: str,
        cloud_skill_id: str,
        candidates: Sequence[_Candidate],
        cfg: SkillEvolutionConfig,
    ) -> list[SkillTraceSummary]:
        if not candidates:
            return []
        semaphore = asyncio.Semaphore(max(1, cfg.summary_concurrency))

        async def run(candidate: _Candidate) -> SkillTraceSummary | None:
            async with semaphore:
                text = await self._summarize_one(candidate, cfg)
            if not text:
                return None
            item = SkillTraceSummary(
                summary_id=candidate.add_record_id,
                project_id=project_id,
                cloud_skill_id=cloud_skill_id,
                add_record_id=candidate.add_record_id,
                skill_name=candidate.skill_name,
                summary=text,
                created_at=candidate.created_at,
                score=candidate.score,
                task_id=candidate.task_id,
            )
            await self._skill.upsert_summary(to_skill_trace_summary_point(item))
            return item

        results = await asyncio.gather(*(run(c) for c in candidates))
        return [item for item in results if item is not None]

    async def _summarize_one(self, candidate: _Candidate, cfg: SkillEvolutionConfig) -> str | None:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": summarize_trajectory_user(candidate.skill_name, candidate.transcript)},
        ]
        try:
            reply = await self.llm.chat(task=cfg.summary_task, messages=messages)
        except Exception as exc:
            logger.warning("skill trajectory summary failed", add_record_id=candidate.add_record_id, error=str(exc))
            return None
        return (reply.content or "").strip() or None

    async def _mint_version(
        self,
        *,
        project_id: str,
        parent_version_id: str,
        skill_name: str,
        skill_md: str,
        batch: Sequence[SkillTraceSummary],
        status: SkillVersionStatus,
        cfg: SkillEvolutionConfig,
    ) -> tuple[SkillVersion | None, str]:
        """Propose+apply a patch for one batch and mint a child version.

        Returns ``(version, new_skill_md)``; ``(None, skill_md)`` if the LLM stage
        or registration failed (the caller stops and leaves the batch unconsumed).
        """

        try:
            patch = await self._propose_patch(skill_name, skill_md, batch, cfg)
            new_md = await self._apply_patch(skill_md, patch, cfg)
            if cfg.rewrite_skill:
                new_md = await self._rewrite(new_md, cfg)
        except Exception as exc:
            logger.warning("skill patch generation failed", parent_version_id=parent_version_id, error=str(exc))
            return None, skill_md

        new_md = (new_md or "").strip()
        if not new_md:
            logger.warning("skill patch produced empty content", parent_version_id=parent_version_id)
            return None, skill_md

        try:
            version = await self.store.create_evolved_version(
                project_id=project_id,
                parent_version_id=parent_version_id,
                name=skill_name,
                content=new_md,
                status=status,
            )
        except SkillBundleError as exc:
            logger.warning("evolved skill content rejected", parent_version_id=parent_version_id, error=str(exc))
            return None, skill_md
        return version, new_md

    async def _propose_patch(
        self, skill_name: str, skill_md: str, batch: Sequence[SkillTraceSummary], cfg: SkillEvolutionConfig
    ) -> str:
        """Propose a skill patch from trajectory summaries."""

        summaries = [s.summary for s in batch]
        scores = [s.score for s in batch]
        use_scores = cfg.use_trajectory_score and any(score is not None for score in scores)
        system = PROPOSE_PATCH_SCORED_SYSTEM if use_scores else PROPOSE_PATCH_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": propose_patch_user(skill_name, skill_md, summaries, scores if use_scores else None),
            },
        ]
        reply = await self.llm.chat(task=cfg.patch_task, messages=messages)
        return (reply.content or "").strip()

    async def _apply_patch(self, skill_md: str, patch: str, cfg: SkillEvolutionConfig) -> str:
        """Apply the change plan as structured edit ops and return the new SKILL.md.

        The model emits only the changed spans (replace/insert/delete); the ops are
        applied deterministically to ``skill_md`` via the chat ``format_parser``. When
        an edit fails to apply (e.g. a non-unique anchor), ``feedback_on_parse_error``
        feeds the failed reply plus the ``SkillEditError`` back into the conversation
        so the retry can self-correct (add surrounding context, fix JSON) instead of
        re-running the identical prompt; the failure is also recorded to ClickHouse via
        the ``llm.chat.parse_error`` span event.
        """

        messages = [
            {"role": "system", "content": APPLY_PATCH_SYSTEM},
            {"role": "user", "content": apply_patch_user(skill_md, patch)},
        ]
        reply = await self.llm.chat(
            task=cfg.apply_task,
            messages=messages,
            format_parser=lambda content: apply_patch_ops(skill_md, content),
            feedback_on_parse_error=True,
        )
        return reply.parsed if reply.parsed is not None else (reply.content or "")

    async def _rewrite(self, skill_md: str, cfg: SkillEvolutionConfig) -> str:
        messages = [
            {"role": "system", "content": REWRITE_SKILL_SYSTEM},
            {"role": "user", "content": rewrite_skill_user(skill_md)},
        ]
        reply = await self.llm.chat(task=cfg.rewrite_task, messages=messages)
        return reply.content or ""

    async def _head_skill_md(self, project_id: str, cloud_skill_id: str, head: SkillVersion) -> str:
        content = await self.store.get_content(
            project_id=project_id, cloud_skill_id=cloud_skill_id, version_id=head.version_id
        )
        files = deserialize_bundle(content.content)
        # The canonical bundle keys are basenames; SKILL.md is the only whitelisted file.
        return files.get("SKILL.md", "")

    @staticmethod
    def _evolved_status(cfg: SkillEvolutionConfig) -> SkillVersionStatus:
        try:
            return SkillVersionStatus(cfg.evolved_status)
        except ValueError:
            logger.warning("invalid evolved_status, defaulting to draft", value=cfg.evolved_status)
            return SkillVersionStatus.DRAFT


def _batches(items: Sequence[SkillTraceSummary], min_size: int, max_size: int) -> Iterator[list[SkillTraceSummary]]:
    """Yield consecutive batches of ``min_size``..``max_size`` in order.

    Greedy: each batch takes up to ``max_size``; the loop stops once fewer than
    ``min_size`` items remain, leaving the remainder for a later evolve call. So
    10 items (min=4, max=8) yields one batch of 8 (2 deferred); 12 yields 8 + 4.
    """

    i = 0
    n = len(items)
    while n - i >= min_size:
        size = min(max_size, n - i)
        yield list(items[i : i + size])
        i += size


def _render_transcript(messages: list[Any], max_chars: int) -> str:
    """Render stored add ``messages`` into a compact transcript for the LLM."""

    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(_truncate(str(message), max_chars))
            continue
        if "role" in message and "content" in message:
            lines.append(f"[{message.get('role', '?')}] {_truncate(str(message.get('content', '')), max_chars)}")
        elif "text" in message:
            lines.append(f"[text] {_truncate(str(message.get('text', '')), max_chars)}")
        elif "url" in message:
            lines.append(f"[url] {message.get('url', '')}")
        elif "file_name" in message:
            lines.append(f"[file] {message.get('file_name', '')}")
        else:
            lines.append(_truncate(json.dumps(message, ensure_ascii=False), max_chars))
    return "\n".join(lines)


def _truncate(text: str, n: int) -> str:
    text = text.strip()
    if n <= 0 or len(text) <= n:
        return text
    head = text[: n // 2]
    tail = text[-n // 2 :]
    return f"{head}\n…[{len(text) - n} chars elided]…\n{tail}"


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return utcnow()


_evolver: Any = None
_evolver_name: str | None = None


def get_skill_evolver() -> Any:
    """Process-global skill-evolve pipeline, selected by config.

    Builds the algorithm version named in ``get_config().pipelines["skill_evolve"]``
    via the pipeline registry, rebuilding if that name changes (config reload).
    """

    global _evolver, _evolver_name
    name = get_config().pipelines["skill_evolve"]
    if _evolver is None or _evolver_name != name:
        _evolver = create_pipeline(type="skill_evolve", name=name)
        _evolver_name = name
    return _evolver
