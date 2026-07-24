"""Episode chunking components for buffered add records."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from ...llm import LLMClient, get_llm_client
from ...logging import get_logger
from ...prompts import CONV_BOUNDARY_DETECTION_PROMPT

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SplitMode = Literal["llm", "rule"]
logger = get_logger(__name__)


@dataclass(slots=True)
class EpisodeBoundary:
    start_idx: int
    end_idx: int
    title: str = ""
    reasoning: str = ""


class EpisodesChunker:
    """Split buffered add records into complete episodes."""

    def __init__(
        self,
        *,
        mode: SplitMode = "llm",
        llm_client: LLMClient | None = None,
        max_messages: int = 15,
        max_minutes_from_first: int = 30,
        split_on_user_speaker: bool = True,
        boundary_prompt: str = CONV_BOUNDARY_DETECTION_PROMPT,
        resplit_prompt: str | None = None,
    ) -> None:
        self.mode = mode
        self.llm_client = llm_client
        self.max_messages = max_messages
        self.max_delta = timedelta(minutes=max_minutes_from_first)
        self.split_on_user_speaker = split_on_user_speaker
        self.boundary_prompt = boundary_prompt
        self.resplit_prompt = resplit_prompt

    async def detect_boundaries(
        self,
        entries: list[dict[str, Any]],
        *,
        force: bool = False,
        boundary_prompt: str | None = None,
        resplit_prompt: str | None = None,
    ) -> list[EpisodeBoundary]:
        """Detect episode boundaries in a conversation message list."""
        if not entries:
            return []
        if self.mode == "rule":
            return self._rule_boundaries(entries, force=force)
        effective_boundary = boundary_prompt or self.boundary_prompt
        effective_resplit = resplit_prompt if resplit_prompt is not None else self.resplit_prompt
        return await self._llm_boundaries(
            entries,
            force=force,
            boundary_prompt=effective_boundary,
            resplit_prompt=effective_resplit,
        )

    async def _llm_boundaries(
        self,
        entries: list[dict[str, Any]],
        *,
        force: bool,
        boundary_prompt: str,
        resplit_prompt: str | None,
    ) -> list[EpisodeBoundary]:
        prompt = boundary_prompt.replace("{conversation_list}", _format_entries(entries))
        llm_client = self.llm_client or get_llm_client()
        response = await llm_client.chat(
            task="memory.add.episode_boundary",
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            boundaries = _coerce_boundaries(response.parsed if response.parsed is not None else response.content)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("episode_boundary_parse_failed", error=str(exc))
            return []
        if force and entries and (not boundaries or boundaries[-1].end_idx < len(entries) - 1):
            start_idx = boundaries[-1].end_idx + 1 if boundaries else 0
            boundaries.append(EpisodeBoundary(start_idx=start_idx, end_idx=len(entries) - 1, title=""))
        boundaries = await self._resplit_oversized(boundaries, entries, resplit_prompt=resplit_prompt)
        return _complete_boundaries(boundaries, len(entries), force=force)

    def _rule_boundaries(self, entries: list[dict[str, Any]], *, force: bool) -> list[EpisodeBoundary]:
        boundaries: list[EpisodeBoundary] = []
        start_idx = 0
        first_time = _entry_time(entries[0])

        for idx, entry in enumerate(entries):
            if idx == start_idx:
                continue
            entry_time = _entry_time(entry)
            speaker = str(entry.get("speaker") or "").lower()
            should_split = False
            reason = ""

            if idx - start_idx >= self.max_messages:
                should_split = True
                reason = f"Reached {self.max_messages} messages."
            elif self.split_on_user_speaker and speaker == "user":
                should_split = True
                reason = "New user message starts a new episode."
            elif first_time is not None and entry_time is not None and entry_time - first_time > self.max_delta:
                should_split = True
                reason = f"More than {self.max_delta} since first message."

            if should_split:
                boundaries.append(EpisodeBoundary(start_idx=start_idx, end_idx=idx - 1, reasoning=reason))
                start_idx = idx
                first_time = entry_time

        if force and start_idx < len(entries):
            boundaries.append(EpisodeBoundary(start_idx=start_idx, end_idx=len(entries) - 1, reasoning="Force split."))
        return _complete_boundaries(boundaries, len(entries), force=force)

    async def _resplit_oversized(
        self,
        boundaries: list[EpisodeBoundary],
        entries: list[dict[str, Any]],
        *,
        resplit_prompt: str | None = None,
    ) -> list[EpisodeBoundary]:
        """Re-split oversized boundaries using LLM, falling back to rule-based chunking."""
        effective_resplit = resplit_prompt if resplit_prompt is not None else self.resplit_prompt
        if not effective_resplit:
            return _enforce_max_size(boundaries, self.max_messages)

        result: list[EpisodeBoundary] = []
        for b in boundaries:
            size = b.end_idx - b.start_idx + 1
            if size <= self.max_messages:
                result.append(b)
                continue

            sub_entries = entries[b.start_idx : b.end_idx + 1]
            num_parts = math.ceil(size / self.max_messages)
            prompt = (
                effective_resplit.replace("{conversation_list}", _format_entries(sub_entries))
                .replace("{num_parts}", str(num_parts))
                .replace("{max_messages}", str(self.max_messages))
            )

            try:
                llm_client = self.llm_client or get_llm_client()
                response = await llm_client.chat(
                    task="memory.add.episode_resplit",
                    messages=[{"role": "user", "content": prompt}],
                )
                sub_boundaries = _coerce_boundaries(
                    response.parsed if response.parsed is not None else response.content
                )
                if not sub_boundaries:
                    raise ValueError("LLM resplit returned empty boundaries")

                remapped = [
                    EpisodeBoundary(
                        start_idx=sb.start_idx + b.start_idx,
                        end_idx=sb.end_idx + b.start_idx,
                        title=sb.title,
                        reasoning=sb.reasoning,
                    )
                    for sb in sub_boundaries
                ]
                remapped = _enforce_max_size(remapped, self.max_messages)
                result.extend(remapped)
                logger.info(
                    "episode_resplit_ok",
                    original_size=size,
                    num_parts=len(remapped),
                    start_idx=b.start_idx,
                    end_idx=b.end_idx,
                )
            except Exception as exc:
                logger.warning(
                    "episode_resplit_failed_fallback_rule",
                    error=str(exc),
                    start_idx=b.start_idx,
                    end_idx=b.end_idx,
                )
                result.extend(_enforce_max_size([b], self.max_messages))

        return result


def _enforce_max_size(boundaries: list[EpisodeBoundary], max_size: int) -> list[EpisodeBoundary]:
    """Split any boundary that exceeds max_size into chunks of at most max_size."""
    result: list[EpisodeBoundary] = []
    for b in boundaries:
        size = b.end_idx - b.start_idx + 1
        if size <= max_size:
            result.append(b)
        else:
            cursor = b.start_idx
            while cursor <= b.end_idx:
                chunk_end = min(cursor + max_size - 1, b.end_idx)
                result.append(
                    EpisodeBoundary(start_idx=cursor, end_idx=chunk_end, title=b.title, reasoning=b.reasoning)
                )
                cursor = chunk_end + 1
    return result


def _complete_boundaries(boundaries: list[EpisodeBoundary], entry_count: int, *, force: bool) -> list[EpisodeBoundary]:
    complete: list[EpisodeBoundary] = []
    expected_start = 0
    for boundary in boundaries:
        start_idx = expected_start
        end_idx = min(boundary.end_idx, entry_count - 1)
        if end_idx < start_idx:
            continue
        if not force and end_idx >= entry_count - 1:
            break
        complete.append(
            EpisodeBoundary(start_idx=start_idx, end_idx=end_idx, title=boundary.title, reasoning=boundary.reasoning)
        )
        expected_start = end_idx + 1
    return complete


def _format_entries(entries: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"idx: {idx}, time: {entry.get('timestamp')}, speaker: {entry.get('speaker')}, content: {entry.get('content')}"
        for idx, entry in enumerate(entries)
    )


def _coerce_boundaries(value: Any) -> list[EpisodeBoundary]:
    if isinstance(value, str):
        value = _parse_json(value)
    if not isinstance(value, list):
        return []
    boundaries: list[EpisodeBoundary] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start_idx = item.get("start_idx", 0)
        end_idx = item.get("end_idx")
        if isinstance(start_idx, int) and isinstance(end_idx, int):
            boundaries.append(
                EpisodeBoundary(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    title=str(item.get("title") or ""),
                    reasoning=str(item.get("reasoning") or ""),
                )
            )
    return boundaries


def _parse_json(content: str) -> Any:
    text = content.strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((idx for idx in [text.find("{"), text.find("[")] if idx >= 0), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _entry_time(entry: dict[str, Any]) -> datetime | None:
    value = entry.get("timestamp")
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    return None
