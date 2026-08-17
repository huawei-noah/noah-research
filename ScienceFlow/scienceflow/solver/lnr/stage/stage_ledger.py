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

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


_STEP_RE = re.compile(r"(?im)^\s*###\s+(S?\d+)\b.*$")
_STAGE_EVENT_RE = re.compile(r"(?im)^\s*##\s+Stage Event for\s+(S?\d+)\s*$")
_HEADING_RE = re.compile(r"(?im)^\s*##+\s+")
_FIELD_RE_TEMPLATE = r"(?im)^\s*(?:\*\*)?{field}(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.+?)\s*$"


@dataclass(frozen=True)
class StageCard:
    stage_id: str
    body: str
    metric: str = ""
    lower_is_better: str = ""
    run_time_sec: str = ""
    metric_type: str = ""
    metric_note: str = ""
    metric_validity: str = ""
    selection_eligible: str = ""
    metric_validity_reason_code: str = ""
    brief: str = ""
    why: str = ""
    files: str = ""
    route_evidence: str = ""
    stage_events: tuple[str, ...] = ()


def normalize_stage_id(raw: str) -> str:
    text = str(raw or "").strip().upper()
    if text.startswith("S"):
        text = text[1:]
    if not text.isdigit():
        return ""
    return f"S{int(text):02d}"


def next_stage_id(cards: list[StageCard]) -> str:
    highest = 0
    for card in cards:
        sid = normalize_stage_id(card.stage_id)
        if sid:
            highest = max(highest, int(sid[1:]))
    return f"S{highest + 1:02d}"


def _field(body: str, field: str) -> str:
    m = re.search(_FIELD_RE_TEMPLATE.format(field=re.escape(field)), body or "")
    return (m.group(1).strip() if m else "")


def parse_stage_cards(text: str) -> list[StageCard]:
    raw = str(text or "")
    matches = list(_STEP_RE.finditer(raw))
    stage_events = _stage_events(raw)
    cards: list[StageCard] = []
    for idx, match in enumerate(matches):
        stage_id = normalize_stage_id(match.group(1))
        if not stage_id:
            continue
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        cards.append(
            StageCard(
                stage_id=stage_id,
                body=body,
                metric=_field(body, "metric"),
                lower_is_better=_field(body, "lower_is_better"),
                run_time_sec=_field(body, "run_time_sec"),
                metric_type=_field(body, "metric_type"),
                metric_note=_field(body, "metric_note"),
                metric_validity=_field(body, "metric_validity"),
                selection_eligible=_field(body, "selection_eligible"),
                metric_validity_reason_code=_field(body, "metric_validity_reason_code"),
                brief=_field(body, "BRIEF"),
                why=_field(body, "WHY"),
                files=_field(body, "FILES"),
                route_evidence=_field(body, "route_evidence"),
                stage_events=stage_events.get(stage_id, ()),
            )
        )
    return cards


def _stage_events(raw: str) -> dict[str, tuple[str, ...]]:
    events: dict[str, list[str]] = {}
    matches = list(_STAGE_EVENT_RE.finditer(raw or ""))
    for idx, match in enumerate(matches):
        stage_id = normalize_stage_id(match.group(1))
        if not stage_id:
            continue
        start = match.end()
        next_heading = _HEADING_RE.search(raw, start)
        next_event = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        end = min(next_heading.start() if next_heading else len(raw), next_event)
        body = raw[start:end].strip()
        summary = re.sub(r"\s+", " ", body).strip()
        if summary:
            events.setdefault(stage_id, []).append(summary[:260])
    return {stage: tuple(lines) for stage, lines in events.items()}


def read_ledger(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def validate_stage_card(text: str, stage_id: str) -> tuple[bool, str, StageCard | None]:
    target = normalize_stage_id(stage_id)
    cards = parse_stage_cards(text)
    matches = [c for c in cards if c.stage_id == target]
    if not matches:
        return False, f"missing {target} heading", None
    if len(matches) > 1:
        return False, f"duplicate {target} heading", None
    card = matches[0]
    missing = []
    if not card.metric:
        missing.append("metric")
    if not card.lower_is_better:
        missing.append("lower_is_better")
    if not card.brief:
        missing.append("BRIEF")
    if not card.why:
        missing.append("WHY")
    if not card.files:
        missing.append("FILES")
    if missing:
        return False, f"{target} missing {','.join(missing)}", card
    return True, "", card


def validate_append_only_stage_commit(before: str, after: str, stage_id: str) -> tuple[bool, str]:
    target = normalize_stage_id(stage_id)
    if not target:
        return False, "invalid stage id"
    old_text = str(before or "")
    new_text = str(after or "")
    if old_text and not new_text.startswith(old_text):
        return False, "stage ledger is append-only; existing content changed"
    old_cards = parse_stage_cards(old_text)
    new_cards = parse_stage_cards(new_text)
    old_ids = [card.stage_id for card in old_cards]
    new_ids = [card.stage_id for card in new_cards]
    if new_ids[: len(old_ids)] != old_ids:
        return False, "existing stage order changed"
    added = new_ids[len(old_ids) :]
    if len(added) != 1:
        return False, f"append must add exactly one stage card, found {len(added)}"
    expected = next_stage_id(old_cards)
    if target != expected:
        return False, f"stage id must be next active stage; expected {expected}, got {target}"
    if added[0] != target:
        return False, f"appended stage must be {target}, got {added[0]}"
    ok, reason, _card = validate_stage_card(new_text, target)
    return ok, reason


def salvage_append_only_stage_commit(before: str, attempted: str, stage_id: str) -> tuple[bool, str, str]:
    """Extract the new target card from a rewritten ledger and append it to before."""
    target = normalize_stage_id(stage_id)
    if not target:
        return False, "", "invalid stage id"
    old_text = str(before or "")
    cards = [card for card in parse_stage_cards(attempted) if card.stage_id == target]
    if not cards:
        return False, "", f"missing {target} heading"
    card = cards[-1]
    sep = ""
    if old_text and not old_text.endswith("\n\n"):
        sep = "\n" if old_text.endswith("\n") else "\n\n"
    candidate = old_text + sep + card.body.strip() + "\n"
    ok, reason = validate_append_only_stage_commit(old_text, candidate, target)
    if not ok:
        return False, "", reason
    return True, candidate, ""


def compact_ledger_for_prompt(text: str, *, max_chars: int = 8000) -> str:
    body = str(text or "").strip()
    if len(body) <= max_chars:
        return body
    keep = max(1000, max_chars - 160)
    return body[: keep // 2] + "\n\n... [middle of stage ledger omitted] ...\n\n" + body[-keep // 2 :]


def _stage_num(stage_id: str) -> int:
    sid = normalize_stage_id(stage_id)
    return int(sid[1:]) if sid else 0


def _format_stage_summary_lines(cards: list[StageCard], *, header: str, max_chars: int) -> str:
    if not cards:
        return ""
    lines = [header]
    for card in cards:
        metric = card.metric or "unknown"
        run_time = re.sub(r"\s+", " ", card.run_time_sec or "").strip()
        metric_type = re.sub(r"\s+", " ", card.metric_type or "").strip()
        metric_validity = re.sub(r"\s+", " ", card.metric_validity or "").strip()
        selection_eligible = re.sub(r"\s+", " ", card.selection_eligible or "").strip()
        reason_code = re.sub(r"\s+", " ", card.metric_validity_reason_code or "").strip()
        brief = re.sub(r"\s+", " ", card.brief or "").strip()
        why = re.sub(r"\s+", " ", card.why or "").strip()
        files = re.sub(r"\s+", " ", card.files or "").strip()
        route_evidence = re.sub(r"\s+", " ", card.route_evidence or "").strip()
        if route_evidence and route_evidence not in why:
            why = f"{why}; {route_evidence}" if why else route_evidence
        piece = f"- {card.stage_id}: metric={metric}; BRIEF={brief}"
        if run_time:
            piece = f"- {card.stage_id}: metric={metric}; run_time_sec={run_time}; BRIEF={brief}"
        if metric_type:
            prefix = f"- {card.stage_id}: metric={metric}"
            if run_time:
                prefix += f"; run_time_sec={run_time}"
            piece = f"{prefix}; metric_type={metric_type}; BRIEF={brief}"
        if metric_validity:
            piece += f"; metric_validity={metric_validity}"
        if selection_eligible:
            piece += f"; selection_eligible={selection_eligible}"
        if reason_code:
            piece += f"; metric_validity_reason={reason_code}"
        if why:
            piece += f"; WHY={why}"
        if files:
            piece += f"; FILES={files}"
        lines.append(piece[:420])
    summary = "\n".join(lines).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 40].rstrip() + "\n... [tail summary truncated]"
    return summary


def tail_summary_after(text: str, target_stage: str, *, max_chars: int = 1200) -> str:
    target = normalize_stage_id(target_stage)
    cards = parse_stage_cards(text)
    if not target:
        return ""
    seen = False
    tail: list[StageCard] = []
    for card in cards:
        if card.stage_id == target:
            seen = True
            continue
        if seen:
            tail.append(card)
    return _format_stage_summary_lines(
        tail,
        header=f"Abandoned trajectory after {target} before estra:",
        max_chars=max_chars,
    )


def tail_summary_after_cards(
    cards: list[StageCard],
    target_stage: str,
    *,
    max_chars: int = 1200,
) -> str:
    target = normalize_stage_id(target_stage)
    if not target:
        return ""
    seen = False
    tail: list[StageCard] = []
    for card in cards:
        if card.stage_id == target:
            seen = True
            continue
        if seen:
            tail.append(card)
    return _format_stage_summary_lines(
        tail,
        header=f"Abandoned trajectory after {target} before estra:",
        max_chars=max_chars,
    )


def tail_summary_from_stage(
    text: str,
    *,
    start_stage: str = "S02",
    target_stage: str = "",
    max_chars: int = 1200,
) -> str:
    start = normalize_stage_id(start_stage) or "S02"
    target = normalize_stage_id(target_stage)
    start_num = _stage_num(start)
    cards = [card for card in parse_stage_cards(text) if _stage_num(card.stage_id) >= start_num]
    label = f" before estra to {target}" if target else " before estra"
    return _format_stage_summary_lines(
        cards,
        header=f"Abandoned compact trajectory from {start} through terminal{label}:",
        max_chars=max_chars,
    )


def _append_ledger_block(path: Path, block: str) -> None:
    text = str(block or "").strip()
    if not text:
        return
    existing = read_ledger(path)
    sep = ""
    if existing and not existing.endswith("\n\n"):
        sep = "\n" if existing.endswith("\n") else "\n\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(sep + text + "\n")


def append_archived_trajectory_summary(path: Path, *, target_stage: str, summary: str) -> None:
    if not summary.strip():
        return
    target = normalize_stage_id(target_stage)
    block = (
        f"## Archived Trajectory before estra to {target}\n"
        "HISTORICAL_EXPLORATION: Abandoned branch evidence after the restored stage; "
        "not the active route or a new instruction.\n"
        "SUMMARY:\n"
        f"{summary.strip()}\n"
    )
    _append_ledger_block(path, block)


def append_estra_summary(path: Path, *, target_stage: str, summary: str) -> None:
    if not summary.strip():
        return
    block = (
        f"## ESTRA Summary after {normalize_stage_id(target_stage)}\n"
        f"SUMMARY: {summary.strip()}\n"
    )
    _append_ledger_block(path, block)


def append_stage_event_summary(path: Path, *, target_stage: str, summary: str) -> None:
    text = re.sub(r"\s+", " ", str(summary or "")).strip()
    if not text:
        return
    target = normalize_stage_id(target_stage)
    if not target:
        return
    block = f"## Stage Event for {target}\n{text[:260]}\n"
    _append_ledger_block(path, block)


def stage_cards_with_overrides(
    cards: list[StageCard],
    overrides: Mapping[str, Mapping[str, Any]],
) -> list[StageCard]:
    """Return cards with adjudicated control-plane fields overlaid by stage id."""
    out: list[StageCard] = []
    allowed = {
        "metric",
        "lower_is_better",
        "run_time_sec",
        "metric_type",
        "metric_note",
        "metric_validity",
        "selection_eligible",
        "metric_validity_reason_code",
        "brief",
        "why",
        "files",
        "route_evidence",
    }
    for card in cards:
        data = overrides.get(card.stage_id, {})
        patch = {
            key: str(value)
            for key, value in data.items()
            if key in allowed and value not in (None, "")
        }
        out.append(replace(card, **patch) if patch else card)
    return out


def render_stage_cards(cards: list[StageCard]) -> str:
    """Render canonical stage cards for prompts without mutating the append-only ledger."""
    chunks: list[str] = []
    for card in cards:
        lines = [
            f"### {card.stage_id}",
            f"metric: {card.metric or 'unknown'}",
        ]
        if card.lower_is_better:
            lines.append(f"lower_is_better: {card.lower_is_better}")
        if card.run_time_sec:
            lines.append(f"run_time_sec: {card.run_time_sec}")
        if card.metric_type:
            lines.append(f"metric_type: {card.metric_type}")
        if card.metric_note:
            lines.append(f"metric_note: {card.metric_note}")
        if card.metric_validity:
            lines.append(f"metric_validity: {card.metric_validity}")
        if card.selection_eligible:
            lines.append(f"selection_eligible: {card.selection_eligible}")
        if card.metric_validity_reason_code:
            lines.append(f"metric_validity_reason_code: {card.metric_validity_reason_code}")
        if card.brief:
            lines.append(f"BRIEF: {card.brief}")
        why = card.why or ""
        if card.route_evidence and card.route_evidence not in why:
            why = f"{why}; {card.route_evidence}" if why else card.route_evidence
        if why:
            lines.append(f"WHY: {why}")
        if card.files:
            lines.append(f"FILES: {card.files}")
        for event in card.stage_events:
            lines.append(f"EVENT: {event}")
        chunks.append("\n".join(lines).rstrip())
    return "\n\n".join(chunks).strip()


def tail_summary_from_cards(
    cards: list[StageCard],
    *,
    start_stage: str = "S02",
    target_stage: str = "",
    max_chars: int = 1200,
) -> str:
    start = normalize_stage_id(start_stage) or "S02"
    target = normalize_stage_id(target_stage)
    start_num = _stage_num(start)
    selected = [card for card in cards if _stage_num(card.stage_id) >= start_num]
    label = f" before estra to {target}" if target else " before estra"
    return _format_stage_summary_lines(
        selected,
        header=f"Abandoned compact trajectory from {start} through terminal{label}:",
        max_chars=max_chars,
    )
