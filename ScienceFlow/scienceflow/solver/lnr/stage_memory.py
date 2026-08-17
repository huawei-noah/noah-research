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

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scienceflow.solver.lnr.stage.stage_ledger import StageCard, normalize_stage_id, render_stage_cards

_STAGE_MEMORY_DIR = ".agent_memory/stage_memory"
_SUMMARIES_FILE = "summaries.jsonl"
_INDEX_FILE = "index.json"
_CURRENT_SEGMENT_FILE = "current_segment.json"
_EVENTS_FILE = "events.jsonl"
_PROMPT_VERSION = "stage-memory-v1"
_DEFAULT_CONTEXT_BUDGET_CHARS = 24_000
_MIN_BUDGET_CHARS = 4_000
_MAX_CARD_CHARS = 1_200


@dataclass(frozen=True)
class StageMemoryView:
    text: str
    raw_chars: int
    view_chars: int
    folded_stage_count: int
    summary_ids: tuple[str, ...]
    verification_stage_ids: tuple[str, ...]
    reused_summary_count: int
    created_summary_count: int


def stage_memory_root(workspace_dir: Path) -> Path:
    return Path(workspace_dir) / _STAGE_MEMORY_DIR


def sync_current_segment(workspace_dir: Path, cards: Iterable[StageCard]) -> None:
    root = stage_memory_root(workspace_dir)
    root.mkdir(parents=True, exist_ok=True)
    card_list = list(cards)
    payload = {
        "prompt_version": _PROMPT_VERSION,
        "stage_ids": [card.stage_id for card in card_list],
        "source_hash": _cards_hash(card_list),
        "updated_at": time.time(),
    }
    _write_json_atomic(root / _CURRENT_SEGMENT_FILE, payload)


def build_stage_memory_view(
    workspace_dir: Path,
    cards: list[StageCard],
    *,
    context_budget_chars: int = _DEFAULT_CONTEXT_BUDGET_CHARS,
    rebuild_on_stale: bool = True,
    target_stage: str = "",
    latest_stage: str = "",
    best_stage: str = "",
) -> StageMemoryView:
    """Build a prompt-safe stage memory view from effective stage cards.

    The append-only ledger remains the source of truth. This view persists folded
    historical summaries as a cache and keeps the current tail plus verification
    raw cards visible for decision-making.
    """
    budget = max(_MIN_BUDGET_CHARS, int(context_budget_chars or _DEFAULT_CONTEXT_BUDGET_CHARS))
    root = stage_memory_root(workspace_dir)
    root.mkdir(parents=True, exist_ok=True)
    sync_current_segment(workspace_dir, cards)

    raw_text = render_stage_cards(cards)
    raw_chars = len(raw_text)
    if not cards or raw_chars <= budget:
        _write_index(root, cards=cards, summaries=[], active_summary_ids=[], verification_stage_ids=[])
        return StageMemoryView(
            text=raw_text,
            raw_chars=raw_chars,
            view_chars=len(raw_text),
            folded_stage_count=0,
            summary_ids=(),
            verification_stage_ids=(),
            reused_summary_count=0,
            created_summary_count=0,
        )

    keep_ids = _raw_keep_stage_ids(cards, target_stage=target_stage, latest_stage=latest_stage, best_stage=best_stage)
    suffix_start = _raw_tail_start(cards, keep_ids=keep_ids)
    folded_cards = cards[:suffix_start]
    raw_tail_cards = cards[suffix_start:]
    if not folded_cards:
        clipped = raw_text[: max(0, budget - 64)].rstrip() + "\n... [stage memory clipped to budget]"
        return StageMemoryView(
            text=clipped,
            raw_chars=raw_chars,
            view_chars=len(clipped),
            folded_stage_count=0,
            summary_ids=(),
            verification_stage_ids=tuple(sorted(keep_ids)),
            reused_summary_count=0,
            created_summary_count=0,
        )

    existing = _load_summaries(root)
    summary, created = _ensure_l_summary(root, existing, folded_cards, rebuild_on_stale=rebuild_on_stale)
    summaries = [summary]
    verification_ids = _verification_stage_ids(folded_cards, target_stage=target_stage, best_stage=best_stage)
    verification_cards = [card for card in cards if card.stage_id in verification_ids and card.stage_id not in {c.stage_id for c in raw_tail_cards}]
    view = _render_view(summaries=summaries, verification_cards=verification_cards, raw_tail_cards=raw_tail_cards)

    # If the first folded view is still too large, summarize the summary layer
    # rather than clipping raw current evidence.
    if len(view) > budget:
        g_summary, g_created = _ensure_g_summary(root, _load_summaries(root), summaries, rebuild_on_stale=rebuild_on_stale)
        summaries = [g_summary]
        created = created or g_created
        view = _render_view(summaries=summaries, verification_cards=verification_cards, raw_tail_cards=raw_tail_cards)

    if len(view) > budget:
        view = _clip_view_preserving_tail(
            summaries=summaries,
            verification_cards=verification_cards,
            raw_tail_cards=raw_tail_cards,
            budget=budget,
        )

    active_summary_ids = tuple(str(s.get("summary_id") or "") for s in summaries if s.get("summary_id"))
    _write_index(
        root,
        cards=cards,
        summaries=summaries,
        active_summary_ids=list(active_summary_ids),
        verification_stage_ids=list(verification_ids),
    )
    _event(
        root,
        {
            "event": "stage_memory_view_built",
            "raw_chars": raw_chars,
            "view_chars": len(view),
            "folded_stage_count": len(folded_cards),
            "summary_ids": list(active_summary_ids),
            "verification_stage_ids": list(verification_ids),
            "created_summary": bool(created),
            "prompt_version": _PROMPT_VERSION,
        },
    )
    return StageMemoryView(
        text=view,
        raw_chars=raw_chars,
        view_chars=len(view),
        folded_stage_count=len(folded_cards),
        summary_ids=active_summary_ids,
        verification_stage_ids=tuple(verification_ids),
        reused_summary_count=0 if created else len(active_summary_ids),
        created_summary_count=1 if created else 0,
    )


def _raw_tail_start(cards: list[StageCard], *, keep_ids: set[str]) -> int:
    _ = keep_ids
    if len(cards) <= 3:
        return 0
    return max(0, len(cards) - 3)


def _raw_keep_stage_ids(
    cards: list[StageCard],
    *,
    target_stage: str,
    latest_stage: str,
    best_stage: str,
) -> set[str]:
    keep = {normalize_stage_id(target_stage), normalize_stage_id(latest_stage), normalize_stage_id(best_stage)}
    keep.discard("")
    for card in cards[-3:]:
        keep.add(card.stage_id)
    return keep


def _verification_stage_ids(cards: list[StageCard], *, target_stage: str, best_stage: str) -> list[str]:
    ids: list[str] = []
    for sid in (normalize_stage_id(best_stage), normalize_stage_id(target_stage)):
        if sid and any(card.stage_id == sid for card in cards):
            ids.append(sid)
    if cards:
        ids.extend([cards[0].stage_id, cards[-1].stage_id])
    invalid = next(
        (
            card.stage_id
            for card in cards
            if str(card.metric_validity or "").lower() in {"low", "invalid", "false"}
            or str(card.selection_eligible or "").lower() in {"false", "0", "no"}
        ),
        "",
    )
    if invalid:
        ids.append(invalid)
    return _dedupe(ids)


def _ensure_l_summary(
    root: Path,
    existing: list[dict[str, Any]],
    cards: list[StageCard],
    *,
    rebuild_on_stale: bool,
) -> tuple[dict[str, Any], bool]:
    source_hash = _cards_hash(cards)
    stage_range = [cards[0].stage_id, cards[-1].stage_id]
    summary_id = f"L_{cards[0].stage_id}_{cards[-1].stage_id}"
    stale_record: dict[str, Any] | None = None
    for item in existing:
        if item.get("summary_id") != summary_id or item.get("prompt_version") != _PROMPT_VERSION:
            continue
        if item.get("source_hash") == source_hash:
            _event(root, {"event": "stage_memory_summary_reused", "summary_id": summary_id})
            return item, False
        stale_record = item
    if stale_record is not None and not rebuild_on_stale:
        _event(root, {"event": "stage_memory_stale_summary_reused", "summary_id": summary_id})
        return stale_record, False
    record = {
        "summary_id": summary_id,
        "kind": "L",
        "stage_range": stage_range,
        "source_stage_ids": [card.stage_id for card in cards],
        "source_hash": source_hash,
        "facts_hash": source_hash,
        "prompt_version": _PROMPT_VERSION,
        "best_stage": _best_stage_id(cards),
        "key_stage_index": _key_stage_index(cards),
        "summary_text": _summarize_cards(cards, header=f"Historical segment {stage_range[0]}-{stage_range[1]}"),
        "created_at": time.time(),
    }
    _append_jsonl(root / _SUMMARIES_FILE, record)
    _event(root, {"event": "stage_memory_summary_created", "summary_id": summary_id, "kind": "L"})
    return record, True


def _ensure_g_summary(
    root: Path,
    existing: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    rebuild_on_stale: bool,
) -> tuple[dict[str, Any], bool]:
    source_hash = _summaries_hash(summaries)
    first = str(summaries[0].get("stage_range", ["S00", "S00"])[0])
    last = str(summaries[-1].get("stage_range", ["S00", "S00"])[-1])
    summary_id = f"G_{first}_{last}"
    stale_record: dict[str, Any] | None = None
    for item in existing:
        if item.get("summary_id") != summary_id or item.get("prompt_version") != _PROMPT_VERSION:
            continue
        if item.get("source_hash") == source_hash:
            _event(root, {"event": "stage_memory_summary_reused", "summary_id": summary_id})
            return item, False
        stale_record = item
    if stale_record is not None and not rebuild_on_stale:
        _event(root, {"event": "stage_memory_stale_summary_reused", "summary_id": summary_id})
        return stale_record, False
    text = "\n".join(f"- {s.get('summary_id')}: {s.get('summary_text', '')}" for s in summaries)
    record = {
        "summary_id": summary_id,
        "kind": "G",
        "stage_range": [first, last],
        "source_stage_ids": [sid for s in summaries for sid in s.get("source_stage_ids", [])],
        "source_hash": source_hash,
        "facts_hash": source_hash,
        "prompt_version": _PROMPT_VERSION,
        "best_stage": next((str(s.get("best_stage") or "") for s in summaries if s.get("best_stage")), ""),
        "key_stage_index": [entry for s in summaries for entry in s.get("key_stage_index", [])],
        "summary_text": _clip("Grouped historical summaries:\n" + text, 1_600),
        "created_at": time.time(),
    }
    _append_jsonl(root / _SUMMARIES_FILE, record)
    _event(root, {"event": "stage_memory_summary_created", "summary_id": summary_id, "kind": "G"})
    return record, True


def _render_view(
    *,
    summaries: list[dict[str, Any]],
    verification_cards: list[StageCard],
    raw_tail_cards: list[StageCard],
) -> str:
    lines = ["## Stage Memory View"]
    if summaries:
        lines.extend(["", "### Historical Summaries"])
        for item in summaries:
            stage_range = item.get("stage_range") or []
            range_text = "-".join(str(x) for x in stage_range) if stage_range else "unknown"
            lines.append(f"- {item.get('summary_id')} {range_text}: {_one_line(item.get('summary_text'), max_chars=700)}")
            key = item.get("key_stage_index") or []
            if key:
                key_text = "; ".join(
                    f"{entry.get('stage_id')}={entry.get('role')}" for entry in key[:4] if isinstance(entry, dict)
                )
                if key_text:
                    lines.append(f"  key_stage_index: {key_text}")
    expand_ids = [str(item.get("summary_id") or "") for item in summaries if item.get("summary_id")]
    if expand_ids:
        lines.extend(["", "### Available Expand IDs", ", ".join(expand_ids)])
    if verification_cards:
        lines.extend(["", "### Verification Raw Cards"])
        lines.append(_render_raw_cards(verification_cards))
    if raw_tail_cards:
        lines.extend(["", "### Current Raw Segment"])
        lines.append(_render_raw_cards(raw_tail_cards))
    return "\n".join(part for part in lines if part is not None).strip()


def _clip_view_preserving_tail(
    *,
    summaries: list[dict[str, Any]],
    verification_cards: list[StageCard],
    raw_tail_cards: list[StageCard],
    budget: int,
) -> str:
    tail = "\n\n".join(
        part
        for part in (
            "### Verification Raw Cards\n" + _render_raw_cards(verification_cards, max_card_chars=700) if verification_cards else "",
            "### Current Raw Segment\n" + _render_raw_cards(raw_tail_cards, max_card_chars=700) if raw_tail_cards else "",
        )
        if part
    )
    head_budget = max(800, budget - len(tail) - 200)
    head_lines = ["## Stage Memory View"]
    expand_ids = [str(item.get("summary_id") or "") for item in summaries if item.get("summary_id")]
    if expand_ids:
        head_lines.extend(["", "### Available Expand IDs", ", ".join(expand_ids)])
    head_lines.extend(["", "### Historical Summaries"])
    per_summary_chars = max(240, head_budget // max(1, len(summaries) + 1))
    for item in summaries:
        head_lines.append(f"- {item.get('summary_id')}: {_one_line(item.get('summary_text'), max_chars=per_summary_chars)}")
    head = _clip("\n".join(head_lines), head_budget)
    return (head + "\n\n" + tail).strip()[:budget]


def _render_raw_cards(cards: list[StageCard], *, max_card_chars: int = _MAX_CARD_CHARS) -> str:
    chunks: list[str] = []
    for card in cards:
        chunks.append(_clip(render_stage_cards([card]), max_card_chars))
    return "\n\n".join(chunks).strip()


def _summarize_cards(cards: list[StageCard], *, header: str) -> str:
    lines = [header]
    best = _best_stage_id(cards)
    if best:
        lines.append(f"best_stage={best}")
    for card in cards[:4]:
        lines.append(_card_summary_line(card))
    if len(cards) > 6:
        lines.append(f"... {len(cards) - 5} intermediate stages summarized ...")
    if len(cards) > 4:
        lines.append(_card_summary_line(cards[-1]))
    return _clip("\n".join(lines), 1_600)


def _card_summary_line(card: StageCard) -> str:
    metric = card.metric or "unknown"
    brief = _one_line(card.brief, max_chars=120)
    why = _one_line(card.why or card.route_evidence, max_chars=140)
    files = _one_line(getattr(card, "files", ""), max_chars=120)
    parts = [f"- {card.stage_id}: metric={metric}"]
    if card.metric_validity:
        parts.append(f"validity={card.metric_validity}")
    if card.selection_eligible:
        parts.append(f"eligible={card.selection_eligible}")
    if brief:
        parts.append(f"brief={brief}")
    if why:
        parts.append(f"why={why}")
    if files:
        parts.append(f"files={files}")
    return "; ".join(parts)


def _key_stage_index(cards: list[StageCard]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    best = _best_stage_id(cards)
    if best:
        out.append({"stage_id": best, "role": "best_valid", "note": "best metric in folded range"})
    if cards:
        out.append({"stage_id": cards[-1].stage_id, "role": "latest", "note": "latest stage in folded range"})
    invalid = next(
        (card for card in cards if str(card.selection_eligible or "").lower() in {"false", "0", "no"}),
        None,
    )
    if invalid is not None:
        out.append({"stage_id": invalid.stage_id, "role": "invalid_or_ineligible", "note": "excluded from selection"})
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in out:
        sid = item["stage_id"]
        if sid not in seen:
            deduped.append(item)
            seen.add(sid)
    return deduped


def _best_stage_id(cards: list[StageCard]) -> str:
    best_id = ""
    best_value: float | None = None
    for card in cards:
        value = _metric_float(card.metric)
        if value is None:
            continue
        lower = str(card.lower_is_better or "").strip().lower() not in {"false", "0", "no"}
        if best_value is None or (lower and value < best_value) or ((not lower) and value > best_value):
            best_value = value
            best_id = card.stage_id
    return best_id


def _metric_float(raw: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(raw or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _load_summaries(root: Path) -> list[dict[str, Any]]:
    path = root / _SUMMARIES_FILE
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _write_index(
    root: Path,
    *,
    cards: list[StageCard],
    summaries: list[dict[str, Any]],
    active_summary_ids: list[str],
    verification_stage_ids: list[str],
) -> None:
    payload = {
        "prompt_version": _PROMPT_VERSION,
        "updated_at": time.time(),
        "stage_ids": [card.stage_id for card in cards],
        "source_hash": _cards_hash(cards),
        "active_summary_ids": active_summary_ids,
        "verification_stage_ids": verification_stage_ids,
        "stage_to_summary": {
            sid: str(item.get("summary_id") or "")
            for item in summaries
            for sid in item.get("source_stage_ids", [])
        },
    }
    _write_json_atomic(root / _INDEX_FILE, payload)


def _cards_hash(cards: list[StageCard]) -> str:
    data = [
        {
            "stage_id": card.stage_id,
            "metric": card.metric,
            "lower_is_better": card.lower_is_better,
            "metric_validity": card.metric_validity,
            "selection_eligible": card.selection_eligible,
            "brief": card.brief,
            "why": card.why,
            "files": getattr(card, "files", ""),
            "route_evidence": card.route_evidence,
            "stage_events": list(card.stage_events),
        }
        for card in cards
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _summaries_hash(summaries: list[dict[str, Any]]) -> str:
    data = [
        {
            "summary_id": item.get("summary_id"),
            "source_hash": item.get("source_hash"),
            "summary_text": item.get("summary_text"),
        }
        for item in summaries
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _event(root: Path, payload: dict[str, Any]) -> None:
    record = {"ts": time.time(), **payload}
    _append_jsonl(root / _EVENTS_FILE, record)


def _one_line(text: Any, *, max_chars: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    return _clip(raw, max_chars)


def _clip(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = normalize_stage_id(item)
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
