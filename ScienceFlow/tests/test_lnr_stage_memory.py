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

from scienceflow.solver.lnr.stage.stage_ledger import StageCard
from scienceflow.solver.lnr.stage_memory import build_stage_memory_view, sync_current_segment


def _card(idx: int, *, metric: float | None = None, why_size: int = 700) -> StageCard:
    sid = f"S{idx:02d}"
    value = metric if metric is not None else float(idx) / 100.0
    why = (f"stage {idx} route evidence " + "x" * why_size).strip()
    return StageCard(
        stage_id=sid,
        body=f"### {sid}\nmetric: {value}\nlower_is_better: false\nBRIEF: stage {idx}\nWHY: {why}",
        metric=str(value),
        lower_is_better="false",
        metric_validity="high",
        selection_eligible="true",
        brief=f"stage {idx} brief",
        why=why,
    )


def test_stage_memory_persists_and_reuses_closed_summary(tmp_path: Path) -> None:
    cards = [_card(i, why_size=650) for i in range(1, 9)]

    first = build_stage_memory_view(tmp_path, cards, context_budget_chars=4000, latest_stage="S08", best_stage="S08")
    second = build_stage_memory_view(tmp_path, cards, context_budget_chars=4000, latest_stage="S08", best_stage="S08")

    assert first.folded_stage_count > 0
    assert first.created_summary_count == 1
    assert second.created_summary_count == 0
    assert second.reused_summary_count >= 1
    assert (tmp_path / ".agent_memory" / "stage_memory" / "summaries.jsonl").exists()
    assert "Historical Summaries" in second.text
    assert "Current Raw Segment" in second.text


def test_stage_memory_keeps_verification_raw_card_for_folded_best(tmp_path: Path) -> None:
    cards = [_card(i, metric=0.99 if i == 2 else 0.10 + i / 100.0, why_size=650) for i in range(1, 9)]

    view = build_stage_memory_view(tmp_path, cards, context_budget_chars=4000, latest_stage="S08", best_stage="S02")

    assert "Verification Raw Cards" in view.text
    assert "### S02" in view.text
    assert "metric: 0.99" in view.text
    assert "Available Expand IDs" in view.text


def test_stage_memory_syncs_current_segment_metadata(tmp_path: Path) -> None:
    cards = [_card(i, why_size=80) for i in range(1, 4)]

    sync_current_segment(tmp_path, cards)

    current = (tmp_path / ".agent_memory" / "stage_memory" / "current_segment.json").read_text()
    assert '"stage_ids"' in current
    assert '"S01"' in current
    assert '"S03"' in current


def test_stage_memory_rebuilds_stale_summary_when_enabled(tmp_path: Path) -> None:
    cards = [_card(i, why_size=650) for i in range(1, 9)]
    build_stage_memory_view(tmp_path, cards, context_budget_chars=4000, latest_stage="S08", best_stage="S08")

    changed = list(cards)
    changed[0] = _card(1, metric=0.75, why_size=650)

    stale_reuse = build_stage_memory_view(
        tmp_path,
        changed,
        context_budget_chars=4000,
        rebuild_on_stale=False,
        latest_stage="S08",
        best_stage="S08",
    )
    rebuilt = build_stage_memory_view(
        tmp_path,
        changed,
        context_budget_chars=4000,
        rebuild_on_stale=True,
        latest_stage="S08",
        best_stage="S08",
    )

    assert stale_reuse.created_summary_count == 0
    assert rebuilt.created_summary_count == 1
