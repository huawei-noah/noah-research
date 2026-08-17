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

from scienceflow.solver.lnr.prompts import build_estra_prompt
from scienceflow.solver.lnr.stage.peer_context import build_peer_route_evidence


def _row(
    *,
    worker: str,
    stage: str,
    metric: str,
    lower: str = "0",
    validation_ok: str = "1",
    selection_eligible: str = "1",
    metric_validity: str = "high",
    candidate_ready: str = "1",
    brief: str = "route",
) -> dict[str, str]:
    return {
        "worker_id": worker,
        "stage_id": stage,
        "candidate_id": f"{worker}:{stage}",
        "metric_value": metric,
        "metric_name": "Final Validation Score",
        "lower_is_better": lower,
        "validation_ok": validation_ok,
        "selection_eligible": selection_eligible,
        "selection_score": metric if selection_eligible == "1" else "",
        "metric_validity": metric_validity,
        "candidate_ready": candidate_ready,
        "submission_status": "ready" if candidate_ready == "1" else "missing_submission",
        "brief": brief,
        "why": brief,
    }


def test_peer_route_evidence_prefers_reliable_better_peer() -> None:
    evidence = build_peer_route_evidence(
        [
            _row(worker="W00", stage="S03", metric="0.81", brief="local baseline"),
            _row(worker="W01", stage="S04", metric="0.87", brief="peer pretrained route"),
            _row(worker="W02", stage="S05", metric="0.99", metric_validity="low", brief="leaky route"),
        ],
        current_worker_id="W00",
    )

    assert evidence.present
    assert evidence.peer_best_worker == "W01"
    assert evidence.peer_best_stage == "S04"
    assert evidence.evidence_strength == "peer_better"
    assert "W01:S04" in evidence.text
    assert "leaky route" not in evidence.text


def test_peer_route_evidence_filters_explicit_ineligible_peer() -> None:
    evidence = build_peer_route_evidence(
        [
            _row(worker="W00", stage="S02", metric="0.80"),
            _row(
                worker="W01",
                stage="S09",
                metric="0.95",
                selection_eligible="0",
                candidate_ready="1",
                brief="not comparable",
            ),
        ],
        current_worker_id="W00",
    )

    assert not evidence.present
    assert evidence.peer_record_count == 1


def test_estra_prompt_includes_peer_and_backtrack_blocks() -> None:
    prompt = build_estra_prompt(
        ledger_filename="Stage Memory View",
        ledger_text="S01 baseline\nS02 cnn route",
        latest_stage="S02",
        switch_candidate_stages=["S01"],
        stage_checkpoint_context="- S01: metric=0.80",
        peer_route_evidence="- Best reliable peer: W01:S03 metric=0.90",
        backtrack_reflection="- Candidate S01: metric=0.80",
        resource_context="pressure_state=GREEN",
    )

    assert "Peer route evidence:" in prompt
    assert "Best reliable peer: W01:S03" in prompt
    assert "Backtrack reflection:" in prompt
    assert "Candidate S01: metric=0.80" in prompt
    assert "ResourceContext snapshot:" in prompt
