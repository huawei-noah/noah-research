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

from scienceflow.solver.lnr.resource_feedback_contract import eta_bucket, resource_feedback_text


def test_eta_bucket_edges() -> None:
    assert eta_bucket(60, confidence="high") == "short"
    assert eta_bucket(300, confidence="medium") == "medium"
    assert eta_bucket(900, confidence="low") == "long"
    assert eta_bucket(None, confidence="low") == "unknown"


def test_resource_feedback_includes_unlock_condition_without_research_advice() -> None:
    text = resource_feedback_text(
        status="DENIED_REPLAN",
        reason="invalid_deliverable_schema_preflight",
        scope="worker_deliverable",
        resource_mode="RED",
        blocked_class="heavy_gpu_candidate",
        gpu_ids=["5"],
        eta_next_train_sec=1476,
        eta_confidence="low",
        unlock_condition="submission_schema_valid",
        blocked_until_unlock=True,
        schema_state="invalid",
    )

    assert text.startswith("RESOURCE_FEEDBACK: DENIED_REPLAN because invalid_deliverable_schema_preflight")
    assert "eta_bucket=long" in text
    assert "eta_confidence=low" in text
    assert "unlock_condition=submission_schema_valid" in text
    assert "blocked_until_unlock=true" in text
    assert "schema_state=invalid" in text
    assert "Available non-GPU work" not in text
    assert "try " not in text.lower()
    assert "fix " not in text.lower()
