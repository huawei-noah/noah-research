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

def build_global_merge_prompt(
    *,
    task_desc: str,
    artifact_path: str,
    candidate_count: int,
    wall_clock_sec: float,
    required_finals: int,
) -> str:
    artifact = artifact_path or "submission.csv"
    minutes = max(1, int(float(wall_clock_sec or 0) // 60))
    required = max(1, int(required_finals))
    final_dirs = ", ".join(f"`finals/final_{idx:02d}/`" for idx in range(required))
    return f"""The exploration phase is complete.

Continue as the same worker and perform the final cross-worker reduction for this task:
{task_desc}

The complete structured history is under `evidence/`. Candidate artifacts from all
workers are under `candidates/`. There are {candidate_count} packed candidates. Read
`evidence/index.json`, `candidates/candidate_index.json`, and candidate metadata before
deciding. Treat validation protocol, metric validity, lineage, route diversity, and
prediction compatibility as evidence; do not assume every score is directly comparable.

Independently decide whether selection, voting, blending, stacking, calibration, or
another task-appropriate aggregation is justified. Implement the reduction yourself.
Your job is not merely to pick one winner: produce defensible final artifact variants
that a downstream system or human can evaluate.

Rules:
- Use only public training/validation data and candidate files visible in this workspace.
- Do not use private/test labels or any private score.
- Do not mechanically average incompatible outputs or class identifiers.
- With small or weak validation sets, prefer route-diverse uniform or strongly
  regularized blends over sharply optimized weights.
- Interpret uncertainty, confidence, variance, and probability columns through the
  task metric; do not assume that increasing or independently averaging them is safer.
- Use optional `merge_payload/` prediction artifacts only after checking sample IDs,
  shapes, class order, semantics, and validation provenance.
- For optimization artifacts, do not blindly average invalid structures. Either copy
  the best valid artifact or construct a new valid artifact from candidate evidence.
- Do not retrain models or copy model weights into this workspace; use retained prediction artifacts when available.
- Do not create root-level output artifacts. Every output must live under
  `finals/final_*`.
- Keep the work reproducible. Write any helper scripts you use.
- Finish within about {minutes} minutes.

Required outputs:
- Create exactly {required} distinct final artifacts under {final_dirs}.
- Each final directory must contain `{artifact}`.
- Each final directory must contain `merge_report.md` explaining inputs and method.
- Do not duplicate byte-identical artifacts or make cosmetic perturbations merely to
  satisfy the count.
- If a blend is not appropriate, use distinct defensible candidate fallbacks.
- If the available evidence cannot support {required} honest variants, produce every
  defensible distinct variant available and explain the shortfall.

The system will run the configured evaluator on every `finals/final_*` directory.
It will expose the final directories as the complete merge output set.
"""
