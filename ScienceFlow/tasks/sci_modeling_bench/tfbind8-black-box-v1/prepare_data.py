#!/usr/bin/env python3
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

"""Materialize the public TFBind8 protocol view and a validation submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sci_modeling_bench.suites.design_bench import (
    TFBind8BlackBoxOptimizationTask,
)


TASK_ID = "sci-modeling-bench-tfbind8-v1"
REPO_ID = "sci-modeling-bench/design-bench"
CONFIG_NAME = "tfbind8"
SPLIT = "six6_ref_r1"
REVISION = "2ee2856f4255bb6a64c11b6c2660a6f41418e654"
PROTOCOL_ID = "design-bench/tfbind8-bottom-percentile-v1"


def prepare(output_dir: Path) -> dict[str, object]:
    task = TFBind8BlackBoxOptimizationTask.from_hub(
        repo_id=REPO_ID,
        config_name=CONFIG_NAME,
        revision=REVISION,
    )
    offline_data = task.build_input()
    rows = list(
        zip(
            offline_data["sequence"],
            offline_data["normalized_e_score"],
            strict=True,
        )
    )

    public_dir = output_dir / "public"
    validation_dir = output_dir / "validation"
    public_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    data_path = public_dir / "offline_data.csv"
    temporary_path = data_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sequence", "normalized_e_score"))
        writer.writerows((sequence, repr(float(score))) for sequence, score in rows)
    temporary_path.replace(data_path)

    visible_top = sorted(rows, key=lambda item: float(item[1]), reverse=True)[:128]
    baseline_path = validation_dir / "visible_top128_submission.json"
    baseline_path.write_text(
        json.dumps(
            {"candidates": [{"sequence": sequence} for sequence, _ in visible_top]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "repo_id": REPO_ID,
        "config_name": CONFIG_NAME,
        "split": SPLIT,
        "revision": REVISION,
        "protocol_id": PROTOCOL_ID,
        "visible_rows": len(rows),
        "visible_max_normalized_e_score": float(visible_top[0][1]),
        "offline_data_sha256": _sha256(data_path),
    }
    (public_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "public_dir": str(public_dir.resolve()),
        "baseline_submission": str(baseline_path.resolve()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/sci_modeling_bench/tfbind8-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
