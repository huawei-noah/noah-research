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
import os
import shutil
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.global_merge.candidate_pack import (
    candidate_artifact_source,
)
from scienceflow.solver.lnr.global_merge.fallback import metric_float, ranked_candidates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_symlink(src: Path, dst: Path) -> None:
    source = src.resolve(strict=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(dst)
    dst.symlink_to(os.path.relpath(source, start=dst.parent.resolve(strict=False)))


def _canonicalize_workspace_finals(
    *,
    workspace_finals: Path,
    finals_dir: Path,
    merge_dir: Path,
    candidates: list[dict[str, Any]],
    artifact_path: str,
    ledger_filename: str,
    max_finals: int,
) -> list[dict[str, Any]]:
    artifact = artifact_path or "submission.csv"
    candidate_by_sha: dict[str, Path] = {}
    for candidate in candidates:
        source = candidate_artifact_source(candidate, artifact_path=artifact)
        if source is None:
            continue
        try:
            candidate_by_sha[_sha256_file(source)] = source.resolve(strict=True)
        except OSError:
            continue

    shutil.rmtree(finals_dir, ignore_errors=True)
    finals_dir.mkdir(parents=True, exist_ok=True)
    promoted: list[dict[str, Any]] = []
    seen_artifact_shas: set[str] = set()
    source_dirs = sorted(p for p in workspace_finals.glob("final_*") if p.is_dir())
    for source_dir in source_dirs:
        if len(promoted) >= max(0, int(max_finals)):
            break
        source_artifact = source_dir / artifact
        if not source_artifact.is_file():
            continue
        artifact_sha = _sha256_file(source_artifact)
        if artifact_sha in seen_artifact_shas:
            continue
        seen_artifact_shas.add(artifact_sha)
        canonical = candidate_by_sha.get(artifact_sha)
        canonical_kind = "candidate_snapshot"
        if canonical is None:
            canonical_kind = "merge_snapshot"
            canonical = (
                merge_dir / "submission_snapshots" / artifact_sha[:16] / artifact
            )
            canonical.parent.mkdir(parents=True, exist_ok=True)
            if canonical.is_file():
                if _sha256_file(canonical) != artifact_sha:
                    raise RuntimeError(f"merge snapshot hash collision: {canonical}")
                source_artifact.unlink()
            else:
                source_artifact.replace(canonical)
            canonical.chmod(0o444)
        elif source_artifact.resolve(strict=True) != canonical:
            source_artifact.unlink()
        _relative_symlink(canonical, source_artifact)

        final_dir = finals_dir / f"final_{len(promoted):02d}"
        _relative_symlink(canonical, final_dir / artifact)
        _copy_optional_stage_files(
            source_dir=source_dir,
            final_dir=final_dir,
            ledger_filename=ledger_filename,
        )
        report = source_dir / "merge_report.md"
        if report.is_file():
            shutil.copy2(report, final_dir / report.name)
        promoted.append(
            {
                "final_id": final_dir.name,
                "artifact_sha": artifact_sha,
                "canonical_artifact": str(canonical),
                "canonical_kind": canonical_kind,
            }
        )
    return promoted


def _expose_dataset(workspace: Path, dataset_source: Path | None) -> None:
    if dataset_source is None or not dataset_source.exists():
        return
    target = workspace / "dataset"
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(
        dataset_source.resolve(strict=True),
        target_is_directory=dataset_source.is_dir(),
    )


def _cleanup_legacy_single_winner_outputs(merge_dir: Path) -> None:
    for name in (
        "submission.csv",
        "selected_submission.csv",
        "selected_candidate.json",
        "selected_solution.py",
        "selected_run_results.md",
        "selected_merge_report.md",
    ):
        try:
            (merge_dir / name).unlink(missing_ok=True)
        except OSError:
            pass
    shutil.rmtree(merge_dir / "selected_artifact", ignore_errors=True)


def _copy_optional_stage_files(
    *,
    source_dir: Path,
    final_dir: Path,
    ledger_filename: str,
) -> None:
    for name in ("solution.py", ledger_filename):
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, final_dir / src.name)


def _write_fallback_finals(
    *,
    finals_dir: Path,
    candidates: list[dict[str, Any]],
    artifact_path: str,
    ledger_filename: str,
    max_finals: int = 3,
    ordered_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    artifact = artifact_path or "submission.csv"
    finals_dir.mkdir(parents=True, exist_ok=True)
    limit = max(0, int(max_finals))
    seen_artifact_shas: set[str] = set()
    for final_dir in sorted(p for p in finals_dir.glob("final_*") if p.is_dir()):
        existing_artifact = final_dir / artifact
        if not existing_artifact.is_file():
            continue
        try:
            seen_artifact_shas.add(_sha256_file(existing_artifact))
        except OSError:
            continue

    next_index = 0
    while (finals_dir / f"final_{next_index:02d}").exists():
        next_index += 1
    sources: list[dict[str, Any]] = []
    for candidate in (
        ordered_candidates
        if ordered_candidates is not None
        else ranked_candidates(candidates)
    ):
        if len(seen_artifact_shas) >= limit:
            break
        src_artifact = candidate_artifact_source(candidate, artifact_path=artifact)
        if src_artifact is None:
            continue
        try:
            artifact_sha = _sha256_file(src_artifact)
        except OSError:
            continue
        if artifact_sha in seen_artifact_shas:
            continue
        final_dir = finals_dir / f"final_{next_index:02d}"
        next_index += 1
        dst_artifact = final_dir / artifact
        _relative_symlink(src_artifact, dst_artifact)
        snapshot = Path(str(candidate.get("snapshot_path") or ""))
        _copy_optional_stage_files(
            source_dir=snapshot,
            final_dir=final_dir,
            ledger_filename=ledger_filename,
        )
        source = dict(candidate)
        source.update(
            {
                "final_id": final_dir.name,
                "artifact_path": artifact,
                "artifact_sha": artifact_sha,
            }
        )
        (final_dir / "source_candidate_metadata.json").write_text(
            json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (final_dir / "merge_report.md").write_text(
            "\n".join(
                [
                    f"# Merge Report: {final_dir.name}",
                    "",
                    "Fallback final generated because the merge agent did not produce enough distinct final artifacts.",
                    f"source_candidate_id: {candidate.get('candidate_id')}",
                    f"source_metric: {candidate.get('metric_value')}",
                    f"metric_validity: {candidate.get('metric_validity') or 'unspecified'}",
                    f"selection_eligible: {candidate.get('selection_eligible')}",
                    f"artifact: {artifact}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        seen_artifact_shas.add(artifact_sha)
        sources.append(source)
    return sources


def _ranked_authoritative_candidates(
    candidates: list[dict[str, Any]],
    *,
    artifact_path: str,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("metric_authoritative") is not True:
            continue
        if candidate.get("candidate_ready") is not True:
            continue
        if candidate.get("validation_ok") is not True:
            continue
        if candidate.get("selection_eligible") is not True:
            continue
        if str(candidate.get("metric_validity") or "").strip().lower() != "high":
            continue
        expected_sha = str(candidate.get("artifact_sha") or "").strip()
        if not expected_sha and Path(artifact_path).suffix.lower() == ".csv":
            expected_sha = str(candidate.get("submission_sha") or "").strip()
        if not expected_sha:
            continue
        if metric_float(candidate) is None:
            continue
        if not isinstance(candidate.get("lower_is_better"), bool):
            continue
        eligible.append(candidate)
    directions = {bool(candidate["lower_is_better"]) for candidate in eligible}
    if len(directions) != 1:
        return []
    lower_is_better = directions.pop()
    return sorted(
        eligible,
        key=lambda candidate: (
            float(metric_float(candidate) or 0.0)
            if lower_is_better
            else -float(metric_float(candidate) or 0.0),
            str(candidate.get("candidate_id") or ""),
        ),
    )


def materialize_best_stage_final(
    *,
    merge_dir: Path,
    candidates: list[dict[str, Any]],
    artifact_path: str,
    ledger_filename: str,
) -> dict[str, Any]:
    """Expose the best eligible historical Stage without invoking a merge Agent."""

    merge_dir.mkdir(parents=True, exist_ok=True)
    finals_dir = merge_dir / "finals"
    shutil.rmtree(finals_dir, ignore_errors=True)
    ranked = _ranked_authoritative_candidates(
        candidates,
        artifact_path=artifact_path,
    )
    sources = _write_fallback_finals(
        finals_dir=finals_dir,
        candidates=ranked,
        artifact_path=artifact_path,
        ledger_filename=ledger_filename,
        max_finals=1,
        ordered_candidates=ranked,
    )
    manifest: dict[str, Any] = {
        "mode": "best_stage",
        "status": "no_valid_candidate",
        "final_count": 0,
    }
    if sources:
        source = sources[0]
        candidate_id = str(source.get("candidate_id") or "")
        original = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("candidate_id") or "") == candidate_id
            ),
            {},
        )
        artifact = artifact_path or "submission.csv"
        expected_sha = str(original.get("artifact_sha") or "").strip().lower()
        if not expected_sha and Path(artifact).suffix.lower() == ".csv":
            expected_sha = str(original.get("submission_sha") or "").strip().lower()
        final_dir = finals_dir / "final_00"
        final_artifact = final_dir / artifact
        actual_sha = ""
        try:
            if final_artifact.is_file():
                actual_sha = _sha256_file(final_artifact)
        except OSError:
            actual_sha = ""
        if not actual_sha:
            manifest["status"] = "missing_artifact"
        elif not expected_sha or actual_sha != expected_sha:
            shutil.rmtree(final_dir, ignore_errors=True)
            manifest.update(
                {
                    "status": "artifact_sha_mismatch",
                    "expected_artifact_sha": expected_sha,
                    "actual_artifact_sha": actual_sha,
                }
            )
        else:
            manifest.update(
                {
                    "status": "success",
                    "final_count": 1,
                    "artifact_path": artifact,
                    "artifact_sha": actual_sha,
                    "final_dir": str(final_dir),
                    "selected_candidate": source,
                }
            )
    manifest_path = merge_dir / "best_stage_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest
