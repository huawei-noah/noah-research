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

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from scienceflow.solver.lnr.global_merge.candidate_evidence import (
    prepare_candidates,
    select_diverse_candidates,
)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_candidate_dir_name(candidate_id: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("-", str(candidate_id or "").strip()).strip(".-")
    return cleaned or "candidate"


def _link_if_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        return False
    source = src.resolve(strict=True)
    dst.symlink_to(os.path.relpath(source, start=dst.parent.resolve(strict=False)))
    return True


def _merge_payload_files(
    snapshot: Path,
    *,
    max_file_bytes: int,
    remaining_bytes: int,
) -> list[Path]:
    payload_root = snapshot / "merge_payload"
    if not payload_root.is_dir():
        return []
    selected: list[Path] = []
    used = 0
    for path in sorted(p for p in payload_root.rglob("*") if p.is_file()):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes or used + size > remaining_bytes:
            continue
        selected.append(path)
        used += size
    return selected


def candidate_artifact_source(candidate: dict[str, Any], *, artifact_path: str) -> Path | None:
    explicit = str(candidate.get("artifact_source") or "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    snapshot_text = str(candidate.get("snapshot_path") or "").strip()
    if not snapshot_text:
        return None
    snapshot = Path(snapshot_text)
    for rel in (str(candidate.get("artifact_path") or "").strip(), artifact_path, "submission.csv"):
        if not rel:
            continue
        path = snapshot / rel
        if path.is_file():
            return path
    return None


def pack_candidates(
    *,
    merge_dir: Path,
    merge_workspace: Path,
    candidates: list[dict[str, Any]],
    artifact_path: str,
    ledger_filename: str,
    max_candidates: int = 24,
    max_prediction_file_bytes: int = 536_870_912,
    max_prediction_total_bytes: int = 2_147_483_648,
) -> list[dict[str, Any]]:
    packed_root = merge_dir / "candidates"
    workspace_root = merge_workspace / "candidates"
    for root in (packed_root, workspace_root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    packed: list[dict[str, Any]] = []
    prediction_bytes = 0
    prepared = prepare_candidates(candidates, artifact_path=artifact_path)
    selected = select_diverse_candidates(prepared, max_candidates=max_candidates)
    for candidate in selected:
        artifact_src = candidate_artifact_source(candidate, artifact_path=artifact_path)
        if artifact_src is None:
            continue
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{len(packed):02d}")
        dir_name = safe_candidate_dir_name(candidate_id)
        dst_dir = packed_root / dir_name
        ws_dir = workspace_root / dir_name
        rel_artifact = str(candidate.get("artifact_path") or artifact_path or artifact_src.name)
        artifact_dst = dst_dir / rel_artifact
        if not _link_if_file(artifact_src, artifact_dst):
            continue
        if not _link_if_file(artifact_src, ws_dir / rel_artifact):
            continue
        snapshot = Path(str(candidate.get("snapshot_path") or ""))
        for name in ("solution.py", ledger_filename):
            _link_if_file(snapshot / name, dst_dir / name)
            _link_if_file(snapshot / name, ws_dir / name)
        payload_files = _merge_payload_files(
            snapshot,
            max_file_bytes=max(0, int(max_prediction_file_bytes)),
            remaining_bytes=max(0, int(max_prediction_total_bytes) - prediction_bytes),
        )
        payload_relpaths: list[str] = []
        for payload_src in payload_files:
            rel = payload_src.relative_to(snapshot)
            if not _link_if_file(payload_src, dst_dir / rel):
                continue
            _link_if_file(payload_src, ws_dir / rel)
            payload_relpaths.append(str(rel))
            prediction_bytes += payload_src.stat().st_size
        meta = dict(candidate)
        meta.update(
            {
                "candidate_dir": dir_name,
                "packed_artifact": str(artifact_dst.relative_to(dst_dir)),
                "source_artifact": str(artifact_src),
                "merge_payload_files": payload_relpaths,
            }
        )
        metadata_text = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (dst_dir / "metadata.json").write_text(metadata_text, encoding="utf-8")
        (ws_dir / "metadata.json").write_text(metadata_text, encoding="utf-8")
        packed.append(meta)

    index = {
        "artifact_path": artifact_path,
        "candidate_count": len(packed),
        "candidates": packed,
        "prediction_payload_bytes": prediction_bytes,
    }
    text = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (packed_root / "candidate_index.json").write_text(text, encoding="utf-8")
    (workspace_root / "candidate_index.json").write_text(text, encoding="utf-8")
    return packed
