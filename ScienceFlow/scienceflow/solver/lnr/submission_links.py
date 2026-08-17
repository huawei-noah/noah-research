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
import time
from pathlib import Path
from typing import Any, Iterable


def _cleanup_previous_links(submission_dir: Path) -> None:
    index_path = submission_dir / ".scienceflow_submission_links.json"
    payload: dict[str, Any] = {}
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    indexed = [
        str(row.get("path") or "").strip()
        for row in payload.get("links") or []
        if str(row.get("path") or "").strip()
    ]
    for rel in (*indexed, "submission.csv", "selected_submission.csv"):
        path = submission_dir / rel
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
        except OSError:
            continue
    for rel_dir in ("candidates", "finals", "selected_artifact", "workers"):
        root = submission_dir / rel_dir
        if not root.is_dir():
            continue
        for path in sorted(
            (p for p in root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
            except OSError:
                continue
        try:
            root.rmdir()
        except OSError:
            continue


def _relative_symlink(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        return None
    source = src.resolve(strict=False)
    rel_target = os.path.relpath(source, start=dst.parent.resolve(strict=False))
    dst.symlink_to(rel_target)
    return {
        "target": rel_target,
        "source": str(source),
        "kind": "symlink",
    }


def _submission_link_relpath(dst: Path, submission_dir: Path) -> str:
    try:
        return str(dst.relative_to(submission_dir))
    except ValueError:
        return dst.name


def _safe_worker_name(worker_root: Path) -> str:
    name = worker_root.name.strip()
    return name or "worker"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_worker_workspace_artifacts(
    *,
    worker_roots: Iterable[Path],
    candidates: Iterable[dict[str, Any]],
    artifact_path: str,
) -> list[dict[str, str]]:
    """Replace completed worker artifacts with links to identical stage snapshots."""

    artifact_rel = Path(artifact_path or "submission.csv")
    if artifact_rel.is_absolute() or ".." in artifact_rel.parts:
        return []

    sources_by_worker: dict[str, list[Path]] = {}
    for candidate in candidates:
        worker_id = str(candidate.get("worker_id") or "").strip().lower()
        if not worker_id:
            worker_id = (
                str(candidate.get("candidate_id") or "")
                .partition(":")[0]
                .strip()
                .lower()
            )
        snapshot_path = str(candidate.get("snapshot_path") or "").strip()
        if not worker_id or not snapshot_path:
            continue
        source = Path(snapshot_path) / artifact_rel
        try:
            if source.is_file():
                sources_by_worker.setdefault(worker_id, []).append(
                    source.resolve(strict=True)
                )
        except OSError:
            continue

    canonicalized: list[dict[str, str]] = []
    for raw_worker_root in worker_roots:
        worker_root = Path(raw_worker_root)
        workspace_artifact = worker_root / "workspace" / artifact_rel
        if workspace_artifact.is_symlink() or not workspace_artifact.is_file():
            continue
        worker_id = _safe_worker_name(worker_root).lower()
        try:
            workspace_sha = _sha256_file(workspace_artifact)
        except OSError:
            continue
        for source in sources_by_worker.get(worker_id, []):
            try:
                if _sha256_file(source) != workspace_sha:
                    continue
                record = _relative_symlink(source, workspace_artifact)
            except OSError:
                continue
            if record is not None:
                canonicalized.append(
                    {
                        "worker_id": worker_id.upper(),
                        "path": str(workspace_artifact),
                        "source": str(source),
                        "target": str(record["target"]),
                    }
                )
            break
    return canonicalized


def refresh_submission_links(
    *,
    submission_dir: Path,
    artifact_path: str,
    merge_dir: Path | None = None,
    worker_roots: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    """Expose LNR submission artifacts under one stable submissions/ directory."""
    submission_dir = Path(submission_dir).expanduser().resolve(strict=False)
    submission_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_previous_links(submission_dir)

    artifact_rel = Path(artifact_path or "submission.csv")
    artifact_name = artifact_rel.name or "submission.csv"
    suffix = Path(artifact_name).suffix or ".artifact"
    links: list[dict[str, Any]] = []

    def add(src: Path, rel_dst: str, *, group: str) -> None:
        dst = submission_dir / rel_dst
        record = _relative_symlink(src, dst)
        if record is not None:
            record["path"] = _submission_link_relpath(dst, submission_dir)
            record["group"] = group
            links.append(record)

    if merge_dir is not None:
        merge_root = Path(merge_dir)
        for candidate_dir in sorted((merge_root / "candidates").glob("*")):
            if candidate_dir.is_dir():
                add(
                    candidate_dir / artifact_rel,
                    f"candidates/{candidate_dir.name}{suffix}",
                    group="merge_candidate",
                )

        for final_dir in sorted((merge_root / "finals").glob("final_*")):
            if final_dir.is_dir():
                add(
                    final_dir / artifact_rel,
                    f"finals/{final_dir.name}{suffix}",
                    group="merge_final",
                )

    seen_workers: set[Path] = set()
    for raw_worker_root in worker_roots:
        worker_root = Path(raw_worker_root)
        worker_key = _safe_worker_name(worker_root)
        resolved = worker_root.resolve(strict=False)
        if resolved in seen_workers:
            continue
        seen_workers.add(resolved)
        add(
            worker_root / "workspace" / artifact_rel,
            f"workers/{worker_key}/workspace{suffix}",
            group="worker_workspace",
        )
        snapshots_dir = worker_root / "snapshots"
        for snapshot_dir in sorted(snapshots_dir.glob("*")):
            if snapshot_dir.is_dir():
                add(
                    snapshot_dir / artifact_rel,
                    f"workers/{worker_key}/snapshots/{snapshot_dir.name}{suffix}",
                    group="worker_snapshot",
                )

    index = {
        "artifact_path": artifact_path or "submission.csv",
        "created_at": time.time(),
        "merge_dir": str(Path(merge_dir).resolve(strict=False))
        if merge_dir is not None
        else "",
        "links": links,
    }
    (submission_dir / ".scienceflow_submission_links.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return links
