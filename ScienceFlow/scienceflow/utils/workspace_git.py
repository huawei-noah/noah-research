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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import fnmatch
import hashlib
import os
import json
import re
import shutil
import subprocess
import time


DEFAULT_WORKSPACE_GIT_TRACK_GLOBS: tuple[str, ...] = ("*.py", "*.md")
WORKSPACE_GIT_CONTROL_EXCLUDES: tuple[str, ...] = (
    ".run_results.md",
    "run_results.md",
    "selected_run_results.md",
    "resource_feedback.md",
    ".agent_memory/**",
    ".memory/**",
    ".logs/**",
    "logs/**",
    "task_logs/**",
    "snapshots/**",
    ".snapshots/**",
    ".estra_archives/**",
    ".scienceflow_checkpoints/**",
    ".scienceflow_trials/**",
    "stage_memory/**",
    "submission_snapshots/**",
    "submission_history/**",
    "submissions/**",
    "cache/**",
    ".cache/**",
    "tmp/**",
    "__pycache__/**",
    "*.pyc",
    "*.log",
)

_GITIGNORE_START = "# >>> scienceflow workspace source tracking >>>"
_GITIGNORE_END = "# <<< scienceflow workspace source tracking <<<"


@dataclass(frozen=True)
class WorkspaceGitInitResult:
    enabled: bool
    ready: bool
    initialized: bool
    committed: bool
    message: str = ""


@dataclass(frozen=True)
class WorkspaceGitCheckpointResult:
    enabled: bool
    ready: bool
    committed: bool = False
    commit_sha: str = ""
    stage_id: str = ""
    submission_snapshot: str = ""
    ledger_path: str = ""
    source_changed: bool = False
    submission_changed: bool = False
    metric_value: float | None = None
    message: str = ""


@dataclass(frozen=True)
class CandidateArtifactArchiveResult:
    enabled: bool
    ready: bool
    archived: bool = False
    artifact_path: str = ""
    artifact_kind: str = ""
    artifact_sha256: str = ""
    snapshot_path: str = ""
    ledger_path: str = ""
    size_bytes: int = 0
    message: str = ""


def normalize_workspace_git_track_globs(value: Any) -> tuple[str, ...]:
    """Normalize YAML/string/list track patterns for workspace source git."""
    if value is None:
        return DEFAULT_WORKSPACE_GIT_TRACK_GLOBS
    if isinstance(value, str):
        parts: Iterable[Any] = value.replace(",", "\n").splitlines()
    elif isinstance(value, Iterable):
        parts = value
    else:
        return DEFAULT_WORKSPACE_GIT_TRACK_GLOBS

    globs: list[str] = []
    for item in parts:
        text = str(item or "").strip()
        if not text:
            continue
        if text.startswith("!"):
            text = text[1:].strip()
        if text and text not in globs:
            globs.append(text)
    return tuple(globs) or DEFAULT_WORKSPACE_GIT_TRACK_GLOBS


def render_workspace_gitignore(track_globs: Iterable[str] | None = None) -> str:
    globs = normalize_workspace_git_track_globs(track_globs)
    lines = [
        _GITIGNORE_START,
        "# Track source/docs only; keep datasets, model weights, logs, and submissions out of git.",
        "*",
        "!*/",
        "!.gitignore",
    ]
    for pattern in globs:
        lines.append(f"!{pattern}")
    lines.extend(
        [
            *WORKSPACE_GIT_CONTROL_EXCLUDES,
            "dataset/**",
            "artifacts/**",
            "submissions/**",
            "logs/**",
            ".logs/**",
            "catboost_info/**",
        ],
    )
    lines.append(_GITIGNORE_END)
    return "\n".join(lines) + "\n"


def _install_workspace_gitignore(workspace_dir: Path, track_globs: Iterable[str]) -> None:
    gitignore = workspace_dir / ".gitignore"
    block = render_workspace_gitignore(track_globs).rstrip("\n")
    if not gitignore.exists():
        gitignore.write_text(block + "\n", encoding="utf-8")
        return

    current = gitignore.read_text(encoding="utf-8")
    for start, end in ((_GITIGNORE_START, _GITIGNORE_END),):
        if start in current and end in current:
            before, rest = current.split(start, 1)
            _, after = rest.split(end, 1)
            updated = before.rstrip() + "\n" + block + after
            gitignore.write_text(updated.lstrip("\n"), encoding="utf-8")
            return

    sep = "" if current.endswith("\n") else "\n"
    gitignore.write_text(current + sep + "\n" + block + "\n", encoding="utf-8")


def _git(
    workspace_dir: Path,
    *args: str,
    timeout_sec: int = 20,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workspace_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=check,
    )


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _head_sha(workspace_dir: Path) -> str:
    proc = _git(workspace_dir, "rev-parse", "--verify", "HEAD")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _source_pathspec(track_globs: Iterable[str] | None) -> list[str]:
    globs = normalize_workspace_git_track_globs(track_globs)
    return [".gitignore", *globs]


def _is_control_source_path(path: str) -> bool:
    normalized = path.strip().lstrip("/")
    if not normalized:
        return False
    for pattern in WORKSPACE_GIT_CONTROL_EXCLUDES:
        if pattern.endswith("/**"):
            prefix = pattern[:-3].rstrip("/") + "/"
            if normalized.startswith(prefix):
                return True
        elif fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def _semantic_changed_paths(workspace_dir: Path, track_globs: Iterable[str] | None) -> list[str]:
    proc = _git(
        workspace_dir,
        "ls-files",
        "-m",
        "-o",
        "--exclude-standard",
        "--",
        *_source_pathspec(track_globs),
    )
    if proc.returncode != 0:
        return []
    paths: list[str] = []
    for raw in proc.stdout.splitlines():
        item = raw.strip()
        if item and not _is_control_source_path(item):
            paths.append(item)
    return paths


def workspace_source_changed(workspace_dir: str | Path, track_globs: Iterable[str] | None = None) -> bool:
    """Return whether semantic tracked source changed, excluding runtime control artifacts."""
    workspace = Path(workspace_dir).expanduser().resolve(strict=False)
    if not (workspace / ".git").is_dir():
        return False
    return bool(_semantic_changed_paths(workspace, track_globs))


def _last_metric_from_ml_run_results(workspace_dir: Path) -> float | None:
    path = workspace_dir / "ml_run_results.md"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        raw = line.strip()
        if not raw or "|" not in raw:
            continue
        lower = raw.lower()
        if "validation_metric" in lower or set(raw.replace("|", "").strip()) <= {"-", ":"}:
            continue
        parts = [p.strip().strip("`* ") for p in raw.strip("|").split("|")]
        candidates = []
        if len(parts) >= 5:
            candidates.append(parts[4])
        candidates.append(raw)
        for candidate in candidates:
            if candidate.lower() in {"", "null", "none", "nan"}:
                continue
            match = re.search(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", candidate)
            if not match:
                continue
            try:
                return float(match.group(0))
            except ValueError:
                continue
    return None


def _safe_metric_label(metric: float | None) -> str:
    if metric is None:
        return "metric_unknown"
    return ("metric_" + f"{metric:.6g}").replace("-", "m").replace(".", "p")


def _safe_stage_label(stage_id: str | None) -> str:
    raw = str(stage_id or "").strip().upper()
    if not raw:
        return ""
    label = re.sub(r"[^A-Z0-9_.-]+", "_", raw).strip("._-")
    return label[:32]


def _workspace_relpath(path: Path, workspace_dir: Path) -> str:
    return os.path.relpath(
        Path(path).resolve(strict=False),
        Path(workspace_dir).resolve(strict=False),
    )


def _checkpoint_ledger_path(workspace_dir: Path, checkpoint_dir: str | Path | None = None) -> Path:
    base = Path(checkpoint_dir) if checkpoint_dir is not None else workspace_dir / ".scienceflow_checkpoints"
    return base / "ledger.jsonl"


def _submission_snapshot_for_sha(
    workspace_dir: Path,
    sha: str,
    stage_id: str = "",
    *,
    checkpoint_dir: str | Path | None = None,
) -> str:
    ledger = _checkpoint_ledger_path(workspace_dir, checkpoint_dir)
    if not ledger.is_file():
        return ""
    stage_label = _safe_stage_label(stage_id)
    try:
        fallback = ""
        for line in ledger.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("submission_sha256") == sha and row.get("submission_snapshot"):
                snap = workspace_dir / str(row["submission_snapshot"])
                if snap.is_file():
                    if not stage_label:
                        return str(row["submission_snapshot"])
                    if row.get("stage_id") == stage_label:
                        return str(row["submission_snapshot"])
                    if not fallback:
                        fallback = str(row["submission_snapshot"])
        if not stage_label:
            return fallback
    except OSError:
        return ""
    return ""


def _submission_sha_seen(
    workspace_dir: Path,
    sha: str,
    *,
    checkpoint_dir: str | Path | None = None,
) -> bool:
    return bool(_submission_snapshot_for_sha(workspace_dir, sha, checkpoint_dir=checkpoint_dir))


def _next_snapshot_path(
    workspace_dir: Path,
    sha: str,
    metric: float | None,
    stage_id: str = "",
    submission_snapshot_dir: str | Path | None = None,
) -> Path:
    snapshots = Path(submission_snapshot_dir) if submission_snapshot_dir is not None else workspace_dir / "submission_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in snapshots.glob("iter_*.csv") if p.is_file())
    seq = len(existing) + 1
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    label = _safe_metric_label(metric)
    stage_label = _safe_stage_label(stage_id)
    stage_part = f"_{stage_label.lower()}" if stage_label else ""
    while True:
        name = f"iter_{seq:04d}{stage_part}_{timestamp}_{label}_sha_{sha[:12]}.csv"
        path = snapshots / name
        if not path.exists():
            return path
        seq += 1


def _safe_artifact_source(workspace_dir: Path, artifact_path: str) -> tuple[Path | None, str]:
    raw = str(artifact_path or "").replace("\\", "/").strip()
    rel = Path(raw)
    if not raw or rel.is_absolute() or ".." in rel.parts:
        return None, "candidate artifact path must be workspace-relative"
    try:
        source = (workspace_dir / rel).resolve(strict=False)
        source.relative_to(workspace_dir)
    except (OSError, ValueError):
        return None, "candidate artifact resolves outside workspace"
    if not source.is_file():
        return None, "candidate artifact is not a regular file"
    return source, rel.as_posix()


def _artifact_snapshot_for_sha(workspace_dir: Path, ledger: Path, sha: str) -> str:
    if not ledger.is_file():
        return ""
    try:
        for line in ledger.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("artifact_sha256") != sha or not row.get("snapshot_path"):
                continue
            snapshot = workspace_dir / str(row["snapshot_path"])
            if snapshot.is_file() and _sha256_file(snapshot) == sha:
                return str(row["snapshot_path"])
    except OSError:
        return ""
    return ""


def _next_artifact_snapshot_path(snapshot_dir: Path, source: Path, sha: str) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in snapshot_dir.glob("iter_*") if path.is_file())
    seq = len(existing) + 1
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._-") or "artifact"
    suffix = "".join(source.suffixes)[-32:]
    while True:
        path = snapshot_dir / f"iter_{seq:04d}_{timestamp}_{stem}_sha_{sha[:12]}{suffix}"
        if not path.exists():
            return path
        seq += 1


def archive_workspace_candidate_artifact(
    workspace_dir: str | Path,
    *,
    artifact_path: str,
    artifact_kind: str = "",
    snapshot_dir: str | Path,
    ledger_path: str | Path,
    trigger: str = "",
    tool_error: bool = False,
) -> CandidateArtifactArchiveResult:
    """Archive each stable candidate artifact revision independently of stage selection.

    The configured candidate may be an MLE submission CSV or a task-specific
    optimization artifact such as JSON. Archives are content-addressed and do
    not imply evaluator validity or selection eligibility.
    """
    workspace = Path(workspace_dir).expanduser().resolve(strict=False)
    source, rel = _safe_artifact_source(workspace, artifact_path)
    if source is None:
        return CandidateArtifactArchiveResult(
            enabled=True,
            ready=True,
            artifact_path=str(artifact_path or ""),
            artifact_kind=str(artifact_kind or ""),
            message=rel,
        )

    snapshots = Path(snapshot_dir).expanduser().resolve(strict=False)
    ledger = Path(ledger_path).expanduser().resolve(strict=False)
    try:
        source_sha = _sha256_file(source)
        if not source_sha:
            raise OSError("could not hash candidate artifact")
        existing = _artifact_snapshot_for_sha(workspace, ledger, source_sha)
        if existing:
            return CandidateArtifactArchiveResult(
                enabled=True,
                ready=True,
                artifact_path=rel,
                artifact_kind=str(artifact_kind or ""),
                artifact_sha256=source_sha,
                snapshot_path=existing,
                ledger_path=_workspace_relpath(ledger, workspace),
                size_bytes=source.stat().st_size,
            )

        snapshot = _next_artifact_snapshot_path(snapshots, source, source_sha)
        temp = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temp)
            if _sha256_file(temp) != source_sha or _sha256_file(source) != source_sha:
                raise OSError("candidate artifact changed while being archived")
            os.replace(temp, snapshot)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

        snapshot_rel = _workspace_relpath(snapshot, workspace)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "capture_type": "candidate_artifact_persisted",
            "trigger": str(trigger or ""),
            "tool_error": bool(tool_error),
            "artifact_path": rel,
            "artifact_kind": str(artifact_kind or ""),
            "artifact_sha256": source_sha,
            "size_bytes": snapshot.stat().st_size,
            "snapshot_path": snapshot_rel,
            "selection_eligible": None,
            "metric_value": None,
        }
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return CandidateArtifactArchiveResult(
            enabled=True,
            ready=True,
            archived=True,
            artifact_path=rel,
            artifact_kind=str(artifact_kind or ""),
            artifact_sha256=source_sha,
            snapshot_path=snapshot_rel,
            ledger_path=_workspace_relpath(ledger, workspace),
            size_bytes=snapshot.stat().st_size,
        )
    except OSError as exc:
        return CandidateArtifactArchiveResult(
            enabled=True,
            ready=False,
            artifact_path=rel,
            artifact_kind=str(artifact_kind or ""),
            message=str(exc),
        )


def auto_checkpoint_workspace_source(
    workspace_dir: str | Path,
    *,
    enabled: bool = True,
    track_globs: Iterable[str] | None = None,
    tool_name: str = "",
    tool_error: bool = False,
    metric_value_override: float | None = None,
    stage_id: str = "",
    submission_snapshot_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> WorkspaceGitCheckpointResult:
    """Lightweight REPL checkpoint hook.

    The hook is intentionally outside the agent's workflow: it commits tracked
    source/docs when they changed, snapshots ``submission.csv`` separately, and
    writes a JSONL ledger binding source commit, metric, and submission hash.
    """
    if not enabled:
        return WorkspaceGitCheckpointResult(enabled=False, ready=False)
    if tool_error:
        return WorkspaceGitCheckpointResult(enabled=True, ready=True, message="tool_error")
    if shutil.which("git") is None:
        return WorkspaceGitCheckpointResult(
            enabled=True,
            ready=False,
            message="git executable not found",
        )

    workspace = Path(workspace_dir).expanduser().resolve(strict=False)
    if not (workspace / ".git").is_dir():
        return WorkspaceGitCheckpointResult(
            enabled=True,
            ready=False,
            message="workspace git is not initialized",
        )

    try:
        stage_label = _safe_stage_label(stage_id)
        changed_paths = _semantic_changed_paths(workspace, track_globs)
        source_changed = bool(changed_paths)
        metric_value = metric_value_override
        if metric_value is None:
            metric_value = _last_metric_from_ml_run_results(workspace)
        commit_sha = _head_sha(workspace)
        committed = False

        if source_changed:
            _git(workspace, "add", "-A", "--", *changed_paths)
            staged = _git(workspace, "diff", "--cached", "--quiet")
            if staged.returncode != 0:
                metric_label = _safe_metric_label(metric_value)
                tool_label = (tool_name or "tool").strip()[:24] or "tool"
                stage_part = f" {stage_label}" if stage_label else ""
                msg = f"ckpt:{stage_part} {metric_label} after {tool_label}"
                commit = _git(workspace, "commit", "-q", "-m", msg, timeout_sec=30)
                if commit.returncode != 0:
                    return WorkspaceGitCheckpointResult(
                        enabled=True,
                        ready=True,
                        stage_id=stage_label,
                        source_changed=True,
                        metric_value=metric_value,
                        message=(commit.stderr or commit.stdout).strip(),
                    )
                committed = True
                commit_sha = _head_sha(workspace)

        submission_snapshot = ""
        submission_changed = False
        submission = workspace / "submission.csv"
        submission_sha = _sha256_file(submission) if submission.is_file() else None
        if submission_sha:
            existing_snapshot = _submission_snapshot_for_sha(
                workspace,
                submission_sha,
                stage_label,
                checkpoint_dir=checkpoint_dir,
            )
            if existing_snapshot:
                submission_snapshot = existing_snapshot
            else:
                snap_path = _next_snapshot_path(
                    workspace,
                    submission_sha,
                    metric_value,
                    stage_label,
                    submission_snapshot_dir=submission_snapshot_dir,
                )
                shutil.copy2(submission, snap_path)
                submission_snapshot = _workspace_relpath(snap_path, workspace)
                submission_changed = True

        ledger_path = ""
        if committed or submission_changed:
            ledger = _checkpoint_ledger_path(workspace, checkpoint_dir)
            ledger.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool_name": tool_name,
                "stage_id": stage_label,
                "commit": commit_sha,
                "committed": committed,
                "source_changed": source_changed,
                "metric_value": metric_value,
                "submission_snapshot": submission_snapshot,
                "submission_sha256": submission_sha,
            }
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            ledger_path = _workspace_relpath(ledger, workspace)

        return WorkspaceGitCheckpointResult(
            enabled=True,
            ready=True,
            committed=committed,
            commit_sha=commit_sha,
            stage_id=stage_label,
            submission_snapshot=submission_snapshot,
            ledger_path=ledger_path,
            source_changed=source_changed,
            submission_changed=submission_changed,
            metric_value=metric_value,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return WorkspaceGitCheckpointResult(
            enabled=True,
            ready=False,
            message=str(exc),
        )


def ensure_workspace_source_git(
    workspace_dir: str | Path,
    *,
    enabled: bool = True,
    track_globs: Iterable[str] | None = None,
    initial_commit: bool = True,
    user_name: str = "ScienceFlow REPL",
    user_email: str = "scienceflow-repl@local",
) -> WorkspaceGitInitResult:
    """Initialize a workspace-local git repo that tracks only source/doc files."""
    if not enabled:
        return WorkspaceGitInitResult(enabled=False, ready=False, initialized=False, committed=False)
    if shutil.which("git") is None:
        return WorkspaceGitInitResult(
            enabled=True,
            ready=False,
            initialized=False,
            committed=False,
            message="git executable not found",
        )

    workspace = Path(workspace_dir).expanduser().resolve(strict=False)
    workspace.mkdir(parents=True, exist_ok=True)
    globs = normalize_workspace_git_track_globs(track_globs)
    had_git = (workspace / ".git").exists()

    try:
        if not had_git:
            init = _git(workspace, "init", "-q")
            if init.returncode != 0:
                return WorkspaceGitInitResult(
                    enabled=True,
                    ready=False,
                    initialized=False,
                    committed=False,
                    message=(init.stderr or init.stdout).strip(),
                )

        _git(workspace, "config", "user.name", user_name)
        _git(workspace, "config", "user.email", user_email)
        _install_workspace_gitignore(workspace, globs)

        has_head = _git(workspace, "rev-parse", "--verify", "HEAD")
        committed = False
        if initial_commit and has_head.returncode != 0:
            initial_paths = _semantic_changed_paths(workspace, globs)
            if initial_paths:
                _git(workspace, "add", "-A", "--", *initial_paths)
            staged = _git(workspace, "diff", "--cached", "--quiet")
            if staged.returncode != 0:
                commit = _git(
                    workspace,
                    "commit",
                    "-q",
                    "-m",
                    "Initialize workspace source tracking",
                    timeout_sec=30,
                )
                committed = commit.returncode == 0
                if commit.returncode != 0:
                    return WorkspaceGitInitResult(
                        enabled=True,
                        ready=True,
                        initialized=not had_git,
                        committed=False,
                        message=(commit.stderr or commit.stdout).strip(),
                    )

        return WorkspaceGitInitResult(
            enabled=True,
            ready=True,
            initialized=not had_git,
            committed=committed,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return WorkspaceGitInitResult(
            enabled=True,
            ready=False,
            initialized=not had_git,
            committed=False,
            message=str(exc),
        )
