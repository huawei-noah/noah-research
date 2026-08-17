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

"""Shadow workspace helpers for safe observation trials.

The helpers are intentionally not wired into BashTool yet. Phase P5 provides
the isolation primitive required before any future observation kill + replay
path can run without polluting the real node workspace.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_PROTECTED_GLOBS: tuple[str, ...] = (
    "submission.csv",
    "result.md",
    "*.ckpt",
    "*.joblib",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.pt",
    "*.pth",
    "*logit*",
    "*pred*.csv",
    "checkpoints/**",
    "models/**",
)


def _safe_job_segment(job_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in job_id)
    return safe[:80] or "job"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _file_digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_digest(path: Path, st: os.stat_result) -> str:
    h = hashlib.blake2b(digest_size=16)
    if stat.S_ISLNK(st.st_mode):
        try:
            text = "symlink:" + os.readlink(path)
        except OSError:
            text = "symlink:<unreadable>"
    else:
        text = f"metadata:{stat.S_IMODE(st.st_mode)}:{int(st.st_size)}:{int(st.st_mtime_ns)}"
    h.update(text.encode("utf-8", errors="surrogateescape"))
    return h.hexdigest()


def _under_named_root(path: Path, root: Path, names: tuple[str, ...]) -> bool:
    if not names:
        return False
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return bool(rel.parts and rel.parts[0] in names)


def _fingerprint(
    paths: list[Path],
    root: Path,
    *,
    metadata_only_roots: tuple[str, ...] = (),
) -> dict[str, tuple[int, int, str]]:
    out: dict[str, tuple[int, int, str]] = {}
    for path in paths:
        try:
            st = path.lstat()
            is_link = stat.S_ISLNK(st.st_mode)
            if not (is_link or stat.S_ISREG(st.st_mode)):
                continue
            rel = path.relative_to(root).as_posix()
            if is_link or _under_named_root(path, root, metadata_only_roots):
                digest = _metadata_digest(path, st)
            else:
                digest = _file_digest(path)
            out[rel] = (int(st.st_size), int(st.st_mtime_ns), digest)
        except OSError:
            continue
    return out


def _mirror_readonly_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        try:
            child_stat = child.lstat()
        except OSError:
            continue
        if stat.S_ISDIR(child_stat.st_mode):
            _mirror_readonly_dir(child, target)
        elif stat.S_ISREG(child_stat.st_mode) or stat.S_ISLNK(child_stat.st_mode):
            target.symlink_to(child)


def _is_dir_no_follow(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _is_fingerprintable_path(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISLNK(mode)


def _collect_protected_paths(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    trials_root = root / ".scienceflow_trials"
    for pattern in patterns:
        for path in root.glob(pattern):
            if _is_relative_to(path, trials_root):
                continue
            if _is_dir_no_follow(path):
                paths.extend(p for p in path.rglob("*") if _is_fingerprintable_path(p))
            else:
                paths.append(path)
    return sorted(set(paths))


@dataclass
class ShadowWorkspace:
    real_workspace: Path
    shadow_workspace: Path
    job_id: str
    protected_globs: tuple[str, ...] = _DEFAULT_PROTECTED_GLOBS
    metadata_only_roots: tuple[str, ...] = ()
    protected_before: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    shadow_protected_before: dict[str, tuple[int, int, str]] = field(default_factory=dict)

    @staticmethod
    def _diff_fingerprints(
        before: dict[str, tuple[int, int, str]],
        after: dict[str, tuple[int, int, str]],
    ) -> list[str]:
        changed: list[str] = []
        for rel, before_sig in before.items():
            if rel not in after:
                changed.append(f"removed:{rel}")
            elif after[rel] != before_sig:
                changed.append(f"modified:{rel}")
        for rel in after:
            if rel not in before:
                changed.append(f"added:{rel}")
        return sorted(changed)

    def validate_real_workspace_unchanged(self) -> tuple[bool, list[str]]:
        after = _fingerprint(
            _collect_protected_paths(self.real_workspace, self.protected_globs),
            self.real_workspace,
            metadata_only_roots=self.metadata_only_roots,
        )
        changed = self._diff_fingerprints(self.protected_before, after)
        return not changed, changed

    def validate_shadow_protected_unchanged(self) -> tuple[bool, list[str]]:
        after = _fingerprint(
            _collect_protected_paths(self.shadow_workspace, self.protected_globs),
            self.shadow_workspace,
            metadata_only_roots=self.metadata_only_roots,
        )
        changed = self._diff_fingerprints(self.shadow_protected_before, after)
        return not changed, changed

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self.shadow_workspace)
        except FileNotFoundError:
            return
        try:
            self.shadow_workspace.parent.rmdir()
        except OSError:
            return


class ShadowWorkspaceManager:
    def __init__(
        self,
        *,
        copy_file_max_bytes: int = 5 * 1024 * 1024,
        readonly_dir_names: tuple[str, ...] = (),
        protected_globs: tuple[str, ...] = _DEFAULT_PROTECTED_GLOBS,
    ) -> None:
        self.copy_file_max_bytes = max(0, int(copy_file_max_bytes))
        self.readonly_dir_names = tuple(readonly_dir_names)
        self.protected_globs = tuple(protected_globs)

    def create(self, workspace_dir: str | Path, job_id: str) -> ShadowWorkspace:
        real = Path(workspace_dir).resolve()
        if not real.is_dir():
            raise FileNotFoundError(f"workspace does not exist: {real}")
        trial_root = real / ".scienceflow_trials"
        shadow = trial_root / f"run_{_safe_job_segment(job_id)}"
        protected_before = _fingerprint(
            _collect_protected_paths(real, self.protected_globs),
            real,
            metadata_only_roots=self.readonly_dir_names,
        )

        if shadow.exists():
            shutil.rmtree(shadow)
        shadow.mkdir(parents=True, exist_ok=True)

        readonly_names = set(self.readonly_dir_names)
        for child in real.iterdir():
            if child.name == ".scienceflow_trials":
                continue
            target = shadow / child.name
            if child.is_dir():
                if child.name in readonly_names:
                    _mirror_readonly_dir(child, target)
                else:
                    target.mkdir(exist_ok=True)
                continue
            if not child.is_file():
                continue
            try:
                size = child.stat().st_size
            except OSError:
                continue
            if size <= self.copy_file_max_bytes:
                shutil.copy2(child, target)
            else:
                target.symlink_to(child)

        shadow_protected_before = _fingerprint(
            _collect_protected_paths(shadow, self.protected_globs),
            shadow,
            metadata_only_roots=self.readonly_dir_names,
        )
        return ShadowWorkspace(
            real_workspace=real,
            shadow_workspace=shadow,
            job_id=job_id,
            protected_globs=self.protected_globs,
            metadata_only_roots=self.readonly_dir_names,
            protected_before=protected_before,
            shadow_protected_before=shadow_protected_before,
        )
