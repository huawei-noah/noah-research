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

import errno
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterable


_TRANSIENT_REMOVE_ERRNOS = {
    errno.ENOENT,
    errno.ENOTEMPTY,
    errno.EBUSY,
    errno.EAGAIN,
}


def clear_directory_contents(
    root: Path,
    *,
    preserve_paths: Iterable[Path] = (),
    trash_dir: Path | None = None,
) -> None:
    """Clear a directory while preserving live runtime paths, including nested ones."""
    if not root.exists():
        return
    preserved = tuple(path.resolve(strict=False) for path in preserve_paths)
    trash = trash_dir.resolve(strict=False) if trash_dir is not None else None

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        child_resolved = child.resolve(strict=False)
        if _is_preserved(child_resolved, preserved):
            continue
        if trash is not None and (child_resolved == trash or _is_relative_to(child_resolved, trash)):
            continue
        if child.is_dir() and not child.is_symlink() and _contains_preserved(child_resolved, preserved):
            clear_directory_contents(child, preserve_paths=preserved, trash_dir=trash)
            continue
        remove_path_for_restore(child, trash_dir=trash)


def remove_path_for_restore(path: Path, *, trash_dir: Path | None = None) -> None:
    """Remove a restore-obsolete path; rename to trash on transient filesystem races."""
    if not path.exists() and not path.is_symlink():
        return
    last_error: OSError | None = None
    for _ in range(3):
        try:
            _remove_once(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if exc.errno not in _TRANSIENT_REMOVE_ERRNOS:
                break
            time.sleep(0.05)

    if trash_dir is not None:
        try:
            _move_to_trash(path, trash_dir)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error


def _remove_once(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _move_to_trash(path: Path, trash_dir: Path) -> None:
    trash_dir.mkdir(parents=True, exist_ok=True)
    target = trash_dir / f"{path.name}.{uuid.uuid4().hex}"
    path.rename(target)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
    elif target.exists() or target.is_symlink():
        try:
            target.unlink()
        except OSError:
            pass


def _is_preserved(path: Path, preserved: tuple[Path, ...]) -> bool:
    return any(path == root or _is_relative_to(path, root) for root in preserved)


def _contains_preserved(path: Path, preserved: tuple[Path, ...]) -> bool:
    return any(_is_relative_to(root, path) for root in preserved)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
