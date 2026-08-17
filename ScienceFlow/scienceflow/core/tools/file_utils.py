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

"""Path sandbox, syntax validation, atomic file writes, summaries."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from pathlib import PurePosixPath


def _normalize_extra_roots(extra_roots: Sequence[str | Path] | None) -> tuple[Path, ...]:
    out: list[Path] = []
    for p in extra_roots or ():
        s = str(p).strip()
        if not s:
            continue
        out.append(Path(s).expanduser().resolve())
    return tuple(out)


def normalize_denied_prefixes(prefixes: Sequence[str | Path] | None) -> tuple[PurePosixPath, ...]:
    out: list[PurePosixPath] = []
    for raw in prefixes or ():
        text = str(raw or "").strip().replace("\\", "/")
        if not text:
            continue
        text = text.lstrip("/")
        parts = [part for part in text.split("/") if part not in ("", ".")]
        if not parts or any(part == ".." for part in parts):
            continue
        path = PurePosixPath(*parts)
        if path not in out:
            out.append(path)
    return tuple(out)


class PathGuard:
    """Restrict file paths to *workspace_dir* when *enabled*; otherwise resolve only.

    When *extra_roots* is non-empty, resolved paths may also lie under any of those
    absolute prefixes (e.g. symlink targets for large read-only datasets).
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        enabled: bool = True,
        extra_roots: Sequence[str | Path] | None = None,
        denied_prefixes: Sequence[str | Path] | None = None,
    ):
        self.root = Path(workspace_dir).resolve()
        self._enabled = enabled
        self._extra_roots = _normalize_extra_roots(extra_roots)
        self._denied_prefixes = normalize_denied_prefixes(denied_prefixes)

    def _under_any_extra(self, candidate: Path) -> bool:
        c = candidate.resolve()
        for prefix in self._extra_roots:
            try:
                c.relative_to(prefix)
                return True
            except ValueError:
                continue
        return False

    def resolve(self, path: str | Path) -> Path:
        """Resolve *path* relative to workspace; raise if outside root when sandbox *enabled*."""
        if isinstance(path, Path) and path.is_absolute():
            candidate = path.resolve()
        else:
            candidate = (self.root / path).resolve()
        if self._enabled:
            try:
                candidate.relative_to(self.root)
            except ValueError as e:
                if not self._under_any_extra(candidate):
                    raise ValueError(f"Path escapes workspace: {path!r}") from e
        if self.is_denied(candidate, requested=path):
            raise ValueError(f"Path is hidden from agent tools: {path!r}")
        return candidate

    def is_denied(self, path: str | Path, *, requested: str | Path | None = None) -> bool:
        if not self._denied_prefixes:
            return False
        if requested is not None and _logical_path_denied(requested, self._denied_prefixes):
            return True
        try:
            candidate = Path(path)
            resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
            rel = resolved.relative_to(self.root)
        except (OSError, ValueError):
            return False
        logical = PurePosixPath(*rel.parts) if rel.parts else PurePosixPath(".")
        return _posix_path_denied(logical, self._denied_prefixes)


def _logical_path_denied(
    value: str | Path,
    denied_prefixes: tuple[PurePosixPath, ...],
) -> bool:
    text = str(value or "").strip().replace("\\", "/")
    if not text or PurePosixPath(text).is_absolute():
        return False
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        return False
    return _posix_path_denied(PurePosixPath(*parts), denied_prefixes)


def _posix_path_denied(path: PurePosixPath, denied_prefixes: tuple[PurePosixPath, ...]) -> bool:
    parts = path.parts
    for prefix in denied_prefixes:
        pparts = prefix.parts
        if len(parts) >= len(pparts) and parts[: len(pparts)] == pparts:
            return True
    return False


def display_under_root(root: Path, resolved: Path) -> str:
    """Human-readable path for tool output; falls back to absolute if outside *root*."""
    try:
        return resolved.relative_to(root).as_posix() or "."
    except ValueError:
        return resolved.as_posix()


def normalize_display_path(path: str | Path | None) -> str:
    """Normalize a model-visible path without resolving symlinks.

    Tools still use :class:`PathGuard` and resolved filesystem paths for access
    checks. This helper is only for the text returned to the model, where a
    symlink target such as ``/work/.../prepared/train.csv`` should remain
    visible as the logical workspace path ``dataset/train.csv``.
    """
    s = str(path or ".").strip().replace("\\", "/")
    if not s:
        return "."
    if PurePosixPath(s).is_absolute():
        return PurePosixPath(s).as_posix()
    parts: list[str] = []
    for part in s.split("/"):
        if part in ("", "."):
            continue
        parts.append(part)
    return "/".join(parts) or "."


def _display_under_any_root(resolved: Path, roots: Sequence[Path]) -> str | None:
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix() or "."
        except ValueError:
            continue
    return None


def display_from_request(
    root: Path,
    resolved: Path,
    requested: str | Path | None,
    *,
    logical_paths: bool = True,
    display_root: str | Path | None = None,
) -> str:
    """Return the logical request path for model output, falling back to root-relative.

    ``requested`` is normally the workspace-relative path supplied to a tool.
    Using it keeps symlinked datasets and reference workspaces spatially stable
    in LLM-visible observations while preserving real resolved paths internally.
    """
    roots = [root]
    if display_root is not None:
        roots.insert(0, Path(display_root))
    if not logical_paths:
        return display_under_root(roots[0].resolve(), resolved)

    if requested is not None and Path(str(requested)).is_absolute():
        rel = _display_under_any_root(resolved, roots)
        if rel is not None:
            return rel
        return resolved.name or "."

    logical = normalize_display_path(requested)
    if logical != "." or normalize_display_path(root) == ".":
        return logical
    rel = _display_under_any_root(resolved, roots)
    if rel is not None:
        return rel
    return resolved.name or "."


def join_display_path(base: str | Path | None, child: str | Path | None = "") -> str:
    """Join logical display path fragments without introducing absolute paths."""
    b = normalize_display_path(base)
    c = normalize_display_path(child)
    if c == ".":
        return b
    if b == ".":
        return c
    return f"{b.rstrip('/')}/{c.lstrip('/')}"


def validate_syntax(path: Path, content: str) -> str | None:
    """Lightweight syntax check by extension. Returns error string or None."""
    ext = path.suffix.lower()
    if ext == ".py":
        try:
            compile(content, str(path), "exec")
        except SyntaxError as e:
            return f"Python syntax error: {e}"
    elif ext == ".json":
        import json

        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {e}"
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            return None
        try:
            yaml.safe_load(content)
        except Exception as e:
            return f"YAML parse error: {e}"
    return None


def atomic_write(path: Path, content: str, *, encoding: str = "utf-8") -> dict:
    """Write via temp file + os.replace. Return metadata dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    raw = content.encode(encoding)
    return {
        "path": str(path),
        "lines": len(content.splitlines()),
        "bytes": len(raw),
        "sha256_short": hashlib.sha256(raw).hexdigest()[:16],
    }


def file_summary(path: Path, max_lines: int = 200) -> str:
    """Return a short preview of a file (for diagnostics / LLM hints)."""
    if not path.is_file():
        return f"(not a file: {path})"
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    if total <= max_lines:
        body = "\n".join(lines)
    else:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines // 2) :]
        body = "\n".join(head) + f"\n... ({total} lines total) ...\n" + "\n".join(tail)
    return f"[{path.name}: {total} lines]\n{body}"
