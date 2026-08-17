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

"""Path helpers for hidden per-node runtime metadata."""

from __future__ import annotations

from pathlib import Path

NODE_LOGS_DIR = ".logs"
LEGACY_NODE_LOGS_DIR = "logs"
NODE_CONTEXT_JSON = "context.json"


def node_logs_dir(node_dir: Path, *, create: bool = False) -> Path:
    """Return the hidden per-node log directory for new writes."""
    path = Path(node_dir) / NODE_LOGS_DIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_node_logs_dir(node_dir: Path) -> Path:
    return Path(node_dir) / LEGACY_NODE_LOGS_DIR


def existing_node_logs_dir(node_dir: Path) -> Path:
    """Return hidden logs if present, otherwise legacy ``logs`` if present."""
    node_dir = Path(node_dir)
    hidden = node_dir / NODE_LOGS_DIR
    if hidden.exists():
        return hidden
    legacy = node_dir / LEGACY_NODE_LOGS_DIR
    if legacy.exists():
        return legacy
    return hidden


def node_log_path(node_dir: Path, *parts: str, create_parent: bool = False) -> Path:
    """Return a path under hidden ``.logs`` for new writes."""
    path = node_logs_dir(Path(node_dir), create=create_parent).joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_node_log_path(node_dir: Path, *parts: str) -> Path:
    """Return an existing hidden/legacy log path, preferring hidden ``.logs``."""
    node_dir = Path(node_dir)
    hidden = (node_dir / NODE_LOGS_DIR).joinpath(*parts)
    if hidden.exists():
        return hidden
    legacy = (node_dir / LEGACY_NODE_LOGS_DIR).joinpath(*parts)
    if legacy.exists():
        return legacy
    return hidden


def node_context_path(node_dir: Path, *, create_parent: bool = False) -> Path:
    path = node_log_path(Path(node_dir), NODE_CONTEXT_JSON, create_parent=create_parent)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_node_context_path(node_dir: Path) -> Path:
    """Return existing hidden context, falling back to legacy root ``context.json``."""
    node_dir = Path(node_dir)
    hidden = node_context_path(node_dir)
    if hidden.exists():
        return hidden
    return node_dir / NODE_CONTEXT_JSON
