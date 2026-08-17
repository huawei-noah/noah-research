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
from typing import Any


@dataclass(frozen=True)
class LnrWorkerLayout:
    root: Path
    workspace: Path
    logs: Path
    snapshots: Path


def lnr_worker_dirname(lhr: Any) -> str:
    dirname = str(getattr(lhr, "worker_dirname", "workers") or "workers").strip()
    return dirname or "workers"


def lnr_worker_layout(task_root: Path, lhr: Any, worker_index: int) -> LnrWorkerLayout:
    root = Path(task_root) / lnr_worker_dirname(lhr) / f"w{int(worker_index):02d}"
    return LnrWorkerLayout(
        root=root,
        workspace=root / "workspace",
        logs=root / "logs",
        snapshots=root / "snapshots",
    )


def ensure_lnr_worker_layout(task_root: Path, lhr: Any, worker_index: int) -> LnrWorkerLayout:
    layout = lnr_worker_layout(task_root, lhr, worker_index)
    for path in (layout.root, layout.workspace, layout.logs, layout.snapshots):
        path.mkdir(parents=True, exist_ok=True)
    return layout
