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

"""Task-neutral candidate artifact I/O helpers."""

from __future__ import annotations

import time
from pathlib import Path


def wait_for_stable_file(
    path: Path,
    *,
    timeout_sec: float = 2.0,
    interval_sec: float = 0.25,
    min_age_sec: float = 0.4,
) -> tuple[bool, str]:
    """Return true once a candidate artifact stops changing briefly."""
    deadline = time.monotonic() + max(0.0, timeout_sec)
    last_sig: tuple[int, int] | None = None
    stable_hits = 0
    last_size = -1
    last_age = 0.0
    while True:
        try:
            stat_result = path.stat()
        except OSError as exc:
            return False, f"cannot stat {path.name}: {exc}"
        sig = (int(stat_result.st_size), int(stat_result.st_mtime_ns))
        last_size = int(stat_result.st_size)
        last_age = max(0.0, time.time() - float(stat_result.st_mtime))
        if sig == last_sig and last_age >= min_age_sec:
            stable_hits += 1
        else:
            stable_hits = 0
            last_sig = sig
        if stable_hits >= 1:
            return True, "stable"
        if time.monotonic() >= deadline:
            return (
                False,
                f"{path.name} appears to still be writing "
                f"(size={last_size} bytes, mtime_age={last_age:.2f}s); "
                "retry validation after it stabilizes",
            )
        time.sleep(max(0.01, interval_sec))
