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

import logging
import subprocess

logger = logging.getLogger("scienceflow")


def parse_cpu_list(cpu_str: str) -> list[int]:
    if not cpu_str or not cpu_str.strip():
        return []
    result: list[int] = []
    for part in cpu_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def parse_gpu_list(gpu_str: str) -> list[int]:
    if not gpu_str or not gpu_str.strip():
        return []
    result: list[int] = []
    for part in gpu_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def query_gpu_free_memory() -> dict[int, int]:
    """Query nvidia-smi for free memory (MiB) per GPU.

    Returns mapping ``{gpu_index: free_memory_mib}``. Empty dict on failure.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi query failed: %s", exc)
        return {}

    result: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                idx = int(parts[0].strip())
                free = int(parts[1].strip())
                result[idx] = free
            except ValueError:
                continue
    return result


def select_least_used_gpu(
    candidates: list[int] | None = None,
    exclude: set[int] | None = None,
    count: int = 1,
) -> str | None:
    """Pick the GPU(s) with the most free memory.

    Parameters
    ----------
    candidates:
        Restrict selection to these GPU indices (physical IDs).
        ``None`` means all GPUs visible to nvidia-smi.
    exclude:
        GPU indices to skip (e.g. already assigned by earlier tasks in this batch).
        This is a **soft** preference: when all candidate GPUs are excluded, the
        function retries without exclusion and picks the least-loaded GPU(s) for
        shared / stacked execution.
    count:
        Number of GPUs to select. Returns comma-separated indices
        (e.g. ``"2,5"``) sorted by free memory descending.

    Returns
    -------
    Comma-separated GPU indices as a string (e.g. ``"3"`` or ``"2,5"``),
    or ``None`` if no GPU is available (nvidia-smi failed or no candidates).
    """
    free_map = query_gpu_free_memory()
    if not free_map:
        return None

    if candidates is not None:
        free_map = {k: v for k, v in free_map.items() if k in candidates}
    if not free_map:
        return None

    pool = {k: v for k, v in free_map.items() if k not in exclude} if exclude else free_map
    shared = False
    if not pool:
        pool = free_map
        shared = True

    count = max(1, min(count, len(pool)))
    sorted_gpus = sorted(pool, key=pool.get, reverse=True)  # type: ignore[arg-type]
    chosen = sorted_gpus[:count]
    tag = " (shared – all GPUs already assigned)" if shared else ""
    logger.info(
        "Auto GPU selection%s: chose GPU %s (free %s) from %s",
        tag,
        ",".join(str(g) for g in chosen),
        ", ".join(f"{g}={pool[g]} MiB" for g in chosen),
        {k: f"{v} MiB" for k, v in sorted(pool.items())},
    )
    return ",".join(str(g) for g in chosen)


def parse_gpu_list_auto(gpu_str: str) -> tuple[bool, int, list[int]]:
    """Parse ``gpu_list`` with ``auto`` mode support.

    Accepted formats::

        "auto"            → 1 GPU, from all
        "auto:2"          → 2 GPUs, from all
        "auto:2:0-7"      → 2 GPUs, from GPUs 0–7
        "auto:2:0,2,4,6"  → 2 GPUs, from GPUs 0/2/4/6
        "auto:0,2,4"      → 1 GPU, from GPUs 0/2/4  (backward compat: first segment not pure int → candidates)
        "4"               → static assignment (original behaviour)

    Returns ``(is_auto, gpu_count, candidate_gpu_ids)``.
    """
    s = (gpu_str or "").strip()
    if not s.lower().startswith("auto"):
        return False, 0, parse_gpu_list(s)

    rest = s[4:].strip()
    if not rest or not rest.startswith(":"):
        return True, 1, []

    segments = rest[1:].split(":", 1)

    # Single segment: could be count ("2") or candidates ("0,2,4" / "0-3")
    if len(segments) == 1:
        seg = segments[0].strip()
        if seg.isdigit() and int(seg) <= 16:
            return True, int(seg), []
        return True, 1, parse_gpu_list(seg)

    # Two segments: count : candidates
    count_str, cand_str = segments
    count = int(count_str.strip()) if count_str.strip().isdigit() else 1
    return True, max(1, count), parse_gpu_list(cand_str.strip())


def trim_long_string(s: str, max_len: int = 5000) -> str:
    if len(s) <= max_len:
        return s
    half = max_len // 2
    truncated = len(s) - max_len
    return f"{s[:half]}\n ... [{truncated} characters truncated] ... \n{s[-half:]}"
