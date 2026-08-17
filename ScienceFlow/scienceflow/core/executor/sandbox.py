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

"""Sandbox utilities: data truncation, submission management."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger("scienceflow")


def truncate_data_wrapper(max_samples: int) -> str:
    """Return Python code that monkey-patches pd.read_csv/read_parquet and json.load
    to limit rows from dataset directories, accelerating fast-debug runs."""
    if max_samples <= 0:
        return ""
    return f'''
if {max_samples} > 0:
    import pandas as pd
    import json as _json_mod
    def _debug_is_dataset_data(path):
        p = (str(path).replace(chr(92), "/") if path else "").lower()
        return "dataset" in p and any(x in p for x in ["/deep/", "/shallow/"])
    _pd_read_csv_orig = pd.read_csv
    def _pd_read_csv_patched(path, *args, **kwargs):
        df = _pd_read_csv_orig(path, *args, **kwargs)
        if _debug_is_dataset_data(path) and len(df) > {max_samples}:
            return df.head({max_samples}).copy()
        return df
    pd.read_csv = _pd_read_csv_patched
    if hasattr(pd, "read_parquet"):
        _pd_read_parquet_orig = pd.read_parquet
        def _pd_read_parquet_patched(path, *args, **kwargs):
            df = _pd_read_parquet_orig(path, *args, **kwargs)
            if _debug_is_dataset_data(path) and len(df) > {max_samples}:
                return df.head({max_samples}).copy()
            return df
        pd.read_parquet = _pd_read_parquet_patched
    _json_load_orig = _json_mod.load
    def _json_load_patched(fp, *args, **kwargs):
        data = _json_load_orig(fp, *args, **kwargs)
        path = getattr(fp, "name", "") or ""
        if not _debug_is_dataset_data(path):
            return data
        if isinstance(data, dict) and ("annotations" in data or "images" in data):
            kept_ids = set()
            for ann in (data.get("annotations") or []):
                img_id = ann.get("image_id")
                if img_id is not None and img_id not in kept_ids:
                    kept_ids.add(img_id)
                    if len(kept_ids) >= {max_samples}:
                        break
            if not kept_ids and data.get("images"):
                for img in data["images"][:{max_samples}]:
                    iid = img.get("id")
                    if iid is not None:
                        kept_ids.add(iid)
            if kept_ids:
                data = dict(data)
                if data.get("annotations") is not None:
                    data["annotations"] = [a for a in data["annotations"] if a.get("image_id") in kept_ids]
                if data.get("images") is not None:
                    data["images"] = [img for img in data["images"] if img.get("id") in kept_ids]
        elif isinstance(data, list) and len(data) > {max_samples}:
            data = data[:{max_samples}]
        return data
    _json_mod.load = _json_load_patched
'''


def _workspace_submission_subdir(file_prefix: str) -> str:
    """Directory under workspace where prep_cfg stores per-node CSVs."""
    return "submissions"


def replace_submission_in_code(
    code: str, node_id: int | str, file_prefix: str = "submission",
) -> str:
    """Replace submission.csv references with per-node files under submissions/."""
    subdir = _workspace_submission_subdir(file_prefix)
    target = f"{file_prefix}_{node_id}.csv"
    rel_path = f"{subdir}/{target}"
    modified = code

    if "submission/submission.csv" in modified:
        modified = modified.replace("submission/submission.csv", rel_path)
    if "submissions/submission.csv" in modified:
        modified = modified.replace("submissions/submission.csv", rel_path)

    if "/submission.csv" in modified:
        modified = modified.replace("/submission.csv", f"/{rel_path}")

    if "./submission.csv" in modified:
        modified = modified.replace("./submission.csv", f"./{rel_path}")

    if "to_csv('submission.csv" in modified:
        modified = modified.replace(
            "to_csv('submission.csv", f"to_csv('{rel_path}",
        )
    if 'to_csv("submission.csv' in modified:
        modified = modified.replace(
            'to_csv("submission.csv', f'to_csv("{rel_path}',
        )

    # Path join: parent already points at submissions/ — use filename only
    modified = modified.replace(
        ' / "submission.csv"', f' / "{target}"',
    )
    modified = modified.replace(
        " / 'submission.csv'", f" / '{target}'",
    )

    if '"submission.csv"' in modified:
        modified = modified.replace('"submission.csv"', f'"{rel_path}"')
    if "'submission.csv'" in modified:
        modified = modified.replace("'submission.csv'", f"'{rel_path}'")

    return modified


_GENERIC_MODEL_NAMES = sorted(
    [
        "best_model.pth", "best_model.bin", "best_model.pt",
        "model_best.pth", "model_best.bin", "model_best.pt",
        "model.pth", "model.pt", "model.bin",
        "checkpoint.pth", "checkpoint.pt", "checkpoint.bin",
    ],
    key=len,
    reverse=True,
)


def replace_model_path_in_code(
    code: str, node_id: int | str, file_flag: str = "",
) -> str:
    """Replace generic model filenames to avoid multi-process conflicts."""
    if ".pth" not in code and ".bin" not in code and ".pt" not in code:
        return code

    modified = code
    for filename in _GENERIC_MODEL_NAMES:
        if filename not in modified:
            continue
        name, ext = filename.rsplit(".", 1)
        prefix = f"{file_flag}_" if file_flag else ""
        new_filename = f"{prefix}{name}_{node_id}.{ext}"

        modified = modified.replace(f"/{filename}", f"/{new_filename}")
        modified = modified.replace(f'"{filename}"', f'"{new_filename}"')
        modified = modified.replace(f"'{filename}'", f"'{new_filename}'")

    return modified


def rename_submission(
    workspace_dir: str | Path,
    submission_dir: str | Path,
    node_id: str,
    *,
    file_prefix: str = "submission",
) -> bool:
    """If a bare submission.csv ended up in the wrong place, move it to
    submission_dir as {file_prefix}_{node_id}.csv. Returns True if rescue happened."""
    wd = Path(workspace_dir).resolve()
    target_dir = Path(submission_dir).resolve()
    filename = f"{file_prefix}_{node_id}.csv"
    target_path = target_dir / filename

    if target_path.exists():
        return False

    candidates = [
        wd / filename,
        wd / "submission.csv",
        wd / "submission" / filename,
        wd / "submissions" / filename,
        wd / "submissions" / "submission.csv",
    ]

    for src in candidates:
        if src.exists() and src.resolve() != target_path.resolve():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target_path))
            logger.info(f"[rescue_submission] Moved {src} -> {target_path}")
            return True
    return False


def check_submission_exists(submission_dir: str | Path, node_id: str) -> bool:
    """Check whether submission_{node_id}.csv exists in submission_dir."""
    return (Path(submission_dir) / f"submission_{node_id}.csv").exists()


_MODEL_PREFIXES = ("fast_debug_", "default_", "")


def strip_node_id_from_text(text: str, node_id: str) -> str:
    """Strip node_id artifacts from a single string, restoring original filenames."""
    if not text or node_id not in text:
        return text
    text = text.replace(f"submission_{node_id}.csv", "submission.csv")
    for prefix in _MODEL_PREFIXES:
        for filename in _GENERIC_MODEL_NAMES:
            name, ext = filename.rsplit(".", 1)
            modified = f"{prefix}{name}_{node_id}.{ext}"
            if modified in text:
                text = text.replace(modified, filename)
    return text


def strip_node_id_from_exec_result(exec_result, node_id: int | str):
    """In-place strip of node_id artifacts from ExecutionResult fields so that the
    LLM sees original filenames (e.g. submission.csv, best_model.pth)."""
    nid = str(node_id)

    if exec_result.term_out:
        exec_result.term_out = [strip_node_id_from_text(s, nid) for s in exec_result.term_out]
    if exec_result.all_term_out:
        exec_result.all_term_out = [strip_node_id_from_text(s, nid) for s in exec_result.all_term_out]
    if exec_result.exc_info and isinstance(exec_result.exc_info, dict):
        exec_result.exc_info = {
            k: strip_node_id_from_text(str(v), nid) if isinstance(v, str) else v
            for k, v in exec_result.exc_info.items()
        }
    if exec_result.exc_stack:
        cleaned = []
        for frame in exec_result.exc_stack:
            if isinstance(frame, (list, tuple)):
                cleaned.append(type(frame)(strip_node_id_from_text(str(item), nid) for item in frame))
            else:
                cleaned.append(frame)
        exec_result.exc_stack = cleaned
    return exec_result


# Backward-compatible re-export (canonical: scienceflow.safety.leakage_detector)
from scienceflow.safety.leakage_detector import validate_no_leakage
