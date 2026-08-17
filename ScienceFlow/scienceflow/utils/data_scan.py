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

"""
Data directory scan (e.g. MLEBench prepared/public).

Used by LNR workspace preparation and dataset preview helpers.
related tooling. Unified flow: traverse, read meta by extension, optional binary
probes, output fixed schema. No task-specific logic.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from collections import Counter
from typing import Any, Callable

# Meta file ext -> reader (unified: Path -> dict|str|None)
META_READERS: dict[str, Any] = {}

# Per-call overrides (set by :func:`scan_data_dir` for balanced budgets).
_META_JSON_MAX_BYTES: int = 50_000
_CSV_MAX_ROWS_TO_COUNT: int = 200_000


class _ScanMetaCsvScope:
    """Temporarily override JSON/CSV scan caps for :func:`scan_data_dir`."""

    __slots__ = ("_meta", "_csv", "_old")

    def __init__(self, meta_sample_max_bytes: int, csv_max_rows_to_scan: int) -> None:
        self._meta = int(meta_sample_max_bytes)
        self._csv = int(csv_max_rows_to_scan)
        self._old: tuple[int, int] = (50_000, 200_000)

    def __enter__(self) -> None:
        global _META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT
        self._old = (_META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT)
        _META_JSON_MAX_BYTES = self._meta
        _CSV_MAX_ROWS_TO_COUNT = self._csv

    def __exit__(self, *exc: object) -> None:
        global _META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT
        _META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT = self._old
        return None

# When a single directory holds more files than this, stop listing them one by one and show only a summary (e.g. *.wav (28,150))
FLAT_DIR_FILE_LIST_THRESHOLD = 500
# When a metadata file has at most this many lines/entries, append the first few sample records to dir_structure to help the LLM understand the actual data format
META_SAMPLE_COUNT = 1
# Maximum number of child items (subdirectories + files) listed when expanding a subdirectory; beyond that, show only the first N items with an omission note
EXPAND_LIST_MAX_ITEMS = 20
# Extensions treated as "metadata" in large flat directories; listed individually (with file names) so prep can see csv/json/md etc.
FLAT_DIR_META_EXTS = (".csv", ".json", ".md", ".yml", ".yaml", ".txt")
# Directory names always excluded during scanning (never walked, never shown)
SCAN_EXCLUDE_DIRS = frozenset({"__pycache__"})
# Upper limit for displayed JSON array/object keys
JSON_KEY_SHOW_LIMIT = 12
# subdirs under a directory: maximum number of subdirectories shown; beyond that, show the first N plus "... (M total)"
SUBDIRS_SHOW_LIMIT = 5
# preview_raw_files: read the first few lines of common non-CSV/JSON data files (.xyz, .txt, .dat, etc.) in subdirectories as a preview
RAW_PREVIEW_EXTS = frozenset({".xyz", ".txt", ".dat", ".sdf", ".mol", ".pdb", ".cif", ".mol2"})
RAW_PREVIEW_MAX_LINES = 10
RAW_PREVIEW_MAX_LINE_LEN = 120
# Read hints for binary/media files, helping the LLM know which library to open them with
BINARY_FILE_HINTS: dict[str, str] = {
    ".dicom": "DICOM medical image; read with pydicom",
    ".dcm":   "DICOM medical image; read with pydicom",
    ".tiff":  "large TIFF image; read with tifffile or openslide",
    ".tif":   "TIFF image; read with tifffile or PIL",
    ".jpg":   "JPEG image; read with PIL or cv2",
    ".jpeg":  "JPEG image; read with PIL or cv2",
    ".png":   "PNG image; read with PIL or cv2",
    ".webp":  "WebP image; read with PIL",
    ".bmp":   "BMP image; read with PIL or cv2",
    ".gif":   "GIF image; read with PIL",
    ".mat":   "Matlab data; read with scipy.io.loadmat",
    ".bin":   "binary data; read with numpy.fromfile or struct",
    ".wav":   "audio waveform; read with scipy.io.wavfile or librosa",
    ".mp3":   "audio; read with librosa",
    ".mp4":   "video; read with cv2.VideoCapture or decord",
    ".avi":   "video; read with cv2.VideoCapture",
    ".npy":   "numpy array; read with numpy.load",
    ".npz":   "numpy archive; read with numpy.load",
    ".h5":    "HDF5; read with h5py",
    ".hdf5":  "HDF5; read with h5py",
    ".parquet": "columnar data; read with pandas.read_parquet",
    ".feather": "columnar data; read with pandas.read_feather",
    ".tfrecord": "TensorFlow record; read with tf.data.TFRecordDataset",
    ".tfrec":   "TensorFlow record; read with tf.data.TFRecordDataset",
    ".nii":   "NIfTI neuroimaging; read with nibabel",
    ".nii.gz": "NIfTI neuroimaging; read with nibabel",
}
# Runtime artifact file extensions; not raw data, hidden from preview
RUNTIME_ARTIFACT_EXTS = frozenset({".pth", ".pt", ".ckpt", ".pkl", ".pickle"})
# Single-line output cap for binary probing (to avoid blowing up the prompt)
BINARY_PROBE_MAX_LINE = 900


def _probe_mat(path: Path) -> str | None:
    try:
        import scipy.io

        d = scipy.io.loadmat(str(path))
        items: list[str] = []
        for k, v in sorted(d.items()):
            if k.startswith("__"):
                continue
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                items.append(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                items.append(f"{k}: type={type(v).__name__}")
        return "; ".join(items) if items else None
    except Exception:
        return None


def _probe_npy(path: Path) -> str | None:
    try:
        import numpy as np

        arr = np.load(str(path), allow_pickle=False)
        return f"shape={tuple(arr.shape)}, dtype={arr.dtype}"
    except Exception:
        return None


def _probe_npz(path: Path) -> str | None:
    try:
        import numpy as np

        with np.load(str(path), allow_pickle=False) as data:
            parts = [
                f"{k}: shape={tuple(data[k].shape)}, dtype={data[k].dtype}"
                for k in sorted(data.files)[:24]
            ]
        return "; ".join(parts) if parts else None
    except Exception:
        return None


def _probe_h5(path: Path) -> str | None:
    try:
        import h5py

        items: list[str] = []
        with h5py.File(str(path), "r") as f:

            def visit(name: str, obj: Any) -> None:
                if len(items) >= 24:
                    return
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    items.append(f"{name}: shape={tuple(obj.shape)}, dtype={obj.dtype}")

            f.visititems(visit)
        return "; ".join(sorted(items)) if items else None
    except Exception:
        return None


def _probe_parquet(path: Path) -> str | None:
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        cols = ", ".join(f"{c}({df[c].dtype})" for c in df.columns[:20])
        more = f", ... ({len(df.columns)} cols)" if len(df.columns) > 20 else ""
        return f"{len(df):,} rows; {cols}{more}"
    except Exception:
        return None


def _probe_image(path: Path) -> str | None:
    """Best-effort image dimension probe using PIL (header only, no pixel decode)."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
        return f"size={w}x{h}, mode={mode}"
    except Exception:
        return None


# ext key matches Counter keys (suffix without dot, lowercased)
BINARY_PROBERS: dict[str, Callable[[Path], str | None]] = {
    "mat": _probe_mat,
    "npy": _probe_npy,
    "npz": _probe_npz,
    "h5": _probe_h5,
    "hdf5": _probe_h5,
    "parquet": _probe_parquet,
    "jpg": _probe_image,
    "jpeg": _probe_image,
    "png": _probe_image,
    "webp": _probe_image,
    "bmp": _probe_image,
    "gif": _probe_image,
    "tif": _probe_image,
    "tiff": _probe_image,
}


def _append_binary_probe_samples(
    dir_path: Path,
    exts_counter: Counter,
    lines_out: list[str],
    *,
    probe_binary_files: bool,
) -> None:
    """Append one probed line per supported binary ext present in *dir_path*."""
    if not probe_binary_files:
        return
    for ext_no_dot in sorted(k for k in BINARY_PROBERS if exts_counter.get(k, 0) > 0):
        suf = "." + ext_no_dot
        try:
            one = next(dir_path.glob(f"*{suf}"), None)
            if one is None or not one.is_file():
                continue
            fn = BINARY_PROBERS[ext_no_dot]
            txt = fn(one)
            if not txt:
                continue
            if len(txt) > BINARY_PROBE_MAX_LINE:
                txt = txt[:BINARY_PROBE_MAX_LINE] + "..."
            lines_out.append(f"│   └── [binary_probe] {one.name}")
            lines_out.append(f"│     | {txt}")
        except OSError:
            pass


def _ext_counter_from_dir_files(dir_path: Path) -> Counter:
    """Non-recursive extension counts under *dir_path* (one sample dir).

    When the parent data root hits WALK_MAX_DIRS and os.walk does not recurse
    into Sample*/ children, aggregated ``dir_stats`` ext counters are empty;
    probing still needs to see .mat/.npy/etc. from a real subdirectory.
    """
    c: Counter = Counter()
    try:
        for p in dir_path.iterdir():
            if p.is_file():
                c[(p.suffix.lstrip(".") or "no_ext").lower()] += 1
    except OSError:
        pass
    return c


def _binary_hint(ext_no_dot: str) -> str:
    """Return a read-hint string for binary/media extensions, or '' if none."""
    key = "." + ext_no_dot.lower()
    return BINARY_FILE_HINTS.get(key, "")


def _ext_comment_with_hint(ext_no_dot: str, count: int) -> str:
    """Build '*.ext # {id}.ext(N)' with optional read-hint appended."""
    ext_dot = "." + ext_no_dot
    hint = _binary_hint(ext_no_dot)
    base = f"*{ext_dot} # {{id}}{ext_dot}({count:,})"
    if hint:
        base += f"  [{hint}]"
    return base


def _ext_list_with_hints(exts_counter: Counter, top_n: int = 5) -> str:
    """Build comma-separated ext list with binary hints for dir comments.
    E.g. '.mp4, .mat [read with scipy.io.loadmat], .wav [read with librosa]'
    Skips runtime artifact extensions (.pth, .pt, etc.)."""
    parts: list[str] = []
    for e, _ in exts_counter.most_common(top_n + len(RUNTIME_ARTIFACT_EXTS)):
        if not e or e == "no_ext":
            continue
        if ("." + e) in RUNTIME_ARTIFACT_EXTS:
            continue
        hint = _binary_hint(e)
        if hint:
            parts.append(f".{e} [{hint}]")
        else:
            parts.append(f".{e}")
        if len(parts) >= top_n:
            break
    return ", ".join(parts) or "mixed"


def _is_numeric_str(s: str) -> bool:
    """True if s looks numeric, empty, or null-like (NaN/NA/None)."""
    s = s.strip()
    if not s or s.lower() in ("nan", "na", "null", "none", "inf", "-inf"):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _json_type_comment(m: dict) -> str:
    """Concise type-aware comment for JSON metadata."""
    typ = m.get("type", "")
    if typ == "array":
        cnt = m.get("count", 0)
        keys = m.get("keys", [])
        if keys:
            show = ", ".join(keys[:8])
            if len(keys) > 8:
                show += ", ..."
            return f" # [array of {cnt:,} objects, each: {{{show}}}]"
        sample = m.get("sample_items", [])
        if sample:
            return f" # [array of {cnt:,} {type(sample[0]).__name__}s]"
        return f" # [array of {cnt:,} items]"
    if typ == "object":
        keys = m.get("keys", [])
        value_counts = m.get("value_counts", {})
        sample = m.get("sample_items", {})
        kc = m.get("key_count", len(keys))
        if keys and sample:
            parts = []
            for k in keys[:JSON_KEY_SHOW_LIMIT]:
                vc = value_counts.get(k)
                v = sample.get(k)
                if isinstance(v, list):
                    parts.append(f"{k}(list[{vc or len(v):,}])")
                elif isinstance(v, dict):
                    parts.append(f"{k}(dict[{vc or len(v):,}])")
                else:
                    parts.append(f"{k}({type(v).__name__})")
            tail = f", ... ({kc} total)" if kc > JSON_KEY_SHOW_LIMIT else ""
            return f" # {{dict: {', '.join(parts)}{tail}}}"
        return f" # (dict, {kc} keys)"
    if typ == "truncated":
        return " # (truncated, could not parse)"
    return ""


def _meta_file_comment(m: dict) -> str:
    """One-line comment for any metadata file (CSV or JSON)."""
    if "columns" in m:
        cols = m["columns"]
        row_count = m.get("row_count", 0)
        return f" # ({len(cols)} cols, {row_count:,} rows)"
    return _json_type_comment(m)


def _hide_from_preview(name: str, include_val_outputs: bool = False) -> bool:
    """Return True if this file/dir should be hidden from data_preview.
    When include_val_outputs is False, hide basenames starting with ``validation`` (aligned with
    dataset preview helpers). ``val_metadata*`` is always hidden."""
    stem = Path(name).stem.lower()
    ext = Path(name).suffix.lower()
    base = Path(name).name.lower()
    if stem == "description":
        return True
    # Runtime artifacts (model weights/checkpoints) are not raw data
    if ext in RUNTIME_ARTIFACT_EXTS:
        return True
    # submission.csv is the reference answer; only sample_submission.csv is kept
    if stem == "submission" and ext == ".csv":
        return True
    # Always hide val_metadata*.json so the agent cannot see validation metadata
    if stem == "val_metadata" or stem.startswith("val_metadata_"):
        return True
    # Hide intermediate/temporary and merged artifacts: _subset, tmp, _combined
    name_lower = name.lower()
    if "_subset" in name_lower or "_combined" in name_lower:
        return True
    if "tmp" in name_lower:
        return True
    if include_val_outputs:
        return False
    if base.startswith("validation"):
        return True
    return False


def _register(ext: str):
    def _inner(fn):
        META_READERS[ext] = fn
        return fn

    return _inner


@_register(".csv")
def _read_csv(path: Path, max_rows: int = 3, max_rows_to_count: int | None = None) -> dict[str, Any] | None:
    try:
        old_limit = csv.field_size_limit()
        cap = max_rows_to_count if max_rows_to_count is not None else _CSV_MAX_ROWS_TO_COUNT
        try:
            csv.field_size_limit(100 * 1024 * 1024)  # 100MB for large fields
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                rows, row_count = [], 0
                for i, row in enumerate(reader):
                    row_count = i + 1
                    if i < max_rows:
                        rows.append(dict(row))
                    if row_count >= cap:
                        break
                # detect columns whose sample values are non-numeric strings
                str_value_cols: dict[str, str] = {}
                for col in columns:
                    for row in rows:
                        val = (row.get(col) or "").strip()
                        if val and not _is_numeric_str(val):
                            str_value_cols[col] = val
                            break
                result: dict[str, Any] = {
                    "columns": columns,
                    "sample_rows": rows,
                    "row_count": row_count,
                }
                if row_count >= cap:
                    result["row_count_truncated"] = True
                if str_value_cols:
                    result["string_value_cols"] = str_value_cols
                return result
        finally:
            csv.field_size_limit(old_limit)
    except Exception:
        return None


@_register(".json")
def _read_json(path: Path, max_bytes: int | None = None) -> dict[str, Any] | None:
    eff = max_bytes if max_bytes is not None else _META_JSON_MAX_BYTES
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            content = content[:eff]
            i, j = content.find("{"), content.find("[")
            start = min(i if i >= 0 else 999999, j if j >= 0 else 999999)
            if start < 999999:
                end = content.rfind("}") if i >= 0 and (j < 0 or i <= j) else content.rfind("]")
                if end > start:
                    try:
                        data = json.loads(content[start : end + 1])
                    except json.JSONDecodeError:
                        return {"type": "truncated"}
                else:
                    return {"type": "truncated"}
            else:
                return {"type": "truncated"}
        if isinstance(data, list):
            keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
            sample = data[:META_SAMPLE_COUNT]
            return {"type": "array", "count": len(data), "keys": keys, "sample_items": sample}
        if isinstance(data, dict):
            keys = list(data.keys())
            value_counts = {}
            sample = {}
            for k, v in data.items():
                if isinstance(v, list):
                    value_counts[k] = len(v)
                    sample[k] = v[:1]
                elif isinstance(v, dict):
                    value_counts[k] = len(v)
                    sample[k] = v
                else:
                    sample[k] = v
            return {"type": "object", "keys": keys, "key_count": len(keys), "sample_items": sample, "value_counts": value_counts}
        return {"type": type(data).__name__}
    except Exception:
        return None


MAX_SAMPLE_LINE_LEN = 120
# Detection/box columns (PredictionString etc.) need more characters, otherwise the class name (e.g. car) is invisible in the preview and the column is easily misjudged as all-numeric
MAX_CELL_LEN_DEFAULT = 60
MAX_CELL_LEN_PREDICTION_STRING = 200
# Per-key value display length cap in JSON preview, so all keys stay fully visible and are not squeezed out by long values
JSON_KEY_VAL_MAX_LEN = 100


def _read_raw_file_preview(path: Path) -> str | None:
    """Read the first few lines of a raw data file for LLM preview."""
    try:
        lines: list[str] = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= RAW_PREVIEW_MAX_LINES:
                    break
                line = line.rstrip("\n\r")
                if len(line) > RAW_PREVIEW_MAX_LINE_LEN:
                    line = line[:RAW_PREVIEW_MAX_LINE_LEN] + "..."
                lines.append(line)
        if not lines:
            return None
        return "\n".join(lines)
    except Exception:
        return None


def _format_raw_preview_lines(preview_text: str, indent: str = "    ") -> list[str]:
    """Format raw file preview text into tree-display lines."""
    lines: list[str] = []
    for line in preview_text.splitlines():
        lines.append(f"{indent}  | {line}")
    return lines


def _format_meta_sample_lines(m: dict, indent: str = "    ") -> list[str]:
    """Format sample data from CSV or JSON meta for display in dir_structure tree.
    First line = header (column names or keys), next line(s) = sample data.
    """
    lines: list[str] = []

    # CSV: header line then sample row(s)
    sample_rows = m.get("sample_rows", [])
    cols = m.get("columns", [])
    if sample_rows and cols:
        header_line = f"{indent}  | {', '.join(cols)}"
        lines.append(header_line)
        has_long_col = any(
            "prediction" in c.lower() or c.lower() in ("predictionstring", "label", "encoding")
            for c in cols
        )
        line_limit = MAX_SAMPLE_LINE_LEN + (MAX_CELL_LEN_PREDICTION_STRING - MAX_CELL_LEN_DEFAULT) if has_long_col else MAX_SAMPLE_LINE_LEN
        for row in sample_rows[:META_SAMPLE_COUNT]:
            vals = []
            for c in cols:
                cell = str(row.get(c, ""))
                max_len = MAX_CELL_LEN_PREDICTION_STRING if (
                    "prediction" in c.lower() or c.lower() in ("predictionstring", "label", "encoding")
                ) else MAX_CELL_LEN_DEFAULT
                vals.append(cell[:max_len])
            line = f"{indent}  | {', '.join(vals)}"
            if len(line) > line_limit:
                line = line[:line_limit] + "..."
            lines.append(line)
        # string-valued columns hint removed to reduce preview noise
        return lines

    # JSON: sample_items
    sample_items = m.get("sample_items")
    if not sample_items:
        return []
    typ = m.get("type", "")
    if typ == "array" and isinstance(sample_items, list):
        keys = m.get("keys", [])
        for item in sample_items[:META_SAMPLE_COUNT]:
            if isinstance(item, dict):
                kvs = keys or list(item.keys())
                # Print key+val in pairs, one "key: value" per line; cap each val length so all keys stay fully visible
                for k in kvs:
                    v = item.get(k, "")
                    if isinstance(v, list):
                        if v:
                            first = json.dumps(v[0], ensure_ascii=False)
                            if len(first) > JSON_KEY_VAL_MAX_LEN:
                                first = first[:JSON_KEY_VAL_MAX_LEN] + "…"
                            val_str = f"[{first}, …] ({len(v)} items)" if len(v) > 1 else f"[{first}]"
                        else:
                            val_str = "[]"
                    elif isinstance(v, dict):
                        val_str = json.dumps(v, ensure_ascii=False)
                        if len(val_str) > JSON_KEY_VAL_MAX_LEN:
                            val_str = val_str[:JSON_KEY_VAL_MAX_LEN] + "…"
                    else:
                        val_str = str(v)
                        if len(val_str) > JSON_KEY_VAL_MAX_LEN:
                            val_str = val_str[:JSON_KEY_VAL_MAX_LEN] + "…"
                    line = f"{indent}  | {k}: {val_str}"
                    lines.append(line)
            else:
                line = f"{indent}  | {json.dumps(item, ensure_ascii=False)}"
                if len(line) > MAX_SAMPLE_LINE_LEN:
                    line = line[:MAX_SAMPLE_LINE_LEN] + "..."
                lines.append(line)
    elif typ == "object" and isinstance(sample_items, dict):
        value_counts = m.get("value_counts", {})
        for k, v in sample_items.items():
            count = value_counts.get(k)
            if isinstance(v, list):
                cnt = count if count is not None else len(v)
                if v:
                    first = json.dumps(v[0], ensure_ascii=False)
                    if len(first) > JSON_KEY_VAL_MAX_LEN:
                        first = first[:JSON_KEY_VAL_MAX_LEN] + "…"
                    val_str = f"[{first}, …] ({cnt:,} items)"
                else:
                    val_str = f"[] (0 items)"
            elif isinstance(v, dict):
                cnt = count if count is not None else len(v)
                val_str = f"{{…{cnt:,} keys}}"
            elif isinstance(v, str):
                val_str = v if len(v) <= JSON_KEY_VAL_MAX_LEN else v[:JSON_KEY_VAL_MAX_LEN] + "…"
            else:
                val_str = json.dumps(v, ensure_ascii=False)
                if len(val_str) > JSON_KEY_VAL_MAX_LEN:
                    val_str = val_str[:JSON_KEY_VAL_MAX_LEN] + "…"
            line = f"{indent}  | {k}: {val_str}"
            lines.append(line)

    return lines


def _try_resolve_isdir(p: Path) -> bool:
    try:
        return p.resolve().is_dir()
    except OSError:
        return False


def _walk_count_files_and_exts(child_path: Path) -> tuple[int, Counter]:
    """Single pass under *child_path* (replaces double ``rglob`` for counts + extensions)."""
    fc = 0
    exts: Counter = Counter()
    try:
        for _dp, _dns, fns in os.walk(str(child_path), topdown=True, followlinks=True):
            for f in fns:
                fc += 1
                suf = Path(f).suffix.lstrip(".").lower() or "no_ext"
                exts[suf] += 1
    except OSError:
        pass
    return fc, exts


def _append_raw_preview_for_dir(dir_path: Path, exts_counter: Counter, lines_out: list[str], nested: bool = False) -> None:
    """For a flat directory, find one raw file per matching extension and append preview lines."""
    raw_exts_in_dir = {e for e in exts_counter if ("." + e) in RAW_PREVIEW_EXTS}
    for raw_ext in sorted(raw_exts_in_dir):
        suf = "." + raw_ext
        try:
            one = next(dir_path.glob(f"*{suf}"), None)
            if one is not None and one.is_file():
                preview = _read_raw_file_preview(one)
                if preview:
                    lines_out.append(f"│   └── [sample] {one.name} (first {RAW_PREVIEW_MAX_LINES} lines)")
                    for sl in _format_raw_preview_lines(preview, indent=""):
                        lines_out.append("│   " + sl)
        except OSError:
            pass


def _append_raw_preview_for_subdir_with_children(
    dir_path: Path, sub_sample: list[str], lines_out: list[str]
) -> None:
    """For a directory with child subdirs (e.g. train/ with 1/, 2/, ...), pick the first child
    and show raw file previews from it."""
    if not sub_sample:
        return
    first_child_name = sub_sample[0].rstrip("/")
    child_path = dir_path / first_child_name
    if not child_path.is_dir():
        return
    try:
        for f in sorted(child_path.iterdir()):
            if f.is_file() and f.suffix.lower() in RAW_PREVIEW_EXTS:
                preview = _read_raw_file_preview(f)
                if preview:
                    rel = f"{first_child_name}/{f.name}"
                    lines_out.append(f"│   └── [sample] {rel} (first {RAW_PREVIEW_MAX_LINES} lines)")
                    for sl in _format_raw_preview_lines(preview, indent=""):
                        lines_out.append("│   " + sl)
                break
    except OSError:
        pass


def _append_raw_preview_for_capped_or_nested(dir_path: Path, lines_out: list[str]) -> None:
    """For capped or deeply nested directories, find the first child subdir containing a raw file
    and show its preview. Falls back to checking direct files in the directory."""
    try:
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir():
                for f in sorted(entry.iterdir()):
                    if f.is_file() and f.suffix.lower() in RAW_PREVIEW_EXTS:
                        preview = _read_raw_file_preview(f)
                        if preview:
                            rel = f"{entry.name}/{f.name}"
                            lines_out.append(f"│   └── [sample] {rel} (first {RAW_PREVIEW_MAX_LINES} lines)")
                            for sl in _format_raw_preview_lines(preview, indent=""):
                                lines_out.append("│   " + sl)
                        return
            elif entry.is_file() and entry.suffix.lower() in RAW_PREVIEW_EXTS:
                preview = _read_raw_file_preview(entry)
                if preview:
                    lines_out.append(f"│   └── [sample] {entry.name} (first {RAW_PREVIEW_MAX_LINES} lines)")
                    for sl in _format_raw_preview_lines(preview, indent=""):
                        lines_out.append("│   " + sl)
                return
    except OSError:
        pass


def _build_subdir_tree_lines(
    subdir_path: Path,
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    include_val_outputs: bool = False,
    preview_raw_files: bool = False,
    probe_binary_files: bool = False,
) -> list[str]:
    """Build tree lines for a subdirectory (e.g. Deep/ or Shallow/). Large directories (files only, above the threshold) produce a single summary line instead of an itemized listing."""
    if not subdir_path.exists() or not subdir_path.is_dir():
        return []

    # Do a lightweight pass first: count items and extensions only, without keeping file names
    n_dirs = 0
    n_files = 0
    ext_counter: Counter = Counter()
    for p in subdir_path.iterdir():
        if p.name in SCAN_EXCLUDE_DIRS:
            continue
        if p.is_dir():
            n_dirs += 1
        elif p.is_symlink() and _try_resolve_isdir(p):
            n_dirs += 1
        else:
            n_files += 1
            ext_counter[(p.suffix.lower() or "no_ext").lstrip(".")] += 1

    # Files only and above the threshold: summarize by extension, and list metadata-type files (csv/json/md, etc.) by name
    if n_dirs == 0 and n_files > FLAT_DIR_FILE_LIST_THRESHOLD:
        meta_exts_set = {e.lstrip(".") for e in FLAT_DIR_META_EXTS}
        parts: list[str] = []
        # 1) Metadata-type extensions: list actual file names (and read meta) so prep can see train_curated.csv etc.
        meta_files: list[tuple[str, str, dict]] = []  # (name, ext, meta_or_empty)
        for p in sorted(subdir_path.iterdir()):
            if p.is_dir() or (p.is_symlink() and _try_resolve_isdir(p)):
                continue
            ext = (p.suffix.lower() or "no_ext").lstrip(".")
            if ext not in meta_exts_set:
                continue
            if _hide_from_preview(p.name, include_val_outputs):
                continue
            meta_key = f"{meta_prefix}/{p.name}"
            m = meta.get(meta_key)
            if m is None and p.suffix in meta_readers:
                m = meta_readers[p.suffix](p) or {}
                if m:
                    meta[meta_key] = m
            meta_files.append((p.name, ext, m or {}))
        max_meta_show = EXPAND_LIST_MAX_ITEMS * 2
        for name, _ext, m in meta_files[:max_meta_show]:
            comment = _meta_file_comment(m)
            parts.append(name + comment)
            if "columns" in m or m.get("sample_items"):
                parts.extend(_format_meta_sample_lines(m, indent=""))
        if len(meta_files) > max_meta_show:
            parts.append(f"… ({len(meta_files)} meta files total, showing first {max_meta_show})")
        # 2) Non-metadata extensions: summarize by extension, e.g. *.wav (28,150), with a read-hint attached
        for ext, cnt in ext_counter.most_common():
            if ext == "no_ext":
                continue
            if ext in meta_exts_set:
                continue
            parts.append(_ext_comment_with_hint(ext, cnt))
        # Add uniform tree prefixes: ├── for the first N-1 lines, └── for the last line
        lines_out = []
        for i, s in enumerate(parts):
            if s.startswith("  |"):
                lines_out.append("│   " + s)
            else:
                prefix = "└── " if i == len(parts) - 1 else "├── "
                lines_out.append(prefix + s)
        return lines_out

    # Otherwise fall back to the original logic: list subdirectories and files (may rglob into subdirectories, but usually not huge)
    lines_out: list[str] = []
    subdirs: list[tuple[str, int, str, list[str], Counter]] = []
    files: list[tuple[str, dict]] = []

    def _is_dir_or_symlink_to_dir(p: Path) -> bool:
        if p.is_dir():
            return True
        if p.is_symlink():
            try:
                return p.resolve().is_dir()
            except OSError:
                pass
        return False

    for p in sorted(subdir_path.iterdir()):
        rel_name = p.name
        if rel_name in SCAN_EXCLUDE_DIRS:
            continue
        if _hide_from_preview(rel_name, include_val_outputs):
            continue
        if _is_dir_or_symlink_to_dir(p):
            child_path = subdir_path / rel_name
            fc, exts = _walk_count_files_and_exts(child_path)
            ext = "." + exts.most_common(1)[0][0] if exts else ""
            try:
                # Subdirectory names only, used for the subdirs: display (directory stats); keep the full list so "N total" can be shown
                children = sorted(
                    c.name + "/"
                    for c in child_path.iterdir()
                    if (c.is_dir() or (c.is_symlink() and _try_resolve_isdir(c)))
                    and c.name not in SCAN_EXCLUDE_DIRS
                )
            except OSError:
                children = []
            subdirs.append((rel_name, fc, ext, children, exts))
        else:
            ext = (p.suffix.lower() or "no_ext").lstrip(".")
            meta_key = f"{meta_prefix}/{rel_name}"
            m = meta.get(meta_key)
            if m is None and p.suffix in meta_readers:
                m = meta_readers[p.suffix](p)
                if m is not None:
                    meta[meta_key] = m
            files.append((rel_name, {"type": "file", "ext": ext, "meta": m}))

    # Show metadata files (csv/json/md, etc.) first so they are not truncated away when there are many subdirectories
    meta_exts_set = {e.lstrip(".").lower() for e in FLAT_DIR_META_EXTS}
    meta_files = [(name, info) for name, info in files if (info.get("ext") or "").lower() in meta_exts_set]
    other_files = [(name, info) for name, info in files if (info.get("ext") or "").lower() not in meta_exts_set]
    items: list[tuple[str, str, Any]] = []
    for name, info in sorted(meta_files, key=lambda x: x[0]):
        items.append((name, "file", info))
    # Group subdirs with identical file signatures (count + ext set) to collapse repetitive entries
    _subdir_sorted = sorted(subdirs, key=lambda x: x[0])
    _sig_groups: dict[tuple, list[tuple]] = {}
    for entry in _subdir_sorted:
        name, fc, ext, children, exts = entry
        sig = (fc, tuple(sorted(exts.items())))
        _sig_groups.setdefault(sig, []).append(entry)

    for sig, group in _sig_groups.items():
        first = group[0]
        name, fc, ext, children, exts = first
        items.append((name, "dir", {"file_count": fc, "ext": ext, "sub_sample": children, "ext_counter": exts}))
        if len(group) > 1:
            last_name = group[-1][0]
            items.append((f"... ({len(group) - 1} more similar dirs, {name} ~ {last_name})", "collapsed", {}))
    for name, info in sorted(other_files, key=lambda x: x[0]):
        items.append((name, "file", info))

    total_items = len(items)
    if total_items > EXPAND_LIST_MAX_ITEMS:
        items = items[:EXPAND_LIST_MAX_ITEMS]

    for i, (name, typ, info) in enumerate(items):
        is_last = i == len(items) - 1 and total_items <= EXPAND_LIST_MAX_ITEMS
        if i == len(items) - 1 and total_items > EXPAND_LIST_MAX_ITEMS:
            prefix = "├── "
        else:
            prefix = "└── " if is_last else "├── "
        if typ == "collapsed":
            lines_out.append(prefix + name)
            continue
        if typ == "dir":
            fc = info.get("file_count", 0)
            ext = info.get("ext", "")
            exts_counter = info.get("ext_counter") or Counter()
            if len(exts_counter) > 1:
                ext_list = _ext_list_with_hints(exts_counter)
                comment = f" # ({fc:,} files: {ext_list})"
            else:
                id_col = "{id}"
                csv_meta = meta.get(f"{meta_prefix}/{name}.csv")
                if csv_meta:
                    cols = csv_meta.get("columns", [])
                    for c in ["image_name", "image_id", "StudyInstanceUID", "Id", "id", "fname"]:
                        if c in cols:
                            id_col = "{" + c + "}"
                            break
                suf = f"{id_col}{ext}" if ext else ""
                comment = f" # {suf}({fc:,})" if suf else f" # {fc:,} files"
                hint = _binary_hint(ext.lstrip(".")) if ext else ""
                if hint:
                    comment += f"  [{hint}]"
            sub_sample = info.get("sub_sample") or []
            subdir_count = info.get("subdir_count", len(sub_sample))
            if sub_sample or subdir_count:
                comment += " subdirs: " + ", ".join(sub_sample[:SUBDIRS_SHOW_LIMIT])
                if subdir_count > SUBDIRS_SHOW_LIMIT:
                    comment += f" ... ({subdir_count} total)"
            lines_out.append(prefix + name + "/" + comment)
            # If the subdirectory has no nested subdirectories, show 1 sample for each json/csv present in it
            if not sub_sample:
                child_path = subdir_path / name
                if child_path.is_dir():
                    for show_ext in ("json", "csv"):
                        if show_ext not in exts_counter:
                            continue
                        suf = "." + show_ext
                        if suf not in meta_readers:
                            continue
                        try:
                            one = next(child_path.glob(f"*{suf}"), None)
                            if one is not None and one.is_file():
                                sample_meta = meta_readers.get(suf)
                                if sample_meta:
                                    m = sample_meta(one)
                                    if m and ("columns" in m or m.get("sample_items") is not None or m.get("type")):
                                        lines_out.append("│   └── [sample] " + one.name)
                                        for sl in _format_meta_sample_lines(m, indent=""):
                                            lines_out.append("│   " + sl)
                        except OSError:
                            pass
                    if preview_raw_files:
                        _append_raw_preview_for_dir(child_path, exts_counter, lines_out, nested=False)
                    if probe_binary_files:
                        _append_binary_probe_samples(
                            child_path, exts_counter, lines_out,
                            probe_binary_files=True,
                        )
            else:
                if preview_raw_files:
                    _append_raw_preview_for_subdir_with_children(subdir_path / name, sub_sample, lines_out)
                if probe_binary_files and sub_sample:
                    first_child_name = sub_sample[0].rstrip("/")
                    fc_path = subdir_path / name / first_child_name
                    if fc_path.is_dir():
                        sub_exts: Counter = Counter()
                        try:
                            for c in fc_path.iterdir():
                                if c.is_file():
                                    sub_exts[(c.suffix.lower() or "no_ext").lstrip(".")] += 1
                        except OSError:
                            sub_exts = Counter()
                        _append_binary_probe_samples(
                            fc_path, sub_exts, lines_out, probe_binary_files=True,
                        )
        else:
            m = info.get("meta") or {}
            comment = _meta_file_comment(m)
            lines_out.append(prefix + name + comment)
            for sl in _format_meta_sample_lines(m, indent=""):
                lines_out.append("│   " + sl)

    if total_items > EXPAND_LIST_MAX_ITEMS:
        lines_out.append(f"└── … ({total_items:,} items total, showing first {EXPAND_LIST_MAX_ITEMS})")

    return lines_out


def scan_data_dir(
    data_dir: str | Path,
    expand_subdirs: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    include_val_outputs: bool = False,
    preview_raw_files: bool = False,
    probe_binary_files: bool = False,
    *,
    walk_budget_dirs: int | None = 200_000,
    walk_budget_files: int | None = 500_000,
    probe_binary_dirs_budget: int = 8,
    preview_raw_dirs_budget: int = 12,
    meta_sample_max_bytes: int = 50_000,
    csv_max_rows_to_scan: int = 200_000,
) -> dict[str, Any]:
    """
    Scan data dir: traverse + read meta + stats. Output fixed schema.

    Args:
        data_dir: path to data dir (e.g. public/)
        expand_subdirs: top-level dir names to expand (e.g. ["Deep", "Shallow"]). Default None.
        exclude_dirs: dir names to exclude from scanning (e.g. ["working", "cache"]). Default None.
        include_val_outputs: when True, do not hide ``validation*`` basenames in the tree
            (split_prep holdout files at dataset root).
        probe_binary_files: when True, run optional Python readers (.mat/.npy/...) on one
            sample file per extension under each listed directory (best-effort).
        walk_budget_dirs: max ``os.walk`` directory visits (None = unlimited).
        walk_budget_files: max files counted across the walk (None = unlimited).
        probe_binary_dirs_budget: max top-level dir entries that run binary probes.
        preview_raw_dirs_budget: max top-level dir entries that append raw text previews.
        meta_sample_max_bytes: cap JSON read size for meta parsing.
        csv_max_rows_to_scan: max CSV rows to scan for row_count / samples.

    Returns:
        {
            "path": str,
            "dir_structure": str,
            "meta": {...},
            "stats": {"total_files": N, "by_extension": {...}},
            "scan_budget": {...},
        }
    """
    if expand_subdirs is None:
        expand_subdirs = []
    if exclude_dirs is None:
        exclude_dirs = []
    exclude_set = set(exclude_dirs) | SCAN_EXCLUDE_DIRS
    root = Path(data_dir).resolve()
    if not root.exists() or not root.is_dir():
        return {"error": "path_not_found", "path": str(root)}

    root_str = str(root)
    with _ScanMetaCsvScope(meta_sample_max_bytes, csv_max_rows_to_scan):
        return _scan_data_dir_body(
            root,
            root_str,
            expand_subdirs,
            exclude_set,
            include_val_outputs,
            preview_raw_files,
            probe_binary_files,
            walk_budget_dirs,
            walk_budget_files,
            probe_binary_dirs_budget,
            preview_raw_dirs_budget,
        )


def _scan_data_dir_body(
    root: Path,
    root_str: str,
    expand_subdirs: list[str],
    exclude_set: set[str],
    include_val_outputs: bool,
    preview_raw_files: bool,
    probe_binary_files: bool,
    walk_budget_dirs: int | None,
    walk_budget_files: int | None,
    probe_binary_dirs_budget: int,
    preview_raw_dirs_budget: int,
) -> dict[str, Any]:
    """Inner implementation (runs under :class:`_ScanMetaCsvScope`)."""
    tree: dict[str, Any] = {}
    meta: dict[str, Any] = {}

    # Collect with a single os.walk: direct children per level and file count/extensions per level, avoiding multiple rglobs over the same subtree
    walk_entries: dict[str, tuple[list[str], list[str]]] = {}  # rel -> (dirnames, filenames)
    capped_dirs: set[str] = set()  # rels not fully traversed due to too many subdirectories; avoids timeouts from 800k+ files such as herbarium
    total_ext_counter: Counter = Counter()
    total_file_count = 0
    # Stop recursing when a single level has more direct children than this; record only "not expanded" to avoid timeouts
    WALK_MAX_DIRS = 80

    dirs_seen = 0
    walk_dirs_exceeded = False
    walk_files_exceeded = False

    # followlinks=True so symlinked dataset entries under workspace get correct file_count
    for dirpath, dirnames, filenames in os.walk(root_str, topdown=True, followlinks=True):
        dirs_seen += 1
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        rel = os.path.relpath(dirpath, root_str) if dirpath != root_str else "."
        rel = str(Path(rel).as_posix())  # normalize to / for consistency with tree keys
        if rel != "." and any(part in exclude_set for part in Path(rel).parts):
            continue
        if walk_budget_dirs is not None and dirs_seen > walk_budget_dirs:
            walk_dirs_exceeded = True
            dirnames[:] = []
        # Save the current level's subdirectory names and file list first, then decide whether to truncate (truncating clears dirnames, which only stops recursion without losing directory names)
        use_files: list[str] = []
        for f in filenames:
            if walk_budget_files is not None and total_file_count >= walk_budget_files:
                walk_files_exceeded = True
                break
            use_files.append(f)
            suf = Path(f).suffix.lstrip(".").lower() or "no_ext"
            total_ext_counter[suf] += 1
            total_file_count += 1
        walk_entries[rel] = (list(dirnames), use_files)
        if len(dirnames) > WALK_MAX_DIRS:
            capped_dirs.add(rel)
            dirnames[:] = []  # stop recursing; hundreds of thousands of files would time out prep (e.g. herbarium)
        if walk_files_exceeded:
            dirnames[:] = []

    # Compute the recursive file_count and ext_counter per directory bottom-up
    dir_stats: dict[str, dict[str, Any]] = {}
    for rel in sorted(walk_entries.keys(), key=lambda r: (-r.count("/"), r)):
        dirs_here, files_here = walk_entries[rel]
        fc = len(files_here)
        ext_c = Counter((Path(f).suffix.lstrip(".").lower() or "no_ext") for f in files_here)
        capped = rel in capped_dirs
        for d in dirs_here:
            sub_rel = f"{rel}/{d}" if rel != "." else d
            if sub_rel in capped_dirs:
                capped = True
            if sub_rel in dir_stats:
                fc += dir_stats[sub_rel].get("file_count") or 0
                ext_c += dir_stats[sub_rel].get("ext_counter") or Counter()
                capped = capped or dir_stats[sub_rel].get("capped", False)
        dir_stats[rel] = {"file_count": fc, "ext_counter": ext_c, "capped": capped}

    # 1. Top level: build tree from the walk's ".", with file_count from dir_stats
    top_dirs, top_files = walk_entries.get(".", ([], []))
    for rel in sorted(top_dirs + top_files):
        if rel in exclude_set or _hide_from_preview(rel, include_val_outputs):
            continue
        if rel in top_dirs:
            fc = dir_stats.get(rel, {}).get("file_count", 0)
            sub_dirs, sub_files = walk_entries.get(rel, ([], []))
            sub_sample = [(n + "/") for n in sub_dirs][:SUBDIRS_SHOW_LIMIT]
            tree[rel] = {"type": "dir", "file_count": fc, "sub_sample": sub_sample, "subdir_count": len(sub_dirs)}
        else:
            p = root / rel
            ext = (p.suffix.lower() or "no_ext").lstrip(".")
            tree[rel] = {"type": "file", "ext": ext}
            if p.suffix in META_READERS:
                m = META_READERS[p.suffix](p)
                if m is not None:
                    meta[rel] = m

    # 2. Meta files in subdirs (only a few csv/json inside top-level subdirectories)
    for d in list(tree.keys()):
        if tree[d].get("type") != "dir":
            continue
        dpath = root / d
        for ext, fn in META_READERS.items():
            for f in list(dpath.glob(f"*{ext}"))[:5]:
                if f.is_file() and not _hide_from_preview(f.name, include_val_outputs):
                    rel = f"{d}/{f.name}"
                    m = fn(f)
                    if m is not None:
                        meta[rel] = m

    # 3. Overall statistics reuse the walk-aggregated results
    ext_counter = total_ext_counter

    # 4. Per-directory dominant extension from dir_stats
    dir_ext: dict[str, str] = {}
    for n in tree:
        if tree[n].get("type") != "dir":
            continue
        exts = dir_stats.get(n, {}).get("ext_counter") or Counter()
        if exts:
            dir_ext[n] = "." + exts.most_common(1)[0][0]

    # 5. Build dir structure string
    dirs = sorted(n for n in tree if tree[n].get("type") == "dir")
    file_entries = [(n, tree[n]) for n in tree if tree[n].get("type") == "file"]

    def _dir_prefix(name: str) -> str:
        m = re.match(r"^([^\d]+)", name)
        return (m.group(1).rstrip("_-") or name) if m else name

    DIR_GROUP_THRESHOLD = 20
    prefix_to_dirs: dict[str, list[str]] = {}
    for d in dirs:
        prefix_to_dirs.setdefault(_dir_prefix(d), []).append(d)

    dir_items: list[tuple[str, str, dict]] = []
    for _pre in sorted(prefix_to_dirs.keys(), key=lambda p: (prefix_to_dirs[p][0] if prefix_to_dirs[p] else "")):
        dlist = prefix_to_dirs[_pre]
        if len(dlist) >= DIR_GROUP_THRESHOLD:
            total_fc = sum(tree[d]["file_count"] for d in dlist)
            exts = Counter()
            for d in dlist:
                exts += dir_stats.get(d, {}).get("ext_counter") or Counter()
            ext = "." + exts.most_common(1)[0][0] if exts else ""
            dir_items.append((f"{_pre}*/", "dir_group", {"dir_count": len(dlist), "file_count": total_fc, "ext": ext, "_first_dir": dlist[0] if dlist else None, "_ext_counter": exts}))
        else:
            for d in sorted(dlist):
                if not _hide_from_preview(d, include_val_outputs):
                    dir_items.append((d, "dir", tree[d]))

    FILE_GROUP_THRESHOLD = 15
    ext_groups: dict[str, list[str]] = {}
    for n, info in file_entries:
        ext = info.get("ext", "no_ext")
        if ext == "no_ext":
            ext = ""
        key = f".{ext}" if ext else "no_ext"
        ext_groups.setdefault(key, []).append(n)

    items_for_tree: list[tuple[str, str, dict]] = []
    for item in dir_items:
        items_for_tree.append(item)
    for ext_key, fnames in sorted(ext_groups.items(), key=lambda x: (x[0] != ".csv", x[0] != ".json", x[0])):
        if len(fnames) > FILE_GROUP_THRESHOLD:
            count = len(fnames)
            ext_dot = ext_key if ext_key not in (".no_ext", "no_ext") else ""
            items_for_tree.append((f"*{ext_dot}({count:,})", "group", {"count": count, "ext": ext_dot}))
        else:
            for n in sorted(fnames):
                if not _hide_from_preview(n, include_val_outputs):
                    items_for_tree.append((n, "file", tree[n]))

    probe_left = max(0, int(probe_binary_dirs_budget))
    raw_left = max(0, int(preview_raw_dirs_budget))

    lines = [root.name + "/"]
    if walk_dirs_exceeded or walk_files_exceeded:
        cap_note = f"# ... scan budget: dirs_visited={dirs_seen:,} files_counted={total_file_count:,}"
        if walk_dirs_exceeded:
            cap_note += " [walk_dirs_cap]"
        if walk_files_exceeded:
            cap_note += " [walk_files_cap]"
        lines.append(cap_note)

    for i, (n, typ, info) in enumerate(items_for_tree):
        is_last = i == len(items_for_tree) - 1
        prefix = "└── " if is_last else "├── "
        if typ == "dir":
            suffix = "/"
            fc = info.get("file_count", 0)
            exts_counter = dir_stats.get(n, {}).get("ext_counter") or Counter()
            ext = dir_ext.get(n, "")
            if dir_stats.get(n, {}).get("capped"):
                comment = " # (many, scan skipped)"
            elif "." in capped_dirs:
                # When the root is truncated, top-level subdirectories are not recursively scanned; only the names are kept
                comment = " # (not expanded)"
            else:
                # With multiple extensions, no longer use a single {id}.ext, which would mislead (e.g. a directory holding .mp4/.wav/.mat at once)
                if len(exts_counter) > 1:
                    ext_list = _ext_list_with_hints(exts_counter)
                    comment = f" # ({fc:,} files: {ext_list})"
                else:
                    id_col = "{id}"
                    for csv_name, csv_meta in meta.items():
                        if csv_name.endswith(".csv") and csv_name.startswith(n + "."):
                            cols = csv_meta.get("columns", [])
                            for c in ["image_name", "image_id", "StudyInstanceUID", "Id", "id", "fname"]:
                                if c in cols:
                                    id_col = "{" + c + "}"
                                    break
                            break
                    suf = f"{id_col}{ext}" if ext else ""
                    comment = f" # {suf}({fc:,})" if suf else f" # {fc:,} files"
                    hint = _binary_hint(ext.lstrip(".")) if ext else ""
                    if hint:
                        comment += f"  [{hint}]"
                sub_sample = info.get("sub_sample") or []
                subdir_count = info.get("subdir_count", len(sub_sample))
                if (sub_sample or subdir_count) and not dir_stats.get(n, {}).get("capped"):
                    comment += " subdirs: " + ", ".join(sub_sample[:SUBDIRS_SHOW_LIMIT])
                    if subdir_count > SUBDIRS_SHOW_LIMIT:
                        comment += f" ... ({subdir_count} total)"
        elif typ == "dir_group":
            suffix = ""
            dc, fc = info.get("dir_count", 0), info.get("file_count", 0)
            ext = info.get("ext", "")
            grp_exts = info.get("_ext_counter") or Counter()
            if "." in capped_dirs:
                comment = f" # (~{dc:,} subdirs, not expanded)"
                fd = info.get("_first_dir")
                if fd:
                    ec = _ext_counter_from_dir_files(root / fd)
                    if ec:
                        ext_list = _ext_list_with_hints(ec)
                        comment += f"; e.g. {fd}/: {ext_list}"
            else:
                if len(grp_exts) > 1:
                    ext_list = _ext_list_with_hints(grp_exts)
                    comment = f" # ({dc:,} subdirs, {fc:,} files: {ext_list})"
                else:
                    suf = f"{{id}}{ext}" if ext else ""
                    hint = _binary_hint(ext.lstrip(".")) if ext else ""
                    comment = f" # {suf}({dc:,} subdirs, {fc:,} files)" if suf else f" # ({dc:,} subdirs, {fc:,} files)"
                    if hint:
                        comment += f"  [{hint}]"

        elif typ == "group":
            comment = ""
        else:
            suffix = ""
            m = meta.get(n, {})
            comment = _meta_file_comment(m)
        lines.append(prefix + n + suffix + comment)
        if typ == "file" and n in meta:
            for sl in _format_meta_sample_lines(meta[n]):
                lines.append("│   " + sl)

        # Unexpanded subdirectories: if JSON/CSV meta is already loaded for them (from section 2), show 1 sample per extension
        if typ == "dir" and n not in expand_subdirs and not dir_stats.get(n, {}).get("capped"):
            for ext in (".json", ".csv"):
                sample_key = next(
                    (k for k in sorted(meta.keys())
                     if k.startswith(n + "/") and k.endswith(ext)),
                    None
                )
                if not sample_key:
                    continue
                sample_name = os.path.basename(sample_key)
                sample_meta = meta.get(sample_key)
                if sample_meta and ("columns" in sample_meta or sample_meta.get("sample_items") is not None or sample_meta.get("type")):
                    lines.append("│   └── [sample] " + sample_name)
                    for sl in _format_meta_sample_lines(sample_meta):
                        lines.append("│   " + sl)

        if typ == "dir" and n not in expand_subdirs and preview_raw_files:
            n_dir = root / n
            is_capped = dir_stats.get(n, {}).get("capped", False)
            sub_sample = info.get("sub_sample") or []
            if raw_left > 0:
                raw_left -= 1
                if is_capped or sub_sample:
                    _append_raw_preview_for_capped_or_nested(n_dir, lines)
                else:
                    n_exts = dir_stats.get(n, {}).get("ext_counter") or Counter()
                    _append_raw_preview_for_dir(n_dir, n_exts, lines, nested=False)

        if (
            typ == "dir"
            and n not in expand_subdirs
            and probe_binary_files
            and not dir_stats.get(n, {}).get("capped")
        ):
            n_dir = root / n
            n_exts_pb = dir_stats.get(n, {}).get("ext_counter") or Counter()
            if probe_left > 0:
                probe_left -= 1
                _append_binary_probe_samples(
                    n_dir, n_exts_pb, lines, probe_binary_files=True,
                )

        # When a large batch of same-prefix directories is merged into a dir_group, still binary-probe one of them
        # (otherwise the LLM cannot see structures like .mat under Sample*/ and is forced to explore via bash)
        if typ == "dir_group" and probe_binary_files:
            first_dir_name = info.get("_first_dir")
            if first_dir_name:
                sample_path = root / first_dir_name
                if sample_path.is_dir():
                    # Merge walk-aggregated exts with a direct listing of the sample dir
                    # (parent may be capped before per-Sample stats exist).
                    grp_exts_pb = Counter(info.get("_ext_counter") or Counter())
                    grp_exts_pb.update(_ext_counter_from_dir_files(sample_path))
                    if probe_left > 0:
                        probe_left -= 1
                        lines.append(
                            f"│   └── [dir_group sample: {first_dir_name}/] "
                            f"(binary probe from 1 of ~{info.get('dir_count', 0):,} dirs)"
                        )
                        _append_binary_probe_samples(
                            sample_path,
                            grp_exts_pb,
                            lines,
                            probe_binary_files=True,
                        )

        # Expand one level of the specified subdirectories (e.g. Deep/, Shallow/)
        if typ == "dir" and n in expand_subdirs:
            subdir_path = root / n
            subtree_lines = _build_subdir_tree_lines(
                subdir_path, meta, n, META_READERS, include_val_outputs,
                preview_raw_files=preview_raw_files,
                probe_binary_files=probe_binary_files,
            )
            for subline in subtree_lines:
                lines.append("│   " + subline)
    dir_structure = "\n".join(lines)

    return {
        "path": str(root),
        "dir_structure": dir_structure,
        "meta": {
            k: ({"columns": v["columns"]} if "columns" in v else {"type": v.get("type"), "keys": v.get("keys", [])[:JSON_KEY_SHOW_LIMIT + 3]})
            for k, v in meta.items()
        },
        "stats": {"total_files": total_file_count, "by_extension": dict(ext_counter.most_common(15))},
        "scan_budget": {
            "walk_dirs_exceeded": walk_dirs_exceeded,
            "walk_files_exceeded": walk_files_exceeded,
            "dirs_visited": dirs_seen,
            "files_counted": total_file_count,
            "probe_binary_remaining": probe_left,
            "preview_raw_remaining": raw_left,
        },
    }


def extract_eval_signature(py_path: Path) -> str | None:
    """Extract the eval() function's def-line + docstring from a workspace_eval.py file.

    Returns a code snippet like:
        def eval(x: List[int]) -> float:
            \"\"\"
            Evaluate predictions for ...
            Args: ...
            Returns: ...
            \"\"\"
    or None if not found / parse error.
    """
    import ast

    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "eval":
            continue

        lines = source.splitlines()
        sig_lines: list[str] = []
        for i in range(node.lineno - 1, min(node.lineno + 10, len(lines))):
            sig_lines.append(lines[i])
            if lines[i].rstrip().endswith(":"):
                break
        sig = "\n".join(sig_lines)

        docstring = ast.get_docstring(node)
        if docstring:
            doc_lines = docstring.splitlines()
            sig += '\n    """\n'
            for dl in doc_lines:
                sig += f"    {dl}\n"
            sig += '    """'

        return sig

    return None


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "./data/mlebench_data/herbarium-2022-fgvc9/prepared/public"
    r = scan_data_dir(path)
    print(r.get("dir_structure", json.dumps(r)))
