"""Dataset preparation helpers for SpreadsheetBench."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path


def prepare_data_root(data_root: Path, data_url: str, *, download: bool) -> Path:
    """Ensure SpreadsheetBench Verified-400 exists locally and return its root."""
    resolved = resolve_data_root(data_root)
    if resolved is not None:
        return resolved
    if not download:
        raise FileNotFoundError(
            f"SpreadsheetBench data not found under {data_root}. Re-run with --download or provide --data-root."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / "spreadsheetbench_verified_400.tar.gz"
    if not archive_path.exists():
        print(f"Downloading SpreadsheetBench data:\n  {data_url}\n  -> {archive_path}")
        urllib.request.urlretrieve(data_url, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    print(f"Extracting {archive_path} into {data_root}")
    safe_extract_tar(archive_path, data_root)
    resolved = resolve_data_root(data_root)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find spreadsheetbench_verified_400/dataset.json after extracting {archive_path}."
        )
    return resolved


def resolve_data_root(path: Path) -> Path | None:
    """Return the directory whose child is ``spreadsheetbench_verified_400``."""
    candidates = [
        path,
        path / "SpreadsheetBench",
        path / "spreadsheetbench_verified_400",
    ]
    for candidate in candidates:
        if candidate.name == "spreadsheetbench_verified_400" and (candidate / "dataset.json").exists():
            return candidate.parent
        if (candidate / "spreadsheetbench_verified_400" / "dataset.json").exists():
            return candidate

    for dataset_json in path.glob("**/spreadsheetbench_verified_400/dataset.json"):
        return dataset_json.parent.parent
    return None


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a tar archive while rejecting path traversal members."""
    destination = destination.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe tar member: {member.name}")
        archive.extractall(destination, filter="data")
