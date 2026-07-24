"""Benchmark matrix manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunManifest:
    """Manifest row emitted after a benchmark run."""

    benchmark: str
    run_id: str
    key_id: str
    project_id: str
    memory_algorithm: str
    api_key_file: str
    request_ids: dict[str, list[str]]
    request_metadata: dict[str, dict[str, Any]]
    eval_result: dict[str, Any]
    started_at: str
    finished_at: str

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest row."""
        return {
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "key_id": self.key_id,
            "project_id": self.project_id,
            "memory_algorithm": self.memory_algorithm,
            "api_key_file": self.api_key_file,
            "request_ids": self.request_ids,
            "request_metadata": self.request_metadata,
            "eval_result": self.eval_result,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def append_manifest(path: str | Path, manifest: BenchmarkRunManifest) -> None:
    """Append one manifest row as JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest.to_jsonable(), ensure_ascii=False) + "\n")


def write_manifests(path: str | Path, manifests: list[BenchmarkRunManifest]) -> None:
    """Overwrite a manifest JSONL file with the current run's rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for manifest in manifests:
            fh.write(json.dumps(manifest.to_jsonable(), ensure_ascii=False) + "\n")
