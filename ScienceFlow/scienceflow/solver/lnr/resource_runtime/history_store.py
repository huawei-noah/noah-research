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

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ResourceHistoryStore:
    """Small task-level runtime history used for conservative admission ETA."""

    def __init__(self, path: Path, *, max_records: int = 200) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.max_records = max(20, int(max_records or 200))

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"version": 1, "records": []}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        if not isinstance(data.get("records"), list):
            data["records"] = []
        data.setdefault("version", 1)
        return data

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=True, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            state = self._read_unlocked()
            try:
                yield state
            finally:
                self._write_unlocked(state)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        clean = sorted(float(x) for x in values if float(x) >= 0.0)
        if not clean:
            return 0.0
        if len(clean) == 1:
            return clean[0]
        pos = max(0.0, min(1.0, float(q))) * (len(clean) - 1)
        lo = int(pos)
        hi = min(len(clean) - 1, lo + 1)
        frac = pos - lo
        return clean[lo] * (1.0 - frac) + clean[hi] * frac

    def record_completion(
        self,
        *,
        job_id: str,
        worker_id: str,
        resource_class: str,
        gpu_ids: list[str],
        elapsed_sec: float,
        status: str,
        command_digest: str = "",
        entrypoint: str = "",
        peak_memory_used_gb: float | None = None,
    ) -> dict[str, Any]:
        elapsed = max(0.0, float(elapsed_sec or 0.0))
        if elapsed <= 0.0:
            return {"recorded": False, "reason": "non_positive_elapsed"}
        now = time.time()
        record = {
            "job_id": str(job_id or ""),
            "worker_id": str(worker_id or ""),
            "resource_class": str(resource_class or ""),
            "gpu_ids": [str(x) for x in (gpu_ids or []) if str(x).strip()],
            "elapsed_sec": elapsed,
            "status": str(status or "finished"),
            "command_digest": str(command_digest or ""),
            "entrypoint": str(entrypoint or ""),
            "peak_memory_used_gb": float(peak_memory_used_gb or 0.0),
            "finished_at": now,
        }
        with self._locked_state() as state:
            records = [x for x in (state.get("records") or []) if isinstance(x, dict)]
            records.append(record)
            if len(records) > self.max_records:
                records = records[-self.max_records :]
            state["records"] = records
            summary = self.summary(resource_class=record["resource_class"], _records=records)
        return {"recorded": True, "record": record, "summary": summary}

    def summary(
        self,
        *,
        resource_class: str = "",
        command_digest: str = "",
        entrypoint: str = "",
        _records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if _records is None:
            state = self._read_unlocked()
            records = [x for x in (state.get("records") or []) if isinstance(x, dict)]
        else:
            records = [x for x in _records if isinstance(x, dict)]
        cls = str(resource_class or "")
        digest = str(command_digest or "")
        ep = str(entrypoint or "")
        filtered = [x for x in records if not cls or str(x.get("resource_class") or "") == cls]
        if digest:
            digest_filtered = [x for x in filtered if str(x.get("command_digest") or "") == digest]
            if digest_filtered:
                filtered = digest_filtered
        if ep:
            ep_filtered = [x for x in filtered if str(x.get("entrypoint") or "") == ep]
            if ep_filtered:
                filtered = ep_filtered
        if not filtered:
            return {"count": 0, "resource_class": cls}
        values = [float(x.get("elapsed_sec") or 0.0) for x in filtered if float(x.get("elapsed_sec") or 0.0) > 0.0]
        if not values:
            return {"count": 0, "resource_class": cls}
        return {
            "count": len(values),
            "resource_class": cls,
            "command_digest": digest,
            "entrypoint": ep,
            "mean_sec": sum(values) / len(values),
            "p50_sec": self._percentile(values, 0.50),
            "p80_sec": self._percentile(values, 0.80),
            "p95_sec": self._percentile(values, 0.95),
            "last_elapsed_sec": values[-1],
        }

    def snapshot(self) -> dict[str, Any]:
        state = self._read_unlocked()
        records = [x for x in (state.get("records") or []) if isinstance(x, dict)]
        classes = sorted({str(x.get("resource_class") or "") for x in records if str(x.get("resource_class") or "")})
        return {
            "version": int(state.get("version") or 1),
            "record_count": len(records),
            "classes": {cls: self.summary(resource_class=cls, _records=records) for cls in classes},
        }
