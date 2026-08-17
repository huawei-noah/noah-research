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

"""Budget / fast-path behavior for :func:`scienceflow.utils.data_scan.scan_data_dir`."""

from __future__ import annotations

from pathlib import Path

from scienceflow.utils.data_scan import scan_data_dir, _walk_count_files_and_exts


def test_walk_count_files_and_exts_single_pass(tmp_path: Path) -> None:
    d = tmp_path / "sub"
    d.mkdir()
    (d / "a.txt").write_text("x")
    (d / "b.txt").write_text("y")
    (d / "nest").mkdir()
    (d / "nest" / "c.bin").write_bytes(b"\x00")
    fc, exts = _walk_count_files_and_exts(d)
    assert fc == 3
    assert exts.get("txt") == 2
    assert exts.get("bin") == 1


def test_scan_data_dir_walk_budget_emits_flag(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    for i in range(5):
        (root / f"d{i}").mkdir()
        (root / f"d{i}" / "f.txt").write_text("ok")
    r = scan_data_dir(
        root,
        walk_budget_dirs=3,
        walk_budget_files=10_000,
        probe_binary_dirs_budget=0,
        preview_raw_dirs_budget=0,
    )
    assert r.get("scan_budget", {}).get("walk_dirs_exceeded") is True
    ds = r.get("dir_structure", "")
    assert "walk_dirs_cap" in ds or "[walk_dirs_cap]" in ds


def test_scan_data_dir_returns_scan_budget_keys(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    r = scan_data_dir(tmp_path)
    sb = r.get("scan_budget")
    assert isinstance(sb, dict)
    assert "files_counted" in sb
    assert "probe_binary_remaining" in sb
