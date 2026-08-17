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

"""Tests for embedded full-run submission handling and memory isolation."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from scienceflow.core.agent.run_control.embedded_fullrun import (
    EmbeddedFullRunMixin,
    _read_last_jsonl_tool_content,
    _rewrite_last_jsonl_tool_content,
    submission_rows_plausible_for_progressive_bare,
)


def _write_csv(path: Path, header: str, n_data: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header]
    lines.extend([f"{i},x" for i in range(n_data)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_submission_rows_plausible_ok_when_matching_test(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    ds = ws / "dataset"
    _write_csv(ds / "test.csv", "id,v", 100)
    _write_csv(ws / "submission.csv", "id,v", 100)
    ok, msg = submission_rows_plausible_for_progressive_bare(ws)
    assert ok, msg


def test_submission_rows_plausible_fails_when_too_short(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    ds = ws / "dataset"
    _write_csv(ds / "test.csv", "id,v", 1000)
    _write_csv(ws / "submission.csv", "id,v", 100)
    ok, msg = submission_rows_plausible_for_progressive_bare(ws)
    assert not ok
    assert "100" in msg and "1000" in msg


def test_submission_rows_plausible_ok_without_baseline_dataset(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    _write_csv(ws / "submission.csv", "id,v", 10)
    ok, msg = submission_rows_plausible_for_progressive_bare(ws)
    assert ok, msg
    assert "compare" in msg.lower()


# ---------------------------------------------------------------------------
# JSONL helper tests
# ---------------------------------------------------------------------------

def _make_tool_jsonl_record(content: str) -> str:
    """Return a single JSONL line that looks like a MemoryRecord with role='tool'."""
    return json.dumps({"message": {"role": "tool", "content": content}, "role": "tool"})


def _make_user_jsonl_record(content: str) -> str:
    return json.dumps({"message": {"role": "user", "content": content}, "role": "user"})


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _last_content(path: Path) -> str:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return json.loads(lines[-1])["message"]["content"]


def test_read_last_jsonl_tool_content_ok(tmp_path: Path) -> None:
    p = tmp_path / "lt.jsonl"
    _write_jsonl(p, [
        _make_user_jsonl_record("first msg"),
        _make_tool_jsonl_record("Final Score: 0.07"),
    ])
    result = _read_last_jsonl_tool_content(p)
    assert result == "Final Score: 0.07"


def test_read_last_jsonl_tool_content_noop_when_last_not_tool(tmp_path: Path) -> None:
    p = tmp_path / "lt.jsonl"
    _write_jsonl(p, [
        _make_tool_jsonl_record("some output"),
        _make_user_jsonl_record("a user message"),
    ])
    assert _read_last_jsonl_tool_content(p) is None


def test_rewrite_last_jsonl_tool_content_ok(tmp_path: Path) -> None:
    p = tmp_path / "st.json"
    _write_jsonl(p, [
        _make_tool_jsonl_record("original content"),
    ])
    ok = _rewrite_last_jsonl_tool_content(p, "original content\n\n[NODE-GATE OK] validator=sdk")
    assert ok
    assert "[NODE-GATE OK]" in _last_content(p)


def test_rewrite_last_jsonl_tool_content_noop_when_not_tool(tmp_path: Path) -> None:
    p = tmp_path / "st.json"
    _write_jsonl(p, [
        _make_user_jsonl_record("user stuff"),
    ])
    original = p.read_bytes()
    ok = _rewrite_last_jsonl_tool_content(p, "should not overwrite")
    assert not ok
    assert p.read_bytes() == original


# ---------------------------------------------------------------------------
# _append_to_last_tool_message / _inject_run_control_ok_message integration tests
# (use real JSONL files, SimpleNamespace stub memory — no MagicMock for memory)
# ---------------------------------------------------------------------------

def _make_stub_agent(tmp_path: Path, *, superloop: bool = False) -> SimpleNamespace:
    """Build agent stub backed by real JSONL files."""
    short = tmp_path / "short_term.json"
    long_ = tmp_path / "long_term.jsonl"
    # Seed both with a prior user message + a tool message
    _write_jsonl(short, [
        _make_user_jsonl_record("task context"),
        _make_tool_jsonl_record("Final Validation Score: 0.0829"),
    ])
    _write_jsonl(long_, [
        _make_user_jsonl_record("task context"),
        _make_tool_jsonl_record("Final Validation Score: 0.0829"),
    ])
    agent = SimpleNamespace(
        _lnr_superloop_enabled=superloop,
        memory=SimpleNamespace(
            chat_history_memory=SimpleNamespace(
                storage=SimpleNamespace(json_path=str(short))
            ),
            long_term_log=str(long_),
        ),
        _short_path=short,
        _long_path=long_,
    )
    agent._append_to_last_tool_message = (
        EmbeddedFullRunMixin._append_to_last_tool_message.__get__(agent, type(agent))
    )
    agent._inject_run_control_ok_message = (
        EmbeddedFullRunMixin._inject_run_control_ok_message.__get__(agent, type(agent))
    )
    return agent


def _write_submission(ws: Path, n: int = 10) -> Path:
    sub = ws / "submission.csv"
    sub.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,pred"] + [f"{i},0.5" for i in range(n)]
    sub.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sub


def test_append_writes_to_both_jsonl_files(tmp_path: Path) -> None:
    """_append_to_last_tool_message rewrites both short_term.json and long_term.jsonl."""
    agent = _make_stub_agent(tmp_path)
    agent._append_to_last_tool_message("[audit-marker] validator=sdk rows=240 exec_wall=18.4s")
    for path in (agent._short_path, agent._long_path):
        content = _last_content(path)
        assert "[audit-marker]" in content
        assert "Final Validation Score" in content  # original content preserved


def test_inject_run_control_ok_noops_for_llm_visible_memory(tmp_path: Path) -> None:
    """Node-gate success is internal and must not mutate inherited tool memory."""
    sub_dir = tmp_path / "ws"
    sub = _write_submission(sub_dir, n=240)
    agent = _make_stub_agent(tmp_path, superloop=True)
    orig_short = agent._short_path.read_bytes()
    orig_long = agent._long_path.read_bytes()

    agent._inject_run_control_ok_message(
        validator="sdk",
        submission_csv=sub,
        wall_sec=18.4,
        context_json={"phase_remaining_sec": 842.6},
    )
    assert agent._short_path.read_bytes() == orig_short
    assert agent._long_path.read_bytes() == orig_long


def test_inject_run_control_ok_no_phase_remaining(tmp_path: Path) -> None:
    """Deprecated inject helper remains a no-op even without phase_remaining."""
    sub_dir = tmp_path / "ws"
    sub = _write_submission(sub_dir, n=10)
    agent = _make_stub_agent(tmp_path, superloop=True)
    orig_short = agent._short_path.read_bytes()
    orig_long = agent._long_path.read_bytes()

    agent._inject_run_control_ok_message(
        validator="light",
        submission_csv=sub,
        wall_sec=5.0,
        context_json={},
    )
    assert agent._short_path.read_bytes() == orig_short
    assert agent._long_path.read_bytes() == orig_long


def test_append_noop_when_last_not_tool(tmp_path: Path) -> None:
    """When last JSONL line is not a tool record, no modification is made."""
    short = tmp_path / "short_term.json"
    long_ = tmp_path / "long_term.jsonl"
    _write_jsonl(short, [_make_user_jsonl_record("user stuff")])
    _write_jsonl(long_, [_make_user_jsonl_record("user stuff")])
    orig_short = short.read_bytes()
    orig_long = long_.read_bytes()

    # Build agent stub manually (last record is user, not tool)
    stub = SimpleNamespace(
        _lnr_superloop_enabled=True,
        memory=SimpleNamespace(
            chat_history_memory=SimpleNamespace(
                storage=SimpleNamespace(json_path=str(short))
            ),
            long_term_log=str(long_),
        ),
    )
    stub._append_to_last_tool_message = (
        EmbeddedFullRunMixin._append_to_last_tool_message.__get__(stub, type(stub))
    )
    stub._append_to_last_tool_message("[audit-marker] validator=sdk rows=5 exec_wall=3.0s")
    # Files must be byte-identical
    assert short.read_bytes() == orig_short
    assert long_.read_bytes() == orig_long


def test_append_works_when_only_long_term_exists(tmp_path: Path) -> None:
    """Only long_term.jsonl present (no short_term.json): still rewrites long_term."""
    long_ = tmp_path / "long_term.jsonl"
    _write_jsonl(long_, [_make_tool_jsonl_record("score 0.05")])
    stub = SimpleNamespace(
        _lnr_superloop_enabled=True,
        memory=SimpleNamespace(
            chat_history_memory=SimpleNamespace(
                storage=SimpleNamespace(json_path=None)
            ),
            long_term_log=str(long_),
        ),
    )
    stub._append_to_last_tool_message = (
        EmbeddedFullRunMixin._append_to_last_tool_message.__get__(stub, type(stub))
    )
    stub._append_to_last_tool_message("[audit-marker] validator=sdk rows=5 exec_wall=3.0s")
    assert "[audit-marker]" in _last_content(long_)


def test_opt_solver_bare_run_skips_legacy_score_contract(tmp_path: Path) -> None:
    """Artifact evaluator profiles must not inject MLEBench score-contract repairs."""
    ws = tmp_path / "ws"
    ws.mkdir()
    injected: list[str] = []
    stub = SimpleNamespace(
        _embedded_full_run_enabled=False,
        _lnr_allow_any_stage_script=True,
        _workspace_dir=ws,
        _scienceflow_task_profile="opt_solver",
        _scienceflow_evaluator_backend="artifact_command",
        _lnr_candidate_artifact_rel="artifacts/best_solution.json",
        _log_info=MagicMock(),
        _log_warning=MagicMock(),
    )
    stub._inject_run_control_user_message = injected.append
    stub._maybe_write_bare_run_tail_snapshot = (
        EmbeddedFullRunMixin._maybe_write_bare_run_tail_snapshot.__get__(stub, type(stub))
    )

    asyncio.run(
        stub._maybe_write_bare_run_tail_snapshot(
            {"command": "python train.py"},
            SimpleNamespace(output="Final Validation Score: 1.0\n", error=False),
        )
    )

    assert stub._lnr_snapshot_ok is False
    assert stub._lnr_snapshot_reason == "artifact_evaluator_primary"
    assert injected == []
