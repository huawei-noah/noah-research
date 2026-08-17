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

import json
from pathlib import Path

import pytest

from scienceflow.solver.lnr.replay_prepare import ReplayPrepareError, prepare_lnr_replay
from scienceflow.solver.lnr.resume.memory_state import inspect_resume_memory_messages


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _records() -> list[dict]:
    return [
        {"role": "user", "message": {"role": "user", "content": "start"}},
        {
            "role": "assistant",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_train",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {"cmd": "python train.py", "yield_time_ms": 1000},
                                sort_keys=True,
                            ),
                        },
                    }
                ],
            },
        },
        {
            "role": "tool",
            "message": {
                "role": "tool",
                "tool_call_id": "call_train",
                "content": "already completed",
            },
        },
        {"role": "assistant", "message": {"role": "assistant", "content": "done"}},
    ]


def _source_case(tmp_path: Path, *, short_has_target: bool = True, write_long: bool = True) -> tuple[Path, Path, Path]:
    source = tmp_path / "source_case"
    agent_dir = (
        source
        / "run"
        / "run_a"
        / "task_a"
        / "task_logs"
        / "memory"
        / "ScienceAgent"
    )
    short_memory = agent_dir / "short_term.json"
    long_memory = agent_dir / "long_term.jsonl"
    records = _records()
    _write_jsonl(short_memory, records if short_has_target else records[:1])
    if write_long:
        _write_jsonl(long_memory, records)
    (source / "replay.yaml").write_text(
        "defaults:\n"
        "  workspace_base: /old/workspace/base\n"
        "tasks:\n"
        "  - run_id: run_a\n"
        "    task_id: task_a\n",
        encoding="utf-8",
    )
    return source, short_memory, long_memory


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_lnr_replay_prefers_long_term_and_materializes_short_term(tmp_path: Path) -> None:
    source, _short_memory, _long_memory = _source_case(tmp_path, short_has_target=False)
    output = tmp_path / "replay"

    result = prepare_lnr_replay(source, output, tool_call_id="call_train")

    copied_short = Path(result.short_term_path)
    copied_long = Path(result.long_term_path)
    short_records = _read_jsonl(copied_short)
    long_records = _read_jsonl(copied_long)
    state = inspect_resume_memory_messages(short_records)

    assert result.memory_source_kind == "long_term"
    assert result.memory_path == result.short_term_path
    assert result.records_before == 4
    assert result.records_after == 2
    assert result.selected_message_index == 1
    assert result.tool_name == "bash"
    assert result.command == "python train.py"
    assert state.action == "execute_pending_tool"
    assert len(state.pending_tool_calls) == 1
    assert state.pending_tool_calls[0].tool_call_id == "call_train"
    assert short_records == long_records
    assert "/replay/run" in (output / "replay.yaml").read_text(encoding="utf-8")
    assert (output / "replay_prepare_manifest.json").is_file()


def test_prepare_lnr_replay_accepts_explicit_short_term_relative_path(tmp_path: Path) -> None:
    source, short_memory, _long_memory = _source_case(tmp_path)
    output = tmp_path / "replay"
    rel_memory = short_memory.relative_to(source)

    result = prepare_lnr_replay(source, output, message_index=1, memory_path=rel_memory)

    assert result.memory_source_kind == "short_term"
    assert result.selected_message_index == 1
    assert Path(result.short_term_path).relative_to(output.resolve()) == rel_memory
    assert Path(result.long_term_path).is_file()
    assert len(_read_jsonl(Path(result.long_term_path))) == 2


def test_prepare_lnr_replay_falls_back_to_short_term_without_long_term(tmp_path: Path) -> None:
    source, _short_memory, _long_memory = _source_case(tmp_path, write_long=False)

    result = prepare_lnr_replay(source, tmp_path / "replay", tool_call_id="call_train")

    assert result.memory_source_kind == "short_term"
    assert Path(result.long_term_path).is_file()


def test_prepare_lnr_replay_rejects_non_tool_call_target(tmp_path: Path) -> None:
    source, _short_memory, _long_memory = _source_case(tmp_path)

    with pytest.raises(ReplayPrepareError, match="not an assistant message"):
        prepare_lnr_replay(source, tmp_path / "bad_replay", message_index=2)
