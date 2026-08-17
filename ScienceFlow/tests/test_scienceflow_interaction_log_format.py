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

"""Tests for workspace interaction.log formatting helpers."""

import pytest

from scienceflow.config.settings import Config, _apply_env
from scienceflow.safety.execution_policy import format_full_run_log_body
from scienceflow.utils.workspace_interaction_log import (
    BashStreamFilter,
    ConsecutiveLineDeduper,
    FullRunLightGBMStreamDeduper,
    collapse_consecutive_lightgbm_warnings_in_text,
    collapse_consecutive_repeated_lines_in_text,
    normalize_lightgbm_warning_line,
    strip_interaction_ansi,
)

from scienceflow.core.agent import (
    _WS_LOG_BODY_CAP,
    _WS_LOG_MAX_LINES,
    format_tool_call_lines_for_interaction_log,
    format_tool_result_for_interaction_log,
    resolve_interaction_log_policy,
    tool_result_text_for_interaction_log,
    truncate_for_interaction_log,
)


def test_write_multiline_and_truncation_when_not_full() -> None:
    content = "line1\nline2\n" + ("x" * (_WS_LOG_BODY_CAP + 100))
    lines = format_tool_call_lines_for_interaction_log(
        "write",
        {"path": "a.py", "content": content},
        full=False,
    )
    # When full=False, only the [tool-call] meta line is emitted (no body).
    assert len(lines) == 1
    assert "[tool-call] write" in lines[0]
    assert "WRITE_OK memory-compression" in lines[0]
    assert "sha256:" in lines[0]


def test_write_full_omits_body() -> None:
    """full=True still logs only meta for write; file content is not duplicated in interaction.log."""
    content = "a\nb\nc\n" + ("y" * 5000)
    lines = format_tool_call_lines_for_interaction_log(
        "write",
        {"path": "b.py", "content": content},
        full=True,
    )
    assert len(lines) == 1
    assert "[tool-call] write" in lines[0]
    assert "WRITE_OK memory-compression" in lines[0]
    assert "sha256:" in lines[0]
    assert content not in lines[0]
    assert "[tool-call-body]" not in lines[0]
    assert "---- begin ----" not in lines[0]


def test_edit_two_bodies() -> None:
    lines = format_tool_call_lines_for_interaction_log(
        "edit",
        {
            "path": "f.py",
            "old_str": "old\nold",
            "new_str": "new",
        },
        full=True,
    )
    assert len(lines) == 3
    assert ".old_str" in lines[1]
    assert "old\nold" in lines[1]
    assert ".new_str" in lines[2]
    assert "new" in lines[2]


def test_bash_command_body() -> None:
    lines = format_tool_call_lines_for_interaction_log(
        "bash",
        {"command": "echo\nhi"},
        full=True,
    )
    assert len(lines) == 2
    assert "echo\nhi" in lines[1]


def test_read_fallback_single_line() -> None:
    lines = format_tool_call_lines_for_interaction_log(
        "read",
        {"path": "x.txt"},
        full=False,
    )
    assert len(lines) == 1
    assert lines[0].startswith("[tool-call] read ")


def test_truncate_for_interaction_log_keeps_first_n_lines() -> None:
    lines = [f"line{i}\n" for i in range(_WS_LOG_MAX_LINES + 5)]
    text = "".join(lines)
    out = truncate_for_interaction_log(text)
    assert out.startswith("line0\n")
    assert "truncated" in out
    assert f"{_WS_LOG_MAX_LINES + 5} lines" in out
    assert str(len(text)) in out
    assert out.count("\n") <= _WS_LOG_MAX_LINES + 3  # kept lines + summary line


def test_truncate_for_interaction_log_short_unchanged() -> None:
    assert truncate_for_interaction_log("a\nb") == "a\nb"


def test_apply_env_scienceflow_interaction_log_llm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config()
    assert cfg.scienceflow_interaction_log_llm_stream is False
    monkeypatch.setenv("SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM", "0")
    _apply_env(cfg)
    assert cfg.scienceflow_interaction_log_llm_stream is False
    monkeypatch.setenv("SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM", "1")
    _apply_env(cfg)
    assert cfg.scienceflow_interaction_log_llm_stream is True


def test_apply_env_scienceflow_interaction_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config()
    assert cfg.scienceflow_interaction_log_level == "minimal"
    monkeypatch.setenv("SCIENCEFLOW_INTERACTION_LOG_LEVEL", "minimal")
    _apply_env(cfg)
    assert cfg.scienceflow_interaction_log_level == "minimal"
    monkeypatch.setenv("SCIENCEFLOW_INTERACTION_LOG_LEVEL", "verbose")
    _apply_env(cfg)
    assert cfg.scienceflow_interaction_log_level == "verbose"


def test_apply_env_feedback_api_keys_do_not_pollute_code(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "API_KEY",
        "API_KEYS",
        "BASE_URL",
        "BASE_URLS",
        "CODE_API_KEY",
        "CODE_API_KEYS",
        "CODE_BASE_URL",
        "CODE_BASE_URLS",
        "FEEDBACK_API_KEY",
        "FEEDBACK_API_KEYS",
        "FEEDBACK_BASE_URL",
        "FEEDBACK_BASE_URLS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FEEDBACK_API_KEYS", "fb-a fb-b")
    monkeypatch.setenv("FEEDBACK_BASE_URL", "https://feedback.example/v1")

    cfg = Config()
    _apply_env(cfg)

    assert cfg.agent.code.api_keys == []
    assert cfg.agent.code.api_key == ""
    assert cfg.agent.feedback.api_keys == ["fb-a", "fb-b"]
    assert cfg.agent.feedback.base_url == "https://feedback.example/v1"


def test_apply_env_code_and_feedback_stage_specific_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("API_KEY", "API_KEYS", "BASE_URL", "BASE_URLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CODE_API_KEY", "code-key")
    monkeypatch.setenv("CODE_BASE_URL", "https://code.example/v1")
    monkeypatch.setenv("FEEDBACK_API_KEY", "feedback-key")
    monkeypatch.setenv("FEEDBACK_BASE_URL", "https://feedback.example/v1")

    cfg = Config()
    _apply_env(cfg)

    assert cfg.agent.code.api_key == "code-key"
    assert cfg.agent.code.base_url == "https://code.example/v1"
    assert cfg.agent.feedback.api_key == "feedback-key"
    assert cfg.agent.feedback.base_url == "https://feedback.example/v1"


def test_bash_command_preview_normal_shows_short_command() -> None:
    pol = resolve_interaction_log_policy(level="normal", legacy_full=False, legacy_llm_stream=False)
    lines = format_tool_call_lines_for_interaction_log(
        "bash",
        {"command": "python3 solution.py"},
        policy=pol,
    )
    assert len(lines) == 1
    assert "python3 solution.py" in lines[0]


def test_bash_command_preview_minimal_truncates_long_command() -> None:
    pol = resolve_interaction_log_policy(level="minimal", legacy_full=False, legacy_llm_stream=False)
    long_cmd = "x" * 120
    lines = format_tool_call_lines_for_interaction_log("bash", {"command": long_cmd}, policy=pol)
    assert len(lines) == 1
    assert "truncated" in lines[0]
    assert "xxxxx" in lines[0]


def test_collapse_consecutive_repeated_lines_in_text_min_repeat() -> None:
    w = "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf"
    block = "\n".join([w] * 20 + ["Final Validation Score: 0.42", w, w, w])
    got = collapse_consecutive_repeated_lines_in_text(block, min_repeat=3)
    assert got.count(w) == 2  # one preserved + one before metric break + one in tail run first line
    assert "log-dedup" in got
    assert "Final Validation Score: 0.42" in got
    assert got.count("log-dedup") == 2


def test_collapse_consecutive_repeated_lines_preserves_order_non_repeated() -> None:
    lines = ["a", "b", "a", "c"]
    got = collapse_consecutive_repeated_lines_in_text("\n".join(lines), min_repeat=3)
    assert got == "a\nb\na\nc"


def test_consecutive_line_deduper_stream_lightgbm_warnings() -> None:
    out: list[str] = []
    d = ConsecutiveLineDeduper(min_repeat=3, summary_prefix="[log-dedup]")
    w = "[LightGBM] [Warning] x"
    for ln in [w] * 20:
        out.extend(d.feed_line(ln))
    out.extend(d.flush())
    assert w in out
    assert sum(1 for x in out if x == w) == 1
    assert any("log-dedup" in x and "19" in x for x in out)
    out2: list[str] = []
    d2 = ConsecutiveLineDeduper(min_repeat=3)
    for ln in ["epoch 1", "epoch 2"]:
        out2.extend(d2.feed_line(ln))
    out2.extend(d2.flush())
    assert out2 == ["epoch 1", "epoch 2"]


def test_bash_stream_filter_dedupes_repeated_kept_lines() -> None:
    emitted: list[str] = []

    def _emit(msg: str) -> None:
        emitted.append(msg)

    f = BashStreamFilter(
        _emit,
        prefix="[bash-stream] ",
        dedup_enabled=True,
        dedup_min_repeat=3,
    )
    # Lines must pass BashStreamFilter whitelist (LightGBM warnings are blacklist-filtered).
    w = "[epoch 1] training progress"
    chunk = "\n".join([w] * 15) + "\n"
    f.feed_chunk(chunk)
    f.flush()
    body = "".join(emitted)
    assert body.count(w) == 1
    assert "log-dedup" in body


def test_bash_tool_result_dedups_before_truncation() -> None:
    pol = resolve_interaction_log_policy(level="normal", legacy_full=False, legacy_llm_stream=False)
    spam = "same spam line"
    body = "[exit=0, 1.0s]\n" + "\n".join([spam] * 10)
    out = tool_result_text_for_interaction_log("bash", body, pol)
    assert spam in out
    assert out.count(spam) == 1
    assert "log-dedup" in out


def test_bash_tool_result_for_interaction_log_strips_stderr_section() -> None:
    pol = resolve_interaction_log_policy(level="normal", legacy_full=False, legacy_llm_stream=False)
    raw = "[exit=0, 6.5s]\nFinal Validation Score: 0.095\n\n[stderr]\nUserWarning: x"
    out = tool_result_text_for_interaction_log("bash", raw, pol)
    assert "Final Validation Score" in out
    assert "[stderr]" not in out
    assert "UserWarning" not in out

    err_prefixed = "Error: non-zero exit code 1\n[exit=1, 1.0s]\nstdout line\n[stderr]\nerr only"
    out2 = tool_result_text_for_interaction_log("bash", err_prefixed, pol)
    assert "stdout line" in out2
    assert "err only" not in out2


def test_tool_result_truncation() -> None:
    s = "z" * 3000
    out = format_tool_result_for_interaction_log(s, full=False)
    assert len(out) < len(s)
    assert "truncated" in out

    out2 = format_tool_result_for_interaction_log(s, full=True)
    assert out2 == s


def test_tool_result_truncation_multiline_keeps_five_lines() -> None:
    lines = [f"line{i}" for i in range(12)]
    s = "\n".join(lines)
    out = format_tool_result_for_interaction_log(s, full=False)
    assert "line0" in out and "line4" in out
    assert "line5" not in out
    assert "truncated" in out
    assert "7 more lines" in out  # 12 - 5


def test_normalize_lightgbm_warning_line_ignores_ansi() -> None:
    raw = "\033[2m[LightGBM] [Warning] No further splits\033[0m"
    key = normalize_lightgbm_warning_line(raw)
    assert key is not None
    assert key.startswith("[LightGBM] [Warning]")
    assert "No further splits" in key
    assert "\033" not in key


def test_normalize_lightgbm_non_warning_returns_none() -> None:
    assert normalize_lightgbm_warning_line("[LightGBM] [Info] x") is None
    assert normalize_lightgbm_warning_line("hello") is None


def test_strip_interaction_ansi() -> None:
    assert strip_interaction_ansi("\033[2ma\033[0m") == "a"


def test_collapse_consecutive_lightgbm_warnings_in_text() -> None:
    w = "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf"
    block = "\n".join([w, w, w, "ok", w, w])
    got = collapse_consecutive_lightgbm_warnings_in_text(block)
    assert w in got
    assert "log-dedup" in got and "2 duplicate" in got
    assert got.count("log-dedup") == 2
    assert "ok" in got


def test_format_full_run_log_body_dedupes_lightgbm() -> None:
    w = "[LightGBM] [Warning] x"
    body = format_full_run_log_body(f"a\n{w}\n{w}", "")
    assert body.count(w) == 1
    assert "log-dedup" in body


def test_full_run_lightgbm_stream_deduper_collapses_and_flushes() -> None:
    out: list[str] = []
    d = FullRunLightGBMStreamDeduper(
        emit=out.append,
        format_line=lambda _s, ln: f"[full-run-stream] {ln}",
    )
    w = "[LightGBM] [Warning] same"
    d("stdout", w)
    d("stdout", w)
    d("stdout", w)
    d("stdout", "Training until validation")
    d("stdout", w)
    d.flush()
    assert out[0] == f"[full-run-stream] {w}"
    assert "log-dedup" in out[1] and "2 duplicate" in out[1]
    assert out[2] == "[full-run-stream] Training until validation"
    assert out[3] == f"[full-run-stream] {w}"
    assert len(out) == 4

    d("stdout", w)
    d("stdout", w)
    d.flush()
    assert "log-dedup" in out[-1] and "1 duplicate" in out[-1]
