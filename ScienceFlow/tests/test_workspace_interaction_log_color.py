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

"""Tests for workspace interaction log ANSI coloring."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tests.fs_root import TEST_WORKSPACE_ROOT

from scienceflow.utils.logging import ScienceFlowColorFormatter, ScienceFlowPlainFormatter
from scienceflow.utils.workspace_interaction_log import (
    InteractionColorFormatter,
    attach_workspace_interaction_logger,
    build_interaction_log_context_tag,
    close_workspace_interaction_logger,
    colorize_interaction_message,
    reset_interaction_log_context_tag,
    set_interaction_log_context_tag,
    write_raw_to_interaction_log,
)


def test_colorize_user_prefix_inserts_sgr_when_enabled() -> None:
    raw = "[user] hello"
    out = colorize_interaction_message(raw, enabled=True)
    assert "\033[" in out
    assert "hello" in out
    assert out.endswith("\033[0m")


def test_colorize_disabled_returns_plain() -> None:
    raw = "[user] hello"
    assert colorize_interaction_message(raw, enabled=False) == raw


def test_colorize_still_colors_when_no_color_env_set() -> None:
    """File interaction log must not be suppressed by terminal NO_COLOR (e.g. Cursor)."""
    raw = "[tool-call] bash {}"
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        out = colorize_interaction_message(raw, enabled=True)
    assert "\033[32m" in out
    assert "bash" in out


def test_colorize_tool_result_err_red() -> None:
    raw = "[tool-result] read exit=err oops"
    out = colorize_interaction_message(raw, enabled=True)
    assert "\033[31m" in out


def test_colorize_unknown_prefix_unchanged() -> None:
    raw = "some freeform line"
    assert colorize_interaction_message(raw, enabled=True) == raw


def test_colorize_multiline_dims_continuation() -> None:
    raw = "[tool-call-body] bash.command (3 chars)\n---- begin ----\nls\n---- end ----"
    out = colorize_interaction_message(raw, enabled=True)
    assert "\033[2m" in out  # dim on continuation
    assert "---- begin ----" in out


def test_interaction_color_formatter_record() -> None:
    fmt = InteractionColorFormatter()
    rec = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="[user] x",
        args=(),
        exc_info=None,
    )
    line = fmt.format(rec)
    assert "202" in line or "-" in line[:20]
    assert "| INFO     |" in line
    assert "\033[96m" in line


def test_interaction_color_formatter_plain_when_no_color_in_message() -> None:
    fmt = InteractionColorFormatter()
    rec = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="unrecognized",
        args=(),
        exc_info=None,
    )
    line = fmt.format(rec)
    assert "\033[" not in line


def test_scienceflow_color_formatter_matches_interaction_user_style() -> None:
    """scienceflow.log uses the same :func:`colorize_interaction_message` rules as interaction.log."""
    fmt = ScienceFlowColorFormatter()
    rec = logging.LogRecord(
        name="scienceflow",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="[user] hello",
        args=(),
        exc_info=None,
    )
    line = fmt.format(rec)
    assert "| scienceflow |" in line
    assert "\033[96m" in line
    ifmt = InteractionColorFormatter()
    irec = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="[user] hello",
        args=(),
        exc_info=None,
    )
    iline = ifmt.format(irec)
    assert line.split("|", 3)[-1].strip() == iline.split("|", 3)[-1].strip()


def test_scienceflow_plain_formatter_has_no_escapes() -> None:
    fmt = ScienceFlowPlainFormatter()
    rec = logging.LogRecord(
        name="scienceflow",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="[user] hello",
        args=(),
        exc_info=None,
    )
    line = fmt.format(rec)
    assert "\033[" not in line
    assert "[user] hello" in line


def test_interaction_log_line_includes_context_tag_from_contextvar() -> None:
    """LNR binds session+phase so each formatted line shows ``[improve|explore]`` etc."""
    with tempfile.TemporaryDirectory(dir=str(TEST_WORKSPACE_ROOT)) as tmp:
        ws = Path(tmp)
        tag = build_interaction_log_context_tag("improve", "explore")
        tok = set_interaction_log_context_tag(tag)
        try:
            lg = attach_workspace_interaction_logger(ws, color=False)
            assert lg is not None
            lg.info("[user] hi")
            text = (ws / ".logs" / "interaction.log").read_text(encoding="utf-8")
        finally:
            reset_interaction_log_context_tag(tok)
    assert "[improve|explore]" in text
    assert "[user] hi" in text


def test_write_raw_to_interaction_log_appends_without_timestamp() -> None:
    with tempfile.TemporaryDirectory(dir=str(TEST_WORKSPACE_ROOT)) as tmp:
        ws = Path(tmp)
        lg = attach_workspace_interaction_logger(ws, color=False)
        assert lg is not None
        lg.info("header line")
        write_raw_to_interaction_log(lg, "stream chunk")
        write_raw_to_interaction_log(lg, "\n")  # mirrors _consume_stream when chunk has no trailing \n
        lg.info("after stream")
        text = (ws / ".logs" / "interaction.log").read_text(encoding="utf-8")
    assert "header line" in text
    assert "stream chunk" in text
    assert "after stream" in text
    # raw chunk must not be prefixed like formatted lines
    idx_chunk = text.index("stream chunk")
    idx_after = text.index("after stream")
    assert idx_chunk < idx_after
    line_with_chunk = [ln for ln in text.splitlines() if "stream chunk" in ln][0]
    assert "| INFO     |" not in line_with_chunk


def test_split_interaction_log_layout_writes_nested_files() -> None:
    with tempfile.TemporaryDirectory(dir=str(TEST_WORKSPACE_ROOT)) as tmp:
        ws = Path(tmp)
        lg = attach_workspace_interaction_logger(ws, color=False, layout="split")
        assert lg is not None
        lg.info("[user] split hi")
        write_raw_to_interaction_log(lg, "stream chunk")
        close_workspace_interaction_logger(ws, color=False, layout="split")
        inter = ws / ".logs" / "interaction" / "interaction.log"
        traj = ws / ".logs" / "traj_interaction" / "traj_interaction.log"
        assert inter.is_file()
        assert traj.is_file()
        assert "split hi" in inter.read_text(encoding="utf-8")
        assert "stream chunk" in traj.read_text(encoding="utf-8")
        assert not (ws / ".logs" / "interaction.log").exists()
