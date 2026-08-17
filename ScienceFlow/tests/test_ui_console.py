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

"""Display helpers for Rich terminal UI."""

from scienceflow.ui.console import shorten_long_lines, truncate_middle, RichUI


def test_truncate_middle_short_unchanged():
    assert truncate_middle("hello", max_len=20) == "hello"


def test_truncate_middle_long():
    s = "a" * 20 + "PATH" + "b" * 20
    out = truncate_middle(s, max_len=16)
    assert "..." in out
    assert out.startswith("a")
    assert out.endswith("b")


def test_shorten_long_lines_per_line():
    short = "ok"
    long_line = "/" + "x" * 120 + "/end"
    text = f"{short}\n{long_line}\n{short}"
    out = shorten_long_lines(text, max_len=40)
    lines = out.split("\n")
    assert lines[0] == "ok"
    assert len(lines[1]) <= 40
    assert "..." in lines[1]
    assert lines[2] == "ok"


def test_richui_render_exec_step_no_crash():
    """render_exec_step with typical args should not raise."""
    ui = RichUI()
    ui.render_exec_step(
        step=1,
        lang="bash",
        code="ls -la /some/very/long/path/that/keeps/going",
        output="total 42\nfile1.csv\nfile2.csv",
        returncode=0,
        exec_time=0.3,
    )


def test_richui_render_exec_step_empty_output():
    """render_exec_step with empty output shows feedback, no crash."""
    ui = RichUI()
    ui.render_exec_step(
        step=2,
        lang="python",
        code="print()",
        output="",
        returncode=0,
        exec_time=0.1,
    )


def test_richui_render_agent_reply_no_crash():
    ui = RichUI()
    ui.render_agent_reply("Hello, this is the **final** answer.")
