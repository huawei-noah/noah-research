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

"""REPL small-talk detection and tool_choice for first round."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from scienceflow.core.agent.run_policy import AutoContinuePolicy

if TYPE_CHECKING:
    from scienceflow.core.agent.run_policy import RunPolicy


def looks_like_repl_small_talk(text: str) -> bool:
    """True when REPL user input is greeting / chat with no workspace action."""
    t = (text or "").strip()
    if not t or len(t) > 200:
        return False
    tl = t.lower()
    task_keywords = (
        "read",
        "grep",
        "write",
        "edit",
        "bash",
        "solution",
        "dataset",
        "python",
        "train",
        ".csv",
        "file",
        "路径",
        "读取",
        "运行",
        "修改",
        "code",
        "error",
        "fix",
        "提交",
        "model",
        "ls ",
        "cat ",
        "pip",
        "import ",
        "def ",
        "class ",
        "run ",
        "/",
    )
    if any(k in tl for k in task_keywords):
        return False
    if re.match(
        r"^(你好|您好|嗨|哈喽|早上好|晚上好|下午好|在吗|在？|在\?|hi|hello|hey|"
        r"thanks|thank you|thx|谢了|谢谢|多谢|再见|拜拜|bye|goodbye|ok|好的|嗯|嗯嗯|早|晚安|哈哈)"
        r"([！!啊呀呢~～\s。,，…]*)?$",
        t,
        re.I,
    ):
        return True
    if re.match(r"^(你好|您好|hi|hello)[\s!！?？。,，]*$", t, re.I):
        return True
    return False


def tool_choice_for_main_loop(
    run_policy: RunPolicy,
    round_idx: int,
    run_request: str | None,
) -> str:
    """REPL: use tool_choice=none on small-talk turns so the model cannot emit tools."""
    if not isinstance(run_policy, AutoContinuePolicy):
        return "auto"
    if round_idx != 0 or not run_request:
        return "auto"
    if looks_like_repl_small_talk(run_request):
        return "none"
    return "auto"
