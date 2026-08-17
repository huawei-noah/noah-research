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

"""Core ScienceAgent package and public helper exports."""

from __future__ import annotations

from scienceflow.core.agent.agent import ScienceAgent
from scienceflow.core.agent.tools.bash_utils import (
    _bash_command_parallel_safe,
    _embedded_failure_user_message,
    _looks_like_quick_test_solution_run,
    _looks_like_write_placeholder_mimicry,
)
from scienceflow.core.agent.shared.constants import (
    _WS_LOG_RECOVERY_CAP,
    _WS_LOG_USER_CAP,
)
from scienceflow.core.agent.io.interaction_log import (
    _WS_LOG_BODY_CAP,
    _WS_LOG_MAX_LINES,
    _WS_LOG_TOOL_CALL_SINGLE_CAP,
    _WS_LOG_TOOL_RESULT_CAP,
    format_tool_call_lines_for_interaction_log,
    format_tool_result_for_interaction_log,
    tool_result_text_for_interaction_log,
    truncate_for_interaction_log,
)
from scienceflow.core.agent.io.interaction_log_policy import (
    InteractionLogPolicy,
    resolve_interaction_log_policy,
)
from scienceflow.core.agent.memory.memory_utils import (
    _compress_tool_call_for_memory,
    _tool_call_args_from_tc,
    inject_thought_into_tool_params,
)
from scienceflow.core.agent.prompts.system_prompt import _DEFAULT_SYSTEM, _default_system_prompt
from scienceflow.core.agent.prompts.write_coaching import (
    _extract_python_code_from_assistant_text,
)

# Re-export common integration points from the agent package root.
from scienceflow.core.tools import create_tool_collection
from scienceflow.safety.execution_policy import (
    clear_embedded_full_run_result,
    ensure_full_execution,
    write_embedded_full_run_result,
)

__all__ = [
    "ScienceAgent",
    "_DEFAULT_SYSTEM",
    "_WS_LOG_BODY_CAP",
    "_WS_LOG_MAX_LINES",
    "_WS_LOG_RECOVERY_CAP",
    "_WS_LOG_TOOL_CALL_SINGLE_CAP",
    "_WS_LOG_TOOL_RESULT_CAP",
    "_WS_LOG_USER_CAP",
    "_bash_command_parallel_safe",
    "_compress_tool_call_for_memory",
    "_default_system_prompt",
    "_embedded_failure_user_message",
    "_extract_python_code_from_assistant_text",
    "_looks_like_quick_test_solution_run",
    "_looks_like_write_placeholder_mimicry",
    "_tool_call_args_from_tc",
    "clear_embedded_full_run_result",
    "create_tool_collection",
    "ensure_full_execution",
    "format_tool_call_lines_for_interaction_log",
    "format_tool_result_for_interaction_log",
    "tool_result_text_for_interaction_log",
    "InteractionLogPolicy",
    "resolve_interaction_log_policy",
    "truncate_for_interaction_log",
    "inject_thought_into_tool_params",
    "write_embedded_full_run_result",
]
